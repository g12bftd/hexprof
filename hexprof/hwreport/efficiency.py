"""Compute / memory efficiency, roofline and stall metrics for the hardware report.

Inputs per graph: the QHAS headline numbers (run time, timeline cycles, DRAM bytes), the
HTP-op-instance aggregates (cycles per unit, critical path), the QNN op type table (cycles,
% active cycles, critical path), the work in MACs (ONNX shapes or analytic), the deployed
weight bytes (AI Hub's compiled model size, or analytic) and the ceilings P (TOPS, per
precision) and B (GB/s) measured by ``hexprof ceilings``.

Every derived quantity says how it is computed; where an input is missing the metric is
left out rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hexprof.hwreport import analytic

THEORETICAL_GBPS = 84.8  # Snapdragon 8 Elite: LPDDR5X-5300 (10.6 GT/s) x 64-bit bus
DATA_MOVEMENT = {"Transpose", "Reshape", "StridedSlice", "Concat", "Split", "Pad", "Gather", "GatherElements",
                 "GatherNd", "Slice", "Tile", "Cast", "Convert", "DepthToSpace", "SpaceToDepth", "Pack", "UnPack",
                 "ChannelShuffle", "ScatterNd", "ScatterElements", "Expand", "Squeeze", "Unsqueeze", "Flatten"}
GRAPH_IO = {"Input", "Output"}
_GEMMA = re.compile(r"(prompt|token)_ar\d+_cl\d+_\d+_of_\d+")


@dataclass
class Ctx:
    P: dict[str, float] = field(default_factory=dict)  # TOPS per precision (ops = 2 x MACs)
    B: float | None = None  # GB/s, measured
    B_series: str | None = None
    theory_gbps: float = THEORETICAL_GBPS
    precisions: dict[str, str] = field(default_factory=dict)  # label -> precision overrides
    gemma_config: str | None = None
    onnx: dict[str, str] = field(default_factory=dict)  # label -> ONNX path overrides
    ceilings_src: dict[str, str] = field(default_factory=dict)


# A graph is limited by a resource when it reaches this share of the measured peak (P or B); below both
# roofs, the time goes elsewhere (vector work, waits, overheads): see the critical path by unit.
LIMIT_PCT = 60.0


def limiter(compute_util_pct: float | None, bw_util_pct: float | None) -> str | None:
    """Which measured peak a graph actually reaches (≥ LIMIT_PCT), if any."""
    if bw_util_pct is not None and bw_util_pct >= LIMIT_PCT:
        return "DRAM bandwidth"
    if compute_util_pct is not None and compute_util_pct >= LIMIT_PCT:
        return "compute (P)"
    if compute_util_pct is not None or bw_util_pct is not None:
        return "neither roof"
    return None


def precision_of(label: str, recorded: str | None, ctx: Ctx) -> str | None:
    if label in ctx.precisions:
        return ctx.precisions[label]
    if recorded:
        return recorded
    if _GEMMA.search(label):
        return "w4a16"
    if "vision" in label.lower():
        return "fp16"  # Qualcomm's Gemma 4 vision encoder: only FP16 compiles end to end
    return None


def _n(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _pct(a: float | None, b: float | None) -> float | None:
    return 100.0 * a / b if a is not None and b else None


# --------------------------------------------------------------------------- per graph


def graph_metrics(label: str, kpis: dict[str, Any], instances: dict[str, Any], qnn_types: list[dict[str, Any]],
                  qnn_nodes: list[dict[str, Any]], precision: str | None, info: dict[str, Any] | None,
                  onnx_path: str | None, ctx: Ctx, resources: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"precision": precision, "warnings": []}
    t_us = kpis.get("time_us")
    t_s = t_us / 1e6 if t_us else None
    tl = kpis.get("timeline_cycles")
    dram = kpis.get("total_dram")
    us_per_cycle = (t_us / tl) if t_us and tl else None

    # ---- work: ONNX shapes > analytic
    counted = None
    if onnx_path and Path(onnx_path).exists():
        try:
            from hexprof.hwreport import macs as M

            counted = M.count(onnx_path, (info or {}).get("shapes"))
            out["warnings"] += counted["warnings"]
        except Exception as e:  # noqa: BLE001 - fall back to analytic
            out["warnings"].append(f"ONNX MAC count failed for {onnx_path}: {type(e).__name__}: {e}")
    ana = analytic.gemma4_text(label, ctx.gemma_config) or analytic.siglip2(label, precision, ctx.gemma_config)
    if counted and counted["total_macs"]:
        out.update(macs=counted["total_macs"], macs_source=f"ONNX shapes ({Path(onnx_path).name}): every MatMul / Conv "
                   f"as compiled", weight_macs=counted["weight_macs"], activation_macs=counted["activation_macs"])
        if ana:
            out["macs_analytic"] = ana["macs"]
    elif ana:
        out.update(macs=ana["macs"], macs_source=ana["source"])
    # ---- deployed weight bytes: AI Hub's compiled model size > analytic
    size = (info or {}).get("compiled_file_size")
    if size:
        out.update(weight_bytes=float(size), weight_source="AI Hub compiled model size (weights as deployed"
                   + ("; shared by the prompt and token graphs of this part)" if _GEMMA.search(label) else ")"))
    elif ana:
        out.update(weight_bytes=ana["weight_bytes"], weight_source=ana["source"].split(";")[0].split(" (useful")[0])
    if ana:
        out["kv_bytes"] = ana.get("kv_bytes", 0.0)

    macs = out.get("macs")
    # mixed precision (Lite-MP) is compared with the measured P of its base precision
    P = (ctx.P.get(precision) or ctx.P.get(precision.split("_mixed")[0])) if precision else None
    B = ctx.B
    if macs and t_s:
        ops = 2 * macs
        out.update(ops=ops, achieved_tops=ops / t_s / 1e12)
        if tl:
            out["macs_per_cycle"] = macs / tl
        if P:
            out.update(P_tops=P, compute_util_pct=100 * out["achieved_tops"] / P)
    if dram and t_s:
        gbps = dram / t_s / 1e9
        out.update(dram_bytes=dram, dram_gbps=gbps, bw_util_theory_pct=100 * gbps / ctx.theory_gbps)
        if B:
            out.update(B_gbps=B, bw_util_pct=100 * gbps / B)
        if macs:
            out["intensity"] = 2 * macs / dram
    if out.get("intensity") and (P or B):
        mem_roof = out["intensity"] * (B or ctx.theory_gbps) / 1e3  # TOPS
        attainable = min(P, mem_roof) if P else mem_roof
        out.update(attainable_tops=attainable, roof_pct=100 * out["achieved_tops"] / attainable,
                   bound="memory" if (not P or mem_roof < P) else "compute",
                   roof_basis=("measured P and B" if P and B else "measured P, theoretical B" if P else
                               "measured B only" if B else "theoretical B only"))
    lim = limiter(out.get("compute_util_pct"), out.get("bw_util_pct"))
    if lim:
        out["limiter"] = lim  # "neither roof": the overview adds the unit that dominates the critical path
    if out.get("weight_bytes") and dram:
        expected = out["weight_bytes"] + out.get("kv_bytes", 0.0)
        out.update(expected_bytes=expected, beyond_bytes=dram - expected, dram_over_expected=dram / expected)
    # ---- sanity checks: a result above its roof means an input is inconsistent
    if out.get("roof_pct") and out["roof_pct"] > 110:
        out["warnings"].append(f"Achieved {out['achieved_tops']:.2f} TOPS is {out['roof_pct']:.0f}% of the roofline: "
                               "check that P was measured at this graph's precision and that the MAC count fits this graph.")
    if out.get("bw_util_pct") and out["bw_util_pct"] > 110:
        out["warnings"].append(f"DRAM traffic runs at {out['bw_util_pct']:.0f}% of the measured B: B is probably "
                               "underestimated (bandwidth sweep weights partly on chip?) or bytes are double counted.")
    if out.get("dram_over_expected") and out["dram_over_expected"] < 0.9:
        out["warnings"].append(f"QHAS measured {out['dram_over_expected']:.2f}× the expected weights + KV bytes: the "
                               "weight size is overestimated, or part of the weights stayed in on-chip memory.")

    # ---- stalls and parallelism
    work = instances.get("total_cycles") or 0.0
    bk = instances.get("by_kind") or {}
    dp_tot = sum(v.get("dp_cycles", 0.0) for v in bk.values())
    st: dict[str, Any] = {"idle_pct": kpis.get("percent_idle"), "work_cycles": work, "timeline_cycles": tl}
    if work and tl:
        st["avg_busy_threads"] = work / tl
        st["hvx_threads"] = (bk.get("HVX (vector)", {}).get("cycles", 0.0)) / tl
        st["hmx_ops_in_flight"] = (bk.get("HMX (matrix)", {}).get("cycles", 0.0)) / tl
    for typ in ("HMX", "HVX"):
        vals = [_n(r.get("utilization")) for r in resources or [] if str(r.get("type", "")).upper() == typ]
        if vals:
            st[f"{typ.lower()}_resource_util_sum"] = sum(vals)
    for k, key in (("DMA wait", "dma_wait"), ("Sync", "sync")):
        st[f"{key}_work_pct"] = _pct(bk.get(k, {}).get("cycles", 0.0), work)
        st[f"{key}_critical_pct"] = _pct(bk.get(k, {}).get("dp_cycles", 0.0), dp_tot)
    tot = sum(_n(r.get("cycles")) for r in qnn_types)
    with_active = [r for r in qnn_types if r.get("percent_active_cycles") is not None]
    if tot and with_active:
        active = sum(_n(r.get("cycles")) * _n(r.get("percent_active_cycles")) / 100 for r in with_active)
        covered = sum(_n(r.get("cycles")) for r in with_active)
        st["active_pct"] = 100 * active / covered
        st["waiting_pct"] = 100 - st["active_pct"]
        st["waiting_by_type"] = sorted(
            ({"op": r.get("op"), "waiting_cycles": _n(r.get("cycles")) * (1 - _n(r.get("percent_active_cycles")) / 100),
              "cycles": _n(r.get("cycles")), "active_pct": _n(r.get("percent_active_cycles"))} for r in with_active),
            key=lambda r: -r["waiting_cycles"])
    dpq = sum(_n(r.get("dominant_path_cycles")) for r in qnn_types)
    if dpq:
        st["data_movement_critical_pct"] = 100 * sum(_n(r.get("dominant_path_cycles")) for r in qnn_types
                                                     if str(r.get("op")) in DATA_MOVEMENT) / dpq
        st["graph_io_critical_pct"] = 100 * sum(_n(r.get("dominant_path_cycles")) for r in qnn_types
                                                if str(r.get("op")) in GRAPH_IO) / dpq
        st["data_movement_ops"] = sorted({str(r.get("op")) for r in qnn_types if str(r.get("op")) in DATA_MOVEMENT})
    out["stalls"] = st

    # ---- per op (ONNX shapes joined onto QHAS QNN nodes)
    if counted and counted["nodes"] and qnn_nodes and us_per_cycle:
        from hexprof.hwreport import macs as M

        j = M.match(qnn_nodes, counted)
        rows = []
        for r in j["rows"]:
            t = r["cycles"] * us_per_cycle * 1e-6
            row = {**{k: r[k] for k in ("name", "op_type", "macs", "weight", "cycles", "dram_bytes", "qnn_nodes")},
                   "out_shape": str(r["out_shape"]), "time_us": t * 1e6,
                   "tops": 2 * r["macs"] / t / 1e12 if t else None,
                   "intensity": 2 * r["macs"] / r["dram_bytes"] if r["dram_bytes"] else None}
            if P and row["tops"] is not None:
                row["compute_util_pct"] = 100 * row["tops"] / P
            if row["intensity"] and row["tops"] is not None:
                roof = row["intensity"] * (B or ctx.theory_gbps) / 1e3
                roof = min(P, roof) if P else roof
                row["roof_pct"] = 100 * row["tops"] / roof
            rows.append(row)
        out["per_op"] = {"rows": rows, "coverage_pct": 100 * j["coverage"], "onnx_nodes": len(counted["nodes"]),
                         "examples": {"onnx": [n["name"] for n in counted["nodes"][:3]],
                                      "qhas": [str(q.get("qnn_op") or q.get("op")) for q in qnn_nodes[:3]]}}
    return out


# --------------------------------------------------------------------------- page blocks


def _fmt(v: float | None, unit: str = "", d: int = 2) -> str:
    if v is None:
        return "–"
    a = abs(v)
    for suf, div in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if a >= div and unit in ("MAC", "op", "B", "cycles"):
            return f"{v / div:.{d}f} {suf}{unit}"
    sep = "" if unit in ("%", "×", "") else " "
    return f"{v:.{d}f}{sep}{unit}"


def _missing_ceilings(ctx: Ctx) -> str:
    miss = []
    if not ctx.P:
        miss.append("P (run `hexprof ceilings compute`, pass --compute-json)")
    if not ctx.B:
        miss.append(f"B (run `hexprof ceilings bandwidth`, pass --bandwidth-json; until then the theoretical "
                    f"{ctx.theory_gbps:g} GB/s is used)")
    return ("Missing measured ceilings: " + "; ".join(miss) + ".") if miss else ""


def graph_sections(m: dict[str, Any], top: int, ctx: Ctx) -> list[dict[str, Any]]:
    secs: list[dict[str, Any]] = []
    rows = [
        {"metric": "Precision", "value": m.get("precision") or "–", "how": "deployed weights / activations"},
        {"metric": "Work (MACs per run)", "value": _fmt(m.get("macs"), "MAC"), "how": m.get("macs_source") or
         "unknown: no ONNX (run `hexprof hw fetch-models --onnx …`) and no analytic model for this graph"},
        {"metric": "Achieved compute", "value": _fmt(m.get("achieved_tops"), "TOPS"), "how": "2 × MACs ÷ QHAS run time"},
        {"metric": "Compute utilization", "value": _fmt(m.get("compute_util_pct"), "%", 1),
         "how": f"achieved ÷ P ({_fmt(m.get('P_tops'), 'TOPS')}, measured at {m.get('precision')})" if m.get("P_tops")
         else "needs P from `hexprof ceilings compute`"},
        {"metric": "MACs per cycle", "value": _fmt(m.get("macs_per_cycle"), "MAC/cycle", 0),
         "how": "MACs ÷ timeline cycles (independent of the clock speed)"},
        {"metric": "DRAM bandwidth", "value": _fmt(m.get("dram_gbps"), "GB/s", 1), "how": "QHAS DRAM bytes ÷ QHAS run time"},
        {"metric": "Bandwidth utilization (measured B)", "value": _fmt(m.get("bw_util_pct"), "%", 1),
         "how": f"÷ B = {_fmt(m.get('B_gbps'), 'GB/s', 1)} ({ctx.B_series})" if m.get("B_gbps")
         else "needs B from `hexprof ceilings bandwidth`"},
        {"metric": "Bandwidth utilization (theoretical)", "value": _fmt(m.get("bw_util_theory_pct"), "%", 1),
         "how": f"÷ {ctx.theory_gbps:g} GB/s (LPDDR5X-5300, 64-bit; never fully reachable)"},
        {"metric": "Arithmetic intensity", "value": _fmt(m.get("intensity"), "ops/byte", 1), "how": "2 × MACs ÷ DRAM bytes"},
        {"metric": "Roofline: attainable", "value": _fmt(m.get("attainable_tops"), "TOPS"),
         "how": f"min(P, intensity × B) ({m.get('roof_basis')})" if m.get("roof_basis") else "needs P and/or B"},
        {"metric": "Roofline: achieved / attainable", "value": _fmt(m.get("roof_pct"), "%", 1),
         "how": f"{m.get('bound')}-bound" if m.get("bound") else "–"},
        {"metric": "Deployed weights", "value": _fmt(m.get("weight_bytes"), "B"), "how": m.get("weight_source") or
         "unknown (run `hexprof hw fetch-models`)"},
        {"metric": "DRAM traffic ÷ (weights + KV cache)", "value": _fmt(m.get("dram_over_expected"), "×", 2),
         "how": ("needs the deployed weight size" if not m.get("expected_bytes") else
                 f"{_fmt(m.get('beyond_bytes'), 'B')} beyond weights + KV: activations, inputs/outputs and spills to "
                 "main memory" if m.get("beyond_bytes", 0) >= 0 else
                 f"{_fmt(-m['beyond_bytes'], 'B')} less than weights + KV: weight size overestimated, or weights "
                 "partly kept on chip")},
    ]
    note = _missing_ceilings(ctx)
    secs.append({"title": "Compute and memory efficiency", "why": "How close this graph runs to the NPU's limits. "
                 "Utilization here means work done ÷ peak rate, unlike QHAS's time-based “% utilization”.",
                 "blocks": [{"type": "note", "text": note, "table": {"open": True, "cols": [
                     {"key": "metric", "label": "Metric", "kind": "str"}, {"key": "value", "label": "Value", "kind": "str"},
                     {"key": "how", "label": "How", "kind": "str"}], "rows": rows}}]})
    st = m.get("stalls") or {}
    srows = [
        {"metric": "Idle (HTP not executing)", "value": _fmt(st.get("idle_pct"), "%", 1), "how": "QHAS percent_idle"},
        {"metric": "Waiting inside ops", "value": _fmt(st.get("waiting_pct"), "%", 1),
         "how": "op cycles not active ÷ op cycles, from Qualcomm's “% active cycles” per QNN op type"},
        {"metric": "DMA wait: share of work / of critical path",
         "value": f"{_fmt(st.get('dma_wait_work_pct'), '%', 1)} / {_fmt(st.get('dma_wait_critical_pct'), '%', 1)}",
         "how": "ops stalled until a DMA transfer (usually from DRAM) completes"},
        {"metric": "Sync: share of work / of critical path",
         "value": f"{_fmt(st.get('sync_work_pct'), '%', 1)} / {_fmt(st.get('sync_critical_pct'), '%', 1)}",
         "how": "ops waiting on other threads"},
        {"metric": "Data movement on the critical path", "value": _fmt(st.get("data_movement_critical_pct"), "%", 1),
         "how": "op types that only move or reshape data: " + (", ".join(st.get("data_movement_ops") or []) or "none")},
        {"metric": "Graph I/O on the critical path", "value": _fmt(st.get("graph_io_critical_pct"), "%", 1),
         "how": "QNN Input / Output ops (writing results out of the graph)"},
        {"metric": "Average busy HTP threads", "value": _fmt(st.get("avg_busy_threads"), "", 2),
         "how": "op cycles summed over threads ÷ timeline cycles"},
        {"metric": "Average busy vector (HVX) threads", "value": _fmt(st.get("hvx_threads"), "", 2),
         "how": "HVX-op cycles ÷ timeline cycles"},
        {"metric": "Average HMX ops in flight", "value": _fmt(st.get("hmx_ops_in_flight"), "", 2),
         "how": "HMX-op cycles ÷ timeline cycles (ops issued from several threads overlap, so this can exceed 1)"},
        {"metric": "HMX / HVX utilization (QHAS resources)",
         "value": f"{_fmt(st.get('hmx_resource_util_sum'), '%', 1)} / {_fmt(st.get('hvx_resource_util_sum'), '%', 1)}",
         "how": "Qualcomm's per-thread resource utilization, summed over threads (see Resource utilization)"},
    ]
    blocks: list[dict[str, Any]] = [{"type": "note", "text": "AI Hub does not expose hardware stall counters; these are "
                                     "the stall and parallelism measures QHAS supports.",
                                     "table": {"open": True, "cols": [
                                         {"key": "metric", "label": "Metric", "kind": "str"},
                                         {"key": "value", "label": "Value", "kind": "str"},
                                         {"key": "how", "label": "How", "kind": "str"}], "rows": srows}}]
    wbt = st.get("waiting_by_type") or []
    wtot = sum(r["waiting_cycles"] for r in wbt)
    if wtot:
        shown = wbt[:top]
        rest = wtot - sum(r["waiting_cycles"] for r in shown)
        hrows = [{"label": str(r["op"]), "value": 100 * r["waiting_cycles"] / wtot,
                  "extra": [["waiting cycles", _fmt(r["waiting_cycles"], "cycles")], ["% active", f"{r['active_pct']:.1f}%"]]}
                 for r in shown] + ([{"label": f"Other ({len(wbt) - len(shown)})", "value": 100 * rest / wtot,
                                      "color": "var(--other)"}] if rest > 0 else [])
        blocks.append({"type": "hbar", "rows": hrows, "kind": "pct", "title": "waiting cycles by QNN op type",
                       "table": {"cols": [{"key": "op", "label": "QNN op type", "kind": "str"},
                                          {"key": "waiting_cycles", "label": "Waiting cycles", "kind": "num"},
                                          {"key": "cycles", "label": "Cycles", "kind": "num"},
                                          {"key": "active_pct", "label": "% active", "kind": "pct"}], "rows": wbt}})
    secs.append({"title": "Stalls and parallelism", "why": "Where the NPU waits instead of computing: idle time, ops "
                 "stalled on memory transfers or other threads, and pure data-movement ops. Waiting cycles by QNN op "
                 "type (below) = cycles × (1 − % active).", "blocks": blocks})
    po = m.get("per_op")
    if po and not po["rows"]:
        ex = po.get("examples") or {}
        secs.append({"title": "Per-op roofline (ONNX shapes × QHAS cycles)",
                     "why": f"The source ONNX has {po['onnx_nodes']} MAC ops, but none of their names matched a QNN op in "
                            "QHAS, so per-op rates can't be computed. ONNX names look like "
                            f"{', '.join(ex.get('onnx') or []) or '–'}; QHAS names look like "
                            f"{', '.join(ex.get('qhas') or []) or '–'}.", "blocks": []})
    if po and po["rows"]:
        pts = [{"label": r["name"], "x": r["intensity"], "y": r["tops"], "group": m.get("precision") or "ops",
                "extra": [["op", r["op_type"]], ["MACs", _fmt(r["macs"], "MAC")], ["time", f"{r['time_us']:.1f} µs"],
                          ["% of roofline", _fmt(r.get("roof_pct"), "%", 1)]]}
               for r in po["rows"][:300] if r.get("intensity") and r.get("tops")]
        secs.append({"title": "Per-op roofline (ONNX shapes × QHAS cycles)",
                     "why": f"MAC-carrying ops matched between the source ONNX ({po['onnx_nodes']} MAC ops) and QHAS: "
                            f"{po['coverage_pct']:.0f}% of the MACs matched. Op time = its cycles (summed over threads) × "
                            "µs per cycle, so ops that run in parallel look slower than they are.",
                     "blocks": [{"type": "roofline", "points": pts, "groups": [m.get("precision") or "ops"],
                                 "P": {k: v for k, v in ctx.P.items() if k == m.get("precision")}, "B": ctx.B,
                                 "Bth": ctx.theory_gbps, "title": "per-op roofline",
                                 "table": {"cols": [{"key": "name", "label": "ONNX node", "kind": "str"},
                                                    {"key": "op_type", "label": "Op", "kind": "str"},
                                                    {"key": "out_shape", "label": "Output shape", "kind": "str"},
                                                    {"key": "macs", "label": "MACs", "kind": "num"},
                                                    {"key": "time_us", "label": "Time", "kind": "us"},
                                                    {"key": "tops", "label": "TOPS", "kind": "num"},
                                                    {"key": "compute_util_pct", "label": "% of P", "kind": "pct"},
                                                    {"key": "dram_bytes", "label": "DRAM", "kind": "bytes"},
                                                    {"key": "intensity", "label": "ops/byte", "kind": "num"},
                                                    {"key": "roof_pct", "label": "% of roofline", "kind": "pct"}],
                                           "rows": po["rows"][:500]}}]})
    return secs


# --------------------------------------------------------------------------- overview


def overview_sections(graphs: list[Any], short_label: Any, ctx: Ctx) -> list[dict[str, Any]]:
    rows, srows = [], []
    for g in graphs:
        m = g.out.get("efficiency") or {}
        if not m:
            continue
        bk = (g.out.get("qhas") or {}).get("by_kind") or {}
        top = max(bk.items(), key=lambda kv: kv[1].get("dp_cycles") or 0, default=(None, {}))
        tot_dp = sum(v.get("dp_cycles") or 0 for v in bk.values())
        lim = m.get("limiter")
        if lim == "neither roof" and top[0] and tot_dp:
            lim = f"neither roof: {top[0]} {100 * (top[1].get('dp_cycles') or 0) / tot_dp:.0f}% of critical path"
        rows.append({"label": short_label(g.label), "precision": m.get("precision"), "macs": m.get("macs"),
                     "limiter": lim,
                     "tops": m.get("achieved_tops"), "compute_util_pct": m.get("compute_util_pct"),
                     "gbps": m.get("dram_gbps"), "bw_util_pct": m.get("bw_util_pct"),
                     "bw_util_theory_pct": m.get("bw_util_theory_pct"), "intensity": m.get("intensity"),
                     "bound": m.get("bound"), "roof_pct": m.get("roof_pct"), "weights": m.get("weight_bytes"),
                     "dram_over_expected": m.get("dram_over_expected"),
                     "macs_source": (m.get("macs_source") or "–").split(":")[0]})
        st = m.get("stalls") or {}
        srows.append({"label": short_label(g.label), "idle": st.get("idle_pct"), "waiting": st.get("waiting_pct"),
                      "dma_crit": st.get("dma_wait_critical_pct"), "sync_crit": st.get("sync_critical_pct"),
                      "move_crit": st.get("data_movement_critical_pct"), "io_crit": st.get("graph_io_critical_pct"),
                      "threads": st.get("avg_busy_threads"), "hmx": st.get("hmx_ops_in_flight")})
    if not rows:
        return []
    note = _missing_ceilings(ctx)
    flagged = [r["label"] for r in rows if (r["roof_pct"] or 0) > 110 or (r["bw_util_pct"] or 0) > 110]
    if flagged:
        note = (note + " " if note else "") + ("Above their roof (inconsistent inputs, see the graph pages): "
                                               + ", ".join(flagged) + ".")
    groups = sorted({r["precision"] for r in rows if r["precision"]})
    pts = [{"label": r["label"], "x": r["intensity"], "y": r["tops"], "group": r["precision"] or "unknown",
            "extra": [["achieved", _fmt(r["tops"], "TOPS")], ["intensity", _fmt(r["intensity"], "ops/byte", 1)],
                      ["% of roofline", _fmt(r["roof_pct"], "%", 1)], ["bound", r["bound"] or "–"]]}
           for r in rows if r["intensity"] and r["tops"]]
    secs: list[dict[str, Any]] = [{
        "title": "Compute and memory efficiency, per graph",
        "why": "Utilization = achieved rate ÷ measured peak (P for compute at the graph's precision, B for DRAM "
               "bandwidth). Roofline: the attainable rate at a graph's arithmetic intensity is min(P, intensity × B); "
               "points far below their roof have room to improve. 'Roofline side' only says which roof is lower at "
               f"that intensity; 'Limited by' says whether the graph actually reaches ≥ {LIMIT_PCT:.0f}% of P or B. "
               "Below both, the run time goes to work the roofline does not count (vector ops, waits, overheads), "
               "named by the unit that dominates the critical path.",
        "blocks": ([{"type": "note", "text": note}] if note else []) + (
            [{"type": "roofline", "points": pts, "groups": groups or ["unknown"],
              "P": {g: v for g, v in ctx.P.items() if g in groups}, "B": ctx.B,
              "Bth": ctx.theory_gbps, "title": "roofline per graph"}] if pts else []) + [
            {"type": "note", "text": "", "table": {"open": True, "cols": [
                {"key": "label", "label": "Graph", "kind": "str"}, {"key": "precision", "label": "Precision", "kind": "str"},
                {"key": "macs", "label": "MACs", "kind": "num"}, {"key": "tops", "label": "TOPS", "kind": "num"},
                {"key": "compute_util_pct", "label": "% of P", "kind": "pct"}, {"key": "gbps", "label": "GB/s", "kind": "num"},
                {"key": "bw_util_pct", "label": "% of B", "kind": "pct"},
                {"key": "bw_util_theory_pct", "label": f"% of {ctx.theory_gbps:g}", "kind": "pct"},
                {"key": "limiter", "label": f"Limited by (≥ {LIMIT_PCT:.0f}% of a peak)", "kind": "str"},
                {"key": "intensity", "label": "ops/byte", "kind": "num"},
                {"key": "bound", "label": "Roofline side", "kind": "str"},
                {"key": "roof_pct", "label": "% of roofline", "kind": "pct"},
                {"key": "weights", "label": "Weights", "kind": "bytes"},
                {"key": "dram_over_expected", "label": "DRAM ÷ (weights + KV)", "kind": "num"},
                {"key": "macs_source", "label": "MACs from", "kind": "str"}], "rows": rows}}]}]
    wrows = [{"label": r["label"], "value": r["waiting"]} for r in srows if r["waiting"] is not None]
    secs.append({"title": "Stalls and parallelism, per graph",
                 "why": "Idle = HTP not executing (the hardware view: low idle means the NPU is busy all the run). "
                        "Waiting = per-thread view: cycles inside ops in which the op's thread was not active "
                        "(Qualcomm's % active cycles). Several HTP threads share the vector and matrix units, so a thread "
                        "often waits while another uses them: high waiting with low idle is not lost NPU time by itself. "
                        "Time actually lost to memory is DMA wait on the critical path. DMA wait, sync, data movement "
                        "and graph I/O are shares of the critical path; average busy threads = op cycles ÷ timeline "
                        "cycles. Hardware stall counters are not exposed by AI Hub.",
                 "blocks": ([{"type": "hbar", "rows": wrows, "kind": "pct", "max": 100,
                              "title": "share of op cycles in which the op's thread waited (per-thread view)"}]
                            if wrows else []) + [
                     {"type": "note", "text": "", "table": {"open": True, "cols": [
                         {"key": "label", "label": "Graph", "kind": "str"}, {"key": "idle", "label": "Idle %", "kind": "pct"},
                         {"key": "waiting", "label": "Waiting in ops (per thread) %", "kind": "pct"},
                         {"key": "dma_crit", "label": "DMA wait (crit.)", "kind": "pct"},
                         {"key": "sync_crit", "label": "Sync (crit.)", "kind": "pct"},
                         {"key": "move_crit", "label": "Data movement (crit.)", "kind": "pct"},
                         {"key": "io_crit", "label": "Graph I/O (crit.)", "kind": "pct"},
                         {"key": "threads", "label": "Busy threads", "kind": "num"},
                         {"key": "hmx", "label": "HMX ops in flight", "kind": "num"}], "rows": srows}}]})
    return secs
