"""Build the hardware report: one HTML page per graph, an overview, CSV tables and a JSON summary.

Input: folders (or files) holding AI Hub artifacts as hexprof downloads them,
e.g. a Gemma 4 export dir (``hw/<part>__<graph>/{qhas,optrace}/*.json`` +
``profiles/<part>__<graph>.profile.json``) or a SigLIP 2 profile run
(``<component>/{qhas,optrace}/*.json`` + ``<component>/<component>.profile.json``).
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from hexprof.aihub.metrics import aihub_metrics
from hexprof.hwreport import efficiency as EF
from hexprof.hwreport import html as H
from hexprof.hwreport import models as MD
from hexprof.hwreport import optrace as OT
from hexprof.hwreport import qhas as QH
from hexprof.io_utils import _jsonable, write_json

KINDS = ["HMX (matrix)", "HVX (vector)", "DMA wait", "DMA", "Sync", "Format change", "Reference op (scalar)", "Other"]
QHAS_DEFS = {
    "util": "Qualcomm: “Percentage of overall timeline HTP Utilization”: the share of the run in which the HTP core "
            "was executing operations (time-based; not compute or bandwidth efficiency).",
    "ips": "Qualcomm: “Inferences per second of overall graph” (1 / run time).",
    "resources": "Per HTP thread and resource type (HMX = matrix unit, HVX = vector units). Qualcomm: utilization = "
                 "“Percentage of utilization of this resource”; the table also lists the cycles the resource was used "
                 "(cycles_used) out of the timeline cycles.",
    "kinds": "Every executed HTP op instance, assigned once to the first unit it used in this order: HMX (matrix unit), "
             "HVX (vector units), reference op (scalar code, slow), format change, DMA wait, sync, DMA. “Work” = op cycles "
             "summed over all HTP threads. Several threads run vector (HVX) ops at the same time, while there is one matrix "
             "unit (HMX), so this shows where the work goes, not where the time goes.",
    "critical": "The same ops, counting only their cycles on the critical (dominant) path: the chain of ops that sets the "
                "run time. This is the wall-time view: a unit with a large share here is what to speed up.",
    "qnn_dp": "Share of the critical path per QNN op type (Qualcomm's num_dominant_path_cycles): the op types that set the "
              "run time. Compare with the work view above: an op type can do a lot of work in parallel without being on it.",
    "qnn": "QNN ops are the operations of the QNN IR graph: what the converter made from your ONNX (one per ONNX node, "
           "with its name). The HTP backend lowers each into one or more HTP ops (Qualcomm: “Number HTP Ops that roll up "
           "into this QNN op type”). Cycles = “total number of clock cycles taken by the specified operation”; % active "
           "cycles = “cycles when the operation is actively executing out of the total cycles” (Qualcomm).",
    "mem": "Qualcomm: DRAM read / write = “Number of bytes of DRAM (Dynamic Random-Access Memory) Read / Write”: traffic to "
           "main memory. VTCM read / write = “Number of bytes of VTCM (Vector Tightly Coupled Memory) Read / Write”: the "
           "HTP's on-chip scratch memory, much faster than DRAM. Bytes, not cycles. A QNN row adds up the HTP ops that roll "
           "up into it. GB/s = bytes ÷ the op's own time (cycles × µs per cycle, derived here); ops overlap, so per-op "
           "rates do not add up to the graph's.",
    "htp": "HTP ops are the kernels the HTP backend actually runs on the NPU after lowering the QNN graph: tiled over "
           "threads and VTCM, with inserted data transfers and layout changes; each knows “the QNN IR Op this HTP op maps "
           "to”. Reference op = “implementation runs purely on scalar code”; format change = the op converts a tensor "
           "layout (Qualcomm). Both are overhead to look for.",
    "dominant": "Dominant (critical) path: Qualcomm “number of cycles on the critical path for the operation”. Ops with a high "
                "share are the ones that set the run time; speeding up anything else does not shorten it.",
    "timeline": "Busy share of each HTP thread over the run, from the start cycle and length of every op instance. "
                "Gaps are idle time: waiting on memory transfers, synchronisation or the host.",
}


@dataclass
class Graph:
    label: str
    qhas: Path | None = None
    optrace: Path | None = None
    profile: Path | None = None
    published: dict[str, Any] | None = None  # AI Hub's published figures for this component, when known
    precision: str | None = None  # recorded by the run (SigLIP 2 profile.json), else inferred
    info: dict[str, Any] | None = None  # AI Hub model facts from `hexprof hw fetch-models`
    io: dict[str, Any] = field(default_factory=dict)  # inputs / outputs and what they mean
    out: dict[str, Any] = field(default_factory=dict)
    job_id: str | None = None  # the AI Hub profile job behind "AI Hub ms" (Workbench page: WORKBENCH_JOB)


WORKBENCH_JOB = "https://workbench.aihub.qualcomm.com/jobs/{}/"


def discover(paths: Sequence[str | Path]) -> list[Graph]:
    graphs: dict[str, Graph] = {}
    files: list[Path] = []
    roots: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            roots.append(p)
            files += [f for f in p.rglob("*.json") if not f.name.endswith((".ops.json", ".classified.json"))]
        elif p.exists():
            files.append(p)
            roots.append(p.parent)
    for f in files:
        name, folder = f.name.lower(), f.parent.name.lower()
        if "qhas" in name or folder == "qhas":  # AI Hub: <job>_main_qhas_summary.json
            graphs.setdefault(QH.label_of(f), Graph(QH.label_of(f))).qhas = f
        elif any(t in name for t in ("profile_trace", "optrace", "chrometrace")) or folder == "optrace":
            graphs.setdefault(OT.label_of(f), Graph(OT.label_of(f))).optrace = f  # <job>_main_profile_trace.json
    for r in roots:
        for pj in r.rglob("profile.json"):
            try:
                d = json.loads(pj.read_text()) if pj.stat().st_size < 20 * 2**20 else None
            except (OSError, ValueError):
                d = None
            comps = d.get("components") if isinstance(d, dict) else None
            for comp, rec in (comps or {}).items():
                if comp in graphs and isinstance(d, dict) and d.get("precision"):
                    graphs[comp].precision = str(d["precision"])
                if comp in graphs and isinstance(rec, dict) and rec.get("published"):
                    graphs[comp].published = {**rec["published"], "what": f"{d.get('precision')}, {d.get('runtime')}"}
    for g in graphs.values():
        for r in roots:
            hit = next(iter(sorted(r.rglob(f"{g.label}.profile.json"))), None)
            if hit:
                g.profile = hit
                break
    return sorted(graphs.values(), key=lambda g: _natural(g.label))


_GRAPH = re.compile(r"^(?:.+?__)?(?P<kind>prompt|token)_ar(?P<ar>\d+)_cl(?P<cl>\d+)_(?P<i>\d+)_of_(?P<n>\d+)$")


def short_label(label: str) -> str:
    """'part_1_of_4__prompt_ar128_cl4096_1_of_4' -> 'part 1/4 · prompt (ar128, cl4096)'."""
    m = _GRAPH.match(label)
    if m:
        return f"part {m['i']}/{m['n']} · {m['kind']} (ar{m['ar']}, cl{m['cl']})"
    return label.replace("__", " · ")


def _size(p: Path) -> str:
    return _fmt_bytes(p.stat().st_size)


def _natural(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def _n(v: Any) -> float:
    return QH.num(v) or 0.0


def _share_rows(rows: list[dict[str, Any]], key: str, total: float, label_fn: Any, top: int,
                extra_fn: Any = None) -> list[dict[str, Any]]:
    out, rest = [], 0.0
    for i, r in enumerate(rows):
        v = _n(r.get(key))
        if i < top:
            out.append({"label": label_fn(r), "value": 100 * v / total if total else 0.0,
                        "extra": extra_fn(r) if extra_fn else []})
        else:
            rest += v
    if rest > 0:
        out.append({"label": f"Other ({len(rows) - top})", "value": 100 * rest / total if total else 0.0,
                    "color": "var(--other)", "extra": []})
    return out


def _kind_color(kind: str) -> str:
    k = "DMA" if kind == "DMA set" else kind  # same slot as in the overview, where DMA set is merged into DMA
    return f"var(--s{KINDS.index(k) + 1})" if k in KINDS[:-1] else "var(--other)"


def _with_rates(rows: list[dict[str, Any]], us_per_cycle: float | None) -> list[dict[str, Any]]:
    """Add DRAM / VTCM bytes and GB/s (bytes ÷ the op's own time) to QHAS op rows, in place."""
    for r in rows:
        r["dram_bytes"] = _n(r.get("dram_read")) + _n(r.get("dram_write"))
        r["vtcm_bytes"] = _n(r.get("vtcm_read")) + _n(r.get("vtcm_write"))
        t = _n(r.get("cycles")) * us_per_cycle * 1e-6 if us_per_cycle else 0.0
        r["dram_gbps"] = r["dram_bytes"] / t / 1e9 if t else None
        r["vtcm_gbps"] = r["vtcm_bytes"] / t / 1e9 if t else None
    return rows


MEM_COLS = [{"key": "dram_read", "label": "DRAM read", "kind": "bytes"},
            {"key": "dram_write", "label": "DRAM write", "kind": "bytes"},
            {"key": "vtcm_read", "label": "VTCM read", "kind": "bytes"},
            {"key": "vtcm_write", "label": "VTCM write", "kind": "bytes"},
            {"key": "dram_gbps", "label": "DRAM GB/s", "kind": "num"},
            {"key": "vtcm_gbps", "label": "VTCM GB/s", "kind": "num"}]


def _mem_charts(rows: list[dict[str, Any]], label_fn: Any, top: int, what: str) -> list[dict[str, Any]]:
    """DRAM and VTCM read/write as two stacked charts (different scales, so not one axis)."""
    out = []
    for mem, title in (("dram", "DRAM (main memory)"), ("vtcm", "VTCM (on-chip)")):
        best = sorted(rows, key=lambda r: -(_n(r.get(f"{mem}_read")) + _n(r.get(f"{mem}_write"))))[:top]
        if not any(_n(r.get(f"{mem}_read")) + _n(r.get(f"{mem}_write")) for r in best):
            continue
        out.append({"type": "note", "text": f"{title}: the {len(best)} {what} with the most traffic"})
        out.append({"type": "stacked", "kind": "bytes", "series": [f"{mem.upper()} read", f"{mem.upper()} write"],
                    "title": f"{title} traffic by {what}",
                    "rows": [{"label": label_fn(r), "values": [_n(r.get(f"{mem}_read")), _n(r.get(f"{mem}_write"))]} for r in best]})
    return out


def _ms(v: Any) -> str:
    return "–" if v is None else f"{float(v):.3f} ms"


def _mb(r: Any) -> str:
    return f"{r.get('min')}–{r.get('max')} MB" if isinstance(r, dict) else "–"


def _fmt_bytes(v: Any) -> str:
    v = _n(v)
    for unit, d in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):  # decimal, consistent with GB/s
        if abs(v) >= d:
            return f"{v / d:.1f} {unit}"
    return f"{v:.0f} B"


def _fmt_n(v: Any) -> str:
    v = _n(v)
    for unit, d in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= d:
            return f"{v / d:.2f}{unit}"
    return f"{v:.0f}"


# --------------------------------------------------------------------------- per graph


def graph_page(g: Graph, out_dir: Path, top: int, bins: int, ctx: EF.Ctx | None = None) -> dict[str, Any]:
    ctx = ctx or EF.Ctx()
    csv_dir = out_dir / g.label
    sections: list[dict[str, Any]] = []
    tiles: list[dict[str, Any]] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {"label": g.label, "files": {k: str(v) for k, v in
                                                          (("qhas", g.qhas), ("optrace", g.optrace), ("profile", g.profile)) if v}}
    if g.qhas:
        rep = QH.load(g.qhas, top_n=max(200, top), timeline_bins=bins)
        warnings += rep.warnings
        k = QH.overall_kpis(rep)
        upc = rep.cycles_to_us()
        for rows_ in (rep.qnn_op_types, rep.qnn_op_instances, rep.htp_op_types):
            _with_rates(rows_, upc)
        summary["qhas"] = {"version": rep.version, "kpis": k, "by_kind": rep.instances.get("by_kind"),
                           "op_instances": rep.instances.get("count"),
                           "qnn_op_types": [{"op": r.get("op"), "cycles": _n(r.get("cycles")),
                                             "dp_cycles": _n(r.get("dominant_path_cycles")),
                                             "dram_bytes": r["dram_bytes"], "vtcm_bytes": r["vtcm_bytes"]}
                                            for r in rep.qnn_op_types]}
        tiles += [
            {"label": "HTP utilization", "value": f"{k['percent_utilization']:.1f}%" if k.get("percent_utilization") is not None else "–",
             "sub": "share of the run the NPU was busy"},
            {"label": "Inferences per second", "value": f"{k['inf_per_s']:.1f}" if k.get("inf_per_s") else "–",
             "sub": f"run time {k['time_us'] / 1e3:.2f} ms" if k.get("time_us") else None},
            {"label": "DRAM traffic", "value": _fmt_bytes(k.get("total_dram")),
             "sub": f"{k['dram_gbps']:.1f} GB/s average" if k.get("dram_gbps") is not None else None},
            {"label": "VTCM traffic", "value": _fmt_bytes(k.get("total_vtcm")),
             "sub": f"{k['vtcm_gbps']:.1f} GB/s average" if k.get("vtcm_gbps") is not None else None},
            {"label": "Timeline cycles", "value": _fmt_n(k.get("timeline_cycles")),
             "sub": f"≈ {k['clock_mhz']:.0f} MHz effective" if k.get("clock_mhz") else None},
            {"label": "HTP op instances", "value": f"{rep.instances.get('count') or 0:,}",
             "sub": f"{len(rep.qnn_op_instances)} QNN op nodes"},
        ]
        _csv(csv_dir / "qhas_overall.csv", rep.overall)
        sections.append({"title": "Overall summary per HTP core", "why": QHAS_DEFS["util"] + " " + QHAS_DEFS["ips"],
                         "blocks": [{"type": "note", "text": "DRAM/VTCM GB/s = bytes moved ÷ run time (derived here).",
                                     "table": {"open": True, "cols": [
                                         {"key": "htp", "label": "HTP", "kind": "str"},
                                         {"key": "percent_utilization", "label": "Util %", "kind": "pct"},
                                         {"key": "percent_idle", "label": "Idle %", "kind": "pct"},
                                         {"key": "inf_per_s", "label": "IPS", "kind": "num"},
                                         {"key": "time_us", "label": "Time", "kind": "us"},
                                         {"key": "timeline_cycles", "label": "Timeline cycles", "kind": "num"},
                                         {"key": "total_dram_read", "label": "DRAM read", "kind": "bytes"},
                                         {"key": "total_dram_write", "label": "DRAM write", "kind": "bytes"},
                                         {"key": "total_vtcm_read", "label": "VTCM read", "kind": "bytes"},
                                         {"key": "total_vtcm_write", "label": "VTCM write", "kind": "bytes"}],
                                         "rows": rep.overall}}]})
        if rep.resources:
            res = sorted(rep.resources, key=lambda r: -_n(r.get("utilization")))
            multi = len(rep.overall) > 1
            rows = [{"label": (f"HTP {r.get('htp')} · " if multi else "") + f"{r.get('type', '?')} · TID {r.get('tid', '?')}",
                     "value": _n(r.get("utilization")),
                     "extra": [["cycles used", _fmt_n(r.get("cycles_used"))], ["timeline cycles", _fmt_n(r.get("timeline_cycles"))]]}
                    for r in res]
            _csv(csv_dir / "qhas_resources.csv", rep.resources)
            sections.append({"title": "Resource utilization (HMX, HVX, …)", "why": QHAS_DEFS["resources"],
                             "blocks": [{"type": "hbar", "rows": rows, "kind": "pct", "max": 100, "title": "resource utilization",
                                         "table": {"cols": [{"key": "htp", "label": "HTP", "kind": "str"},
                                                            {"key": "type", "label": "Type", "kind": "str"},
                                                            {"key": "tid", "label": "TID", "kind": "str"},
                                                            {"key": "utilization", "label": "Util %", "kind": "pct"},
                                                            {"key": "cycles_used", "label": "Cycles used", "kind": "num"},
                                                            {"key": "timeline_cycles", "label": "Timeline cycles", "kind": "num"},
                                                            {"key": "start_cycle", "label": "Start", "kind": "num"},
                                                            {"key": "end_cycle", "label": "End", "kind": "num"}],
                                                   "rows": res}}]})
        inst = rep.instances
        total = inst.get("total_cycles") or 0.0
        kinds = sorted(inst.get("by_kind", {}).items(), key=lambda kv: -kv[1]["cycles"])
        if kinds:
            krows = [{"label": k, "value": 100 * v["cycles"] / total if total else 0.0,
                      "color": _kind_color(k),
                      "extra": [["cycles", _fmt_n(v["cycles"])], ["op instances", _fmt_n(v["count"])],
                                ["DRAM bytes", _fmt_bytes(v["dram_bytes"])]]} for k, v in kinds]
            dptot = sum(v.get("dp_cycles", 0.0) for _, v in kinds)
            ktable = [{"kind": k, "share": 100 * v["cycles"] / total if total else 0.0,
                       "dp_share": 100 * v.get("dp_cycles", 0.0) / dptot if dptot else None, **v} for k, v in kinds]
            dprows = [{"label": k, "value": 100 * v.get("dp_cycles", 0.0) / dptot, "color": _kind_color(k),
                       "extra": [["critical-path cycles", _fmt_n(v.get("dp_cycles"))]]}
                      for k, v in sorted(kinds, key=lambda kv: -kv[1].get("dp_cycles", 0.0))] if dptot else []
            types = sorted(inst.get("by_type", {}).items(), key=lambda kv: -kv[1]["cycles"])
            ttable = [{"type": t, "share": 100 * v["cycles"] / total if total else 0.0, **v} for t, v in types]
            _csv(csv_dir / "htp_instances_by_kind.csv", ktable)
            _csv(csv_dir / "htp_instances_by_type.csv", ttable)
            if dprows:
                sections.append({"title": "What sets the run time: critical path by unit", "why": QHAS_DEFS["critical"],
                                 "blocks": [{"type": "hbar", "rows": dprows, "kind": "pct", "title": "critical path by unit"}]})
            sections.append({"title": "Where the work goes: HTP op instances by unit", "why": QHAS_DEFS["kinds"],
                             "blocks": [{"type": "hbar", "rows": krows, "kind": "pct", "title": "op cycles by unit",
                                         "table": {"cols": [{"key": "kind", "label": "Unit", "kind": "str"},
                                                            {"key": "share", "label": "% of op cycles", "kind": "pct"},
                                                            {"key": "dp_share", "label": "% of critical path", "kind": "pct"},
                                                            {"key": "cycles", "label": "Cycles", "kind": "num"},
                                                            {"key": "dp_cycles", "label": "Critical-path cycles", "kind": "num"},
                                                            {"key": "count", "label": "Op instances", "kind": "num"},
                                                            {"key": "dram_bytes", "label": "DRAM bytes", "kind": "bytes"}],
                                                   "rows": ktable}},
                                        {"type": "note", "text": "By Qualcomm's exact “Type” column (all units an op used):",
                                         "table": {"cols": [{"key": "type", "label": "Type", "kind": "str"},
                                                            {"key": "share", "label": "% of cycles", "kind": "pct"},
                                                            {"key": "cycles", "label": "Cycles", "kind": "num"},
                                                            {"key": "count", "label": "Op instances", "kind": "num"}],
                                                   "rows": ttable}}]})
        if rep.qnn_op_types:
            qtot = sum(_n(r.get("cycles")) for r in rep.qnn_op_types)
            qdp = sum(_n(r.get("dominant_path_cycles")) for r in rep.qnn_op_types)
            _csv(csv_dir / "qnn_op_types.csv", rep.qnn_op_types)
            _csv(csv_dir / "qnn_op_instances.csv", rep.qnn_op_instances)
            sections.append({"title": "QNN op types", "why": QHAS_DEFS["qnn"], "blocks": [
                {"type": "note", "text": "Work (op cycles summed over threads):"},
                {"type": "hbar", "kind": "pct", "title": "cycles by QNN op type",
                 "rows": _share_rows(rep.qnn_op_types, "cycles", qtot, lambda r: str(r.get("op")), top,
                                     lambda r: [["cycles", _fmt_n(r.get("cycles"))],
                                                ["% active cycles", f"{_n(r.get('percent_active_cycles')):.1f}%"],
                                                ["HTP ops", _fmt_n(r.get("num_htp_ops"))],
                                                ["DRAM", _fmt_bytes(_n(r.get("dram_read")) + _n(r.get("dram_write")))],
                                                ["critical path", f"{_n(r.get('dominant_path_percent')):.1f}%"]]),
                 "table": {"cols": [{"key": "op", "label": "QNN op type", "kind": "str"},
                                    {"key": "cycles", "label": "Cycles", "kind": "num"},
                                    {"key": "percent_active_cycles", "label": "% active", "kind": "pct"},
                                    {"key": "num_htp_ops", "label": "HTP ops", "kind": "num"},
                                    {"key": "dram_read", "label": "DRAM read", "kind": "bytes"},
                                    {"key": "dram_write", "label": "DRAM write", "kind": "bytes"},
                                    {"key": "vtcm_read", "label": "VTCM read", "kind": "bytes"},
                                    {"key": "vtcm_write", "label": "VTCM write", "kind": "bytes"},
                                    {"key": "dominant_path_percent", "label": "% critical path", "kind": "pct"}],
                           "rows": rep.qnn_op_types}},
                *([{"type": "note", "text": "Critical path (wall-time view): " + QHAS_DEFS["qnn_dp"]},
                   {"type": "hbar", "kind": "pct", "title": "critical path by QNN op type",
                    "rows": _share_rows(sorted(rep.qnn_op_types, key=lambda r: -_n(r.get("dominant_path_cycles"))),
                                        "dominant_path_cycles", qdp, lambda r: str(r.get("op")), top,
                                        lambda r: [["critical-path cycles", _fmt_n(r.get("dominant_path_cycles"))],
                                                   ["share of work", f"{100 * _n(r.get('cycles')) / qtot:.1f}%" if qtot else "–"]])}]
                  if qdp else []),
                {"type": "note", "text": f"QNN op instances (nodes), slowest {min(200, len(rep.qnn_op_instances))}:",
                 "table": {"cols": [{"key": "qnn_op", "label": "QNN op", "kind": "str"},
                                    {"key": "cycles", "label": "Cycles", "kind": "num"},
                                    {"key": "percent_active_cycles", "label": "% active", "kind": "pct"},
                                    {"key": "num_htp_ops", "label": "HTP ops", "kind": "num"},
                                    *MEM_COLS,
                                    {"key": "dominant_path_percent", "label": "% critical path", "kind": "pct"}],
                           "rows": rep.qnn_op_instances[:200]}}]})
            totals = inst.get("mem_totals") or {}
            check = []
            for key, lbl in (("dram_read", "DRAM read"), ("dram_write", "DRAM write"), ("vtcm_read", "VTCM read"),
                             ("vtcm_write", "VTCM write")):
                overall = k.get(f"total_{key}")
                row = {"what": lbl, "overall": overall,
                       "qnn_types": sum(_n(r.get(key)) for r in rep.qnn_op_types),
                       "htp_types": sum(_n(r.get(key)) for r in rep.htp_op_types),
                       "htp_instances": totals.get(key)}
                row["qnn_vs_overall"] = 100 * row["qnn_types"] / overall if overall else None
                check.append(row)
            _csv(csv_dir / "memory_totals_check.csv", check)
            qnn_by_dram = sorted(rep.qnn_op_instances, key=lambda r: -r["dram_bytes"])
            sections.append({"title": "Memory traffic: QNN ops (DRAM and VTCM)", "why": QHAS_DEFS["mem"], "blocks": [
                {"type": "note", "text": "Check: per-op bytes add up to QHAS's overall totals (they should agree; this "
                                         "also confirms the columns are bytes).",
                 "table": {"open": True, "cols": [
                     {"key": "what", "label": "", "kind": "str"},
                     {"key": "overall", "label": "Overall (QHAS summary)", "kind": "bytes"},
                     {"key": "qnn_types", "label": "Σ QNN op types", "kind": "bytes"},
                     {"key": "htp_types", "label": "Σ HTP op types", "kind": "bytes"},
                     {"key": "htp_instances", "label": "Σ HTP op instances", "kind": "bytes"},
                     {"key": "qnn_vs_overall", "label": "QNN ÷ overall", "kind": "pct"}], "rows": check}},
                *_mem_charts(rep.qnn_op_types, lambda r: str(r.get("op")), top, "QNN op types"),
                {"type": "note", "text": "All QNN op types:", "table": {"cols": [
                    {"key": "op", "label": "QNN op type", "kind": "str"}, {"key": "num_htp_ops", "label": "HTP ops", "kind": "num"},
                    {"key": "cycles", "label": "Cycles", "kind": "num"}, *MEM_COLS], "rows": rep.qnn_op_types}},
                *_mem_charts(rep.qnn_op_instances, lambda r: str(r.get("qnn_op") or r.get("op")), top, "QNN ops (nodes)"),
                {"type": "note", "text": f"QNN ops (nodes) with the most DRAM traffic (top {min(500, len(qnn_by_dram))}; "
                                         "all nodes in qnn_op_instances.csv):",
                 "table": {"cols": [{"key": "qnn_op", "label": "QNN op", "kind": "str"},
                                    {"key": "op", "label": "Type", "kind": "str"},
                                    {"key": "cycles", "label": "Cycles", "kind": "num"}, *MEM_COLS],
                           "rows": qnn_by_dram[:500]}}]})
        if rep.htp_op_types:
            htot = sum(_n(r.get("cycles")) for r in rep.htp_op_types)
            _csv(csv_dir / "htp_op_types.csv", rep.htp_op_types)

            def hlabel(r: dict[str, Any]) -> str:
                tags = [t for t, key in (("ref", "reference_op"), ("fmt", "format_change")) if QH.flag(r.get(key))]
                return str(r.get("op")) + (f" [{', '.join(tags)}]" if tags else "")

            sections.append({"title": "HTP op types", "why": QHAS_DEFS["htp"], "blocks": [
                {"type": "hbar", "kind": "pct", "title": "cycles by HTP op type",
                 "rows": _share_rows(rep.htp_op_types, "cycles", htot, hlabel, top,
                                     lambda r: [["cycles", _fmt_n(r.get("cycles"))], ["instances", _fmt_n(r.get("instances"))],
                                                ["% active cycles", f"{_n(r.get('percent_active_cycles')):.1f}%"],
                                                ["reference op", str(QH.flag(r.get("reference_op")))],
                                                ["format change", str(QH.flag(r.get("format_change")))]]),
                 "table": {"cols": [{"key": "op", "label": "HTP op", "kind": "str"},
                                    {"key": "cycles", "label": "Cycles", "kind": "num"},
                                    {"key": "instances", "label": "Instances", "kind": "num"},
                                    {"key": "percent_active_cycles", "label": "% active", "kind": "pct"},
                                    {"key": "reference_op", "label": "Ref op", "kind": "str"},
                                    {"key": "format_change", "label": "Format change", "kind": "str"},
                                    *MEM_COLS,
                                    {"key": "dominant_path_percent", "label": "% critical path", "kind": "pct"}],
                           "rows": rep.htp_op_types}}]})
            top_dram = inst.get("top_by_dram") or []
            if top_dram:
                _csv(csv_dir / "htp_op_instances_most_dram.csv", top_dram)
            sections.append({"title": "Memory traffic: HTP ops (DRAM and VTCM)", "why": QHAS_DEFS["mem"], "blocks": [
                *_mem_charts(rep.htp_op_types, hlabel, top, "HTP op types"),
                *_mem_charts(top_dram, lambda r: f"{r.get('htp_op')} ({r.get('qnn_op')})", top, "HTP op instances"),
                {"type": "note", "text": f"HTP op instances with the most DRAM traffic (top {len(top_dram)}):",
                 "table": {"cols": _instance_cols(), "rows": top_dram}} if top_dram else
                {"type": "note", "text": "No HTP op instance moved DRAM data."}]})
        dp = inst.get("top_by_dominant_path") or []
        if dp:
            sections.append({"title": "Critical path: ops that set the run time", "why": QHAS_DEFS["dominant"], "blocks": [
                {"type": "hbar", "kind": "pct", "title": "critical path share",
                 "rows": [{"label": f"{r.get('htp_op')} ({r.get('qnn_op')})", "value": _n(r.get("percent_dominant_path_cycles")),
                           "extra": [["cycles", _fmt_n(r.get("cycles"))], ["type", str(r.get("type"))], ["op id", str(r.get("op_id"))]]}
                          for r in dp[:top]],
                 "table": {"cols": _instance_cols(), "rows": dp}}]})
        slow = inst.get("top_by_cycles") or []
        if slow:
            _csv(csv_dir / "htp_op_instances_slowest.csv", slow)
            kind = "us" if slow[0].get("time_us") is not None else "num"
            sections.append({"title": "Slowest HTP op instances", "why": "Individual op executions ranked by cycles; time and "
                             "DRAM GB/s are derived from the core's µs per cycle (run time ÷ timeline cycles).", "blocks": [
                                 {"type": "hbar", "kind": kind, "title": "slowest op instances",
                                  "rows": [{"label": f"{r.get('htp_op')} ({r.get('qnn_op')})",
                                            "value": r.get("time_us") if kind == "us" else r.get("cycles"),
                                            "extra": [["cycles", _fmt_n(r.get("cycles"))], ["type", str(r.get("type"))],
                                                      ["DRAM", _fmt_bytes(_n(r.get("dram_read")) + _n(r.get("dram_write")))],
                                                      ["DRAM GB/s", f"{r['dram_gbps']:.1f}" if r.get("dram_gbps") is not None else "–"]]}
                                           for r in slow[:top]],
                                  "table": {"cols": _instance_cols(), "rows": slow}}]})
        # ---- efficiency, roofline and stalls (placed right after the overall summary)
        prec = EF.precision_of(g.label, g.precision, ctx)
        onnx = ctx.onnx.get(g.label) or (g.info or {}).get("onnx_path")
        eff = EF.graph_metrics(g.label, k, inst, rep.qnn_op_types, rep.qnn_op_instances, prec, g.info, onnx, ctx,
                               rep.resources)
        warnings += eff.pop("warnings", [])
        summary["efficiency"] = {kk: vv for kk, vv in eff.items() if kk != "per_op"}
        if eff.get("per_op"):
            summary["efficiency"]["per_op_coverage_pct"] = eff["per_op"]["coverage_pct"]
            _csv(csv_dir / "per_op_roofline.csv", eff["per_op"]["rows"])
        _csv(csv_dir / "efficiency.csv", [{kk: vv for kk, vv in eff.items() if not isinstance(vv, (dict, list))}])
        _csv(csv_dir / "waiting_by_qnn_op_type.csv", (eff.get("stalls") or {}).get("waiting_by_type") or [])
        pos = next((i + 1 for i, sec in enumerate(sections) if sec["title"] == "Overall summary per HTP core"), len(sections))
        sections[pos:pos] = EF.graph_sections(eff, top, ctx)
        if eff.get("achieved_tops") is not None:
            tiles.append({"label": "Achieved compute", "value": f"{eff['achieved_tops']:.2f} TOPS",
                          "sub": (f"{eff['compute_util_pct']:.1f}% of P ({prec})" if eff.get("compute_util_pct") is not None
                                  else "P not measured yet")})
        if eff.get("dram_gbps") is not None:
            tiles.append({"label": "Bandwidth utilization",
                          "value": f"{eff['bw_util_pct']:.1f}%" if eff.get("bw_util_pct") is not None
                          else f"{eff['bw_util_theory_pct']:.1f}%",
                          "sub": f"of measured B ({ctx.B:.1f} GB/s)" if eff.get("bw_util_pct") is not None
                          else f"of the theoretical {ctx.theory_gbps:g} GB/s"})
        if inst.get("timeline"):
            sections.append({"title": "QHAS timeline: HTP thread occupancy", "why": QHAS_DEFS["timeline"],
                             "blocks": [{"type": "lanes", "timeline": inst["timeline"], "title": "HTP thread occupancy",
                                         "table": {"cols": [{"key": "lane", "label": "Thread", "kind": "str"},
                                                            {"key": "busy_share", "label": "Busy share", "kind": "frac"}],
                                                   "rows": inst["timeline"]["lanes"]}}]})
    if g.optrace:
        ot = OT.load(g.optrace, bins=bins)
        kq = summary.get("qhas", {}).get("kpis", {})
        OT.calibrate(ot, kq.get("timeline_cycles"), kq.get("time_us"))
        tk = "us" if ot.unit == "µs" else "num"

        def tfmt(v: float, ms: bool = False) -> str:
            if ot.unit != "µs":
                return f"{v:.4g} {ot.unit}"
            return f"{v / 1e3:.3f} ms" if ms else f"{v:.2f} µs"

        summary["optrace"] = {"events": ot.events, "span_us": ot.span_us, "unit": ot.unit, "unit_note": ot.unit_note,
                              "raw_span": ot.raw_span, "declared_unit": ot.declared_unit, "lanes": ot.lanes,
                              "counters": [{k: v for k, v in c.items() if k != "series"} for c in ot.counters],
                              "top_ops": ot.ops[:50]}
        _csv(csv_dir / "optrace_threads.csv", ot.lanes)
        _csv(csv_dir / "optrace_ops.csv", ot.ops)
        _csv(csv_dir / "optrace_counters.csv", [{k: v for k, v in c.items() if k != "series"} for c in ot.counters])
        if not g.qhas:
            tiles += [{"label": "Trace span", "value": tfmt(ot.span or 0, ms=True), "sub": f"{_fmt_n(ot.events)} events"}]
        blocks: list[dict[str, Any]] = []
        if ot.lanes:
            lrows = sorted(ot.lanes, key=lambda r: -(r["busy_share"] or 0))
            blocks.append({"type": "hbar", "kind": "frac", "max": 1, "title": "busy share per thread",
                           "rows": [{"label": r["name"], "value": r["busy_share"],
                                     "extra": [["busy", tfmt(r["busy_us"], ms=True)], ["slices", _fmt_n(r["slices"])]]} for r in lrows],
                           "table": {"cols": [{"key": "name", "label": "Thread", "kind": "str"},
                                              {"key": "busy_share", "label": "Busy share", "kind": "frac"},
                                              {"key": "busy_us", "label": "Busy", "kind": tk},
                                              {"key": "slices", "label": "Slices", "kind": "num"}], "rows": lrows}})
        sections.append({"title": "Optrace: busy share per thread",
                         "why": f"{_fmt_n(ot.events)} trace events over {tfmt(ot.span or 0, ms=True)}. {ot.unit_note} "
                                "Busy share = time covered by top-level op slices on the thread ÷ trace span.",
                         "blocks": blocks or [{"type": "note", "text": "No op slices found."}]})
        if ot.timeline:
            sections.append({"title": "Optrace: timeline occupancy", "why": "Busy share of each trace thread over time.",
                             "blocks": [{"type": "lanes", "timeline": ot.timeline, "title": "optrace occupancy",
                                         "table": {"cols": [{"key": "name", "label": "Thread", "kind": "str"},
                                                            {"key": "busy_share", "label": "Busy share", "kind": "frac"},
                                                            {"key": "busy_us", "label": "Busy", "kind": tk}],
                                                   "rows": ot.lanes}}]})
        if ot.counters:
            cblocks: list[dict[str, Any]] = []
            for c in ot.counters[:6]:
                cblocks.append({"type": "line", "points": c["series"], "kind": "num", "title": c["name"], "xunit": ot.unit,
                                "subtitle": f"{c['name']} ({c['process']}): mean {c['time_weighted_mean']:.4g}, "
                                            f"min {c['min']:.4g}, max {c['max']:.4g}, {c['points']} samples"})
            cblocks[-1]["table"] = {"cols": [{"key": "process", "label": "Process", "kind": "str"},
                                             {"key": "name", "label": "Counter", "kind": "str"},
                                             {"key": "time_weighted_mean", "label": "Mean (time-weighted)", "kind": "num"},
                                             {"key": "min", "label": "Min", "kind": "num"},
                                             {"key": "max", "label": "Max", "kind": "num"},
                                             {"key": "points", "label": "Samples", "kind": "num"}],
                                    "rows": [{k: v for k, v in c.items() if k != "series"} for c in ot.counters]}
            sections.append({"title": "Optrace: counters (memory tracks)",
                             "why": "Counter tracks recorded in the trace, e.g. memory bandwidth, plotted over time "
                                    "(units as recorded by the trace). Long traces are bucketed: line = bucket mean, "
                                    "tooltip also shows the bucket max.", "blocks": cblocks})
        if ot.ops:
            ttot = sum(o["total_us"] for o in ot.ops)
            sections.append({"title": "Optrace: time by op name", "why": "Total slice time per op name across all threads and "
                             "nesting levels.", "blocks": [
                                 {"type": "hbar", "kind": "pct", "title": "time by op name",
                                  "rows": _share_rows(ot.ops, "total_us", ttot, lambda r: r["name"], top,
                                                      lambda r: [["total", tfmt(r["total_us"], ms=True)], ["count", _fmt_n(r["count"])],
                                                                 ["mean", tfmt(r["mean_us"])], ["max", tfmt(r["max_us"])]]),
                                  "table": {"cols": [{"key": "name", "label": "Op", "kind": "str"},
                                                     {"key": "total_us", "label": "Total", "kind": tk},
                                                     {"key": "count", "label": "Count", "kind": "num"},
                                                     {"key": "mean_us", "label": "Mean", "kind": tk},
                                                     {"key": "max_us", "label": "Max", "kind": tk}], "rows": ot.ops[:500]}}]})
        if ot.unit != "µs":
            warnings.append(ot.unit_note)
        if ot.other_phases:  # instant / flow / async markers carry no duration or value
            sections[-1]["blocks"].append({"type": "note", "text": "Other trace events (markers, not timed): " +
                                           ", ".join(f"{n:,} × ph={k!r}" for k, n in sorted(ot.other_phases.items()))})
    if g.profile:
        prof = json.loads(Path(g.profile).read_text())
        layers = sorted(prof.get("execution_detail", []), key=lambda r: -_n(r.get("execution_time")))
        m = aihub_metrics(prof) if prof.get("execution_summary") else {}
        summary["aihub"] = {**m, "inference_time_ms": m.get("inference_time_milliseconds"), "published": g.published}
        lc, pk = m.get("layer_counts") or {}, m.get("estimated_peak_memory_range_mb") or {}
        pub = g.published or {}
        tiles.append({"label": "AI Hub inference time",
                      "value": f"{m['inference_time_milliseconds']:.3f} ms" if m.get("inference_time_milliseconds") else "–",
                      "sub": (f"published {pub['inference_time_milliseconds']} ms" if pub.get("inference_time_milliseconds")
                              else ", ".join(f"{k.upper()} {v}" for k, v in lc.items() if k != "total" and v) + " layers")})
        if pk:
            lp = m.get("first_load_peak_memory_range_mb") or {}
            tiles.append({"label": "AI Hub inference peak memory", "value": f"{pk.get('min')}–{pk.get('max')} MB",
                          "sub": (f"load peak {lp.get('max')} MB; " if lp else "")
                                 + f"{lc.get('npu', 0)} of {lc.get('total', 0)} layers on the NPU"})
        rows_m = [{"metric": "Inference time", "value": _ms(m.get("inference_time_milliseconds")),
                   "published": _ms(pub.get("inference_time_milliseconds"))},
                  {"metric": "Peak memory (inference)", "value": _mb(pk), "published": _mb(pub.get("estimated_peak_memory_range_mb"))},
                  {"metric": "Layers on NPU / GPU / CPU", "value": f"{lc.get('npu', 0)} / {lc.get('gpu', 0)} / {lc.get('cpu', 0)}",
                   "published": f"{pub['layer_counts'].get('npu', 0)} / {pub['layer_counts'].get('total', 0)} total"
                   if isinstance(pub.get("layer_counts"), dict) else "–"},
                  {"metric": "First load time", "value": _ms(m.get("first_load_time_milliseconds")), "published": "–"},
                  {"metric": "Warm load time", "value": _ms(m.get("warm_load_time_milliseconds")), "published": "–"},
                  {"metric": "First / warm load peak memory",
                   "value": f"{_mb(m.get('first_load_peak_memory_range_mb'))} / {_mb(m.get('warm_load_peak_memory_range_mb'))}",
                   "published": "–"}]
        spread = m.get("all_inference_times_milliseconds") or {}
        if spread.get("n"):
            rows_m.append({"metric": f"Spread over {spread['n']} runs (min / median / max)",
                           "value": f"{spread['min']:.3f} / {spread['median']:.3f} / {spread['max']:.3f} ms", "published": "–"})
        sections.append({"title": "AI Hub profile summary", "why": "AI Hub's own measurement of this graph (what Workbench shows "
                         "on the job page), next to AI Hub's published figure when one exists"
                         + (f" ({pub.get('what')}, job {pub.get('job_id')})." if pub else "."),
                         "blocks": [{"type": "note", "text": "", "table": {"open": True, "cols": [
                             {"key": "metric", "label": "Metric", "kind": "str"}, {"key": "value", "label": "This job", "kind": "str"},
                             {"key": "published", "label": "Published", "kind": "str"}], "rows": rows_m}}]})
        if layers and any(r.get("execution_time") is not None for r in layers):
            _csv(csv_dir / "aihub_layers.csv", layers)
            sections.append({"title": "AI Hub per-layer profile", "why": "AI Hub's own per-layer timing (Runtime Layer Analysis) "
                             "and the compute unit each layer ran on.", "blocks": [
                                 {"type": "hbar", "kind": "us", "title": "slowest layers",
                                  "rows": [{"label": str(r.get("name")), "value": _n(r.get("execution_time")),
                                            "extra": [["type", str(r.get("type"))], ["unit", str(r.get("compute_unit"))]]} for r in layers[:top]],
                                  "table": {"cols": [{"key": "name", "label": "Layer", "kind": "str"},
                                                     {"key": "type", "label": "Type", "kind": "str"},
                                                     {"key": "compute_unit", "label": "Unit", "kind": "str"},
                                                     {"key": "execution_time", "label": "Time", "kind": "us"},
                                                     {"key": "execution_cycles", "label": "Cycles", "kind": "num"}],
                                            "rows": layers[:500]}}]})
    io_sec = _io_section(g)
    if io_sec:
        sections.insert(0, io_sec)
    summary["io"] = g.io
    g.out = summary
    files = ", ".join(p.name for p in (g.qhas, g.optrace, g.profile) if p)
    return {"title": short_label(g.label), "subtitle": f"{g.label} · sources: {files}", "nav": {"href": "index.html", "text": "← All graphs"},
            "tiles": tiles, "sections": sections, "warnings": warnings,
            "footer": f"Tables as CSV in {g.label}/. Definitions quoted from Qualcomm's QAIRT Visualizer. "
                      f"QHAS schema {summary.get('qhas', {}).get('version', '–')}."}


def _io_section(g: Graph) -> dict[str, Any] | None:
    """Inputs and outputs of the graph as compiled (from `hexprof hw fetch-models`), and what they mean."""
    from hexprof.hwreport import io as IO

    info = g.info or {}
    ins, outs = info.get("inputs") or {}, info.get("outputs") or {}
    g.io = {**IO.describe(g.label, ins, outs), "inputs": ins, "outputs": outs}
    if not (ins or outs):
        d = {k: v for k, v in g.io.items() if k not in ("inputs", "outputs")}
        if not d:
            return None
        text = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in d.items())
        return {"title": "Inputs and outputs", "why": "From the graph name only. Run `hexprof hw fetch-models` for the "
                "exact shapes AI Hub compiled.", "blocks": [{"type": "note", "text": text}]}
    d = g.io
    facts = [f"{k}: {d[k]}" for k in ("role", "image", "image_patches", "tokens_per_call", "context_length", "batch",
                                     "image_tokens_out", "logit_positions", "vocab", "embedding_dim") if d.get(k) is not None]
    return {"title": "Inputs and outputs", "why": "What this compiled graph takes and returns on the device, as AI Hub "
            "reports it for the model it profiled (shapes are per call; a static graph always runs at these sizes, "
            "padding shorter inputs). " + "; ".join(f.replace("_", " ") for f in facts) + ".",
            "blocks": [{"type": "note", "text": "", "table": {"open": True, "cols": [
                {"key": "direction", "label": "", "kind": "str"}, {"key": "name", "label": "Tensor", "kind": "str"},
                {"key": "shape", "label": "Shape", "kind": "str"}, {"key": "dtype", "label": "Type", "kind": "str"},
                {"key": "meaning", "label": "Meaning", "kind": "str"}], "rows": IO.table(ins, outs)}}]}


def requests_section(paths: Sequence[str | Path], graphs: list[Graph]) -> dict[str, Any] | None:
    """Your actual requests (`gemma4 aihub` / `siglip2 infer` results.json): image sizes and token counts."""
    rows: list[dict[str, Any]] = []
    fed = {g.label: (g.io or {}).get("image") for g in graphs if (g.io or {}).get("image")}
    fed_img = next((v for k, v in fed.items() if "vision" in k), None) or next(iter(fed.values()), None)
    for p in map(Path, paths):
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for r in d.get("results") or []:  # gemma4 aihub
            b, v = r.get("breakdown") or {}, r.get("vision_encoder") or {}
            img_tok = v.get("image_tokens")
            total = b.get("prefill_tokens")
            rows.append({"model": "Gemma 4", "image": Path(r.get("image", "")).name, "original": _img_size(r.get("image")),
                         "fed": (f"{r['image_size_fed'][0]}×{r['image_size_fed'][1]}" if r.get("image_size_fed")
                                 else fed_img or "448×448"),
                         "prompt": (r.get("prompt") or "")[:80], "image_tokens": img_tok,
                         "text_tokens": total - img_tok if total and img_tok else None, "prompt_tokens": total,
                         "chunks": f"{b.get('prefill_chunks')} × {b.get('prefill_chunk_tokens')}" if b.get("prefill_chunks") else None,
                         "generated_tokens": b.get("decode_tokens"), "generated_source": b.get("decode_tokens_source"),
                         "ttft_ms": b.get("ttft_ms"), "e2e_ms": b.get("e2e_ms"), "source": str(p)})
        if d.get("embedding_dim") and d.get("images"):  # siglip2 infer
            for img in d["images"]:
                rows.append({"model": "SigLIP 2", "image": Path(img).name, "original": _img_size(img),
                             "fed": fed.get("image_encoder") or "224×224", "prompt": ", ".join(d.get("labels") or [])[:80],
                             "text_tokens": f"{len(d.get('labels') or [])} label(s) × 64 (padded)" if d.get("labels") else None,
                             "output": f"embedding of size {d['embedding_dim']}", "source": str(p)})
    if not rows:
        return None
    return {"title": "Your requests: images and tokens", "why": "From the inference runs on your own image and prompt. "
            "Prompt tokens include the chat template and the image tokens the vision encoder produced; the prompt is "
            "processed in chunks of the prompt graph's length. Generated tokens: the length of the answer (from the local "
            "llama.cpp run when available, else --max-new-tokens). Times are AI Hub-based estimates.",
            "blocks": [{"type": "note", "text": "", "table": {"open": True, "cols": [
                {"key": "model", "label": "Model", "kind": "str"}, {"key": "image", "label": "Image", "kind": "str"},
                {"key": "original", "label": "Original size", "kind": "str"},
                {"key": "fed", "label": "Fed to the encoder", "kind": "str"},
                {"key": "prompt", "label": "Prompt / labels", "kind": "str"},
                {"key": "image_tokens", "label": "Image tokens", "kind": "num"},
                {"key": "text_tokens", "label": "Text + template tokens", "kind": "str"},
                {"key": "prompt_tokens", "label": "Prompt tokens (total)", "kind": "num"},
                {"key": "chunks", "label": "Prompt chunks", "kind": "str"},
                {"key": "generated_tokens", "label": "Generated tokens", "kind": "num"},
                {"key": "generated_source", "label": "Generated from", "kind": "str"},
                {"key": "ttft_ms", "label": "Time to first token (ms)", "kind": "num"},
                {"key": "e2e_ms", "label": "End to end (ms)", "kind": "num"},
                {"key": "output", "label": "Output", "kind": "str"}], "rows": rows}}]}


# Example requests for the end-to-end table: (what, image tokens?, text + chat-template tokens, output tokens).
# Text + template is an assumption (Gemma's chat template adds about 15-20 tokens around a short question);
# `hexprof gemma4 aihub --image ... --prompt ...` counts the real tokens of your own requests.
E2E_EXAMPLES = [("text question, 128-token answer", False, 40, 128),
                ("1 image + question, 128-token answer", True, 40, 128),
                ("1 image + question, 512-token answer", True, 40, 512),
                ("1 image + long prompt (500 tokens), 256-token answer", True, 500, 256)]


def e2e_examples(tok: dict[str, Any], pre: dict[str, Any], vis: list[Graph]) -> dict[str, Any]:
    """Time to first token and end to end for typical requests, from the profiled graphs."""
    chunk, chunk_ms, step_ms, ctx = int(pre["ar"]), float(pre["aihub_ms"]), float(tok["aihub_ms"]), int(tok["cl"])
    vg = vis[0] if vis else None
    vis_ms = float(vg.out["aihub"]["inference_time_ms"]) if vg else None
    img_tok = int((vg.io or {}).get("image_tokens_out") or 256) if vg else 256
    rows = []
    for what, image, text, out_tok in E2E_EXAMPLES:
        if image and vis_ms is None:
            continue
        prompt = text + (img_tok if image else 0)
        chunks = -(-prompt // chunk)
        ttft = (vis_ms if image else 0.0) + chunks * chunk_ms
        total = ttft + (out_tok - 1) * step_ms
        rows.append({"what": what, "image_tokens": img_tok if image else 0, "text_tokens": text, "prompt_tokens": prompt,
                     "chunks": f"{chunks} × {chunk}", "vision_ms": vis_ms if image else 0.0, "prefill_ms": chunks * chunk_ms,
                     "ttft_ms": ttft, "output_tokens": out_tok, "decode_ms": (out_tok - 1) * step_ms, "total_s": total / 1e3,
                     "fits": "yes" if prompt + out_tok <= ctx else f"no (> {ctx})"})
    return {"type": "note", "text": (
        f"Worked examples. Time to first token = vision encoder (one image: {img_tok} image tokens) + "
        f"ceil(prompt tokens ÷ {chunk}) × {chunk_ms:.1f} ms (the prompt graphs run once per {chunk}-token chunk, padded); "
        f"each further token = {step_ms:.1f} ms (one pass of the token graphs). The text + chat-template token counts are "
        "assumptions; your own requests (`hexprof gemma4 aihub --image … --prompt …`, then this report with "
        "--requests) are counted exactly."),
        "table": {"open": True, "cols": [
            {"key": "what", "label": "Request", "kind": "str"},
            {"key": "image_tokens", "label": "Image tokens", "kind": "num"},
            {"key": "text_tokens", "label": "Text + template tokens", "kind": "num"},
            {"key": "prompt_tokens", "label": "Prompt tokens", "kind": "num"},
            {"key": "chunks", "label": "Prefill chunks", "kind": "str"},
            {"key": "vision_ms", "label": "Vision (ms)", "kind": "num"},
            {"key": "prefill_ms", "label": "Prefill (ms)", "kind": "num"},
            {"key": "ttft_ms", "label": "Time to first token (ms)", "kind": "num"},
            {"key": "output_tokens", "label": "Output tokens", "kind": "num"},
            {"key": "decode_ms", "label": "Decode (ms)", "kind": "num"},
            {"key": "total_s", "label": "End to end (s)", "kind": "num"},
            {"key": "fits", "label": f"Fits {ctx} context", "kind": "str"}], "rows": rows}}


def deployment_section(paths: Sequence[str | Path]) -> dict[str, Any] | None:
    """What was deployed, from the run manifest(s) hexprof wrote next to the results (hexprof.runlog)."""
    found: list[dict[str, Any]] = []
    for p in map(Path, paths):
        root = p if p.is_dir() else p.parent
        for f in sorted(root.glob("run_manifest.json")) + sorted(root.glob("*/run_manifest.json")):
            try:
                found.append(json.loads(f.read_text()))
            except (OSError, ValueError):
                continue
    if not found:
        return None
    blocks: list[dict[str, Any]] = []
    for m in found:
        hw, mod = m.get("hardware") or {}, m.get("model") or {}
        rep = hw.get("as_reported_by_aihub") or {}
        facts = [{"what": "verdict", "value": m.get("verdict")}, {"what": "model", "value": mod.get("name")},
                 {"what": "weights", "value": mod.get("weights")},
                 {"what": "architecture", "value": ", ".join(f"{k} {v}" for k, v in (mod.get("architecture") or {}).items())},
                 {"what": "device", "value": f"{hw.get('device')} ({hw.get('soc')}, Hexagon {hw.get('hexagon')})"},
                 {"what": "ran on (AI Hub)", "value": "; ".join(f"{k} {', '.join(v)}" for k, v in rep.items()) or None},
                 {"what": "runtime", "value": hw.get("runtime")},
                 {"what": "run", "value": f"{m.get('run_id')}: {m.get('status')}, {m.get('duration_min')} min, "
                                          f"hexprof {(m.get('environment') or {}).get('hexprof_commit')}"}]
        setting_rows = [{"changed": "changed" if x.get("changed") else "", "group": x.get("group"),
                         "setting": x.get("setting"), "value": str(x.get("value")),
                         "base": "" if x.get("base") is None else str(x.get("base")), "note": x.get("note") or ""}
                        for x in m.get("settings") or []]
        blocks.append({"type": "note", "text": m.get("title") or "", "table": {"open": True, "cols": [
            {"key": "what", "label": "", "kind": "str"}, {"key": "value", "label": "", "kind": "str"}],
            "rows": [r for r in facts if r["value"]]}})
        if setting_rows:
            blocks.append({"type": "note", "text": "Settings (base = Qualcomm's / the real model's value)",
                           "table": {"open": True, "cols": [
                               {"key": "changed", "label": "", "kind": "str"}, {"key": "group", "label": "Group", "kind": "str"},
                               {"key": "setting", "label": "Setting", "kind": "str"},
                               {"key": "value", "label": "Value", "kind": "str"},
                               {"key": "base", "label": "Base", "kind": "str"},
                               {"key": "note", "label": "Note", "kind": "str"}], "rows": setting_rows}})
    return {"title": "What was deployed", "why": "From run_manifest.json, written by the command that deployed these "
            "graphs: the model, every setting next to Qualcomm's (or the real model's) value, and the hardware AI Hub "
            "reported for the profile jobs.", "blocks": blocks}


def _img_size(path: Any) -> str | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return f"{im.size[0]}×{im.size[1]}"
    except Exception:  # noqa: BLE001 - the image may not be on this machine
        return None


def _instance_cols() -> list[dict[str, str]]:
    return [{"key": "htp_op", "label": "HTP op", "kind": "str"}, {"key": "qnn_op", "label": "QNN op", "kind": "str"},
            {"key": "type", "label": "Type", "kind": "str"}, {"key": "tid", "label": "TID", "kind": "str"},
            {"key": "cycles", "label": "Cycles", "kind": "num"}, {"key": "time_us", "label": "Time", "kind": "us"},
            {"key": "percent_dominant_path_cycles", "label": "% critical path", "kind": "pct"},
            {"key": "dram_read", "label": "DRAM read", "kind": "bytes"}, {"key": "dram_write", "label": "DRAM write", "kind": "bytes"},
            {"key": "vtcm_read", "label": "VTCM read", "kind": "bytes"}, {"key": "vtcm_write", "label": "VTCM write", "kind": "bytes"},
            {"key": "dram_gbps", "label": "DRAM GB/s", "kind": "num"}, {"key": "vtcm_gbps", "label": "VTCM GB/s", "kind": "num"},
            {"key": "dims", "label": "Dims", "kind": "str"},
            {"key": "dtype", "label": "Dtype", "kind": "str"}]


# --------------------------------------------------------------------------- overview


def llm_rollup(graphs: list[Graph]) -> dict[str, dict[str, Any]]:
    """Sum the parts of split LLM graphs (``part_i_of_n__{prompt|token}_ar.._cl..``) into per-step times."""
    groups: dict[tuple[str, int, int], dict[int, tuple[int, Graph]]] = {}
    for g in graphs:
        m = _GRAPH.match(g.label)
        if m:
            groups.setdefault((m["kind"], int(m["ar"]), int(m["cl"])), {})[int(m["i"])] = (int(m["n"]), g)
    out: dict[str, dict[str, Any]] = {}
    for (kind, ar, cl), parts in groups.items():
        n = next(iter(parts.values()))[0]
        if sorted(parts) != list(range(1, n + 1)):
            continue
        ah = [p[1].out.get("aihub", {}).get("inference_time_ms") for p in parts.values()]
        qh = [p[1].out.get("qhas", {}).get("kpis", {}).get("time_us") for p in parts.values()]
        out[kind] = {"ar": ar, "cl": cl, "parts": n, "aihub_ms": sum(ah) if all(ah) else None,
                     "qhas_ms": sum(qh) / 1e3 if all(qh) else None}
    return out


def _published_llm(cl: int) -> dict[str, Any] | None:
    try:
        from hexprof.reference import published_gemma4_llm_metric

        return published_gemma4_llm_metric(cl, "npu")
    except Exception:  # noqa: BLE001 - reference data is optional
        return None


def _stack_rows(graphs: list[Graph], key: str) -> list[dict[str, Any]]:
    rows = []
    for g in graphs:
        bk = g.out.get("qhas", {}).get("by_kind") or {}
        tot = sum(v.get(key, 0.0) for v in bk.values())
        if tot:
            merged = dict.fromkeys(KINDS, 0.0)
            for k, v in bk.items():
                k2 = "DMA" if k == "DMA set" else (k if k in merged else "Other")
                merged[k2] += 100 * v.get(key, 0.0) / tot
            rows.append({"label": short_label(g.label), "values": [merged[k] for k in KINDS]})
    return rows


def _stack_block(rows: list[dict[str, Any]], series: list[str], title: str) -> dict[str, Any]:
    return {"type": "stacked", "rows": rows, "series": series, "kind": "pct", "max": 100, "title": title,
            "table": {"cols": [{"key": "label", "label": "Graph", "kind": "str"}] + [{"key": k, "label": k, "kind": "pct"} for k in series],
                      "rows": [{"label": r["label"], **dict(zip(series, r["values"]))} for r in rows]}}


def overview_page(graphs: list[Graph], title: str, ctx: EF.Ctx | None = None,
                  requests_sec: dict[str, Any] | None = None, deployed_sec: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = []
    for g in graphs:
        k = g.out.get("qhas", {}).get("kpis", {})
        ah = g.out.get("aihub", {})
        pk = ah.get("estimated_peak_memory_range_mb") or {}
        ot = g.out.get("optrace") or {}
        rows.append({"label": short_label(g.label), "folder": g.label, "href": f"{g.label}.html",
                     "job": g.job_id, "job_url": WORKBENCH_JOB.format(g.job_id) if g.job_id else None,
                     "time_ms": k["time_us"] / 1e3 if k.get("time_us") else None,
                     "util": k.get("percent_utilization"), "ips": k.get("inf_per_s"), "dram": k.get("total_dram"),
                     "dram_gbps": k.get("dram_gbps"), "vtcm": k.get("total_vtcm"), "vtcm_gbps": k.get("vtcm_gbps"),
                     "aihub_ms": ah.get("inference_time_ms"), "peak_mb": pk.get("max"),
                     "load_peak_mb": (ah.get("first_load_peak_memory_range_mb") or {}).get("max"),
                     "published_ms": (g.published or {}).get("inference_time_milliseconds"),
                     "trace_ms": ot["span_us"] / 1e3 if ot.get("span_us") else None})
    tiles: list[dict[str, Any]] = []
    blocks_e2e: list[dict[str, Any]] = []
    roll = llm_rollup(graphs)
    tok, pre = roll.get("token"), roll.get("prompt")
    if tok and tok.get("aihub_ms"):
        tiles.append({"label": "Decode (AI Hub)", "value": f"{1e3 / tok['aihub_ms']:.1f} tok/s",
                      "sub": f"{tok['parts']} token graphs, {tok['aihub_ms']:.1f} ms per token"
                             + (f"; QHAS {1e3 / tok['qhas_ms']:.1f} tok/s" if tok.get("qhas_ms") else "")})
    if pre and pre.get("aihub_ms"):
        tiles.append({"label": "Prefill (AI Hub)", "value": f"{pre['ar'] * 1e3 / pre['aihub_ms']:.0f} tok/s",
                      "sub": f"{pre['ar']}-token chunk in {pre['aihub_ms']:.1f} ms"
                             + (f"; QHAS {pre['ar'] * 1e3 / pre['qhas_ms']:.0f} tok/s" if pre.get("qhas_ms") else "")})
    vis = [g for g in graphs if "vision" in g.label.lower() and g.out.get("aihub", {}).get("inference_time_ms")]
    for g in vis:
        tiles.append({"label": "Vision encoder (AI Hub)", "value": f"{g.out['aihub']['inference_time_ms']:.0f} ms",
                      "sub": "per image"})
    if tok or pre:
        cl = (tok or pre)["cl"]
        pub = _published_llm(cl)
        text = ("Sums of the parts, each profiled on its own: switching between graphs, embedding lookups and sampling "
                "on the CPU are not included, so an end-to-end run is somewhat slower.")
        if pub:
            text += (f" Qualcomm publishes Gemma 4 E4B on the S25 NPU only for a different runtime and precision "
                     f"(q4_0 through llama.cpp/GenieX), at {cl} context: decode {pub['tokens_per_second']:.2f} tok/s, "
                     f"prefill {pub['prefill_tokens_per_second']:.0f} tok/s, time to first token from "
                     f"{pub['time_to_first_token_range_milliseconds']['min']:.0f} ms. Same ballpark expected, not equality.")
        blocks_e2e.append({"type": "note", "text": text})
    if tok and pre and tok.get("aihub_ms") and pre.get("aihub_ms"):
        blocks_e2e.append(e2e_examples(tok, pre, vis))
    sections: list[dict[str, Any]] = []
    io_rows = [{"label": short_label(g.label), **{k: v for k, v in (g.io or {}).items() if k not in ("inputs", "outputs")}}
               for g in graphs if g.io]
    if io_rows:
        sections.append({"title": "Inputs and outputs per graph",
                         "why": "Per call of each compiled graph (static shapes). Exact when `hexprof hw fetch-models` "
                                "has recorded AI Hub's input / output specs, else read from the graph name.",
                         "blocks": [{"type": "note", "text": "", "table": {"open": True, "cols": [
                             {"key": "label", "label": "Graph", "kind": "str"}, {"key": "role", "label": "Role", "kind": "str"},
                             {"key": "image", "label": "Image (px)", "kind": "str"},
                             {"key": "image_patches", "label": "Patches", "kind": "num"},
                             {"key": "tokens_per_call", "label": "Tokens per call", "kind": "num"},
                             {"key": "context_length", "label": "Context", "kind": "num"},
                             {"key": "batch", "label": "Batch", "kind": "num"},
                             {"key": "image_tokens_out", "label": "Image tokens out", "kind": "num"},
                             {"key": "output", "label": "Output", "kind": "str"}], "rows": io_rows}}]})
    if requests_sec:
        sections.append(requests_sec)
    if deployed_sec:
        sections.insert(0, deployed_sec)
    if blocks_e2e:
        sections.append({"title": "End to end (from the parts)", "why": "Decode = 1 ÷ (sum of the token graphs); prefill = "
                         "chunk length ÷ (sum of the prompt graphs); time to first token = vision encoder + prefill "
                         "chunks; end to end = time to first token + one decode step per further token.",
                         "blocks": blocks_e2e})
    cols = [{"key": "label", "label": "Graph", "kind": "str", "link": "href"},
            {"key": "aihub_ms", "label": "AI Hub ms", "kind": "num"}]
    if any(r["job"] for r in rows):  # click through to the Workbench job page that shows this inference time
        cols.append({"key": "job", "label": "AI Hub profile job", "kind": "str", "link": "job_url"})
    if any(r["published_ms"] for r in rows):
        cols.append({"key": "published_ms", "label": "Published ms", "kind": "num"})
    cols += [{"key": "time_ms", "label": "QHAS ms", "kind": "num"},
             {"key": "util", "label": "Util %", "kind": "pct"},
             {"key": "ips", "label": "IPS", "kind": "num"},
             {"key": "dram", "label": "DRAM", "kind": "bytes"},
             {"key": "dram_gbps", "label": "DRAM GB/s", "kind": "num"},
             {"key": "vtcm", "label": "VTCM", "kind": "bytes"},
             {"key": "vtcm_gbps", "label": "VTCM GB/s", "kind": "num"},
             {"key": "peak_mb", "label": "Inference peak MB", "kind": "num"},
             {"key": "load_peak_mb", "label": "Load peak MB", "kind": "num"},
             {"key": "trace_ms", "label": "Trace ms", "kind": "num"},
             {"key": "folder", "label": "Folder", "kind": "str"}]
    ctx_ = ctx or EF.Ctx()
    eff_secs = EF.overview_sections(graphs, short_label, ctx_)
    sections.append({"title": "Graphs", "why": "One row per compiled graph. Click a name for its full report.",
                     "blocks": [{"type": "note", "text": "AI Hub ms and peak memory: AI Hub's profile job (what Workbench "
                                 "shows). Inference peak = extra memory used while running (weights are allocated at "
                                 "load time, so for large models see Load peak). QHAS ms, utilization, inferences per "
                                 "second and memory traffic: the QHAS file of the detailed profiling run. Trace ms: "
                                 "optrace span, converted to ms. Bytes are decimal (1 GB = 10⁹ B), like GB/s.",
                                 "table": {"open": True, "cols": cols, "rows": rows}}]})
    sections += eff_secs
    for key, title_, kind, why in (("aihub_ms", "Run time per graph (AI Hub, ms)", "num", "AI Hub's inference time of one run."),
                                   ("util", "HTP utilization per graph", "pct", QHAS_DEFS["util"]),
                                   ("dram_gbps", "Average DRAM bandwidth per graph (GB/s)", "num",
                                    "DRAM bytes moved ÷ QHAS run time. The S25's memory peak is 84.8 GB/s in theory "
                                    "(LPDDR5X-5300, 64-bit); compare with the measured bandwidth ceiling.")):
        data = [{"label": r["label"], "value": r[key]} for r in rows if r.get(key) is not None]
        if data:
            sections.append({"title": title_, "why": why, "blocks": [{"type": "hbar", "rows": data, "kind": kind,
                                                                        "max": 100 if kind == "pct" else None, "title": title_}]})
    dp_rows = _stack_rows(graphs, "dp_cycles")
    if dp_rows:
        sections.append({"title": "What sets the run time: critical path by unit, per graph", "why": QHAS_DEFS["critical"],
                         "blocks": [_stack_block(dp_rows, KINDS, "critical path by unit")]})
    work_rows = _stack_rows(graphs, "cycles")
    if work_rows:
        sections.append({"title": "Where the work goes: op cycles by unit, per graph", "why": QHAS_DEFS["kinds"],
                         "blocks": [_stack_block(work_rows, KINDS, "op cycles by unit")]})
    # critical path by QNN op type: the 7 op types with the largest share anywhere keep their colour on every graph
    shares: dict[str, dict[str, float]] = {}
    for g in graphs:
        ts = g.out.get("qhas", {}).get("qnn_op_types") or []
        tot = sum(t["dp_cycles"] for t in ts)
        if tot:
            shares[g.label] = {str(t["op"]): 100 * t["dp_cycles"] / tot for t in ts}
    if shares:
        best: dict[str, float] = {}
        for sh in shares.values():
            for op, v in sh.items():
                best[op] = max(best.get(op, 0.0), v)
        keep = [op for op, _ in sorted(best.items(), key=lambda kv: -kv[1])[:7]]
        series = keep + ["Other"]
        op_rows = [{"label": short_label(lbl), "values": [sh.get(op, 0.0) for op in keep] + [sum(v for op, v in sh.items() if op not in keep)]}
                   for lbl, sh in shares.items()]
        sections.append({"title": "What sets the run time: critical path by QNN op type, per graph", "why": QHAS_DEFS["qnn_dp"],
                         "blocks": [_stack_block(op_rows, series, "critical path by QNN op type")]})
    for mem, name in (("dram_bytes", "DRAM"), ("vtcm_bytes", "VTCM")):
        shares_m: dict[str, dict[str, float]] = {}
        for g in graphs:
            ts = g.out.get("qhas", {}).get("qnn_op_types") or []
            tot = sum(t.get(mem, 0.0) for t in ts)
            if tot:
                shares_m[g.label] = {str(t["op"]): 100 * t.get(mem, 0.0) / tot for t in ts}
        if not shares_m:
            continue
        best_m: dict[str, float] = {}
        for sh in shares_m.values():
            for op, v in sh.items():
                best_m[op] = max(best_m.get(op, 0.0), v)
        keep_m = [op for op, _ in sorted(best_m.items(), key=lambda kv: -kv[1])[:7]]
        sections.append({"title": f"Where the {name} traffic goes: by QNN op type, per graph",
                         "why": QHAS_DEFS["mem"].split(" A QNN row")[0],
                         "blocks": [_stack_block([{"label": short_label(lbl), "values": [sh.get(op, 0.0) for op in keep_m]
                                                   + [sum(v for op, v in sh.items() if op not in keep_m)]}
                                                  for lbl, sh in shares_m.items()], keep_m + ["Other"],
                                                 f"{name} bytes by QNN op type")]})
    return {"title": title, "subtitle": f"{len(graphs)} graph(s): QHAS summaries, optrace timelines and AI Hub profiles.",
            "tiles": tiles, "sections": sections,
            "footer": "Generated by hexprof hw report. Definitions quoted from Qualcomm's QAIRT Visualizer."}


def build(paths: Sequence[str | Path], out_dir: str | Path, top: int = 15, bins: int = 400,
          title: str | None = None, compute_json: str | Path | None = None, bandwidth_json: str | Path | None = None,
          theory_gbps: float = EF.THEORETICAL_GBPS, precisions: dict[str, str] | None = None,
          gemma_config: str | None = None, onnx: dict[str, str] | None = None,
          requests: Sequence[str | Path] = ()) -> Path:
    from hexprof.ceilings.experiments import load_ceilings

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graphs = discover(paths)
    if not graphs:
        raise FileNotFoundError("no QHAS (*qhas*.json) or optrace (*profile_trace*.json) files found under "
                                + ", ".join(map(str, paths)))
    ceil = load_ceilings(compute_json, bandwidth_json)
    ctx = EF.Ctx(P={k: v for k, v in (ceil.get("P_tops") or {}).items() if v}, B=ceil.get("B_gbps"),
                 B_series=ceil.get("B_series"), theory_gbps=theory_gbps, precisions=precisions or {},
                 gemma_config=gemma_config, onnx=onnx or {},
                 ceilings_src={"compute": str(compute_json or ""), "bandwidth": str(bandwidth_json or "")})
    print("Ceilings: " + (", ".join(f"P[{k}] = {v:.2f} TOPS" for k, v in ctx.P.items()) or "no P")
          + "; " + (f"B = {ctx.B:.1f} GB/s ({ctx.B_series})" if ctx.B else f"no B (theoretical {theory_gbps:g} GB/s used)"))
    infos = MD.load_infos([Path(p) if Path(p).is_dir() else Path(p).parent for p in paths])
    refs = MD.job_refs(paths)  # export_profiles.json / profile.json: the profile job of every graph
    for g in graphs:
        g.info = infos.get(g.label)
        g.job_id = (refs.get(g.label) or {}).get("job_id")
    if infos:
        print(f"AI Hub model facts for {sum(1 for g in graphs if g.info)} of {len(graphs)} graphs")
    for i, g in enumerate(graphs, 1):
        t = time.time()
        print(f"[{i}/{len(graphs)}] {g.label}: " + ", ".join(f"{p.name} ({_size(p)})" for p in (g.qhas, g.optrace, g.profile) if p),
              flush=True)
        page = graph_page(g, out, top, bins, ctx)
        (out / f"{g.label}.html").write_text(H.render_page(_jsonable(page)), encoding="utf-8")
        print(f"    done in {time.time() - t:.1f} s", flush=True)
    ov = overview_page(graphs, title or "NPU hardware report", ctx, requests_section(requests, graphs),
                       deployment_section(paths))
    (out / "index.html").write_text(H.render_page(_jsonable(ov)), encoding="utf-8")
    write_json(out / "summary.json", {"_ceilings": {"P_tops": ctx.P, "B_gbps": ctx.B, "B_series": ctx.B_series,
                                                    "theoretical_gbps": theory_gbps, **ctx.ceilings_src},
                                      **{g.label: g.out for g in graphs}})
    print(f"\nReport: {out / 'index.html'}  ({len(graphs)} graph pages, CSV tables per graph, summary.json)")
    return out / "index.html"
