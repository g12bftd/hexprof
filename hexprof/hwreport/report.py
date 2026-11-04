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

from hexprof.hwreport import html as H
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
             "HVX (vector units), reference op (scalar code, slow), format change, DMA wait, sync, DMA. Share of all op cycles.",
    "qnn": "QNN ops are the graph operations you exported; each lowers to one or more HTP ops. Cycles = total clock cycles "
           "spent in that op type; % active cycles = cycles actively executing out of its total (Qualcomm).",
    "mem": "Bytes moved between the HTP and DRAM (main memory) per QNN op type. VTCM is the HTP's on-chip memory.",
    "htp": "HTP ops are what actually runs on the NPU. Reference op = “implementation runs purely on scalar code”; "
           "format change = the op converts a tensor layout (Qualcomm). Both are overhead to look for.",
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
    out: dict[str, Any] = field(default_factory=dict)


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


def _fmt_bytes(v: Any) -> str:
    v = _n(v)
    for unit, d in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
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


def graph_page(g: Graph, out_dir: Path, top: int, bins: int) -> dict[str, Any]:
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
        summary["qhas"] = {"version": rep.version, "kpis": k, "by_kind": rep.instances.get("by_kind"),
                           "op_instances": rep.instances.get("count")}
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
            ktable = [{"kind": k, "share": 100 * v["cycles"] / total if total else 0.0, **v} for k, v in kinds]
            types = sorted(inst.get("by_type", {}).items(), key=lambda kv: -kv[1]["cycles"])
            ttable = [{"type": t, "share": 100 * v["cycles"] / total if total else 0.0, **v} for t, v in types]
            _csv(csv_dir / "htp_instances_by_kind.csv", ktable)
            _csv(csv_dir / "htp_instances_by_type.csv", ttable)
            sections.append({"title": "Where the cycles go: HTP op instances by unit", "why": QHAS_DEFS["kinds"],
                             "blocks": [{"type": "hbar", "rows": krows, "kind": "pct", "title": "cycles by unit",
                                         "table": {"cols": [{"key": "kind", "label": "Unit", "kind": "str"},
                                                            {"key": "share", "label": "% of cycles", "kind": "pct"},
                                                            {"key": "cycles", "label": "Cycles", "kind": "num"},
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
            _csv(csv_dir / "qnn_op_types.csv", rep.qnn_op_types)
            _csv(csv_dir / "qnn_op_instances.csv", rep.qnn_op_instances)
            sections.append({"title": "QNN op types", "why": QHAS_DEFS["qnn"], "blocks": [
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
                {"type": "note", "text": f"QNN op instances (nodes), slowest {min(200, len(rep.qnn_op_instances))}:",
                 "table": {"cols": [{"key": "qnn_op", "label": "QNN op", "kind": "str"},
                                    {"key": "cycles", "label": "Cycles", "kind": "num"},
                                    {"key": "percent_active_cycles", "label": "% active", "kind": "pct"},
                                    {"key": "num_htp_ops", "label": "HTP ops", "kind": "num"},
                                    {"key": "dram_read", "label": "DRAM read", "kind": "bytes"},
                                    {"key": "dram_write", "label": "DRAM write", "kind": "bytes"},
                                    {"key": "dominant_path_percent", "label": "% critical path", "kind": "pct"}],
                           "rows": rep.qnn_op_instances[:200]}}]})
            mem = sorted(rep.qnn_op_types, key=lambda r: -(_n(r.get("dram_read")) + _n(r.get("dram_write"))))[:top]
            if any(_n(r.get("dram_read")) + _n(r.get("dram_write")) for r in mem):
                sections.append({"title": "Memory traffic by QNN op type", "why": QHAS_DEFS["mem"], "blocks": [
                    {"type": "stacked", "kind": "bytes", "series": ["DRAM read", "DRAM write"], "title": "DRAM traffic",
                     "rows": [{"label": str(r.get("op")), "values": [_n(r.get("dram_read")), _n(r.get("dram_write"))]} for r in mem],
                     "table": {"cols": [{"key": "op", "label": "QNN op type", "kind": "str"},
                                        {"key": "dram_read", "label": "DRAM read", "kind": "bytes"},
                                        {"key": "dram_write", "label": "DRAM write", "kind": "bytes"},
                                        {"key": "vtcm_read", "label": "VTCM read", "kind": "bytes"},
                                        {"key": "vtcm_write", "label": "VTCM write", "kind": "bytes"}],
                               "rows": mem}}]})
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
                                    {"key": "dram_read", "label": "DRAM read", "kind": "bytes"},
                                    {"key": "dram_write", "label": "DRAM write", "kind": "bytes"},
                                    {"key": "dominant_path_percent", "label": "% critical path", "kind": "pct"}],
                           "rows": rep.htp_op_types}}]})
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
        if inst.get("timeline"):
            sections.append({"title": "QHAS timeline: HTP thread occupancy", "why": QHAS_DEFS["timeline"],
                             "blocks": [{"type": "lanes", "timeline": inst["timeline"], "title": "HTP thread occupancy",
                                         "table": {"cols": [{"key": "lane", "label": "Thread", "kind": "str"},
                                                            {"key": "busy_share", "label": "Busy share", "kind": "frac"}],
                                                   "rows": inst["timeline"]["lanes"]}}]})
    if g.optrace:
        ot = OT.load(g.optrace, bins=bins)
        summary["optrace"] = {"events": ot.events, "span_us": ot.span_us, "lanes": ot.lanes,
                              "counters": [{k: v for k, v in c.items() if k != "series"} for c in ot.counters],
                              "top_ops": ot.ops[:50]}
        _csv(csv_dir / "optrace_threads.csv", ot.lanes)
        _csv(csv_dir / "optrace_ops.csv", ot.ops)
        _csv(csv_dir / "optrace_counters.csv", [{k: v for k, v in c.items() if k != "series"} for c in ot.counters])
        if not g.qhas:
            tiles += [{"label": "Trace span", "value": f"{(ot.span_us or 0) / 1e3:.2f} ms", "sub": f"{_fmt_n(ot.events)} events"}]
        blocks: list[dict[str, Any]] = []
        if ot.lanes:
            lrows = sorted(ot.lanes, key=lambda r: -(r["busy_share"] or 0))
            blocks.append({"type": "hbar", "kind": "frac", "max": 1, "title": "busy share per thread",
                           "rows": [{"label": r["name"], "value": r["busy_share"],
                                     "extra": [["busy", f"{r['busy_us'] / 1e3:.2f} ms"], ["slices", _fmt_n(r["slices"])]]} for r in lrows],
                           "table": {"cols": [{"key": "name", "label": "Thread", "kind": "str"},
                                              {"key": "busy_share", "label": "Busy share", "kind": "frac"},
                                              {"key": "busy_us", "label": "Busy", "kind": "us"},
                                              {"key": "slices", "label": "Slices", "kind": "num"}], "rows": lrows}})
        sections.append({"title": "Optrace: busy share per thread",
                         "why": f"{_fmt_n(ot.events)} trace events over {(ot.span_us or 0) / 1e3:.2f} ms (trace time units: µs). "
                                "Busy share = time covered by top-level op slices on the thread ÷ trace span.",
                         "blocks": blocks or [{"type": "note", "text": "No op slices found."}]})
        if ot.timeline:
            sections.append({"title": "Optrace: timeline occupancy", "why": "Busy share of each trace thread over time.",
                             "blocks": [{"type": "lanes", "timeline": ot.timeline, "title": "optrace occupancy",
                                         "table": {"cols": [{"key": "name", "label": "Thread", "kind": "str"},
                                                            {"key": "busy_share", "label": "Busy share", "kind": "frac"},
                                                            {"key": "busy_us", "label": "Busy", "kind": "us"}],
                                                   "rows": ot.lanes}}]})
        if ot.counters:
            cblocks: list[dict[str, Any]] = []
            for c in ot.counters[:6]:
                cblocks.append({"type": "line", "points": c["series"], "kind": "num", "title": c["name"],
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
                                                      lambda r: [["total", f"{r['total_us'] / 1e3:.3f} ms"], ["count", _fmt_n(r["count"])],
                                                                 ["mean", f"{r['mean_us']:.2f} µs"], ["max", f"{r['max_us']:.2f} µs"]]),
                                  "table": {"cols": [{"key": "name", "label": "Op", "kind": "str"},
                                                     {"key": "total_us", "label": "Total", "kind": "us"},
                                                     {"key": "count", "label": "Count", "kind": "num"},
                                                     {"key": "mean_us", "label": "Mean", "kind": "us"},
                                                     {"key": "max_us", "label": "Max", "kind": "us"}], "rows": ot.ops[:500]}}]})
        if ot.other_phases:  # instant / flow / async markers carry no duration or value
            sections[-1]["blocks"].append({"type": "note", "text": "Other trace events (markers, not timed): " +
                                           ", ".join(f"{n:,} × ph={k!r}" for k, n in sorted(ot.other_phases.items()))})
    if g.profile:
        prof = json.loads(Path(g.profile).read_text())
        layers = sorted(prof.get("execution_detail", []), key=lambda r: -_n(r.get("execution_time")))
        es = prof.get("execution_summary", {})
        summary["aihub"] = {"inference_time_ms": _n(es.get("estimated_inference_time")) / 1e3}
        units: dict[str, int] = {}
        for r in layers:
            units[str(r.get("compute_unit"))] = units.get(str(r.get("compute_unit")), 0) + 1
        tiles.append({"label": "AI Hub inference time", "value": f"{_n(es.get('estimated_inference_time')) / 1e3:.2f} ms",
                      "sub": ", ".join(f"{k} {v}" for k, v in sorted(units.items())) + " layers"})
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
    g.out = summary
    files = ", ".join(p.name for p in (g.qhas, g.optrace, g.profile) if p)
    return {"title": short_label(g.label), "subtitle": f"{g.label} · sources: {files}", "nav": {"href": "index.html", "text": "← All graphs"},
            "tiles": tiles, "sections": sections, "warnings": warnings,
            "footer": f"Tables as CSV in {g.label}/. Definitions quoted from Qualcomm's QAIRT Visualizer. "
                      f"QHAS schema {summary.get('qhas', {}).get('version', '–')}."}


def _instance_cols() -> list[dict[str, str]]:
    return [{"key": "htp_op", "label": "HTP op", "kind": "str"}, {"key": "qnn_op", "label": "QNN op", "kind": "str"},
            {"key": "type", "label": "Type", "kind": "str"}, {"key": "tid", "label": "TID", "kind": "str"},
            {"key": "cycles", "label": "Cycles", "kind": "num"}, {"key": "time_us", "label": "Time", "kind": "us"},
            {"key": "percent_dominant_path_cycles", "label": "% critical path", "kind": "pct"},
            {"key": "dram_read", "label": "DRAM read", "kind": "bytes"}, {"key": "dram_write", "label": "DRAM write", "kind": "bytes"},
            {"key": "dram_gbps", "label": "DRAM GB/s", "kind": "num"}, {"key": "dims", "label": "Dims", "kind": "str"},
            {"key": "dtype", "label": "Dtype", "kind": "str"}]


# --------------------------------------------------------------------------- overview


def overview_page(graphs: list[Graph], title: str) -> dict[str, Any]:
    rows = []
    for g in graphs:
        q = g.out.get("qhas", {})
        k = q.get("kpis", {})
        rows.append({"label": short_label(g.label), "folder": g.label, "href": f"{g.label}.html", "time_ms": (k.get("time_us") or 0) / 1e3 if k.get("time_us") else None,
                     "util": k.get("percent_utilization"), "ips": k.get("inf_per_s"), "dram": k.get("total_dram"),
                     "dram_gbps": k.get("dram_gbps"), "vtcm": k.get("total_vtcm"),
                     "aihub_ms": g.out.get("aihub", {}).get("inference_time_ms"),
                     "trace_ms": (g.out.get("optrace", {}).get("span_us") or 0) / 1e3 if g.out.get("optrace") else None})
    sections: list[dict[str, Any]] = [{"title": "Graphs", "why": "One row per compiled graph. Click a name for its full report.",
                                       "blocks": [{"type": "note", "text": "Time and utilization from QHAS; AI Hub time from the profile job.",
                                                   "table": {"open": True, "cols": [
                                                       {"key": "label", "label": "Graph", "kind": "str", "link": "href"},
                                                       {"key": "time_ms", "label": "QHAS ms", "kind": "num"},
                                                       {"key": "aihub_ms", "label": "AI Hub ms", "kind": "num"},
                                                       {"key": "util", "label": "Util %", "kind": "pct"},
                                                       {"key": "ips", "label": "IPS", "kind": "num"},
                                                       {"key": "dram", "label": "DRAM", "kind": "bytes"},
                                                       {"key": "dram_gbps", "label": "DRAM GB/s", "kind": "num"},
                                                       {"key": "vtcm", "label": "VTCM", "kind": "bytes"},
                                                       {"key": "trace_ms", "label": "Trace ms", "kind": "num"},
                                                       {"key": "folder", "label": "Folder", "kind": "str"}],
                                                       "rows": rows}}]}]
    for key, title_, kind, why in (("time_ms", "Run time per graph (ms)", "num", "QHAS run time of one inference."),
                                   ("util", "HTP utilization per graph", "pct", QHAS_DEFS["util"]),
                                   ("dram_gbps", "Average DRAM bandwidth per graph (GB/s)", "num",
                                    "DRAM bytes moved ÷ run time. Compare with the measured bandwidth ceiling.")):
        data = [{"label": r["label"], "value": r[key]} for r in rows if r.get(key) is not None]
        if data:
            sections.append({"title": title_, "why": why, "blocks": [{"type": "hbar", "rows": data, "kind": kind,
                                                                        "max": 100 if kind == "pct" else None, "title": title_}]})
    kind_rows = []
    for g in graphs:
        bk = g.out.get("qhas", {}).get("by_kind") or {}
        tot = sum(v["cycles"] for v in bk.values())
        if tot:
            merged = dict.fromkeys(KINDS, 0.0)
            for k, v in bk.items():
                k2 = "DMA" if k == "DMA set" else (k if k in merged else "Other")
                merged[k2] += 100 * v["cycles"] / tot
            kind_rows.append({"label": short_label(g.label), "values": [merged[k] for k in KINDS]})
    if kind_rows:
        sections.append({"title": "Where the cycles go, per graph", "why": QHAS_DEFS["kinds"], "blocks": [
            {"type": "stacked", "rows": kind_rows, "series": KINDS, "kind": "pct", "max": 100, "title": "cycles by unit",
             "table": {"cols": [{"key": "label", "label": "Graph", "kind": "str"}] +
                       [{"key": k, "label": k, "kind": "pct"} for k in KINDS],
                       "rows": [{"label": r["label"], **dict(zip(KINDS, r["values"]))} for r in kind_rows]}}]})
    return {"title": title, "subtitle": f"{len(graphs)} graph(s): QHAS summaries, optrace timelines and AI Hub profiles.",
            "sections": sections,
            "footer": "Generated by hexprof hw report. Definitions quoted from Qualcomm's QAIRT Visualizer."}


def build(paths: Sequence[str | Path], out_dir: str | Path, top: int = 15, bins: int = 400,
          title: str | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graphs = discover(paths)
    if not graphs:
        raise FileNotFoundError("no QHAS (*qhas*.json) or optrace (*profile_trace*.json) files found under "
                                + ", ".join(map(str, paths)))
    for i, g in enumerate(graphs, 1):
        t = time.time()
        print(f"[{i}/{len(graphs)}] {g.label}: " + ", ".join(f"{p.name} ({_size(p)})" for p in (g.qhas, g.optrace, g.profile) if p),
              flush=True)
        page = graph_page(g, out, top, bins)
        (out / f"{g.label}.html").write_text(H.render_page(_jsonable(page)), encoding="utf-8")
        print(f"    done in {time.time() - t:.1f} s", flush=True)
    ov = overview_page(graphs, title or "NPU hardware report")
    (out / "index.html").write_text(H.render_page(_jsonable(ov)), encoding="utf-8")
    write_json(out / "summary.json", {g.label: g.out for g in graphs})
    print(f"\nReport: {out / 'index.html'}  ({len(graphs)} graph pages, CSV tables per graph, summary.json)")
    return out / "index.html"
