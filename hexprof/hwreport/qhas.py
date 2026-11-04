"""QHAS (QNN HTP Analysis Summary) reader.

Column names and their meaning are Qualcomm's (QAIRT Visualizer tooltips):

htp_overall_summary (one row per HTP core)
    percent_utilization  "Percentage of overall timeline HTP Utilization"
    percent_idle, inf_per_s ("Inferences per second of overall graph"), time_us,
    timeline_cycles ("Total number of cycles taken by operations for the core"),
    total_dram[_read|_write] / total_vtcm[_read|_write] (bytes moved),
    htp_resources.data: per HTP thread + resource type (HMX, HVX, ...):
        utilization ("Percentage of utilization of this resource"), cycles_used,
        timeline_cycles, start_cycle, end_cycle
qnn_op_types / qnn_op_instances_nodes / htp_op_types
    cycles, percent_active_cycles, dram_read/write, vtcm_read/write, num_htp_ops,
    instances, reference_op ("runs purely on scalar code"), format_change,
    num/percent_dominant_path_cycles_htp_<core> (cycles on the critical path)
htp_op_instances (one row per executed HTP op; the big table)
    htp, op_id, htp_op, qnn_op, tid, start_cycle, cycles, dram_read/write,
    vtcm_read/write, flags hmx / hvx / dma / dma_set / dma_wait / sync /
    ref_op / fmt_change, num/percent_dominant_path_cycles, dims, dtype, ...
"""

from __future__ import annotations

import heapq
import re
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hexprof.hwreport import jsonstream

SUPPORTED_MAJOR = 1  # QAIRT Visualizer accepts "1.0.0 - 2.0.0"
FLAG_LABELS = {"hmx": "HMX", "hvx": "HVX", "dma": "DMA", "dma_set": "DMA Set", "dma_wait": "DMA Wait",
               "sync": "Sync", "ref_op": "Ref Op", "fmt_change": "Format Change"}
# One kind per op instance for time shares (an op can use several units; first match wins).
KIND_ORDER = [("hmx", "HMX (matrix)"), ("hvx", "HVX (vector)"), ("ref_op", "Reference op (scalar)"),
              ("fmt_change", "Format change"), ("dma_wait", "DMA wait"), ("sync", "Sync"),
              ("dma", "DMA"), ("dma_set", "DMA set")]
_DP = re.compile(r"^(num|percent)_dominant_path_cycles_htp_(\d+)$")


def num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "").rstrip("%")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def flag(v: Any) -> bool:
    if v is True or v is False or v is None:
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


_TEXT_COLS = {"op", "qnn_op", "htp_op", "htp", "tid", "type", "dims", "dtype", "input_nodes", "op_id"}


def _conv(k: str, v: Any) -> Any:
    """Numeric columns -> float (QHAS sometimes stores numbers as strings); names/ids untouched."""
    if k in _TEXT_COLS or isinstance(v, bool):
        return v
    n = num(v)
    return n if n is not None else v


def _norm_row(r: dict[str, Any]) -> dict[str, Any]:
    """Numbers as floats; dominant-path columns gathered per HTP core."""
    out: dict[str, Any] = {}
    dp: dict[int, dict[str, float | None]] = defaultdict(dict)
    for k, v in r.items():
        m = _DP.match(k)
        if m:
            dp[int(m.group(2))][m.group(1)] = num(v)
            continue
        out[k] = _conv(k, v)
    if dp:
        out["dominant_path"] = [{"core": c, "cycles": d.get("num"), "percent": d.get("percent")} for c, d in sorted(dp.items())]
        out["dominant_path_cycles"] = sum((d.get("num") or 0.0) for d in dp.values())
        out["dominant_path_percent"] = max((d.get("percent") or 0.0) for d in dp.values())
    return out


def instance_kind(r: dict[str, Any]) -> str:
    for key, label in KIND_ORDER:
        if flag(r.get(key)):
            return label
    return "Other"


def instance_type(r: dict[str, Any]) -> str:
    """Qualcomm's "Type" column: every active resource/flag, comma-joined."""
    active = [label for key, label in FLAG_LABELS.items() if flag(r.get(key))]
    return ", ".join(active) if active else "N/A"


@dataclass
class QhasReport:
    path: str
    header: dict[str, Any]
    overall: list[dict[str, Any]]
    resources: list[dict[str, Any]]
    qnn_op_types: list[dict[str, Any]]
    qnn_op_instances: list[dict[str, Any]]
    htp_op_types: list[dict[str, Any]]
    instances: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def version(self) -> str:
        v = self.header.get("version") or {}
        if isinstance(v, dict):
            return f"{v.get('major', '?')}.{v.get('minor', '?')}.{v.get('patch', '?')}"
        return str(v)

    def cycles_to_us(self, core: Any = None) -> float | None:
        """µs per cycle from the overall summary (time_us / timeline_cycles)."""
        rows = [r for r in self.overall if core is None or str(r.get("htp")) == str(core)] or self.overall
        for r in rows:
            t, c = num(r.get("time_us")), num(r.get("timeline_cycles"))
            if t and c:
                return t / c
        return None


def load_overall(path: str | Path) -> QhasReport:
    """Header + overall summary only (one streaming pass; for headline numbers)."""
    path = str(path)
    header = jsonstream.first(path, "header", {}) or {}
    overall = [{k: _conv(k, v) for k, v in r.items() if k != "htp_resources"}
               for r in jsonstream.items(path, "data.htp_overall_summary.data.item")]
    return QhasReport(path, header, overall, [], [], [], [])


def load(path: str | Path, top_n: int = 200, timeline_bins: int = 400) -> QhasReport:
    path = str(path)
    header = jsonstream.first(path, "header", {}) or {}
    warnings = []
    art = str(header.get("artifact_type", "")).lower()
    if art and art != "qhas":
        warnings.append(f"header.artifact_type is {header.get('artifact_type')!r}, expected 'QHAS'")
    ver = header.get("version") or {}
    if isinstance(ver, dict) and ver.get("major") not in (None, SUPPORTED_MAJOR):
        warnings.append(f"QHAS schema {ver} is outside the 1.x range this reader was built for")

    overall, resources = [], []
    for r in jsonstream.items(path, "data.htp_overall_summary.data.item"):
        res = (r.get("htp_resources") or {}).get("data") if isinstance(r.get("htp_resources"), dict) else r.get("htp_resources")
        row = {k: _conv(k, v) for k, v in r.items() if k != "htp_resources"}
        overall.append(row)
        for x in res or []:
            resources.append({"htp": r.get("htp"), **{k: _conv(k, v) for k, v in x.items()}})
    qnn_types = [_norm_row(r) for r in jsonstream.items(path, "data.qnn_op_types.data.item")]
    qnn_inst = [_norm_row(r) for r in jsonstream.items(path, "data.qnn_op_instances_nodes.data.item")]
    htp_types = [_norm_row(r) for r in jsonstream.items(path, "data.htp_op_types.data.item")]
    for rows in (qnn_types, qnn_inst, htp_types):
        rows.sort(key=lambda r: -(num(r.get("cycles")) or 0.0))
    rep = QhasReport(path, header, overall, resources, qnn_types, qnn_inst, htp_types, warnings=warnings)
    rep.instances = _aggregate_instances(path, top_n, timeline_bins, rep)
    return rep


_FLAG_KEYS = tuple(FLAG_LABELS)
_KIND_IDX = [(_FLAG_KEYS.index(k), label) for k, label in KIND_ORDER]


def _kind_and_type(flags: tuple[bool, ...], cache: dict[tuple[bool, ...], tuple[str, str]]) -> tuple[str, str]:
    kt = cache.get(flags)
    if kt is None:
        kind = next((label for i, label in _KIND_IDX if flags[i]), "Other")
        typ = ", ".join(FLAG_LABELS[k] for k, f in zip(_FLAG_KEYS, flags) if f) or "N/A"
        kt = cache[flags] = (kind, typ)
    return kt


def _slim(r: dict[str, Any], c: float, s: float | None, kind: str, typ: str) -> dict[str, Any]:
    return {
        "op_id": r.get("op_id"), "htp_op": r.get("htp_op"), "qnn_op": r.get("qnn_op"), "htp": r.get("htp"),
        "tid": r.get("tid"), "type": typ, "kind": kind, "start_cycle": s, "cycles": c,
        "dram_read": num(r.get("dram_read")), "dram_write": num(r.get("dram_write")),
        "vtcm_read": num(r.get("vtcm_read")), "vtcm_write": num(r.get("vtcm_write")),
        "num_dominant_path_cycles": num(r.get("num_dominant_path_cycles")),
        "percent_dominant_path_cycles": num(r.get("percent_dominant_path_cycles")),
        "dims": r.get("dims"), "dtype": r.get("dtype"),
    }


def _aggregate_instances(path: str, top_n: int, bins: int, rep: QhasReport, progress_every: int = 1_000_000) -> dict[str, Any]:
    """Single streaming pass over htp_op_instances (can be millions of rows)."""
    count = 0
    total_cycles = 0.0
    by_kind: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "cycles": 0.0, "dram_bytes": 0.0,
                                                                "dp_cycles": 0.0})
    by_type: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "cycles": 0.0, "dp_cycles": 0.0})
    kt_cache: dict[tuple[bool, ...], tuple[str, str]] = {}
    lanes: dict[str, int] = {}
    starts, cycles_arr, lane_idx = array("d"), array("d"), array("i")
    top_cycles: list[tuple[float, int, dict[str, Any]]] = []
    top_dp: list[tuple[float, int, dict[str, Any]]] = []
    t_start = time.time()
    for r in jsonstream.items(path, "data.htp_op_instances.data.item"):
        count += 1
        if progress_every and count % progress_every == 0:
            print(f"    … {count:,} HTP op instances ({time.time() - t_start:.0f} s)", flush=True)
        c = num(r.get("cycles")) or 0.0
        total_cycles += c
        kind, typ = _kind_and_type(tuple(flag(r.get(k)) for k in _FLAG_KEYS), kt_cache)
        k = by_kind[kind]
        k["count"] += 1
        k["cycles"] += c
        k["dram_bytes"] += (num(r.get("dram_read")) or 0.0) + (num(r.get("dram_write")) or 0.0)
        dpc = num(r.get("num_dominant_path_cycles")) or 0.0
        k["dp_cycles"] += dpc
        t = by_type[typ]
        t["count"] += 1
        t["cycles"] += c
        t["dp_cycles"] += dpc
        lane = f"HTP {r.get('htp', 0)} · TID {r.get('tid', '?')}"
        li = lanes.setdefault(lane, len(lanes))
        s = num(r.get("start_cycle"))
        if s is not None:
            starts.append(s)
            cycles_arr.append(c)
            lane_idx.append(li)
        slim = None
        if len(top_cycles) < top_n or c > top_cycles[0][0]:
            slim = _slim(r, c, s, kind, typ)
            (heapq.heappush if len(top_cycles) < top_n else heapq.heapreplace)(top_cycles, (c, count, slim))
        dpv = num(r.get("percent_dominant_path_cycles")) or 0.0
        if dpv > 0 and (len(top_dp) < top_n or dpv > top_dp[0][0]):
            slim = slim or _slim(r, c, s, kind, typ)
            (heapq.heappush if len(top_dp) < top_n else heapq.heapreplace)(top_dp, (dpv, count, slim))

    timeline = _lane_occupancy(np.frombuffer(starts, dtype=np.float64), np.frombuffer(cycles_arr, dtype=np.float64),
                               np.frombuffer(lane_idx, dtype=np.int32), lanes, bins)
    us_per_cycle = rep.cycles_to_us()
    top = [x[2] for x in sorted(top_cycles, key=lambda t: -t[0])]
    dp_rows = [x[2] for x in sorted(top_dp, key=lambda t: -t[0])]
    if us_per_cycle:
        for x in {id(x): x for x in top + dp_rows}.values():
            x["time_us"] = x["cycles"] * us_per_cycle
            b = (x["dram_read"] or 0.0) + (x["dram_write"] or 0.0)
            x["dram_gbps"] = b / (x["time_us"] * 1e-6) / 1e9 if x["time_us"] else None
    return {
        "count": count,
        "total_cycles": total_cycles,
        "by_kind": dict(by_kind),
        "by_type": dict(by_type),
        "top_by_cycles": top,
        "top_by_dominant_path": dp_rows,
        "timeline": timeline,
        "us_per_cycle": us_per_cycle,
    }


def _lane_occupancy(start: np.ndarray, dur: np.ndarray, lane: np.ndarray, lanes: dict[str, int],
                    bins: int) -> dict[str, Any] | None:
    """Busy fraction per lane per time bin (exact for non-overlapping ops on a lane)."""
    if start.size == 0:
        return None
    t0 = float(start.min())
    t1 = float((start + dur).max())
    if t1 <= t0:
        return None
    edges = np.linspace(t0, t1, bins + 1)
    width = edges[1] - edges[0]
    out = []
    for name, li in sorted(lanes.items(), key=lambda kv: kv[1]):
        m = lane == li
        s, d = start[m], dur[m]
        if s.size == 0:
            continue
        order = np.argsort(s)
        s, d = s[order], d[order]
        e = s + d
        csum = np.concatenate([[0.0], np.cumsum(d)])
        ends_sorted = np.maximum.accumulate(e)  # tolerate small overlaps
        idx = np.searchsorted(ends_sorted, edges, side="right")
        busy_upto = csum[idx]
        nxt = np.minimum(idx, s.size - 1)
        partial = np.where((idx < s.size) & (s[nxt] < edges), np.minimum(edges, e[nxt]) - s[nxt], 0.0)
        busy_upto = busy_upto + np.clip(partial, 0, None)
        frac = np.clip(np.diff(busy_upto) / width, 0.0, 1.0)
        out.append({"lane": name, "busy": [round(float(x), 4) for x in frac],
                    "busy_share": float(d.sum() / (t1 - t0))})
    return {"t0": t0, "t1": t1, "unit": "cycles", "lanes": out}


def overall_kpis(rep: QhasReport) -> dict[str, Any]:
    """Headline numbers of the first HTP core (+ derived DRAM / VTCM bandwidth)."""
    if not rep.overall:
        return {}
    r = rep.overall[0]
    t_us = num(r.get("time_us"))
    out = {k: num(r.get(k)) for k in ("percent_utilization", "percent_idle", "inf_per_s", "time_us", "timeline_cycles",
                                      "total_dram", "total_dram_read", "total_dram_write", "total_vtcm",
                                      "total_vtcm_read", "total_vtcm_write")}
    out["htp"] = r.get("htp")
    if t_us and out.get("total_dram") is not None:
        out["dram_gbps"] = out["total_dram"] / (t_us * 1e-6) / 1e9
    if t_us and out.get("total_vtcm") is not None:
        out["vtcm_gbps"] = out["total_vtcm"] / (t_us * 1e-6) / 1e9
    if t_us and out.get("timeline_cycles"):
        out["clock_mhz"] = out["timeline_cycles"] / t_us
    out["cores"] = len(rep.overall)
    return out


def label_of(path: str | Path) -> str:
    p = Path(path)
    return p.parent.parent.name if p.parent.name in ("qhas", "optrace") else p.stem
