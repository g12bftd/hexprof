"""Optrace (QNN HTP op trace) reader: Chrome trace events, streamed.

Mirrors how QAIRT Visualizer reads the file: metadata events (``ph: "M"``)
name processes and threads, complete events (``ph: "X"``) are op slices on a
(process, thread) lane nested by ``args.stackDepth``, and counter events
(``ph: "C"``) are memory tracks whose value is the first entry of ``args``
(e.g. "Memory Bandwidth"). Times follow the Chrome trace convention (µs).
"""

from __future__ import annotations

from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hexprof.hwreport import jsonstream


@dataclass
class OptraceReport:
    path: str
    events: int
    t0: float | None
    t1: float | None
    processes: dict[str, str]
    lanes: list[dict[str, Any]]  # per (pid, tid): name, busy_us, busy_share, slices
    ops: list[dict[str, Any]]  # per op name: count, total_us, max_us, mean_us
    counters: list[dict[str, Any]]  # per (process, counter name): stats + downsampled series
    timeline: dict[str, Any] | None = None
    other_phases: dict[str, int] = field(default_factory=dict)

    @property
    def span_us(self) -> float | None:
        return None if self.t0 is None or self.t1 is None else self.t1 - self.t0


def _f(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(path: str | Path, bins: int = 400, max_points: int = 1500) -> OptraceReport:
    path = str(path)
    prefix = "item" if jsonstream.top_level_is_list(path) else "traceEvents.item"
    proc_name: dict[Any, str] = {}
    thread_name: dict[tuple[Any, Any], str] = {}
    lane_ids: dict[tuple[Any, Any], int] = {}
    starts, durs, lane_of = array("d"), array("d"), array("i")
    op_stats: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0])  # count, total, max
    counters: dict[tuple[Any, str], tuple[array, array]] = {}
    other: dict[str, int] = defaultdict(int)
    n = 0
    t0, t1 = float("inf"), float("-inf")
    for ev in jsonstream.items(path, prefix):
        n += 1
        ph = ev.get("ph")
        if ph == "M":
            args = ev.get("args") or {}
            if ev.get("name") == "process_name" and "name" in args:
                proc_name[ev.get("pid")] = str(args["name"])
            elif ev.get("name") == "thread_name" and "name" in args:
                thread_name[(ev.get("pid"), ev.get("tid"))] = str(args["name"])
        elif ph == "X":
            ts, dur = _f(ev.get("ts")), _f(ev.get("dur")) or 0.0
            if ts is None:
                continue
            name = str(ev.get("name", "?"))
            st = op_stats[name]
            st[0] += 1
            st[1] += dur
            st[2] = max(st[2], dur)
            t0, t1 = min(t0, ts), max(t1, ts + dur)
            depth = (ev.get("args") or {}).get("stackDepth") or 0
            if not depth:  # top-level slices only, so nesting is not double counted
                key = (ev.get("pid"), ev.get("tid"))
                lane_of.append(lane_ids.setdefault(key, len(lane_ids)))
                starts.append(ts)
                durs.append(dur)
        elif ph == "C":
            ts = _f(ev.get("ts"))
            args = ev.get("args") or {}
            if ts is None or not args:
                continue
            for k, v in args.items():
                val = _f(v)
                if val is None:
                    continue
                cname = str(ev.get("name", "counter")) + ("" if len(args) == 1 else f" · {k}")
                xs, ys = counters.setdefault((ev.get("pid"), cname), (array("d"), array("d")))
                xs.append(ts)
                ys.append(val)
        else:
            other[str(ph)] += 1

    if t0 == float("inf"):
        t0 = t1 = None  # type: ignore[assignment]
    s = np.frombuffer(starts, dtype=np.float64)
    d = np.frombuffer(durs, dtype=np.float64)
    li = np.frombuffer(lane_of, dtype=np.int32)
    span = (t1 - t0) if t0 is not None else None

    def pname(pid: Any) -> str:
        return proc_name.get(pid, f"Process {pid}")

    lanes = []
    for (pid, tid), idx in sorted(lane_ids.items(), key=lambda kv: kv[1]):
        m = li == idx
        busy = float(d[m].sum())
        lanes.append({"pid": pid, "tid": tid, "name": f"{pname(pid)} · {thread_name.get((pid, tid), f'Thread {tid}')}",
                      "slices": int(m.sum()), "busy_us": busy, "busy_share": busy / span if span else None})
    ops = sorted(({"name": k, "count": int(v[0]), "total_us": v[1], "max_us": v[2], "mean_us": v[1] / v[0] if v[0] else 0.0}
                  for k, v in op_stats.items()), key=lambda r: -r["total_us"])
    ctr = []
    for (pid, cname), (xs, ys) in counters.items():
        x, y = np.frombuffer(xs, dtype=np.float64), np.frombuffer(ys, dtype=np.float64)
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        ctr.append({"process": pname(pid), "name": cname, "points": int(x.size),
                    "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()),
                    "time_weighted_mean": _time_weighted_mean(x, y),
                    "series": _downsample(x, y, max_points)})
    ctr.sort(key=lambda c: (0 if "bandwidth" in c["name"].lower() else 1, c["process"], c["name"]))
    timeline = None
    if span and s.size:
        timeline = _occupancy(s, d, li, [lane["name"] for lane in lanes], t0, t1, bins)
    return OptraceReport(path, n, t0, t1, {str(k): v for k, v in proc_name.items()}, lanes, ops, ctr, timeline, dict(other))


def _time_weighted_mean(x: np.ndarray, y: np.ndarray) -> float:
    """Counters are step functions: each value holds until the next sample."""
    if x.size < 2:
        return float(y.mean()) if y.size else 0.0
    w = np.diff(x)
    return float((y[:-1] * w).sum() / w.sum()) if w.sum() > 0 else float(y.mean())


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> list[list[float]]:
    if x.size <= max_points:
        return [[float(a), float(b)] for a, b in zip(x, y)]
    edges = np.linspace(x[0], x[-1], max_points + 1)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, max_points - 1)
    sums = np.bincount(idx, weights=y, minlength=max_points)
    cnt = np.bincount(idx, minlength=max_points)
    mx = np.full(max_points, -np.inf)
    np.maximum.at(mx, idx, y)
    centers = (edges[:-1] + edges[1:]) / 2
    keep = cnt > 0
    # mean per bucket keeps the level; the bucket max is kept alongside for peaks
    return [[float(c), float(s / k), float(m)] for c, s, k, m in zip(centers[keep], sums[keep], cnt[keep], mx[keep])]


def _occupancy(s: np.ndarray, d: np.ndarray, li: np.ndarray, names: list[str], t0: float, t1: float,
               bins: int) -> dict[str, Any]:
    edges = np.linspace(t0, t1, bins + 1)
    width = edges[1] - edges[0]
    lanes = []
    for i, name in enumerate(names):
        m = li == i
        ss, dd = s[m], d[m]
        if ss.size == 0:
            continue
        o = np.argsort(ss)
        ss, dd = ss[o], dd[o]
        ee = ss + dd
        csum = np.concatenate([[0.0], np.cumsum(dd)])
        ends = np.maximum.accumulate(ee)
        idx = np.searchsorted(ends, edges, side="right")
        nxt = np.minimum(idx, ss.size - 1)
        partial = np.where((idx < ss.size) & (ss[nxt] < edges), np.minimum(edges, ee[nxt]) - ss[nxt], 0.0)
        upto = csum[idx] + np.clip(partial, 0, None)
        frac = np.clip(np.diff(upto) / width, 0.0, 1.0)
        lanes.append({"lane": name, "busy": [round(float(x), 4) for x in frac]})
    return {"t0": t0, "t1": t1, "unit": "µs", "lanes": lanes}


def label_of(path: str | Path) -> str:
    p = Path(path)
    return p.parent.parent.name if p.parent.name in ("qhas", "optrace") else p.stem
