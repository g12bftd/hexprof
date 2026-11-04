"""Optrace (QNN HTP op trace) reader: Chrome trace events, streamed.

Mirrors how QAIRT Visualizer reads the file: metadata events (``ph: "M"``)
name processes and threads, complete events (``ph: "X"``) are op slices on a
(process, thread) lane nested by ``args.stackDepth``, and counter events
(``ph: "C"``) are memory tracks whose value is the first entry of ``args``
(e.g. "Memory Bandwidth"). Times follow the Chrome trace convention (µs).
"""

from __future__ import annotations

import math
import re
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
    unit: str = "µs"  # unit of all times below; "µs" once identified, else "trace units"
    unit_note: str = "Times assumed to be µs (Chrome trace convention); not cross-checked."
    declared_unit: str | None = None  # "displayTimeUnit" in the file, if any (display hint only)
    raw_span: float | None = None  # span in the file's own timestamp units

    @property
    def span(self) -> float | None:
        return None if self.t0 is None or self.t1 is None else self.t1 - self.t0

    @property
    def span_us(self) -> float | None:
        return self.span if self.unit == "µs" else None

    def rescale(self, factor: float, unit: str, note: str) -> None:
        """Convert every time quantity (not shares) by ``factor`` into ``unit``."""
        if self.t0 is not None:
            self.t0, self.t1 = self.t0 * factor, self.t1 * factor
        for ln in self.lanes:
            ln["busy_us"] *= factor
        for o in self.ops:
            for k in ("total_us", "max_us", "mean_us"):
                o[k] *= factor
        for c in self.counters:
            c["series"] = [[pt[0] * factor, *pt[1:]] for pt in c["series"]]
        if self.timeline:
            self.timeline.update(t0=self.timeline["t0"] * factor, t1=self.timeline["t1"] * factor, unit=unit)
        self.unit, self.unit_note = unit, note

    def mark_unknown(self, note: str) -> None:
        self.unit, self.unit_note = "trace units", note
        if self.timeline:
            self.timeline["unit"] = "trace units"


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
    rep = OptraceReport(path, n, t0, t1, {str(k): v for k, v in proc_name.items()}, lanes, ops, ctr, timeline, dict(other))
    rep.declared_unit = declared_unit(path)
    rep.raw_span = span
    return rep


def declared_unit(path: str | Path, window: int = 1 << 16) -> str | None:
    """``displayTimeUnit`` from the head/tail of the file (top-level keys sit at either end)."""
    p = Path(path)
    with open(p, "rb") as f:
        head = f.read(window)
        f.seek(max(0, p.stat().st_size - window))
        tail = f.read()
    m = re.search(rb'"displayTimeUnit"\s*:\s*"(\w+)"', head + tail)
    return m.group(1).decode() if m else None


def calibrate(rep: OptraceReport, timeline_cycles: float | None, time_us: float | None, tol: float = 1.25) -> None:
    """Identify the timestamp unit by matching the trace span against the same graph's QHAS run.

    QNN HTP traces are not always in µs. The span is compared with the QHAS run in HTP cycles,
    µs and ns; the first match within ``tol`` sets the conversion to µs. No match: times stay in
    "trace units" (shares and busy fractions are unit-free and stay valid).
    """
    span = rep.span
    if not span:
        return
    checks = []
    if timeline_cycles and time_us:
        checks.append(("HTP clock cycles", span / timeline_cycles, time_us / timeline_cycles))
    if time_us:
        checks += [("µs", span / time_us, 1.0), ("ns", span / (time_us * 1e3), 1e-3)]
    # closest match wins (a ~1 GHz clock makes cycles and ns look alike; the nearer one is right)
    for name, ratio, factor in sorted(checks, key=lambda c: abs(math.log(c[1])) if c[1] > 0 else math.inf)[:1]:
        if 1 / tol <= ratio <= tol:
            note = (f"Trace timestamps are in {name}: the trace spans {ratio:.3f}× the QHAS run "
                    f"({time_us / 1e3:.3f} ms), so times were converted to µs" + (
                        f" with QHAS's {factor * 1e3:.4f} ns per cycle." if name.startswith("HTP") else "."))
            if factor != 1.0 or name != "µs":
                rep.rescale(factor, "µs", note)
            else:
                rep.unit_note = note
            return
    if time_us:
        detail = ", ".join(f"{r:.3g}× in {n}" for n, r, _ in checks)
        rep.mark_unknown(f"Time unit not identified: compared with the QHAS run, the trace span is {detail}. "
                         "Times are shown in the file's own units; busy shares are unaffected.")


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
