"""Straight-line fits for the ceilings.

Compute:   latency = t0 + ops / P    -> P = 1 / slope (ops/s), t0 = intercept
Bandwidth: latency = t0 + bytes / B  -> B = 1 / slope (bytes/s), t0 = intercept

Latency per point is AI Hub's estimated inference time (its estimate over the
repeated on-device runs).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class LinearFit:
    slope: float
    intercept: float
    slope_se: float
    intercept_se: float
    r2: float
    n: int

    def predict(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.intercept + self.slope * np.asarray(x)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_line(x: Sequence[float], y: Sequence[float]) -> LinearFit:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    n = len(xa)
    if n < 2 or np.ptp(xa) == 0:
        raise ValueError("need at least two points with distinct x to fit a line")
    A = np.vstack([xa, np.ones(n)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, ya, rcond=None)
    resid = ya - (slope * xa + intercept)
    ss_res = float(resid @ resid)
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    if n > 2:
        s2 = ss_res / (n - 2)
        sxx = float(((xa - xa.mean()) ** 2).sum())
        slope_se = math.sqrt(s2 / sxx)
        intercept_se = math.sqrt(s2 * (1.0 / n + xa.mean() ** 2 / sxx))
    else:
        slope_se = intercept_se = float("nan")
    return LinearFit(float(slope), float(intercept), slope_se, intercept_se, r2, n)


def _inverse_with_range(slope: float, se: float) -> tuple[float, float | None, float | None]:
    val = 1.0 / slope if slope > 0 else float("inf")
    if not math.isfinite(se) or slope <= 0:
        return val, None, None
    lo_s, hi_s = slope + 2 * se, slope - 2 * se
    return val, 1.0 / lo_s, (1.0 / hi_s if hi_s > 0 else None)


def compute_ceiling(ops: Sequence[float], latency_ms: Sequence[float]) -> dict[str, Any]:
    """P (TOPS) and fixed cost t0 (ms) from (ops per inference, latency) points."""
    f = fit_line(ops, [v / 1e3 for v in latency_ms])  # seconds
    p, p_lo, p_hi = _inverse_with_range(f.slope, f.slope_se)
    return {
        "P_tops": p / 1e12,
        "P_tops_2se_range": [None if p_lo is None else p_lo / 1e12, None if p_hi is None else p_hi / 1e12],
        "t0_ms": f.intercept * 1e3,
        "t0_ms_se": f.intercept_se * 1e3,
        "fit": f.to_dict(),
        "attained_tops": [o / (t / 1e3) / 1e12 for o, t in zip(ops, latency_ms)],
    }


def bandwidth_ceiling(nbytes: Sequence[float], latency_ms: Sequence[float]) -> dict[str, Any]:
    """B (GB/s) and fixed cost per step t0 (ms) from (bytes per token, latency) points."""
    f = fit_line(nbytes, [v / 1e3 for v in latency_ms])
    b, b_lo, b_hi = _inverse_with_range(f.slope, f.slope_se)
    return {
        "B_gbps": b / 1e9,
        "B_gbps_2se_range": [None if b_lo is None else b_lo / 1e9, None if b_hi is None else b_hi / 1e9],
        "t0_ms": f.intercept * 1e3,
        "t0_ms_se": f.intercept_se * 1e3,
        "fit": f.to_dict(),
        "attained_gbps": [x / (t / 1e3) / 1e9 for x, t in zip(nbytes, latency_ms)],
    }
