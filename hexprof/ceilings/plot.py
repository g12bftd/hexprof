"""Plots for the ceiling sweeps (matplotlib optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_PALETTE = ["#2a6fdb", "#e0663a", "#2e9e6a", "#8e5bd6", "#c4a000", "#d6336c"]


def plot_fits(series: dict[str, dict[str, Any]], kind: str, path: str | Path) -> Path | None:
    """One scatter + fit line per series.

    ``series[label] = {"x": [...], "y_ms": [...], "fit": {...}, "headline": "P = 9.8 TOPS"}``
    ``kind``: "compute" (x in ops -> GOP) or "bandwidth" (x in bytes -> MB).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (pip install 'hexprof[plots]')")
        return None
    scale, xlabel = (1e9, "Operations per inference (GOP)") if kind == "compute" else (2**20, "Weight bytes read per token (MB)")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for i, (label, s) in enumerate(series.items()):
        c = _PALETTE[i % len(_PALETTE)]
        x = np.asarray(s["x"], float)
        y = np.asarray(s["y_ms"], float)
        ax.scatter(x / scale, y, color=c, s=22, zorder=3)
        f = s.get("fit")
        if f:
            xs = np.linspace(0, x.max() * 1.05, 50)
            ax.plot(xs / scale, (f["intercept"] + f["slope"] * xs) * 1e3, color=c, lw=1.6,
                    label=f"{label}: {s.get('headline', '')}")
        else:
            ax.plot([], [], color=c, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Latency per inference (ms)" if kind == "compute" else "Latency per token (ms)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#dddddd", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Compute ceiling: latency = t0 + ops / P" if kind == "compute" else "Bandwidth ceiling: latency = t0 + bytes / B",
                 fontsize=10)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path
