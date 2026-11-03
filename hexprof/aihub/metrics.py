"""Turn an AI Hub profile into the same metrics AI Hub / AI Hub Models report.

The conversions mirror ``qai_hub_models.scorecard.results.scorecard_job`` and
``qai_hub_models.utils.printing.print_profile_metrics_from_job`` (v0.65):

* ``inference_time_milliseconds`` = ``execution_summary.estimated_inference_time`` / 1000
  (AI Hub's estimate over its repeated on-device runs, in microseconds).
* ``estimated_peak_memory_range_mb`` = ``inference_memory_peak_range`` bytes / 2**20, rounded.
* ``layer_counts`` and ``primary_compute_unit`` from ``execution_detail[*].compute_unit``.

Anything AI Hub does not measure (power, energy, temperature, clocks, stall
counters) is deliberately absent rather than estimated.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from hexprof.io_utils import markdown_table, summarize

_MB = 1 << 20


def layer_counts(profile: dict) -> dict[str, int]:
    counts = Counter(str(d.get("compute_unit", "UNK")).upper() for d in profile.get("execution_detail", []))
    npu, gpu, cpu = counts.get("NPU", 0), counts.get("GPU", 0), counts.get("CPU", 0)
    return {"total": npu + gpu + cpu, "npu": npu, "gpu": gpu, "cpu": cpu}


def primary_compute_unit(counts: dict[str, int]) -> str:
    npu, gpu, cpu = counts["npu"], counts["gpu"], counts["cpu"]
    if npu == 0 and gpu == 0 and cpu == 0:
        return "null"
    top = max(cpu, gpu, npu)
    if top == npu:
        return "NPU"
    if top == gpu:
        return "GPU"
    return "CPU"


def _us_to_ms(v: Any) -> float | None:
    return None if v is None else float(v) / 1000.0


def _range_mb(rng: Any) -> dict[str, int] | None:
    if not rng:
        return None
    lo, hi = rng
    return {"min": round(lo / _MB), "max": round(hi / _MB)}


def aihub_metrics(profile: dict) -> dict[str, Any]:
    """Metrics in AI Hub Models' perf.yaml vocabulary (plus load times and run spread)."""
    es = profile["execution_summary"]
    counts = layer_counts(profile)
    out: dict[str, Any] = {
        "inference_time_milliseconds": _us_to_ms(es.get("estimated_inference_time")),
        "estimated_peak_memory_range_mb": _range_mb(es.get("inference_memory_peak_range")),
        "primary_compute_unit": primary_compute_unit(counts),
        "layer_counts": counts,
        "first_load_time_milliseconds": _us_to_ms(es.get("first_load_time")),
        "warm_load_time_milliseconds": _us_to_ms(es.get("warm_load_time")),
        "first_load_peak_memory_range_mb": _range_mb(es.get("first_load_memory_peak_range")),
        "warm_load_peak_memory_range_mb": _range_mb(es.get("warm_load_memory_peak_range")),
    }
    all_times = es.get("all_inference_times")
    if all_times:
        out["all_inference_times_milliseconds"] = summarize(t / 1000.0 for t in all_times)
    return out


def per_layer(profile: dict) -> list[dict[str, Any]]:
    """Per-op rows: name, type, compute unit, time (us) and cycles when present."""
    rows = []
    for i, d in enumerate(profile.get("execution_detail", [])):
        rows.append(
            {
                "index": i,
                "name": d.get("name"),
                "type": d.get("type"),
                "compute_unit": d.get("compute_unit"),
                "execution_time_us": d.get("execution_time"),
                "execution_cycles": d.get("execution_cycles"),
            }
        )
    return rows


def op_type_breakdown(profile: dict) -> list[dict[str, Any]]:
    """Total per-op time grouped by op type, sorted by time (needs per-layer timing)."""
    agg: dict[tuple[str, str], dict[str, float]] = {}
    for row in per_layer(profile):
        key = (str(row["type"]), str(row["compute_unit"]))
        a = agg.setdefault(key, {"count": 0, "time_us": 0.0, "cycles": 0.0})
        a["count"] += 1
        a["time_us"] += float(row["execution_time_us"] or 0)
        a["cycles"] += float(row["execution_cycles"] or 0)
    total = sum(a["time_us"] for a in agg.values()) or 1.0
    out = [
        {"type": t, "compute_unit": cu, **a, "share": a["time_us"] / total}
        for (t, cu), a in agg.items()
    ]
    return sorted(out, key=lambda r: -r["time_us"])


def metrics_markdown(title: str, m: dict[str, Any]) -> str:
    peak = m.get("estimated_peak_memory_range_mb") or {}
    lc = m["layer_counts"]
    rows = [
        ["Estimated inference time (ms)", m.get("inference_time_milliseconds")],
        ["Estimated peak memory (MB)", f"[{peak.get('min', '-')}, {peak.get('max', '-')}]"],
        ["Primary compute unit", m["primary_compute_unit"]],
        ["Layers (NPU / GPU / CPU)", f"{lc['npu']} / {lc['gpu']} / {lc['cpu']} (total {lc['total']})"],
        ["First load (ms)", m.get("first_load_time_milliseconds")],
        ["Warm load (ms)", m.get("warm_load_time_milliseconds")],
    ]
    spread = m.get("all_inference_times_milliseconds")
    if spread and spread.get("n"):
        rows.append(
            [
                f"All inference runs (ms, n={spread['n']})",
                f"min {spread['min']:.3f} / median {spread['median']:.3f} / max {spread['max']:.3f}",
            ]
        )
    return f"### {title}\n\n" + markdown_table(["Metric", "Value"], rows, digits=3)
