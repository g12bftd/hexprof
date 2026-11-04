"""The ceiling experiments from the plan.

Compute ceiling
    Standalone MatMuls, once per weight/activation precision used by the
    deployment. Square shapes, multiples of 32, L = 1, 2, 4, 8... chained in one
    graph. Fit latency = t0 + ops / P per precision.

Bandwidth ceiling
    Width sweep (primary): single-token stacks of a fixed number of linear layers
    at the deployed weight precision, increasing their width.
    Depth sweep: Gemma-4-shaped decoder subsets of 4, 8, 12... layers (each layer
    adds bytes *and* fixed per-op overhead, so it underestimates B).
    Fit latency = t0 + bytes / B. Every configuration must exceed on-chip
    memory (VTCM + system cache) so weights stream from DRAM on every run.

KV-cache check
    The same compiled decoder subset, profiled with an empty and a full KV cache.
    If latency differs, cache bytes belong on the bandwidth x-axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from hexprof.aihub import client as hc
from hexprof.aihub.metrics import aihub_metrics
from hexprof.ceilings import fit, graphs
from hexprof.ceilings.plot import plot_fits
from hexprof.ceilings.runner import PointSpec, Sweep
from hexprof.io_utils import markdown_table, read_json, write_json

DEFAULT_ONCHIP_MB = 16.0  # 8 MB VTCM + system-level cache (conservative); weights must exceed this many times over


def _series(records: list[dict[str, Any]], x_key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        if r.get("latency_ms") is None:
            continue
        s = out.setdefault(r["series"], {"x": [], "y_ms": [], "keys": []})
        s["x"].append(r["graph"][x_key] if x_key in r["graph"] else r["graph"]["weight_bytes"] + r["graph"]["kv_bytes"])
        s["y_ms"].append(r["latency_ms"])
        s["keys"].append(r["key"])
    return out


def _qhas(rec: dict[str, Any]) -> dict[str, Any]:
    """QHAS headline numbers of a point's profile job, when its artifacts were downloaded (--hw)."""
    files = [f for f in ((rec.get("hw_artifacts") or {}).get("qhas") or []) if str(f).endswith(".json")]
    if not files or not Path(files[0]).exists():
        return {}
    try:
        from hexprof.hwreport import qhas as QH

        return QH.overall_kpis(QH.load_overall(files[0]))
    except Exception:  # noqa: BLE001 - optional cross-check
        return {}


def _points_md(records: list[dict[str, Any]], x_key: str, x_label: str, qhas_bytes: bool = False) -> str:
    """Per point: x, AI Hub latency and its run-to-run spread (min / median / max over the profile's runs)."""
    head = ["point", x_label, "latency ms", "min", "median", "max", "runs"]
    if qhas_bytes:
        head += ["QHAS DRAM bytes", "QHAS ÷ assumed"]
    rows = []
    for r in records:
        if r.get("latency_ms") is None:
            continue
        sp = (r.get("metrics") or {}).get("all_inference_times_milliseconds") or {}
        x = r["graph"].get(x_key)
        row = [r["key"], x, r["latency_ms"], sp.get("min"), sp.get("median"), sp.get("max"), sp.get("n")]
        if qhas_bytes:
            q = _qhas(r).get("total_dram")
            row += [q, (q / x) if q and x else None]
        rows.append(row)
    return markdown_table(head, rows, 4)


# --------------------------------------------------------------------------- compute


def compute_sweep(sweep: Sweep, precisions: Sequence[str], size: int, chain: Sequence[int], rows: int | None = None) -> dict[str, Any]:
    specs = []
    for p in precisions:
        prec = hc.parse_precision(p)
        for L in chain:
            specs.append(PointSpec(f"compute-{prec.name}-s{size}-m{rows or size}-l{L}", prec.name, prec,
                                   lambda d, L=L: graphs.matmul_chain(size, L, rows, out_dir=d)))
    records = sweep.run(specs)
    if sweep.dry_run:
        return {"records": records}
    series = _series(records, "ops")
    result: dict[str, Any] = {"experiment": "compute", "size": size, "rows": rows or size, "chain": list(chain),
                              "device": sweep.device_name, "runtime": sweep.runtime, "series": {},
                              "qairt": sorted({r["qairt"] for r in records if r.get("qairt")})}
    rows_md = []
    for label, s in series.items():
        entry: dict[str, Any] = {"points": [{"key": k, "ops": x, "latency_ms": y} for k, x, y in zip(s["keys"], s["x"], s["y_ms"])]}
        if len(set(s["x"])) >= 2:
            c = fit.compute_ceiling(s["x"], s["y_ms"])
            entry.update(c)
            s["fit"], s["headline"] = c["fit"], f"P={c['P_tops']:.2f} TOPS, t0={c['t0_ms']:.2f} ms"
            rows_md.append([label, c["P_tops"], c["t0_ms"], c["fit"]["r2"], len(s["x"]), max(c["attained_tops"])])
        result["series"][label] = entry
    write_json(sweep.out / "compute_ceiling.json", result)
    md = "## Compute ceiling (latency = t0 + ops / P)\n\n" + markdown_table(
        ["precision", "P (TOPS)", "t0 (ms)", "R²", "points", "best attained TOPS"], rows_md, 3)
    md += "\n\n### Points (AI Hub latency and its spread over the profile's runs)\n\n" + _points_md(records, "ops", "ops")
    (sweep.out / "compute_ceiling.md").write_text(md + "\n", encoding="utf-8")
    plot_fits(series, "compute", sweep.out / "compute_ceiling.png")
    print(md)
    return result


# --------------------------------------------------------------------------- bandwidth


def _check_offchip(records: list[dict[str, Any]], onchip_mb: float, factor: float = 4.0) -> list[str]:
    small = [r["key"] for r in records if r["graph"]["weight_bytes"] < factor * onchip_mb * 2**20]
    if small:
        print(f"WARNING: {len(small)} config(s) hold < {factor:g}x the {onchip_mb:g} MB of on-chip memory; "
              f"weights may stay on chip between runs and inflate B: {', '.join(small)}")
    return small


def bandwidth_sweep(
    sweep: Sweep,
    precision: str,
    widths: Sequence[int] = (),
    width_layers: int = 32,
    gemma4_layers: Sequence[int] = (),
    gemma4_dims: graphs.Gemma4Dims | None = None,
    context: int = 4096,
    include_kv: bool = False,
    onchip_mb: float = DEFAULT_ONCHIP_MB,
) -> dict[str, Any]:
    prec = hc.parse_precision(precision)
    specs = []
    for w in widths:
        specs.append(PointSpec(f"bw-width-{prec.name}-w{w}-l{width_layers}", f"width sweep ({width_layers} layers)", prec,
                               lambda d, w=w: graphs.linear_stack(w, width_layers, out_dir=d)))
    dims = gemma4_dims or (graphs.load_gemma4_dims() if gemma4_layers else None)
    for n in gemma4_layers:
        specs.append(PointSpec(f"bw-gemma4-{prec.name}-n{n}-c{context}", "Gemma 4 decoder subsets", prec,
                               lambda d, n=n: graphs.gemma4_decoder_stack(n, dims, context, out_dir=d)))  # type: ignore[arg-type]
    records = sweep.run(specs)
    small = _check_offchip(records, onchip_mb)
    if sweep.dry_run:
        return {"records": records}
    x_key = "weight_bytes_plus_kv" if include_kv else "weight_bytes"
    series = _series(records, x_key)
    result: dict[str, Any] = {"experiment": "bandwidth", "precision": prec.name, "device": sweep.device_name,
                              "runtime": sweep.runtime, "x_axis": x_key, "possibly_on_chip": small,
                              "qairt": sorted({r["qairt"] for r in records if r.get("qairt")}),
                              "gemma4_dims": dims.__dict__ if dims else None, "series": {}}
    rows_md = []
    for label, s in series.items():
        entry: dict[str, Any] = {"points": [{"key": k, "bytes": x, "latency_ms": y} for k, x, y in zip(s["keys"], s["x"], s["y_ms"])]}
        if len(set(s["x"])) >= 2:
            b = fit.bandwidth_ceiling(s["x"], s["y_ms"])
            entry.update(b)
            s["fit"], s["headline"] = b["fit"], f"B={b['B_gbps']:.1f} GB/s, t0={b['t0_ms']:.2f} ms"
            rows_md.append([label, b["B_gbps"], b["t0_ms"], b["fit"]["r2"], len(s["x"]), max(b["attained_gbps"])])
        result["series"][label] = entry
    # cross-check: refit on the DRAM bytes QHAS measured on the device instead of the assumed weight bytes
    measured = [(r["key"], _qhas(r).get("total_dram"), r["latency_ms"]) for r in records
                if r.get("latency_ms") is not None and r.get("series", "").startswith("width")]
    measured = [m for m in measured if m[1]]
    if len({m[1] for m in measured}) >= 2:
        b = fit.bandwidth_ceiling([m[1] for m in measured], [m[2] for m in measured])
        result["series"]["width sweep, QHAS-measured bytes"] = {
            "points": [{"key": k, "bytes": x, "latency_ms": y} for k, x, y in measured], **b}
        rows_md.append(["width sweep, QHAS-measured bytes", b["B_gbps"], b["t0_ms"], b["fit"]["r2"], len(measured),
                        max(b["attained_gbps"])])
    write_json(sweep.out / "bandwidth_ceiling.json", result)
    md = f"## Bandwidth ceiling at {prec.name} (latency = t0 + bytes / B)\n\n" + markdown_table(
        ["sweep", "B (GB/s)", "t0 (ms)", "R²", "points", "best attained GB/s"], rows_md, 3)
    md += ("\n\nThe width sweep is the primary estimate; the depth sweep adds per-layer overhead and underestimates B. "
           "With --hw, the width sweep is refit on the DRAM bytes QHAS measured on the device; the two B values "
           "should agree.")
    md += "\n\n### Points\n\n" + _points_md(records, x_key, "assumed bytes", qhas_bytes=True)
    (sweep.out / "bandwidth_ceiling.md").write_text(md + "\n", encoding="utf-8")
    plot_fits(series, "bandwidth", sweep.out / "bandwidth_ceiling.png")
    print(md)
    return result


# --------------------------------------------------------------------------- KV cache check


def kv_cache_check(
    out_dir: str | Path,
    device_name: str,
    precision: str,
    n_layers: int,
    context: int,
    dims: graphs.Gemma4Dims | None = None,
    runtime: str = "qnn_context_binary",
    qairt_version: str | None = None,
    tolerance: float = 0.03,
) -> dict[str, Any]:
    """Profile one compiled decoder subset with an empty and a full KV cache."""
    prec = hc.parse_precision(precision)
    dims = dims or graphs.load_gemma4_dims()
    # Float I/O so the hand-made inputs need no quantization parameters.
    sweep = Sweep(out_dir, device_name, runtime=runtime, qairt_version=qairt_version, quantize_io=False)
    key = f"kvcheck-{prec.name}-n{n_layers}-c{context}"
    spec = PointSpec(key, "kv-check", prec, lambda d: graphs.gemma4_decoder_stack(n_layers, dims, context, out_dir=d))
    sweep._build_and_upload([spec])
    sweep._quantize([spec])
    sweep._compile([spec])
    sweep._link([spec])
    st = sweep.state[key]
    if "error" in st:
        raise hc.HubJobError(st["error"])
    model = sweep.deployable(st)
    dev = hc.device(device_name)
    opts = f"--qairt_version {qairt_version}" if qairt_version else ""
    jobs = {}
    for kind in ("kv_empty", "kv_full"):
        if not st.get(f"{kind}_job"):
            job = hc.submit_inference(model, dev, sweep._inputs(st, 1, kind), profile=True,
                                      name=f"hexprof {key} {kind}", options=opts)
            st[f"{kind}_job"] = job.job_id
            sweep._save()
        jobs[kind] = hc.get_job(st[f"{kind}_job"])
    lat = {}
    for kind, job in jobs.items():
        prof, _ = hc.save_profile(job, sweep.out / "profiles", f"{key}-{kind}")
        lat[kind] = aihub_metrics(prof)["inference_time_milliseconds"]
    delta = (lat["kv_full"] - lat["kv_empty"]) / lat["kv_empty"]
    result = {
        "experiment": "kv_cache_check",
        "precision": prec.name,
        "layers": n_layers,
        "context": context,
        "latency_ms": lat,
        "relative_change": delta,
        "kv_bytes": st["graph"]["kv_bytes"],
        "weight_bytes": st["graph"]["weight_bytes"],
        "verdict": ("independent of KV contents: plot weight bytes only" if abs(delta) <= tolerance
                    else "depends on KV contents: add cache bytes to the x-axis (bandwidth --include-kv)"),
        "jobs": {k: j.url for k, j in jobs.items()},
    }
    write_json(sweep.out / f"{key}.json", result)
    print(f"KV cache check ({n_layers} layers, ctx {context}): empty {lat['kv_empty']:.3f} ms, "
          f"full {lat['kv_full']:.3f} ms ({delta:+.1%}) -> {result['verdict']}")
    return result


def load_ceilings(compute_json: str | Path | None, bandwidth_json: str | Path | None, series: str | None = None) -> dict[str, Any]:
    """Pick P and B out of earlier results (for operator classification)."""
    out: dict[str, Any] = {}
    if compute_json and Path(compute_json).exists():
        c = read_json(compute_json)
        out["P_tops"] = {k: v.get("P_tops") for k, v in c["series"].items()}
    if bandwidth_json and Path(bandwidth_json).exists():
        b = read_json(bandwidth_json)
        ser = b["series"]
        pick = series or next((k for k in ser if k.startswith("width")), next(iter(ser), None))
        out["B_gbps"] = ser.get(pick, {}).get("B_gbps") if pick else None
        out["B_series"] = pick
    return out
