"""How well each compiled Gemma 4 part uses the S25's NPU: achieved TOPS and GB/s.

AI Hub / QHAS report *time-based* numbers per graph: run time, inferences per
second and "% of overall timeline HTP utilization" (how much of the run the NPU
was busy, not how close it got to its compute or bandwidth peak). This module
adds the *work* side, per part:

* operations per run  = 2 x weight parameters x tokens + attention
* bytes per run       = weights at their deployed precision (+ per-channel scales)
                        + the KV cache read by attention
* achieved TOPS = operations / time,  achieved GB/s = bytes / time,
  arithmetic intensity = operations / byte

Counts come from the model config (Qualcomm's checkpoint ``config.json``) and
the structure of Qualcomm's Gemma 4 export (``qai_hub_models`` 0.65): 4 parts
split at layers [0:24] [24:33] [33:42] + lm_head; sliding-window and global
attention layers with their own head sizes; KV-shared layers without K/V
projections and with a double-wide MLP; per-layer-embedding gate/projection.
Deployed precision follows Qualcomm's w4a16 recipe: int4 transformer weights,
int8 ``down_proj``, int4 ``lm_head`` (QHAS measures 344.5 MB of DRAM traffic
for the lm_head token graph: 262144 x 2560 weights at 4 bits = 335.5 MB, at 8
bits it would be 671 MB), int8 KV cache. Embedding lookups run on the host and
are excluded. Byte counts are analytic estimates unless the graph's QHAS file
is present: then the DRAM bytes it measured are used (``bytes_source``).
``--measure-binaries`` downloads the compiled context binaries as a further check.

With the measured ceilings (``hexprof ceilings compute/bandwidth``) it also
reports compute / bandwidth utilisation and the roofline bound per part.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hexprof.io_utils import markdown_table, read_json, write_json

SPLITTING_POINTS = (24, 33)  # qai_hub_models gemma_4_e4b_it: layers[0:24] | [24:33] | [33:42] | lm_head
SCALE_BYTES_PER_CHANNEL = 4  # fp32 per-output-channel scale
LM_HEAD_BITS = 4  # measured: the lm_head token graph moves 344.5 MB per run (4-bit = 335.5 MB, 8-bit = 671 MB)
_GRAPH_RE = re.compile(r"(prompt|token)_ar(\d+)_cl(\d+)_(\d+)_of_(\d+)")
_PART_RE = re.compile(r"part(\d+)_of_(\d+)")


# --------------------------------------------------------------------------- config


def find_config(explicit: str | Path | None = None) -> Path:
    """Qualcomm's checkpoint config.json (downloaded by the export) or an explicit file."""
    if explicit:
        return Path(explicit)
    root = os.environ.get("QAIHM_STORE_ROOT") or os.environ.get("HEXPROF_DOWNLOAD_DIR") or str(Path.home())
    hits = sorted(glob.glob(os.path.join(root, ".qaihm", "qai-hub-models", "models", "gemma_4_e4b_it", "v*",
                                         "gemma_4_e4b_it_w4a16", "config.json")))
    if not hits:
        raise FileNotFoundError(
            f"no Gemma 4 config.json under {root}/.qaihm (run the export first, or pass --config)")
    return Path(hits[-1])


@dataclass
class TextConfig:
    hidden: int
    intermediate: list[int]
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    global_head_dim: int
    layer_types: list[str]
    sliding_window: int
    kv_shared_layers: int
    double_wide_mlp: bool
    ple_dim: int
    vocab: int

    @classmethod
    def load(cls, path: str | Path) -> "TextConfig":
        raw = json.loads(Path(path).read_text())
        t = raw.get("text_config", raw)
        n = int(t["num_hidden_layers"])
        inter = t["intermediate_size"]
        inter = [int(x) for x in inter] if isinstance(inter, list) else [int(inter)] * n
        types = list(t.get("layer_types") or ["sliding_attention"] * n)
        return cls(
            hidden=int(t["hidden_size"]),
            intermediate=inter,
            layers=n,
            heads=int(t["num_attention_heads"]),
            kv_heads=int(t["num_key_value_heads"]),
            head_dim=int(t.get("head_dim") or 256),
            global_head_dim=int(t.get("global_head_dim") or t.get("head_dim") or 256),
            layer_types=types,
            sliding_window=int(t.get("sliding_window") or 512),
            kv_shared_layers=int(t.get("num_kv_shared_layers") or 0),
            double_wide_mlp=bool(t.get("use_double_wide_mlp", False)),
            ple_dim=int(t.get("hidden_size_per_layer_input") or 0),
            vocab=int(t.get("vocab_size") or 262144),
        )

    def is_global(self, i: int) -> bool:
        return self.layer_types[i] != "sliding_attention"

    def is_kv_shared(self, i: int) -> bool:
        first = self.layers - self.kv_shared_layers
        return 0 < first <= i


# --------------------------------------------------------------------------- analytic counts


def _linear(d_in: int, d_out: int, bits: int) -> tuple[int, float]:
    params = d_in * d_out
    return params, params * bits / 8 + d_out * SCALE_BYTES_PER_CHANNEL


def layer_counts(c: TextConfig, i: int, seq: int, ctx: int, kv_bytes: float = 1.0) -> dict[str, float]:
    """Weight params / bytes, operations and KV bytes of decoder layer ``i`` for one run of ``seq`` tokens."""
    hd = c.global_head_dim if c.is_global(i) else c.head_dim
    span = ctx if c.is_global(i) else min(c.sliding_window, ctx)
    inter = c.intermediate[i] * (2 if c.double_wide_mlp and c.is_kv_shared(i) else 1)
    mats = [(c.hidden, c.heads * hd, 4), (c.heads * hd, c.hidden, 4)]  # q, o
    if not c.is_kv_shared(i):
        mats += [(c.hidden, c.kv_heads * hd, 4)] * 2  # k, v
    mats += [(c.hidden, inter, 4), (c.hidden, inter, 4), (inter, c.hidden, 8)]  # gate, up (int4), down (int8)
    if c.ple_dim:
        mats += [(c.hidden, c.ple_dim, 4), (c.ple_dim, c.hidden, 4)]  # per-layer-embedding gate / projection
    params = 0
    wbytes = 0.0
    for d_in, d_out, bits in mats:
        p, b = _linear(d_in, d_out, bits)
        params += p
        wbytes += b
    attn_ops = 2 * 2 * c.heads * seq * span * hd  # QK^T and AV
    kv = 2 * c.kv_heads * span * hd * kv_bytes  # K and V read
    return {"params": params, "weight_bytes": wbytes, "ops": 2 * params * seq + attn_ops, "kv_bytes": kv}


def splits_for(c: TextConfig, num_parts: int, splits: tuple[int, ...] | None = None) -> tuple[int, ...]:
    """Layer split points for ``num_parts`` (last part = lm_head); Qualcomm's [24, 33] for Gemma 4 E4B."""
    splits = tuple(splits) if splits is not None else SPLITTING_POINTS
    n_text = num_parts - 1
    if len(splits) == n_text - 1 and all(0 < s < c.layers for s in splits):
        return splits
    return tuple(round(c.layers * k / n_text) for k in range(1, n_text))


def part_counts(c: TextConfig, part: int, num_parts: int, seq: int, ctx: int,
                splits: tuple[int, ...] | None = None) -> dict[str, float]:
    bounds = [0, *splits_for(c, num_parts, splits), c.layers]
    if part == num_parts and len(bounds) - 1 < num_parts:  # last part = lm_head
        p, b = _linear(c.hidden, c.vocab, LM_HEAD_BITS)
        return {"layers": "lm_head", "params": p, "weight_bytes": b, "ops": 2 * p * seq, "kv_bytes": 0.0}
    lo, hi = bounds[part - 1], bounds[part]
    tot = {"params": 0.0, "weight_bytes": 0.0, "ops": 0.0, "kv_bytes": 0.0}
    for i in range(lo, hi):
        for k, v in layer_counts(c, i, seq, ctx).items():
            tot[k] += v
    tot["layers"] = f"{lo}-{hi - 1}"
    return tot


# --------------------------------------------------------------------------- QHAS summary fields


_SCALAR_KEY = re.compile(r"util|inference|per_?sec|throughput|ips\b|fps", re.I)


def qhas_summary_fields(path: str | Path, max_mb: float = 1024, window_mb: float = 64) -> dict[str, Any]:
    """Graph-level utilisation / inferences-per-second fields of a QHAS summary.

    Small files are parsed fully and per-op tables (long lists) are skipped; for
    large files the first/last ``window_mb`` are scanned for ``"key": number``
    pairs and only keys that occur once (graph-level) are kept.
    """
    p = Path(path)
    out: dict[str, Any] = {}
    if p.stat().st_size <= max_mb * 2**20:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))

        def walk(o: Any, path: str) -> None:
            if isinstance(o, dict):
                for k, v in o.items():
                    q = f"{path}.{k}"
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and _SCALAR_KEY.search(k):
                        out[q] = v
                    elif isinstance(v, (dict, list)):
                        walk(v, q)
            elif isinstance(o, list) and len(o) <= 8:  # long lists are per-op tables
                for idx, v in enumerate(o):
                    walk(v, f"{path}[{idx}]")

        walk(data, "$")
        return out
    size = p.stat().st_size
    w = int(window_mb * 2**20)
    with open(p, "rb") as f:
        chunks = [f.read(w)]
        if size > w:
            f.seek(max(size - w, w))
            chunks.append(f.read(w))
    counts: dict[str, int] = {}
    first: dict[str, float] = {}
    for ch in chunks:
        for m in re.finditer(rb'"([^"]{1,80})"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', ch):
            k = m.group(1).decode(errors="replace")
            if _SCALAR_KEY.search(k):
                counts[k] = counts.get(k, 0) + 1
                first.setdefault(k, float(m.group(2)))
    return {f"~{k}": first[k] for k, n in counts.items() if n == 1}


# --------------------------------------------------------------------------- report


def _graph_info(rec: dict[str, Any]) -> tuple[str, int, int, int, int] | None:
    m = _GRAPH_RE.search(rec.get("graph", ""))
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))


def _ceilings(compute_json: str | None, bandwidth_json: str | None, peak_tops: float | None,
              peak_gbps: float | None, precision: str = "w4a16") -> tuple[float | None, float | None, str]:
    P, B, src = peak_tops, peak_gbps, "given"
    if compute_json and P is None:
        series = read_json(compute_json).get("series", {})
        P = (series.get(precision) or next(iter(series.values()), {})).get("P_tops")
        src = "measured"
    if bandwidth_json and B is None:
        series = read_json(bandwidth_json).get("series", {})
        pick = next((v for k, v in series.items() if k.startswith("width")), next(iter(series.values()), {}))
        B = pick.get("B_gbps")
        src = "measured"
    return P, B, src


def efficiency(
    export_dir: str | Path,
    config: str | Path | None = None,
    compute_json: str | None = None,
    bandwidth_json: str | None = None,
    peak_tops: float | None = None,
    peak_gbps: float | None = None,
    measure_binaries: bool = False,
    qhas_max_mb: float = 1024,
    splits: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    export_dir = Path(export_dir)
    summary = read_json(export_dir / "export_profiles.json")
    cfg_path = find_config(config)
    c = TextConfig.load(cfg_path)
    P, B, ceil_src = _ceilings(compute_json, bandwidth_json, peak_tops, peak_gbps)
    binaries: dict[str, float] = {}
    if measure_binaries:
        binaries = _binary_sizes(summary, export_dir / "binaries")

    rows: list[dict[str, Any]] = []
    for label, rec in sorted(summary["parts"].items()):
        t_ms = float(rec["metrics"]["inference_time_milliseconds"])
        row: dict[str, Any] = {"graph": label, "time_ms": t_ms, "inferences_per_s": 1e3 / t_ms if t_ms else None}
        info = _graph_info(rec)
        if info:
            kind, seq, ctx, part, nparts = info
            cnt = part_counts(c, part, nparts, seq, ctx, splits)
            row.update(kind="prefill" if kind == "prompt" else "decode", tokens=seq, context=ctx, layers=cnt["layers"],
                       weight_params=cnt["params"], weight_bytes=cnt["weight_bytes"], kv_bytes=cnt["kv_bytes"],
                       ops=cnt["ops"])
            est_bytes = cnt["weight_bytes"] + cnt["kv_bytes"]
            measured = _measured_dram(export_dir, label)
            total_bytes = measured or est_bytes
            row["bytes_estimate"] = est_bytes
            row["bytes"] = total_bytes
            row["bytes_source"] = "QHAS" if measured else "estimate"
            row["achieved_tops"] = cnt["ops"] / (t_ms / 1e3) / 1e12
            row["achieved_gbps"] = total_bytes / (t_ms / 1e3) / 1e9
            row["intensity_ops_per_byte"] = cnt["ops"] / total_bytes
            if P:
                row["compute_util"] = row["achieved_tops"] / P
            if B:
                row["bandwidth_util"] = row["achieved_gbps"] / B
            if P and B:
                t_compute = cnt["ops"] / (P * 1e12) * 1e3
                t_memory = total_bytes / (B * 1e9) * 1e3
                row["roofline_ms"] = max(t_compute, t_memory)
                row["roofline_bound"] = "memory" if t_memory >= t_compute else "compute"
                row["roofline_efficiency"] = row["roofline_ms"] / t_ms
        else:
            row["kind"] = "vision encoder" if "vision" in label else "other"
        comp = rec.get("component", label.split("/")[0])
        if comp in binaries:
            row["binary_bytes"] = binaries[comp]
            row["binary_gbps"] = binaries[comp] / (t_ms / 1e3) / 1e9
        qhas_files = sorted(glob.glob(str(export_dir / "hw" / label.replace("/", "__") / "qhas" / "*.json")))
        if qhas_files:
            try:
                row["qhas"] = qhas_summary_fields(qhas_files[0], qhas_max_mb)
            except Exception as e:  # noqa: BLE001 - best effort
                row["qhas_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)

    decode = [r for r in rows if r.get("kind") == "decode"]
    prefill = [r for r in rows if r.get("kind") == "prefill"]
    model = {}
    if decode:
        t = sum(r["time_ms"] for r in decode)
        model["decode_ms_per_token"] = t
        model["decode_tokens_per_s"] = 1e3 / t
        model["decode_achieved_gbps"] = sum(r["bytes"] for r in decode) / (t / 1e3) / 1e9
    if prefill:
        t = sum(r["time_ms"] for r in prefill)
        seq = prefill[0]["tokens"]
        model["prefill_ms_per_chunk"] = t
        model["prefill_tokens_per_s"] = seq * 1e3 / t
        model["prefill_achieved_tops"] = sum(r["ops"] for r in prefill) / (t / 1e3) / 1e12
    report = {"config": str(cfg_path), "ceilings": {"P_tops": P, "B_gbps": B, "source": ceil_src if (P or B) else None},
              "model": model, "graphs": rows}
    write_json(export_dir / "efficiency.json", report)
    md = efficiency_markdown(report)
    (export_dir / "efficiency.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved {export_dir}/efficiency.md and efficiency.json")
    return report


def _measured_dram(export_dir: Path, label: str) -> float | None:
    """DRAM bytes per run that QHAS measured for this graph (hw report summary first, else the QHAS file)."""
    stem = label.replace("/", "__")
    summ = export_dir / "hw_report" / "summary.json"
    if summ.exists():
        k = (read_json(summ).get(stem) or {}).get("qhas", {}).get("kpis") or {}
        if k.get("total_dram"):
            return float(k["total_dram"])
    files = sorted(glob.glob(str(export_dir / "hw" / stem / "qhas" / "*.json")))
    if not files:
        return None
    try:
        from hexprof.hwreport import qhas as QH

        k = QH.overall_kpis(QH.load_overall(files[0]))
        return float(k["total_dram"]) if k.get("total_dram") else None
    except Exception:  # noqa: BLE001 - not a QHAS 1.x file: fall back to the estimate
        return None


def _binary_sizes(summary: dict[str, Any], out: Path) -> dict[str, float]:
    """Download each component's compiled context binary once and return its size."""
    from hexprof.aihub import client as hc

    out.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, float] = {}
    for rec in summary["parts"].values():
        comp = rec.get("component")
        if not comp or comp in sizes or not rec.get("model_id"):
            continue
        path = out / f"{comp}.bin"
        if not path.exists():
            print(f"Downloading compiled {comp} ...")
            path = Path(hc.get_model(rec["model_id"]).download(str(path)))
        sizes[comp] = float(path.stat().st_size)
    return sizes


def _gb(x: float | None) -> str:
    return "-" if x is None else f"{x / 1e9:.2f}"


def efficiency_markdown(r: dict[str, Any]) -> str:
    P, B = r["ceilings"]["P_tops"], r["ceilings"]["B_gbps"]
    head = ["graph", "ms", "inf/s", "GOP", "GB read", "ops/byte", "TOPS", "GB/s"]
    if P:
        head.append("% of P")
    if B:
        head.append("% of B")
    if P and B:
        head += ["bound", "roofline eff."]
    rows = []
    for g in r["graphs"]:
        row = [g["graph"], g["time_ms"], g["inferences_per_s"]]
        if "ops" in g:
            row += [g["ops"] / 1e9, _gb(g["bytes"]) + ("" if g.get("bytes_source") == "QHAS" else " (est.)"),
                    g["intensity_ops_per_byte"], g["achieved_tops"], g["achieved_gbps"]]
        else:
            row += ["-", _gb(g.get("binary_bytes")), "-", "-", g.get("binary_gbps", "-")]
        if P:
            row.append(f"{100 * g['compute_util']:.1f}%" if "compute_util" in g else "-")
        if B:
            row.append(f"{100 * g['bandwidth_util']:.1f}%" if "bandwidth_util" in g else "-")
        if P and B:
            row += [g.get("roofline_bound", "-"),
                    f"{100 * g['roofline_efficiency']:.0f}%" if "roofline_efficiency" in g else "-"]
        rows.append(row)
    m = r["model"]
    lines = ["## Gemma 4 E4B parts on the S25: work, bytes and achieved rates", "",
             markdown_table(head, rows, 2), ""]
    if m:
        lines.append(f"Whole model: decode {m.get('decode_tokens_per_s', 0):.2f} tok/s "
                     f"({m.get('decode_ms_per_token', 0):.1f} ms/token, {m.get('decode_achieved_gbps', 0):.1f} GB/s), "
                     f"prefill {m.get('prefill_tokens_per_s', 0):.0f} tok/s ({m.get('prefill_achieved_tops', 0):.2f} TOPS).")
    if not (P or B):
        lines.append("No ceilings given: add --compute-json / --bandwidth-json (from `hexprof ceilings`) "
                     "or --peak-tops / --peak-gbps for % of peak and the roofline bound.")
    qrows = [[g["graph"], k, v] for g in r["graphs"] for k, v in (g.get("qhas") or {}).items()]
    if qrows:
        lines += ["", "### QHAS graph-level fields (time-based: NPU busy share, runs per second)", "",
                  markdown_table(["graph", "field", "value"], qrows, 3)]
    if any("binary_bytes" in g for g in r["graphs"]):
        lines += ["", "Compiled binary sizes (both graphs of a part share one binary and its weights): "
                  + ", ".join(sorted({f"{g['graph'].split('/')[0]} {_gb(g['binary_bytes'])} GB"
                                      for g in r["graphs"] if 'binary_bytes' in g}))]
    lines += ["", "GOP is an analytic estimate from the model config and Qualcomm's w4a16 recipe (int4 weights and "
              "lm_head, int8 down_proj and KV cache; host-side embedding lookups excluded). GB read is what QHAS "
              "measured for the graph; (est.) marks the analytic estimate, used when no QHAS file is present."]
    return "\n".join(lines)
