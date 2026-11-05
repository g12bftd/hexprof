"""Work (MACs) and weight bytes per graph from the model architecture, when no ONNX is at hand.

* Gemma 4 E4B text parts: the per-part counts of :mod:`hexprof.models.gemma4_efficiency`
  (Qualcomm's checkpoint ``config.json`` and w4a16 recipe).
* SigLIP 2 image / text encoders: the standard ViT / text-transformer formulas with the
  ``google/siglip2-base-patch16-224`` config (read from the Hugging Face cache when present).

MACs count the useful multiply-accumulates of the weight layers and attention; elementwise
ops (norms, activations, residual adds) are not MACs and are left out. A static graph may
execute more (e.g. attention over the full context window); an ONNX count captures that.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_GEMMA = re.compile(r"(prompt|token)_ar(\d+)_cl(\d+)_(\d+)_of_(\d+)")
SIGLIP2_REPO = "google/siglip2-base-patch16-224"
SIGLIP2_BASE = {  # google/siglip2-base-patch16-224
    "vision": {"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12, "image_size": 224,
               "patch_size": 16, "num_channels": 3},
    "text": {"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12, "max_position_embeddings": 64},
}
BITS = {"w4a16": 4, "w4a8": 4, "w8a16": 8, "w8a8": 8, "w16a16": 16, "fp16": 16, "float": 16}


def precision_bits(precision: str | None) -> int:
    return BITS.get((precision or "").lower(), 8)


def gemma4_text(label: str, config_path: str | Path | None = None) -> dict[str, Any] | None:
    m = _GEMMA.search(label)
    if not m:
        return None
    from hexprof.models import gemma4_efficiency as ge

    try:
        c = ge.TextConfig.load(ge.find_config(config_path))
    except (FileNotFoundError, KeyError, ValueError):
        return None
    seq, ctx, part, nparts = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    cnt = ge.part_counts(c, part, nparts, seq, ctx)
    return {"macs": cnt["ops"] / 2, "weight_bytes": cnt["weight_bytes"], "kv_bytes": cnt["kv_bytes"],
            "weight_params": cnt["params"], "layers": cnt["layers"], "tokens": seq, "context": ctx,
            "precision": f"w{c.uniform_bits}a16" if c.uniform_bits and c.uniform_bits < 16 else
                         ("fp16" if c.uniform_bits else "w4a16"),
            "source": ("analytic: dummy config, uniform AI Hub quantization" if c.uniform_bits else
                       "analytic: Gemma 4 config + Qualcomm's w4a16 recipe") +
                      " (useful MACs; attention over the sliding window / filled context)"}


def _siglip_config(config_path: str | Path | None = None) -> dict[str, dict[str, int]]:
    cfg = {k: dict(v) for k, v in SIGLIP2_BASE.items()}
    if config_path and Path(config_path).is_file():  # e.g. a hexprof dummy checkpoint
        try:
            raw = json.loads(Path(config_path).read_text())
        except (OSError, ValueError):
            raw = {}
        if "siglip" in str(raw.get("model_type", "")):
            for side in ("vision", "text"):
                for k, v in (raw.get(f"{side}_config") or {}).items():
                    if k in cfg[side] and isinstance(v, int):
                        cfg[side][k] = v
            return cfg
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(SIGLIP2_REPO, "config.json")
        if isinstance(p, str):
            raw = json.loads(Path(p).read_text())
            for side in ("vision", "text"):
                for k, v in (raw.get(f"{side}_config") or {}).items():
                    if k in cfg[side] and isinstance(v, int):
                        cfg[side][k] = v
    except Exception:  # noqa: BLE001 - offline / no hub: built-in base config
        pass
    return cfg


def _block(tokens: int, h: int, inter: int) -> tuple[float, float]:
    """MACs and weight parameters of one pre-norm transformer block (q, k, v, o + 2-matrix MLP)."""
    params = 4 * h * h + 2 * h * inter
    return tokens * params + 2 * tokens * tokens * h, params


def siglip2(component: str, precision: str | None = "w8a16", config_path: str | Path | None = None) -> dict[str, Any] | None:
    if component not in ("image_encoder", "text_encoder"):
        return None
    cfg = _siglip_config(config_path)
    bits = precision_bits(precision)
    if component == "image_encoder":
        v = cfg["vision"]
        h, inter, n = v["hidden_size"], v["intermediate_size"], v["num_hidden_layers"]
        t = (v["image_size"] // v["patch_size"]) ** 2
        patch_in = v["num_channels"] * v["patch_size"] ** 2
        blk_macs, blk_params = _block(t, h, inter)
        # attention-pooling head: 1 probe query over t tokens, then an MLP
        head_macs = h * h + 2 * t * h * h + 2 * t * h + h * h + 2 * h * inter
        head_params = 4 * h * h + 2 * h * inter
        macs = t * patch_in * h + n * blk_macs + head_macs
        params = patch_in * h + t * h + n * blk_params + head_params
        shape = f"{t} patches, {n} layers, hidden {h}"
    else:
        x = cfg["text"]
        h, inter, n, s = x["hidden_size"], x["intermediate_size"], x["num_hidden_layers"], x["max_position_embeddings"]
        blk_macs, blk_params = _block(s, h, inter)
        macs = n * blk_macs + h * h  # head projection on the pooled token
        params = s * h + n * blk_params + h * h  # token-embedding rows are gathered, not streamed
        shape = f"{s} tokens, {n} layers, hidden {h}"
    return {"macs": float(macs), "weight_params": float(params), "weight_bytes": params * bits / 8,
            "kv_bytes": 0.0, "precision": precision or "w8a16",
            "source": f"analytic: {SIGLIP2_REPO} ({shape}); weight layers and attention"}
