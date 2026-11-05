"""What a shrunk model is: the knobs, the Hugging Face config they produce, and what the device must show.

Pure Python (no torch). Three jobs:

* :func:`resolve` turns ``{"layers": 12, "hidden": 1024}`` into a full Hugging Face config, starting
  from the real model's config, so every field that is not overridden keeps its real value;
* :func:`validate` rejects requests Qualcomm's code cannot build (and warns about sizes that build
  but run with padding);
* :func:`expected_io` computes, independently of the code that builds the model, the input and
  output shapes the deployed graphs must have. The verification compares them with what AI Hub
  reports for the binaries that ran on the phone.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

TARGETS = ("gemma4-text", "gemma4-vision", "siglip2-image", "siglip2-text")
FAMILY = {"gemma4-text": "gemma4", "gemma4-vision": "gemma4", "siglip2-image": "siglip2", "siglip2-text": "siglip2"}

# knob -> (config section, key). Sections: "text_config" / "vision_config" (or "" for the top level).
KNOBS: dict[str, dict[str, tuple[str, str]]] = {
    "gemma4-text": {
        "layers": ("text_config", "num_hidden_layers"),
        "kv-shared-layers": ("text_config", "num_kv_shared_layers"),
        "hidden": ("text_config", "hidden_size"),
        "intermediate": ("text_config", "intermediate_size"),
        "heads": ("text_config", "num_attention_heads"),
        "kv-heads": ("text_config", "num_key_value_heads"),
        "head-dim": ("text_config", "head_dim"),
        "global-head-dim": ("text_config", "global_head_dim"),
        "sliding-window": ("text_config", "sliding_window"),
        "pattern": ("text_config", "layer_types"),  # every N-th layer is full (global) attention
        "ple-dim": ("text_config", "hidden_size_per_layer_input"),
        "vocab": ("text_config", "vocab_size"),
    },
    "gemma4-vision": {
        "layers": ("vision_config", "num_hidden_layers"),
        "hidden": ("vision_config", "hidden_size"),
        "intermediate": ("vision_config", "intermediate_size"),
        "heads": ("vision_config", "num_attention_heads"),
        "kv-heads": ("vision_config", "num_key_value_heads"),
        "head-dim": ("vision_config", "head_dim"),
        "patch-size": ("vision_config", "patch_size"),
        "pooling-kernel": ("vision_config", "pooling_kernel_size"),
        "soft-tokens": ("vision_config", "default_output_length"),
        "text-hidden": ("text_config", "hidden_size"),  # width of the image tokens handed to the LM
    },
    "siglip2-image": {
        "layers": ("vision_config", "num_hidden_layers"),
        "hidden": ("vision_config", "hidden_size"),
        "intermediate": ("vision_config", "intermediate_size"),
        "heads": ("vision_config", "num_attention_heads"),
        "patch-size": ("vision_config", "patch_size"),
        "image-size": ("vision_config", "image_size"),
    },
    "siglip2-text": {
        "layers": ("text_config", "num_hidden_layers"),
        "hidden": ("text_config", "hidden_size"),
        "intermediate": ("text_config", "intermediate_size"),
        "heads": ("text_config", "num_attention_heads"),
        "vocab": ("text_config", "vocab_size"),
        "seq-len": ("text_config", "max_position_embeddings"),
        "projection": ("text_config", "projection_size"),
    },
}

# Deployment settings (graph shapes and precision), per target, with Qualcomm's values as defaults.
DEPLOY: dict[str, dict[str, Any]] = {
    "gemma4-text": {"context_length": 4096, "prompt_length": 128, "precision": "w4a16", "splits": None},
    # Gemma 4's image processor resizes every image to fill max_soft_tokens x pooling_kernel^2 patches,
    # so the graph is set by the soft-token budget, not by the input image size.
    "gemma4-vision": {"precision": "float"},
    "siglip2-image": {"batch": 1, "precision": "w8a16"},
    "siglip2-text": {"batch": 1, "precision": "w8a16"},
}

# Qualcomm's Gemma 4 export bounds (torch.export dims in templates/gemma4/model.py)
GEMMA_MAX_SEQ, GEMMA_MAX_KV, GEMMA_MAX_CTX = 4096, 4096, 8192
PRECISIONS = {"float", "fp16", "w8a16", "w8a8", "w4a16", "w4a8", "w16a16"}  # AI Hub quantize jobs: INT4/8/16
GEMMA_SOFT_TOKENS = (70, 140, 280, 560, 1120)  # transformers' Gemma4ImageProcessor budgets
HMX = 32  # HMX tiles are 32 wide: other sizes compile but run padded

# Knobs where "not a multiple of 32" means padded matrix work on the HMX (a warning, not an error).
_MATRIX_KNOBS = {"hidden", "intermediate", "head-dim", "global-head-dim", "vocab", "ple-dim", "text-hidden", "projection"}


def parse_dims(items: list[str] | None) -> dict[str, int]:
    """``["layers=12", "hidden=1024"]`` -> ``{"layers": 12, "hidden": 1024}``."""
    out: dict[str, int] = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"dimension {it!r} must be knob=value, e.g. layers=12")
        k, v = it.split("=", 1)
        k = k.strip().replace("_", "-")
        try:
            out[k] = int(v)
        except ValueError:
            raise ValueError(f"dimension {k} takes an integer, got {v!r}") from None
    return out


def knobs(target: str) -> dict[str, tuple[str, str]]:
    if target not in KNOBS:
        raise ValueError(f"unknown dummy target {target!r}; choose from {', '.join(TARGETS)}")
    return KNOBS[target]


# --------------------------------------------------------------------------- Gemma 4 layer layout


def layer_types(n: int, pattern: int) -> list[str]:
    """Qualcomm's ``get_gemma4_layer_types``: every ``pattern``-th layer is full attention, and the last."""
    types = ["full_attention" if (i + 1) % pattern == 0 else "sliding_attention" for i in range(n)]
    if types:
        types[-1] = "full_attention"
    return types


def pattern_of(types: list[str]) -> int:
    """Inverse of :func:`layer_types` for the real configs (the first full-attention layer)."""
    return types.index("full_attention") + 1 if "full_attention" in types else len(types) + 1


def kv_sharing_ok(n: int, shared: int, pattern: int) -> bool:
    """Every KV-shared layer reads the KV of the last earlier non-shared layer of its own type."""
    if shared == 0:
        return True
    if not 0 < shared < n:
        return False
    types = layer_types(n, pattern)
    first = n - shared
    return set(types[first:]) <= set(types[:first])


def default_splits(n: int, shared: int) -> list[int]:
    """Qualcomm's cut for E4B, generalised: [non-shared | first half of shared | rest] (+ lm_head part).

    E4B: 42 layers, 18 shared -> [24, 33]. Without KV sharing: thirds.
    """
    if shared:
        pts = [n - shared, n - shared + math.ceil(shared / 2)]
    else:
        pts = [math.ceil(n / 3), math.ceil(2 * n / 3)]
    return sorted({p for p in pts if 0 < p < n})


def gemma_parts(n: int, splits: list[int]) -> list[dict[str, Any]]:
    """Layer range of every part, then the lm_head part. Names follow Qualcomm: part1_of_4 ..."""
    bounds = [0, *splits, n]
    total = len(bounds)  # decoder parts + lm_head
    parts = [{"component": f"part{i + 1}_of_{total}", "part_id": i + 1, "layers": [bounds[i], bounds[i + 1]]}
             for i in range(len(bounds) - 1)]
    parts.append({"component": f"part{total}_of_{total}", "part_id": total, "layers": None})
    return parts


# --------------------------------------------------------------------------- resolve


@dataclass
class Resolved:
    target: str
    config: dict[str, Any]  # the full Hugging Face config to write
    dims: dict[str, int]  # every knob's value in that config (overridden or real)
    requested: dict[str, int]  # what was asked
    deploy: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    real: dict[str, int] = field(default_factory=dict)  # the knobs' values in the real model

    def to_json(self) -> dict[str, Any]:
        return {"target": self.target, "requested": self.requested, "dims": self.dims, "deploy": self.deploy,
                "real": self.real, "notes": self.notes, "warnings": self.warnings}


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    if not name:
        return cfg
    sec = cfg.get(name)
    if sec is None:
        sec = cfg[name] = {}
    return sec


def _scalar(v: Any) -> Any:
    return v[0] if isinstance(v, list) and v and isinstance(v[0], (int, float)) else v


def read_dims(target: str, cfg: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, (sec, key) in knobs(target).items():
        s = cfg.get(sec, {}) if sec else cfg
        if key == "layer_types":
            lt = s.get("layer_types")
            if lt:
                out[k] = pattern_of(lt)
            continue
        v = _scalar(s.get(key))
        if v is not None:
            out[k] = int(v)
    if target == "gemma4-text":
        t = cfg.get("text_config", {})
        out.setdefault("kv-shared-layers", int(t.get("num_kv_shared_layers") or 0))
        out.setdefault("global-head-dim", int(t.get("global_head_dim") or t.get("head_dim") or 0))
    if target == "siglip2-text":
        t = cfg.get("text_config", {})
        out.setdefault("projection", int(t.get("projection_size") or t.get("hidden_size") or 0))
    return out


def resolve(target: str, base: dict[str, Any], requested: dict[str, int], deploy: dict[str, Any] | None = None) -> Resolved:
    """Apply the requested knobs to (a copy of) the real config; derived fields follow."""
    kn = knobs(target)
    unknown = sorted(set(requested) - set(kn))
    cfg = copy.deepcopy(base)
    dep = {**DEPLOY[target], **{k: v for k, v in (deploy or {}).items() if v is not None}}
    r = Resolved(target, cfg, {}, dict(requested), dep)
    if unknown:
        r.errors.append(f"{target} has no knob(s) {unknown}; available: {', '.join(kn)}")
        return r
    real = read_dims(target, base)
    r.real = dict(real)
    missing = sorted(set(kn) - set(real) - {"kv-shared-layers", "global-head-dim", "projection", "soft-tokens"})
    if missing:
        sec = kn[missing[0]][0]
        r.errors.append(f"the base config has no {sec} values for {missing} (a text-only config cannot make a "
                        f"vision dummy): pass --base-config with the full model's config.json or its HF id")
        return r
    want = {**real, **requested}

    if target == "gemma4-text":
        _resolve_gemma_text(r, real, want)
    else:
        for k, (sec, key) in kn.items():
            if k in requested:
                s = _section(cfg, sec)
                old = s.get(key)
                s[key] = [want[k]] * len(old) if isinstance(old, list) else want[k]
        if target == "siglip2-text" and "hidden" in requested and "projection" not in requested:
            _section(cfg, "text_config")["projection_size"] = want["hidden"]  # as in the real model (768 = 768)
            r.notes.append("projection follows hidden (the real model projects 768 -> 768)")
        if target == "siglip2-text" and "seq-len" in requested:
            dep["seq_len"] = want["seq-len"]
        if target == "siglip2-text":
            dep.setdefault("seq_len", want.get("seq-len", 64))
        if target == "siglip2-image":
            dep["image_size"] = want.get("image-size", 224)
        if target == "gemma4-vision":
            dep["soft_tokens"] = want.get("soft-tokens") or 280
            # Qualcomm's VisionEncoder reads the pooler's output length from here; keep it equal to the budget
            _section(cfg, "vision_config")["default_output_length"] = dep["soft_tokens"]
    r.dims = read_dims(target, cfg)
    validate(r)
    return r


def _resolve_gemma_text(r: Resolved, real: dict[str, int], want: dict[str, int]) -> None:
    t = _section(r.config, "text_config")
    n = want["layers"]
    pattern = want.get("pattern") or real.get("pattern") or 6
    if "kv-shared-layers" in r.requested:
        shared = want["kv-shared-layers"]
    elif n != real.get("layers"):
        # keep the real fraction of KV-shared layers (E4B: 18 of 42) when only the depth changes
        shared = round(n * real.get("kv-shared-layers", 0) / max(real.get("layers", 1), 1))
        while shared and not kv_sharing_ok(n, shared, pattern):
            shared -= 1
        r.notes.append(f"kv-shared-layers not given: {shared} of {n} layers share KV "
                       f"(the real {real.get('kv-shared-layers')}/{real.get('layers')}, rounded to a valid layout)")
    else:
        shared = real.get("kv-shared-layers", 0)
    inter = t.get("intermediate_size")
    t["num_hidden_layers"] = n
    t["num_kv_shared_layers"] = shared
    t["layer_types"] = layer_types(n, pattern)
    if isinstance(inter, list):
        t["intermediate_size"] = [want["intermediate"]] * n if "intermediate" in r.requested else (
            (inter + [inter[-1]] * n)[:n])
    elif "intermediate" in r.requested:
        t["intermediate_size"] = want["intermediate"]
    for k in ("hidden", "heads", "kv-heads", "head-dim", "global-head-dim", "sliding-window", "ple-dim", "vocab"):
        if k in r.requested:
            t[KNOBS["gemma4-text"][k][1]] = want[k]
    if "vocab" in r.requested and "vocab_size_per_layer_input" in t:
        r.notes.append("vocab changes the lm_head part; the per-layer and token embedding tables are host-side "
                       "(not in any NPU graph)")
    # The per-layer-embedding table is gathered on the host (graph input per_layer_inputs), so its row count
    # never reaches the NPU; a small table keeps the dummy checkpoint small.
    if t.get("vocab_size_per_layer_input"):
        t["vocab_size_per_layer_input"] = min(int(t["vocab_size_per_layer_input"]), 1024)
    splits = r.deploy.get("splits")
    r.deploy["splits"] = list(splits) if splits else default_splits(n, shared)


# --------------------------------------------------------------------------- validate


def validate(r: Resolved) -> Resolved:
    d, dep, t = r.dims, r.deploy, r.target
    err, warn = r.errors, r.warnings
    for k, v in d.items():
        if v <= 0 and not (k == "kv-shared-layers" and v == 0):
            err.append(f"{k} must be positive (got {v})")
    if err:
        return r
    if t == "gemma4-text":
        n, shared, pattern = d["layers"], d.get("kv-shared-layers", 0), d.get("pattern", 6)
        if d["heads"] % d["kv-heads"]:
            err.append(f"heads ({d['heads']}) must be a multiple of kv-heads ({d['kv-heads']})")
        if not 0 <= shared < n:
            err.append(f"kv-shared-layers ({shared}) must be smaller than layers ({n})")
        elif not kv_sharing_ok(n, shared, pattern):
            err.append(f"with {n} layers, pattern {pattern} and {shared} KV-shared layers some shared layer has no "
                       f"earlier non-shared layer of its type to read KV from; lower kv-shared-layers or the pattern")
        for k in ("head-dim", "global-head-dim"):
            if d.get(k, 2) % 2:
                err.append(f"{k} must be even (RoPE rotates pairs)")
        ctx, prompt = int(dep["context_length"]), int(dep["prompt_length"])
        if not 1 < prompt < ctx:
            err.append(f"prompt-length ({prompt}) must be > 1 and < context-length ({ctx})")
        if ctx > GEMMA_MAX_CTX or ctx - 1 > GEMMA_MAX_KV or prompt > GEMMA_MAX_SEQ:
            err.append(f"Qualcomm's Gemma 4 export bounds: context <= {GEMMA_MAX_CTX}, KV <= {GEMMA_MAX_KV}, "
                       f"sequence <= {GEMMA_MAX_SEQ}")
        sp = dep.get("splits") or []
        if sorted(set(sp)) != list(sp) or any(not 0 < p < n for p in sp):
            err.append(f"splits {sp} must be increasing layer indices between 1 and {n - 1}")
    elif t == "gemma4-vision":
        if d["heads"] % d.get("kv-heads", d["heads"]):
            err.append(f"heads ({d['heads']}) must be a multiple of kv-heads ({d.get('kv-heads')})")
        if d.get("soft-tokens", 280) not in GEMMA_SOFT_TOKENS:
            err.append(f"soft-tokens must be one of {GEMMA_SOFT_TOKENS} (Gemma 4 image processor budgets)")
        if d["head-dim"] % 2:
            err.append("head-dim must be even (2-D RoPE)")
    elif t in ("siglip2-image", "siglip2-text"):
        if d["hidden"] % d["heads"]:
            err.append(f"hidden ({d['hidden']}) must be a multiple of heads ({d['heads']}): SigLIP derives head_dim")
        if t == "siglip2-image" and d["image-size"] % d["patch-size"]:
            err.append(f"image-size {d['image-size']} must be a multiple of patch-size {d['patch-size']}")
        if int(dep.get("batch", 1)) < 1:
            err.append("batch must be >= 1")
    prec = str(dep.get("precision") or "")
    if prec not in PRECISIONS:
        err.append(f"precision {prec!r} must be one of {sorted(PRECISIONS)}")
    for k, v in d.items():
        if k in _MATRIX_KNOBS and k in r.requested and v % HMX:
            warn.append(f"{k}={v} is not a multiple of {HMX}: it compiles, but the HMX pads it (part of what you "
                        f"measure, and different from the real model's aligned sizes)")
    if t in ("siglip2-image", "siglip2-text") and "hidden" in r.requested and (d["hidden"] // d["heads"]) % HMX:
        warn.append(f"head_dim = hidden/heads = {d['hidden'] // d['heads']} is not a multiple of {HMX}")
    return r


# --------------------------------------------------------------------------- what the device must show


def gemma_graph_names(dep: dict[str, Any], n_parts: int, part_id: int) -> dict[str, tuple[int, int]]:
    """Qualcomm's graph names: {prompt|token}_ar{seq}_cl{ctx}_{i}_of_{n}."""
    ctx = int(dep["context_length"])
    return {f"{'token' if s == 1 else 'prompt'}_ar{s}_cl{ctx}_{part_id}_of_{n_parts}": (s, ctx)
            for s in (int(dep["prompt_length"]), 1)}


def gemma_kv_inputs(d: dict[str, int], ar: int, ctx: int) -> dict[str, list[int]]:
    """Every past-KV input of the whole model for one graph: per non-shared layer and KV head."""
    n, shared, pattern = d["layers"], d.get("kv-shared-layers", 0), d.get("pattern", 6)
    types = layer_types(n, pattern)
    kv_len = ctx - ar
    swa = min(d["sliding-window"], kv_len)
    out: dict[str, list[int]] = {}
    for i in range(n - shared):
        sliding = types[i] == "sliding_attention"
        hd = d["head-dim"] if sliding else d.get("global-head-dim", d["head-dim"])
        ln = swa if sliding else kv_len
        pfx = "swa" if sliding else "past"
        for h in range(d["kv-heads"]):
            out[f"{pfx}_key_{i}_h{h}_in"] = [1, 1, hd, ln]
            out[f"{pfx}_value_{i}_h{h}_in"] = [1, 1, ln, hd]
    return out


def gemma_fixed_inputs(d: dict[str, int], ar: int, ctx: int) -> dict[str, list[int]]:
    """Shapes of the named (non-KV) inputs wherever a part has them."""
    swa = min(d["sliding-window"], ctx - ar)
    return {
        "inputs_embeds": [1, ar, d["hidden"]],
        "per_layer_inputs": [1, ar, d["layers"], d["ple-dim"]],
        "attention_mask": [1, 1, ar, ctx],
        "swa_attention_mask": [1, 1, ar, swa + ar],
        "swa_position_ids_cos": [1, 1, ar, d["head-dim"] // 2],
        "swa_position_ids_sin": [1, 1, ar, d["head-dim"] // 2],
        "position_ids_cos": [1, 1, ar, d.get("global-head-dim", d["head-dim"]) // 2],
        "position_ids_sin": [1, 1, ar, d.get("global-head-dim", d["head-dim"]) // 2],
    }


def expected_io(r: Resolved | dict[str, Any]) -> dict[str, Any]:
    """Shapes the deployed graphs must have, from the knobs alone (not from the code that built them)."""
    rj = r.to_json() if isinstance(r, Resolved) else r
    t, d, dep = rj["target"], rj["dims"], rj["deploy"]
    if t == "gemma4-text":
        n_parts = len(dep["splits"]) + 2
        graphs = {}
        for p in gemma_parts(d["layers"], dep["splits"]):
            for g, (ar, ctx) in gemma_graph_names(dep, n_parts, p["part_id"]).items():
                graphs[f"{p['component']}/{g}"] = {"ar": ar, "ctx": ctx, "part": p}
        return {"graphs": graphs, "logits": {"last_dim": d["vocab"]}, "n_parts": n_parts}
    if t == "siglip2-image":
        s, b = d["image-size"], int(dep.get("batch", 1))
        return {"inputs": {"image": [b, 3, s, s]}, "output_last_dim": d["hidden"], "batch": b,
                "patches": (s // d["patch-size"]) ** 2}
    if t == "siglip2-text":
        b = int(dep.get("batch", 1))
        return {"inputs": {"input_ids": [b, int(dep.get("seq_len") or d["seq-len"])]},
                "output_last_dim": d.get("projection") or d["hidden"], "batch": b}
    if t == "gemma4-vision":
        k, tokens = d["pooling-kernel"], d.get("soft-tokens", 280)
        return {"inputs": {"pixel_values": [1, tokens * k * k, 3 * d["patch-size"] ** 2],
                           "image_position_ids": [1, tokens * k * k, 2]},
                "output_last_dim": d["text-hidden"], "max_tokens": tokens}
    raise ValueError(t)


# --------------------------------------------------------------------------- sizes (informational)


def param_estimate(r: Resolved | dict[str, Any]) -> dict[str, float]:
    """Matrix weights of the deployed graphs (norms and biases ignored), for the summary table."""
    rj = r.to_json() if isinstance(r, Resolved) else r
    t, d = rj["target"], rj["dims"]
    if t == "gemma4-text":
        n, shared, H = d["layers"], d.get("kv-shared-layers", 0), d["hidden"]
        types = layer_types(n, d.get("pattern", 6))
        decoder = 0
        for i in range(n):
            hd = d["head-dim"] if types[i] == "sliding_attention" else d.get("global-head-dim", d["head-dim"])
            decoder += 2 * H * d["heads"] * hd  # q, o
            if i < n - shared:
                decoder += 2 * H * d["kv-heads"] * hd  # k, v
            decoder += 3 * H * d["intermediate"]  # gate, up, down
            decoder += 2 * H * d["ple-dim"]  # per-layer-embedding gate / projection
        decoder += H * n * d["ple-dim"]  # per-layer model projection
        return {"decoder": float(decoder), "lm_head": float(H * d["vocab"]), "total": float(decoder + H * d["vocab"])}
    if t in ("siglip2-image", "siglip2-text", "gemma4-vision"):
        H, I, n = d["hidden"], d["intermediate"], d["layers"]
        per = 4 * H * H + 2 * H * I
        return {"encoder": float(n * per), "total": float(n * per)}
    return {}
