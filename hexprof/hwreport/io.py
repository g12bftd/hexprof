"""Model inputs and outputs: read every spec format, and say what the shapes mean.

Formats handled by :func:`normalize`
* AI Hub ``Model.input_spec`` / ``output_spec`` (IOSpec): ``{graph name or None: [TensorSpec(name,
  dtype, shape, ...)]}``. A linked Gemma 4 part holds both its prompt and token graphs.
* AI Hub job ``shapes`` (InputSpecs): ``{name: shape}`` or ``{name: (shape, dtype)}``.
* qai_hub_models ``InputSpec``: ``{name: TensorSpec(shape=..., dtype=...)}`` or ``(shape, dtype)``.
* hexprof's own JSON: ``{name: {"shape": [...], "dtype": "..."}}``.

:func:`describe` turns the shapes into the numbers people ask for: tokens per call, context
length, image size, image tokens produced, batch, and what comes out (vocabulary scores,
hidden states, embeddings).
"""

from __future__ import annotations

import math
import re
from typing import Any

_GRAPH = re.compile(r"(prompt|token)_ar(\d+)_cl(\d+)")


def _dims(x: Any) -> list[int] | None:
    try:
        return [int(d) for d in x]
    except (TypeError, ValueError):
        return None


def _one(name: str, s: Any) -> dict[str, Any] | None:
    if isinstance(s, dict) and "shape" in s:
        d = _dims(s.get("shape") or [])
        return {"shape": d, "dtype": str(s.get("dtype")) if s.get("dtype") is not None else None} if d is not None else None
    shape = getattr(s, "shape", None)
    if shape is not None and not isinstance(s, (list,)):
        d = _dims(shape)
        return {"shape": d, "dtype": str(getattr(s, "dtype", None))} if d is not None else None
    if isinstance(s, (tuple, list)) and s and isinstance(s[0], (tuple, list)):
        d = _dims(s[0])
        return {"shape": d, "dtype": str(s[1]) if len(s) > 1 else None} if d is not None else None
    d = _dims(s)
    return {"shape": d, "dtype": None} if d is not None else None


def normalize(spec: Any, graph: str | None = None) -> dict[str, dict[str, Any]]:
    """``{tensor name: {"shape": [...], "dtype": ...}}`` from any supported spec (``graph`` picks one graph)."""
    if not spec:
        return {}
    items = dict(spec)
    lists = {k: v for k, v in items.items() if isinstance(v, list) and v and hasattr(v[0], "_fields")}
    if lists:  # AI Hub IOSpec, keyed by graph
        if graph is not None and graph in lists:
            chosen = lists[graph]
        elif graph is not None:
            hit = next((k for k in lists if k and (graph in k or k in graph)), None)
            chosen = lists[hit] if hit else [t for v in lists.values() for t in v]
        else:
            chosen = [t for v in lists.values() for t in v]
        return {t.name: {"shape": _dims(t.shape) if t.shape is not None else None, "dtype": str(t.dtype)} for t in chosen}
    out = {}
    for k, v in items.items():
        one = _one(str(k), v)
        if one is not None:
            out[str(k)] = one
    return out


def shapes(spec: Any, graph: str | None = None) -> dict[str, list[int]]:
    return {k: v["shape"] for k, v in normalize(spec, graph).items() if v.get("shape") is not None}


# --------------------------------------------------------------------------- meaning


def _meaning(name: str, shape: list[int] | None, direction: str, ctx: dict[str, Any]) -> str:
    if not shape:
        return ""
    n = name.lower()
    if direction == "in":
        if n == "image" or (len(shape) == 4 and 3 in (shape[1], shape[-1])):
            h, w = (shape[2], shape[3]) if shape[1] == 3 else (shape[1], shape[2])
            return f"{shape[0]} image(s) of {w}×{h} px, RGB"
        if "pixel_values" in n and len(shape) == 3:
            p = int(round(math.sqrt(shape[-1] / 3))) if shape[-1] % 3 == 0 else None
            side = int(round(math.sqrt(shape[1]))) if p else None
            px = f" ≈ {side * p}×{side * p} px image" if side and side * side == shape[1] else ""
            return f"{shape[1]} patches of {p}×{p} px{px}" if p and p * p * 3 == shape[-1] else f"{shape[1]} image patches"
        if "position" in n and "image" in n:
            return f"position of each of {shape[1]} patches" if len(shape) >= 2 else ""
        if "input_ids" in n and len(shape) == 2:
            return f"{shape[0]} text(s) × {shape[1]} tokens"
        if any(k in n for k in ("embed", "hidden")) and len(shape) == 3:
            return f"{shape[1]} token(s) per call, hidden size {shape[2]}"
        if "mask" in n and shape:
            return f"attention over {shape[-1]} positions (context)"
        if any(k in n for k in ("past", "key", "value", "kv")):
            return "KV cache in"
        return ""
    if len(shape) >= 2 and shape[-1] >= 32000:
        return f"vocabulary scores: {shape[-2] if len(shape) >= 2 else 1} position(s) × {shape[-1]:,} words"
    if "vision_embedding" in n or ("image" in n and len(shape) == 3):
        return f"{shape[1]} image tokens for the language model, hidden size {shape[2]}"
    if any(k in n for k in ("past", "key", "value", "kv")):
        return "KV cache out (new tokens)"
    if len(shape) == 2:
        return f"{shape[0]} embedding(s) of size {shape[1]}"
    if len(shape) == 3 and any(k in n for k in ("hidden", "embed", "output", "out")):
        return f"hidden states for {shape[1]} token(s), size {shape[2]}"
    return ""


def table(inputs: dict[str, dict[str, Any]], outputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    kv_in = [k for k in inputs if any(t in k.lower() for t in ("past", "key_", "value_", "kv"))]
    kv_out = [k for k in outputs if any(t in k.lower() for t in ("past", "key_", "value_", "kv"))]
    for direction, spec, kv in (("in", inputs, kv_in), ("out", outputs, kv_out)):
        for name, s in spec.items():
            if name in kv[1:]:  # many identical KV tensors: show the first, count the rest
                continue
            rows.append({"direction": direction, "name": name + (f" (+{len(kv) - 1} more KV tensors)" if name in kv[:1] and len(kv) > 1 else ""),
                         "shape": "×".join(map(str, s.get("shape") or [])) or "–", "dtype": s.get("dtype") or "–",
                         "meaning": _meaning(name, s.get("shape"), direction, {})})
    return rows


def describe(label: str, inputs: dict[str, dict[str, Any]], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Headline numbers of a graph's I/O (``None`` where the shapes don't say)."""
    d: dict[str, Any] = {}
    m = _GRAPH.search(label)
    if m:
        d.update(role="prompt (prefill)" if m.group(1) == "prompt" else "token (decode)",
                 tokens_per_call=int(m.group(2)), context_length=int(m.group(3)))
    for name, s in inputs.items():
        sh, n = s.get("shape") or [], name.lower()
        if not sh:
            continue
        if n == "image" or (len(sh) == 4 and 3 in (sh[1], sh[-1])):
            h, w = (sh[2], sh[3]) if sh[1] == 3 else (sh[1], sh[2])
            d.update(role=d.get("role", "image encoder"), image=f"{w}×{h}", batch=sh[0])
        elif "pixel_values" in n and len(sh) == 3:
            d.update(role=d.get("role", "vision encoder"), image_patches=sh[1])
            if sh[-1] % 3 == 0:
                p = int(round(math.sqrt(sh[-1] / 3)))
                side = int(round(math.sqrt(sh[1])))
                if p * p * 3 == sh[-1] and side * side == sh[1]:
                    d.update(patch=p, image=f"{side * p}×{side * p}")
        elif "input_ids" in n and len(sh) == 2:
            d.update(role=d.get("role", "text encoder"), batch=sh[0], tokens_per_call=sh[1])
        elif any(k in n for k in ("embed", "hidden")) and len(sh) == 3 and "tokens_per_call" not in d:
            d["tokens_per_call"] = sh[1]
        elif "mask" in n and "context_length" not in d:
            d["context_length"] = sh[-1]
    outs = []
    for name, s in outputs.items():
        sh, n = s.get("shape") or [], name.lower()
        if not sh or any(k in n for k in ("past", "key_", "value_", "kv")):
            continue
        if len(sh) >= 2 and sh[-1] >= 32000:
            d["vocab"] = sh[-1]
            d["logit_positions"] = sh[-2]
            outs.append(f"vocabulary scores for {sh[-2]} position(s)")
        elif "vision_embedding" in n or ("image" in n and len(sh) == 3):
            d["image_tokens_out"] = sh[1]
            outs.append(f"{sh[1]} image tokens × {sh[2]}")
        elif len(sh) == 2:
            d["embedding_dim"] = sh[1]
            outs.append(f"embedding of size {sh[1]}")
        elif len(sh) == 3:
            outs.append(f"hidden states {sh[1]} × {sh[2]}")
    if outs:
        d["output"] = "; ".join(outs)
    if any(any(k in n.lower() for k in ("past", "key_", "value_", "kv")) for n in outputs):
        d["output"] = (d.get("output", "") + "; " if d.get("output") else "") + "KV cache for the new tokens"
    return d
