"""Gemma 4 vision encoder and SigLIP 2 encoders: Qualcomm's wrappers on a shrunk random checkpoint.

Qualcomm's ``from_pretrained`` loads any Hugging Face checkpoint folder (SigLIP 2: ``weights=``, Gemma 4
vision: ``checkpoint=``) and applies the same NPU adaptations it applies to the real weights, so the
existing component runner (:mod:`hexprof.study.components`: upload -> quantize -> compile -> link ->
profile) deploys a dummy unchanged once it is pointed at the folder.

Gemma 4's vision graph geometry comes from its image processor, which resizes every image to fill
``max_soft_tokens * pooling_kernel_size**2`` patches (padding the rest). :func:`gemma_processor` sets
that budget (70 / 140 / 280 / 560 / 1120 soft tokens) and, for random checkpoints that ship no
tokenizer, stands in for the full processor with transformers' own ``Gemma4ImageProcessor``, which
produces the same ``pixel_values`` / ``image_position_ids``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

CKPT_KW = {"gemma4-vision": "checkpoint", "siglip2-image": "weights", "siglip2-text": "weights"}


class _ImageOnlyProcessor:
    """What Qualcomm's VisionEncoder.from_pretrained needs from AutoProcessor: pixel_values + positions."""

    def __init__(self, image_processor: Any) -> None:
        self.image_processor = image_processor

    def apply_chat_template(self, messages: list[dict[str, Any]], **_: Any) -> Any:
        images = [c["image"] for m in messages for c in m.get("content", []) if c.get("type") == "image"]
        return self.image_processor(images=images, return_tensors="pt")


GEMMA_REPO = "google/gemma-4-E4B-it"


def _image_processor(src: Any, used: dict[str, Any] | None = None, **overrides: Any) -> Any:
    """Gemma 4's own image processor (its settings from the checkpoint, else from Google's repo; it needs
    no tokenizer), else transformers' Gemma4ImageProcessor with its defaults."""
    from transformers import AutoImageProcessor

    ip = None
    for cand in filter(None, (src, GEMMA_REPO)):
        try:
            ip = AutoImageProcessor.from_pretrained(cand)
            if used is not None:
                used["image_processor_source"] = str(cand)
            break
        except Exception:  # noqa: BLE001 - random checkpoints have no preprocessor config; the Hub may be unreachable
            continue
    if ip is None:
        if used is not None:
            used["image_processor_source"] = "transformers Gemma4ImageProcessor defaults (Gemma 4's own config unreachable)"
        try:
            from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor as IP
        except ImportError:  # pragma: no cover - older transformers
            from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil as IP  # type: ignore
        ip = IP()
    for k, v in overrides.items():
        if v is not None:
            setattr(ip, k, v)
    return ip


@contextlib.contextmanager
def gemma_processor(max_soft_tokens: int | None = None, patch_size: int | None = None,
                    pooling_kernel_size: int | None = None, image_only: bool = False) -> Iterator[dict[str, Any]]:
    """Patch ``transformers.AutoProcessor.from_pretrained`` while Qualcomm's VisionEncoder loads.

    ``image_only``: never load a tokenizer (random checkpoints); otherwise the real processor is loaded
    and only its image settings are overridden. The dict yielded records what was used.
    """
    import transformers

    orig = transformers.AutoProcessor.from_pretrained
    used: dict[str, Any] = {}
    over = {"max_soft_tokens": max_soft_tokens, "patch_size": patch_size, "pooling_kernel_size": pooling_kernel_size}

    def patched(src: Any, *a: Any, **kw: Any) -> Any:
        proc = None
        if not image_only:
            try:
                proc = orig(src, *a, **kw)
            except Exception:  # noqa: BLE001
                proc = None
        if proc is None:
            proc = _ImageOnlyProcessor(_image_processor(src, used, **over))
        else:
            for k, v in over.items():
                if v is not None:
                    setattr(proc.image_processor, k, v)
        ip = proc.image_processor
        used.update(kind=type(proc).__name__, max_soft_tokens=getattr(ip, "max_soft_tokens", None),
                    patch_size=getattr(ip, "patch_size", None), pooling_kernel_size=getattr(ip, "pooling_kernel_size", None))
        return proc

    transformers.AutoProcessor.from_pretrained = patched  # type: ignore[method-assign]
    try:
        yield used
    finally:
        transformers.AutoProcessor.from_pretrained = orig  # type: ignore[method-assign]


def processor_context(target: str, s: dict[str, Any]) -> Any:
    """The context Qualcomm's ``from_pretrained`` runs in for these settings (a no-op except for Gemma vision)."""
    if target.replace("dummy-", "") != "gemma4-vision" or not (s.get("soft_tokens") or s.get("dummy")):
        return contextlib.nullcontext({})
    proc = s.get("processor") or {}
    return gemma_processor(max_soft_tokens=s.get("soft_tokens"), patch_size=proc.get("patch_size"),
                           pooling_kernel_size=proc.get("pooling_kernel_size"), image_only=bool(s.get("dummy")))


def fix_static_pooler(model: Any, hidden: int) -> int:
    """Qualcomm's static vision pooler sizes its padding with ``768`` when ``root_hidden_size`` is a plain
    number (it is), i.e. it assumes the real Gemma 4 vision width. For other widths the padding tensor is
    rebuilt with the encoder's own width; the real model (768) is untouched. Returns how many were fixed."""
    import torch

    fixed = 0
    for mod in model.modules():
        pad = getattr(mod, "pad_zeros", None)
        if isinstance(pad, torch.Tensor) and pad.dim() == 3 and pad.shape[-1] != hidden:
            mod.pad_zeros = torch.zeros(pad.shape[0], pad.shape[1], hidden, dtype=pad.dtype, device=pad.device)
            fixed += 1
    return fixed


def loaded_config(model: Any) -> dict[str, Any]:
    """The Hugging Face config of the module Qualcomm's wrapper actually built (what got traced)."""
    for path in ("model.config", "veg.vision_tower.config"):
        obj: Any = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "to_dict"):
            d = obj.to_dict()
            keep = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                    "head_dim", "patch_size", "image_size", "pooling_kernel_size", "vocab_size", "max_position_embeddings",
                    "projection_size", "default_output_length")
            out = {k: d[k] for k in keep if k in d}
            for side in ("vision_config", "text_config"):
                if isinstance(d.get(side), dict):
                    out[side] = {k: d[side][k] for k in keep if k in d[side]}
            return out
    return {}
