"""Write a Hugging Face checkpoint with the requested dimensions and random weights.

The config starts from the real model's ``config.json`` (every field not overridden keeps its real
value: activations, norms, RoPE settings, soft-capping ...). Weights are drawn by the model class's
own initializer under a fixed seed. Needs ``torch`` and ``transformers`` (the versions
``qai-hub-models install gemma_4_e4b_it`` / ``siglip2`` pin).

Parts of the checkpoint that never reach the NPU are kept small, which changes nothing that runs
on the phone:

* Gemma 4: the audio tower is dropped (the AI Hub deployment is image + text); the per-layer
  embedding table has at most 1024 rows (it is gathered on the host, ``per_layer_inputs``); for the
  vision target the language model has one small layer (only its width reaches the vision graph),
  for the text target the vision tower is dropped.
* SigLIP 2: the tower the target does not trace is reduced to one small layer.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any

from hexprof.dummy import spec as S
from hexprof.io_utils import read_json, write_json

GEMMA_REPO = "google/gemma-4-E4B-it"
SIGLIP_REPO = "google/siglip2-base-patch16-224"


def _qaihm_gemma_config() -> Path | None:
    roots = [os.environ.get("QAIHM_STORE_ROOT"), os.environ.get("HEXPROF_DOWNLOAD_DIR"), str(Path.home())]
    for root in filter(None, roots):
        hits = sorted(glob.glob(os.path.join(root, ".qaihm", "qai-hub-models", "models", "gemma_4_e4b_it", "v*",
                                             "*", "config.json")))
        if hits:
            return Path(hits[-1])
    return None


def _normalize(family: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Gemma 4 text-only configs (``model_type: gemma4_text``) become the multimodal layout the knobs address."""
    if family == "gemma4" and "text_config" not in cfg and str(cfg.get("model_type", "")).startswith("gemma4"):
        return {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
                "text_config": {**cfg, "model_type": "gemma4_text"}}
    return cfg


def base_config(family: str, base: str | Path | None = None) -> tuple[dict[str, Any], str]:
    """(the real model's config, where it came from). ``base``: a config.json, a checkpoint folder or an HF id."""
    cfg, src = _base_config(family, base)
    return _normalize(family, cfg), src


def _base_config(family: str, base: str | Path | None = None) -> tuple[dict[str, Any], str]:
    if base:
        p = Path(base)
        if p.is_dir():
            p = p / "config.json"
        if p.is_file():
            return read_json(p), str(p)
        repo = str(base)
    else:
        repo = GEMMA_REPO if family == "gemma4" else SIGLIP_REPO
        if family == "gemma4":
            hit = _qaihm_gemma_config()
            if hit:
                return read_json(hit), str(hit)
    try:
        from huggingface_hub import hf_hub_download

        return read_json(hf_hub_download(repo, "config.json")), f"huggingface:{repo}"
    except Exception as e:  # noqa: BLE001
        if family == "siglip2" and not base:
            return siglip2_builtin(), "built-in google/siglip2-base-patch16-224 values (Hugging Face unreachable)"
        raise RuntimeError(
            f"could not read the real {family} config ({type(e).__name__}: {e}). Pass --base-config with the "
            f"model's config.json (Gemma 4: Qualcomm's export stores it under ~/.qaihm/.../gemma_4_e4b_it_w4a16/), "
            f"or log in to Hugging Face (`hf auth login`) after accepting the model licence."
        ) from e


def siglip2_builtin() -> dict[str, Any]:
    """google/siglip2-base-patch16-224, for when the Hub is unreachable (SigLIP architecture, Gemma vocabulary)."""
    common = {"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12, "num_attention_heads": 12,
              "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6, "attention_dropout": 0.0}
    return {"model_type": "siglip", "architectures": ["SiglipModel"], "initializer_factor": 1.0,
            "text_config": {**common, "model_type": "siglip_text_model", "vocab_size": 256000,
                            "max_position_embeddings": 64, "projection_size": 768},
            "vision_config": {**common, "model_type": "siglip_vision_model", "image_size": 224, "patch_size": 16,
                              "num_channels": 3}}


_TINY_SIGLIP = {"num_hidden_layers": 1, "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 1}


def _trim_unused(target: str, cfg: dict[str, Any], notes: list[str]) -> None:
    if target == "siglip2-image":  # the text tower is never traced for the image encoder
        cfg.setdefault("text_config", {}).update(**_TINY_SIGLIP, vocab_size=1000, projection_size=64)
        notes.append("text tower reduced to 1 small layer (the image encoder graph does not use it)")
    elif target == "siglip2-text":  # and the vision tower never for the text encoder
        cfg.setdefault("vision_config", {}).update(**_TINY_SIGLIP, image_size=32, patch_size=16)
        notes.append("vision tower reduced to 1 small layer (the text encoder graph does not use it)")
    if S.FAMILY[target] != "gemma4":
        return
    if cfg.pop("audio_config", None) is not None:
        notes.append("audio tower dropped (not part of the AI Hub image + text deployment)")
    t = cfg.get("text_config") or {}
    if t.get("vocab_size_per_layer_input"):
        t["vocab_size_per_layer_input"] = min(int(t["vocab_size_per_layer_input"]), 1024)
    if target == "gemma4-vision":
        # Only the language model's width reaches the vision graph (embed_vision projects to it).
        t.update(num_hidden_layers=1, num_kv_shared_layers=0, layer_types=["full_attention"],
                 vocab_size=min(int(t.get("vocab_size") or 1024), 1024))
        if isinstance(t.get("intermediate_size"), list):
            t["intermediate_size"] = t["intermediate_size"][:1]
        notes.append("language model reduced to 1 layer: the vision encoder only uses its hidden size")
    elif target == "gemma4-text":
        if cfg.pop("vision_config", None) is not None:
            notes.append("vision tower dropped (the text parts do not use it)")


def processor_overrides(r: S.Resolved) -> dict[str, Any]:
    """Image-processor settings that must agree with a shrunk Gemma 4 vision config."""
    if r.target != "gemma4-vision":
        return {}
    v = r.config.get("vision_config") or {}
    return {"patch_size": v.get("patch_size"), "pooling_kernel_size": v.get("pooling_kernel_size"),
            "max_soft_tokens": int(r.deploy.get("soft_tokens") or 280)}


def make(r: S.Resolved, out_dir: str | Path, seed: int = 0, base_source: str = "",
         dtype: str = "bfloat16") -> dict[str, Any]:
    """Write ``config.json`` + random ``*.safetensors`` with the resolved dimensions."""
    if r.errors:
        raise ValueError("; ".join(r.errors))
    out = Path(out_dir)
    marker = out / "dummy.json"
    want = {"resolved": r.to_json(), "seed": seed, "dtype": dtype}
    if marker.exists():
        old = read_json(marker)
        if {k: old.get(k) for k in want} == want and any(out.glob("*.safetensors")):
            print(f"    checkpoint exists: {out}")
            return old
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers

    cfg = json.loads(json.dumps(r.config))
    notes = list(r.notes)
    _trim_unused(r.target, cfg, notes)
    cfg.pop("transformers_version", None)
    cfg.pop("_name_or_path", None)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    config = transformers.AutoConfig.from_pretrained(str(out))
    torch.manual_seed(seed)
    tdtype = getattr(torch, dtype)
    auto = transformers.AutoModelForImageTextToText if S.FAMILY[r.target] == "gemma4" else transformers.AutoModel
    try:
        model = auto.from_config(config, dtype=tdtype)
    except TypeError:  # transformers < 4.56
        model = auto.from_config(config, torch_dtype=tdtype)
    n_params = sum(p.numel() for p in model.parameters())
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="4GB")
    del model

    proc = processor_overrides(r)
    if proc:
        notes.append(f"Gemma 4 image processor settings for this checkpoint: {proc} "
                     f"(applied by hexprof.dummy.components.gemma_processor; no tokenizer is needed)")
    info = {**want, "base_config": base_source, "checkpoint": str(out), "parameters": n_params, "notes": notes,
            "transformers": transformers.__version__, "torch": torch.__version__}
    write_json(marker, info)
    print(f"    wrote {out}: {n_params / 1e6:.1f} M parameters ({dtype}, seed {seed})")
    return info
