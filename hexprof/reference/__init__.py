"""Published AI Hub reference numbers bundled with hexprof."""

from __future__ import annotations

from importlib import resources
from typing import Any

import yaml


def published_s25() -> dict[str, Any]:
    text = resources.files(__package__).joinpath("aihub_published_s25.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def published_gemma4_llm_metric(context_length: int, compute: str, runtime: str = "geniex_llamacpp") -> dict | None:
    ref = published_s25()["gemma_4_e4b_it"]["q4_0"].get(runtime, {})
    for m in ref.get("llm_metrics", []):
        if m["context_length"] == context_length and m["desired_compute_unit"] == compute:
            return m
    return None


def published_siglip2(precision: str, component: str, runtime: str) -> dict | None:
    return published_s25()["siglip2"].get(precision, {}).get(component, {}).get(runtime)
