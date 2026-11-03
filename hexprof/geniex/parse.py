"""Parse geniex-bench output (per-cell JSON + ``--accuracy`` stdout) into metrics.

geniex-bench (GenieX SDK, schema 3..7) writes one JSON per cell::

    {"cell_id": ..., "plugin": ..., "device": ..., "params": {...},
     "runs": [{"ttft_us", "media_us", "prompt_time_us", "decode_time_us",
               "prompt_tokens", "gen_tokens", "prefill_tps", "decode_tps",
               "stop_reason"}, ...],
     "agg": {"ttft_ms": {"median",...}, "prefill_tps": {...}, "decode_tps": {...},
             "gen_tokens": {"median"}, "prompt_tokens": {"median"}, "media_ms": {"median"}}}

``media_us`` is the vision (or audio) encoder, ``prompt_time_us`` the prefill of
text + media tokens, ``decode_time_us`` the generation loop; on a VLM run TTFT is
roughly media + prefill.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hexprof.io_utils import summarize

# Gemma 4's prompt processor (AR-128) and the QAIRT engine both prefill in
# 128-token chunks; AI Hub's TTFT range starts at one such chunk.
PROMPT_CHUNK = 128


@dataclass
class RunRecord:
    run_idx: int
    ttft_ms: float
    media_ms: float
    prompt_ms: float
    decode_ms: float
    prompt_tokens: int
    gen_tokens: int
    prefill_tps: float
    decode_tps: float
    stop_reason: str | None = None

    @property
    def e2e_ms(self) -> float:
        """Request latency: time to first token + the decode loop."""
        return self.ttft_ms + self.decode_ms

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["e2e_ms"] = self.e2e_ms
        return d


@dataclass
class Cell:
    cell_id: str
    plugin: str
    device: str
    model_id: str | None
    model_path: str | None
    schema_version: str
    versions: dict[str, Any]
    params: dict[str, Any]
    runs: list[RunRecord]
    agg: dict[str, Any] = field(default_factory=dict)

    @property
    def context_length(self) -> int | None:
        _, sep, suffix = self.cell_id.rpartition("-c")
        if sep and suffix.isdigit():
            return int(suffix)
        n_ctx = self.params.get("n_ctx")
        return int(n_ctx) if n_ctx else None


def _run_from_json(r: dict) -> RunRecord:
    return RunRecord(
        run_idx=int(r.get("run_idx", 0)),
        ttft_ms=float(r.get("ttft_us", 0)) / 1000.0,
        media_ms=float(r.get("media_us", 0) or 0) / 1000.0,
        prompt_ms=float(r.get("prompt_time_us", 0) or 0) / 1000.0,
        decode_ms=float(r.get("decode_time_us", 0) or 0) / 1000.0,
        prompt_tokens=int(r.get("prompt_tokens", 0) or 0),
        gen_tokens=int(r.get("gen_tokens", 0) or 0),
        prefill_tps=float(r.get("prefill_tps", 0) or 0),
        decode_tps=float(r.get("decode_tps", 0) or 0),
        stop_reason=r.get("stop_reason"),
    )


def parse_cell(src: str | Path | dict) -> Cell:
    data = src if isinstance(src, dict) else json.loads(Path(src).read_text(encoding="utf-8"))
    if "cell_id" not in data or "runs" not in data:
        raise ValueError(f"not a geniex-bench cell JSON: {src if not isinstance(src, dict) else '<dict>'}")
    return Cell(
        cell_id=str(data.get("cell_id", "")),
        plugin=str(data.get("plugin", "")),
        device=str(data.get("device", "")),
        model_id=data.get("model_id"),
        model_path=data.get("model_path"),
        schema_version=str(data.get("schema_version", "")),
        versions={k: data.get(k) for k in ("geniex_version", "qairt_version", "llama_cpp_version") if k in data},
        params=dict(data.get("params") or {}),
        runs=[_run_from_json(r) for r in data.get("runs") or []],
        agg=dict(data.get("agg") or {}),
    )


def _agg_median(cell: Cell, key: str) -> float | None:
    entry = cell.agg.get(key) or {}
    if "median" in entry:
        return float(entry["median"])
    return None


# --------------------------------------------------------------------------- AI Hub LLM metrics


def aihub_llm_metrics(cell: Cell, desired_compute_unit: str | None = None) -> dict[str, Any]:
    """The LLM metrics AI Hub Models publishes (perf.yaml ``llm_metrics``) for one cell.

    Follows ``qai_hub_models.utils.llm.geniex.jobs`` (median over the measured runs):

    * ``tokens_per_second``          = median decode tok/s
    * ``prefill_tokens_per_second``  = median prefill tok/s
    * ``time_to_first_token_range_milliseconds`` = measured TTFT rescaled from the
      measured prompt length to [one 128-token chunk, the full context]. This
      reproduces the published Galaxy S25 Gemma 4 values exactly (see tests).
    """
    ttft = _agg_median(cell, "ttft_ms")
    prefill = _agg_median(cell, "prefill_tps")
    decode = _agg_median(cell, "decode_tps")
    prompt_tokens = _agg_median(cell, "prompt_tokens")
    if ttft is None and cell.runs:  # recompute from runs if agg is missing
        ttft = summarize(r.ttft_ms for r in cell.runs)["median"]
        prefill = summarize(r.prefill_tps for r in cell.runs)["median"]
        decode = summarize(r.decode_tps for r in cell.runs)["median"]
        prompt_tokens = summarize(r.prompt_tokens for r in cell.runs)["median"]
    ctx = cell.context_length
    ttft_range = None
    if ttft is not None and prompt_tokens:
        ttft_range = {
            "min": ttft * PROMPT_CHUNK / prompt_tokens,
            "max": ttft * (ctx or prompt_tokens) / prompt_tokens,
        }
    return {
        "context_length": ctx,
        "desired_compute_unit": desired_compute_unit or cell.device,
        "tokens_per_second": decode,
        "prefill_tokens_per_second": prefill,
        "time_to_first_token_range_milliseconds": ttft_range,
        "measured": {
            "ttft_ms": ttft,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": _agg_median(cell, "gen_tokens"),
            "runs": len(cell.runs),
        },
        "plugin": cell.plugin,
        "tool_versions": {k: v for k, v in cell.versions.items() if v},
    }


# --------------------------------------------------------------------------- VQA latency breakdown


def stage_breakdown(runs: list[RunRecord]) -> dict[str, Any]:
    """Median (and spread) of each stage of a VQA request over repeated runs."""

    def s(attr: str) -> dict[str, Any]:
        return summarize(getattr(r, attr) for r in runs)

    return {
        "n_runs": len(runs),
        "vision_encoder_ms": s("media_ms"),
        "prefill_ms": s("prompt_ms"),
        "prefill_tokens": s("prompt_tokens"),
        "prefill_tps": s("prefill_tps"),
        "decode_ms": s("decode_ms"),
        "decode_tokens": s("gen_tokens"),
        "decode_tps": s("decode_tps"),
        "ttft_ms": s("ttft_ms"),
        "e2e_ms": summarize(r.e2e_ms for r in runs),
    }


# --------------------------------------------------------------------------- --accuracy stdout


_SEP = re.compile(r"^\[sep \] prompt (\d+)/(\d+)")


def parse_accuracy_stdout(text: str) -> list[dict[str, Any]]:
    """Split ``geniex-bench --accuracy`` stdout into one entry per prompt segment.

    Each segment holds the generated text (``[gen ]`` lines), the SDK stop reason
    and any self-abort warnings.
    """
    segments: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    summary: list[str] = []

    def new(idx: int) -> dict[str, Any]:
        seg = {"index": idx, "lines": [], "stop_reason": None, "warnings": []}
        segments.append(seg)
        return seg

    for raw in text.splitlines():
        line = raw.rstrip("\r")
        m = _SEP.match(line)
        if m:
            cur = new(int(m.group(1)) - 1)
            continue
        if line.startswith("[gen ]"):
            cur = cur or new(0)
            cur["lines"].append(line[7:] if len(line) > 6 else "")
        elif line.startswith("[info] stop_reason="):
            cur = cur or new(0)
            cur["stop_reason"] = line.split("=", 1)[1].strip()
        elif line.startswith("[warn]"):
            cur = cur or new(0)
            cur["warnings"].append(line[6:].strip())
        elif line.startswith("[ok  ]"):
            summary.append(line)
    for seg in segments:
        seg["text"] = "\n".join(seg.pop("lines")).strip()
    if summary and segments:
        segments[-1]["summary_line"] = summary[-1]
    return segments
