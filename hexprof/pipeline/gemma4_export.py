"""Fine-tuned Gemma 4 E4B -> QNN context binaries -> GenieX bundle, profiled on the S25.

Starts from Qualcomm's own Gemma 4 export (``qai_hub_models.models.gemma_4_e4b_it``),
which already does the model-specific NPU adaptation: static input shapes
(128-token prompt processor + 1-token generator), explicit KV cache, 4-way split
(layers 0-23 | 24-32 | 33-41 | lm_head) and a separate FP16 vision encoder.

Stages
------
1. ``quantize``  Qualcomm's recipe: ONNX export (CPU) + AIMET calibration
   (SeqMSE + calibration on WikiText, needs a CUDA GPU) -> w4a16 checkpoint
   (int4 per-channel blocks, int8 lm_head and down_proj, int16 activations, int8 KV).
   An AI Hub quantize job only takes one weight dtype and one activation dtype per
   graph, so it cannot express this mixed recipe; Qualcomm's calibration is used
   for the language model.
2. ``export``    ``qai-hub-models export gemma_4_e4b_it`` in-process: AI Hub compile
   jobs per (part, graph) -> link jobs -> QNN context binaries -> profile jobs on the
   S25 -> GenieX QAIRT bundle. hexprof then collects AI Hub metrics, per-layer timing
   and optrace / QHAS for every profile job.
3. Run the bundle on the phone with ``hexprof gemma4 infer --plugin qairt --model <bundle>``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from hexprof.aihub import client as hc
from hexprof.aihub.metrics import aihub_metrics, metrics_markdown, per_layer
from hexprof.io_utils import markdown_table, write_json

MODEL_ID = "gemma_4_e4b_it"


def _require_qaihm() -> None:
    try:
        import qai_hub_models  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Qualcomm AI Hub Models is required: pip install 'hexprof[pipeline]' && "
            f"qai-hub-models install {MODEL_ID}"
        ) from e


def quantize(checkpoint: str | Path, out_dir: str | Path, num_samples: int = 20,
             context_length: int | None = None, sequence_length: int | None = None,
             export_only: bool = False, extra: Sequence[str] = ()) -> Path:
    """Stage 1: Qualcomm's two-stage Gemma 4 quantization on a (fine-tuned) HF checkpoint."""
    _require_qaihm()
    cmd = [sys.executable, "-m", f"qai_hub_models.models.{MODEL_ID}.quantize", "--checkpoint", str(checkpoint),
           "-o", str(out_dir)]
    if export_only:
        cmd.append("--export-only")
    else:
        cmd += ["--num-samples", str(num_samples)]
    if context_length:
        cmd += ["--context-length", str(context_length)]
    if sequence_length:
        cmd += ["--sequence-length", str(sequence_length)]
    cmd += list(extra)
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return Path(out_dir)


def export(checkpoint: str | Path, out_dir: str | Path, device_name: str = "Samsung Galaxy S25",
           target_runtime: str = "geniex_qairt", precision: str = "w4a16", extra: Sequence[str] = (),
           hw: bool = True) -> dict[str, Any]:
    """Stage 2: compile + link + profile on AI Hub with the official recipe, then collect metrics."""
    _require_qaihm()
    from qai_hub_models.cli.dispatch import build_export_parser_for  # type: ignore[import-not-found]
    from qai_hub_models.utils.export.context import resolve_recipe_dir  # type: ignore[import-not-found]
    from qai_hub_models.utils.export.dispatch import select_pipeline  # type: ignore[import-not-found]

    out = Path(out_dir)
    source_dir = resolve_recipe_dir(MODEL_ID)
    argv = ["--checkpoint", str(checkpoint), "--device", device_name, "--target-runtime", target_runtime,
            "--precision", precision, "--output-dir", str(out / "export_assets"), *extra]
    parser = build_export_parser_for(source_dir)
    args = parser.parse_args(argv)
    print(f"qai-hub-models export {MODEL_ID} " + " ".join(argv))
    result = select_pipeline(source_dir)(**vars(args))
    return collect(result, out, hw)


def collect(result: Any, out_dir: Path, hw: bool = True) -> dict[str, Any]:
    """AI Hub metrics (+ optrace/QHAS) for every profile job of an export result."""
    jobs = getattr(result, "profile_jobs", None) or {}
    summary: dict[str, Any] = {"bundle": str(getattr(result, "download_path", "") or ""), "parts": {}}
    rows = []
    md = []
    for key, job in jobs.items():
        label = "/".join(key) if isinstance(key, tuple) else str(key)
        stem = label.replace("/", "__")
        prof, _ = hc.save_profile(job, out_dir / "profiles", stem)
        m = aihub_metrics(prof)
        rec: dict[str, Any] = {"job": hc.job_info(job), "metrics": m}
        write_json(out_dir / "profiles" / f"{stem}.per_layer.json", per_layer(prof))
        if hw:
            rec["hw_artifacts"] = hc.download_hw_artifacts(job, out_dir / "hw" / stem)
        summary["parts"][label] = rec
        peak = m.get("estimated_peak_memory_range_mb") or {}
        rows.append([label, m["inference_time_milliseconds"], f"[{peak.get('min')}, {peak.get('max')}]",
                     m["layer_counts"]["npu"], m["layer_counts"]["total"], job.url])
        md.append(metrics_markdown(label, m))
    write_json(out_dir / "export_profiles.json", summary)
    table = markdown_table(["part / graph", "inference ms", "peak MB", "NPU layers", "layers", "job"], rows, 3)
    (out_dir / "export_profiles.md").write_text(table + "\n\n" + "\n\n".join(md) + "\n", encoding="utf-8")
    print(table)
    if summary["bundle"]:
        print(f"\nGenieX bundle: {summary['bundle']}\nRun it on the phone:\n"
              f"  hexprof gemma4 infer --plugin qairt --model {summary['bundle']} --image <img> --prompt '<question>'")
    return summary
