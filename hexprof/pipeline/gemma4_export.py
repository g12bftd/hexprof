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


def export_argv(options: set[str], checkpoint: str | Path, device_name: str, target_runtime: str | None,
                precision: str | None, output_dir: str | Path, extra: Sequence[str] = ()) -> list[str]:
    """Arguments for Qualcomm's export parser, using only the flags it offers.

    LLM recipes with a separate quantize script (Gemma 4) have no ``--precision``:
    the parser derives it from ``--checkpoint`` (w4a16 for ``DEFAULT``).
    """
    argv = ["--checkpoint", str(checkpoint), "--device", device_name, "--output-dir", str(output_dir)]
    if target_runtime:
        flag = next((f for f in ("--target-runtime", "--runtime") if f in options), None)
        if flag:
            argv += [flag, target_runtime]
    if precision and "--precision" in options:
        argv += ["--precision", precision]
    return argv + list(extra)


def run_official_export(checkpoint: str | Path, out_dir: str | Path, device_name: str = "Samsung Galaxy S25",
                        target_runtime: str = "geniex_qairt", precision: str = "w4a16",
                        extra: Sequence[str] = ()) -> Any:
    """``qai-hub-models export gemma_4_e4b_it`` in-process; returns its export result (with job handles).

    ``checkpoint="DEFAULT"`` uses Qualcomm's published w4a16 checkpoint (an 18.7 GB
    download into ``$QAIHM_STORE_ROOT/.qaihm``; no GPU needed).
    """
    _require_qaihm()
    from qai_hub_models.cli.dispatch import build_export_parser_for  # type: ignore[import-not-found]
    from qai_hub_models.utils.export.context import resolve_recipe_dir  # type: ignore[import-not-found]
    from qai_hub_models.utils.export.dispatch import select_pipeline  # type: ignore[import-not-found]

    out = Path(out_dir)
    source_dir = resolve_recipe_dir(MODEL_ID)
    parser = build_export_parser_for(source_dir)
    argv = export_argv(set(parser._option_string_actions), checkpoint, device_name, target_runtime, precision,
                       out / "export_assets", extra)
    args = parser.parse_args(argv)
    print(f"qai-hub-models export {MODEL_ID} " + " ".join(argv)
          + f"  (precision: {getattr(args, 'precision', None)}, runtime: {getattr(args, 'target_runtime', None)})")
    try:  # batch jobs (e.g. Slurm) have no stdin: answer Qualcomm's yes/no prompts with "yes"
        from qai_hub_models.utils.asset_loaders import always_answer_prompts  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - older qai_hub_models
        always_answer_prompts = None
    if always_answer_prompts is None:
        return select_pipeline(source_dir)(**vars(args))
    with always_answer_prompts(True):
        return select_pipeline(source_dir)(**vars(args))


def parse_export_args(checkpoint: str | Path, out_dir: str | Path, device_name: str = "Samsung Galaxy S25",
                      target_runtime: str = "geniex_qairt", precision: str = "w4a16",
                      extra: Sequence[str] = ()) -> dict[str, Any]:
    """Parse the export arguments with Qualcomm's own parser, without running anything.

    Unknown flags or invalid values raise here (argparse exits), so a study fails before
    submitting jobs; the returned values are exactly what the export will receive.
    """
    _require_qaihm()
    from qai_hub_models.cli.dispatch import build_export_parser_for  # type: ignore[import-not-found]
    from qai_hub_models.utils.export.context import resolve_recipe_dir  # type: ignore[import-not-found]

    parser = build_export_parser_for(resolve_recipe_dir(MODEL_ID))
    argv = export_argv(set(parser._option_string_actions), checkpoint, device_name, target_runtime, precision,
                       Path(out_dir) / "export_assets", extra)
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        raise ValueError(f"Qualcomm's export parser rejected: {' '.join(argv)}") from e
    return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, list, tuple, type(None)))}


def export(checkpoint: str | Path, out_dir: str | Path, device_name: str = "Samsung Galaxy S25",
           target_runtime: str = "geniex_qairt", precision: str = "w4a16", extra: Sequence[str] = (),
           hw: bool = True) -> dict[str, Any]:
    """Stage 2: compile + link + profile on AI Hub with the official recipe, then collect metrics."""
    result = run_official_export(checkpoint, out_dir, device_name, target_runtime, precision, extra)
    return collect(result, Path(out_dir), hw)


def collect(result: Any, out_dir: Path, hw: bool = True) -> dict[str, Any]:
    """AI Hub metrics (+ optrace/QHAS) for every profile job of an export result."""
    jobs = getattr(result, "profile_jobs", None) or {}
    summary: dict[str, Any] = {"bundle": str(getattr(result, "download_path", "") or ""), "parts": {}}
    rows = []
    md = []
    for key, job in jobs.items():
        component, graph = (key if isinstance(key, tuple) and len(key) == 2 else (str(key), str(key)))
        label = f"{component}/{graph}"
        stem = label.replace("/", "__")
        prof, _ = hc.save_profile(job, out_dir / "profiles", stem)
        m = aihub_metrics(prof)
        model = getattr(job, "model", None)
        rec: dict[str, Any] = {
            "component": component,
            "graph": graph,
            "job": hc.job_info(job),
            # what was profiled, so the same graph can be re-run with real inputs
            "model_id": getattr(model, "model_id", None),
            "options": getattr(job, "options", "") or "",
            "metrics": m,
        }
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
