"""hexprof command line.

    hexprof doctor
    hexprof gemma4   aihub                            (AI Hub hosted S25, no phone)
    hexprof gemma4   setup | infer | profile          (GenieX on an adb-connected phone)
    hexprof siglip2  fetch | infer | profile          (AI Hub hosted Galaxy S25)
    hexprof hub      profile | metrics                (any model / existing job)
    hexprof hw       fields | classify                (optrace / QHAS)
    hexprof ceilings compute | bandwidth | kv-check   (NPU ceilings via AI Hub)
    hexprof pipeline gemma4 quantize | export         (fine-tuned Gemma 4 -> S25)

Global option (any position): --download-dir DIR puts every download under DIR
(AI Hub Models assets, Hugging Face files, llama.cpp models, geniex-bench).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from hexprof import __version__
from hexprof.devices import GALAXY_S25
from hexprof.io_utils import run_dir

DEFAULT_HUB_DEVICE = GALAXY_S25.hub_name


def apply_download_dir(path: str | None) -> Path | None:
    """Point every downloader at ``path`` (must run before those packages are imported)."""
    path = path or os.environ.get("HEXPROF_DOWNLOAD_DIR")
    if not path:
        return None
    d = Path(path).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    os.environ["HEXPROF_DOWNLOAD_DIR"] = str(d)
    os.environ["QAIHM_STORE_ROOT"] = str(d)  # AI Hub Models cache -> <d>/.qaihm (Gemma 4 checkpoint)
    os.environ["HF_HUB_CACHE"] = str(d / "huggingface")  # processors, tokenizers, configs
    os.environ["LLAMA_CACHE"] = str(d / "llama.cpp")  # llama-mtmd-cli -hf downloads
    os.environ["HEXPROF_CACHE"] = str(d / "hexprof")  # geniex-bench packages
    return d


def _pop_download_dir(argv: list[str]) -> tuple[str | None, list[str]]:
    out, value, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--download-dir" and i + 1 < len(argv):
            value, i = argv[i + 1], i + 2
            continue
        if a.startswith("--download-dir="):
            value = a.split("=", 1)[1]
        else:
            out.append(a)
        i += 1
    return value, out


def _assets_dir() -> Path:
    d = os.environ.get("HEXPROF_DOWNLOAD_DIR")
    return Path(d) if d else Path("assets")


# --------------------------------------------------------------------------- helpers


def _phone(args: argparse.Namespace):  # noqa: ANN202
    from hexprof.geniex.adb import Adb
    from hexprof.geniex.bench import GenieXPhone

    return GenieXPhone(Adb(serial=args.serial, dry_run=args.dry_run, verbose=args.verbose))


def _read_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompts_file:
        text = Path(args.prompts_file).read_text(encoding="utf-8")
        # one prompt per paragraph (blank-line separated) so prompts can span lines
        prompts += [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if not prompts:
        raise SystemExit("give at least one --prompt or --prompts-file")
    return prompts


def _expand_images(items: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    for item in items or []:
        p = Path(item)
        if p.is_dir():
            out += sorted(str(f) for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
        else:
            out.append(item)
    return out


def _add_phone_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--serial", help="adb serial of the phone (if several are connected)")
    p.add_argument("--dry-run", action="store_true", help="print the adb/geniex-bench commands without running them")
    p.add_argument("-v", "--verbose", action="store_true", help="echo every adb command")


def _add_gemma_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--plugin", choices=["llama_cpp", "qairt"], default="llama_cpp",
                   help="GenieX backend: llama_cpp = AI Hub's published Gemma 4 path (Q4_0 GGUF); "
                        "qairt = compiled QNN bundle (w4a16)")
    p.add_argument("--model", help="model-manager id, local .gguf, or local GenieX QAIRT bundle dir "
                                   "(default: AI Hub's id for the plugin)")
    p.add_argument("--mmproj", help="vision projector .gguf for llama_cpp (default: from the model manager)")


# --------------------------------------------------------------------------- doctor


def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True

    def line(name: str, good: bool, detail: str, optional: bool = False) -> None:
        nonlocal ok
        ok &= good or optional
        print(f"[{'ok' if good else ('--' if optional else '!!')}] {name}: {detail}")

    for mod, why, optional in [("qai_hub", "AI Hub jobs", False), ("onnx", "ceiling graphs", False),
                               ("ijson", "streaming large QHAS / optrace files (hw report)", True),
                               ("qai_hub_models_cli", "fetch AI Hub assets", True),
                               ("transformers", "SigLIP 2 zero-shot labels / Gemma 4 dims", True),
                               ("matplotlib", "plots", True)]:
        found = importlib.util.find_spec(mod) is not None
        line(mod, found, ("installed" if found else "not installed (optional)" if optional else "missing") + f" ({why})",
             optional)
    cfg = Path(os.environ.get("QAIHUB_CLIENT_INI", Path.home() / ".qai_hub" / "client.ini"))
    line("AI Hub token", cfg.exists(), str(cfg) if cfg.exists() else "run `qai-hub configure --api_token <TOKEN>`")
    adb = shutil.which("adb")
    line("adb", adb is not None, adb or "install Android platform-tools (only needed for Gemma 4 / GenieX)", optional=True)
    if adb:
        phone = _phone(args)
        try:
            devs = phone.adb.devices()
            line("phone", bool(devs), ", ".join(devs) or "no device; enable USB debugging and accept the prompt")
            if devs:
                info = phone.check_device()
                line("SoC", info["soc_model"] == GALAXY_S25.soc_model, f"{info['model']} / {info['soc_model']} / Android {info['android']}")
                v = phone.installed_version()
                line("geniex-bench on phone", v is not None, v or "run `hexprof gemma4 setup`")
        except Exception as e:  # noqa: BLE001
            line("phone", False, str(e))
    return 0 if ok else 1


# --------------------------------------------------------------------------- gemma4


def cmd_gemma4_setup(args: argparse.Namespace) -> int:
    from hexprof.models import gemma4

    res = gemma4.setup(_phone(args), args.geniex_version, args.plugin, args.model, args.mmproj,
                       pull=not args.no_pull, vlm=not args.text_only, refresh=args.refresh)
    for k, v in res.items():
        print(f"{k}: {v}")
    return 0


def cmd_gemma4_infer(args: argparse.Namespace) -> int:
    from hexprof.models import gemma4

    cfg = gemma4.InferConfig(
        plugin=args.plugin, compute=args.compute, model=args.model, mmproj=args.mmproj,
        max_new_tokens=args.max_new_tokens, context_length=args.context_length, repeats=args.repeats,
        warmup=args.warmup, system_prompt=args.system_prompt, think=args.think, temperature=args.temperature,
        seed=args.seed, power_mode=args.power_mode, max_gen_time_s=args.max_gen_time_s,
        images_together=args.images_together,
    )
    gemma4.infer(_phone(args), cfg, _expand_images(args.image), _read_prompts(args), run_dir(args.out, "gemma4-infer"))
    return 0


def cmd_gemma4_profile(args: argparse.Namespace) -> int:
    from hexprof.models import gemma4

    gemma4.profile_aihub(_phone(args), run_dir(args.out, "gemma4-profile"), plugin=args.plugin, model=args.model,
                         computes=args.compute, contexts=args.contexts, repetitions=args.repetitions,
                         n_gen=args.n_gen, power_mode=args.power_mode, qairt_lib=args.qairt_lib)
    return 0


def cmd_gemma4_aihub(args: argparse.Namespace) -> int:
    from hexprof.models import gemma4_aihub

    export_dir = Path(args.export_dir) if args.export_dir else gemma4_aihub.default_export_dir()
    if args.export_only:
        gemma4_aihub.ensure_export(export_dir, args.device, args.checkpoint, hw=not args.no_hw, force=args.force_export)
        return 0
    if args.force_export:
        gemma4_aihub.ensure_export(export_dir, args.device, args.checkpoint, hw=not args.no_hw, force=True)
    images = _expand_images(args.image)
    if not images:
        raise SystemExit("give at least one --image (file or folder)")
    gemma4_aihub.infer(images, _read_prompts(args), run_dir(args.out, "gemma4-aihub"), device_name=args.device,
                       export_dir=export_dir, checkpoint=args.checkpoint, hw=not args.no_hw,
                       max_new_tokens=args.max_new_tokens, local=args.local_answer, llama_cli=args.llama_cli,
                       gguf=args.gguf, mmproj=args.mmproj, hf_repo=args.hf_repo)
    return 0


def cmd_gemma4_efficiency(args: argparse.Namespace) -> int:
    from hexprof.models import gemma4_aihub, gemma4_efficiency

    gemma4_efficiency.efficiency(args.export_dir or gemma4_aihub.default_export_dir(), config=args.config,
                                 compute_json=args.compute_json, bandwidth_json=args.bandwidth_json,
                                 peak_tops=args.peak_tops, peak_gbps=args.peak_gbps,
                                 measure_binaries=args.measure_binaries, qhas_max_mb=args.qhas_max_mb)
    return 0


# --------------------------------------------------------------------------- siglip2


def cmd_siglip2_fetch(args: argparse.Namespace) -> int:
    from hexprof.models import siglip2

    path = siglip2.fetch(args.out or _assets_dir(), args.precision, args.runtime)
    print(f"SigLIP 2 bundle: {path}")
    return 0


def _bundle(args: argparse.Namespace) -> Path:
    from hexprof.models import siglip2

    b = Path(args.bundle) if args.bundle else _assets_dir() / f"siglip2-qnn_dlc-{args.precision}"
    if not (b / "metadata.json").exists():
        print(f"{b} has no SigLIP 2 bundle yet; fetching the AI Hub Models asset ({args.precision}, qnn_dlc)...")
        b = siglip2.fetch(b.parent if b.name.startswith("siglip2-") else b, args.precision, "qnn_dlc")
    return b


def cmd_siglip2_infer(args: argparse.Namespace) -> int:
    from hexprof.models import siglip2

    images = _expand_images(args.image)
    if not images:
        raise SystemExit("give at least one --image (file or folder)")
    siglip2.infer(images, _bundle(args), run_dir(args.out, "siglip2-infer"), labels=args.labels,
                  device_name=args.device, qairt_version=args.qairt_version, options=args.options, hw=not args.no_hw,
                  logit_scale=args.logit_scale, logit_bias=args.logit_bias)
    return 0


def cmd_siglip2_profile(args: argparse.Namespace) -> int:
    from hexprof.models import siglip2

    siglip2.profile(_bundle(args), run_dir(args.out, "siglip2-profile"), components=args.components,
                    device_name=args.device, qairt_version=args.qairt_version, options=args.options, hw=not args.no_hw)
    return 0


# --------------------------------------------------------------------------- hub / hw


def cmd_hub_profile(args: argparse.Namespace) -> int:
    from hexprof.aihub import client as hc
    from hexprof.aihub.metrics import aihub_metrics, metrics_markdown

    out = run_dir(args.out, "hub-profile")
    model: Any = args.model if Path(args.model).exists() else hc.get_model(args.model)
    job = hc.submit_inference(model, hc.device(args.device), None, profile=True, name="hexprof profile",
                              options=args.options)
    print(f"profile job {job.job_id}: {job.url}")
    prof, _ = hc.save_profile(job, out, job.job_id)
    print(metrics_markdown(f"{args.model} on {args.device}", aihub_metrics(prof)))
    if args.hw:
        print(hc.download_hw_artifacts(job, out / "hw"))
    return 0


def cmd_hub_metrics(args: argparse.Namespace) -> int:
    from hexprof.aihub import client as hc
    from hexprof.aihub.metrics import aihub_metrics, metrics_markdown, op_type_breakdown, per_layer
    from hexprof.io_utils import markdown_table, write_json

    out = run_dir(args.out, f"job-{args.job_id}")
    job = hc.get_job(args.job_id)
    prof, _ = hc.save_profile(job, out, args.job_id)
    m = aihub_metrics(prof)
    write_json(out / "metrics.json", {"job": hc.job_info(job), "metrics": m, "per_layer": per_layer(prof)})
    print(metrics_markdown(f"{args.job_id} ({hc.job_info(job)['device']})", m))
    ops = op_type_breakdown(prof)[:12]
    if ops:
        print(markdown_table(["type", "unit", "count", "time us", "share"],
                             [[o["type"], o["compute_unit"], o["count"], o["time_us"], o["share"]] for o in ops], 3))
    if args.hw:
        print(hc.download_hw_artifacts(job, out / "hw"))
    return 0


def cmd_hw_fields(args: argparse.Namespace) -> int:
    import json

    from hexprof.aihub import hwtrace

    for f in args.files:
        print(json.dumps(hwtrace.describe(f), indent=2)[: args.max_chars])
    return 0


def _parse_map(items: Sequence[str] | None) -> dict[str, str]:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip()
    return out


def cmd_hw_classify(args: argparse.Namespace) -> int:
    from hexprof.aihub import client as hc
    from hexprof.aihub import hwtrace
    from hexprof.aihub.metrics import per_layer
    from hexprof.ceilings.experiments import load_ceilings
    from hexprof.io_utils import markdown_table, read_json, write_json

    out = run_dir(args.out, "hw-classify")
    files = list(args.trace or [])
    layers: list[dict] = []
    if args.job:
        job = hc.get_job(args.job)
        prof, _ = hc.save_profile(job, out, args.job)
        layers = per_layer(prof)
        arts = hc.download_hw_artifacts(job, out)
        files += [p for k in ("qhas", "optrace") for p in arts.get(k, []) if str(p).endswith(".json")]
    elif args.profile:
        layers = per_layer(read_json(args.profile))
    if not files:
        raise SystemExit("no optrace / QHAS JSON found (AI Hub only attaches them to NPU profile jobs)")
    ceil = load_ceilings(args.compute_json, args.bandwidth_json)
    bw = args.bandwidth_gbps or ceil.get("B_gbps")
    for f in files:
        rs = hwtrace.best_record_set(f)
        if rs is None:
            print(f"{f}: no per-op table found; inspect with `hexprof hw fields {f}`")
            continue
        rows = hwtrace.normalize(rs, _parse_map(args.field_map))
        hwtrace.merge_aihub_timing(rows, layers)
        hwtrace.classify(rows, bw, args.util_threshold, args.bw_fraction)
        summ = hwtrace.class_summary(rows)
        write_json(out / (Path(f).stem + ".classified.json"), {"source": f, "record_path": rs.path,
                                                               "fields": hwtrace.match_fields(rs.keys, _parse_map(args.field_map)),
                                                               "bandwidth_gbps": bw, "summary": summ, "ops": rows})
        print(f"\n{f} ({rs.path}, {len(rows)} ops, B = {bw} GB/s)")
        print(markdown_table(["class", "ops", "time us", "share"],
                             [[k, v["ops"], v["time_us"], v["time_share"]] for k, v in summ.items()], 3))
        top = sorted(rows, key=lambda r: -(r.get("time_us") or 0))[:15]
        print(markdown_table(["op", "type", "time us", "HMX", "HVX", "GB/s", "class"],
                             [[r["name"][:48], r.get("op_type"), r.get("time_us"), r.get("hmx_util"), r.get("hvx_util"),
                               r.get("achieved_gbps"), r["bound"]] for r in top], 2))
    return 0


def cmd_hw_report(args: argparse.Namespace) -> int:
    from hexprof.hwreport import report

    paths = args.paths
    if not paths:
        from hexprof.models import gemma4_aihub

        paths = [str(gemma4_aihub.default_export_dir())]
    out = args.out or str(Path(paths[0] if Path(paths[0]).is_dir() else Path(paths[0]).parent) / "hw_report")
    report.build(paths, out, top=args.top, bins=args.bins, title=args.title, compute_json=args.compute_json,
                 bandwidth_json=args.bandwidth_json, theory_gbps=args.theoretical_gbps,
                 precisions=_parse_map(args.precision), gemma_config=args.gemma_config, onnx=_parse_map(args.onnx))
    return 0


def cmd_hw_fetch_models(args: argparse.Namespace) -> int:
    from hexprof.hwreport import models

    paths = args.paths
    if not paths:
        from hexprof.models import gemma4_aihub

        paths = [str(gemma4_aihub.default_export_dir())]
    models.fetch(paths, onnx=args.onnx, max_onnx_gb=args.max_onnx_gb, refresh=args.refresh)
    return 0


# --------------------------------------------------------------------------- studies

STUDY_FIXED = {  # fixed setting -> (targets it applies to, the axis it would conflict with)
    "context_length": ({"gemma4-text"}, "context-length"), "prompt_length": ({"gemma4-text"}, "prompt-length"),
    "components": ({"gemma4-text"}, None), "checkpoint": ({"gemma4-text"}, None),
    "target_runtime": ({"gemma4-text"}, None), "precision": ({"gemma4-vision", "siglip2-image", "siglip2-text"}, "precision"),
    "qairt": ({"gemma4-vision", "siglip2-image", "siglip2-text"}, "qairt"),
    "batch": ({"siglip2-image", "siglip2-text"}, "batch"), "image_size": ({"gemma4-vision"}, "image-size"),
    "calibration_samples": ({"gemma4-vision", "siglip2-image", "siglip2-text"}, None),
}


def _study_fixed(args: argparse.Namespace) -> dict:
    fixed = {}
    for key, (targets, axis) in STUDY_FIXED.items():
        val = getattr(args, key, None)
        if val is None:
            continue
        if args.target not in targets:
            raise SystemExit(f"--{key.replace('_', '-')} does not apply to {args.target}")
        if axis == args.axis:
            raise SystemExit(f"--{key.replace('_', '-')} is the swept axis; give its values with --values")
        fixed[key] = val
    return fixed


def cmd_study_run(args: argparse.Namespace) -> int:
    from hexprof.study import run as S

    S.run(args.target, args.axis, args.values or [], _study_fixed(args), args.out, args.device, args.force,
          compare=not args.no_compare, compute_json=args.compute_json, bandwidth_json=args.bandwidth_json)
    return 0


def cmd_study_verify(args: argparse.Namespace) -> int:
    from hexprof.study import run as S

    st = S.reverify(args.study_dir)
    bad = [v for v, r in st["variants"].items() if r.get("status") == "profiled" and not r.get("verified")]
    return 1 if bad else 0


def cmd_study_compare(args: argparse.Namespace) -> int:
    from hexprof.study import compare as C
    from hexprof.study import run as S

    dc, db = S.default_ceilings()
    C.compare(args.study_dir, args.compute_json or dc, args.bandwidth_json or db, rebuild=args.rebuild)
    return 0


def cmd_study_list(args: argparse.Namespace) -> int:
    from hexprof.study import run as S

    print(S.describe())
    return 0


def cmd_hub_frameworks(args: argparse.Namespace) -> int:
    from hexprof.aihub import client as hc

    for fw in hc.hub().get_frameworks():
        print(f"{fw.name:<8} {fw.api_version:<8} {fw.full_version:<24} {', '.join(fw.api_tags)}")
    return 0


# --------------------------------------------------------------------------- ceilings


def _sweep(args: argparse.Namespace, prefix: str):  # noqa: ANN202
    from hexprof.ceilings.runner import Sweep

    return Sweep(run_dir(args.out, prefix), args.device, runtime=args.runtime, compile_options=args.compile_options,
                 profile_options=args.profile_options, quantize_options=args.quantize_options,
                 qairt_version=args.qairt_version, quantize_io=not args.no_quantize_io, n_calib=args.calib_samples,
                 hw=args.hw, dry_run=args.dry_run)


def _add_sweep_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default=DEFAULT_HUB_DEVICE, help="AI Hub device name")
    p.add_argument("--runtime", default="qnn_context_binary", choices=["qnn_context_binary", "qnn_dlc"],
                   help="AI Hub target runtime (context binary = what GenieX QAIRT loads)")
    p.add_argument("--qairt-version", help="pin QAIRT on AI Hub (e.g. 2.50)")
    p.add_argument("--compile-options", default="")
    p.add_argument("--profile-options", default="", help="e.g. '--max_profiler_iterations 200'")
    p.add_argument("--quantize-options", default="")
    p.add_argument("--no-quantize-io", action="store_true", help="keep float graph I/O (default: --quantize_io)")
    p.add_argument("--calib-samples", type=int, default=4)
    p.add_argument("--hw", action="store_true", help="also download optrace / QHAS for every point")
    p.add_argument("--dry-run", action="store_true", help="build and save the graphs only")
    p.add_argument("--out", help="output dir; re-use it to resume a sweep")


def cmd_ceilings_compute(args: argparse.Namespace) -> int:
    from hexprof.ceilings.experiments import compute_sweep

    compute_sweep(_sweep(args, "ceil-compute"), args.precisions, args.size, args.chain, args.rows)
    return 0


def _dims(args: argparse.Namespace):  # noqa: ANN202
    from dataclasses import replace

    from hexprof.ceilings import graphs

    dims = graphs.load_gemma4_dims(args.hf_repo) if not args.no_hf_config else graphs.Gemma4Dims()
    if args.intermediate_size:
        dims = replace(dims, intermediate=args.intermediate_size, source=dims.source + "+override")
    return dims


def cmd_ceilings_bandwidth(args: argparse.Namespace) -> int:
    from hexprof.ceilings.experiments import bandwidth_sweep

    bandwidth_sweep(_sweep(args, "ceil-bandwidth"), args.precision, widths=args.widths, width_layers=args.width_layers,
                    gemma4_layers=args.gemma4_layers, gemma4_dims=_dims(args) if args.gemma4_layers else None,
                    context=args.context, include_kv=args.include_kv, onchip_mb=args.onchip_mb)
    return 0


def cmd_ceilings_kv(args: argparse.Namespace) -> int:
    from hexprof.ceilings.experiments import kv_cache_check

    kv_cache_check(run_dir(args.out, "ceil-kv"), args.device, args.precision, args.layers, args.context, _dims(args),
                   runtime=args.runtime, qairt_version=args.qairt_version, tolerance=args.tolerance)
    return 0


def _add_dims_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hf-repo", default="google/gemma-4-E4B-it", help="read decoder dims from this HF config")
    p.add_argument("--no-hf-config", action="store_true", help="use built-in dims without contacting Hugging Face")
    p.add_argument("--intermediate-size", type=int, help="override the MLP width")


# --------------------------------------------------------------------------- pipeline


def cmd_pipeline_quantize(args: argparse.Namespace) -> int:
    from hexprof.pipeline import gemma4_export

    gemma4_export.quantize(args.checkpoint, args.out, args.num_samples, args.context_length, args.sequence_length,
                           args.export_only, args.extra)
    return 0


def cmd_pipeline_export(args: argparse.Namespace) -> int:
    from hexprof.pipeline import gemma4_export

    gemma4_export.export(args.checkpoint, run_dir(args.out, "gemma4-export"), args.device, args.target_runtime,
                         args.precision, args.extra, hw=not args.no_hw)
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hexprof", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"hexprof {__version__}")
    ap.add_argument("--download-dir", help="put all downloads under this folder (also: env HEXPROF_DOWNLOAD_DIR); "
                                           "accepted anywhere on the command line")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="check AI Hub token, adb, phone and geniex-bench")
    _add_phone_args(p)
    p.set_defaults(func=cmd_doctor)

    # ---- gemma4
    g = sub.add_parser("gemma4", help="Gemma 4 E4B: on AI Hub (aihub) or through GenieX on a USB phone").add_subparsers(dest="sub", required=True)
    p = g.add_parser("aihub", help="your image + prompt on AI Hub's hosted S25 (no phone): vision encoder run on "
                                   "your image, per-stage timing from the compiled graphs, answer via local llama.cpp")
    p.add_argument("--image", action="append", help="image file or folder (repeatable)")
    p.add_argument("--prompt", action="append", help="question / instruction (repeatable)")
    p.add_argument("--prompts-file", help="text file, one prompt per blank-line-separated paragraph")
    p.add_argument("--device", default=DEFAULT_HUB_DEVICE, help="AI Hub device name")
    p.add_argument("--export-dir", help="where the one-time compile/profile results live "
                                        "(default: <download-dir>/gemma4-s25-export or runs/gemma4-s25-export)")
    p.add_argument("--export-only", action="store_true", help="only run the one-time compile + profile step")
    p.add_argument("--force-export", action="store_true", help="redo the compile + profile step")
    p.add_argument("--checkpoint", default="DEFAULT", help="DEFAULT = Qualcomm's published w4a16 checkpoint, "
                                                           "or a quantized checkpoint dir from `pipeline gemma4 quantize`")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="answer length cap; also M for the decode estimate when no local answer is generated")
    p.add_argument("--local-answer", choices=["auto", "on", "off"], default="auto",
                   help="generate the answer text with llama.cpp on this computer (auto = if llama-mtmd-cli is installed)")
    p.add_argument("--llama-cli", help="path to llama-mtmd-cli")
    p.add_argument("--gguf", default="google/gemma-4-E4B-it-qat-q4_0-gguf:Q4_0",
                   help="Hugging Face repo[:quant] or local .gguf for the local answer (default: what GenieX runs)")
    p.add_argument("--mmproj", help="vision projector .gguf (only if the GGUF repo has none)")
    p.add_argument("--hf-repo", default="google/gemma-4-E4B-it", help="processor/tokenizer used for token counts")
    p.add_argument("--no-hw", action="store_true", help="skip optrace / QHAS download")
    p.add_argument("--out", help="output dir (default runs/gemma4-aihub-<time>)")
    p.set_defaults(func=cmd_gemma4_aihub)

    p = g.add_parser("efficiency", help="per compiled part: operations, bytes, achieved TOPS and GB/s, "
                                        "QHAS utilization and inferences/s (after `gemma4 aihub --export-only`)")
    p.add_argument("--export-dir", help="default: <download-dir>/gemma4-s25-export or runs/gemma4-s25-export")
    p.add_argument("--config", help="Gemma 4 config.json (default: the one in Qualcomm's downloaded checkpoint)")
    p.add_argument("--compute-json", help="compute_ceiling.json from `hexprof ceilings compute` (gives %% of P)")
    p.add_argument("--bandwidth-json", help="bandwidth_ceiling.json from `hexprof ceilings bandwidth` (gives %% of B)")
    p.add_argument("--peak-tops", type=float, help="use this P instead of a ceilings file")
    p.add_argument("--peak-gbps", type=float, help="use this B instead of a ceilings file")
    p.add_argument("--measure-binaries", action="store_true",
                   help="download the compiled context binaries to cross-check byte counts")
    p.add_argument("--qhas-max-mb", type=float, default=1024,
                   help="parse QHAS files up to this size fully; larger ones are scanned (vision encoder: ~3.5 GB)")
    p.set_defaults(func=cmd_gemma4_efficiency)

    p = g.add_parser("setup", help="install geniex-bench on the phone and pull the model there")
    _add_phone_args(p)
    _add_gemma_model_args(p)
    p.add_argument("--geniex-version", help="pin a geniex-bench release, e.g. 0.10.0 (default: latest stable)")
    p.add_argument("--no-pull", action="store_true", help="install only; the model downloads on first use")
    p.add_argument("--text-only", action="store_true", help="do not pull the vision projector")
    p.add_argument("--refresh", action="store_true", help="re-download and re-push geniex-bench")
    p.set_defaults(func=cmd_gemma4_setup)

    p = g.add_parser("infer", help="answer prompts about images; prints answers + per-stage latency")
    _add_phone_args(p)
    _add_gemma_model_args(p)
    p.add_argument("--image", action="append", help="image file or folder (repeatable)")
    p.add_argument("--prompt", action="append", help="question / instruction (repeatable)")
    p.add_argument("--prompts-file", help="text file, one prompt per blank-line-separated paragraph")
    p.add_argument("--images-together", action="store_true", help="send all images in one turn instead of one run per image")
    p.add_argument("--compute", default="npu", choices=["npu", "cpu", "gpu", "hybrid", "auto"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=3, help="measured runs per (image, prompt)")
    p.add_argument("--warmup", type=int, default=1, help="unmeasured runs per (image, prompt)")
    p.add_argument("--system-prompt")
    p.add_argument("--think", action="store_true", help="enable Gemma 4 thinking (default off)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--power-mode", help="HTP power mode, e.g. burst (default) or sustained_high_performance")
    p.add_argument("--max-gen-time-s", type=int, default=600)
    p.add_argument("--out", help="output dir (default runs/gemma4-infer-<time>)")
    p.set_defaults(func=cmd_gemma4_infer)

    p = g.add_parser("profile", help="AI Hub's GenieX llm_metrics sweep (TPS, prefill TPS, TTFT range) vs published")
    _add_phone_args(p)
    _add_gemma_model_args(p)
    p.add_argument("--compute", nargs="+", default=["npu"], choices=["npu", "cpu", "gpu", "hybrid"])
    p.add_argument("--contexts", nargs="+", type=int, default=[512, 4096])
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--n-gen", type=int, default=128)
    p.add_argument("--power-mode")
    p.add_argument("--qairt-lib", help="device path of another QAIRT runtime (qairt plugin)")
    p.add_argument("--out")
    p.set_defaults(func=cmd_gemma4_profile)

    # ---- siglip2
    s = sub.add_parser("siglip2", help="SigLIP 2 image encoder on an AI Hub hosted S25").add_subparsers(dest="sub", required=True)
    p = s.add_parser("fetch", help="download AI Hub Models' pre-built SigLIP 2 bundle")
    p.add_argument("--precision", default="w8a16", choices=["w8a16", "float"])
    p.add_argument("--runtime", default="qnn_dlc", choices=["qnn_dlc", "onnx", "tflite"])
    p.add_argument("--out", help="target folder (default: --download-dir, else ./assets)")
    p.set_defaults(func=cmd_siglip2_fetch)
    for name, fn, helptext in [("infer", cmd_siglip2_infer, "embed your images (and score labels) on the S25"),
                               ("profile", cmd_siglip2_profile, "AI Hub profile job, same as AI Hub's published numbers")]:
        p = s.add_parser(name, help=helptext)
        p.add_argument("--bundle", help="folder from `hexprof siglip2 fetch` (default: <download-dir or ./assets>/"
                                        "siglip2-qnn_dlc-<precision>; fetched automatically if missing)")
        p.add_argument("--precision", default="w8a16", choices=["w8a16", "float"], help="used when the bundle is fetched")
        p.add_argument("--device", default=DEFAULT_HUB_DEVICE)
        p.add_argument("--qairt-version", help="AI Hub QAIRT version (default: the one the asset was verified with; 'default' = AI Hub's)")
        p.add_argument("--options", default="", help="extra AI Hub inference/profile options")
        p.add_argument("--no-hw", action="store_true", help="skip optrace / QHAS download")
        p.add_argument("--out")
        if name == "infer":
            p.add_argument("--image", action="append", help="image file or folder (repeatable)")
            p.add_argument("--labels", nargs="+", help="text labels for zero-shot scores (runs the text encoder too)")
            p.add_argument("--logit-scale", type=float, help="skip the HF download of SigLIP's logit scale")
            p.add_argument("--logit-bias", type=float)
        else:
            p.add_argument("--components", nargs="+", default=["image_encoder"], choices=["image_encoder", "text_encoder"])
        p.set_defaults(func=fn)

    # ---- hub
    h = sub.add_parser("hub", help="generic AI Hub profiling").add_subparsers(dest="sub", required=True)
    p = h.add_parser("profile", help="profile any compiled model (.dlc/.bin/.tflite/.onnx or AI Hub model id)")
    p.add_argument("model")
    p.add_argument("--device", default=DEFAULT_HUB_DEVICE)
    p.add_argument("--options", default="")
    p.add_argument("--hw", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_hub_profile)
    p = h.add_parser("frameworks", help="QAIRT versions available on AI Hub (values for --qairt)")
    p.set_defaults(func=cmd_hub_frameworks)
    p = h.add_parser("metrics", help="AI Hub metrics of an existing profile job")
    p.add_argument("job_id")
    p.add_argument("--hw", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_hub_metrics)

    # ---- hw
    w = sub.add_parser("hw", help="optrace / QHAS hardware metrics").add_subparsers(dest="sub", required=True)
    p = w.add_parser("fields", help="list record tables and fields in optrace / QHAS JSON files")
    p.add_argument("files", nargs="+")
    p.add_argument("--max-chars", type=int, default=20000)
    p.set_defaults(func=cmd_hw_fields)
    p = w.add_parser("report", help="HTML report (charts + tables + CSV) from QHAS / optrace / AI Hub profile files")
    p.add_argument("paths", nargs="*", help="folders or files to scan (default: the Gemma 4 export dir)")
    p.add_argument("--out", help="output folder (default: <first path>/hw_report)")
    p.add_argument("--top", type=int, default=15, help="rows per ranked chart (rest folded into Other)")
    p.add_argument("--bins", type=int, default=400, help="time bins for the occupancy timelines")
    p.add_argument("--title", help="overview page title")
    p.add_argument("--compute-json", help="compute_ceiling.json from `hexprof ceilings compute` (P per precision)")
    p.add_argument("--bandwidth-json", help="bandwidth_ceiling.json from `hexprof ceilings bandwidth` (B)")
    p.add_argument("--theoretical-gbps", type=float, default=84.8,
                   help="theoretical DRAM bandwidth (default: S25, LPDDR5X-5300 x 64-bit)")
    p.add_argument("--precision", nargs="*", help="override a graph's precision, e.g. image_encoder=w8a16")
    p.add_argument("--gemma-config", help="Gemma 4 config.json (default: the one the export downloaded)")
    p.add_argument("--onnx", nargs="*", help="ONNX for per-op MACs, e.g. vision_encoder__gemma_4_e4b_it=/path/model.onnx "
                                             "(default: the ones `hw fetch-models --onnx` downloaded)")
    p.set_defaults(func=cmd_hw_report)
    p = w.add_parser("fetch-models", help="record each profiled graph's compiled size, parameters and source ONNX "
                                          "from AI Hub (for the report's weight bytes and MAC counts)")
    p.add_argument("paths", nargs="*", help="result folders (default: the Gemma 4 export dir)")
    p.add_argument("--onnx", choices=["none", "vision", "all"], default="none",
                   help="also download source ONNX models: vision = only the vision encoder; all = every graph "
                        "(the Gemma 4 text parts are several GB each)")
    p.add_argument("--max-onnx-gb", type=float, default=20.0, help="skip source models larger than this")
    p.add_argument("--refresh", action="store_true", help="ignore cached models/*.model.json")
    p.set_defaults(func=cmd_hw_fetch_models)
    p = w.add_parser("classify", help="compute- / memory-bound / overhead per operator")
    p.add_argument("--job", help="AI Hub profile job id (downloads profile + artifacts)")
    p.add_argument("--trace", action="append", help="optrace / QHAS JSON file(s)")
    p.add_argument("--profile", help="AI Hub profile JSON for per-layer timing")
    p.add_argument("--compute-json", help="compute_ceiling.json from `hexprof ceilings compute`")
    p.add_argument("--bandwidth-json", help="bandwidth_ceiling.json from `hexprof ceilings bandwidth`")
    p.add_argument("--bandwidth-gbps", type=float, help="B directly (overrides --bandwidth-json)")
    p.add_argument("--util-threshold", type=float, default=0.6)
    p.add_argument("--bw-fraction", type=float, default=0.7)
    p.add_argument("--field-map", nargs="*", help="override field mapping, e.g. hmx_util=hmx.utilization_pct")
    p.add_argument("--out")
    p.set_defaults(func=cmd_hw_classify)

    # ---- studies
    st = sub.add_parser("study", help="sweep a deployment setting on AI Hub's S25, verify each value ran, compare"
                        ).add_subparsers(dest="sub", required=True)
    p = st.add_parser("list", help="targets, axes and default sweep values")
    p.set_defaults(func=cmd_study_list)
    p = st.add_parser("run", help="build, compile and profile every value on the device, verify, compare")
    p.add_argument("target", choices=["gemma4-text", "gemma4-vision", "siglip2-image", "siglip2-text"])
    p.add_argument("--axis", required=True, help="setting to sweep (see `hexprof study list`)")
    p.add_argument("--values", nargs="+", help="values to sweep (default: the list in `hexprof study list`)")
    p.add_argument("--out", help="study folder (default: $HEXPROF_DOWNLOAD_DIR/studies/<target>__<axis>); "
                                 "re-running resumes")
    p.add_argument("--device", default=DEFAULT_HUB_DEVICE)
    p.add_argument("--force", action="store_true", help="rebuild values that were already profiled")
    p.add_argument("--no-compare", action="store_true", help="skip the comparison page")
    p.add_argument("--compute-json", help="ceilings for the reports (default: runs/ceil-compute/compute_ceiling.json)")
    p.add_argument("--bandwidth-json", help="(default: runs/ceil-bandwidth/bandwidth_ceiling.json)")
    g = p.add_argument_group("fixed settings, gemma4-text")
    g.add_argument("--context-length", type=int, help="context length when not swept (Qualcomm: 4096)")
    g.add_argument("--prompt-length", type=int, help="prompt chunk when not swept (Qualcomm: 128)")
    g.add_argument("--components", nargs="+", help="export these parts (default: the 4 text parts)")
    g.add_argument("--checkpoint", help="DEFAULT = Qualcomm's published w4a16 checkpoint")
    g.add_argument("--target-runtime", help="Qualcomm export runtime (default geniex_qairt)")
    g = p.add_argument_group("fixed settings, gemma4-vision / siglip2-*")
    g.add_argument("--precision", help="float, w8a16, w8a8, w4a16 (default: Qualcomm's: float vision, w8a16 SigLIP 2)")
    g.add_argument("--qairt", help="QAIRT version on AI Hub: default, latest or 2.xx (see `hexprof hub frameworks`)")
    g.add_argument("--batch", type=int, help="SigLIP 2 batch size when not swept")
    g.add_argument("--image-size", type=int, help="vision encoder input size when not swept (Qualcomm: 448)")
    g.add_argument("--calibration-samples", type=int, help="quantize-job calibration samples (default: Qualcomm's)")
    p.set_defaults(func=cmd_study_run)
    p = st.add_parser("verify", help="re-check every profiled value of a study against AI Hub")
    p.add_argument("study_dir")
    p.set_defaults(func=cmd_study_verify)
    p = st.add_parser("compare", help="(re)build the comparison page of a study")
    p.add_argument("study_dir")
    p.add_argument("--compute-json")
    p.add_argument("--bandwidth-json")
    p.add_argument("--rebuild", action="store_true", help="rebuild every variant's hardware report")
    p.set_defaults(func=cmd_study_compare)

    # ---- ceilings
    c = sub.add_parser("ceilings", help="measure NPU compute / bandwidth ceilings on AI Hub").add_subparsers(dest="sub", required=True)
    p = c.add_parser("compute", help="chained square MatMuls per precision -> P, t0")
    _add_sweep_args(p)
    p.add_argument("--precisions", nargs="+", default=["w8a16", "w4a16", "fp16"],
                   help="weight/activation pairs of the deployment (fp16 = Gemma 4 vision encoder)")
    p.add_argument("--size", type=int, default=2048, help="K = N (multiple of 32)")
    p.add_argument("--rows", type=int, help="M (default = size)")
    p.add_argument("--chain", nargs="+", type=int, default=[1, 2, 4, 8, 16], help="L values")
    p.set_defaults(func=cmd_ceilings_compute)

    p = c.add_parser("bandwidth", help="single-token linear stacks (width) and Gemma 4 decoder subsets (depth) -> B, t0")
    _add_sweep_args(p)
    _add_dims_args(p)
    p.add_argument("--precision", default="w4a16", help="deployed weight precision (Gemma 4 blocks: w4a16)")
    p.add_argument("--widths", nargs="*", type=int, default=[3072, 4096, 5120, 6144])
    p.add_argument("--width-layers", type=int, default=16)
    p.add_argument("--gemma4-layers", nargs="*", type=int, default=[], help="depth sweep, e.g. 4 8 12")
    p.add_argument("--context", type=int, default=4096, help="static KV-cache length of the decoder subsets")
    p.add_argument("--include-kv", action="store_true", help="add KV-cache bytes to the x-axis")
    p.add_argument("--onchip-mb", type=float, default=16.0, help="VTCM + system cache, for the off-chip check")
    p.set_defaults(func=cmd_ceilings_bandwidth)

    p = c.add_parser("kv-check", help="decode latency with empty vs full KV cache")
    _add_dims_args(p)
    p.add_argument("--device", default=DEFAULT_HUB_DEVICE)
    p.add_argument("--runtime", default="qnn_context_binary", choices=["qnn_context_binary", "qnn_dlc"])
    p.add_argument("--qairt-version")
    p.add_argument("--precision", default="w4a16")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--tolerance", type=float, default=0.03)
    p.add_argument("--out")
    p.set_defaults(func=cmd_ceilings_kv)

    # ---- pipeline
    pl = sub.add_parser("pipeline", help="fine-tuned model -> AI Hub -> phone").add_subparsers(dest="model", required=True)
    pg = pl.add_parser("gemma4", help="fine-tuned Gemma 4 E4B (Qualcomm's export recipe)").add_subparsers(dest="sub", required=True)
    p = pg.add_parser("quantize", help="ONNX export + AIMET w4a16 calibration (GPU)")
    p.add_argument("--checkpoint", required=True, help="fine-tuned HF checkpoint dir")
    p.add_argument("--out", required=True)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--context-length", type=int, help="shorter KV cache = less memory and attention per step")
    p.add_argument("--sequence-length", type=int, help="prompt-processor chunk (default 128)")
    p.add_argument("--export-only", action="store_true", help="stage A only (CPU)")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra args for the Qualcomm script (after --)")
    p.set_defaults(func=cmd_pipeline_quantize)
    p = pg.add_parser("export", help="compile + link + profile on AI Hub, build the GenieX bundle")
    p.add_argument("--checkpoint", required=True, help="quantized checkpoint from `pipeline gemma4 quantize`")
    p.add_argument("--device", default=DEFAULT_HUB_DEVICE)
    p.add_argument("--target-runtime", default="geniex_qairt", choices=["geniex_qairt", "genie", "qnn_context_binary"])
    p.add_argument("--precision", default="w4a16")
    p.add_argument("--no-hw", action="store_true")
    p.add_argument("--out")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra `qai-hub-models export` args (after --)")
    p.set_defaults(func=cmd_pipeline_export)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    download_dir, rest = _pop_download_dir(list(sys.argv[1:] if argv is None else argv))
    d = apply_download_dir(download_dir)
    if d is not None:
        print(f"Downloads go to {d}")
    args = build_parser().parse_args(rest)
    if hasattr(args, "extra") and args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted (AI Hub jobs keep running; re-run with the same --out to resume)", file=sys.stderr)
        return 130
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        if os.environ.get("HEXPROF_DEBUG"):
            raise
        print(f"error: {e}\n(set HEXPROF_DEBUG=1 for the full traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
