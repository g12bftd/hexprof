"""Single-graph sweeps (Gemma 4 vision encoder, SigLIP 2 image / text encoder) with Qualcomm's own code.

Each variant is built exactly as ``qai-hub-models export`` builds a single model, by calling
its stage functions: ``Model.from_pretrained(...)`` -> ``model.get_input_spec(...)`` ->
``upload_source`` -> (quantized precisions: ONNX compile + ``run_quantize``) ->
``run_compile`` to a QNN context binary with the model's own compile options; then an AI Hub
profile job on the device, the profile and the QHAS / optrace files.

Axes
    image-size  (gemma4-vision)  VisionEncoder.from_pretrained(image_size=V)
    batch       (siglip2-*)      get_input_spec(image_batch_size=V / text_batch_size=V)
    precision   (all)            float | w8a16 | w8a8 | w4a16 ... via an AI Hub quantize job
    mixed-precision (all)        Workbench Lite-MP: V% most sensitive layers kept at --override
                                 (int16 / fp16), the rest at the --precision base (w8a16 / w8a8)
    qairt       (all)            --qairt_version V on compile and profile (default, latest, 2.xx)

Verification compares what AI Hub reports for the compiled model and the jobs (input shapes,
quantize dtypes, the model the compile job consumed, QAIRT version, device) with the request,
and checks across the sweep that every value produced a different graph where it should.
"""

from __future__ import annotations

import importlib
import math
import re
from pathlib import Path
from typing import Any

from hexprof.io_utils import read_json, write_json
from hexprof.study import core

TARGETS = {
    "gemma4-vision": ("qai_hub_models.models.gemma_4_e4b_it", "VisionEncoder", "vision_encoder", "float"),
    "siglip2-image": ("qai_hub_models.models.siglip2.model", "SigLIP2ImageEncoder", "image_encoder", "w8a16"),
    "siglip2-text": ("qai_hub_models.models.siglip2.model", "SigLIP2TextEncoder", "text_encoder", "w8a16"),
}
BATCH_KW = {"siglip2-image": "image_batch_size", "siglip2-text": "text_batch_size"}


def settings(target: str, axis: str, value: str, fixed: dict[str, Any]) -> dict[str, Any]:
    _, _, component, default_prec = TARGETS[target]
    s: dict[str, Any] = {"component": component, "precision": fixed.get("precision") or default_prec,
                         "qairt": fixed.get("qairt"), "from_kwargs": {}, "spec_kwargs": {}, "litemp": None}
    if fixed.get("image_size") and target == "gemma4-vision":
        s["from_kwargs"]["image_size"] = int(fixed["image_size"])
    if fixed.get("batch") and target in BATCH_KW:
        s["spec_kwargs"][BATCH_KW[target]] = int(fixed["batch"])
    if axis == "image-size":
        s["from_kwargs"]["image_size"] = int(value)
    elif axis == "batch":
        s["spec_kwargs"][BATCH_KW[target]] = int(value)
    elif axis == "precision":
        s["precision"] = value
    elif axis == "qairt":
        s["qairt"] = value
    elif axis == "mixed-precision":
        base = fixed.get("precision") or "w8a16"
        if base not in core.QUANTIZED:
            raise ValueError(f"mixed precision needs a quantized base --precision (w8a16, w8a8, ...), not {base}")
        pct = int(value)
        override = fixed.get("override") or "fp16"
        s["precision"] = f"{base}_mixed_{override}" if pct else base
        s["litemp"] = {"percentage": pct, "override": override} if pct else None
    return s


def _staged(state: dict[str, Any], path: Path, key: str, submit: Any) -> Any:
    """Reuse a job recorded on an earlier (interrupted) run, else submit and record it."""
    from hexprof.aihub import client as hc

    if state.get(key):
        return hc.get_job(state[key]) if key.endswith("_job") else hc.get_model(state[key])
    obj = submit()
    state[key] = obj.job_id if key.endswith("_job") else obj.model_id
    write_json(path, state)
    return obj


def run_one(target: str, axis: str, value: str, vdir: Path, fixed: dict[str, Any], device: str) -> dict[str, Any]:
    import qai_hub as hub  # type: ignore[import-not-found]
    from qai_hub_models import Precision, TargetRuntime  # type: ignore[import-not-found]
    from qai_hub_models.utils.export.compile import run_compile  # type: ignore[import-not-found]
    from qai_hub_models.utils.export.quantize import run_quantize  # type: ignore[import-not-found]
    from qai_hub_models.utils.export.upload import upload_source  # type: ignore[import-not-found]

    from hexprof.aihub import client as hc
    from hexprof.aihub.metrics import aihub_metrics

    mod, cls_name, component, _ = TARGETS[target]
    s = settings(target, axis, value, fixed)
    vdir.mkdir(parents=True, exist_ok=True)
    state_path = vdir / "jobs.json"
    state: dict[str, Any] = read_json(state_path) if state_path.exists() else {}
    model = getattr(importlib.import_module(mod), cls_name).from_pretrained(**s["from_kwargs"])
    spec = model.get_input_spec(**s["spec_kwargs"])
    write_json(vdir / "variant.json", {"target": target, "axis": axis, "value": value, "settings": s,
                                       "local_input_spec": core.shapes_of(spec), "local_dtypes": core.dtypes_of(spec)})
    prec = Precision.parse(s["precision"])
    dev = hc.device(device)
    name = f"hexprof_{target}_{axis}_{value}".replace(".", "_")
    qflag = f"--qairt_version {s['qairt']}" if s["qairt"] else ""
    print(f"    {cls_name}.from_pretrained({s['from_kwargs']}), get_input_spec({s['spec_kwargs']}) -> "
          f"{core.shapes_of(spec)}; precision {prec}; {qflag or 'AI Hub default QAIRT'}", flush=True)

    source = _staged(state, state_path, "source_model_id", lambda: upload_source(model, spec))
    if prec != Precision.float:
        onnx_job = _staged(state, state_path, "onnx_job",
                           lambda: run_compile(model, name, dev, TargetRuntime.ONNX, prec, source, spec))
        onnx_model = hc.target_model(onnx_job)

        if s.get("litemp"):  # the percentage Qualcomm's quantize options read from the model
            pct = s["litemp"]["percentage"]
            model.get_hub_litemp_percentage = lambda _p, _pct=pct: _pct

        def quantize() -> Any:
            try:
                return run_quantize(prec, model, name, onnx_model, fixed.get("calibration_samples"), input_spec=spec)
            except Exception as e:  # noqa: BLE001 - no calibration dataset offline: the model's own sample inputs
                print(f"    Qualcomm calibration data unavailable ({type(e).__name__}: {e}); using sample inputs")
                state["calibration"] = "model sample inputs"
                return hub.submit_quantize_job(model=onnx_model, calibration_data=model.sample_inputs(spec),
                                               weights_dtype=prec.weights_type, activations_dtype=prec.activations_type,
                                               name=name, options=model.get_hub_quantize_options(prec))

        qjob = _staged(state, state_path, "quantize_job", quantize)
        source = hc.target_model(qjob)
        state["quantized_model_id"] = source.model_id
    cjob = _staged(state, state_path, "compile_job",
                   lambda: run_compile(model, name, dev, TargetRuntime.QNN_CONTEXT_BINARY, prec, source, spec,
                                       extra_options=qflag))
    compiled = hc.target_model(cjob)
    pjob = _staged(state, state_path, "profile_job",
                   lambda: hc.submit_inference(compiled, dev, None, profile=True, name=name, options=qflag))
    hc.wait_success(pjob, "profile")
    comp_dir = vdir / component
    prof, _ = hc.save_profile(pjob, comp_dir, component)
    arts = hc.download_hw_artifacts(pjob, comp_dir)
    m = aihub_metrics(prof)
    write_json(vdir / "profile.json", {"precision": s["precision"], "runtime": "qnn_context_binary", "device": device,
                                       "components": {component: {"job": hc.job_info(pjob), "metrics": m,
                                                                  "hw_artifacts": arts}}})
    write_json(state_path, state)
    print(f"    {component}: {m.get('inference_time_milliseconds')} ms on {device} ({pjob.url})", flush=True)
    return {"settings": s, "jobs": {k: v for k, v in state.items() if k.endswith(("_job", "_id"))},
            "inference_ms": m.get("inference_time_milliseconds")}


def _same_dims(a: list[int], b: list[int]) -> bool:
    """Equal up to layout (AI Hub may move channels last)."""
    return sorted(a) == sorted(b)


def verify(target: str, axis: str, value: str, vdir: Path, fixed: dict[str, Any], device: str) -> dict[str, Any]:
    from hexprof.aihub import client as hc

    var = read_json(vdir / "variant.json")
    jobs = read_json(vdir / "jobs.json")
    s = var["settings"]
    local = var["local_input_spec"]
    checks: list[dict[str, Any]] = []
    cjob = hc.get_job(jobs["compile_job"])
    pjob = hc.get_job(jobs["profile_job"])
    sent = core.shapes_of(getattr(cjob, "shapes", None))
    core.check(checks, "compile job input shapes", local, sent,
               set(sent) == set(local) and all(_same_dims(sent[k], local[k]) for k in local))
    compiled = hc.target_model(cjob)
    built = core.shapes_of(getattr(compiled, "input_spec", None))
    core.check(checks, "compiled model input shapes (AI Hub)", local, built,
               set(built) == set(local) and all(_same_dims(built[k], local[k]) for k in local))
    if getattr(pjob, "model", None) is not None:
        core.check(checks, "profiled model = compiled model", compiled.model_id, pjob.model.model_id,
                   pjob.model.model_id == compiled.model_id)
    core.device_check(checks, pjob, device, "profile device")
    if axis == "batch":
        b = int(value)
        core.check(checks, "batch dimension", b, {k: d[0] for k, d in built.items()},
                   bool(built) and all(d and d[0] == b for d in built.values()))
    if axis == "image-size":
        sizes = {k: math.prod(d) for k, d in local.items()}
        core.check(checks, "input elements traced for this image size", "differ from every other size",
                   sizes, _unique_among_siblings(vdir, var))
    if s["precision"] == "float":
        core.check(checks, "precision float: no quantize job", None, jobs.get("quantize_job"), not jobs.get("quantize_job"))
    else:
        q = hc.get_job(jobs["quantize_job"]) if jobs.get("quantize_job") else None
        mw = re.match(r"w(\d+)a(\d+)", s["precision"])
        want = (mw.group(1), mw.group(2)) if mw else ("?", "?")
        got_w = str(getattr(q, "weights_dtype", "")).upper()
        got_a = str(getattr(q, "activations_dtype", "")).upper()
        core.check(checks, "quantize job dtypes", f"INT{want[0]} weights, INT{want[1]} activations", f"{got_w} / {got_a}",
                   q is not None and got_w.endswith(f"INT{want[0]}") and got_a.endswith(f"INT{want[1]}"))
        opts = str(getattr(q, "options", "") or "")
        lm = s.get("litemp")
        if lm:
            flag = f"--lite_mp percentage={lm['percentage']};override_qtype={lm['override']}"
            core.check(checks, "mixed precision on the quantize job", flag, opts, flag in opts)
        elif axis == "mixed-precision":
            core.check(checks, "no mixed precision on the quantize job", "no --lite_mp", opts, "--lite_mp" not in opts)
        cm = getattr(cjob, "model", None)
        core.check(checks, "compile job consumed the quantized model", jobs.get("quantized_model_id"),
                   getattr(cm, "model_id", None), getattr(cm, "model_id", None) == jobs.get("quantized_model_id"))
    q_used = core.qairt_of(pjob)
    if s.get("qairt"):
        want_q = core.resolve_qairt(s["qairt"])
        ok = bool(q_used and want_q and (q_used == want_q["full_version"] or q_used.startswith(want_q["api_version"])))
        core.check(checks, "QAIRT version on device", want_q or s["qairt"], q_used, ok,
                   "tags like 'latest' are resolved now; re-verify soon after the run")
    return core.summarize(checks, qairt=q_used, calibration=jobs.get("calibration", "Qualcomm's calibration data"))


def _unique_among_siblings(vdir: Path, var: dict[str, Any]) -> bool:
    mine = var["local_input_spec"]
    for other in vdir.parent.glob("*/variant.json"):
        if other.parent == vdir:
            continue
        o = read_json(other)
        if o.get("value") != var.get("value") and o.get("local_input_spec") == mine:
            return False
    return True
