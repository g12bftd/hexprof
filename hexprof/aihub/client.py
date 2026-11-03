"""Thin, version-tolerant wrappers around the ``qai_hub`` Python client.

``qai_hub`` >= 0.56 folded profile jobs into inference jobs
(``submit_inference_job(..., profile=True)``); older clients only have
``submit_profile_job``. Everything here works with both.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hexprof.io_utils import write_json


class HubJobError(RuntimeError):
    pass


def hub() -> Any:
    try:
        import qai_hub
    except ImportError as e:  # pragma: no cover - depends on env
        raise RuntimeError(
            "qai_hub is not installed. `pip install qai-hub` and run "
            "`qai-hub configure --api_token <TOKEN>` (token: workbench.aihub.qualcomm.com "
            "-> Account -> Settings -> API Token)."
        ) from e
    return qai_hub


def device(name: str, os: str = "") -> Any:
    return hub().Device(name, os) if os else hub().Device(name)


# --------------------------------------------------------------------------- precision


@dataclass(frozen=True)
class Precision:
    """Weight / activation precision of a quantized graph (None = float)."""

    name: str
    weight_bits: int | None
    act_bits: int | None

    @property
    def is_float(self) -> bool:
        return self.weight_bits is None

    @property
    def weight_bytes(self) -> float:
        return (self.weight_bits or 16) / 8.0

    @property
    def act_bytes(self) -> float:
        return (self.act_bits or 16) / 8.0


def parse_precision(text: str) -> Precision:
    t = text.strip().lower()
    if t in ("fp16", "float", "float16", "w16a16f"):
        return Precision("fp16", None, None)
    m = re.fullmatch(r"w(4|8|16)a(8|16)", t)
    if not m:
        raise ValueError(f"Unsupported precision {text!r}; use fp16, w8a8, w8a16, w4a8, w4a16, w16a16")
    return Precision(t, int(m.group(1)), int(m.group(2)))


def quantize_dtype(bits: int) -> Any:
    qd = hub().QuantizeDtype
    return {4: qd.INT4, 8: qd.INT8, 16: qd.INT16}[bits]


# --------------------------------------------------------------------------- submission


def upload_model(path: str | Path, name: str | None = None) -> Any:
    kwargs = {"name": name} if name else {}
    return hub().upload_model(str(path), **kwargs)


def submit_quantize(model: Any, calibration_data: Any, precision: Precision, name: str, options: str = "") -> Any:
    assert not precision.is_float
    return hub().submit_quantize_job(
        model=model,
        calibration_data=calibration_data,
        weights_dtype=quantize_dtype(precision.weight_bits),  # type: ignore[arg-type]
        activations_dtype=quantize_dtype(precision.act_bits),  # type: ignore[arg-type]
        name=name,
        options=options,
    )


def submit_compile(model: Any, dev: Any, name: str, options: str = "", input_specs: Any = None) -> Any:
    kwargs: dict[str, Any] = dict(model=model, device=dev, name=name, options=options)
    if input_specs is not None:
        kwargs["input_specs"] = input_specs
    job = hub().submit_compile_job(**kwargs)
    return job[0] if isinstance(job, list) else job


def submit_link(models: list[Any], dev: Any, name: str, options: str = "") -> Any:
    return hub().submit_link_job(models, device=dev, name=name, options=options)


def submit_inference(
    model: Any,
    dev: Any,
    inputs: Any = None,
    *,
    profile: bool = True,
    name: str | None = None,
    options: str = "",
) -> Any:
    """Inference job, optionally with per-layer profiling (AI Hub's profile job)."""
    h = hub()
    params = inspect.signature(h.submit_inference_job).parameters
    if "profile" in params:
        job = h.submit_inference_job(model=model, device=dev, inputs=inputs, name=name, options=options, profile=profile)
    elif profile and inputs is None:
        job = h.submit_profile_job(model=model, device=dev, name=name, options=options)
    else:
        job = h.submit_inference_job(model=model, device=dev, inputs=inputs, name=name, options=options)
    return job[0] if isinstance(job, list) else job


def get_job(job_id: str) -> Any:
    return hub().get_job(job_id)


def get_model(model_id: str) -> Any:
    return hub().get_model(model_id)


# --------------------------------------------------------------------------- results


def wait_success(job: Any, what: str = "job") -> Any:
    status = job.wait()
    if not status.success:
        raise HubJobError(f"{what} {job.job_id} failed: {getattr(status, 'message', status)} ({job.url})")
    return job


def target_model(job: Any) -> Any:
    wait_success(job, "job")
    model = job.get_target_model()
    if model is None:
        raise HubJobError(f"job {job.job_id} produced no target model ({job.url})")
    return model


def download_profile(job: Any) -> dict:
    wait_success(job, "profile job")
    profile = job.download_profile()
    if not isinstance(profile, dict) or not profile:
        raise HubJobError(f"job {job.job_id} has no profile (was it submitted with profile=True?)")
    return profile


def download_hw_artifacts(job: Any, out_dir: str | Path) -> dict[str, list[str]]:
    """Download optrace (PROFILE_TRACE) and QHAS summary JSONs when AI Hub has them.

    These exist for NPU (QNN/HTP) placements when the job ran with profiling; for
    other runtimes the lists are simply empty.
    """
    h = hub()
    art = getattr(h, "JobArtifactType", None)
    out: dict[str, list[str]] = {}
    if art is None or not hasattr(job, "download_artifacts_for_type"):
        return out
    for key, attr in (("optrace", "PROFILE_TRACE"), ("qhas", "QHAS_SUMMARY")):
        kind = getattr(art, attr, None)
        if kind is None:
            continue
        try:
            out[key] = job.download_artifacts_for_type(str(Path(out_dir) / key), kind)
        except Exception as e:  # noqa: BLE001 - artifacts are best-effort
            out[key] = []
            out[f"{key}_error"] = [f"{type(e).__name__}: {e}"]
    return out


def job_info(job: Any) -> dict[str, Any]:
    dev = getattr(job, "device", None)
    return {
        "job_id": job.job_id,
        "url": getattr(job, "url", None),
        "name": getattr(job, "name", None),
        "device": getattr(dev, "name", None),
        "device_os": getattr(dev, "os", None),
        "options": getattr(job, "options", None),
    }


def save_profile(job: Any, out_dir: str | Path, stem: str) -> tuple[dict, Path]:
    profile = download_profile(job)
    path = write_json(Path(out_dir) / f"{stem}.profile.json", profile)
    return profile, path
