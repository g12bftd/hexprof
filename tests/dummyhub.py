"""A fake AI Hub with the job chain hexprof's dummy runner and its verification read.

Jobs remember what they consumed (``model``), compile jobs the shapes they were given; compiled
models report their I/O spec keyed by graph name, as AI Hub does. ``outputs_fn(path, specs)`` gives
output shapes (tests use a table; a local end-to-end check runs ONNX Runtime).
"""

from __future__ import annotations

import itertools
import re
from typing import Any, NamedTuple


class TS(NamedTuple):
    name: str
    dtype: str
    shape: tuple
    scale: Any = None
    zero_point: Any = None


class Dev:
    def __init__(self, name: str) -> None:
        self.name, self.os = name, "15"


class Model:
    def __init__(self, mid: str, payload: dict[str, Any], input_spec: Any = None, output_spec: Any = None,
                 producer: Any = None) -> None:
        self.model_id, self.payload, self.input_spec, self.output_spec, self._producer = (
            mid, payload, input_spec, output_spec, producer)
        self.name = payload.get("name", mid)

    def get_producer(self) -> Any:
        return self._producer


class Status:
    def __init__(self, ok: bool, msg: str = "") -> None:
        self.success, self.message = ok, msg


class Job:
    def __init__(self, jid: str, kind: str, **kw: Any) -> None:
        self.job_id, self.kind = jid, kind
        self.model = kw.pop("model", None)
        self.models = kw.pop("models", None)
        self.options = kw.pop("options", "")
        self.device = kw.pop("device", None)
        self.target = kw.pop("target", None)
        self.fail = kw.pop("fail", False)
        self.__dict__.update(kw)

    @property
    def url(self) -> str:
        return f"https://workbench.aihub.qualcomm.com/jobs/{self.job_id}/"

    def wait(self) -> Status:
        return Status(not self.fail, "fake failure" if self.fail else "")

    def get_target_model(self) -> Any:
        return self.target

    def download_profile(self) -> dict:
        return self.profile


class DummyHub:
    def __init__(self, outputs_fn: Any, latency_us: float = 5000.0, lie: str | None = None,
                 fail_shared_token_compile: bool = False) -> None:
        self.ids = itertools.count()
        self.jobs: dict[str, Job] = {}
        self.models: dict[str, Model] = {}
        self.outputs_fn, self.latency_us, self.lie = outputs_fn, latency_us, lie
        self.fail_shared_token_compile = fail_shared_token_compile
        self.calls: list[tuple[str, str]] = []

    def _id(self, p: str) -> str:
        return f"{p}{next(self.ids)}"

    def _model(self, payload: dict[str, Any], **kw: Any) -> Model:
        m = Model(self._id("m"), payload, **kw)
        self.models[m.model_id] = m
        return m

    def _job(self, kind: str, **kw: Any) -> Job:
        j = Job(self._id("j"), kind, **kw)
        self.jobs[j.job_id] = j
        return j

    def _hw(self, job: Any, out: Any) -> dict[str, list[str]]:
        """QHAS + optrace files like AI Hub's, from the repo's generators."""
        from pathlib import Path

        from tests.hwfakes import optrace_doc, qhas_doc, write_json

        seed = int(job.job_id[1:])
        q = write_json(Path(out) / "qhas" / f"{job.job_id}_main_qhas_summary.json", qhas_doc(120, seed=seed))
        o = write_json(Path(out) / "optrace" / f"{job.job_id}_main_profile_trace.json", optrace_doc(80, seed=seed))
        return {"qhas": [str(q)], "optrace": [str(o)]}

    def install(self, monkeypatch: Any) -> "DummyHub":
        import hexprof.aihub.client as hc

        monkeypatch.setattr(hc, "upload_model", lambda path, name=None: self._model({"path": str(path), "name": name}))
        monkeypatch.setattr(hc, "get_model", lambda mid: self.models[mid])
        monkeypatch.setattr(hc, "get_job", lambda jid: self.jobs[jid])
        monkeypatch.setattr(hc, "device", lambda name, os="": Dev(name))
        monkeypatch.setattr(hc, "qairt_version", lambda job: "2.50.0")
        monkeypatch.setattr(hc, "download_hw_artifacts", self._hw)

        def quantize(model: Model, calib: Any, prec: Any, name: str, options: str = "") -> Job:
            self.calls.append(("quantize", name))
            shapes = {k: list(v[0].shape) for k, v in calib.items()}
            tgt = self._model({**model.payload, "quantized": prec.name, "calibrated": shapes})
            j = self._job("quantize", model=model, options=options, target=tgt,
                          weights_dtype=f"QuantizeDtype.INT{prec.weight_bits}",
                          activations_dtype=f"QuantizeDtype.INT{prec.act_bits}")
            tgt._producer = j
            return j

        def compile_(model: Model, dev: Any, name: str, options: str = "", input_specs: Any = None) -> Job:
            self.calls.append(("compile", name))
            graph = re.search(r"context_enable_graphs=(\S+)", options).group(1)
            specs = {k: (list(v[0]), v[1]) for k, v in (input_specs or {}).items()}
            outs = self.outputs_fn(model.payload.get("path"), specs)
            if self.lie == "hidden":  # a compiler that built something else than it was asked for
                specs = {k: ([*s[:-1], s[-1] * 2] if k == "inputs_embeds" else s, d) for k, (s, d) in specs.items()}
            cal = model.payload.get("calibrated") or {}
            # an AI Hub whose quantized models keep the calibration shapes: other shapes do not compile
            fail = self.fail_shared_token_compile and any(k in cal and cal[k] != s for k, (s, _) in specs.items())
            tgt = self._model({**model.payload, "compile_options": options},
                              input_spec={graph: [TS(k, d, tuple(s)) for k, (s, d) in specs.items()]},
                              output_spec={graph: [TS(k, "float32", tuple(s)) for k, s in outs.items()]})
            j = self._job("compile", model=model, options=options, target=tgt, fail=fail,
                          shapes={k: (tuple(s), d) for k, (s, d) in specs.items()})
            tgt._producer = j
            return j

        def link(models: list[Model], dev: Any, name: str, options: str = "") -> Job:
            self.calls.append(("link", name))
            spec = {g: v for m in models for g, v in (m.input_spec or {}).items()}
            tgt = self._model({"linked": [m.model_id for m in models]}, input_spec=spec)
            j = self._job("link", models=list(models), options=options, target=tgt)
            tgt._producer = j
            return j

        def infer(model: Model, dev: Any, inputs: Any = None, *, profile: bool = True, name: str | None = None,
                  options: str = "") -> Job:
            self.calls.append(("inference", name or ""))
            from tests.fakes import fake_profile

            return self._job("profile", model=model, options=options, device=dev,
                             profile=fake_profile(self.latency_us))

        monkeypatch.setattr(hc, "submit_quantize", quantize)
        monkeypatch.setattr(hc, "submit_compile", compile_)
        monkeypatch.setattr(hc, "submit_link", link)
        monkeypatch.setattr(hc, "submit_inference", infer)
        return self


class MonkeyPatch:
    """Minimal stand-in for pytest's monkeypatch outside pytest."""

    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, Any]] = []

    def setattr(self, obj: Any, name: str, value: Any) -> None:
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self) -> None:
        for obj, name, value in reversed(self._undo):
            setattr(obj, name, value)
