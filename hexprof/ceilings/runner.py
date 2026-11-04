"""Run ceiling sweeps on AI Hub: build -> upload -> quantize -> compile -> link -> profile.

A QNN context binary (what GenieX QAIRT loads) is made the way AI Hub requires it now and the way
Qualcomm's own export does it: compile to ``qnn_dlc``, then a link job turns the DLC into a context
binary (AI Hub no longer accepts ``--target_runtime qnn_context_binary`` on compile jobs).

All jobs of a stage are submitted before waiting, so AI Hub runs them in
parallel. Job ids are written to ``<out>/state.json`` after every submission;
re-running the same command resumes instead of resubmitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from hexprof.aihub import client as hc
from hexprof.aihub.metrics import aihub_metrics, per_layer
from hexprof.ceilings.graphs import Graph, make_inputs
from hexprof.io_utils import write_json


@dataclass
class PointSpec:
    key: str
    series: str
    precision: hc.Precision
    build: Callable[[Path], Graph]  # streams weights into the given directory


class Sweep:
    def __init__(
        self,
        out_dir: str | Path,
        device_name: str,
        runtime: str = "qnn_context_binary",
        compile_options: str = "",
        profile_options: str = "",
        quantize_options: str = "",
        qairt_version: str | None = None,
        quantize_io: bool = True,
        n_calib: int = 4,
        hw: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device_name = device_name
        self.runtime = runtime
        self.compile_options = compile_options
        self.profile_options = profile_options
        self.quantize_options = quantize_options
        self.qairt_version = qairt_version
        self.quantize_io = quantize_io
        self.n_calib = n_calib
        self.hw = hw
        self.dry_run = dry_run
        self.state_path = self.out / "state.json"
        self.state: dict[str, dict[str, Any]] = (
            json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        )

    # ------------------------------------------------------------------ helpers

    def _save(self) -> None:
        write_json(self.state_path, self.state)

    @property
    def links(self) -> bool:
        return self.runtime == "qnn_context_binary"

    def _compile_opts(self, precision: hc.Precision) -> str:
        opts = [f"--target_runtime {'qnn_dlc' if self.links else self.runtime}"]
        if self.quantize_io and not precision.is_float:
            opts.append("--quantize_io")
        if self.qairt_version:
            opts.append(f"--qairt_version {self.qairt_version}")
        if self.compile_options:
            opts.append(self.compile_options)
        return " ".join(opts)

    def _link_opts(self) -> str:
        return f"--qairt_version {self.qairt_version}" if self.qairt_version else ""

    def deployable(self, st: dict[str, Any]) -> Any:
        """The model that runs on the device: the linked context binary, or the compiled model."""
        return hc.target_model(hc.get_job(st.get("link_job") or st["compile_job"]))

    def _profile_opts(self) -> str:
        opts = []
        if self.qairt_version:
            opts.append(f"--qairt_version {self.qairt_version}")
        if self.profile_options:
            opts.append(self.profile_options)
        return " ".join(opts)

    # ------------------------------------------------------------------ stages

    def _build_and_upload(self, specs: Sequence[PointSpec]) -> None:
        for s in specs:
            st = self.state.setdefault(s.key, {"key": s.key, "series": s.series, "precision": s.precision.name})
            if st.get("source_model_id") or (self.dry_run and st.get("model_dir")):
                continue
            model_dir = self.out / "models" / f"{s.key}.onnx"
            g = s.build(model_dir)
            st["graph"] = {
                "name": g.name,
                "meta": g.meta,
                "ops": g.ops,
                "matmul_ops": g.matmul_ops,
                "attention_ops": g.attention_ops,
                "weight_params": g.weight_params,
                "weight_bytes": g.weight_bytes(s.precision.weight_bits),
                "weight_bytes_no_scales": g.weight_bytes(s.precision.weight_bits, include_scales=False),
                "kv_bytes": g.kv_bytes(s.precision.act_bits),
                "inputs": {k: [list(v[0]), v[1]] for k, v in g.inputs.items()},
            }
            g.save(model_dir)
            st["model_dir"] = str(model_dir)
            mb = st["graph"]["weight_bytes"] / 2**20
            print(f"  built {s.key}: {g.ops / 1e9:.2f} GOP, {mb:.1f} MB weights at {s.precision.name}")
            if not self.dry_run:
                st["source_model_id"] = hc.upload_model(model_dir, name=f"hexprof_{s.key}").model_id
            self._save()

    def _wait_all(self, pending: dict[str, Any], kind: str, job_key: str) -> None:
        """Wait for every job; failed points get an error and are retried on the next run."""
        for key, job in pending.items():
            try:
                hc.wait_success(job, kind)
            except hc.HubJobError as e:
                print(f"WARNING: {e}")
                st = self.state[key]
                st["error"] = str(e)
                st["failed_" + job_key] = st.pop(job_key)
        self._save()

    def _ok(self, specs: Sequence[PointSpec]) -> list[PointSpec]:
        return [s for s in specs if "error" not in self.state[s.key]]

    def _inputs(self, st: dict[str, Any], n: int, kind: str = "random") -> dict[str, list]:
        specs = {k: (tuple(v[0]), v[1]) for k, v in st["graph"]["inputs"].items()}
        return make_inputs(specs, n=n, seed=0, kind=kind)

    def _quantize(self, specs: Sequence[PointSpec]) -> None:
        pending: dict[str, Any] = {}
        for s in self._ok(specs):
            st = self.state[s.key]
            if s.precision.is_float or st.get("quantize_job"):
                continue
            job = hc.submit_quantize(hc.get_model(st["source_model_id"]), self._inputs(st, self.n_calib), s.precision,
                                     name=f"hexprof {s.key}", options=self.quantize_options)
            st["quantize_job"] = job.job_id
            pending[s.key] = job
            self._save()
        self._wait_all(pending, "quantize", "quantize_job")

    def _compile(self, specs: Sequence[PointSpec]) -> None:
        dev = hc.device(self.device_name)
        pending: dict[str, Any] = {}
        for s in self._ok(specs):
            st = self.state[s.key]
            if st.get("compile_job"):
                continue
            src = (hc.target_model(hc.get_job(st["quantize_job"])) if st.get("quantize_job")
                   else hc.get_model(st["source_model_id"]))
            opts = self._compile_opts(s.precision)
            job = hc.submit_compile(src, dev, name=f"hexprof {s.key}", options=opts)
            st["compile_job"], st["compile_options"] = job.job_id, opts
            pending[s.key] = job
            self._save()
        self._wait_all(pending, "compile", "compile_job")

    def _link(self, specs: Sequence[PointSpec]) -> None:
        if not self.links:
            return
        dev = hc.device(self.device_name)
        pending: dict[str, Any] = {}
        for s in self._ok(specs):
            st = self.state[s.key]
            if st.get("link_job"):
                continue
            dlc = hc.target_model(hc.get_job(st["compile_job"]))
            job = hc.submit_link([dlc], dev, name=f"hexprof {s.key}", options=self._link_opts())
            job = job[0] if isinstance(job, list) else job
            st["link_job"] = job.job_id
            pending[s.key] = job
            self._save()
        self._wait_all(pending, "link", "link_job")

    def _profile(self, specs: Sequence[PointSpec]) -> None:
        dev = hc.device(self.device_name)
        pending: dict[str, Any] = {}
        for s in self._ok(specs):
            st = self.state[s.key]
            if st.get("profile_job"):
                continue
            job = hc.submit_inference(self.deployable(st), dev, None, profile=True, name=f"hexprof {s.key}",
                                      options=self._profile_opts())
            st["profile_job"] = job.job_id
            pending[s.key] = job
            self._save()
        self._wait_all(pending, "profile", "profile_job")

    def _collect(self, specs: Sequence[PointSpec]) -> None:
        for s in self._ok(specs):
            st = self.state[s.key]
            if st.get("metrics"):
                continue
            job = hc.get_job(st["profile_job"])
            prof, _ = hc.save_profile(job, self.out / "profiles", s.key)
            st["metrics"] = aihub_metrics(prof)
            st["qairt"] = hc.qairt_version(job)
            st["latency_ms"] = st["metrics"]["inference_time_milliseconds"]
            st["job_url"] = job.url
            write_json(self.out / "profiles" / f"{s.key}.per_layer.json", per_layer(prof))
            if self.hw:
                st["hw_artifacts"] = hc.download_hw_artifacts(job, self.out / "hw" / s.key)
            self._save()

    def run(self, specs: Sequence[PointSpec]) -> list[dict[str, Any]]:
        print(f"{len(specs)} graph(s) -> {self.out}")
        self._build_and_upload(specs)
        if self.dry_run:
            print("dry run: graphs saved, no AI Hub jobs submitted")
            return [self.state[s.key] for s in specs]
        for s in specs:  # a re-run retries points that failed last time
            self.state[s.key].pop("error", None)
        for stage in (self._quantize, self._compile, self._link, self._profile, self._collect):
            stage(specs)
        failed = [s.key for s in specs if "error" in self.state[s.key]]
        if failed:
            print(f"{len(failed)} point(s) failed: {', '.join(failed)} (re-run the same command to retry them)")
        return [self.state[s.key] for s in specs]
