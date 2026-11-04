"""Test doubles: a simulated phone running geniex-bench and a simulated AI Hub."""

from __future__ import annotations

import itertools
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hexprof.geniex.adb import Adb, ShellResult

# --------------------------------------------------------------------------- phone


class FakePhone(Adb):
    """adb + geniex-bench simulator backed by an in-memory filesystem."""

    def __init__(self, media_ms: float = 1600.0, prefill_tps: float = 300.0, decode_tps: float = 15.0) -> None:
        super().__init__(serial="FAKE", dry_run=False)
        self.fs: dict[str, bytes] = {}
        self.media_ms, self.prefill_tps, self.decode_tps = media_ms, prefill_tps, decode_tps
        self.bench_calls: list[list[str]] = []

    def _base(self) -> list[str]:  # never shell out
        return ["adb"]

    def devices(self) -> list[str]:
        return ["FAKE"]

    def getprop(self, name: str) -> str:
        return {"ro.soc.model": "SM8750", "ro.product.model": "SM-S931B", "ro.build.version.release": "16"}.get(name, "")

    def push(self, src: str | Path, dst: str) -> None:
        self.history.append(f"push {src} {dst}")
        p = Path(src)
        self.fs[dst] = p.read_bytes() if p.is_file() else b"<dir>"

    def pull(self, src: str, dst: str | Path) -> bool:
        if src not in self.fs:
            return False
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(self.fs[src])
        return True

    def exists(self, path: str) -> bool:
        return path in self.fs

    def shell(self, cmd: str, *, env: Any = None, secret_env: Any = None, check: bool = True, timeout: float | None = None) -> ShellResult:
        self.history.append(cmd)
        assert "{ENV}" not in cmd or env is not None
        m = re.search(r"\./bin/geniex-bench (.*) >(\S+) 2>(\S+)$", cmd)
        if m:
            return self._bench(shlex.split(m.group(1)), m.group(2), m.group(3))
        if cmd.startswith("cat "):
            path = cmd.split()[1]
            return ShellResult(cmd, 0 if path in self.fs else 1, self.fs.get(path, b"").decode())
        if cmd.startswith("ls -1 "):
            d = cmd.split()[2].rstrip("/") + "/"
            names = sorted({k[len(d):].split("/")[0] for k in self.fs if k.startswith(d)})
            return ShellResult(cmd, 0, "\n".join(names) + "\n")
        return ShellResult(cmd, 0, "")

    # -- geniex-bench model: TTFT = media + prefill; tokens = words of the prompt + 256 per image
    def _bench(self, args: list[str], out: str, err: str) -> ShellResult:
        self.bench_calls.append(args)

        def opt(name: str, default: str | None = None) -> str | None:
            return args[args.index(name) + 1] if name in args else default

        if "--matrix-file" in args:
            return self._matrix(args, opt, out, err)
        n_img = args.count("--image")
        prompts = self.fs[opt("--prompt-file")].decode().strip().split("\n---\n")
        lines, runs = [], []
        for i, p in enumerate(prompts):
            if len(prompts) > 1:
                lines.append(f"[sep ] prompt {i + 1}/{len(prompts)}")
            answer = f"Answer to: {p.strip()} ({n_img} image(s))"
            lines += ["[info] stop_reason=eos", f"[gen ] {answer}"]
            runs.append(self._run(i, len(p.split()) + 256 * n_img, len(answer.split()), n_img))
        lines.append("[ok  ] cell  plugin=llama_cpp device=npu ngl=-1 ttft=1.0ms prefill=1.0tps decode=1.0tps gen=1 tok")
        self.fs[out] = ("\n".join(lines) + "\n").encode()
        self.fs[err] = b""
        self.fs[opt("--output-json")] = json.dumps(self._cell(opt("--cell-id"), args, runs)).encode()
        return ShellResult("bench", 0, "")

    def _matrix(self, args: list[str], opt: Any, out: str, err: str) -> ShellResult:
        rows = [r.split("\t") for r in self.fs[opt("--matrix-file")].decode().splitlines() if r.strip()]
        reps = int(opt("-r", "5"))
        p = int(opt("-p", "512"))
        for row in rows:
            runs = [self._run(i, p, int(opt("-n")), 0) for i in range(reps)]
            self.fs[f"{opt('--output-json-dir')}/{row[0]}.json"] = json.dumps(self._cell(row[0], args, runs, row[2])).encode()
        self.fs[out], self.fs[err] = b"", b""
        return ShellResult("bench", 0, "")

    def _run(self, i: int, prompt_tokens: int, gen: int, n_img: int) -> dict[str, Any]:
        media_us = int(self.media_ms * 1000) if n_img else 0
        prompt_us = int(prompt_tokens / self.prefill_tps * 1e6)
        decode_us = int(gen / self.decode_tps * 1e6)
        return {"run_idx": i, "ttft_us": media_us + prompt_us, "media_us": media_us, "prompt_time_us": prompt_us,
                "decode_time_us": decode_us, "prompt_tokens": prompt_tokens, "gen_tokens": gen,
                "prefill_tps": self.prefill_tps, "decode_tps": self.decode_tps, "stop_reason": "eos"}

    @staticmethod
    def _cell(cell_id: str, args: list[str], runs: list[dict], device: str = "npu") -> dict[str, Any]:
        def med(k: str) -> float:
            return float(np.median([r[k] for r in runs]))

        return {
            "schema_version": "7", "cell_id": cell_id, "plugin": "llama_cpp", "device": device,
            "geniex_version": "v0.10.0", "llama_cpp_version": "b9999", "qairt_version": None,
            "params": {"n_ctx": int(args[args.index("-c") + 1]) if "-c" in args else 0},
            "runs": runs,
            "agg": {"ttft_ms": {"median": med("ttft_us") / 1000}, "prefill_tps": {"median": med("prefill_tps")},
                    "decode_tps": {"median": med("decode_tps")}, "gen_tokens": {"median": med("gen_tokens")},
                    "prompt_tokens": {"median": med("prompt_tokens")}, "media_ms": {"median": med("media_us") / 1000}},
        }


# --------------------------------------------------------------------------- AI Hub


@dataclass
class FakeStatus:
    success: bool = True
    message: str = ""


@dataclass
class FakeModel:
    model_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeJob:
    job_id: str
    profile: dict | None = None
    outputs: dict | None = None
    target: FakeModel | None = None
    fail: bool = False
    model: FakeModel | None = None
    options: str = ""

    @property
    def url(self) -> str:
        return f"https://workbench.aihub.qualcomm.com/jobs/{self.job_id}/"

    def wait(self) -> FakeStatus:
        return FakeStatus(not self.fail, "boom" if self.fail else "")

    def get_target_model(self) -> FakeModel | None:
        return self.target

    def download_profile(self) -> dict:
        return self.profile or {}

    def download_output_data(self) -> dict | None:
        return self.outputs


def fake_profile(latency_us: float, n_npu: int = 10, peak: tuple[int, int] = (0, 180 * 2**20)) -> dict[str, Any]:
    return {
        "execution_summary": {
            "estimated_inference_time": latency_us,
            "inference_memory_peak_range": peak,
            "first_load_time": 1_200_000,
            "warm_load_time": 90_000,
            "all_inference_times": [latency_us, latency_us * 1.02, latency_us * 1.1],
        },
        "execution_detail": [
            {"name": f"op{i}", "type": "Conv2d" if i % 2 else "MatMul", "compute_unit": "NPU", "execution_time": 10 + i}
            for i in range(n_npu)
        ],
    }


class FakeHub:
    """Monkeypatches hexprof.aihub.client. Latency comes from ``latency_fn(payload)``."""

    def __init__(self, latency_fn: Any, output_fn: Any = None) -> None:
        self.ids = itertools.count()
        self.jobs: dict[str, FakeJob] = {}
        self.models: dict[str, FakeModel] = {}
        self.latency_fn = latency_fn
        self.output_fn = output_fn
        self.calls: list[tuple[str, Any]] = []

    def _model(self, payload: dict) -> FakeModel:
        m = FakeModel(f"m{next(self.ids)}", payload)
        self.models[m.model_id] = m
        return m

    def _job(self, **kw: Any) -> FakeJob:
        j = FakeJob(f"j{next(self.ids)}", **kw)
        self.jobs[j.job_id] = j
        return j

    def install(self, monkeypatch: Any) -> "FakeHub":
        import hexprof.aihub.client as hc

        monkeypatch.setattr(hc, "upload_model", lambda path, name=None: self._model({"path": str(path), "name": name}))
        monkeypatch.setattr(hc, "get_model", lambda mid: self.models[mid])
        monkeypatch.setattr(hc, "get_job", lambda jid: self.jobs[jid])
        monkeypatch.setattr(hc, "device", lambda name, os="": name)
        monkeypatch.setattr(hc, "download_hw_artifacts", lambda job, out: {"optrace": [], "qhas": []})

        def quantize(model: FakeModel, calibration_data: Any, precision: Any, name: str, options: str = "") -> FakeJob:
            self.calls.append(("quantize", name, options))
            return self._job(target=self._model({**model.payload, "precision": precision.name}))

        def compile_(model: FakeModel, dev: Any, name: str, options: str = "", input_specs: Any = None) -> FakeJob:
            if "--target_runtime qnn_context_binary" in options:  # as AI Hub answers today
                raise RuntimeError("argument --target_runtime: Unsupported target runtime: qnn_context_binary. "
                                   "Expecting one of 'onnx', 'precompiled_qnn_onnx', 'qnn_dlc', 'tflite'")
            self.calls.append(("compile", name, options))
            return self._job(target=self._model({**model.payload, "compile_options": options}))

        def link(models: list[FakeModel], dev: Any, name: str, options: str = "") -> FakeJob:
            assert all("--target_runtime qnn_dlc" in m.payload.get("compile_options", "") for m in models)
            self.calls.append(("link", name, options))
            return self._job(target=self._model({**models[0].payload, "linked": True}))

        def infer(model: FakeModel, dev: Any, inputs: Any = None, *, profile: bool = True, name: str | None = None,
                  options: str = "") -> FakeJob:
            self.calls.append(("inference", name, options))
            outputs = self.output_fn(model, inputs) if (self.output_fn and inputs is not None) else None
            self.last_inputs = inputs
            return self._job(profile=fake_profile(self.latency_fn(model.payload, inputs)), outputs=outputs,
                             model=model, options=options)

        monkeypatch.setattr(hc, "submit_quantize", quantize)
        monkeypatch.setattr(hc, "submit_compile", compile_)
        monkeypatch.setattr(hc, "submit_link", link)
        monkeypatch.setattr(hc, "qairt_version", lambda job: "2.50.0")
        monkeypatch.setattr(hc, "submit_inference", infer)
        return self
