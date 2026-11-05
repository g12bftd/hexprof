"""Study bookkeeping (resumable state per study) and the on-device verification helpers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from hexprof.io_utils import read_json, write_json

DEFAULT_DEVICE = "Samsung Galaxy S25"

# target -> axis -> default sweep values (strings, as typed on the command line)
AXES: dict[str, dict[str, list[str]]] = {
    "gemma4-text": {
        "context-length": ["512", "1024", "2048", "4096"],
        "prompt-length": ["64", "128", "256", "512"],
        "decode-tokens": ["1", "2", "4", "8"],
    },
    "gemma4-vision": {
        # Gemma 4 resizes every image to fill this many soft tokens x pooling_kernel^2 patches: the token
        # budget sets the graph; the input image size does not (any square image gives the same graph)
        "soft-tokens": ["70", "140", "280", "560", "1120"],
        "precision": ["float", "w8a16"],
        "mixed-precision": ["0", "5", "10", "25", "50"],
        "qairt": ["qualcomm", "default", "latest"],
    },
    "siglip2-image": {
        "batch": ["1", "2", "4", "8"],
        "precision": ["float", "w8a16", "w8a8", "w4a16"],
        "mixed-precision": ["0", "5", "10", "25", "50"],
        "qairt": ["qualcomm", "default", "latest"],
    },
    "siglip2-text": {
        "batch": ["1", "2", "4", "8", "16"],
        "precision": ["float", "w8a16", "w8a8", "w4a16"],
        "mixed-precision": ["0", "5", "10", "25", "50"],
        "qairt": ["qualcomm", "default", "latest"],
    },
    # shrunk models with random weights (hexprof dummy): one knob swept, the others fixed (--dims) or real
    "dummy-gemma4-text": {
        "layers": ["6", "12", "24", "42"],
        "hidden": ["640", "1280", "1920", "2560"],
        "intermediate": ["2560", "5120", "10240", "15360"],
        "heads": ["2", "4", "8", "16"],
        "kv-heads": ["1", "2", "4", "8"],
        "head-dim": ["64", "128", "256"],
        "global-head-dim": ["128", "256", "512"],
        "sliding-window": ["128", "256", "512", "1024"],
        "kv-shared-layers": ["0", "6", "12", "18"],
        "ple-dim": ["64", "128", "256"],
        "vocab": ["32768", "65536", "131072", "262144"],
        "context-length": ["1024", "2048", "4096"],
        "prompt-length": ["64", "128", "256"],
        "precision": ["w4a16", "w8a16"],
    },
    "dummy-gemma4-vision": {
        "layers": ["4", "8", "16"],
        "hidden": ["384", "768", "1152"],
        "intermediate": ["1536", "3072", "4608"],
        "heads": ["6", "12"],
        "soft-tokens": ["70", "140", "280", "560", "1120"],
        "pooling-kernel": ["2", "3", "4"],
        "text-hidden": ["1280", "2560"],
    },
    "dummy-siglip2-image": {
        "layers": ["3", "6", "12"],
        "hidden": ["384", "768", "1024"],
        "intermediate": ["1536", "3072", "4096"],
        "heads": ["6", "12", "16"],
        "image-size": ["224", "256", "384", "512"],
        "patch-size": ["14", "16", "32"],
        "batch": ["1", "2", "4"],
        "precision": ["float", "w8a16", "w8a8", "w4a16"],
    },
    "dummy-siglip2-text": {
        "layers": ["3", "6", "12"],
        "hidden": ["384", "768", "1024"],
        "intermediate": ["1536", "3072", "4096"],
        "heads": ["6", "12", "16"],
        "vocab": ["32000", "64000", "256000"],
        "seq-len": ["16", "32", "64", "128"],
        "batch": ["1", "2", "4", "8"],
        "precision": ["float", "w8a16", "w8a8", "w4a16"],
    },
}
NUMERIC_AXES = {"context-length", "prompt-length", "decode-tokens", "soft-tokens", "batch", "mixed-precision",
                # dummy knobs
                "layers", "hidden", "intermediate", "heads", "kv-heads", "head-dim", "global-head-dim",
                "sliding-window", "kv-shared-layers", "ple-dim", "vocab", "patch-size", "pooling-kernel",
                "text-hidden", "image-size", "seq-len", "projection"}
ZERO_OK = {"kv-shared-layers"}
BASELINE = {  # Qualcomm's shipped setting per axis: the reference every study includes
    ("gemma4-text", "context-length"): "4096", ("gemma4-text", "prompt-length"): "128",
    ("gemma4-text", "decode-tokens"): "1", ("gemma4-vision", "soft-tokens"): "280",
    ("gemma4-vision", "precision"): "float", ("siglip2-image", "batch"): "1", ("siglip2-text", "batch"): "1",
    ("siglip2-image", "precision"): "w8a16", ("siglip2-text", "precision"): "w8a16",
    # qairt: "qualcomm" = the version qai_hub_models pins (what its exports use); default / latest = AI Hub tags
    ("gemma4-vision", "qairt"): "qualcomm", ("siglip2-image", "qairt"): "qualcomm", ("siglip2-text", "qairt"): "qualcomm",
    # mixed precision: 0 = no layers overridden (the plain base precision)
    ("gemma4-vision", "mixed-precision"): "0", ("siglip2-image", "mixed-precision"): "0",
    ("siglip2-text", "mixed-precision"): "0",
}
QUANTIZED = {"w8a16", "w8a8", "w4a16", "w4a8", "w16a16"}


def default_study_dir(target: str, axis: str) -> Path:
    root = os.environ.get("HEXPROF_DOWNLOAD_DIR") or "."
    return Path(root) / "studies" / f"{target}__{axis}"


def validate(target: str, axis: str, values: Sequence[str]) -> list[str]:
    if target not in AXES:
        raise ValueError(f"unknown target {target!r}; choose from {', '.join(AXES)}")
    if axis not in AXES[target]:
        raise ValueError(f"{target} has no axis {axis!r}; choose from {', '.join(AXES[target])}")
    vals = [str(v) for v in (values or AXES[target][axis])]
    if axis == "mixed-precision":
        bad = [v for v in vals if not v.isdigit() or not 0 <= int(v) < 100]
        if bad:
            raise ValueError(f"mixed-precision takes the percentage of layers kept at higher precision, 0-99; got {bad}")
    elif axis in NUMERIC_AXES:
        bad = [v for v in vals if not v.isdigit() or (int(v) <= 0 and axis not in ZERO_OK)]
        if bad:
            raise ValueError(f"{axis} takes positive integers; got {bad}")
        if axis == "soft-tokens" and any(int(v) not in (70, 140, 280, 560, 1120) for v in vals):
            raise ValueError("soft-tokens values must be Gemma 4 image-processor budgets: 70, 140, 280, 560, 1120")
    if axis == "precision":
        ok = {"float", "w8a16", "w8a8", "w4a16", "w4a8", "w16a16"}
        bad = [v for v in vals if v not in ok]
        if bad:
            raise ValueError(f"precision values must be among {sorted(ok)}; got {bad}")
    if len(set(vals)) != len(vals):
        raise ValueError(f"duplicate values in {vals}")
    return vals


@dataclass
class Study:
    target: str
    axis: str
    values: list[str]
    fixed: dict[str, Any]
    out: Path
    device: str = DEFAULT_DEVICE
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def state_path(self) -> Path:
        return self.out / "study.json"

    @classmethod
    def open(cls, target: str, axis: str, values: Sequence[str], fixed: dict[str, Any], out: str | Path | None,
             device: str = DEFAULT_DEVICE) -> "Study":
        vals = validate(target, axis, values)
        s = cls(target, axis, vals, fixed, Path(out) if out else default_study_dir(target, axis), device)
        s.out.mkdir(parents=True, exist_ok=True)
        if s.state_path.exists():
            s.state = read_json(s.state_path)
            prev = {k: s.state.get(k) for k in ("target", "axis", "fixed", "device")}
            now = {"target": target, "axis": axis, "fixed": fixed, "device": device}
            if prev != now:
                raise ValueError(f"{s.out} holds a study with other settings ({prev}); use another --out")
        s.state.update(target=target, axis=axis, fixed=fixed, device=device,
                       values=sorted(set(s.state.get("values", [])) | set(vals), key=_sort_key))
        s.state.setdefault("variants", {})
        s.save()
        return s

    def save(self) -> None:
        write_json(self.state_path, self.state)

    def variant_dir(self, value: str) -> Path:
        return self.out / f"{self.axis}-{value}"

    def record(self, value: str, **kw: Any) -> None:
        v = self.state["variants"].setdefault(value, {"value": value, "dir": str(self.variant_dir(value))})
        v.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.save()

    def run(self, one: Callable[[str, Path], dict[str, Any]], verify: Callable[[str, Path], dict[str, Any]],
            force: bool = False) -> dict[str, Any]:
        """Build + profile every value on the device (resuming finished ones), then verify all of them.

        Verification runs after the whole sweep so that cross-variant checks (does each value give a
        different graph?) see every variant.
        """
        for i, value in enumerate(self.values, 1):
            vdir = self.variant_dir(value)
            st = self.state["variants"].get(value, {})
            print(f"\n=== [{i}/{len(self.values)}] {self.target} {self.axis} = {value} -> {vdir}", flush=True)
            if st.get("status") == "profiled" and not force:
                print("    already profiled (use --force to redo)")
                continue
            self.record(value, status="running")
            try:
                info = one(value, vdir)
                self.record(value, status="profiled", info=info, error=None)
            except Exception as e:  # noqa: BLE001 - one failing value must not stop the sweep
                self.record(value, status="failed", error=f"{type(e).__name__}: {e}")
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
        self.verify_all(verify)
        return self.state

    def verify_all(self, verify: Callable[[str, Path], dict[str, Any]]) -> None:
        print("\n=== verification: what ran on the device vs what was requested", flush=True)
        for value in self.values:
            if self.state["variants"].get(value, {}).get("status") != "profiled":
                continue
            vdir = self.variant_dir(value)
            try:
                ver = verify(value, vdir)
            except Exception as e:  # noqa: BLE001
                ver = {"ok": False, "checks": [], "error": f"{type(e).__name__}: {e}"}
            write_json(vdir / "verification.json", ver)
            self.record(value, verified=ver["ok"])
            bad = [c for c in ver.get("checks", []) if not c["ok"]]
            print(f"    {self.axis} = {value}: {'VERIFIED' if ver['ok'] else 'NOT VERIFIED'} "
                  f"({len(ver.get('checks', []))} checks"
                  + (f", {len(bad)} failed: " + "; ".join(f"{c['what']}: expected {c['expected']}, got {c['actual']}"
                                                         for c in bad[:3]) if bad else "") + ")"
                  + (f" {ver['error']}" if ver.get("error") else ""), flush=True)


def kind(target: str) -> str:
    """The deployment a target's results look like: 'gemma4-text' (split LLM parts) or 'component'."""
    return "gemma4-text" if target.replace("dummy-", "") == "gemma4-text" else "component"


def baseline_of(st: dict[str, Any]) -> str | None:
    """The reference value of a study: Qualcomm's setting, or for a dummy the knob's unswept value."""
    return st.get("baseline") or BASELINE.get((st.get("target"), st.get("axis")))


def _sort_key(v: str) -> tuple[int, Any]:
    return (0, int(v)) if str(v).isdigit() else (1, str(v))


# --------------------------------------------------------------------------- verification helpers


def check(checks: list[dict[str, Any]], what: str, expected: Any, actual: Any, ok: bool, note: str = "") -> None:
    checks.append({"what": what, "expected": expected, "actual": actual, "ok": bool(ok), "note": note})


def shapes_of(spec: Any, graph: str | None = None) -> dict[str, list[int]]:
    """{input name: dims} from an AI Hub job's ``shapes``, a model's ``input_spec`` (keyed by graph) or an InputSpec."""
    from hexprof.hwreport import io as IO

    return IO.shapes(spec, graph)


def dtypes_of(spec: Any, graph: str | None = None) -> dict[str, str]:
    from hexprof.hwreport import io as IO

    return {k: v["dtype"] for k, v in IO.normalize(spec, graph).items() if v.get("dtype")}


def qairt_of(job: Any) -> str | None:
    """QAIRT version AI Hub used for this job (Qualcomm's ToolVersions.from_job)."""
    from hexprof.aihub import client as hc

    return hc.qairt_version(job)


def qualcomm_qairt() -> str | None:
    """The QAIRT version Qualcomm's qai_hub_models pins for context binaries (used when none is given)."""
    try:
        from qai_hub_models import TargetRuntime  # type: ignore[import-not-found]

        v = TargetRuntime.QNN_CONTEXT_BINARY.default_qairt_version
        return str(getattr(v, "api_version", None) or v)
    except Exception:  # noqa: BLE001
        return None


def resolve_qairt(tag: str) -> dict[str, str] | None:
    """'default' / 'latest' / '2.xx' -> AI Hub's framework entry (api_version, full_version)."""
    from hexprof.aihub import client as hc

    for fw in hc.hub().get_frameworks():
        if getattr(fw, "name", "").upper() != "QAIRT":
            continue
        tags = list(getattr(fw, "api_tags", []) or [])
        if tag in tags or tag in (fw.api_version, fw.full_version):
            return {"api_version": fw.api_version, "full_version": fw.full_version, "tags": ",".join(tags)}
    return None


def device_check(checks: list[dict[str, Any]], job: Any, device: str, what: str = "device") -> None:
    name = getattr(getattr(job, "device", None), "name", None)
    check(checks, what, device, name, name == device)


def npu_check(checks: list[dict[str, Any]], profile_path: Path | None, hw_dir: Path | None, what: str) -> None:
    """Did every op of the profiled graph run on the NPU? From AI Hub's per-layer compute units.

    A QNN context binary is built for the HTP backend as a whole (an op the HTP cannot run makes the compile
    or link fail; there is no silent CPU fallback inside it), but this reads it back from the profile anyway:
    any CPU / GPU layer fails the check. Without per-layer data, a QHAS file (the HTP analyzer's output)
    is taken as evidence that the graph ran on the HTP."""
    import json

    detail: list[dict[str, Any]] = []
    if profile_path and Path(profile_path).is_file():
        try:
            detail = json.loads(Path(profile_path).read_text()).get("execution_detail") or []
        except (OSError, ValueError):
            detail = []
    units: dict[str, int] = {}
    for d in detail:
        u = str(d.get("compute_unit", "UNKNOWN")).upper()
        units[u] = units.get(u, 0) + 1
    if detail:
        npu = units.get("NPU", 0)
        check(checks, f"{what}: every op on the NPU", f"{len(detail)} of {len(detail)} on NPU",
              ", ".join(f"{v} {k}" for k, v in sorted(units.items())), npu == len(detail),
              "AI Hub's per-layer compute units of the profile job")
        return
    qhas = sorted(Path(hw_dir).rglob("*qhas*.json")) if hw_dir and Path(hw_dir).exists() else []
    check(checks, f"{what}: ran on the NPU", "per-layer compute units, or the HTP analyzer's QHAS file",
          f"QHAS {qhas[0].name}" if qhas else "no per-layer data and no QHAS file", bool(qhas),
          "the profile had no per-layer detail")


def summarize(checks: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"ok": bool(checks) and all(c["ok"] for c in checks), "checks": checks, **extra}
