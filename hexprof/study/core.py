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
        "image-size": ["224", "336", "448", "560"],
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
}
NUMERIC_AXES = {"context-length", "prompt-length", "decode-tokens", "image-size", "batch", "mixed-precision"}
BASELINE = {  # Qualcomm's shipped setting per axis: the reference every study includes
    ("gemma4-text", "context-length"): "4096", ("gemma4-text", "prompt-length"): "128",
    ("gemma4-text", "decode-tokens"): "1", ("gemma4-vision", "image-size"): "448",
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
        bad = [v for v in vals if not v.isdigit() or int(v) <= 0]
        if bad:
            raise ValueError(f"{axis} takes positive integers; got {bad}")
        if axis == "image-size" and any(int(v) % 112 for v in vals):
            raise ValueError("image-size values must be multiples of 112 (whole patches for 14- or 16-px patches)")
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


def summarize(checks: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"ok": bool(checks) and all(c["ok"] for c in checks), "checks": checks, **extra}
