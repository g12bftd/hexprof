"""Sweep one knob of a shrunk model: ``hexprof study run dummy-<target> --axis <knob> --values ...``.

Every value is a full ``hexprof dummy run`` (random checkpoint -> Qualcomm's export -> AI Hub ->
profile) in ``<study>/<knob>-<value>/``; the other knobs come from ``--dims`` or the real model. The
study's verification adds one cross-variant check: each value deployed a different model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hexprof.dummy import spec as S
from hexprof.io_utils import read_json

DEPLOY_AXES = {"context-length": "context_length", "prompt-length": "prompt_length", "precision": "precision",
               "batch": "batch"}


def base(target: str) -> str:
    return target[len("dummy-"):] if target.startswith("dummy-") else target


def request(target: str, axis: str | None, value: str | None, fixed: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
    t = base(target)
    dims = S.parse_dims(fixed.get("dims"))
    deploy = {k: fixed[k] for k in ("context_length", "prompt_length", "precision", "batch", "splits")
              if fixed.get(k) is not None}
    if axis and value is not None:
        if axis in S.knobs(t):
            dims[axis] = int(value)
        elif axis in DEPLOY_AXES:
            deploy[DEPLOY_AXES[axis]] = value if axis == "precision" else int(value)
        else:
            raise ValueError(f"{target} has no axis {axis}")
    return dims, deploy


def baseline(target: str, axis: str, fixed: dict[str, Any]) -> str | None:
    """The swept knob's value when it is not swept (from --dims, else the real model)."""
    t = base(target)
    dims, deploy = request(target, None, None, fixed)
    if axis in dims:
        return str(dims[axis])
    if axis in DEPLOY_AXES:
        v = deploy.get(DEPLOY_AXES[axis], S.DEPLOY[t].get(DEPLOY_AXES[axis]))
        return None if v is None else str(v)
    try:
        from hexprof.dummy import run as R

        r, _ = R.plan(t, dims, deploy, fixed.get("base_config"))
        return str(r.dims.get(axis)) if r.dims.get(axis) is not None else None
    except Exception:  # noqa: BLE001 - the base config may need the network; the study still runs
        return None


def run_one(target: str, axis: str, value: str, vdir: Path, fixed: dict[str, Any], device: str) -> dict[str, Any]:
    from hexprof.dummy import run as R

    dims, deploy = request(target, axis, value, fixed)
    ver = R.run(base(target), dims, deploy, vdir, device, base=fixed.get("base_config"), seed=int(fixed.get("seed") or 0),
                qairt=fixed.get("qairt"), hw=True, report=False, n_calib=fixed.get("calibration_samples"))
    return {"dims": dims, "deploy": deploy, "verified_at_run": ver.get("ok")}


def verify(target: str, axis: str, value: str, vdir: Path, fixed: dict[str, Any], device: str) -> dict[str, Any]:
    from hexprof.dummy import run as R
    from hexprof.study import core

    ver = R.verify(vdir, device)
    mine = read_json(vdir / "dummy_run.json")["resolved"]
    key = axis if axis in mine["dims"] else DEPLOY_AXES.get(axis)
    got = mine["dims"].get(axis) if axis in mine["dims"] else mine["deploy"].get(key or "")
    core.check(ver["checks"], f"{axis} of the deployed model", value, got, str(got) == str(value))
    clash = []
    for other in vdir.parent.glob("*/dummy_run.json"):
        if other.parent == vdir:
            continue
        o = read_json(other)["resolved"]
        if (o["dims"], o["deploy"]) == (mine["dims"], mine["deploy"]):
            clash.append(other.parent.name)
    core.check(ver["checks"], "differs from every other value of the sweep", "unique dims / deployment",
               clash or "unique", not clash)
    return core.summarize(ver["checks"], **{k: v for k, v in ver.items() if k not in ("ok", "checks")})


def component_and_batch(target: str, axis: str, value: str, fixed: dict[str, Any]) -> tuple[str, int]:
    from hexprof.study.components import TARGETS

    _, deploy = request(target, axis, value, fixed)
    return TARGETS[base(target)][2], int(deploy.get("batch") or 1)
