"""Entry points behind ``hexprof study run | verify | compare | list``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from hexprof.study import core


def _runners(target: str) -> tuple[Any, Any]:
    if target.startswith("dummy-"):
        from hexprof.dummy import study as m
    elif target == "gemma4-text":
        from hexprof.study import gemma4_text as m  # type: ignore[no-redef]
    else:
        from hexprof.study import components as m  # type: ignore[no-redef]
    return m.run_one, m.verify


def default_ceilings() -> tuple[str | None, str | None]:
    root = Path(os.environ.get("HEXPROF_DOWNLOAD_DIR") or ".") / "runs"
    c, b = root / "ceil-compute" / "compute_ceiling.json", root / "ceil-bandwidth" / "bandwidth_ceiling.json"
    return (str(c) if c.exists() else None, str(b) if b.exists() else None)


def run(target: str, axis: str, values: Sequence[str], fixed: dict[str, Any], out: str | None = None,
        device: str = core.DEFAULT_DEVICE, force: bool = False, compare: bool = True,
        compute_json: str | None = None, bandwidth_json: str | None = None) -> Path:
    study = core.Study.open(target, axis, values, fixed, out, device)
    if target.startswith("dummy-") and "baseline" not in study.state:
        from hexprof.dummy import study as DS

        study.state["baseline"] = DS.baseline(target, axis, fixed)
        study.save()
    one, verify = _runners(target)
    print(f"Study {target} / {axis}: values {study.values}; fixed {fixed}; device {device}; results in {study.out}")
    if target == "gemma4-text":
        study.run(lambda v, d: one(axis, v, d, fixed, device), lambda v, d: verify(axis, v, d, fixed, device), force)
    else:
        study.run(lambda v, d: one(target, axis, v, d, fixed, device),
                  lambda v, d: verify(target, axis, v, d, fixed, device), force)
    if compare:
        from hexprof.study.compare import compare as cmp

        dc, db = default_ceilings()
        cmp(study.out, compute_json or dc, bandwidth_json or db)
    return study.out


def reverify(study_dir: str | Path) -> dict[str, Any]:
    from hexprof.io_utils import read_json

    st = read_json(Path(study_dir) / "study.json")
    study = core.Study.open(st["target"], st["axis"], st["values"], st.get("fixed") or {}, study_dir, st["device"])
    _, verify = _runners(st["target"])
    if st["target"] == "gemma4-text":
        study.verify_all(lambda v, d: verify(st["axis"], v, d, st.get("fixed") or {}, st["device"]))
    else:
        study.verify_all(lambda v, d: verify(st["target"], st["axis"], v, d, st.get("fixed") or {}, st["device"]))
    return study.state


def describe() -> str:
    lines = ["target          axis             default values (Qualcomm's setting marked *)"]
    for t, axes in core.AXES.items():
        for a, vals in axes.items():
            base = core.BASELINE.get((t, a))
            lines.append(f"{t:<15} {a:<16} " + " ".join(v + ("*" if v == base else "") for v in vals))
    return "\n".join(lines)
