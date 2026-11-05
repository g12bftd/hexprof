"""Compare a dummy's AI Hub jobs with the real deployment's: same compile / link / profile options?

The reference is your Qualcomm export (``hexprof gemma4 aihub --export-only``, default
``$HEXPROF_DOWNLOAD_DIR/gemma4-s25-export``). For the component it holds (the Gemma 4 vision encoder),
the profile job recorded in ``export_profiles.json`` is followed back to its link and compile jobs on
AI Hub. A dummy of that component (same precision, QAIRT left alone) is compiled, linked and profiled
with exactly those jobs' options, with the QAIRT version they ran with pinned explicitly, and the
verification checks it: same options, same QAIRT version for every job. The only differences between
the two deployments are then the model's dimensions and its (random) weights.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hexprof.io_utils import read_json

COMPONENT = {"gemma4-vision": "vision_encoder"}


def default_reference() -> Path | None:
    from hexprof.models.gemma4_aihub import default_export_dir

    d = default_export_dir()
    return d if (d / "export_profiles.json").exists() else None


def reference_jobs(export_dir: str | Path, component: str) -> dict[str, Any] | None:
    """{"profile", "link", "compile"} AI Hub jobs of ``component`` in a Qualcomm export, or None."""
    from hexprof.aihub import client as hc

    summ = read_json(Path(export_dir) / "export_profiles.json")
    rec = next((r for r in (summ.get("parts") or {}).values() if r.get("component") == component), None)
    if not rec or not (rec.get("job") or {}).get("job_id"):
        return None
    prof = hc.get_job(rec["job"]["job_id"])
    out: dict[str, Any] = {"profile": prof, "graph": rec.get("graph")}
    link = prof.model.get_producer() if getattr(prof, "model", None) is not None else None
    if link is not None and isinstance(getattr(link, "models", None), list):
        out["link"] = link
        comp = link.models[0].get_producer() if link.models else None
        if comp is not None:
            out["compile"] = comp
    elif link is not None:
        out["compile"] = link
    return out


_QAIRT = re.compile(r"--qairt_version\s+\S+")


def _norm(opts: Any) -> str:
    """Options without the QAIRT flag (compared separately, as the version each job actually ran with:
    Qualcomm's code leaves the flag out when its pinned version is AI Hub's default at that moment)."""
    return " ".join(_QAIRT.sub("", str(opts or "")).split())


def _api(version: str | None) -> str | None:
    """'2.50.0.250924' -> '2.50'."""
    m = re.match(r"(\d+)\.(\d+)", str(version or ""))
    return f"{m.group(1)}.{m.group(2)}" if m else None


def pin(opts: str, api: str | None) -> str:
    """The options with an explicit ``--qairt_version`` (the reference's), so AI Hub's default moving on
    since the reference ran cannot change the version."""
    opts = " ".join(str(opts or "").split())
    return opts if (not api or "--qairt_version" in opts) else f"{opts} --qairt_version {api}"


def reference_options(export_dir: str | Path, component: str) -> dict[str, Any] | None:
    """The reference jobs' options and the QAIRT version they ran with, to reuse for a dummy."""
    from hexprof.aihub import client as hc

    ref = reference_jobs(export_dir, component)
    if not ref or not all(ref.get(k) is not None for k in ("compile", "link", "profile")):
        return None
    q = hc.qairt_version(ref["compile"]) or hc.qairt_version(ref["profile"])
    return {"compile": str(ref["compile"].options or ""), "link": str(ref["link"].options or ""),
            "profile": str(ref["profile"].options or ""), "qairt": _api(q), "qairt_full": q,
            "jobs": {k: ref[k].job_id for k in ("compile", "link", "profile")}, "export_dir": str(export_dir)}


def compare(checks: list[dict[str, Any]], mine: dict[str, Any], ref: dict[str, Any], export_dir: str | Path,
            same_inputs: bool) -> None:
    """Add checks: each job's options and QAIRT version equal the reference's; the compiled input shapes too."""
    from hexprof.aihub import client as hc
    from hexprof.study import core

    for kind in ("compile", "link", "profile"):
        a, b = mine.get(kind), ref.get(kind)
        if a is None or b is None:
            core.check(checks, f"{kind} options = Qualcomm's export ({export_dir})", "both jobs found",
                       f"dummy {'found' if a else 'missing'}, reference {'found' if b else 'missing'}", False)
            continue
        oa, ob = _norm(getattr(a, "options", "")), _norm(getattr(b, "options", ""))
        core.check(checks, f"{kind} options = Qualcomm's export (job {b.job_id})", ob, oa, oa == ob,
                   "identical options (QAIRT compared below): only the model (dimensions, random weights) differs")
        qa, qb = hc.qairt_version(a), hc.qairt_version(b)
        if qa or qb:
            core.check(checks, f"{kind} job QAIRT version = Qualcomm's export", qb, qa, qa == qb)
    if same_inputs and mine.get("compile") is not None and ref.get("compile") is not None:
        sa, sb = core.shapes_of(getattr(mine["compile"], "shapes", None)), core.shapes_of(getattr(ref["compile"], "shapes", None))
        core.check(checks, "compiled input shapes = Qualcomm's export", sb, sa, bool(sa) and sa == sb)
