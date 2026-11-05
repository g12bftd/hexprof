"""Compare a dummy's AI Hub jobs with the real deployment's: same compile / link / profile options?

The reference is your Qualcomm export (``hexprof gemma4 aihub --export-only``, default
``$HEXPROF_DOWNLOAD_DIR/gemma4-s25-export``). For the component it holds (the Gemma 4 vision encoder),
the profile job recorded in ``export_profiles.json`` is followed back to its link and compile jobs on
AI Hub, and their options are compared with the dummy's own jobs. With the dummy's precision and QAIRT
left at Qualcomm's settings, the options must be identical: then the only differences between the two
deployments are the model's dimensions and its (random) weights.
"""

from __future__ import annotations

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


def _norm(opts: Any) -> str:
    return " ".join(str(opts or "").split())


def compare(checks: list[dict[str, Any]], mine: dict[str, Any], ref: dict[str, Any], export_dir: str | Path,
            same_inputs: bool) -> None:
    """Add checks: each job's options equal the reference's; the compiled input shapes too when expected."""
    from hexprof.study import core

    for kind in ("compile", "link", "profile"):
        a, b = mine.get(kind), ref.get(kind)
        if a is None or b is None:
            core.check(checks, f"{kind} options = Qualcomm's export ({export_dir})", "both jobs found",
                       f"dummy {'found' if a else 'missing'}, reference {'found' if b else 'missing'}", False)
            continue
        oa, ob = _norm(getattr(a, "options", "")), _norm(getattr(b, "options", ""))
        core.check(checks, f"{kind} options = Qualcomm's export (job {b.job_id})", ob, oa, oa == ob,
                   "identical options: only the model (dimensions, random weights) differs")
    if same_inputs and mine.get("compile") is not None and ref.get("compile") is not None:
        sa, sb = core.shapes_of(getattr(mine["compile"], "shapes", None)), core.shapes_of(getattr(ref["compile"], "shapes", None))
        core.check(checks, "compiled input shapes = Qualcomm's export", sb, sa, bool(sa) and sa == sb)
    from hexprof.aihub import client as hc

    qa, qb = hc.qairt_version(mine["profile"]), hc.qairt_version(ref["profile"]) if ref.get("profile") else None
    if qa and qb:
        core.check(checks, "QAIRT version = Qualcomm's export", qb, qa, qa == qb)
