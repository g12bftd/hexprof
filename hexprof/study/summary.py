"""Ablation summary across studies: every verified change against Qualcomm's setting.

Each study varies one setting with everything else at Qualcomm's defaults (unless fixed
otherwise), so a study's baseline value is the shipped deployment and every other value is a
one-factor ablation. For each verified value this page shows the throughput ratio against the
baseline and the change in the hardware metrics that explain it (DRAM traffic and bandwidth,
achieved TOPS, waiting inside ops, memory).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from hexprof.io_utils import _jsonable, read_json, write_json
from hexprof.study import core
from hexprof.study.compare import AXIS_LABEL

# (metric, label, kind, higher is better); the first is the headline throughput
HEADLINE = {
    "gemma4-text": [("decode_tokens_per_s", "decode tok/s", "num", True),
                    ("prefill_tokens_per_s", "prefill tok/s", "num", True),
                    ("dram_per_token", "DRAM per token", "bytes", False),
                    ("decode_dram_gbps", "decode DRAM GB/s", "num", None),
                    ("prefill_tops", "prefill TOPS", "num", True)],
    "gemma4-text/decode-tokens": [("max_tokens_per_s", "tok/s if all drafts accepted", "num", True),
                                  ("step_ms", "step ms", "num", False),
                                  ("dram_per_step", "DRAM per step", "bytes", False),
                                  ("step_dram_gbps", "DRAM GB/s", "num", None)],
    "component": [("items_per_s", "images or texts / s", "num", True), ("ms_per_item", "ms per item", "num", False),
                  ("dram_gbps", "DRAM GB/s", "num", None), ("tops", "achieved TOPS", "num", True),
                  ("waiting_pct", "waiting %", "pct", False), ("peak_mb", "inference peak MB", "num", False)],
}


def _headline(target: str, axis: str) -> list[tuple[str, str, str, bool | None]]:
    if target.startswith("dummy-"):
        target = core.kind(target)
    return HEADLINE.get(f"{target}/{axis}") or HEADLINE.get(target) or HEADLINE["component"]


def summarize(studies_dir: str | Path | None = None) -> Path:
    from hexprof.hwreport import html as H
    from hexprof.study.compare import compare

    root = Path(studies_dir or Path(os.environ.get("HEXPROF_DOWNLOAD_DIR") or ".") / "studies")
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for sj in sorted(root.glob("*/study.json")):
        sdir = sj.parent
        cj = sdir / "comparison" / "comparison.json"
        if not cj.exists():
            compare(sdir)
        data = read_json(cj)
        st = data["study"]
        target, axis = st["target"], st["axis"]
        base_value = core.baseline_of(st)
        by = {v["value"]: v for v in data["variants"]}
        base = by.get(base_value)
        if not base or not base.get("verified"):
            notes.append(f"{target} / {axis}: Qualcomm's setting ({base_value}) is missing or unverified, so its values "
                         "can't be compared with the baseline. Add it to --values.")
            continue
        mets = _headline(target, axis)
        for v in data["variants"]:
            if v["value"] == base_value or not v.get("verified"):
                continue
            r: dict[str, Any] = {"study": f"{target} / {axis}", "change": f"{AXIS_LABEL.get(axis, axis)}: {base_value} → {v['value']}",
                                 "target": target, "axis": axis, "value": v["value"], "baseline": base_value,
                                 "report": f"../{sdir.name}/comparison/index.html", "fixed": st.get("fixed") or {}}
            for mk, _, _, _ in mets:
                b, x = base["metrics"].get(mk), v["metrics"].get(mk)
                r[f"{mk}"] = x
                r[f"{mk}__base"] = b
                r[f"{mk}__ratio"] = x / b if x is not None and b else None
            rows.append(r)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    page = _page(rows, notes)
    (out / "index.html").write_text(H.render_page(_jsonable(page)), encoding="utf-8")
    write_json(out / "summary.json", {"ablations": rows, "notes": notes})
    if rows:
        keys = sorted({k for r in rows for k in r if not isinstance(r[k], dict)})
        with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"Ablation summary: {out / 'index.html'} ({len(rows)} verified changes from {len(set(r['study'] for r in rows))} studies)")
    return out / "index.html"


def _page(rows: list[dict[str, Any]], notes: list[str]) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        t = core.kind(r["target"]) if r["target"].startswith("dummy-") else r["target"]
        key = "gemma4-text/decode-tokens" if r["study"] == "gemma4-text / decode-tokens" else (
            t if t in HEADLINE else "component")
        groups.setdefault(key, []).append(r)
    for key, rs in groups.items():
        mets = HEADLINE[key]
        mk, label, _, _ = mets[0]
        bars = sorted(({"label": r["change"] + ("" if r["target"] == "gemma4-text" else f" ({r['target']})"),
                        "value": r[f"{mk}__ratio"],
                        "extra": [["baseline", f"{r[f'{mk}__base']:.4g}"], ["this change", f"{r[mk]:.4g}"]]}
                       for r in rs if r.get(f"{mk}__ratio") is not None), key=lambda b: -b["value"])
        cols = [{"key": "change", "label": "Change", "kind": "str", "link": "report"}]
        for m, lbl, kind, _ in mets:
            cols += [{"key": f"{m}__base", "label": f"{lbl} (base)", "kind": kind},
                     {"key": m, "label": lbl, "kind": kind}, {"key": f"{m}__ratio", "label": "× base", "kind": "num"}]
        title = {"gemma4-text": "Gemma 4 text parts", "gemma4-text/decode-tokens": "Gemma 4 multi-token decode step",
                 "component": "Vision encoder and SigLIP 2"}[key]
        sections.append({"title": f"{title}: {label} relative to Qualcomm's setting",
                         "why": "1.00 = the shipped deployment; above 1 is faster. Every row is one setting changed on "
                                "its own and verified on the device; the table adds the hardware metrics behind it.",
                         "blocks": ([{"type": "hbar", "rows": bars, "kind": "num", "title": f"{label} × baseline"}]
                                    if bars else []) +
                                   [{"type": "note", "text": "", "table": {"open": True, "cols": cols, "rows": rs}}]})
    return {"title": "Ablation summary", "subtitle": f"{len(rows)} verified one-factor changes against Qualcomm's "
            "deployment on the Galaxy S25", "warnings": notes, "sections": sections or
            [{"title": "No verified changes yet", "why": "Run studies with `hexprof study run` (or slurm/study.sh).",
              "blocks": []}],
            "footer": "Generated by hexprof study summary from each study's comparison.json."}
