"""Comparison page for a study: one hardware report per variant, then every metric against the swept value.

Only verified variants enter the charts and the "best" tiles; unverified or failed ones are
listed with the checks that failed, so a number on the page always belongs to a setting that
was confirmed to have run on the device.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from hexprof.io_utils import _jsonable, read_json, write_json
from hexprof.study import core

GRAPH = re.compile(r"part_?(\d+)_of_(\d+)__(prompt|token)_ar(\d+)_cl(\d+)")
AXIS_LABEL = {"context-length": "context length (tokens)", "prompt-length": "prompt chunk (tokens)",
              "decode-tokens": "tokens per decode step", "image-size": "image size (px)", "batch": "batch size",
              "precision": "precision", "qairt": "QAIRT version",
              "mixed-precision": "% of layers kept at higher precision"}


def _n(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def gemma_text_metrics(summary: dict[str, Any], axis: str, value: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label, g in summary.items():
        m = GRAPH.search(label)
        if not m or not isinstance(g, dict):
            continue
        k = (g.get("qhas") or {}).get("kpis") or {}
        rows.append({"part": int(m.group(1)), "kind": m.group(3), "ar": int(m.group(4)), "cl": int(m.group(5)),
                     "aihub_ms": (g.get("aihub") or {}).get("inference_time_ms"),
                     "qhas_ms": (k.get("time_us") or 0) / 1e3 or None, "dram": k.get("total_dram"),
                     "macs": (g.get("efficiency") or {}).get("macs"),
                     "waiting": ((g.get("efficiency") or {}).get("stalls") or {}).get("waiting_pct")})
    out: dict[str, Any] = {"graphs": sorted(rows, key=lambda r: (r["kind"], r["ar"], r["part"]))}

    def total(sel: list[dict[str, Any]], key: str) -> float | None:
        vals = [r[key] for r in sel]
        return sum(vals) if sel and all(v is not None for v in vals) else None

    if axis == "decode-tokens":
        k = int(value)
        step = total(rows, "aihub_ms")
        out.update(step_ms=step, ms_per_token=step / k if step else None,
                   max_tokens_per_s=1e3 * k / step if step else None, dram_per_step=total(rows, "dram"))
        q = total(rows, "qhas_ms")
        out["step_dram_gbps"] = out["dram_per_step"] / (q / 1e3) / 1e9 if out["dram_per_step"] and q else None
        for r in rows:
            out[f"part {r['part']} ms"] = r["aihub_ms"]
        return out
    tok = [r for r in rows if r["ar"] == 1]
    pro = [r for r in rows if r["ar"] > 1]
    d, p = total(tok, "aihub_ms"), total(pro, "aihub_ms")
    ar = pro[0]["ar"] if pro else None
    out.update(decode_ms_per_token=d, decode_tokens_per_s=1e3 / d if d else None,
               prefill_chunk_ms=p, prefill_tokens_per_s=ar * 1e3 / p if p and ar else None,
               dram_per_token=total(tok, "dram"))
    qd, qp = total(tok, "qhas_ms"), total(pro, "qhas_ms")
    out["decode_dram_gbps"] = out["dram_per_token"] / (qd / 1e3) / 1e9 if out["dram_per_token"] and qd else None
    macs = total(pro, "macs")
    out["prefill_tops"] = 2 * macs / (qp / 1e3) / 1e12 if macs and qp else None
    for r in rows:
        out[f"{r['kind']} part {r['part']} ms"] = r["aihub_ms"]
    return out


def component_metrics(summary: dict[str, Any], component: str, batch: int) -> dict[str, Any]:
    g = summary.get(component) or {}
    a, k, e = g.get("aihub") or {}, (g.get("qhas") or {}).get("kpis") or {}, g.get("efficiency") or {}
    ms = a.get("inference_time_ms")
    return {"aihub_ms": ms, "ms_per_item": ms / batch if ms else None,
            "items_per_s": 1e3 * batch / ms if ms else None, "qhas_ms": (k.get("time_us") or 0) / 1e3 or None,
            "dram_bytes": k.get("total_dram"), "dram_gbps": k.get("dram_gbps"), "vtcm_gbps": k.get("vtcm_gbps"),
            "util_pct": k.get("percent_utilization"), "tops": e.get("achieved_tops"),
            "compute_util_pct": e.get("compute_util_pct"), "bw_util_pct": e.get("bw_util_pct"),
            "waiting_pct": (e.get("stalls") or {}).get("waiting_pct"),
            "peak_mb": (a.get("estimated_peak_memory_range_mb") or {}).get("max"),
            "load_peak_mb": (a.get("first_load_peak_memory_range_mb") or {}).get("max"),
            "npu_layers": (a.get("layer_counts") or {}).get("npu"), "total_layers": (a.get("layer_counts") or {}).get("total")}


METRICS = {  # (key, label, kind, higher_is_better)
    "gemma4-text": [("decode_tokens_per_s", "Decode (tok/s)", "num", True),
                    ("decode_ms_per_token", "Decode ms per token", "num", False),
                    ("prefill_tokens_per_s", "Prefill (tok/s)", "num", True),
                    ("prefill_chunk_ms", "Prefill ms per chunk", "num", False),
                    ("dram_per_token", "DRAM bytes per decoded token", "bytes", False),
                    ("decode_dram_gbps", "Decode DRAM GB/s", "num", True),
                    ("prefill_tops", "Prefill achieved TOPS", "num", True)],
    "gemma4-text/decode-tokens": [("max_tokens_per_s", "Tokens/s if every draft is accepted", "num", True),
                                  ("step_ms", "Decode step ms (all parts)", "num", False),
                                  ("ms_per_token", "ms per token in the step", "num", False),
                                  ("dram_per_step", "DRAM bytes per step", "bytes", False),
                                  ("step_dram_gbps", "DRAM GB/s", "num", True)],
    "component": [("aihub_ms", "AI Hub inference ms", "num", False),
                  ("ms_per_item", "ms per image / text", "num", False),
                  ("items_per_s", "Images / texts per second", "num", True),
                  ("dram_gbps", "DRAM GB/s", "num", True), ("tops", "Achieved TOPS", "num", True),
                  ("compute_util_pct", "% of P", "pct", True), ("waiting_pct", "Waiting inside ops %", "pct", False),
                  ("peak_mb", "Inference peak MB", "num", False)],
}


def compare(study_dir: str | Path, compute_json: str | None = None, bandwidth_json: str | None = None,
            rebuild: bool = False, top: int = 15) -> Path:
    from hexprof.hwreport import html as H
    from hexprof.hwreport import report

    sdir = Path(study_dir)
    st = read_json(sdir / "study.json")
    target, axis = st["target"], st["axis"]
    values = sorted(st.get("variants", {}), key=core._sort_key)
    variants: list[dict[str, Any]] = []
    for value in values:
        v = st["variants"][value]
        vdir = Path(v.get("dir") or sdir / f"{axis}-{value}")
        ver = read_json(vdir / "verification.json") if (vdir / "verification.json").exists() else {}
        row: dict[str, Any] = {"value": value, "status": v.get("status"), "verified": bool(ver.get("ok")),
                               "error": v.get("error") or ver.get("error"),
                               "failed_checks": [c for c in ver.get("checks", []) if not c["ok"]],
                               "n_checks": len(ver.get("checks", [])), "qairt": ver.get("qairt"),
                               "report": None, "metrics": {}}
        if v.get("status") == "profiled":
            rdir = vdir / "hw_report"
            if rebuild or not (rdir / "summary.json").exists():
                try:
                    report.build([vdir], rdir, top=top, compute_json=compute_json, bandwidth_json=bandwidth_json,
                                 title=f"{target}: {axis} = {value}")
                except FileNotFoundError as e:
                    row["error"] = str(e)
            if (rdir / "summary.json").exists():
                summ = read_json(rdir / "summary.json")
                row["report"] = ("../" + (rdir / "index.html").relative_to(sdir).as_posix()  # page lives in <study>/comparison/
                                 if rdir.is_relative_to(sdir) else (rdir / "index.html").as_uri())
                if target == "gemma4-text":
                    row["metrics"] = gemma_text_metrics(summ, axis, value)
                else:
                    from hexprof.study.components import TARGETS, settings

                    s = settings(target, axis, value, st.get("fixed") or {})
                    b = list(s["spec_kwargs"].values())[0] if s["spec_kwargs"] else 1
                    row["metrics"] = component_metrics(summ, TARGETS[target][2], int(b))
        variants.append(row)
    out = sdir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    page = _page(st, variants)
    (out / "index.html").write_text(H.render_page(_jsonable(page)), encoding="utf-8")
    write_json(out / "comparison.json", {"study": {k: st.get(k) for k in ("target", "axis", "values", "fixed", "device")},
                                         "variants": variants})
    flat = [{"value": r["value"], "verified": r["verified"], "status": r["status"],
             **{k: v for k, v in r["metrics"].items() if not isinstance(v, (list, dict))}} for r in variants]
    keys = sorted({k for r in flat for k in r}, key=lambda k: (k not in ("value", "verified", "status"), k))
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    print(f"Comparison: {out / 'index.html'} ({sum(r['verified'] for r in variants)} of {len(variants)} variants verified)")
    return out / "index.html"


def _page(st: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    target, axis = st["target"], st["axis"]
    numeric = axis in core.NUMERIC_AXES
    key = f"{target}/{axis}" if f"{target}/{axis}" in METRICS else (target if target in METRICS else "component")
    mets = METRICS[key]
    good = [v for v in variants if v["verified"] and v["metrics"]]
    base = core.BASELINE.get((target, axis))
    tiles = []
    for mk, label, kind, higher in mets[:3]:
        pts = [(v["value"], v["metrics"].get(mk)) for v in good if v["metrics"].get(mk) is not None]
        if not pts:
            continue
        best = (max if higher else min)(pts, key=lambda p: p[1])
        b = next((p for p in pts if p[0] == base), None)
        fmt = (lambda x: f"{x / 1e9:.2f} GB") if kind == "bytes" else (lambda x: f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}")
        tiles.append({"label": f"Best {label}", "value": fmt(best[1]),
                      "sub": f"at {axis} = {best[0]}" + (f"; Qualcomm's setting ({base}): {fmt(b[1])}" if b else "")})
    vrows = [{"value": v["value"], "status": v["status"] or "–",
              "verified": "yes" if v["verified"] else "NO", "checks": v["n_checks"], "qairt": v["qairt"] or "–",
              "problem": v["error"] or "; ".join(f"{c['what']}: expected {c['expected']}, got {c['actual']}"
                                                 for c in v["failed_checks"][:3]) or "–",
              "href": v["report"], "link": "report" if v["report"] else "–"} for v in variants]
    sections: list[dict[str, Any]] = [{
        "title": "Variants and verification",
        "why": "Each value was built, compiled and profiled on the device. Verification reads back from AI Hub what "
               "actually ran (graph names, input shapes, quantization dtypes, the model the compiler consumed, QAIRT "
               "version, device) and compares it with the requested setting. Only verified variants enter the charts.",
        "blocks": [{"type": "note", "text": "Fixed settings: " + (", ".join(f"{k} = {v}" for k, v in (st.get("fixed") or {}).items())
                                                                  or "none (Qualcomm's defaults)") + f"; device: {st.get('device')}.",
                    "table": {"open": True, "cols": [
                        {"key": "value", "label": AXIS_LABEL.get(axis, axis), "kind": "str"},
                        {"key": "status", "label": "Status", "kind": "str"},
                        {"key": "verified", "label": "Verified", "kind": "str"},
                        {"key": "checks", "label": "Checks", "kind": "str"},
                        {"key": "qairt", "label": "QAIRT", "kind": "str"},
                        {"key": "problem", "label": "Problem", "kind": "str"},
                        {"key": "link", "label": "Hardware report", "kind": "str", "link": "href"}], "rows": vrows}}]}]
    for mk, label, kind, _ in mets:
        pts = [(v["value"], v["metrics"].get(mk)) for v in good if v["metrics"].get(mk) is not None]
        if not pts:
            continue
        if numeric:
            block = {"type": "xy", "series": [{"name": label, "points": [[int(x), y] for x, y in pts]}],
                     "kind": kind, "xlabel": AXIS_LABEL.get(axis, axis), "xlog": True, "title": label}
        else:
            block = {"type": "hbar", "rows": [{"label": x, "value": y} for x, y in pts], "kind": kind, "title": label}
        sections.append({"title": label, "why": "", "blocks": [block]})
    part_keys = sorted({k for v in good for k in v["metrics"] if k.endswith(" ms") and "part" in k})
    if part_keys and numeric:
        sections.append({"title": "Per part (AI Hub ms)", "why": "Each compiled graph separately: which part the "
                         "setting moves.", "blocks": [{"type": "xy", "kind": "num", "xlabel": AXIS_LABEL.get(axis, axis),
                                                       "xlog": True, "title": "per part",
                                                       "series": [{"name": pk, "points": [[int(v["value"]), v["metrics"][pk]]
                                                                                          for v in good if v["metrics"].get(pk) is not None]}
                                                                  for pk in part_keys]}]})
    table_cols = [{"key": "value", "label": AXIS_LABEL.get(axis, axis), "kind": "str"},
                  {"key": "verified", "label": "Verified", "kind": "str"}] + [
        {"key": mk, "label": label, "kind": kind} for mk, label, kind, _ in mets] + [
        {"key": pk, "label": pk, "kind": "num"} for pk in part_keys]
    sections.append({"title": "All metrics", "why": "Also in comparison.csv / comparison.json.", "blocks": [
        {"type": "note", "text": "", "table": {"open": True, "cols": table_cols, "rows": [
            {"value": v["value"], "verified": "yes" if v["verified"] else "NO", **v["metrics"]} for v in variants]}}]})
    return {"title": f"Study: {target} · {axis}",
            "subtitle": f"values {', '.join(v['value'] for v in variants)} on {st.get('device')}; "
                        f"{len(good)} of {len(variants)} verified",
            "tiles": tiles, "sections": sections,
            "footer": "Generated by hexprof study compare. Per-variant hardware reports are linked in the table."}
