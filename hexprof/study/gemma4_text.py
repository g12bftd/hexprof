"""Gemma 4 E4B text-part sweeps through Qualcomm's own export (``qai-hub-models export gemma_4_e4b_it``).

Axes (one full export, compile, link and on-device profile per value)

    context-length  --context-lengths V  --sequence-lengths P 1   KV-cache / attention window
    prompt-length   --sequence-lengths V 1  --context-lengths C   prompt chunk (prefill graph)
    decode-tokens   --sequence-lengths V  --context-lengths C     tokens per decode step: the
                                                                  graph speculative decoding
                                                                  would verify drafts with

Only the text parts are exported (``--components``), the vision encoder is unaffected by
these settings. Before any job is submitted the arguments are parsed by Qualcomm's own
parser; afterwards every profiled graph is checked against the request: graph names (built
by Qualcomm from the requested lengths), the compile job's input shapes behind each linked
graph, the graph selected for profiling, the device and the QAIRT version.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hexprof.io_utils import read_json
from hexprof.study import core

GRAPH = re.compile(r"(prompt|token)_ar(\d+)_cl(\d+)")
TEXT_COMPONENTS = ["part1_of_4", "part2_of_4", "part3_of_4", "part4_of_4"]


def export_flags(axis: str, value: str, fixed: dict[str, Any]) -> list[str]:
    v = str(int(value))
    ctx, prompt = str(fixed.get("context_length", 4096)), str(fixed.get("prompt_length", 128))
    if axis == "context-length":
        flags = ["--context-lengths", v, "--sequence-lengths", prompt, "1"]
    elif axis == "prompt-length":
        flags = ["--sequence-lengths", v, "1", "--context-lengths", ctx]
    elif axis == "decode-tokens":
        flags = ["--sequence-lengths", v, "--context-lengths", ctx]
    else:
        raise ValueError(f"gemma4-text has no axis {axis}")
    comps = fixed.get("components") or TEXT_COMPONENTS
    return flags + ["--components", *comps]


def expected(axis: str, value: str, fixed: dict[str, Any]) -> tuple[set[int], int]:
    """(sequence lengths, context length) the exported graphs must have."""
    v = int(value)
    ctx, prompt = int(fixed.get("context_length", 4096)), int(fixed.get("prompt_length", 128))
    if axis == "context-length":
        return {prompt, 1}, v
    if axis == "prompt-length":
        return {v, 1}, ctx
    return {v}, ctx


def run_one(axis: str, value: str, vdir: Path, fixed: dict[str, Any], device: str) -> dict[str, Any]:
    from hexprof.pipeline import gemma4_export

    flags = export_flags(axis, value, fixed)
    checkpoint = fixed.get("checkpoint", "DEFAULT")
    runtime = fixed.get("target_runtime", "geniex_qairt")
    parsed = gemma4_export.parse_export_args(checkpoint, vdir, device, runtime, "w4a16", flags)
    seqs, ctx = expected(axis, value, fixed)
    got_seq = sorted(int(x) for x in parsed.get("sequence_lengths") or [])
    got_ctx = [int(x) for x in parsed.get("context_lengths") or []]
    if set(got_seq) != seqs or got_ctx != [ctx]:
        raise ValueError(f"Qualcomm's parser read sequence_lengths={got_seq}, context_lengths={got_ctx}; "
                         f"expected {sorted(seqs)} and [{ctx}]")
    print(f"    export flags: {' '.join(flags)}  (parsed: sequence_lengths={got_seq}, context_lengths={got_ctx}, "
          f"components={parsed.get('components')})", flush=True)
    summary = gemma4_export.export(checkpoint, vdir, device, runtime, "w4a16", flags, hw=True)
    return {"flags": flags, "parsed": {k: parsed.get(k) for k in ("sequence_lengths", "context_lengths", "components")},
            "graphs": sorted(summary.get("parts", {}))}


def _shape_evidence(shapes: dict[str, list[int]], ar: int, ctx: int) -> tuple[bool, bool, str]:
    """Do the graph's input shapes carry this sequence length and context length?"""
    ev = []
    seq_ok = False
    for name, d in shapes.items():
        n = name.lower()
        if any(k in n for k in ("embed", "input_ids", "hidden")) and len(d) >= 2 and d[1] == ar:
            seq_ok = True
            ev.append(f"{name}{d}")
    ctx_ok = False
    for name, d in shapes.items():
        n = name.lower()
        if "mask" in n and d and d[-1] == ctx:
            ctx_ok = True
            ev.append(f"{name}{d}")
        elif any(k in n for k in ("past", "key", "value", "kv")) and (ctx - ar in d or ctx in d):
            ctx_ok = True
            ev.append(f"{name}{d}")
    if not seq_ok:  # no recognisable input name: fall back to the sequence dimension of any input
        seq_ok = any(len(d) >= 2 and d[1] == ar for d in shapes.values())
    if not (seq_ok and ctx_ok):  # show the inputs that should carry the lengths, matched or not
        ev = [f"{n}{d}" for n, d in shapes.items()
              if any(k in n.lower() for k in ("embed", "input_ids", "mask", "past", "key", "value", "kv"))][:4] or ev
    return seq_ok, ctx_ok, ", ".join(ev[:4])


def verify(axis: str, value: str, vdir: Path, fixed: dict[str, Any], device: str) -> dict[str, Any]:
    from hexprof.aihub import client as hc
    from hexprof.hwreport import models as MD

    seqs, ctx = expected(axis, value, fixed)
    summ = read_json(vdir / "export_profiles.json")
    checks: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    qairt: set[str] = set()
    for label, rec in sorted(summ.get("parts", {}).items()):
        comp, graph = rec.get("component", label.split("/")[0]), rec.get("graph", label.split("/")[-1])
        m = GRAPH.search(graph)
        ar, cl = (int(m.group(2)), int(m.group(3))) if m else (None, None)
        core.check(checks, f"{label}: graph name", f"ar in {sorted(seqs)}, cl{ctx}", graph,
                   m is not None and ar in seqs and cl == ctx)
        if ar is not None:
            seen.add((comp, ar))
        job_id = (rec.get("job") or {}).get("job_id")
        if not job_id:
            core.check(checks, f"{label}: profile job", "recorded", None, False)
            continue
        job = hc.get_job(job_id)
        core.device_check(checks, job, device, f"{label}: device")
        opts = str(getattr(job, "options", "") or "")
        if "context_enable_graphs" in opts:
            core.check(checks, f"{label}: graph profiled", graph, opts, graph in opts)
        q = core.qairt_of(job)
        if q:
            qairt.add(q)
        src, cjob = MD.source_of(job.model, graph)
        shapes = core.shapes_of(getattr(cjob, "shapes", None)) if cjob is not None else {}
        if shapes and ar is not None:
            seq_ok, ctx_ok, ev = _shape_evidence(shapes, ar, ctx)
            core.check(checks, f"{label}: compiled input shapes", f"sequence {ar}, context {ctx}", ev or str(shapes)[:300],
                       seq_ok and ctx_ok)
        else:
            core.check(checks, f"{label}: compiled input shapes", f"sequence {ar}, context {ctx}",
                       "compile job not found behind the profiled model", False)
    comps = fixed.get("components") or TEXT_COMPONENTS
    for comp in comps:
        for ar in sorted(seqs):
            core.check(checks, f"{comp}: graph with sequence {ar}", "profiled", "yes" if (comp, ar) in seen else "missing",
                       (comp, ar) in seen)
    return core.summarize(checks, qairt=sorted(qairt), expected={"sequence_lengths": sorted(seqs), "context_length": ctx})
