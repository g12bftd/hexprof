"""``hexprof dummy plan | run | verify``: one shrunk model, from knobs to a verified hardware report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hexprof.dummy import spec as S
from hexprof.io_utils import read_json, write_json


def plan(target: str, dims: dict[str, int], deploy: dict[str, Any], base: str | None = None) -> tuple[S.Resolved, str]:
    from hexprof.dummy import checkpoint as C

    cfg, src = C.base_config(S.FAMILY[target], base)
    return S.resolve(target, cfg, dims, deploy), src


def describe(r: S.Resolved, src: str) -> str:
    real = r.real
    lines = [f"{r.target}: base config {src}", "", f"{'knob':<18} {'real':>10} {'dummy':>10}"]
    for k in S.knobs(r.target):
        a, b = real.get(k), r.dims.get(k)
        lines.append(f"{k:<18} {a if a is not None else '-':>10} {b if b is not None else '-':>10}" + ("  *" if a != b else ""))
    lines += ["", "deployment: " + ", ".join(f"{k}={v}" for k, v in r.deploy.items())]
    est = S.param_estimate(r)
    if est:
        lines.append("matrix weights (estimate): " + ", ".join(f"{k} {v / 1e6:.1f} M" for k, v in est.items()))
    exp = S.expected_io(r)
    if r.target == "gemma4-text":
        lines.append(f"parts: {exp['n_parts']} ({', '.join(str(p['layers'] or 'lm_head') for p in S.gemma_parts(r.dims['layers'], r.deploy['splits']))})")
        lines.append("graphs: " + ", ".join(exp["graphs"]))
        kv = S.gemma_kv_inputs(r.dims, 1, int(r.deploy["context_length"]))
        lines.append(f"KV-cache inputs per token graph: {len(kv)} tensors")
    else:
        lines.append(f"inputs: {exp.get('inputs')}; output width {exp.get('output_last_dim')}")
    for n in r.notes:
        lines.append(f"note: {n}")
    for w in r.warnings:
        lines.append(f"warning: {w}")
    for e in r.errors:
        lines.append(f"ERROR: {e}")
    return "\n".join(lines)


def component_fixed(r: S.Resolved, ckpt: Path, qairt: str | None, n_calib: int | None) -> dict[str, Any]:
    """The settings the component runner (hexprof.study.components) needs for this dummy."""
    from hexprof.dummy import checkpoint as C

    dep, d = r.deploy, r.dims
    fixed: dict[str, Any] = {"checkpoint_dir": str(ckpt), "precision": dep.get("precision"), "qairt": qairt,
                             "calibration_samples": n_calib}
    if r.target in ("siglip2-image", "siglip2-text"):
        fixed["batch"] = int(dep.get("batch", 1))
    if r.target == "siglip2-image":
        fixed["image_size"] = d["image-size"]
    if r.target == "siglip2-text":
        fixed["seq_len"] = int(dep.get("seq_len") or d["seq-len"])
    if r.target == "gemma4-vision":
        fixed["processor"] = C.processor_overrides(r)
        fixed["soft_tokens"] = int(dep.get("soft_tokens") or 280)
        fixed["vision_hidden"] = d["hidden"]
    return {k: v for k, v in fixed.items() if v is not None}


def run(target: str, dims: dict[str, int], deploy: dict[str, Any], out: str | Path, device: str,
        base: str | None = None, seed: int = 0, qairt: str | None = None, hw: bool = True, report: bool = True,
        n_calib: int | None = None, compute_json: str | None = None, bandwidth_json: str | None = None) -> dict[str, Any]:
    from hexprof.dummy import checkpoint as C

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    r, src = plan(target, dims, deploy, base)
    print(describe(r, src), flush=True)
    if r.errors:
        raise ValueError("; ".join(r.errors))
    ckpt = out / "checkpoint"
    info = C.make(r, ckpt, seed=seed, base_source=src)
    write_json(out / "dummy_run.json", {"target": target, "resolved": r.to_json(), "base_config": src, "seed": seed,
                                        "checkpoint": str(ckpt), "parameters": info.get("parameters"),
                                        "qairt": qairt, "device": device, "expected": S.expected_io(r),
                                        "param_estimate": S.param_estimate(r)})
    if target == "gemma4-text":
        from hexprof.dummy import gemma4_text as G

        parts = G.export(r, ckpt, out, qairt)
        G.Deploy(out, parts["parts"], str(r.deploy["precision"]), device, n_calib or 8, hw).run()
        ver = G.verify(out, device)
    else:
        from hexprof.study import components as SC

        fixed = component_fixed(r, ckpt, qairt, n_calib)
        SC.run_one(target, "dummy", "-", out, fixed, device)
        ver = verify_component(out, device)
    write_json(out / "verification.json", ver)
    _print_verification(ver)
    if report:
        from hexprof.hwreport import report as R

        try:
            R.build([out], out / "hw_report", title=f"dummy {target}: " + (", ".join(f"{k}={v}" for k, v in dims.items())
                                                                          or "full size"),
                    compute_json=compute_json, bandwidth_json=bandwidth_json, gemma_config=str(ckpt / "config.json"))
        except FileNotFoundError as e:  # e.g. --no-hw, or AI Hub produced no QHAS / optrace
            print(f"no hardware report: {e}")
    return ver


def verify(out: str | Path, device: str | None = None) -> dict[str, Any]:
    out = Path(out)
    meta = read_json(out / "dummy_run.json")
    dev = device or meta.get("device")
    if meta["target"] == "gemma4-text":
        from hexprof.dummy import gemma4_text as G

        ver = G.verify(out, dev)
    else:
        ver = verify_component(out, dev)
    write_json(out / "verification.json", ver)
    _print_verification(ver)
    return ver


def _same(a: list[int] | None, b: list[int] | None) -> bool:
    return a is not None and b is not None and sorted(a) == sorted(b)


def verify_component(out: Path, device: str) -> dict[str, Any]:
    """The component runner's checks (job chain, device, precision, QAIRT) + the requested dimensions."""
    from hexprof.aihub import client as hc
    from hexprof.study import components as SC
    from hexprof.study import core

    meta = read_json(out / "dummy_run.json")
    r = meta["resolved"]
    t, d, dep = r["target"], r["dims"], r["deploy"]
    fixed = component_fixed(S.Resolved(t, {}, d, r["requested"], dep), Path(meta["checkpoint"]), meta.get("qairt"), None)
    base = SC.verify(t, "dummy", "-", out, fixed, device)
    checks = list(base.get("checks", []))
    exp = S.expected_io(r)
    var = read_json(out / "variant.json")
    jobs = read_json(out / "jobs.json")
    compiled = hc.target_model(hc.get_job(jobs["compile_job"]))
    built = core.shapes_of(getattr(compiled, "input_spec", None))
    for name, want in (exp.get("inputs") or {}).items():
        core.check(checks, f"input {name} (from the requested dimensions)", want, built.get(name), _same(built.get(name), want))
    outs = core.shapes_of(getattr(compiled, "output_spec", None))
    if outs:
        last = [v[-1] for v in outs.values() if v]
        core.check(checks, "output width (from the requested dimensions)", exp["output_last_dim"], last,
                   exp["output_last_dim"] in last)
    loaded = var.get("loaded_config") or {}
    side = "vision_config" if t in ("siglip2-image", "gemma4-vision") else "text_config"
    got = loaded.get(side) or loaded
    pairs = {"layers": "num_hidden_layers", "hidden": "hidden_size", "intermediate": "intermediate_size",
             "heads": "num_attention_heads", "patch-size": "patch_size", "image-size": "image_size", "vocab": "vocab_size",
             "seq-len": "max_position_embeddings"}
    for knob, key in pairs.items():
        if knob in d and key in got:
            g = got[key][0] if isinstance(got[key], list) else got[key]
            core.check(checks, f"Qualcomm's wrapper built {knob}", d[knob], g, int(g) == int(d[knob]))
    if t == "gemma4-vision":
        proc = var.get("processor") or {}
        core.check(checks, "image processor token budget", dep.get("soft_tokens"), proc.get("max_soft_tokens"),
                   proc.get("max_soft_tokens") == dep.get("soft_tokens"))
    return core.summarize(checks, qairt=base.get("qairt"), dims=d, deploy=dep)


def _print_verification(ver: dict[str, Any]) -> None:
    bad = [c for c in ver.get("checks", []) if not c["ok"]]
    print(f"\nverification: {'VERIFIED' if ver.get('ok') else 'NOT VERIFIED'} ({len(ver.get('checks', []))} checks"
          + (f", {len(bad)} failed" if bad else "") + ")", flush=True)
    for c in bad[:12]:
        print(f"  FAILED {c['what']}: expected {c['expected']}, got {c['actual']}")
