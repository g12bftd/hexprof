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


NAMES = {"gemma4-text": "Gemma 4 E4B text decoder", "gemma4-vision": "Gemma 4 E4B vision encoder",
         "siglip2-image": "SigLIP 2 image encoder", "siglip2-text": "SigLIP 2 text encoder"}
QUALCOMM_CODE = {"gemma4-text": "Qualcomm's Gemma4_E4B_PreSplit / PartBase (export + split), subclassed with these sizes",
                 "gemma4-vision": "Qualcomm's gemma_4_e4b_it VisionEncoder.from_pretrained",
                 "siglip2-image": "Qualcomm's SigLIP2ImageEncoder.from_pretrained",
                 "siglip2-text": "Qualcomm's SigLIP2TextEncoder.from_pretrained"}


def run_manifest(r: S.Resolved, src: str, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """(model, settings against the real model / Qualcomm, other facts) for the run log of a dummy."""
    from hexprof import runlog as RL

    t = r.target
    st = [RL.setting(k, r.dims.get(k), r.real.get(k), group="architecture") for k in S.knobs(t) if k in r.dims]
    for k, v in r.deploy.items():
        if k in ("image_size", "seq_len", "soft_tokens"):  # the same value as a knob listed above
            continue
        base = S.DEPLOY[t].get(k)
        if k == "splits" and r.real:
            base = S.default_splits(r.real["layers"], r.real.get("kv-shared-layers", 0))
        st.append(RL.setting(k.replace("_", " "), v, base, group="deployment"))
    if t == "gemma4-text":
        st.append(RL.setting("quantization", "AI Hub quantize job (one weight dtype, random calibration data)",
                             "AIMET recipe (int4 + int8 down_proj / lm_head)", group="deployment",
                             note="Qualcomm's AIMET calibration needs an x86 GPU"))
    model = {"name": f"{NAMES[t]} (SHRUNK DUMMY)", "weights": f"random dummy weights (seed {seed})",
             "source": f"{QUALCOMM_CODE[t]}; base config {src}",
             "architecture": {k: v for k, v in r.dims.items()}}
    extra = {"notes": r.notes, "warnings": r.warnings} if (r.notes or r.warnings) else {}
    return model, st, extra


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
        n_calib: int | None = None, compute_json: str | None = None, bandwidth_json: str | None = None,
        reference: str | None = None) -> dict[str, Any]:
    from hexprof.dummy import checkpoint as C

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    r, src = plan(target, dims, deploy, base)
    print(describe(r, src), flush=True)
    if r.errors:
        raise ValueError("; ".join(r.errors))
    ckpt = out / "checkpoint"
    from hexprof import runlog

    model, settings, extra = run_manifest(r, src, seed)
    with runlog.run(out, kind="dummy", title=f"{NAMES[target]}: shrunk dummy on AI Hub", model=model,
                    settings=settings, device=device, qairt=qairt,
                    runtime="QNN context binary (compile to qnn_dlc, link" + (" prompt + token graphs per part)"
                                                                             if target == "gemma4-text" else ")"),
                    extra=extra) as rl:
        info = C.make(r, ckpt, seed=seed, base_source=src)
        rl.add(model={"parameters": f"{info.get('parameters', 0) / 1e6:.1f} M in the checkpoint"})
        write_json(out / "dummy_run.json", {"target": target, "resolved": r.to_json(), "base_config": src, "seed": seed,
                                            "checkpoint": str(ckpt), "parameters": info.get("parameters"),
                                            "qairt": qairt, "device": device, "reference": reference,
                                            "expected": S.expected_io(r),
                                            "param_estimate": S.param_estimate(r)})
        if target == "gemma4-text":
            from hexprof.dummy import gemma4_text as G
            from hexprof.pipeline.gemma4_export import latency_results

            parts = G.export(r, ckpt, out, qairt)
            summary = G.Deploy(out, parts["parts"], str(r.deploy["precision"]), device, n_calib or 8, hw).run()
            rl.result(**latency_results(summary))
            ver = G.verify(out, device)
        else:
            from hexprof.study import components as SC

            fixed = component_fixed(r, ckpt, qairt, n_calib)
            ro = reference_options(target, r, qairt, reference)
            if ro:
                fixed["reference_options"] = ro
                rl.add(extra={"job options": f"copied from your real export's jobs {ro['jobs']} "
                                             f"(QAIRT {ro.get('qairt')} pinned)"})
            SC.run_one(target, "dummy", "-", out, fixed, device)
            ver = verify_component(out, device, reference)
        write_json(out / "verification.json", ver)
        _print_verification(ver)
        bad = [c["what"] for c in ver.get("checks", []) if not c["ok"]]
        rl.result(verification=("VERIFIED" if ver.get("ok") else "NOT VERIFIED") + f" ({len(ver.get('checks', []))} checks"
                  + (f"; failed: {'; '.join(bad[:5])}" if bad else "") + ")")
    if report:
        from hexprof.hwreport import report as R

        try:
            R.build([out], out / "hw_report", title=f"dummy {target}: " + (", ".join(f"{k}={v}" for k, v in dims.items())
                                                                          or "full size"),
                    compute_json=compute_json, bandwidth_json=bandwidth_json, gemma_config=str(ckpt / "config.json"))
        except FileNotFoundError as e:  # e.g. --no-hw, or AI Hub produced no QHAS / optrace
            print(f"no hardware report: {e}")
    return ver


def verify(out: str | Path, device: str | None = None, reference: str | None = None) -> dict[str, Any]:
    out = Path(out)
    meta = read_json(out / "dummy_run.json")
    dev = device or meta.get("device")
    if meta["target"] == "gemma4-text":
        from hexprof.dummy import gemma4_text as G

        ver = G.verify(out, dev)
    else:
        ver = verify_component(out, dev, reference or meta.get("reference"))
    write_json(out / "verification.json", ver)
    _print_verification(ver)
    return ver


def _same(a: list[int] | None, b: list[int] | None) -> bool:
    return a is not None and b is not None and sorted(a) == sorted(b)


R_COMPONENT = {"gemma4-vision": "vision_encoder"}  # components with a Qualcomm export to compare against


def reference_options(target: str, r: S.Resolved, qairt: str | None, reference: str | None) -> dict[str, Any] | None:
    """The real export's job options to deploy this dummy with (None: Qualcomm's option builders are used)."""
    if target not in R_COMPONENT or qairt or r.deploy.get("precision") != S.DEPLOY[target]["precision"]:
        return None
    from hexprof.dummy import reference as REF

    ref_dir = reference or REF.default_reference()
    if ref_dir is None:
        print("    no real export to copy job options from: Qualcomm's option builders are used")
        return None
    try:
        ro = REF.reference_options(ref_dir, R_COMPONENT[target])
    except Exception as e:  # noqa: BLE001 - e.g. no AI Hub access to the reference jobs
        print(f"    could not read the reference jobs in {ref_dir}: {type(e).__name__}: {e}")
        return None
    if ro:
        print(f"    job options copied from your real export ({ref_dir}): compile '{ro['compile']}', QAIRT {ro['qairt']}")
    return ro


def verify_component(out: Path, device: str, reference: str | Path | None = None) -> dict[str, Any]:
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
    notes: list[str] = []
    if t == "gemma4-vision":
        proc = var.get("processor") or {}
        core.check(checks, "image processor token budget", dep.get("soft_tokens"), proc.get("max_soft_tokens"),
                   proc.get("max_soft_tokens") == dep.get("soft_tokens"))
        notes.append(f"image processor: {proc.get('image_processor_source')}")
    if t in R_COMPONENT:
        from hexprof.dummy import reference as REF

        ref_dir = reference or REF.default_reference()
        if ref_dir is None:
            notes.append("no Qualcomm export to compare with (run slurm/gemma4_profile.sh, or pass --reference)")
        else:
            ref = REF.reference_jobs(ref_dir, R_COMPONENT[t])
            if ref is None:
                notes.append(f"{ref_dir} has no {R_COMPONENT[t]} profile job to compare with")
            elif dep.get("precision") != S.DEPLOY[t]["precision"] or meta.get("qairt"):
                notes.append("precision or QAIRT changed on purpose: options not compared with Qualcomm's export")
            else:
                geometry = all(d.get(k) == (r.get("real") or {}).get(k) for k in ("patch-size", "pooling-kernel", "soft-tokens"))
                mine = {"compile": hc.get_job(jobs["compile_job"]), "link": hc.get_job(jobs["link_job"]) if jobs.get("link_job") else None,
                        "profile": hc.get_job(jobs["profile_job"])}
                REF.compare(checks, mine, ref, ref_dir, same_inputs=geometry)
    return core.summarize(checks, qairt=base.get("qairt"), dims=d, deploy=dep, notes=notes)


def _print_verification(ver: dict[str, Any]) -> None:
    bad = [c for c in ver.get("checks", []) if not c["ok"]]
    print(f"\nverification: {'VERIFIED' if ver.get('ok') else 'NOT VERIFIED'} ({len(ver.get('checks', []))} checks"
          + (f", {len(bad)} failed" if bad else "") + ")", flush=True)
    for c in bad[:12]:
        print(f"  FAILED {c['what']}: expected {c['expected']}, got {c['actual']}")
    for n in ver.get("notes") or []:
        print(f"  note: {n}")
