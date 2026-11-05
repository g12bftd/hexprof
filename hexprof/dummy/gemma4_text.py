"""Shrunk Gemma 4 text model on the S25 NPU: Qualcomm's export -> AI Hub -> profile -> verification.

1. ``export`` (CPU, no AIMET, no GPU). Qualcomm's own E4B classes, subclassed with the requested
   dimensions (``Gemma4_E4B_PreSplit`` / ``Gemma4_E4B_PartBase``), load the random checkpoint, apply
   their NPU adaptations (single-head attention as Conv, external embeddings + per-layer inputs,
   per-head KV-cache I/O, separate sliding-window / global masks and RoPE, KV sharing), export the
   dynamic-shape ONNX and split it at the requested layer boundaries (+ the lm_head part). For every
   part, Qualcomm's ``get_graph_input_spec`` gives the static shapes of its prompt graph
   (``prompt_ar{P}_cl{C}``) and token graph (``token_ar1_cl{C}``), and Qualcomm's option builders give
   the compile / link / profile options (QAIRT pin, ``--output_names``, ``--quantize_io``,
   ``context_enable_graphs``).
2. ``deploy`` on AI Hub, per part: upload -> quantize job (Qualcomm calibrates with AIMET on a GPU;
   here an AI Hub quantize job, uniform int4 or int8 weights with int16 activations, on random
   inputs) -> one compile job per graph (qnn_dlc) -> one link job (both graphs in one context
   binary, weights shared, as Qualcomm ships them) -> one profile job per graph with the QHAS /
   optrace files. Results use the layout of ``hexprof pipeline gemma4 export``
   (``export_profiles.json``), so ``hexprof hw report`` reads them unchanged.
3. ``verify``: the shapes AI Hub reports for the binaries that ran are compared with shapes computed
   from the requested dimensions alone (``spec.gemma_fixed_inputs`` / ``spec.gemma_kv_inputs``), plus
   the job chain (uploaded part -> quantized -> compiled -> linked -> profiled), device, precision
   and QAIRT version, and the layer ranges found in each exported part.
"""

from __future__ import annotations

import contextlib
import re
import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from hexprof.dummy import spec as S
from hexprof.io_utils import read_json, write_json

KV = re.compile(r"^(swa|past)_(key|value)_(\d+)_h(\d+)_(in|out)$")
LAYER = re.compile(r"layers\.(\d+)\.")
MASK_MIN = -50.0  # finite "minus infinity", as in hexprof's ceiling graphs


# --------------------------------------------------------------------------- export (Qualcomm's code)


@contextlib.contextmanager
def _tolerant_tokenizer() -> Iterator[None]:
    """Qualcomm's LLM base loads a tokenizer it only uses for demos and calibration text; a random checkpoint has none."""
    import qai_hub_models.models.templates.llm.model as LM  # type: ignore[import-not-found]

    orig = LM.get_tokenizer

    def tolerant(ckpt: Any) -> Any:
        try:
            return orig(ckpt)
        except Exception:  # noqa: BLE001
            return None

    LM.get_tokenizer = tolerant
    try:
        yield
    finally:
        LM.get_tokenizer = orig


def arch_attrs(d: dict[str, int]) -> dict[str, Any]:
    """Class attributes Qualcomm's Gemma 4 PreSplit / Part classes read (same names, shrunk values)."""
    return dict(num_layers=d["layers"], hidden_size=d["hidden"], num_attention_heads=d["heads"],
                num_key_value_heads=d["kv-heads"], head_dim=d["head-dim"], global_head_dim=d["global-head-dim"],
                num_kv_shared_layers=d.get("kv-shared-layers", 0), sliding_window_pattern=d["pattern"],
                sliding_window=d["sliding-window"], hidden_size_per_layer_input=d["ple-dim"])


def trace_context(dep: dict[str, Any], d: dict[str, int]) -> int:
    """Qualcomm's export ties the sliding-window mask width to sequence + sliding_window, so it is traced
    with a context of at least prompt + window; smaller contexts compile from the same dynamic graph."""
    return max(int(dep["context_length"]), int(dep["prompt_length"]) + d["sliding-window"])


def onnx_facts(path: Path) -> dict[str, Any]:
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    # matrix weights only: the exporter merges identical small tensors (e.g. freshly initialised norm
    # weights) and names the shared copy after the first layer that had it
    layers = sorted({int(x) for i in m.graph.initializer if len(i.dims) >= 2 for x in LAYER.findall(i.name)})
    params = sum(int(np.prod(i.dims)) if i.dims else 1 for i in m.graph.initializer)
    ops: dict[str, int] = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return {"decoder_layers": layers, "parameters": params, "nodes": len(m.graph.node),
            "op_types": dict(sorted(ops.items(), key=lambda kv: -kv[1])[:12]),
            "inputs": [i.name for i in m.graph.input], "outputs": [o.name for o in m.graph.output]}


def export(r: S.Resolved, ckpt: Path, out: Path, qairt: str | None = None) -> dict[str, Any]:
    """Qualcomm's export + split of the shrunk model; returns (and caches) ``<out>/onnx/parts.json``."""
    pj = out / "onnx" / "parts.json"
    if pj.exists():
        old = read_json(pj)
        if old.get("resolved") == r.to_json() and old.get("qairt") == qairt and all(
                Path(p["onnx"]).exists() for p in old["parts"].values()):
            print(f"    ONNX parts exist: {pj.parent}")
            return old
    import onnx
    import torch
    from qai_hub_models import Precision, TargetRuntime  # type: ignore[import-not-found]
    from qai_hub_models.models.gemma_4_e4b_it import model as E  # type: ignore[import-not-found]

    d, dep = r.dims, r.deploy
    prompt, ctx = int(dep["prompt_length"]), int(dep["context_length"])
    parts_meta = S.gemma_parts(d["layers"], list(dep["splits"]))
    n_parts = len(parts_meta)
    arch = arch_attrs(d)
    tctx = trace_context(dep, d)
    pre_cls = type("HexprofDummyGemma4_PreSplit", (E.Gemma4_E4B_PreSplit,), {
        **arch, "hf_repo_name": str(ckpt),
        # model_id "" keeps Qualcomm's on-disk ONNX cache (keyed by model id) out of the way, both ways
        "model_id": "", "model_asset_version": 0, "default_checkpoint": {}, "min_memory_recommended": 0,
        "split_model_name": "Gemma4_Dummy", "num_splits": n_parts, "splitting_points": list(dep["splits"]),
        "default_sequence_lengths": [prompt], "default_context_lengths": [tctx]})
    part_base = type("HexprofDummyGemma4_PartBase", (E.Gemma4_E4B_PartBase,), {
        **arch, "num_splits": n_parts, "fp_presplit_cls": pre_cls})
    print(f"    Qualcomm export: {d['layers']} layers ({d.get('kv-shared-layers', 0)} KV-shared), hidden {d['hidden']}, "
          f"split {dep['splits']} + lm_head; traced at seq {prompt}, ctx {tctx}", flush=True)
    with _tolerant_tokenizer():
        pre = pre_cls.from_pretrained(checkpoint=str(ckpt), host_device=torch.device("cpu"))
    pre.sequence_length, pre.context_length = prompt, tctx
    pre.embedding = pre.EmbeddingClass(max_length=tctx, config=pre.llm_config)
    qflag = f"--qairt_version {qairt}" if qairt else ""
    prec = Precision.parse(str(dep["precision"]).replace("fp16", "float"))
    rt = TargetRuntime.QNN_CONTEXT_BINARY
    result: dict[str, Any] = {"resolved": r.to_json(), "qairt": qairt, "trace_context": tctx, "parts": {}}
    for meta in parts_meta:
        i, comp = meta["part_id"], meta["component"]
        bundle = pre.convert_to_onnx_and_split(i)
        dest = out / "onnx" / f"{comp}.onnx"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        model = onnx.load(str(bundle.onnx_graph_path), load_external_data=True)
        # torch's exporter writes IR 11 (onnx 1.18), which ONNX Runtime up to 1.22 refuses; these graphs use
        # nothing beyond IR 10, and AI Hub's quantize job runs ONNX Runtime on them
        model.ir_version = min(model.ir_version, 10)
        onnx.save_model(model, str(dest / "model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                        location="model.data", size_threshold=1024)
        del model
        part = type(f"HexprofDummyGemma4_Part{i}", (part_base,), {"part_id": i})(
            pre, precision=Precision.float, sequence_lengths=[prompt, 1], context_lengths=[ctx])
        graphs: dict[str, Any] = {}
        for g in part.graph_names:
            spec = part.get_graph_input_spec(g)
            graphs[g] = {
                "inputs": {k: [list(int(x) for x in v.shape), str(v.dtype)] for k, v in spec.items()},
                "compile_options": part.get_graph_hub_compile_options(g, rt, prec, qflag),
                "profile_options": part.get_graph_hub_inference_options(g, rt, qflag),
                "ar": part._graph_names[g][0], "ctx": part._graph_names[g][1]}
        result["parts"][comp] = {"part_id": i, "layers": meta["layers"], "onnx": str(dest),
                                 "outputs": part._get_onnx_output_names(), "graphs": graphs,
                                 "link_options": part.get_hub_link_options(rt, qflag),
                                 "facts": onnx_facts(dest / "model.onnx")}
        print(f"      {comp}: layers {meta['layers'] or 'lm_head'}, {result['parts'][comp]['facts']['parameters'] / 1e6:.1f} M "
              f"parameters, graphs {', '.join(graphs)}", flush=True)
    pre.free_memory()
    try:
        from importlib.metadata import version

        result["qai_hub_models"] = version("qai-hub-models")
    except Exception:  # noqa: BLE001
        pass
    write_json(pj, result)
    return result


# --------------------------------------------------------------------------- AI Hub


def calibration(inputs: dict[str, list[Any]], n: int, seed: int = 0) -> dict[str, list[np.ndarray]]:
    """Random but well-formed inputs: real RoPE cos/sin, causal masks, unit-variance activations and KV."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[np.ndarray]] = {}
    for name, (shape, dtype) in inputs.items():
        vals = []
        for _ in range(n):
            if "position_ids_cos" in name or "position_ids_sin" in name:
                pos = rng.integers(0, 4096, size=shape[2])[:, None]
                half = shape[-1]
                ang = pos / (10000 ** (np.arange(half) / half))
                v = np.cos(ang) if name.endswith("cos") else np.sin(ang)
                v = np.broadcast_to(v, shape)
            elif name.endswith("attention_mask"):
                seq, width = shape[2], shape[3]
                v = np.zeros(shape, np.float32)
                for q in range(seq):  # the last `seq` columns are the new tokens: causal among them
                    v[..., q, width - seq + q + 1:] = MASK_MIN
            else:
                v = rng.standard_normal(shape)
            vals.append(np.ascontiguousarray(np.asarray(v, dtype=np.dtype(dtype if dtype != "float" else "float32"))))
        out[name] = vals
    return out


def _input_specs(inputs: dict[str, list[Any]]) -> dict[str, tuple[tuple[int, ...], str]]:
    return {k: (tuple(v[0]), v[1]) for k, v in inputs.items()}


class Deploy:
    """Resumable AI Hub stages; job ids land in ``<out>/jobs.json`` after every submission."""

    def __init__(self, out: Path, parts: dict[str, Any], precision: str, device: str, n_calib: int = 8,
                 hw: bool = True) -> None:
        from hexprof.aihub import client as hc

        self.hc = hc
        self.out, self.parts, self.device, self.n_calib, self.hw = out, parts, device, n_calib, hw
        self.prec = hc.parse_precision(precision)
        self.path = out / "jobs.json"
        self.state: dict[str, Any] = read_json(self.path) if self.path.exists() else {}
        self.state.setdefault("parts", {})

    def _save(self) -> None:
        write_json(self.path, self.state)

    def _st(self, comp: str) -> dict[str, Any]:
        return self.state["parts"].setdefault(comp, {"graphs": {}})

    def _wait(self, jobs: dict[tuple[str, str | None], Any], kind: str) -> None:
        for (comp, g), job in jobs.items():
            try:
                self.hc.wait_success(job, kind)
            except self.hc.HubJobError as e:
                st = self._st(comp) if g is None else self._st(comp)["graphs"].setdefault(g, {})
                st.setdefault("errors", []).append(str(e))
                st[f"failed_{kind}_job"] = st.pop(f"{kind}_job", None)
                print(f"    WARNING: {e}", flush=True)
        self._save()

    def run(self) -> dict[str, Any]:
        hc, dev = self.hc, self.hc.device(self.device)
        name = f"hexprof_dummy_{self.out.name}"
        # upload + quantize (one per part, calibrated at the prompt graph's shapes)
        pending: dict[tuple[str, str | None], Any] = {}
        for comp, p in self.parts.items():
            st = self._st(comp)
            if not st.get("source_model_id"):
                st["source_model_id"] = hc.upload_model(p["onnx"], name=f"{name}_{comp}").model_id
                self._save()
            if not self.prec.is_float and not st.get("quantize_job"):
                prompt_g = max(p["graphs"], key=lambda g: p["graphs"][g]["ar"])
                calib = calibration(p["graphs"][prompt_g]["inputs"], self.n_calib)
                job = hc.submit_quantize(hc.get_model(st["source_model_id"]), calib, self.prec, f"{name}_{comp}")
                st["quantize_job"], st["quantize_calibrated_on"] = job.job_id, prompt_g
                pending[(comp, None)] = job
                self._save()
        self._wait(pending, "quantize")
        # compile every graph of every part
        pending = {}
        for comp, p in self.parts.items():
            st = self._st(comp)
            for g, gi in p["graphs"].items():
                gs = st["graphs"].setdefault(g, {})
                if gs.get("compile_job"):
                    continue
                src = self._compile_source(comp, st, gs, p, g)
                if src is None:
                    continue
                job = hc.submit_compile(src, dev, f"{name}_{g}", options=gi["compile_options"],
                                        input_specs=_input_specs(gi["inputs"]))
                gs["compile_job"], gs["compiled_from"] = job.job_id, src.model_id
                pending[(comp, g)] = job
                self._save()
        self._wait(pending, "compile")
        self._retry_per_graph(dev, name)
        # link the graphs of each part into one context binary
        pending = {}
        for comp, p in self.parts.items():
            st = self._st(comp)
            if st.get("link_job") or any(not st["graphs"].get(g, {}).get("compile_job") for g in p["graphs"]):
                continue
            dlcs = [hc.target_model(hc.get_job(st["graphs"][g]["compile_job"])) for g in p["graphs"]]
            job = hc.submit_link(dlcs, dev, f"{name}_{comp}", options=p["link_options"])
            st["link_job"], st["linked_models"] = job.job_id, [m.model_id for m in dlcs]
            pending[(comp, None)] = job
            self._save()
        self._wait(pending, "link")
        # profile every graph of the linked binary
        pending = {}
        for comp, p in self.parts.items():
            st = self._st(comp)
            if not st.get("link_job"):
                continue
            binary = hc.target_model(hc.get_job(st["link_job"]))
            for g, gi in p["graphs"].items():
                gs = st["graphs"][g]
                if gs.get("profile_job"):
                    continue
                job = hc.submit_inference(binary, dev, None, profile=True, name=f"{name}_{g}", options=gi["profile_options"])
                gs["profile_job"] = job.job_id
                pending[(comp, g)] = job
                self._save()
        self._wait(pending, "profile")
        return self.collect()

    def _compile_source(self, comp: str, st: dict[str, Any], gs: dict[str, Any], p: dict[str, Any], g: str) -> Any:
        hc = self.hc
        if self.prec.is_float:
            return hc.get_model(st["source_model_id"])
        qjob = gs.get("quantize_job") or st.get("quantize_job")
        if not qjob:
            return None
        return hc.target_model(hc.get_job(qjob))

    def _retry_per_graph(self, dev: Any, name: str) -> None:
        """If a graph did not compile from the part's shared quantized model (calibrated at the prompt
        shapes), quantize again with that graph's own shapes and compile from that."""
        hc = self.hc
        if self.prec.is_float:
            return
        pending: dict[tuple[str, str | None], Any] = {}
        for comp, p in self.parts.items():
            st = self._st(comp)
            for g, gi in p["graphs"].items():
                gs = st["graphs"].get(g, {})
                if gs.get("compile_job") or gs.get("quantize_job") or not gs.get("failed_compile_job"):
                    continue
                calib = calibration(gi["inputs"], self.n_calib, seed=1)
                job = hc.submit_quantize(hc.get_model(st["source_model_id"]), calib, self.prec, f"{name}_{g}")
                gs["quantize_job"] = job.job_id
                pending[(comp, g)] = job
                print(f"    {g}: compile from the shared quantized model failed; quantizing with its own shapes", flush=True)
                self._save()
        if not pending:
            return
        self._wait(pending, "quantize")
        cpending: dict[tuple[str, str | None], Any] = {}
        for comp, g in [k for k in pending]:
            gs = self._st(comp)["graphs"][g]
            if not gs.get("quantize_job"):
                continue
            src = hc.target_model(hc.get_job(gs["quantize_job"]))
            gi = self.parts[comp]["graphs"][g]
            job = hc.submit_compile(src, dev, f"{name}_{g}", options=gi["compile_options"],
                                    input_specs=_input_specs(gi["inputs"]))
            gs["compile_job"], gs["compiled_from"] = job.job_id, src.model_id
            cpending[(comp, g)] = job
            self._save()
        self._wait(cpending, "compile")

    def collect(self) -> dict[str, Any]:
        """``export_profiles.json`` + profiles + QHAS / optrace, in the layout of Qualcomm's export."""
        from hexprof.aihub.metrics import aihub_metrics, per_layer

        hc = self.hc
        summary: dict[str, Any] = {"bundle": "", "dummy": True, "parts": {}}
        for comp, p in self.parts.items():
            st = self._st(comp)
            for g, gi in p["graphs"].items():
                gs = st["graphs"].get(g, {})
                if not gs.get("profile_job"):
                    continue
                job = hc.get_job(gs["profile_job"])
                label = f"{comp}/{g}"
                stem = label.replace("/", "__")
                prof, _ = hc.save_profile(job, self.out / "profiles", stem)
                rec: dict[str, Any] = {"component": comp, "graph": g, "job": hc.job_info(job),
                                       "model_id": getattr(getattr(job, "model", None), "model_id", None),
                                       "options": getattr(job, "options", "") or "", "metrics": aihub_metrics(prof),
                                       "qairt": hc.qairt_version(job)}
                write_json(self.out / "profiles" / f"{stem}.per_layer.json", per_layer(prof))
                if self.hw:
                    rec["hw_artifacts"] = hc.download_hw_artifacts(job, self.out / "hw" / stem)
                summary["parts"][label] = rec
                # the local part ONNX, for exact per-op MAC counts in the hardware report
                write_json(self.out / "models" / f"{stem}.model.json", {
                    "label": stem, "job_id": gs["profile_job"], "onnx_path": str(Path(p["onnx"]) / "model.onnx"),
                    "shapes": {k: v[0] for k, v in gi["inputs"].items()},
                    "inputs": {k: {"shape": v[0], "dtype": v[1]} for k, v in gi["inputs"].items()},
                    "outputs": {}, "source": "hexprof dummy export (local)"})
                print(f"    {label}: {rec['metrics']['inference_time_milliseconds']} ms ({job.url})", flush=True)
        write_json(self.out / "export_profiles.json", summary)
        return summary


# --------------------------------------------------------------------------- verification


def _kv_shape_ok(name: str, shape: list[int], d: dict[str, int], ar: int, ctx: int) -> tuple[bool, Any]:
    m = KV.match(name)
    if not m:
        return True, None
    layer = int(m.group(3))
    types = S.layer_types(d["layers"], d["pattern"])
    sliding = types[layer] == "sliding_attention"
    hd = d["head-dim"] if sliding else d["global-head-dim"]
    ln = (min(d["sliding-window"], ctx - ar) if sliding else ctx - ar) if m.group(5) == "in" else ar
    want = [1, 1, hd, ln] if m.group(2) == "key" else [1, 1, ln, hd]
    prefix_ok = (m.group(1) == "swa") == sliding
    return shape == want and prefix_ok and int(m.group(4)) < d["kv-heads"], want


def verify(out: Path, device: str) -> dict[str, Any]:
    from hexprof.aihub import client as hc
    from hexprof.study import core

    meta = read_json(out / "dummy_run.json")
    parts = read_json(out / "onnx" / "parts.json")["parts"]
    jobs = read_json(out / "jobs.json")["parts"]
    r = meta["resolved"]
    d, dep = r["dims"], r["deploy"]
    exp = S.expected_io(r)
    prec = hc.parse_precision(str(dep["precision"]))
    checks: list[dict[str, Any]] = []
    qairt: set[str] = set()
    kv_seen: dict[int, set[str]] = {}
    for label, e in exp["graphs"].items():
        comp, g = label.split("/")
        ar, ctx, pm = e["ar"], e["ctx"], e["part"]
        st, p = jobs.get(comp, {}), parts.get(comp, {})
        gs = st.get("graphs", {}).get(g, {})
        if not gs.get("profile_job"):
            core.check(checks, f"{label}: profiled", "profile job", gs.get("errors") or "missing", False)
            continue
        pjob = hc.get_job(gs["profile_job"])
        core.device_check(checks, pjob, device, f"{label}: device")
        opts = str(getattr(pjob, "options", "") or "")
        core.check(checks, f"{label}: graph selected for profiling", f"context_enable_graphs={g}", opts,
                   f"context_enable_graphs={g}" in opts)
        q = core.qairt_of(pjob)
        if q:
            qairt.add(q)
        cjob = hc.get_job(gs["compile_job"])
        compiled = hc.target_model(cjob)
        shapes = core.shapes_of(getattr(compiled, "input_spec", None), g) or core.shapes_of(getattr(cjob, "shapes", None))
        fixed = S.gemma_fixed_inputs(d, ar, ctx)
        bad = []
        for name, shp in shapes.items():
            if name in fixed and shp != fixed[name]:
                bad.append(f"{name}{shp} != {fixed[name]}")
            ok, want = _kv_shape_ok(name, shp, d, ar, ctx)
            if not ok:
                bad.append(f"{name}{shp} != {want}")
            if KV.match(name) and name.endswith("_in"):
                kv_seen.setdefault(ar, set()).add(name)
            if name.startswith("add") and shp != [1, ar, d["hidden"]] and len(shp) == 3 and shp[1] != 1:
                bad.append(f"hidden state {name}{shp} != [1, {ar}, {d['hidden']}]")
        core.check(checks, f"{label}: compiled input shapes match the requested dimensions",
                   "hidden, layers, PLE, heads, head dims, window, context, sequence", "; ".join(bad) or
                   f"{len(shapes)} inputs match", bool(shapes) and not bad)
        if pm["part_id"] == 1:
            core.check(checks, f"{label}: token embeddings in", fixed["inputs_embeds"], shapes.get("inputs_embeds"),
                       shapes.get("inputs_embeds") == fixed["inputs_embeds"])
        if pm["layers"] is not None:
            core.check(checks, f"{label}: per-layer inputs carry every layer", fixed["per_layer_inputs"],
                       shapes.get("per_layer_inputs"), shapes.get("per_layer_inputs") == fixed["per_layer_inputs"])
        else:
            outs = core.shapes_of(getattr(compiled, "output_spec", None), g)
            logits = next((v for k, v in outs.items() if "logit" in k.lower()), None)
            if outs:
                core.check(checks, f"{label}: logits", [1, ar, d["vocab"]], logits, logits == [1, ar, d["vocab"]])
        # job chain: uploaded part -> (quantized) -> compiled -> linked -> profiled
        cm = getattr(getattr(cjob, "model", None), "model_id", None)
        core.check(checks, f"{label}: compile job consumed", gs.get("compiled_from"), cm, cm == gs.get("compiled_from"))
        binary = hc.target_model(hc.get_job(st["link_job"]))
        core.check(checks, f"{label}: profiled model = linked binary", binary.model_id,
                   getattr(getattr(pjob, "model", None), "model_id", None),
                   getattr(getattr(pjob, "model", None), "model_id", None) == binary.model_id)
        ljob = hc.get_job(st["link_job"])
        linked = [m.model_id for m in (getattr(ljob, "models", None) or [])]
        core.check(checks, f"{label}: link job consumed this graph", compiled.model_id, linked, compiled.model_id in linked)
    for comp, st in jobs.items():
        p = parts.get(comp, {})
        meta_layers = p.get("layers")
        facts = p.get("facts", {})
        if meta_layers is not None:
            want = list(range(*meta_layers))
            core.check(checks, f"{comp}: decoder layers in the exported part", want, facts.get("decoder_layers"),
                       facts.get("decoder_layers") == want)
        if prec.is_float:
            core.check(checks, f"{comp}: float, no quantize job", None, st.get("quantize_job"), not st.get("quantize_job"))
            continue
        qids = [st.get("quantize_job")] + [gs.get("quantize_job") for gs in st.get("graphs", {}).values()]
        for qid in filter(None, qids):
            qj = hc.get_job(qid)
            gw, ga = str(getattr(qj, "weights_dtype", "")).upper(), str(getattr(qj, "activations_dtype", "")).upper()
            core.check(checks, f"{comp}: quantize job dtypes", f"INT{prec.weight_bits} / INT{prec.act_bits}", f"{gw} / {ga}",
                       gw.endswith(f"INT{prec.weight_bits}") and ga.endswith(f"INT{prec.act_bits}"))
            src = getattr(getattr(qj, "model", None), "model_id", None)
            core.check(checks, f"{comp}: quantize job consumed the uploaded part", st.get("source_model_id"), src,
                       src == st.get("source_model_id"))
    want_kv = set(S.gemma_kv_inputs(d, 1, int(dep["context_length"])))
    for ar, seen in kv_seen.items():
        core.check(checks, f"KV-cache inputs, sequence {ar}: one per non-shared layer and KV head",
                   f"{len(want_kv)} tensors", f"{len(seen)} tensors" + (f"; missing {sorted(want_kv - seen)[:4]}" if want_kv - seen else "")
                   + (f"; extra {sorted(seen - want_kv)[:4]}" if seen - want_kv else ""), seen == want_kv)
    return core.summarize(checks, qairt=sorted(qairt), dims=d, deploy=dep)
