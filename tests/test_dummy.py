"""hexprof dummy: shrunk models with random weights (no torch needed: export and checkpoint are faked)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hexprof.dummy import spec as S
from tests.dummyhub import DummyHub

E4B_TYPES = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(42)]
E4B_TYPES[-1] = "full_attention"
E4B = {  # the fields of google/gemma-4-E4B-it's config.json that the knobs read
    "model_type": "gemma4",
    "text_config": {"model_type": "gemma4_text", "hidden_size": 2560, "intermediate_size": 10240, "num_hidden_layers": 42,
                    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256, "global_head_dim": 512,
                    "sliding_window": 512, "layer_types": E4B_TYPES, "num_kv_shared_layers": 18,
                    "hidden_size_per_layer_input": 256, "vocab_size": 262144, "vocab_size_per_layer_input": 262144},
    "vision_config": {"model_type": "gemma4_vision", "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 16,
                      "num_attention_heads": 12, "num_key_value_heads": 12, "head_dim": 64, "patch_size": 16,
                      "pooling_kernel_size": 3, "default_output_length": 280},
}
TINY = {"layers": 8, "hidden": 256, "intermediate": 512, "heads": 4, "kv-heads": 1, "head-dim": 64,
        "global-head-dim": 128, "vocab": 4096, "ple-dim": 64}


def test_full_size_request_reproduces_qualcomms_layout():
    r = S.resolve("gemma4-text", E4B, {})
    assert not r.errors and r.dims == {**r.real} and r.deploy["splits"] == [24, 33]
    exp = S.expected_io(r)
    assert exp["n_parts"] == 4
    assert "part1_of_4/prompt_ar128_cl4096_1_of_4" in exp["graphs"] and "part4_of_4/token_ar1_cl4096_4_of_4" in exp["graphs"]
    kv = S.gemma_kv_inputs(r.dims, 1, 4096)
    assert len(kv) == 24 * 2 * 2  # non-shared layers x KV heads x (key, value)
    assert kv["swa_key_0_h0_in"] == [1, 1, 256, 512] and kv["past_value_5_h1_in"] == [1, 1, 4095, 512]


def test_shrinking_depth_keeps_the_kv_shared_fraction_and_derives_layout():
    r = S.resolve("gemma4-text", E4B, {"layers": 12})
    assert not r.errors
    assert r.dims["kv-shared-layers"] == 5 and r.config["text_config"]["layer_types"][5] == "full_attention"
    assert r.config["text_config"]["layer_types"][-1] == "full_attention"
    assert r.deploy["splits"] == [7, 10] and any("kv-shared" in n for n in r.notes)
    assert r.config["text_config"]["vocab_size_per_layer_input"] <= 1024  # host-side table, never on the NPU


@pytest.mark.parametrize("dims,deploy,msg", [
    ({"heads": 6, "kv-heads": 4}, {}, "multiple of kv-heads"),
    ({"layers": 6, "kv-shared-layers": 3}, {}, "no earlier non-shared layer"),
    ({"bogus": 3}, {}, "no knob"),
    ({}, {"prompt_length": 4096}, "prompt-length"),
    ({}, {"splits": [30, 10]}, "splits"),
    ({"head-dim": 63}, {}, "even"),
    ({}, {"precision": "w2a16"}, "precision"),  # no 2-bit on AI Hub quantize jobs
])
def test_invalid_requests_are_rejected(dims, deploy, msg):
    r = S.resolve("gemma4-text", E4B, dims, deploy)
    assert any(msg in e for e in r.errors), r.errors


def test_component_knobs_and_warnings():
    from hexprof.dummy.checkpoint import siglip2_builtin

    r = S.resolve("siglip2-image", siglip2_builtin(), {"hidden": 100, "heads": 4, "image-size": 160}, {"batch": 2})
    assert not r.errors and any("multiple of 32" in w for w in r.warnings)
    assert S.expected_io(r)["inputs"] == {"image": [2, 3, 160, 160]}
    assert any("multiple of heads" in e for e in S.resolve("siglip2-image", siglip2_builtin(), {"hidden": 100}).errors)
    t = S.resolve("siglip2-text", siglip2_builtin(), {"hidden": 256, "seq-len": 32})
    assert S.expected_io(t) == {"inputs": {"input_ids": [1, 32]}, "output_last_dim": 256, "batch": 1}
    v = S.resolve("gemma4-vision", E4B, {"soft-tokens": 70, "hidden": 384, "heads": 6, "kv-heads": 6})
    assert not v.errors and S.expected_io(v)["inputs"]["pixel_values"] == [1, 630, 768]
    assert any("soft-tokens" in e for e in S.resolve("gemma4-vision", E4B, {"soft-tokens": 100}).errors)


def test_cli_plan(tmp_path, capsys):
    from hexprof.cli import main

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(E4B))
    assert main(["dummy", "plan", "gemma4-text", "--base-config", str(cfg), "--dims", "layers=12", "hidden=1024"]) == 0
    out = capsys.readouterr().out
    assert "hidden" in out and "1024" in out and "graphs:" in out
    assert main(["dummy", "plan", "gemma4-text", "--base-config", str(cfg), "--dims", "heads=6", "kv-heads=4"]) == 1
    assert main(["dummy", "knobs", "siglip2-text"]) == 0


# --------------------------------------------------------------------------- the run, with export and AI Hub faked


def fake_export(r: S.Resolved, ckpt: Path, out: Path, qairt: str | None = None) -> dict:
    """parts.json as hexprof.dummy.gemma4_text.export writes it, built the way Qualcomm's split names things."""
    d, dep = r.dims, r.deploy
    parts = {}
    types = S.layer_types(d["layers"], d["pattern"])
    first_shared = d["layers"] - d["kv-shared-layers"]
    metas = S.gemma_parts(d["layers"], dep["splits"])
    for meta in metas:
        comp, i = meta["component"], meta["part_id"]
        graphs = {}
        for g, (ar, ctx) in S.gemma_graph_names(dep, len(metas), i).items():
            fixed = S.gemma_fixed_inputs(d, ar, ctx)
            kv = S.gemma_kv_inputs(d, ar, ctx)
            if meta["layers"] is None:
                inputs = {"add_9": [[1, ar, d["hidden"]], "float32"]}
            else:
                lo, hi = meta["layers"]
                inputs = {k: [v, "float32"] for k, v in fixed.items() if i == 1 or k != "inputs_embeds"}
                if i > 1:
                    inputs["add_1"] = [[1, ar, d["hidden"]], "float32"]
                for name, shp in kv.items():
                    layer = int(name.split("_")[2])
                    shared_src = any(types[j] == types[layer] for j in range(max(lo, first_shared), hi)) and layer < first_shared
                    if lo <= layer < hi or shared_src:
                        inputs[name] = [shp, "float32"]
            graphs[g] = {"inputs": inputs, "ar": ar, "ctx": ctx,
                         "compile_options": f"--target_runtime qnn_dlc --qairt_version 2.50 --quantize_io "
                                            f"--qnn_options context_enable_graphs={g}",
                         "profile_options": f"--qairt_version 2.50 --qnn_options context_enable_graphs={g}"}
        onnx_dir = out / "onnx" / f"{comp}.onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        parts[comp] = {"part_id": i, "layers": meta["layers"], "onnx": str(onnx_dir), "graphs": graphs,
                       "outputs": ["logits"] if meta["layers"] is None else ["add_9"],
                       "link_options": " --qairt_version 2.50",
                       "facts": {"decoder_layers": list(range(*meta["layers"])) if meta["layers"] else []}}
    res = {"resolved": r.to_json(), "qairt": qairt, "parts": parts}
    (out / "onnx" / "parts.json").write_text(json.dumps(res))
    return res


def fake_make(r: S.Resolved, out_dir, seed=0, base_source="", dtype="bfloat16") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(r.config))
    info = {"resolved": r.to_json(), "seed": seed, "dtype": dtype, "parameters": 123}
    (out / "dummy.json").write_text(json.dumps(info))
    return info


def _outputs(vocab: int):
    def fn(path, specs):
        ar = next(iter(specs.values()))[0][1]
        return {"logits": [1, ar, vocab]} if path and "part4" in str(path) else {"add_9": [1, ar, 256]}
    return fn


@pytest.fixture
def faked(monkeypatch, tmp_path):
    from hexprof.dummy import checkpoint as C
    from hexprof.dummy import gemma4_text as G

    cfg = tmp_path / "base.json"
    cfg.write_text(json.dumps(E4B))
    monkeypatch.setattr(G, "export", fake_export)
    monkeypatch.setattr(C, "make", fake_make)
    return cfg


def _run(tmp_path, cfg, name="run", **hub_kw):
    from hexprof.dummy import run as R

    return R.run("gemma4-text", dict(TINY), {"context_length": 1024, "prompt_length": 64}, tmp_path / name,
                 "Samsung Galaxy S25", base=str(cfg), report=True)


def test_dummy_text_run_verifies_and_reports(tmp_path, monkeypatch, faked):
    hub = DummyHub(_outputs(4096)).install(monkeypatch)
    ver = _run(tmp_path, faked)
    assert ver["ok"], [c for c in ver["checks"] if not c["ok"]]
    kinds = [c[0] for c in hub.calls]
    assert kinds.count("quantize") == 4 and kinds.count("compile") == 8 and kinds.count("link") == 4
    assert kinds.count("inference") == 8
    summ = json.loads((tmp_path / "run" / "export_profiles.json").read_text())
    assert len(summ["parts"]) == 8 and "part1_of_4/prompt_ar64_cl1024_1_of_4" in summ["parts"]
    page = (tmp_path / "run" / "hw_report" / "index.html").read_text()
    assert "What was deployed" in page and "SHRUNK DUMMY" in page  # from run_manifest.json
    man = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert man["status"] == "succeeded" and man["results"]["verification"].startswith("VERIFIED")
    changed = {x["setting"] for x in man["settings"] if x["changed"]}
    assert {"hidden", "layers", "context length", "quantization"} <= changed
    what = {c["what"] for c in ver["checks"]}
    assert any("KV-cache inputs" in w for w in what) and any("logits" in w for w in what)


def test_dummy_text_lying_compiler_is_caught(tmp_path, monkeypatch, faked):
    DummyHub(_outputs(4096), lie="hidden").install(monkeypatch)
    ver = _run(tmp_path, faked)
    bad = [c for c in ver["checks"] if not c["ok"]]
    assert not ver["ok"] and any("inputs_embeds" in str(c["actual"]) for c in bad)


def test_dummy_text_falls_back_to_per_graph_quantization(tmp_path, monkeypatch, faked):
    hub = DummyHub(_outputs(4096), fail_shared_token_compile=True).install(monkeypatch)
    ver = _run(tmp_path, faked)
    assert ver["ok"], [c for c in ver["checks"] if not c["ok"]]
    assert [c[0] for c in hub.calls].count("quantize") == 8  # 4 shared + 4 at the token graphs' own shapes
    jobs = json.loads((tmp_path / "run" / "jobs.json").read_text())
    assert all(jobs["parts"][p]["graphs"][g].get("quantize_job") for p in jobs["parts"]
               for g in jobs["parts"][p]["graphs"] if g.startswith("token"))


def test_dummy_study_sweeps_layers(tmp_path, monkeypatch, faked):
    from hexprof.study import run as SR

    DummyHub(_outputs(4096)).install(monkeypatch)
    dims = [f"{k}={v}" for k, v in TINY.items() if k != "layers"]
    fixed = {"dims": dims, "base_config": str(faked), "context_length": 1024, "prompt_length": 64}
    out = SR.run("dummy-gemma4-text", "layers", ["6", "8"], fixed, str(tmp_path / "study"))
    st = json.loads((out / "study.json").read_text())
    assert st["baseline"] == "42"  # the real depth: --dims does not set layers
    assert all(v["verified"] for v in st["variants"].values()), st["variants"]
    ver = json.loads((out / "layers-6" / "verification.json").read_text())
    assert any(c["what"] == "layers of the deployed model" and c["ok"] for c in ver["checks"])
    assert (out / "comparison" / "index.html").exists()


def test_gemma4_vision_soft_token_axis():
    from hexprof.study import core

    assert core.validate("gemma4-vision", "soft-tokens", ["70", "280"]) == ["70", "280"]
    with pytest.raises(ValueError):
        core.validate("gemma4-vision", "soft-tokens", ["300"])
    with pytest.raises(ValueError):
        core.validate("gemma4-vision", "image-size", ["448"])  # the processor resizes: not a graph setting
    assert core.validate("dummy-gemma4-text", "kv-shared-layers", ["0", "6"]) == ["0", "6"]


def test_analytic_counts_follow_a_dummy_checkpoint(tmp_path):
    from hexprof.models import gemma4_efficiency as ge

    r = S.resolve("gemma4-text", E4B, dict(TINY), {"precision": "w8a16"})
    (tmp_path / "config.json").write_text(json.dumps(r.config))
    (tmp_path / "dummy.json").write_text(json.dumps({"resolved": r.to_json()}))
    c = ge.TextConfig.load(tmp_path / "config.json")
    assert c.splits == tuple(r.deploy["splits"]) and c.uniform_bits == 8
    head = ge.part_counts(c, 4, 4, 1, 1024)
    assert head["layers"] == "lm_head" and head["params"] == 256 * 4096
