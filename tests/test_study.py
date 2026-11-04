import json
import re
import sys
import types
from pathlib import Path

import pytest

import hwfakes
from hexprof.aihub import client as hc
from hexprof.study import core
from hexprof.study import run as S


class Dev:
    def __init__(self, name="Samsung Galaxy S25"):
        self.name = name


class Model:
    def __init__(self, mid, input_spec=None, producer=None, name="m"):
        self.model_id, self.input_spec, self._producer, self.name = mid, input_spec, producer, name

    def get_producer(self):
        return self._producer


class CompileJob:
    def __init__(self, jid, shapes, source=None, options=""):
        self.job_id, self.shapes, self.model, self.options = jid, shapes, source, options
        self.device = Dev()

    def get_target_model(self):
        return None


class Link:
    def __init__(self, models):
        self.models = models


class ProfileJob:
    def __init__(self, jid, model, options="", device="Samsung Galaxy S25"):
        self.job_id, self.model, self.options, self.device = jid, model, options, Dev(device)
        self.url = f"https://hub/{jid}"


def _page(path):
    return json.loads(re.search(r'id="data">(.*?)</script>', Path(path).read_text(), re.S).group(1).replace("<\\/", "</"))


# --------------------------------------------------------------------------- gemma4-text


def test_validation_and_defaults(tmp_path):
    assert core.validate("gemma4-text", "context-length", []) == ["512", "1024", "2048", "4096"]
    with pytest.raises(ValueError):
        core.validate("gemma4-text", "image-size", ["448"])
    with pytest.raises(ValueError):
        core.validate("gemma4-vision", "image-size", ["300"])  # not whole patches
    with pytest.raises(ValueError):
        core.validate("siglip2-text", "precision", ["int3"])
    s = core.Study.open("siglip2-text", "batch", ["1", "4"], {"precision": "w8a16"}, tmp_path / "st")
    with pytest.raises(ValueError):  # same folder, different fixed settings: refuse to mix results
        core.Study.open("siglip2-text", "batch", ["1"], {"precision": "w4a16"}, s.out)


def test_gemma_text_flags():
    from hexprof.study import gemma4_text as G

    assert G.export_flags("context-length", "1024", {"prompt_length": 128})[:5] == [
        "--context-lengths", "1024", "--sequence-lengths", "128", "1"]
    assert G.export_flags("decode-tokens", "4", {})[:4] == ["--sequence-lengths", "4", "--context-lengths", "4096"]
    assert G.export_flags("prompt-length", "256", {})[-5:] == ["--components", *G.TEXT_COMPONENTS]
    assert G.expected("prompt-length", "256", {"context_length": 2048}) == ({256, 1}, 2048)


def _fake_gemma(monkeypatch, tmp_path, lie_about=None):
    """Simulate Qualcomm's export + AI Hub. `lie_about`: a context length whose graphs secretly run at 4096."""
    from hexprof.pipeline import gemma4_export as ge

    jobs = {}

    def parse(checkpoint, out_dir, device, runtime, precision, flags):
        seq = flags[flags.index("--sequence-lengths") + 1:]
        seq = [int(x) for x in seq[:next((i for i, x in enumerate(seq) if x.startswith("--")), len(seq))]]
        ctx = [int(flags[flags.index("--context-lengths") + 1])]
        return {"sequence_lengths": seq, "context_lengths": ctx, "components": flags[flags.index("--components") + 1:]}

    def export(checkpoint, out_dir, device, runtime, precision, flags, hw=True):
        p = parse(checkpoint, out_dir, device, runtime, precision, flags)
        ctx = p["context_lengths"][0]
        real_ctx = 4096 if ctx == lie_about else ctx
        parts = {}
        for i in (1, 2, 3, 4):
            comp = f"part{i}_of_4"
            linked = []
            for ar in p["sequence_lengths"]:
                kind = "token" if ar == 1 else "prompt"
                graph = f"{kind}_ar{ar}_cl{ctx}_{i}_of_4"
                shapes = {"input_embeds": ((1, ar, 2560), "float32"), "attention_mask": ((1, 1, ar, real_ctx), "uint16"),
                          "past_key_0_in": ((1, 2, 256, real_ctx - ar), "uint8")}
                src = Model(f"onnx_{graph}", name=f"{graph}.onnx")
                cm = Model(f"m_{graph}_{ctx}", name=graph, producer=CompileJob(f"c_{graph}_{ctx}", shapes, source=src))
                linked.append(cm)
                pid = f"p_{graph}_{ctx}"
                jobs[pid] = ProfileJob(pid, Model(f"ml_{comp}_{ctx}", producer=Link(linked)),
                                       options=f"--qnn_options context_enable_graphs={graph}")
                label = f"{comp}__{graph}"
                ms = (40.0 if i == 1 else 10.0) * (ctx / 4096) ** 0.3 * (1 if ar == 1 else 1.6)
                q = hwfakes.qhas_doc(60, seed=i + ar)
                hwfakes.write_json(Path(out_dir) / "hw" / label / "qhas" / "j_main_qhas_summary.json", q)
                hwfakes.write_json(Path(out_dir) / "profiles" / f"{label}.profile.json", hwfakes.profile_doc(ms=ms))
                parts[f"{comp}/{graph}"] = {"component": comp, "graph": graph, "job": {"job_id": pid}, "model_id": "x"}
        summary = {"parts": parts}
        hwfakes.write_json(Path(out_dir) / "export_profiles.json", summary)
        return summary

    monkeypatch.setattr(ge, "parse_export_args", parse)
    monkeypatch.setattr(ge, "export", export)
    monkeypatch.setattr(hc, "get_job", lambda jid: jobs[jid])
    monkeypatch.setattr(core, "qairt_of", lambda job: "2.50.0")


def test_gemma_context_study_runs_verifies_and_compares(tmp_path, monkeypatch):
    _fake_gemma(monkeypatch, tmp_path, lie_about=1024)
    out = S.run("gemma4-text", "context-length", ["512", "1024", "4096"], {}, str(tmp_path / "study"))
    st = json.loads((out / "study.json").read_text())
    assert {v: r["verified"] for v, r in st["variants"].items()} == {"512": True, "1024": False, "4096": True}
    ver = json.loads((out / "context-length-1024" / "verification.json").read_text())
    bad = [c for c in ver["checks"] if not c["ok"]]
    assert bad and all("compiled input shapes" in c["what"] for c in bad)  # names say 1024, shapes say 4096
    cmp = json.loads((out / "comparison" / "comparison.json").read_text())
    by = {v["value"]: v for v in cmp["variants"]}
    m = by["512"]["metrics"]
    assert m["decode_ms_per_token"] == pytest.approx(sum((40.0 if i == 1 else 10.0) * (512 / 4096) ** 0.3 for i in (1, 2, 3, 4)))
    assert m["decode_tokens_per_s"] == pytest.approx(1e3 / m["decode_ms_per_token"])
    assert m["prefill_tokens_per_s"] == pytest.approx(128e3 / m["prefill_chunk_ms"])
    page = _page(out / "comparison" / "index.html")
    chart = next(s for s in page["sections"] if s["title"] == "Decode (tok/s)")["blocks"][0]
    assert chart["type"] == "xy" and [p[0] for p in chart["series"][0]["points"]] == [512, 4096]  # 1024 excluded
    vt = page["sections"][0]["blocks"][0]["table"]["rows"]
    assert [r["verified"] for r in vt] == ["yes", "NO", "yes"] and vt[0]["href"].startswith("../context-length-512/")
    assert (out / "context-length-512" / "hw_report" / "index.html").exists()
    # resuming does not rebuild, and re-verification gives the same answer
    calls = []
    monkeypatch.setattr("hexprof.pipeline.gemma4_export.export", lambda *a, **k: calls.append(a))
    S.run("gemma4-text", "context-length", ["512", "1024", "4096"], {}, str(out), compare=False)
    assert calls == []
    assert S.reverify(out)["variants"]["1024"]["verified"] is False


def test_decode_tokens_metrics(tmp_path, monkeypatch):
    _fake_gemma(monkeypatch, tmp_path)
    out = S.run("gemma4-text", "decode-tokens", ["1", "4"], {}, str(tmp_path / "dk"))
    cmp = json.loads((out / "comparison" / "comparison.json").read_text())
    m4 = next(v for v in cmp["variants"] if v["value"] == "4")["metrics"]
    assert m4["ms_per_token"] == pytest.approx(m4["step_ms"] / 4)
    assert m4["max_tokens_per_s"] == pytest.approx(4e3 / m4["step_ms"])
    assert all(v["verified"] for v in cmp["variants"])


# --------------------------------------------------------------------------- single-graph components


def _fake_qaihm(monkeypatch, tmp_path, ignore_batch=False):
    class Precision:
        def __init__(self, name):
            self.name = name
            m = re.match(r"w(\d+)a(\d+)", name)
            w, a = (m.group(1), m.group(2)) if m else (None, None)
            self.weights_type = f"QuantizeDtype.INT{w}" if w else None
            self.activations_type = f"QuantizeDtype.INT{a}" if a else None

        @staticmethod
        def parse(s):
            return Precision(s)

        def __eq__(self, other):
            return isinstance(other, Precision) and other.name == self.name

        def __str__(self):
            return self.name

    Precision.float = Precision("float")

    class TargetRuntime:
        ONNX = "onnx"
        QNN_CONTEXT_BINARY = "qnn_context_binary"

    class TextEncoder:
        @classmethod
        def from_pretrained(cls, **kw):
            return cls()

        def get_input_spec(self, text_batch_size=1, text_seq_len=64):
            return {"input_ids": ((1 if ignore_batch else text_batch_size, text_seq_len), "int32")}

        def sample_inputs(self, spec):
            return {"input_ids": [0]}

        def get_hub_litemp_percentage(self, prec):
            return None

        def get_hub_inference_options(self, rt, other=""):
            return (other or "--qairt_version 2.50").strip()

        def get_hub_quantize_options(self, prec):  # as Qualcomm's build_quantize_options
            if "_mixed_" in prec.name:
                pct = self.get_hub_litemp_percentage(prec)
                if pct is None:
                    raise ValueError("litemp_percentage is required for mixed precision quantization.")
                return f"--range_scheme min_max --lite_mp percentage={pct};override_qtype={prec.name.split('_mixed_')[1]}"
            return "--range_scheme min_max" if prec.name == "w8a16" else ""

    store = {"n": 0, "jobs": {}, "models": {}}

    def new_id(p):
        store["n"] += 1
        return f"{p}{store['n']}"

    def upload_source(model, spec):
        m = Model(new_id("src"), input_spec=spec)
        store["models"][m.model_id] = m
        return m

    def run_compile(model, name, dev, rt, prec, source, spec, extra_options=""):
        j = CompileJob(new_id("cj"), spec, source=source, options=extra_options)
        j.target = Model(new_id("cm"), input_spec=spec, producer=j)
        store["jobs"][j.job_id] = j
        store["models"][j.target.model_id] = j.target
        return j

    def run_quantize(prec, model, name, onnx_model, n, input_spec=None):
        j = types.SimpleNamespace(job_id=new_id("qj"), weights_dtype=prec.weights_type,
                                  activations_dtype=prec.activations_type,
                                  options=model.get_hub_quantize_options(prec))
        j.target = Model(new_id("qm"), input_spec=input_spec)
        store["jobs"][j.job_id] = j
        store["models"][j.target.model_id] = j.target
        return j

    def run_link(compiled, dev, name, model, rt, extra_options=""):
        j = types.SimpleNamespace(job_id=new_id("lj"), models=[compiled], options=extra_options)
        j.target = Model(new_id("bin"), input_spec=compiled.input_spec)
        store["jobs"][j.job_id] = j
        store["models"][j.target.model_id] = j.target
        return j

    def submit_inference(model, dev, inputs, profile=True, name="", options=""):
        j = ProfileJob(new_id("pj"), model, options)
        store["jobs"][j.job_id] = j
        return j

    def save_profile(job, out_dir, stem):
        b = job.model.input_spec["input_ids"][0][0]
        prof = hwfakes.profile_doc(ms=1.0 + 0.2 * b)
        return prof, hwfakes.write_json(Path(out_dir) / f"{stem}.profile.json", prof)

    def download_hw_artifacts(job, out_dir):
        p = hwfakes.write_json(Path(out_dir) / "qhas" / "j_main_qhas_summary.json", hwfakes.qhas_doc(40))
        return {"qhas": [str(p)], "optrace": []}

    mods = {
        "qai_hub_models": types.SimpleNamespace(Precision=Precision, TargetRuntime=TargetRuntime),
        "qai_hub_models.utils": types.ModuleType("u"), "qai_hub_models.utils.export": types.ModuleType("e"),
        "qai_hub_models.utils.export.compile": types.SimpleNamespace(run_compile=run_compile),
        "qai_hub_models.utils.export.quantize": types.SimpleNamespace(run_quantize=run_quantize),
        "qai_hub_models.utils.export.upload": types.SimpleNamespace(upload_source=upload_source),
        "qai_hub_models.utils.export.link": types.SimpleNamespace(run_link=run_link),
        "qai_hub_models.models": types.ModuleType("m"), "qai_hub_models.models.siglip2": types.ModuleType("s"),
        "qai_hub_models.models.siglip2.model": types.SimpleNamespace(SigLIP2TextEncoder=TextEncoder,
                                                                     SigLIP2ImageEncoder=TextEncoder),
        "qai_hub": types.SimpleNamespace(),
    }
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(hc, "device", lambda name: Dev(name))
    monkeypatch.setattr(hc, "target_model", lambda job: job.target)
    monkeypatch.setattr(hc, "submit_inference", submit_inference)
    monkeypatch.setattr(hc, "wait_success", lambda job, kind="": None)
    monkeypatch.setattr(hc, "save_profile", save_profile)
    monkeypatch.setattr(hc, "download_hw_artifacts", download_hw_artifacts)
    monkeypatch.setattr(hc, "get_job", lambda jid: store["jobs"][jid])
    monkeypatch.setattr(hc, "get_model", lambda mid: store["models"][mid])
    monkeypatch.setattr(hc, "job_info", lambda job: {"job_id": job.job_id, "url": job.url})
    monkeypatch.setattr(core, "qairt_of", lambda job: "2.50.0")
    return store


def test_siglip_text_batch_study(tmp_path, monkeypatch):
    _fake_qaihm(monkeypatch, tmp_path)
    out = S.run("siglip2-text", "batch", ["1", "4", "16"], {}, str(tmp_path / "b"))
    cmp = json.loads((out / "comparison" / "comparison.json").read_text())
    by = {v["value"]: v for v in cmp["variants"]}
    assert all(v["verified"] for v in by.values())
    assert by["16"]["metrics"]["ms_per_item"] == pytest.approx((1.0 + 0.2 * 16) / 16)
    assert by["4"]["metrics"]["items_per_s"] == pytest.approx(4e3 / (1.0 + 0.8))
    ver = json.loads((out / "batch-4" / "verification.json").read_text())
    names = {c["what"] for c in ver["checks"]}
    assert {"compile job input shapes", "compiled model input shapes (AI Hub)", "batch dimension",
            "compile job consumed the quantized model", "quantize job dtypes", "profile device",
            "link job consumed the compiled model", "profiled model = linked context binary"} <= names
    jobs = json.loads((out / "batch-4" / "jobs.json").read_text())
    assert jobs["link_job"].startswith("lj")
    v = json.loads((out / "batch-4" / "variant.json").read_text())
    assert v["settings"]["spec_kwargs"] == {"text_batch_size": 4} and v["settings"]["precision"] == "w8a16"


def test_component_setting_that_does_not_reach_the_graph_is_caught(tmp_path, monkeypatch):
    _fake_qaihm(monkeypatch, tmp_path, ignore_batch=True)  # the model silently ignores the batch size
    out = S.run("siglip2-text", "batch", ["1", "8"], {}, str(tmp_path / "nb"), compare=False)
    st = json.loads((out / "study.json").read_text())
    assert st["variants"]["8"]["verified"] is False and st["variants"]["1"]["verified"] is True


def test_precision_study_float_and_quantized(tmp_path, monkeypatch):
    _fake_qaihm(monkeypatch, tmp_path)
    out = S.run("siglip2-text", "precision", ["float", "w4a16"], {}, str(tmp_path / "p"), compare=False)
    st = json.loads((out / "study.json").read_text())
    assert all(r["verified"] for r in st["variants"].values())
    jobs_f = json.loads((out / "precision-float" / "jobs.json").read_text())
    jobs_q = json.loads((out / "precision-w4a16" / "jobs.json").read_text())
    assert "quantize_job" not in jobs_f and jobs_q["quantize_job"]
    prof = json.loads((out / "precision-w4a16" / "profile.json").read_text())
    assert prof["precision"] == "w4a16"


def test_cli_conflicting_fixed_setting():
    from hexprof import cli

    with pytest.raises(SystemExit):
        cli.main(["study", "run", "siglip2-text", "--axis", "batch", "--batch", "4"])
    with pytest.raises(SystemExit):
        cli.main(["study", "run", "siglip2-text", "--axis", "batch", "--context-length", "1024"])


def test_mixed_precision_study_sets_and_verifies_lite_mp(tmp_path, monkeypatch):
    _fake_qaihm(monkeypatch, tmp_path)
    out = S.run("siglip2-text", "mixed-precision", ["0", "10", "25"], {"precision": "w8a8", "override": "int16"},
                str(tmp_path / "mp"), compare=False)
    st = json.loads((out / "study.json").read_text())
    assert all(r["verified"] for r in st["variants"].values())
    v10 = json.loads((out / "mixed-precision-10" / "verification.json").read_text())
    lm = next(c for c in v10["checks"] if c["what"] == "mixed precision on the quantize job")
    assert lm["ok"] and "percentage=10;override_qtype=int16" in lm["actual"]
    v0 = json.loads((out / "mixed-precision-0" / "variant.json").read_text())
    assert v0["settings"]["precision"] == "w8a8" and v0["settings"]["litemp"] is None
    with pytest.raises(ValueError):  # a float base cannot be mixed
        from hexprof.study import components as C
        C.settings("siglip2-text", "mixed-precision", "10", {"precision": "float"})
    with pytest.raises(ValueError):
        core.validate("siglip2-text", "mixed-precision", ["100"])


def test_ablation_summary_against_baseline(tmp_path, monkeypatch):
    from hexprof.study import summary

    _fake_qaihm(monkeypatch, tmp_path)
    root = tmp_path / "studies"
    S.run("siglip2-text", "batch", ["1", "8"], {}, str(root / "siglip2-text__batch"))
    S.run("siglip2-image", "batch", ["2", "4"], {}, str(root / "siglip2-image__batch"))  # no baseline (1)
    page = summary.summarize(root)
    data = json.loads((page.parent / "summary.json").read_text())
    (row,) = data["ablations"]
    assert row["change"].endswith("1 → 8")
    assert row["items_per_s__ratio"] == pytest.approx((8e3 / (1.0 + 1.6)) / (1e3 / 1.2))
    assert any("siglip2-image / batch" in n for n in data["notes"])
    p = _page(page)
    bars = next(b for s in p["sections"] for b in s["blocks"] if b["type"] == "hbar")
    assert bars["rows"][0]["value"] == pytest.approx(row["items_per_s__ratio"])
