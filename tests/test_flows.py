"""End-to-end flows against a simulated phone (geniex-bench) and a simulated AI Hub."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hexprof.cli import main
from hexprof.geniex.bench import GenieXPhone
from hexprof.models import gemma4, siglip2

from fakes import FakeHub, FakePhone

DATA = Path(__file__).parent / "data"


def _img(path: Path, color=(200, 30, 30), size=(320, 240)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


# --------------------------------------------------------------------------- Gemma 4 / GenieX


def test_gemma4_infer_runs_each_image_with_warmup_and_repeats(tmp_path):
    adb = FakePhone(media_ms=1600, prefill_tps=600, decode_tps=14)
    phone = GenieXPhone(adb)
    imgs = [_img(tmp_path / "a.jpg"), _img(tmp_path / "b.png", (0, 0, 255))]
    cfg = gemma4.InferConfig(repeats=3, warmup=1, max_new_tokens=64)
    res = gemma4.infer(phone, cfg, imgs, ["What color is it?", "Describe it\n---\nbriefly."], tmp_path / "out")

    assert len(adb.bench_calls) == 2  # one geniex-bench process per image
    args = adb.bench_calls[0]
    for flag in ("--accuracy", "--vlm", "--image", "--prompt-file", "--output-json", "--no-think", "--chipset"):
        assert flag in args
    assert args[args.index("-m") + 1] == "google/gemma-4-E4B-it-qat-q4_0-gguf:Q4_0"
    assert args[args.index("--plugin") + 1] == "llama_cpp" and args[args.index("--device") + 1] == "npu"
    assert args[args.index("--chipset") + 1] == "qualcomm-snapdragon-8-elite-for-galaxy"
    # 2 prompts x (1 warm-up + 3 measured) segments; the '---' inside a prompt was neutralised
    prompt_file = adb.fs[args[args.index("--prompt-file") + 1]].decode()
    assert prompt_file.count("\n---\n") == 7 and "- - -" in prompt_file

    assert len(res["results"]) == 4
    r = res["results"][0]
    assert r["answer"].startswith("Answer to: What color is it?") and r["consistent_across_runs"]
    b = r["breakdown"]
    assert b["n_runs"] == 3
    assert b["vision_encoder_ms"]["median"] == pytest.approx(1600)
    assert b["prefill_tokens"]["median"] == 4 + 256
    assert b["e2e_ms"]["median"] == pytest.approx(b["ttft_ms"]["median"] + b["decode_ms"]["median"])
    assert sum(run["warmup"] for run in r["runs"]) == 1
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert saved["aihub_runtime"] == "geniex_llamacpp"
    assert "Vision encoder" in (tmp_path / "out" / "results.md").read_text()
    # environment mirrors AI Hub's own GenieX scorecard
    assert phone.env()["ADSP_LIBRARY_PATH"].endswith("/geniex/lib")


def test_gemma4_text_only_and_images_together(tmp_path):
    adb = FakePhone()
    phone = GenieXPhone(adb)
    res = gemma4.infer(phone, gemma4.InferConfig(repeats=1, warmup=0), [], ["Hi"], tmp_path / "t")
    assert "--vlm" not in adb.bench_calls[-1] and res["results"][0]["breakdown"]["vision_encoder_ms"]["median"] == 0
    imgs = [_img(tmp_path / "a.jpg"), _img(tmp_path / "b.jpg")]
    res = gemma4.infer(phone, gemma4.InferConfig(repeats=1, warmup=0, images_together=True), imgs, ["Compare"], tmp_path / "u")
    assert adb.bench_calls[-1].count("--image") == 2 and len(res["results"]) == 1


def test_gemma4_profile_reproduces_aihub_sweep(tmp_path):
    adb = FakePhone(prefill_tps=727.9, decode_tps=13.36)
    phone = GenieXPhone(adb)
    res = gemma4.profile_aihub(phone, tmp_path / "p", computes=["npu", "cpu"], contexts=[512, 4096])
    first = adb.bench_calls[0]
    assert first[first.index("-r") + 1] == "3"
    assert first[first.index("-c") + 1] == "640" and first[first.index("-p") + 1] == "384" and first[first.index("-n") + 1] == "128"
    tsv = adb.fs[first[first.index("--matrix-file") + 1]].decode().splitlines()
    assert tsv[0].split("\t")[:4] == ["Gemma-4-E4B-it-llama_cpp-npu-c512", "llama_cpp", "npu",
                                      "google/gemma-4-E4B-it-qat-q4_0-gguf:Q4_0"]
    assert len(tsv[0].split("\t")) == 8
    ms = {(m["context_length"], m["desired_compute_unit"]): m for m in res["metrics"]}
    assert set(ms) == {(512, "npu"), (512, "cpu"), (4096, "npu"), (4096, "cpu")}
    m = ms[(4096, "npu")]
    assert m["tokens_per_second"] == pytest.approx(13.36)
    tr = m["time_to_first_token_range_milliseconds"]
    assert tr["max"] / tr["min"] == pytest.approx(4096 / 128)
    perf = (tmp_path / "p" / "perf.yaml").read_text()
    assert "geniex_llamacpp" in perf and "llm_metrics" in perf
    assert "AI Hub" in (tmp_path / "p" / "metrics.md").read_text()


def test_gemma4_cli_dry_run_prints_commands(capsys, tmp_path):
    img = _img(tmp_path / "x.jpg")
    rc = main(["gemma4", "infer", "--dry-run", "--image", str(img), "--prompt", "What is this?", "--out", str(tmp_path / "d")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "./bin/geniex-bench" in out and "--vlm" in out
    # env vars must apply to geniex-bench, i.e. come after `cd ... &&`
    line = next(ln for ln in out.splitlines() if "./bin/geniex-bench" in ln)
    assert line.index("&& LD_LIBRARY_PATH=") < line.index("./bin/geniex-bench") and "{ENV}" not in line


# --------------------------------------------------------------------------- SigLIP 2 / AI Hub


def _bundle(tmp_path: Path) -> Path:
    b = tmp_path / "siglip2-qnn_dlc-w8a16"
    b.mkdir()
    shutil.copy(DATA / "siglip2_w8a16_metadata.json", b / "metadata.json")
    (b / "image_encoder.dlc").write_bytes(b"dlc-image")
    (b / "text_encoder.dlc").write_bytes(b"dlc-text")
    return b


def test_siglip2_io_conversion_matches_metadata(tmp_path):
    bundle = siglip2.Bundle.load(_bundle(tmp_path))
    spec = bundle.io("image_encoder")["inputs"]["image"]
    x = siglip2.preprocess(_img(tmp_path / "a.png", (255, 0, 0)))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    q = siglip2.to_model_input(x, spec)
    assert q.shape == (1, 224, 224, 3) and q.dtype == np.uint16
    assert q[0, 0, 0].tolist() == [65535, 0, 0]
    out_spec = bundle.io("image_encoder")["outputs"]["image_embeds"]
    y = np.full((1, 768), out_spec["quantization_parameters"]["zero_point"], np.uint16)
    assert np.allclose(siglip2.from_model_output(y, out_spec), 0.0)
    assert bundle.qairt_version.startswith("2.50")


def test_siglip2_infer_and_profile_flow(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    protos = {c: siglip2.l2n(rng.standard_normal(768)) for c in ("red", "blue")}

    def outputs(model, inputs):
        name = next(iter(inputs))
        res = []
        for arr in inputs[name]:
            if name == "image":
                color = "red" if arr[0, 0, 0, 0] > arr[0, 0, 0, 2] else "blue"
                emb = protos[color]
                spec = json.loads((DATA / "siglip2_w8a16_metadata.json").read_text())["model_files"]["image_encoder.dlc"]["outputs"]["image_embeds"]["quantization_parameters"]
                res.append(np.round(emb / spec["scale"] + spec["zero_point"]).astype(np.uint16)[None])
            else:
                res.append(protos["red" if arr[0, 1] % 2 == 0 else "blue"][None].astype(np.float32))
        return {"image_embeds" if name == "image" else "text_embeds": res}

    hub = FakeHub(lambda payload, inputs: 3319.0, outputs).install(monkeypatch)
    monkeypatch.setattr(siglip2, "tokenize", lambda labels: np.array([[1, 0], [1, 1]], np.int32)[: len(labels)])
    b = _bundle(tmp_path)
    imgs = [_img(tmp_path / "r.jpg", (250, 10, 10)), _img(tmp_path / "b.jpg", (10, 10, 250))]
    res = siglip2.infer(imgs, b, tmp_path / "o", labels=["a red thing", "a blue thing"], logit_scale=10.0, logit_bias=-5.0)
    E = np.load(tmp_path / "o" / "image_embeddings.npy")
    assert E.shape == (2, 768) and np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-4)
    assert res["best_label"] == {"r.jpg": "a red thing", "b.jpg": "a blue thing"}
    assert res["image_encoder"]["metrics"]["inference_time_milliseconds"] == pytest.approx(3.319)
    assert all(c[2] == "--qairt_version 2.50" for c in hub.calls if c[0] == "inference")
    prof = siglip2.profile(b, tmp_path / "p", components=["image_encoder", "text_encoder"])
    assert prof["components"]["image_encoder"]["published"]["inference_time_milliseconds"] == 3.319
    cache = json.loads((b / ".hexprof_hub_models.json").read_text())
    assert set(cache) == {"image_encoder.dlc", "text_encoder.dlc"}


# --------------------------------------------------------------------------- ceilings / AI Hub


def test_compute_and_bandwidth_sweeps_recover_simulated_ceilings(tmp_path, monkeypatch):
    from hexprof.ceilings import experiments
    from hexprof.ceilings.runner import Sweep

    P = {"w8a16": 9.0e12, "fp16": 4.0e12}
    B = 55e9

    def latency_us(payload, inputs):
        key = Path(payload["path"]).name.removesuffix(".onnx")
        st = json.loads((tmp_path / "state.json").read_text()) if (tmp_path / "state.json").exists() else {}
        st.update(json.loads((tmp_path / "bw" / "state.json").read_text()) if (tmp_path / "bw" / "state.json").exists() else {})
        g = st[key]["graph"]
        if key.startswith("compute"):
            return (0.5e-3 + g["ops"] / P[st[key]["precision"]]) * 1e6
        return (0.2e-3 + g["weight_bytes"] / B) * 1e6

    hub = FakeHub(latency_us).install(monkeypatch)
    res = experiments.compute_sweep(Sweep(tmp_path, "Samsung Galaxy S25"), ["w8a16", "fp16"], 64, [1, 2, 4, 8])
    assert res["series"]["w8a16"]["P_tops"] == pytest.approx(9.0, rel=1e-3)
    assert res["series"]["fp16"]["P_tops"] == pytest.approx(4.0, rel=1e-3)
    assert ("quantize", "hexprof compute-w8a16-s64-m64-l1", "") in hub.calls
    assert not any(c[0] == "quantize" and "fp16" in c[1] for c in hub.calls)  # float graphs skip quantization
    compiles = {name: opts for kind, name, opts in hub.calls if kind == "compile"}
    assert compiles["hexprof compute-w8a16-s64-m64-l1"] == "--target_runtime qnn_dlc --quantize_io"
    assert compiles["hexprof compute-fp16-s64-m64-l1"] == "--target_runtime qnn_dlc"
    links = [c for c in hub.calls if c[0] == "link"]
    assert len(links) == 8  # every point: DLC -> context binary, as Qualcomm's export does
    profiled = [j.model.payload for j in hub.jobs.values() if j.profile]
    assert profiled and all(m.get("linked") for m in profiled)
    assert res["qairt"] == ["2.50.0"]

    n_before = len(hub.calls)
    experiments.compute_sweep(Sweep(tmp_path, "Samsung Galaxy S25"), ["w8a16", "fp16"], 64, [1, 2, 4, 8])
    assert len(hub.calls) == n_before  # resumed from state.json: nothing resubmitted

    bw = experiments.bandwidth_sweep(Sweep(tmp_path / "bw", "Samsung Galaxy S25"), "w4a16", widths=[128, 192, 256],
                                     width_layers=4, onchip_mb=0.001)
    s = bw["series"]["width sweep (4 layers)"]
    assert s["B_gbps"] == pytest.approx(55, rel=1e-3) and s["t0_ms"] == pytest.approx(0.2, abs=1e-6)


def test_sweep_marks_failed_points_and_continues(tmp_path, monkeypatch):
    from hexprof.ceilings import experiments
    from hexprof.ceilings.runner import Sweep

    FakeHub(lambda payload, inputs: 1000.0).install(monkeypatch)
    import hexprof.aihub.client as hc

    real_compile = hc.submit_compile

    def flaky(model, dev, name, options="", input_specs=None):
        job = real_compile(model, dev, name, options, input_specs)
        job.fail = "l2" in name
        return job

    monkeypatch.setattr(hc, "submit_compile", flaky)
    sweep = Sweep(tmp_path, "Samsung Galaxy S25")
    res = experiments.compute_sweep(sweep, ["fp16"], 64, [1, 2, 4])
    assert "error" in sweep.state["compute-fp16-s64-m64-l2"]
    assert len(res["series"]["fp16"]["points"]) == 2
