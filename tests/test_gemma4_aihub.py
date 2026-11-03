"""Gemma 4 on AI Hub (no phone): export reuse, real-image vision encoder job, estimates, local answer, download dir."""

import json
import math
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hexprof import cli
from hexprof.models import gemma4_aihub
from hexprof.pipeline import gemma4_export

from fakes import FakeHub, FakeJob, FakeModel, fake_profile

PREFILL_MS, DECODE_MS, VISION_MS = 30.0, 15.0, 900.0


def _export_summary() -> dict:
    parts = {
        "vision_encoder/vision_encoder": {
            "component": "vision_encoder", "graph": "vision_encoder", "model_id": "m-veg",
            "options": "--qnn_options context_enable_graphs=vision_encoder",
            "metrics": {"inference_time_milliseconds": 1000.0, "layer_counts": {"npu": 500, "total": 500},
                        "estimated_peak_memory_range_mb": {"min": 0, "max": 400}},
        }
    }
    for i in range(1, 5):
        for kind, ar, ms in (("prompt", 128, PREFILL_MS), ("token", 1, DECODE_MS)):
            g = f"{kind}_ar{ar}_cl4096_{i}_of_4"
            parts[f"part{i}_of_4/{g}"] = {
                "component": f"part{i}_of_4", "graph": g, "model_id": f"m-{g}", "options": "",
                "metrics": {"inference_time_milliseconds": ms, "layer_counts": {"npu": 100, "total": 100},
                            "estimated_peak_memory_range_mb": {"min": 0, "max": 900}},
            }
    return {"bundle": "", "parts": parts}


class FakeProcessor:
    """Mimics Gemma 4's processor: 256 image soft tokens + 4 template tokens + 1 per word."""

    def apply_chat_template(self, messages, **kw):
        content = messages[0]["content"]
        text = next(c["text"] for c in content if c["type"] == "text")
        img = next(c["image"] for c in content if c["type"] == "image")
        assert img.size == (448, 448)  # resized to the compiled vision encoder geometry
        n = 256 + 4 + len(text.split())
        return {"input_ids": np.zeros((1, n), np.int64), "pixel_values": np.ones((1, 16, 12), np.float32),
                "image_position_ids": np.zeros((1, 16, 2), np.int64)}


FAKE_LLAMA = """#!{python}
import sys, os
a = sys.argv[1:]
assert a[a.index("--temp") + 1] == "0" and "--jinja" in a and "--image" in a
print("load_model: loading ...")
print(os.environ["MTMD_TEST_RESPONSE_MARKER"])
print("The newspaper is The New York Times.")
sys.stderr.write("llama_perf_context_print: prompt eval time =  900.00 ms /   270 tokens\\n")
sys.stderr.write("llama_perf_context_print:        eval time =     550.00 ms /    11 runs   (50 ms per token)\\n")
"""


@pytest.fixture
def setup_env(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "export_profiles.json").write_text(json.dumps(_export_summary()))
    hub = FakeHub(lambda payload, inputs: VISION_MS * 1000,
                  lambda model, inputs: {"vision_embedding": [np.zeros((1, 256, 8), np.float32)]}).install(monkeypatch)
    hub.models["m-veg"] = FakeModel("m-veg", {"path": "veg.bin"})
    monkeypatch.setattr(gemma4_aihub, "_processor", lambda repo: FakeProcessor())
    img = tmp_path / "newspaper.jpg"
    Image.new("RGB", (640, 488), (240, 240, 240)).save(img)
    return hub, export_dir, img


def test_aihub_infer_with_local_answer(tmp_path, setup_env):
    hub, export_dir, img = setup_env
    llama = tmp_path / "llama-mtmd-cli"
    llama.write_text(FAKE_LLAMA.format(python=sys.executable))
    llama.chmod(llama.stat().st_mode | stat.S_IEXEC)
    prompt = "What is the publisher name of the newspaper?"
    r = gemma4_aihub.infer([img], [prompt], tmp_path / "out", export_dir=export_dir, llama_cli=str(llama), local="on")

    # the vision encoder ran on the user's image with the profiled graph's own options
    inference_calls = [c for c in hub.calls if c[0] == "inference"]
    assert len(inference_calls) == 1 and "context_enable_graphs=vision_encoder" in inference_calls[0][2]
    assert hub.last_inputs["image_position_ids"][0].dtype == np.int32
    res = r["results"][0]
    assert res["answer"]["answer"] == "The newspaper is The New York Times."
    assert res["vision_encoder"]["image_tokens"] == 256
    assert Path(res["vision_encoder"]["embedding_file"]).exists()
    b = res["breakdown"]
    n = 256 + 4 + len(prompt.split())
    assert b["prefill_tokens"] == n and b["prefill_chunks"] == math.ceil(n / 128)
    assert b["prefill_ms"] == pytest.approx(math.ceil(n / 128) * 4 * PREFILL_MS)
    assert b["decode_ms_per_token"] == pytest.approx(4 * DECODE_MS)
    assert b["decode_tokens"] == 11 and b["decode_tokens_source"] == "local answer"
    assert b["vision_encoder_ms"] == pytest.approx(VISION_MS)
    assert b["e2e_ms"] == pytest.approx(VISION_MS + b["prefill_ms"] + 11 * 4 * DECODE_MS)
    md = (tmp_path / "out" / "results.md").read_text()
    assert "New York Times" in md and "Vision encoder (your image, measured)" in md


def test_aihub_infer_without_llama_uses_max_new_tokens(tmp_path, setup_env, monkeypatch):
    hub, export_dir, img = setup_env
    monkeypatch.setattr(gemma4_aihub.shutil, "which", lambda name: None)
    r = gemma4_aihub.infer([img], ["Describe it.", "Count the people."], tmp_path / "o", export_dir=export_dir,
                           max_new_tokens=64)
    assert len([c for c in hub.calls if c[0] == "inference"]) == 1  # one vision job per image, not per prompt
    assert all(x["answer"] is None and x["breakdown"]["decode_tokens"] == 64 for x in r["results"])
    with pytest.raises(RuntimeError):
        gemma4_aihub.infer([img], ["x"], tmp_path / "p", export_dir=export_dir, local="on")


def test_ensure_export_runs_once_and_records_graph_handles(tmp_path, monkeypatch):
    FakeHub(lambda p, i: 1000.0).install(monkeypatch)
    calls = []

    class Result:
        download_path = None
        profile_jobs = {
            ("vision_encoder", "vision_encoder"): FakeJob("j1", profile=fake_profile(900_000), model=FakeModel("m-veg"),
                                                          options="--qnn_options context_enable_graphs=vision_encoder"),
            ("part1_of_4", "prompt_ar128_cl4096_1_of_4"): FakeJob("j2", profile=fake_profile(30_000), model=FakeModel("m-p1")),
            ("part1_of_4", "token_ar1_cl4096_1_of_4"): FakeJob("j3", profile=fake_profile(15_000), model=FakeModel("m-t1")),
        }

    def fake_run(checkpoint, out_dir, device, runtime, precision, extra=()):
        calls.append((checkpoint, device, runtime, precision, list(extra)))
        return Result()

    monkeypatch.setattr(gemma4_export, "run_official_export", fake_run)
    monkeypatch.setattr(gemma4_export, "_require_qaihm", lambda: None)
    d = tmp_path / "exp"
    s = gemma4_aihub.ensure_export(d, "Samsung Galaxy S25", hw=False)
    assert calls == [("DEFAULT", "Samsung Galaxy S25", "geniex_qairt", "w4a16", ["--skip-downloading"])]
    veg = s["parts"]["vision_encoder/vision_encoder"]
    assert veg["model_id"] == "m-veg" and "context_enable_graphs" in veg["options"]
    parts = gemma4_aihub.split_parts(s)
    assert parts["seq_len"] == 128 and parts["context_length"] == 4096
    assert len(parts["prefill"]) == 1 and len(parts["decode"]) == 1
    gemma4_aihub.ensure_export(d, "Samsung Galaxy S25", hw=False)
    assert len(calls) == 1  # reused


def test_download_dir_routes_every_cache(tmp_path, monkeypatch):
    for k in ("HEXPROF_DOWNLOAD_DIR", "QAIHM_STORE_ROOT", "HF_HUB_CACHE", "LLAMA_CACHE", "HEXPROF_CACHE"):
        monkeypatch.delenv(k, raising=False)
    value, rest = cli._pop_download_dir(["siglip2", "infer", "--download-dir", str(tmp_path / "dl"), "--image", "a.jpg"])
    assert value == str(tmp_path / "dl") and rest == ["siglip2", "infer", "--image", "a.jpg"]
    assert cli._pop_download_dir([f"--download-dir={tmp_path}", "doctor"]) == (str(tmp_path), ["doctor"])
    d = cli.apply_download_dir(str(tmp_path / "dl"))
    assert os.environ["QAIHM_STORE_ROOT"] == str(d)
    assert os.environ["HF_HUB_CACHE"] == str(d / "huggingface")
    assert os.environ["LLAMA_CACHE"] == str(d / "llama.cpp")
    assert cli._assets_dir() == d
    assert gemma4_aihub.default_export_dir() == d / "gemma4-s25-export"
    args = cli.build_parser().parse_args(["siglip2", "infer", "--image", "a.jpg"])
    bundle = d / "siglip2-qnn_dlc-w8a16"
    bundle.mkdir()
    (bundle / "metadata.json").write_text("{}")
    assert cli._bundle(args) == bundle


def test_export_argv_matches_qualcomm_parser():
    # Option strings from the real `qai-hub-models export gemma_4_e4b_it` usage (qai_hub_models 0.65, no --precision).
    gemma_opts = {"--runtime", "--target-runtime", "--checkpoint", "--host-device", "--no--skip-quantsim-creation",
                  "--sequence-lengths", "--context-lengths", "--zip-assets", "--inference-options",
                  "--skip-profiling", "--skip-downloading", "--skip-summary", "--output-dir", "--compile-options",
                  "--link-options", "--device", "--chipset", "--device-os"}
    argv = gemma4_export.export_argv(gemma_opts, "DEFAULT", "Samsung Galaxy S25", "geniex_qairt", "w4a16", "out",
                                     ["--skip-downloading"])
    assert "--precision" not in argv
    assert argv == ["--checkpoint", "DEFAULT", "--device", "Samsung Galaxy S25", "--output-dir", "out",
                    "--target-runtime", "geniex_qairt", "--skip-downloading"]
    assert set(a for a in argv if a.startswith("--")) <= gemma_opts
    # recipes that do take --precision still get it
    assert "--precision" in gemma4_export.export_argv(gemma_opts | {"--precision"}, "c", "d", "r", "w4a16", "o")
