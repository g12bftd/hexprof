"""Run manifest + job log (hexprof.runlog)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hexprof import runlog


class _Status:
    def __init__(self, ok=True, msg=""):
        self.success, self.message, self.code = ok, msg, "SUCCESS" if ok else "FAILED"


class _Job:
    def __init__(self, jid, name, ok=True):
        self.job_id, self.name, self.url = jid, name, f"https://workbench.aihub.qualcomm.com/jobs/{jid}/"
        self.device = SimpleNamespace(name="Samsung Galaxy S25", os="15")
        self.options, self._ok = "--qairt_version 2.50", ok

    def get_status(self):
        return _Status(self._ok, "" if self._ok else "compile error")


def _fake_qai_hub(monkeypatch):
    import hexprof.aihub.client as hc

    ids = iter(range(100))
    mod = SimpleNamespace(
        get_devices=lambda name=None: [SimpleNamespace(os="15", attributes=["chipset:qualcomm-snapdragon-8-elite-for-galaxy",
                                                                            "hexagon:v79", "os:android"])],
        upload_model=lambda path, name=None: SimpleNamespace(model_id=f"m{next(ids)}", name=name),
        submit_compile_job=lambda model, device, name="", options="", input_specs=None: _Job(f"jc{next(ids)}", name),
        submit_inference_job=lambda model, device, inputs=None, name="", options="", profile=False: _Job(f"jp{next(ids)}", name),
        submit_link_job=lambda models, device, name="", options="": _Job(f"jl{next(ids)}", name, ok=False),
    )
    monkeypatch.setattr(hc, "hub", lambda: mod)
    monkeypatch.setattr(hc, "qairt_version", lambda job: "2.50.1")
    return mod


def test_run_logs_model_settings_hardware_and_every_job(tmp_path, monkeypatch, capsys):
    mod = _fake_qai_hub(monkeypatch)
    st = [runlog.setting("hidden", 1280, 2560, group="architecture"), runlog.setting("precision", "w4a16", "w4a16")]
    with runlog.run(tmp_path / "r", kind="t", title="test deployment", model={"name": "Gemma 4", "weights": "random dummy weights"},
                    settings=st, device="Samsung Galaxy S25", runtime="QNN context binary") as run:
        m = mod.upload_model("x.onnx", name="part1")
        mod.submit_compile_job(m, "dev", name="prompt_ar128")  # as Qualcomm's own code calls it: hub.submit_*
        mod.submit_inference_job(m, "dev", name="prompt_ar128", profile=True)
        mod.submit_link_job([m], "dev", name="part1")
        run.result(decode_ms=12.5)
    out = capsys.readouterr().out
    assert "hexprof deployment run: test deployment" in out and "SHRUNK DUMMY" in out
    assert "hidden = 1280  (CHANGED; base 2560)" in out and "Hexagon v79" in out
    assert "RAN ON    device Samsung Galaxy S25; os 15; qairt 2.50.1" in out
    assert "not successful: link" in out
    man = json.loads((tmp_path / "r" / "run_manifest.json").read_text())
    assert man["status"] == "succeeded" and man["job_counts"] == {"upload": 1, "compile": 1, "profile": 1, "link": 1}
    assert man["hardware"]["chipset"] == "qualcomm-snapdragon-8-elite-for-galaxy"
    assert man["verdict"].startswith("SHRUNK DUMMY (random weights), 1 change(s): hidden 2560 -> 1280")
    assert man["results"] == {"decode_ms": 12.5}
    lines = (tmp_path / "r" / "jobs_log.jsonl").read_text().splitlines()
    assert len(lines) == 4 and json.loads(lines[1])["job_id"].startswith("jc")
    md = (tmp_path / "r" / "run_manifest.md").read_text()
    assert "| **changed** | architecture | hidden | 1280 | 2560 |" in md and "## AI Hub jobs" in md
    assert mod.submit_compile_job.__name__ == "<lambda>"  # unwrapped again after the run


def test_capture_keeps_signatures(tmp_path, monkeypatch):
    """hexprof.aihub.client.submit_inference inspects submit_inference_job for ``profile=``."""
    import inspect

    import hexprof.aihub.client as hc

    mod = _fake_qai_hub(monkeypatch)
    calls = []

    def submit_inference_job(model, device, inputs=None, name="", options="", profile=False):
        calls.append(profile)
        return _Job("ji", name)

    mod.submit_inference_job = submit_inference_job
    with runlog.run(tmp_path, kind="t", title="t", model={"name": "m", "weights": "w"}):
        assert "profile" in inspect.signature(mod.submit_inference_job).parameters
        hc.submit_inference(SimpleNamespace(model_id="m1"), "dev", {"x": [1]}, profile=True)
    assert calls == [True]


def test_nested_run_in_the_same_folder_adds_details_and_failure_is_recorded(tmp_path, monkeypatch):
    _fake_qai_hub(monkeypatch)
    with pytest.raises(RuntimeError):
        with runlog.run(tmp_path, kind="outer", title="outer", model={"name": "SigLIP 2", "weights": "Qualcomm AI Hub asset"},
                        settings=[runlog.setting("precision", "w8a16", "w8a16")]) as outer:
            with runlog.run(tmp_path, kind="inner", title="inner", model={"name": "x", "source": "inner code"},
                            settings=[runlog.setting("batch", 4, 1), runlog.setting("precision", "w8a16", "w8a16")]) as inner:
                assert inner is outer
            raise RuntimeError("boom")
    man = json.loads((tmp_path / "run_manifest.json").read_text())
    assert [s["setting"] for s in man["settings"]] == ["precision", "batch"]
    assert man["model"]["name"] == "SigLIP 2" and man["model"]["source"] == "inner code"
    assert man["status"].startswith("failed: RuntimeError: boom")
    assert man["verdict"] == "Qualcomm AI Hub base model, 1 change(s): batch 1 -> 4"


def test_qualcomm_settings_verdict():
    st = [runlog.setting("context lengths", [4096], [4096]), runlog.setting("checkpoint", "DEFAULT", "DEFAULT")]
    assert runlog.verdict({"weights": "Qualcomm AI Hub published checkpoint"}, st) == \
        "Qualcomm AI Hub base model, Qualcomm's settings"


def test_gemma_export_manifest_marks_changed_flags(monkeypatch):
    from hexprof.models import gemma4_aihub as GA
    from hexprof.pipeline import gemma4_export as ge

    def parse(checkpoint, out_dir, device, runtime, precision, flags):
        cl = flags[flags.index("--context-lengths") + 1] if "--context-lengths" in flags else "4096"
        return {"context_lengths": [int(cl)], "sequence_lengths": [128, 1], "components": None,
                "precision": None, "target_runtime": runtime}

    monkeypatch.setattr(ge, "parse_export_args", parse)
    model, settings, info = ge.manifest("DEFAULT", "/tmp/x", "Samsung Galaxy S25", "geniex_qairt", "w4a16",
                                        ["--skip-downloading", *GA.export_flags(2048)])
    ch = {s["setting"]: (s["base"], s["value"]) for s in settings if s["changed"]}
    assert ch == {"context lengths": ([4096], [2048])}
    assert "Qualcomm AI Hub published checkpoint" in model["weights"]
    assert runlog.verdict(model, settings) == "Qualcomm AI Hub base model, 1 change(s): context lengths [4096] -> [2048]"
    assert GA.default_export_dir(2048).name == "gemma4-s25-export-cl2048" and GA.default_export_dir().name == "gemma4-s25-export"


def test_export_folder_refuses_other_flags(tmp_path):
    from hexprof.models import gemma4_aihub as GA

    (tmp_path / "export_profiles.json").write_text(json.dumps({"parts": {"a": {}}, "export_flags": []}))
    assert GA.ensure_export(tmp_path, "Samsung Galaxy S25")["parts"]
    with pytest.raises(ValueError, match="another --export-dir"):
        GA.ensure_export(tmp_path, "Samsung Galaxy S25", context_length=2048)
