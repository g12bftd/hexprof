"""Run the real Adb/GenieXPhone code through a stand-in `adb` that executes commands with /bin/sh."""

import json
import shutil
import stat
import sys
from pathlib import Path

import pytest

from hexprof.geniex.adb import Adb, AdbError
from hexprof.geniex.bench import GenieXPhone
from hexprof.models import gemma4

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")

FAKE_ADB = """#!/bin/sh
if [ "$1" = "-s" ]; then shift 2; fi
cmd="$1"; shift
case "$cmd" in
  shell) exec /bin/sh -c "$*" ;;
  push) mkdir -p "$(dirname "$2")"; cp -r "$1" "$2" ;;
  pull) cp "$1" "$2" ;;
  devices) printf 'List of devices attached\\nFAKE\\tdevice\\n' ;;
esac
"""

FAKE_BENCH = """#!{python}
import json, os, sys
a = sys.argv[1:]
opt = lambda k: a[a.index(k) + 1]
segs = open(opt("--prompt-file")).read().strip().split("\\n---\\n")
runs = []
for i, p in enumerate(segs):
    if len(segs) > 1:
        print(f"[sep ] prompt {{i + 1}}/{{len(segs)}}")
    print("[info] stop_reason=eos")
    print(f"[gen ] LD={{os.environ.get('LD_LIBRARY_PATH', 'UNSET')}} TOKEN={{os.environ.get('GENIEX_HFTOKEN', 'UNSET')}}")
    runs.append(dict(run_idx=i, ttft_us=2000000, media_us=1500000, prompt_time_us=500000, decode_time_us=2000000,
                     prompt_tokens=280, gen_tokens=25, prefill_tps=560.0, decode_tps=12.5, stop_reason="eos"))
json.dump(dict(schema_version="7", cell_id=opt("--cell-id"), plugin="llama_cpp", device="npu", runs=runs,
               geniex_version="v0.10.0"), open(opt("--output-json"), "w"))
"""


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_real_adb_wrapper_env_placement_and_exit_codes(tmp_path, monkeypatch):
    adb_bin = _exe(tmp_path / "adb", FAKE_ADB)
    root = tmp_path / "device"
    phone = GenieXPhone(Adb(serial="FAKE", adb_path=str(adb_bin)), root=str(root))
    # the "installed" package
    (root / "geniex" / "bin").mkdir(parents=True)
    _exe(root / "geniex" / "bin" / "geniex-bench", FAKE_BENCH.format(python=sys.executable))
    for d in ("inputs", "results", "models", "geniex-cache"):
        (root / d).mkdir()

    assert phone.adb.shell("exit 3", check=False).returncode == 3
    with pytest.raises(AdbError):
        phone.adb.shell("false")
    assert phone.adb.shell("echo hi").stdout == "hi\n"

    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    phone.adb.verbose = True
    img = tmp_path / "img.png"
    from PIL import Image

    Image.new("RGB", (32, 32)).save(img)
    res = gemma4.infer(phone, gemma4.InferConfig(repeats=2, warmup=1), [img], ["What is this?"], tmp_path / "out")
    r = res["results"][0]
    assert f"LD={root}/geniex/lib:" in r["answer"]  # env reached geniex-bench, not just `cd`
    assert "TOKEN=hf_secret_value" in r["answer"]  # HF token forwarded for model downloads...
    assert all("hf_secret_value" not in h for h in phone.adb.history)  # ...but never logged
    assert r["breakdown"]["n_runs"] == 2 and r["breakdown"]["vision_encoder_ms"]["median"] == 1500
    raw = json.loads((tmp_path / "out" / "results.json").read_text())
    assert raw["tool_versions"]["geniex_version"] == "v0.10.0"
