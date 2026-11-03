"""Stage and drive ``geniex-bench`` on an Android phone over adb.

The on-device layout and environment mirror Qualcomm AI Hub Models' own
GenieX scorecard for Galaxy S25/S26 (``qai_hub_models/utils/llm/geniex``):

* the release tarball ``geniex-bench-android-arm64[-vX.Y.Z].tar.gz`` is pushed
  to the phone, ``lib/qairt/htp-files/*.so`` and ``lib/llama_cpp/*.so`` are
  copied next to ``lib/libgeniex.so``;
* commands run with ``LD_LIBRARY_PATH=lib:lib/llama_cpp:lib/qairt``,
  ``ADSP_LIBRARY_PATH=lib`` and ``GENIEX_PLUGIN_PATH=lib``;
* models are pulled on the phone by GenieX's model manager into
  ``--mm-data-dir`` (so the phone needs internet), or pushed from the host.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from hexprof.devices import GALAXY_S25, TargetDevice
from hexprof.geniex.adb import Adb, AdbError, ShellResult

GENIEX_S3 = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex"
DEVICE_ROOT = "/data/local/tmp/hexprof"
HOST_CACHE = Path(os.environ.get("HEXPROF_CACHE", Path.home() / ".cache" / "hexprof"))


def bench_url(version: str | None = None) -> str:
    """Release URL of the Android geniex-bench package (``None`` = latest stable)."""
    suffix = f"-v{version.lstrip('v')}" if version else ""
    return f"{GENIEX_S3}/geniex-bench-android-arm64{suffix}.tar.gz"


def _safe_extract(tgz: Path, dest: Path) -> Path:
    """Extract a tarball refusing absolute paths / traversal; return its top dir."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as tf:
        members = tf.getmembers()
        base = dest.resolve()
        for m in members:
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(base) + os.sep):
                raise ValueError(f"refusing unsafe tar member {m.name!r}")
            if m.issym() or m.islnk():
                link = (target.parent / m.linkname).resolve()
                if not str(link).startswith(str(base) + os.sep):
                    raise ValueError(f"refusing tar link {m.name!r} -> {m.linkname!r}")
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, members=members, filter="data")
        else:  # pragma: no cover - Python < 3.10.12
            tf.extractall(dest, members=members)  # noqa: S202 - members validated above
        top = {m.name.split("/", 1)[0] for m in members if m.name}
    if len(top) != 1:
        raise ValueError(f"expected one top-level directory in {tgz}, got {sorted(top)}")
    return dest / top.pop()


def fetch_host_package(version: str | None = None, cache: Path = HOST_CACHE, refresh: bool = False) -> Path:
    """Download + extract the Android geniex-bench package on the host (cached)."""
    url = bench_url(version)
    root = cache / "geniex"
    root.mkdir(parents=True, exist_ok=True)
    tgz = root / Path(url).name
    if refresh or not tgz.exists():
        tmp = tgz.with_suffix(".part")
        print(f"Downloading {url}")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:  # noqa: S310 - fixed https URL
            shutil.copyfileobj(resp, f)
        tmp.replace(tgz)
    extract_root = root / (tgz.name.removesuffix(".tar.gz"))
    marker = extract_root / ".extracted"
    if refresh or not marker.exists():
        if extract_root.exists():
            shutil.rmtree(extract_root)
        pkg = _safe_extract(tgz, extract_root)
        marker.write_text(pkg.name)
    pkg = extract_root / marker.read_text().strip()
    if not (pkg / "bin" / "geniex-bench").exists():
        raise FileNotFoundError(f"{pkg} has no bin/geniex-bench")
    return pkg


@dataclass
class GenieXPhone:
    """A phone with geniex-bench installed under ``root``."""

    adb: Adb
    target: TargetDevice = GALAXY_S25
    root: str = DEVICE_ROOT
    hf_token: str | None = field(default=None, repr=False)

    @property
    def pkg(self) -> str:
        return f"{self.root}/geniex"

    @property
    def cache(self) -> str:
        return f"{self.root}/geniex-cache"

    @property
    def inputs(self) -> str:
        return f"{self.root}/inputs"

    @property
    def results(self) -> str:
        return f"{self.root}/results"

    @property
    def models(self) -> str:
        return f"{self.root}/models"

    # ------------------------------------------------------------------ setup

    def check_device(self) -> dict[str, str]:
        if not self.adb.dry_run and not self.adb.serial and not self.adb.devices():
            raise AdbError("No adb device. Connect the phone with USB debugging enabled (`adb devices`).")
        info = {
            "model": self.adb.getprop("ro.product.model"),
            "soc_model": self.adb.getprop("ro.soc.model"),
            "android": self.adb.getprop("ro.build.version.release"),
            "build": self.adb.getprop("ro.build.fingerprint"),
        }
        if info["soc_model"] and self.target.soc_model and info["soc_model"] != self.target.soc_model:
            print(
                f"WARNING: phone SoC is {info['soc_model']}, expected {self.target.soc_model} "
                f"({self.target.description}). --chipset {self.target.chipset} may pull the wrong bundle."
            )
        return info

    def install(self, version: str | None = None, refresh: bool = False) -> str:
        """Push geniex-bench to the phone (idempotent unless ``refresh``)."""
        pkg = fetch_host_package(version, refresh=refresh) if not self.adb.dry_run else Path(f"geniex-bench-android-arm64-{version or 'latest'}")
        marker = f"{self.pkg}/.hexprof-version"
        if not refresh and not self.adb.dry_run:
            cur = self.adb.shell(f"cat {marker} 2>/dev/null", check=False).stdout.strip()
            if cur == pkg.name:
                return pkg.name
        self.adb.shell(f"rm -rf {self.pkg} && mkdir -p {self.root} {self.cache} {self.inputs} {self.results} {self.models}")
        self.adb.push(pkg, self.pkg)
        self.adb.shell(f"find {self.pkg}/bin -type f -exec chmod 755 {{}} +")
        self.adb.shell(f"cp {self.pkg}/lib/qairt/htp-files/*.so {self.pkg}/lib/")
        self.adb.shell(f"cp {self.pkg}/lib/llama_cpp/*.so {self.pkg}/lib/")
        self.adb.shell(f"echo {shlex.quote(pkg.name)} > {marker}")
        return pkg.name

    def installed_version(self) -> str | None:
        if self.adb.dry_run:
            return None
        out = self.adb.shell(f"cat {self.pkg}/.hexprof-version 2>/dev/null", check=False).stdout.strip()
        return out or None

    # ------------------------------------------------------------------ running

    def env(self) -> dict[str, str]:
        lib = f"{self.pkg}/lib"
        return {
            "LD_LIBRARY_PATH": f"{lib}:{lib}/llama_cpp:{lib}/qairt",
            "ADSP_LIBRARY_PATH": lib,
            "GENIEX_PLUGIN_PATH": lib,
        }

    def _secret_env(self) -> dict[str, str]:
        token = self.hf_token or os.environ.get("GENIEX_HFTOKEN") or os.environ.get("HF_TOKEN")
        return {"GENIEX_HFTOKEN": token} if token else {}

    def push_input(self, local: str | Path) -> str:
        """Push a host file into the inputs dir under a content-addressed name."""
        local = Path(local)
        digest = hashlib.sha1(local.read_bytes()).hexdigest()[:12] if local.exists() else "missing"
        remote = f"{self.inputs}/{digest}-{local.name.replace(' ', '_')}"
        if not self.adb.exists(remote):
            self.adb.push(local, remote)
        return remote

    def push_model_file(self, local: str | Path) -> str:
        local = Path(local)
        remote = f"{self.models}/{local.name}"
        if not self.adb.exists(remote):
            self.adb.push(local, remote)
        return remote

    def run_bench(
        self,
        args: Sequence[str],
        remote_stdout: str,
        remote_stderr: str,
        timeout: float | None = None,
    ) -> ShellResult:
        """Run ``geniex-bench <args>`` on the phone, redirecting output to files."""
        quoted = " ".join(shlex.quote(a) for a in args)
        cmd = f"cd {self.pkg} && {{ENV}} ./bin/geniex-bench {quoted} >{remote_stdout} 2>{remote_stderr}"
        return self.adb.shell(cmd, env=self.env(), secret_env=self._secret_env(), check=False, timeout=timeout)

    def read_remote(self, path: str) -> str:
        if self.adb.dry_run:
            return ""
        return self.adb.shell(f"cat {path} 2>/dev/null", check=False).stdout

    def model_args(
        self,
        plugin: str,
        model: str,
        mmproj: str | None = None,
    ) -> list[str]:
        """``-m`` (+ ``--mmproj-path``) with host paths pushed to the phone first."""
        args: list[str] = []
        if Path(model).exists():
            m = Path(model)
            remote = self.push_model_file(m) if m.is_file() else self._push_dir(m)
            args += ["-m", remote]
        else:
            args += ["-m", model]
        if mmproj:
            args += ["--mmproj-path", self.push_model_file(mmproj) if Path(mmproj).exists() else mmproj]
        return ["--plugin", plugin] + args

    def _push_dir(self, local: Path) -> str:
        remote = f"{self.models}/{local.name}"
        if not self.adb.exists(remote):
            self.adb.push(local, remote)
        return remote

    def common_args(self) -> list[str]:
        return ["--mm-data-dir", self.cache, "--chipset", self.target.chipset]
