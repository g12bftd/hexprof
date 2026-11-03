"""Minimal adb wrapper that recovers on-device exit codes.

``adb shell`` drops the remote exit status on some adb versions, so every
command is suffixed with ``; echo __RC__:$?`` (the same trick Qualcomm's own
QDC scripts use) and the code is parsed back out.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

_RC = "__RC__:"


class AdbError(RuntimeError):
    pass


@dataclass
class ShellResult:
    cmd: str
    returncode: int
    stdout: str


@dataclass
class Adb:
    serial: str | None = None
    adb_path: str = "adb"
    dry_run: bool = False
    verbose: bool = False
    history: list[str] = field(default_factory=list)

    def _base(self) -> list[str]:
        exe = self.adb_path
        if not self.dry_run and shutil.which(exe) is None and not Path(exe).exists():
            raise AdbError("adb not found. Install Android platform-tools and enable USB debugging on the phone.")
        return [exe] + (["-s", self.serial] if self.serial else [])

    def _log(self, text: str) -> None:
        self.history.append(text)
        if self.verbose or self.dry_run:
            print(f"$ {text}")

    def shell(
        self,
        cmd: str,
        *,
        env: Mapping[str, str] | None = None,
        secret_env: Mapping[str, str] | None = None,
        check: bool = True,
        timeout: float | None = None,
    ) -> ShellResult:
        """Run ``cmd`` on the device. ``secret_env`` values are never echoed.

        Environment assignments replace an ``{ENV}`` placeholder in ``cmd`` when
        present, otherwise they prefix the whole command.
        """
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in (env or {}).items())
        secret = " ".join(f"{k}={shlex.quote(v)}" for k, v in (secret_env or {}).items())
        shown_secret = " ".join(f"{k}=***" for k in (secret_env or {}))
        real_env = " ".join(p for p in (secret, prefix) if p)
        shown_env = " ".join(p for p in (shown_secret, prefix) if p)
        if "{ENV}" in cmd:
            # Variable assignments must sit right before the program they are for
            # (`cd x && VAR=1 prog`), not before a builtin like `cd`.
            full = cmd.replace("{ENV}", real_env)
            shown = cmd.replace("{ENV}", shown_env)
        else:
            full = " ".join(p for p in (real_env, cmd) if p)
            shown = " ".join(p for p in (shown_env, cmd) if p)
        self._log("adb shell " + shown)
        if self.dry_run:
            return ShellResult(cmd, 0, "")
        proc = subprocess.run(
            self._base() + ["shell", f"{full}; echo {_RC}$?"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        out, rc = proc.stdout, proc.returncode
        lines = out.rstrip("\n").split("\n") if out else []
        if lines and lines[-1].startswith(_RC):
            try:
                rc = int(lines[-1][len(_RC):])
                out = "\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else "")
            except ValueError:
                pass
        if check and rc != 0:
            raise AdbError(f"device command failed (exit {rc}): {cmd}\n{out[-4000:]}")
        return ShellResult(cmd, rc, out)

    def push(self, src: str | Path, dst: str) -> None:
        self._log(f"adb push {src} {dst}")
        if self.dry_run:
            return
        proc = subprocess.run(self._base() + ["push", str(src), dst], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise AdbError(f"adb push {src} -> {dst} failed:\n{proc.stdout}{proc.stderr}")

    def pull(self, src: str, dst: str | Path) -> bool:
        self._log(f"adb pull {src} {dst}")
        if self.dry_run:
            return False
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(self._base() + ["pull", src, str(dst)], capture_output=True, text=True, check=False)
        return proc.returncode == 0

    def exists(self, path: str) -> bool:
        if self.dry_run:
            return False
        return self.shell(f"test -e {shlex.quote(path)}", check=False).returncode == 0

    def getprop(self, name: str) -> str:
        if self.dry_run:
            return ""
        return self.shell(f"getprop {name}", check=False).stdout.strip()

    def devices(self) -> list[str]:
        if self.dry_run:
            return []
        proc = subprocess.run(self._base()[:1] + ["devices"], capture_output=True, text=True, check=False)
        return [ln.split()[0] for ln in proc.stdout.splitlines()[1:] if ln.strip().endswith("device")]
