"""Run manifest and job log for every deployment: which model, which settings, which hardware.

Every command that deploys something on AI Hub opens a run::

    with runlog.run(out_dir, kind="dummy", title=..., model=..., settings=[...], device=...) as run:
        ...
        run.result(decode_ms=...)

The run

* prints a banner at the start (it lands at the top of the Slurm ``.out`` log) and writes
  ``<out>/run_manifest.json`` and ``<out>/run_manifest.md``:

  - **model**: name, weights (Qualcomm's published asset, random dummy weights, your checkpoint),
    architecture (layers, hidden size, ...);
  - **settings**: every setting with Qualcomm's (or the real model's) value next to it, changes
    marked, and a one-line verdict (e.g. "Qualcomm AI Hub base model, Qualcomm's settings" or
    "shrunk dummy: 3 changes");
  - **hardware**: AI Hub device, chipset, Hexagon version, NPU facts, target runtime, QAIRT version;
  - **environment**: hexprof commit, command line, host, Slurm job, package versions;

* logs every AI Hub job submitted while it is open (``<out>/jobs_log.jsonl`` and one stdout line
  each), including the ones Qualcomm's own export code submits;
* at the end records status, duration, what AI Hub reports it actually used (device, OS, QAIRT
  version of the profile jobs), the final state of every job and the headline results.

A run opened for the same folder while one is already open (e.g. the component runner inside a
dummy run) adds its details to the open run instead of starting a second one. Nothing here ever
stops a deployment: logging problems are reported and ignored.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator, Sequence

from hexprof.io_utils import _jsonable, write_json

PACKAGES = ("qai-hub", "qai-hub-models", "torch", "transformers", "onnx", "onnxruntime", "numpy")
_SUBMIT = ("upload_model", "submit_compile_job", "submit_quantize_job", "submit_link_job", "submit_profile_job",
           "submit_inference_job")
_ACTIVE: list["Run"] = []


def setting(name: str, value: Any, base: Any = None, note: str = "", group: str = "deployment") -> dict[str, Any]:
    """One setting. ``base``: Qualcomm's (or the real model's) value; None = no reference."""
    changed = base is not None and _norm(value) != _norm(base)
    return {"group": group, "setting": name, "value": value, "base": base, "changed": changed, "note": note}


def _norm(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return str(v).lower() if v is not None else None


# --------------------------------------------------------------------------- facts


def environment() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    pkgs = {}
    for p in PACKAGES:
        try:
            pkgs[p] = version(p)
        except PackageNotFoundError:
            pass
    repo = Path(__file__).resolve().parent.parent
    commit = None
    try:
        commit = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%h %s"], capture_output=True,
                                text=True, timeout=5).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        if commit and dirty:
            commit += " (+ local changes)"
    except Exception:  # noqa: BLE001 - not a git checkout
        pass
    slurm = {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CLUSTER_NAME", "SLURM_JOB_PARTITION")
             if os.environ.get(k)}
    return {"hexprof_commit": commit, "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
            "host": platform.node(), "python": platform.python_version(), "packages": pkgs, "slurm": slurm,
            "download_dir": os.environ.get("HEXPROF_DOWNLOAD_DIR")}


def hardware(device: str, runtime: str = "", qairt: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """What we know about the target: hexprof's device facts + AI Hub's device attributes (when reachable)."""
    from hexprof.devices import get_device

    d = get_device(device)
    hw: dict[str, Any] = {"device": d.hub_name, "where": "AI Hub hosted device (Qualcomm Workbench)",
                          "soc": d.description or None, "chipset": d.chipset or None,
                          "hexagon": d.hexagon_arch or None, "compute_unit": "NPU (Hexagon HTP)",
                          "npu_facts": ({"vtcm_mb": d.vtcm_bytes / 2**20, "hvx_units": d.hvx_units,
                                         "hmx_units": d.hmx_units, "dram_peak_gbps": d.dram_peak_gbps}
                                        if d.vtcm_bytes else None),
                          "runtime": runtime or None, "qairt_requested": qairt or "pinned by qai_hub_models / AI Hub default"}
    try:
        from hexprof.aihub import client as hc

        devs = hc.hub().get_devices(name=d.hub_name)
        if devs:
            attrs = [str(a) for a in (getattr(devs[0], "attributes", None) or [])]
            hw["aihub_os"] = getattr(devs[0], "os", None)
            hw["aihub_attributes"] = attrs
            for a in attrs:
                k, _, v = a.partition(":")
                if k in ("chipset", "hexagon", "soc-model") and v:
                    hw[{"chipset": "chipset", "hexagon": "hexagon", "soc-model": "soc_model"}[k]] = v
    except Exception:  # noqa: BLE001 - offline / no token: hexprof's own facts only
        pass
    hw.update(extra or {})
    return hw


def verdict(model: dict[str, Any], settings: Sequence[dict[str, Any]]) -> str:
    changed = [s for s in settings if s.get("changed")]
    weights = str(model.get("weights", "")).lower()
    if "random" in weights or "dummy" in weights:
        head = "SHRUNK DUMMY (random weights)"
    elif "qualcomm" in weights:
        head = "Qualcomm AI Hub base model"
    elif "synthetic" in weights:
        head = "synthetic benchmark graphs"
    else:
        head = "custom weights"
    if not changed:
        return head + (", Qualcomm's settings" if "qualcomm" in head.lower() else ", reference settings")
    return f"{head}, {len(changed)} change(s): " + ", ".join(f"{s['setting']} {s['base']} -> {s['value']}" for s in changed)


# --------------------------------------------------------------------------- the run


class Run:
    def __init__(self, out_dir: str | Path, kind: str, title: str, model: dict[str, Any],
                 settings: Sequence[dict[str, Any]], hw: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
        self.out = Path(out_dir)
        self.t0 = time.time()
        self.jobs: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        self.m: dict[str, Any] = {
            "run_id": f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}", "kind": kind, "title": title,
            "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "status": "running", "out_dir": str(self.out),
            "model": model, "settings": list(settings), "verdict": verdict(model, settings), "hardware": hw,
            "environment": environment(), "extra": dict(extra or {}), "results": {}}

    # ---- additions while running
    def add(self, model: dict[str, Any] | None = None, settings: Sequence[dict[str, Any]] = (),
            extra: dict[str, Any] | None = None, hw: dict[str, Any] | None = None) -> None:
        if model:
            for k, v in model.items():
                if isinstance(v, dict) and isinstance(self.m["model"].get(k), dict):
                    self.m["model"][k].update(v)
                elif v is not None:
                    self.m["model"].setdefault(k, v)
        known = {(s["group"], s["setting"]) for s in self.m["settings"]}
        new = [s for s in settings if (s["group"], s["setting"]) not in known]
        self.m["settings"] += new
        self.m["extra"].update(extra or {})
        self.m["hardware"].update({k: v for k, v in (hw or {}).items() if v is not None})
        self.m["verdict"] = verdict(self.m["model"], self.m["settings"])
        if new:
            print("[hexprof] more settings: " + "; ".join(_fmt_setting(s) for s in new), flush=True)
        self.save()

    def result(self, **kw: Any) -> None:
        self.m["results"].update(kw)
        self.save()

    def job(self, kind: str, obj: Any, inputs: Any = None) -> None:
        """Log one submitted AI Hub job (or uploaded model)."""
        try:
            rec: dict[str, Any] = {"time": time.strftime("%H:%M:%S"), "kind": kind}
            if kind == "upload":
                rec.update(model_id=getattr(obj, "model_id", None), name=getattr(obj, "name", None))
            else:
                dev = getattr(obj, "device", None)
                rec.update(job_id=getattr(obj, "job_id", None), name=getattr(obj, "name", None),
                           url=getattr(obj, "url", None), device=getattr(dev, "name", None) or (dev if isinstance(dev, str) else None),
                           options=getattr(obj, "options", None) or None)
                self._handles.append((rec, obj))
            if inputs is not None:
                rec["input_models"] = inputs
            self.jobs.append(rec)
            with open(self.out / "jobs_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(rec)) + "\n")
            what = rec.get("job_id") or rec.get("model_id")
            print(f"[hexprof] {rec['time']} {kind:<9} {what} {rec.get('name') or ''}"
                  + (f" on {rec['device']}" if rec.get("device") else "") + (f"  {rec['url']}" if rec.get("url") else ""),
                  flush=True)
        except Exception as e:  # noqa: BLE001 - logging must never stop a deployment
            print(f"[hexprof] (job log: {type(e).__name__}: {e})", flush=True)

    # ---- files
    def save(self) -> None:
        try:
            self.out.mkdir(parents=True, exist_ok=True)
            write_json(self.out / "run_manifest.json", self.m)
            (self.out / "run_manifest.md").write_text(markdown(self.m), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[hexprof] (run manifest: {type(e).__name__}: {e})", flush=True)

    def finish(self, error: BaseException | None) -> None:
        self.m["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        self.m["duration_min"] = round((time.time() - self.t0) / 60, 1)
        self.m["status"] = "succeeded" if error is None else f"failed: {type(error).__name__}: {error}"
        if error is not None:
            self.m["traceback_tail"] = traceback.format_exception(error)[-3:]
        self._final_job_states()
        counts: dict[str, int] = {}
        for j in self.jobs:
            counts[j["kind"]] = counts.get(j["kind"], 0) + 1
        self.m["job_counts"] = counts
        self.save()
        print(banner_end(self.m), flush=True)

    def _final_job_states(self) -> None:
        from hexprof.aihub import client as hc

        used: dict[str, set[str]] = {"device": set(), "os": set(), "qairt": set()}
        for rec, job in self._handles:
            try:
                st = job.get_status() if hasattr(job, "get_status") else None
                if st is not None:
                    rec["final_status"] = str(getattr(st, "code", None) or ("SUCCESS" if getattr(st, "success", False) else st))
                    if getattr(st, "message", None) and not getattr(st, "success", True):
                        rec["message"] = str(st.message)[:300]
                if rec["kind"] in ("profile", "inference"):
                    dev = getattr(job, "device", None)
                    if getattr(dev, "name", None):
                        used["device"].add(dev.name)
                    if getattr(dev, "os", None):
                        used["os"].add(str(dev.os))
                    q = hc.qairt_version(job)
                    if q:
                        used["qairt"].add(q)
            except Exception:  # noqa: BLE001
                continue
        self.m["jobs"] = self.jobs
        if any(used.values()):
            self.m["hardware"]["as_reported_by_aihub"] = {k: sorted(v) for k, v in used.items() if v}


def _fmt_setting(s: dict[str, Any]) -> str:
    v = s["value"]
    if s.get("changed"):
        return f"{s['setting']} = {v}  (CHANGED; base {s['base']})"
    return f"{s['setting']} = {v}" + ("  (as base)" if s.get("base") is not None else "")


def banner_start(m: dict[str, Any]) -> str:
    hw, mod, env = m["hardware"], m["model"], m["environment"]
    w = 96
    lines = ["=" * w, f"hexprof deployment run: {m['title']}", f"run id {m['run_id']}, started {m['started']}", "-" * w,
             f"VERDICT   {m['verdict']}", "",
             f"MODEL     {mod.get('name')}", f"          weights: {mod.get('weights')}"]
    if mod.get("source"):
        lines.append(f"          source: {mod['source']}")
    arch = mod.get("architecture") or {}
    if arch:
        lines.append("          architecture: " + ", ".join(f"{k} {v}" for k, v in arch.items()))
    if mod.get("parameters"):
        lines.append(f"          parameters: {mod['parameters']}")
    lines += ["", "SETTINGS  (value; base = Qualcomm's / the real model's value)"]
    for s in m["settings"]:
        lines.append(f"  {'*' if s.get('changed') else ' '} [{s['group']}] {_fmt_setting(s)}" + (f"  - {s['note']}" if s.get("note") else ""))
    soc = str(hw.get("soc") or "")
    lines += ["", f"HARDWARE  {hw.get('device')} ({hw.get('where')})",
              f"          SoC {soc or '?'}" + ("" if hw.get("hexagon") and str(hw["hexagon"]) in soc else f"; Hexagon {hw.get('hexagon')}")
              + f"; chipset {hw.get('chipset')}; compute unit {hw.get('compute_unit')}"]
    if hw.get("npu_facts"):
        f = hw["npu_facts"]
        lines.append(f"          VTCM {f['vtcm_mb']:g} MB, {f['hvx_units']} HVX, {f['hmx_units']} HMX, "
                     f"DRAM peak {f['dram_peak_gbps']} GB/s (theoretical)")
    lines.append(f"          runtime {hw.get('runtime')}; QAIRT {hw.get('qairt_requested')}")
    if hw.get("aihub_attributes"):
        lines.append("          AI Hub attributes: " + ", ".join(hw["aihub_attributes"][:12]))
    lines += ["", f"ENV       hexprof {env.get('hexprof_commit')}; python {env.get('python')}; host {env.get('host')}"
              + (f"; Slurm job {env['slurm'].get('SLURM_JOB_ID')}" if env.get("slurm") else ""),
              "          " + ", ".join(f"{k} {v}" for k, v in env.get("packages", {}).items()),
              f"          command: {env.get('command')}", f"OUTPUT    {m['out_dir']}", "=" * w]
    for k, v in (m.get("extra") or {}).items():
        if v not in (None, "", [], {}):
            lines.insert(-1, f"EXTRA     {k}: {v}")
    return "\n".join(lines)


def banner_end(m: dict[str, Any]) -> str:
    w = 96
    lines = ["=" * w, f"hexprof run {m['run_id']}: {m['status']} after {m.get('duration_min')} min",
             f"VERDICT   {m['verdict']}"]
    rep = (m["hardware"].get("as_reported_by_aihub") or {})
    if rep:
        lines.append("RAN ON    " + "; ".join(f"{k} {', '.join(v)}" for k, v in rep.items()))
    if m.get("job_counts"):
        lines.append("JOBS      " + ", ".join(f"{v} {k}" for k, v in m["job_counts"].items()))
    bad = [j for j in m.get("jobs", []) if j.get("final_status") and "SUCCESS" not in j["final_status"].upper()]
    for j in bad[:8]:
        lines.append(f"  not successful: {j['kind']} {j.get('job_id')} {j.get('name') or ''}: {j['final_status']} "
                     f"{j.get('message') or ''}")
    for k, v in (m.get("results") or {}).items():
        lines.append(f"RESULT    {k}: {v}")
    lines += [f"MANIFEST  {Path(m['out_dir']) / 'run_manifest.md'}", "=" * w]
    return "\n".join(lines)


def markdown(m: dict[str, Any]) -> str:
    hw, mod = m["hardware"], m["model"]
    out = [f"# {m['title']}", "", f"**{m['verdict']}**", "",
           f"Run `{m['run_id']}`: {m['status']}; started {m['started']}"
           + (f", finished {m['finished']} ({m.get('duration_min')} min)" if m.get("finished") else ""), "",
           "## Model", "", f"* name: {mod.get('name')}", f"* weights: {mod.get('weights')}"]
    if mod.get("source"):
        out.append(f"* source: {mod['source']}")
    for k, v in (mod.get("architecture") or {}).items():
        out.append(f"* {k}: {v}")
    out += ["", "## Settings", "", "| | group | setting | value | base (Qualcomm / real model) | note |", "|---|---|---|---|---|---|"]
    for s in m["settings"]:
        out.append(f"| {'**changed**' if s.get('changed') else ''} | {s['group']} | {s['setting']} | {s['value']} | "
                   f"{'' if s.get('base') is None else s['base']} | {s.get('note') or ''} |")
    out += ["", "## Hardware", ""]
    for k, v in hw.items():
        out.append(f"* {k}: {v}")
    out += ["", "## Environment", ""]
    for k, v in m["environment"].items():
        out.append(f"* {k}: {v}")
    if m.get("extra"):
        out += ["", "## Other", ""] + [f"* {k}: {v}" for k, v in m["extra"].items()]
    if m.get("results"):
        out += ["", "## Results", ""] + [f"* {k}: {v}" for k, v in m["results"].items()]
    jobs = m.get("jobs") or []
    if jobs:
        out += ["", "## AI Hub jobs", "", "| time | kind | id | name | device | final status |", "|---|---|---|---|---|---|"]
        for j in jobs:
            jid = j.get("job_id") or j.get("model_id")
            link = f"[{jid}]({j['url']})" if j.get("url") else str(jid)
            out.append(f"| {j['time']} | {j['kind']} | {link} | {j.get('name') or ''} | {j.get('device') or ''} | "
                       f"{j.get('final_status') or ''} |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- capture + context


@contextlib.contextmanager
def _capture(run: Run) -> Iterator[None]:
    """Wrap qai_hub's submit functions so every job (hexprof's or Qualcomm's) is logged."""
    try:
        from hexprof.aihub import client as hc

        mod = hc.hub()
    except Exception:  # noqa: BLE001 - no qai_hub: nothing to capture
        yield
        return
    originals = {n: getattr(mod, n) for n in _SUBMIT if hasattr(mod, n)}
    kinds = {"upload_model": "upload", "submit_compile_job": "compile", "submit_quantize_job": "quantize",
             "submit_link_job": "link", "submit_profile_job": "profile", "submit_inference_job": "inference"}

    def wrap(name: str, fn: Any) -> Any:
        @functools.wraps(fn)  # keep the signature: hexprof.aihub.client inspects it (e.g. ``profile=``)
        def wrapped(*a: Any, **kw: Any) -> Any:
            res = fn(*a, **kw)
            kind = kinds[name]
            if kind == "inference" and kw.get("profile"):
                kind = "profile"
            src = kw.get("model", a[0] if a else None)
            ins = ([getattr(x, "model_id", None) for x in src] if isinstance(src, (list, tuple))
                   else getattr(src, "model_id", None))
            for obj in (res if isinstance(res, list) else [res]):
                for r in list(_ACTIVE):
                    r.job(kind, obj, ins if kind != "upload" else None)
            return res
        return wrapped

    for n, fn in originals.items():
        setattr(mod, n, wrap(n, fn))
    try:
        yield
    finally:
        for n, fn in originals.items():
            setattr(mod, n, fn)


@contextlib.contextmanager
def run(out_dir: str | Path, *, kind: str, title: str, model: dict[str, Any], settings: Sequence[dict[str, Any]] = (),
        device: str = "Samsung Galaxy S25", runtime: str = "", qairt: str | None = None,
        hw_extra: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> Iterator[Run]:
    out = Path(out_dir)
    same = next((r for r in _ACTIVE if r.out.resolve() == out.resolve()), None) if out.exists() else None
    if same is not None:  # e.g. the component runner inside a dummy run: one run, more details
        same.add(model=model, settings=settings, extra={**(extra or {}), "runner": title},
                 hw={"runtime": runtime or None})
        yield same
        return
    r = Run(out, kind, title, model, settings, hardware(device, runtime, qairt, hw_extra), extra)
    out.mkdir(parents=True, exist_ok=True)
    r.save()
    print(banner_start(r.m), flush=True)
    _ACTIVE.append(r)
    err: BaseException | None = None
    try:
        with _capture(r) if len(_ACTIVE) == 1 else contextlib.nullcontext():
            yield r
    except BaseException as e:
        err = e
        raise
    finally:
        _ACTIVE.remove(r)
        r.finish(err)


def current() -> Run | None:
    return _ACTIVE[-1] if _ACTIVE else None
