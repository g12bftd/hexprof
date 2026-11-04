"""Facts about each profiled graph's model, fetched from AI Hub once and cached next to the results.

For every graph whose profile job is recorded (Gemma 4: ``export_profiles.json``; SigLIP 2:
``profile.json``) this follows the model AI Hub profiled back to its source:

    profiled model --(link job)--> compiled graph --(compile job)--> source model (ONNX)

and writes ``<results>/models/<graph>.model.json`` with

* ``compiled_file_size``: size of what runs on the phone (≈ weights at their deployed
  precision; both graphs of a Gemma part share one context binary and its weights),
* ``source_num_parameters`` and the compile job's input ``shapes``,
* ``onnx_path`` when the source ONNX was downloaded (``--onnx``), for per-op MAC counts.

Needs the AI Hub token; the report itself stays offline and reads these files.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Sequence

from hexprof.io_utils import read_json, write_json


def job_refs(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for root in (Path(p) if Path(p).is_dir() else Path(p).parent for p in paths):
        for f in root.rglob("export_profiles.json"):
            for key, rec in (read_json(f).get("parts") or {}).items():
                refs[key.replace("/", "__")] = {"job_id": (rec.get("job") or {}).get("job_id"),
                                                "model_id": rec.get("model_id"), "graph": rec.get("graph"),
                                                "root": str(f.parent)}
        for f in root.rglob("profile.json"):
            try:
                d = read_json(f)
            except (OSError, ValueError):
                continue
            for comp, rec in ((d.get("components") or {}) if isinstance(d, dict) else {}).items():
                job = (rec or {}).get("job") or {}
                if job.get("job_id"):
                    refs[comp] = {"job_id": job["job_id"], "graph": comp, "precision": d.get("precision"),
                                  "root": str(f.parent)}
    return refs


def _graph_tokens(graph: str | None) -> int | None:
    m = re.search(r"_ar(\d+)_", graph or "")
    return int(m.group(1)) if m else None


def _pick_linked(models: list[Any], graph: str | None) -> Any:
    """The input of a link job that is this graph (by name, else by its sequence length)."""
    if not models:
        return None
    for m in models:
        if graph and graph in (getattr(m, "name", "") or ""):
            return m
    ar = _graph_tokens(graph)
    if ar:
        for m in models:
            job = m.get_producer()
            shapes = getattr(job, "shapes", None) or {}
            first = next(iter(shapes.values()), None) if isinstance(shapes, dict) else None
            dims = first[0] if isinstance(first, tuple) and first and isinstance(first[0], (tuple, list)) else first
            if dims and len(dims) > 1 and int(dims[1]) == ar:
                return m
    return models[0]


def source_of(model: Any, graph: str | None, max_hops: int = 6) -> tuple[Any, Any]:
    """(source model, compile job) behind a compiled model; (None, None) when not traceable."""
    for _ in range(max_hops):
        job = model.get_producer() if hasattr(model, "get_producer") else None
        if job is None:
            return None, None
        if isinstance(getattr(job, "models", None), list):  # link job: one compiled model per graph
            model = _pick_linked(job.models, graph)
            if model is None:
                return None, None
            continue
        src = getattr(job, "model", None)
        if src is None:
            return None, None
        if type(job).__name__ == "CompileJob" or hasattr(job, "get_target_model"):
            return src, job
        model = src  # e.g. a quantize job: keep walking
    return None, None


def _shapes(job: Any) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for name, spec in (getattr(job, "shapes", None) or {}).items():
        dims = spec[0] if isinstance(spec, tuple) and spec and isinstance(spec[0], (tuple, list)) else spec
        try:
            out[name] = [int(x) for x in dims]
        except (TypeError, ValueError):
            pass
    return out


def _extract_onnx(path: Path) -> Path | None:
    if path.is_file() and zipfile.is_zipfile(path):
        dest = path.with_suffix("")
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
        path = dest
    if path.is_dir():
        hits = sorted(path.rglob("*.onnx"))
        return hits[0] if hits else None
    return path if path.suffix == ".onnx" else None


def fetch(paths: Sequence[str | Path], onnx: str = "none", max_onnx_gb: float = 20.0, refresh: bool = False) -> dict[str, Any]:
    """Write ``models/<graph>.model.json`` for every graph with a recorded job. ``onnx``: none | vision | all."""
    from hexprof.aihub import client as hc

    refs = job_refs(paths)
    if not refs:
        raise FileNotFoundError("no export_profiles.json / profile.json with job ids under " + ", ".join(map(str, paths)))
    out: dict[str, Any] = {}
    for label, ref in sorted(refs.items()):
        dest = Path(ref["root"]) / "models" / f"{label}.model.json"
        info: dict[str, Any] = read_json(dest) if dest.exists() and not refresh else {}
        want_onnx = onnx == "all" or (onnx == "vision" and "vision" in label)
        if info and not info.get("error") and (info.get("onnx_path") or info.get("onnx_skipped") or not want_onnx):
            print(f"{label}: cached ({dest})")
            out[label] = info
            continue
        try:
            model = hc.get_model(ref["model_id"]) if ref.get("model_id") else hc.get_job(ref["job_id"]).model
            info = {"label": label, "job_id": ref.get("job_id"), "compiled_model_id": model.model_id,
                    "compiled_name": getattr(model, "name", None), "compiled_type": str(getattr(model, "model_type", "")),
                    "compiled_file_size": getattr(model, "file_size", None),
                    "compiled_num_parameters": getattr(model, "num_parameters", None)}
            src, cjob = source_of(model, ref.get("graph"))
            if src is not None:
                info.update(source_model_id=src.model_id, source_name=getattr(src, "name", None),
                            source_type=str(getattr(src, "model_type", "")), source_file_size=getattr(src, "file_size", None),
                            source_num_parameters=getattr(src, "num_parameters", None),
                            compile_job_id=getattr(cjob, "job_id", None), shapes=_shapes(cjob))
            if want_onnx and src is not None:
                size = getattr(src, "file_size", None) or 0
                if size > max_onnx_gb * 1e9:
                    info["onnx_skipped"] = f"source model is {size / 1e9:.1f} GB > --max-onnx-gb {max_onnx_gb:g}"
                else:
                    folder = dest.parent / label
                    folder.mkdir(parents=True, exist_ok=True)
                    print(f"{label}: downloading source model {src.model_id} ({size / 1e9:.2f} GB) …", flush=True)
                    got = Path(src.download(str(folder / "source")))
                    onnx_file = _extract_onnx(got)
                    if onnx_file:
                        info["onnx_path"] = str(onnx_file)
                    else:
                        info["onnx_skipped"] = f"source is not ONNX ({got.name})"
        except Exception as e:  # noqa: BLE001 - one graph failing should not stop the others
            info = {**info, "label": label, "error": f"{type(e).__name__}: {e}"}
        write_json(dest, info)
        sz = info.get("compiled_file_size")
        print(f"{label}: compiled {sz / 1e9:.3f} GB" if sz else f"{label}: {info.get('error') or 'no size reported'}",
              (f"; ONNX {info['onnx_path']}" if info.get("onnx_path") else ""), flush=True)
        out[label] = info
    return out


def load_infos(roots: Sequence[Path]) -> dict[str, dict[str, Any]]:
    infos: dict[str, dict[str, Any]] = {}
    for r in roots:
        for f in r.rglob("*.model.json"):
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and d.get("label"):
                infos[d["label"]] = d
    return infos
