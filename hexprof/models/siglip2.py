"""SigLIP 2 (siglip2-base-patch16-224) on an AI Hub hosted Galaxy S25.

GenieX only serves LLMs/VLMs, so SigLIP 2 runs the way AI Hub publishes it: the
pre-built W8A16 QAIRT DLC from AI Hub Models (``qai-hub-models fetch siglip2
-r qnn_dlc -p w8a16``) executed by an AI Hub inference/profile job on the
hosted S25 (QNN HTP backend, i.e. the NPU).

* :func:`infer` – your images in, L2-normalised 768-d embeddings out (saved as
  ``.npy``), plus image-image cosine similarity and, with ``labels``, SigLIP
  zero-shot scores (text encoder also run on the S25). The same job is profiled.
* :func:`profile` – a profile job with AI Hub's default random inputs, i.e. the
  measurement behind the published 3.319 ms (image encoder, W8A16, QAIRT DLC).
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hexprof.aihub import client as hc
from hexprof.aihub import hwtrace
from hexprof.aihub.metrics import aihub_metrics, metrics_markdown, op_type_breakdown, per_layer
from hexprof.io_utils import markdown_table, write_json
from hexprof.reference import published_siglip2

WEIGHTS = "google/siglip2-base-patch16-224"
IMAGE_SIZE = 224
TEXT_SEQ_LEN = 64  # AI Hub export's fixed text length
COMPONENTS = ("image_encoder", "text_encoder")


# --------------------------------------------------------------------------- assets


def fetch(out_dir: str | Path, precision: str = "w8a16", runtime: str = "qnn_dlc") -> Path:
    """Download AI Hub Models' pre-built SigLIP 2 bundle; return the extracted folder."""
    try:
        from qai_hub_models_cli.fetch import fetch as qaihm_fetch  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install 'hexprof[assets]' (qai_hub_models_cli) to fetch AI Hub Models assets") from e
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(qaihm_fetch(model="siglip2", runtime=runtime, precision=precision, output_dir=out, extract=True))
    if path.suffix == ".zip":
        dest = path.with_suffix("")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
        path = dest
    if not (path / "metadata.json").exists():
        nested = [p.parent for p in path.rglob("metadata.json")]
        if nested:
            path = nested[0]
    return path


@dataclass
class Bundle:
    root: Path
    metadata: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path) -> "Bundle":
        root = Path(root)
        meta = json.loads((root / "metadata.json").read_text()) if (root / "metadata.json").exists() else {}
        return cls(root, meta)

    def model_file(self, component: str) -> Path:
        files = self.metadata.get("model_files", {})
        for name in files:
            if name.startswith(component):
                return self.root / name
        hits = sorted(self.root.glob(f"{component}.*"))
        if not hits:
            raise FileNotFoundError(f"no {component} model in {self.root}")
        return hits[0]

    def io(self, component: str) -> dict[str, Any]:
        return self.metadata.get("model_files", {}).get(self.model_file(component).name, {})

    @property
    def precision(self) -> str:
        return str(self.metadata.get("precision", "unknown"))

    @property
    def runtime(self) -> str:
        return str(self.metadata.get("runtime", self.model_file("image_encoder").suffix.lstrip(".")))

    @property
    def qairt_version(self) -> str | None:
        return (self.metadata.get("tool_versions") or {}).get("qairt")

    def hub_model(self, component: str) -> Any:
        """Upload the component once and cache its AI Hub model id next to the bundle."""
        path = self.model_file(component)
        cache_file = self.root / ".hexprof_hub_models.json"
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        digest = _file_digest(path)
        if (entry := cache.get(path.name)) and entry.get("sha") == digest:
            try:
                return hc.get_model(entry["model_id"])
            except Exception:  # noqa: BLE001 - expired / deleted model: re-upload
                pass
        print(f"Uploading {path.name} ({path.stat().st_size / 2**20:.0f} MB) to AI Hub...")
        model = hc.upload_model(path, name=f"siglip2_{component}_{self.precision}")
        cache[path.name] = {"sha": digest, "model_id": model.model_id}
        cache_file.write_text(json.dumps(cache, indent=2))
        return model


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- pre/post-processing


def preprocess(image_path: str | Path) -> np.ndarray:
    """Float32 [1, 3, 224, 224] RGB in [0, 1] — same as AI Hub Models' SigLIP2App.

    (Bilinear resize to 224x224, no crop; the model maps [0, 1] to [-1, 1] itself.)
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def to_model_input(x_nchw: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Match the compiled model's layout and IO dtype (quantized IO -> integer)."""
    shape = spec.get("shape")
    x = x_nchw
    if shape and len(shape) == 4 and shape[-1] == 3 and x.shape[1] == 3:
        x = x.transpose(0, 2, 3, 1)  # NCHW -> NHWC (AI Hub compiles images channel-last)
    dtype = spec.get("dtype", "float32")
    q = spec.get("quantization_parameters")
    if q and dtype in ("uint8", "uint16", "int8", "int16"):
        info = np.iinfo(dtype)
        x = np.clip(np.round(x / q["scale"] + q["zero_point"]), info.min, info.max).astype(dtype)
    return np.ascontiguousarray(x.astype(dtype, copy=False))


def from_model_output(y: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Dequantize integer outputs: real = (q - zero_point) * scale."""
    q = spec.get("quantization_parameters")
    if np.issubdtype(y.dtype, np.integer) and q:
        return (y.astype(np.float32) - q["zero_point"]) * q["scale"]
    return y.astype(np.float32)


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def tokenize(labels: Sequence[str]) -> np.ndarray:
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install 'hexprof[siglip-text]' for zero-shot labels") from e
    tok = AutoTokenizer.from_pretrained(WEIGHTS, use_fast=True)
    out = tok(list(labels), return_tensors="np", padding="max_length", truncation=True, max_length=TEXT_SEQ_LEN)
    return out["input_ids"].astype(np.int32)


def logit_scale_bias() -> tuple[float, float]:
    """SigLIP's learned temperature and bias (as AI Hub Models' SigLIP2App reads them)."""
    import math

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(WEIGHTS, filename="model.safetensors")
    with safe_open(path, framework="np") as f:
        return float(math.exp(float(f.get_tensor("logit_scale").reshape(-1)[0]))), float(
            f.get_tensor("logit_bias").reshape(-1)[0]
        )


# --------------------------------------------------------------------------- jobs


def _inference_options(bundle: Bundle, qairt_version: str | None, extra: str) -> str:
    opts = []
    if qairt_version is None and bundle.qairt_version:
        qairt_version = ".".join(bundle.qairt_version.split(".")[:2])  # the version AI Hub verified it with
    if qairt_version and qairt_version != "default":
        opts.append(f"--qairt_version {qairt_version}")
    if extra:
        opts.append(extra)
    return " ".join(opts)


def _run_component(
    bundle: Bundle,
    component: str,
    inputs: list[np.ndarray],
    dev: Any,
    out_dir: Path,
    options: str,
    hw: bool,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    io = bundle.io(component)
    in_name, in_spec = next(iter(io.get("inputs", {}).items()), ("image" if component == "image_encoder" else "input_ids", {}))
    out_name, out_spec = next(iter(io.get("outputs", {}).items()), (None, {}))
    model = bundle.hub_model(component)
    data = {in_name: [to_model_input(x, in_spec) if component == "image_encoder" else x for x in inputs]}
    job = hc.submit_inference(model, dev, data, profile=True, name=f"hexprof siglip2 {component} {bundle.precision}",
                              options=options)
    print(f"  {component}: inference job {job.job_id} ({len(inputs)} input(s)) -> {job.url}")
    hc.wait_success(job, f"{component} inference")
    outputs = job.download_output_data()
    key = out_name if out_name in outputs else next(iter(outputs))
    embeds = [from_model_output(np.asarray(y), out_spec).reshape(-1) for y in outputs[key]]
    record = _collect_profile(job, out_dir / component, component, hw)
    return embeds, record


def _collect_profile(job: Any, out_dir: Path, stem: str, hw: bool) -> dict[str, Any]:
    profile, _ = hc.save_profile(job, out_dir, stem)
    rec: dict[str, Any] = {"job": hc.job_info(job), "metrics": aihub_metrics(profile)}
    rows = per_layer(profile)
    write_json(out_dir / f"{stem}.per_layer.json", rows)
    rec["op_types"] = op_type_breakdown(profile)[:15]
    if hw:
        arts = hc.download_hw_artifacts(job, out_dir)
        rec["hw_artifacts"] = arts
        rec["hw"] = summarize_hw(arts, rows)
    return rec


def summarize_hw(arts: dict[str, list[str]], layer_rows: list[dict[str, Any]], bandwidth_gbps: float | None = None) -> dict[str, Any]:
    """Classify ops from the downloaded optrace / QHAS files (best effort)."""
    out: dict[str, Any] = {}
    for kind in ("qhas", "optrace"):
        for path in arts.get(kind, []):
            if not str(path).endswith(".json"):
                continue
            rs = hwtrace.best_record_set(path)
            if rs is None:
                continue
            rows = hwtrace.normalize(rs)
            hwtrace.merge_aihub_timing(rows, layer_rows)
            hwtrace.classify(rows, bandwidth_gbps)
            out[kind] = {"file": str(path), "record_path": rs.path, "fields": hwtrace.match_fields(rs.keys),
                         "classes": hwtrace.class_summary(rows)}
            write_json(Path(path).with_suffix(".ops.json"), rows)
    return out


def infer(
    images: Sequence[str | Path],
    bundle_dir: str | Path,
    out_dir: str | Path,
    labels: Sequence[str] | None = None,
    device_name: str = "Samsung Galaxy S25",
    qairt_version: str | None = None,
    options: str = "",
    hw: bool = True,
    logit_scale: float | None = None,
    logit_bias: float | None = None,
) -> dict[str, Any]:
    bundle = Bundle.load(bundle_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = hc.device(device_name)
    opts = _inference_options(bundle, qairt_version, options)
    paths = [Path(p) for p in images]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    print(f"SigLIP 2 {bundle.precision} {bundle.runtime} on {device_name} (AI Hub hosted), options: '{opts}'")
    img_embeds, img_rec = _run_component(bundle, "image_encoder", [preprocess(p) for p in paths], dev, out, opts, hw)
    E = l2n(np.stack(img_embeds))
    np.save(out / "image_embeddings.npy", E)
    result: dict[str, Any] = {
        "device": device_name,
        "bundle": str(bundle.root),
        "precision": bundle.precision,
        "runtime": bundle.runtime,
        "images": [str(p) for p in paths],
        "embedding_dim": int(E.shape[1]),
        "embeddings_file": str(out / "image_embeddings.npy"),
        "embedding_preview": {p.name: [round(float(v), 5) for v in e[:8]] for p, e in zip(paths, E)},
        "raw_norms": [float(np.linalg.norm(e)) for e in img_embeds],
        "image_cosine_similarity": (E @ E.T).round(4).tolist(),
        "image_encoder": img_rec,
    }
    if labels:
        ids = tokenize(labels)
        txt_embeds, txt_rec = _run_component(bundle, "text_encoder", [ids[i : i + 1] for i in range(len(labels))], dev,
                                             out, opts, hw)
        T = l2n(np.stack(txt_embeds))
        np.save(out / "text_embeddings.npy", T)
        if logit_scale is None or logit_bias is None:
            logit_scale, logit_bias = logit_scale_bias()
        logits = logit_scale * (E @ T.T) + logit_bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        result.update(
            labels=list(labels),
            text_encoder=txt_rec,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            logits=logits.round(4).tolist(),
            probabilities=probs.round(4).tolist(),
            best_label={p.name: labels[int(np.argmax(logits[i]))] for i, p in enumerate(paths)},
        )
    write_json(out / "results.json", result)
    md = infer_markdown(result)
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved embeddings and results to {out}")
    return result


def infer_markdown(r: dict[str, Any]) -> str:
    names = [Path(p).name for p in r["images"]]
    lines = [f"## SigLIP 2 ({r['precision']}, {r['runtime']}) on {r['device']}", ""]
    lines.append(f"Embeddings: {len(names)} x {r['embedding_dim']} (L2-normalised) -> `{r['embeddings_file']}`\n")
    for n in names:
        lines.append(f"- {n}: first values {r['embedding_preview'][n]}")
    if len(names) > 1:
        lines += ["", "Image-image cosine similarity:", "",
                  markdown_table(["", *names], [[n, *row] for n, row in zip(names, r["image_cosine_similarity"])], 3)]
    if r.get("labels"):
        lines += ["", "Zero-shot (sigmoid(scale * cos + bias)):", "",
                  markdown_table(["image", *r["labels"]], [[n, *row] for n, row in zip(names, r["probabilities"])], 3)]
    lines += ["", metrics_markdown("image_encoder (AI Hub metrics, this job)", r["image_encoder"]["metrics"])]
    if r.get("text_encoder"):
        lines += ["", metrics_markdown("text_encoder (AI Hub metrics, this job)", r["text_encoder"]["metrics"])]
    lines += ["", f"Job: {r['image_encoder']['job']['url']}"]
    return "\n".join(lines)


def profile(
    bundle_dir: str | Path,
    out_dir: str | Path,
    components: Sequence[str] = ("image_encoder",),
    device_name: str = "Samsung Galaxy S25",
    qairt_version: str | None = None,
    options: str = "",
    hw: bool = True,
) -> dict[str, Any]:
    """AI Hub profile job (random inputs) per component, like AI Hub's own scorecard."""
    bundle = Bundle.load(bundle_dir)
    out = Path(out_dir)
    dev = hc.device(device_name)
    opts = _inference_options(bundle, qairt_version, options)
    jobs = {}
    for comp in components:
        model = bundle.hub_model(comp)
        jobs[comp] = hc.submit_inference(model, dev, None, profile=True,
                                         name=f"hexprof siglip2 {comp} {bundle.precision} profile", options=opts)
        print(f"  {comp}: profile job {jobs[comp].job_id} -> {jobs[comp].url}")
    result: dict[str, Any] = {"device": device_name, "precision": bundle.precision, "runtime": bundle.runtime,
                              "options": opts, "components": {}}
    md = [f"## SigLIP 2 {bundle.precision} {bundle.runtime} on {device_name}"]
    for comp, job in jobs.items():
        rec = _collect_profile(job, out / comp, comp, hw)
        rec["published"] = published_siglip2(bundle.precision, comp, bundle.runtime)
        result["components"][comp] = rec
        md.append(metrics_markdown(f"{comp} (measured)", rec["metrics"]))
        if rec["published"]:
            pub = rec["published"]
            md.append(f"\nAI Hub published: {pub['inference_time_milliseconds']} ms, peak memory "
                      f"[{pub['estimated_peak_memory_range_mb']['min']}, {pub['estimated_peak_memory_range_mb']['max']}] MB, "
                      f"{pub['layer_counts']['npu']} NPU layers (job {pub['job_id']})")
        if rec.get("op_types"):
            md.append("\nTop op types by time:\n")
            md.append(markdown_table(["type", "unit", "count", "time us", "share"],
                                     [[o["type"], o["compute_unit"], o["count"], o["time_us"], o["share"]]
                                      for o in rec["op_types"][:10]], 3))
        md.append(f"\nJob: {rec['job']['url']}\n")
    write_json(out / "profile.json", result)
    text = "\n\n".join(md)
    (out / "profile.md").write_text(text, encoding="utf-8")
    print(text)
    return result
