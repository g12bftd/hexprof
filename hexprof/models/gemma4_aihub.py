"""Gemma 4 E4B on AI Hub Workbench's hosted Galaxy S25 — no phone, no GenieX.

What Workbench can and cannot do for an LLM
-------------------------------------------
Workbench runs one compiled graph per job with no state between jobs. Gemma 4
is compiled (by Qualcomm's own recipe) into a vision encoder plus 4 text parts,
each with a 128-token prefill graph and a 1-token decode graph. Generating text
would need 4 chained cloud jobs per token, so it is not done here. Instead:

1. One-time **export** (Qualcomm's ``qai-hub-models export gemma_4_e4b_it`` with
   the published w4a16 checkpoint): compile + link every graph and profile each
   on the S25 -> real per-graph latency / memory / optrace.
2. Per request, with **your image and prompt**:

   * your image goes through Gemma 4's processor (resized to the compiled
     448x448 geometry, exactly like Qualcomm's VEG calibration) and the compiled
     vision encoder runs on it on the S25 (real inference job, profiled);
   * the same processor call gives N, the exact number of prompt tokens
     (text + image soft tokens) the compiled model would prefill;
   * stage times: vision encoder (measured on your image), prefill =
     ceil(N/128) x the 4 prefill graphs, decode = M x the 4 decode graphs.
     This is NPU graph time; host-side work (embedding lookup, sampling) is not
     included.
3. Optionally, the **answer text** is generated on this computer by llama.cpp
   with the same Q4_0 GGUF GenieX runs on the S25 (CPU numerics can differ
   slightly from the NPU's, so the text may not match the phone word for word).
   Its token count is used as M.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hexprof.aihub import client as hc
from hexprof.aihub.metrics import aihub_metrics
from hexprof.io_utils import markdown_table, read_json, write_json

HF_REPO = "google/gemma-4-E4B-it"
LOCAL_GGUF = "google/gemma-4-E4B-it-qat-q4_0-gguf:Q4_0"  # what GenieX runs on the S25
VEG_IMAGE_SIZE = 448  # Qualcomm traces the vision encoder at 448x448
EXPORT_STATE = "export_profiles.json"
_RESPONSE_MARKER = "<HEXPROF_RESPONSE_START>"
_GRAPH_RE = re.compile(r"(prompt|token)_ar(\d+)_cl(\d+)")


# --------------------------------------------------------------------------- export


def default_export_dir(context_length: int | None = None, prompt_length: int | None = None) -> Path:
    """``gemma4-s25-export`` for Qualcomm's settings; a separate folder per changed context / prompt length."""
    base = os.environ.get("HEXPROF_DOWNLOAD_DIR")
    name = "gemma4-s25-export" + (f"-cl{context_length}" if context_length else "") + (
        f"-ar{prompt_length}" if prompt_length else "")
    return Path(base) / name if base else Path("runs") / name


def export_flags(context_length: int | None = None, prompt_length: int | None = None) -> list[str]:
    """Qualcomm export flags for a changed KV-cache length / prompt chunk (None = Qualcomm's 4096 / 128)."""
    flags: list[str] = []
    if context_length:
        flags += ["--context-lengths", str(int(context_length))]
    if prompt_length:
        flags += ["--sequence-lengths", str(int(prompt_length)), "1"]
    return flags


def ensure_export(export_dir: str | Path, device_name: str, checkpoint: str = "DEFAULT",
                  target_runtime: str = "geniex_qairt", hw: bool = True, force: bool = False,
                  context_length: int | None = None, prompt_length: int | None = None) -> dict[str, Any]:
    """Compile + profile Gemma 4 on AI Hub once; later calls reuse ``export_profiles.json``."""
    export_dir = Path(export_dir)
    state = export_dir / EXPORT_STATE
    flags = export_flags(context_length, prompt_length)
    if state.exists() and not force:
        summary = read_json(state)
        if summary.get("parts"):
            had = summary.get("export_flags", [])
            if had != flags:
                raise ValueError(f"{export_dir} holds an export made with flags {had or '(none)'}, not {flags or '(none)'}; "
                                 "use another --export-dir or --force-export")
            print(f"Re-using the compiled Gemma 4 graphs profiled in {export_dir}")
            return summary
    from hexprof import runlog
    from hexprof.pipeline import gemma4_export

    gemma4_export._require_qaihm()
    print("One-time step: compiling Gemma 4 E4B (w4a16) on AI Hub and profiling every graph on "
          f"{device_name}. This downloads Qualcomm's 18.7 GB checkpoint and can take a few hours.")
    extra = ["--skip-downloading", *flags]
    model, settings, info = gemma4_export.manifest(checkpoint, export_dir, device_name, target_runtime, "w4a16", extra)
    with runlog.run(export_dir, kind="gemma4-export", title="Gemma 4 E4B on AI Hub: Qualcomm export, every graph profiled",
                    model=model, settings=settings, device=device_name, runtime=info["runtime"],
                    extra=info["extra"]) as run:
        result = gemma4_export.run_official_export(checkpoint, export_dir, device_name, target_runtime,
                                                   "w4a16", extra=extra)
        summary = gemma4_export.collect(result, export_dir, hw)
        summary.update(device=device_name, checkpoint=checkpoint, target_runtime=target_runtime, export_flags=flags)
        write_json(state, summary)
        run.result(**gemma4_export.latency_results(summary))
    return summary


def split_parts(summary: dict[str, Any]) -> dict[str, Any]:
    """Group profiled graphs into vision encoder / prefill / decode."""
    out: dict[str, Any] = {"vision": None, "prefill": [], "decode": [], "seq_len": None, "context_length": None}
    for label, rec in summary["parts"].items():
        comp = rec.get("component", label.split("/")[0])
        graph = rec.get("graph", label.split("/")[-1])
        m = _GRAPH_RE.search(graph)
        if "vision" in comp:
            out["vision"] = rec
        elif m and m.group(1) == "prompt":
            out["prefill"].append(rec)
            out["seq_len"] = int(m.group(2))
            out["context_length"] = int(m.group(3))
        elif m and m.group(1) == "token":
            out["decode"].append(rec)
    return out


def _ms(rec: dict[str, Any]) -> float:
    return float(rec["metrics"]["inference_time_milliseconds"])


# --------------------------------------------------------------------------- per-request


def _processor(hf_repo: str) -> Any:
    try:
        from transformers import AutoProcessor
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Gemma 4's processor needs transformers: `qai-hub-models install gemma_4_e4b_it` "
                           "or pip install 'hexprof[siglip-text]'") from e
    return AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)


def _as_np(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def prepare(processor: Any, image_path: str | Path, prompt: str, image_size: int = VEG_IMAGE_SIZE) -> dict[str, Any]:
    """Gemma 4 chat template on (image, prompt): token count + the vision encoder's inputs."""
    from PIL import Image

    with Image.open(image_path) as im:
        original = list(im.size)
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                           add_generation_prompt=True)
    ids = _as_np(inputs["input_ids"])
    return {
        "prompt_tokens": int(ids.shape[-1]),
        "image_size_original": original,
        "image_size_fed": [image_size, image_size],
        "pixel_values": _as_np(inputs["pixel_values"]).astype(np.float32),
        # AI Hub runs int64 graph inputs as int32 (as qai_hub_models does)
        "image_position_ids": _as_np(inputs["image_position_ids"]).astype(np.int32),
    }


def run_vision_encoder(vision: dict[str, Any], prepared: dict[str, Any], device_name: str,
                       out_dir: Path, stem: str) -> dict[str, Any]:
    """Real AI Hub inference (profiled) of the compiled vision encoder on the user's image."""
    model = hc.get_model(vision["model_id"])
    data = {"pixel_values": [prepared["pixel_values"]], "image_position_ids": [prepared["image_position_ids"]]}
    job = hc.submit_inference(model, hc.device(device_name), data, profile=True,
                              name=f"hexprof gemma4 vision encoder {stem}", options=vision.get("options", ""))
    print(f"  vision encoder on your image: job {job.job_id} -> {job.url}")
    hc.wait_success(job, "vision encoder inference")
    outputs = job.download_output_data() or {}
    emb = None
    if outputs:
        key = "vision_embedding" if "vision_embedding" in outputs else next(iter(outputs))
        emb = np.asarray(outputs[key][0], dtype=np.float32)
        np.save(out_dir / f"{stem}.vision_embedding.npy", emb)
    prof, _ = hc.save_profile(job, out_dir / "profiles", f"{stem}.vision_encoder")
    m = aihub_metrics(prof)
    return {
        "job": hc.job_info(job),
        "metrics": m,
        "embedding_shape": list(emb.shape) if emb is not None else None,
        "image_tokens": int(emb.shape[-2]) if emb is not None and emb.ndim >= 2 else None,
        "embedding_file": str(out_dir / f"{stem}.vision_embedding.npy") if emb is not None else None,
    }


def find_llama_cli(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    return shutil.which("llama-mtmd-cli")


def local_answer(llama_cli: str, image: str | Path, prompt: str, gguf: str = LOCAL_GGUF, mmproj: str | None = None,
                 max_new_tokens: int = 256, context_length: int = 4096, timeout: float | None = 1800) -> dict[str, Any]:
    """Answer with llama.cpp (``llama-mtmd-cli -hf <gguf> --image ... -p ...``, greedy) on this computer."""
    cmd = [llama_cli, "-hf" if not Path(gguf).exists() else "-m", gguf, "--image", str(image), "-p", prompt,
           "--temp", "0", "--seed", "42", "-n", str(max_new_tokens), "-c", str(context_length), "--jinja"]
    if mmproj:
        cmd += ["--mmproj", mmproj]
    env = dict(os.environ, MTMD_TEST_RESPONSE_MARKER=_RESPONSE_MARKER)
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env, timeout=timeout, check=False)
    out, err = proc.stdout, proc.stderr
    text = out.split(_RESPONSE_MARKER, 1)[1] if _RESPONSE_MARKER in out else out
    m = re.search(r"\beval time =\s*[\d.]+ ms /\s*(\d+) runs", err)
    rec = {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "answer": text.strip(),
        "generated_tokens": int(m.group(1)) if m else None,
    }
    if proc.returncode != 0:
        rec["error"] = err[-2000:]
    return rec


def estimate(parts: dict[str, Any], prompt_tokens: int, gen_tokens: int, vision_ms: float | None) -> dict[str, Any]:
    seq = parts["seq_len"] or 128
    prefill_chunk_ms = sum(_ms(r) for r in parts["prefill"])
    decode_token_ms = sum(_ms(r) for r in parts["decode"])
    chunks = math.ceil(prompt_tokens / seq)
    vis = vision_ms if vision_ms is not None else (_ms(parts["vision"]) if parts["vision"] else 0.0)
    prefill_ms = chunks * prefill_chunk_ms
    decode_ms = gen_tokens * decode_token_ms
    return {
        "vision_encoder_ms": vis,
        "prefill_tokens": prompt_tokens,
        "prefill_chunks": chunks,
        "prefill_chunk_tokens": seq,
        "prefill_ms": prefill_ms,
        "prefill_tps": prompt_tokens / (prefill_ms / 1e3) if prefill_ms else None,
        "decode_tokens": gen_tokens,
        "decode_ms_per_token": decode_token_ms,
        "decode_ms": decode_ms,
        "decode_tps": 1e3 / decode_token_ms if decode_token_ms else None,
        "ttft_ms": vis + prefill_ms,
        "e2e_ms": vis + prefill_ms + decode_ms,
    }


def infer(
    images: Sequence[str | Path],
    prompts: Sequence[str],
    out_dir: str | Path,
    device_name: str = "Samsung Galaxy S25",
    export_dir: str | Path | None = None,
    checkpoint: str = "DEFAULT",
    hw: bool = True,
    max_new_tokens: int = 256,
    local: str = "auto",
    llama_cli: str | None = None,
    gguf: str = LOCAL_GGUF,
    mmproj: str | None = None,
    hf_repo: str = HF_REPO,
    context_length: int | None = None,
    prompt_length: int | None = None,
) -> dict[str, Any]:
    if not images or not prompts:
        raise ValueError("give at least one --image and one --prompt")
    for i in images:
        if not Path(i).exists():
            raise FileNotFoundError(i)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_dir = export_dir or default_export_dir(context_length, prompt_length)
    summary = ensure_export(export_dir, device_name, checkpoint, hw=hw, context_length=context_length,
                            prompt_length=prompt_length)
    from hexprof import runlog

    with runlog.run(out, kind="gemma4-aihub", title="Gemma 4 E4B on AI Hub: your images and prompts",
                    model={"name": "Gemma 4 E4B-it (google/gemma-4-E4B-it)",
                           "weights": f"the Qualcomm export in {export_dir} (checkpoint {checkpoint})"},
                    settings=[runlog.setting("export flags", " ".join(export_flags(context_length, prompt_length)) or
                                             "(none)", "(none)", group="export")],
                    device=device_name, runtime="QNN context binary (vision encoder run on your images)",
                    extra={"images": [str(i) for i in images], "prompts": len(prompts), "max_new_tokens": max_new_tokens}) as run:
        report = _infer(images, prompts, out, device_name, export_dir, summary, local, llama_cli, gguf, mmproj, hf_repo,
                        max_new_tokens)
        run.result(requests=len(report["results"]), context_length=report["context_length"])
    return report


def _infer(images: Sequence[str | Path], prompts: Sequence[str], out: Path, device_name: str, export_dir: str | Path,
           summary: dict[str, Any], local: str, llama_cli: str | None, gguf: str, mmproj: str | None, hf_repo: str,
           max_new_tokens: int) -> dict[str, Any]:
    parts = split_parts(summary)
    if not parts["prefill"] or not parts["decode"]:
        raise RuntimeError("the export has no prefill/decode profiles; re-run with --force-export")
    cli = find_llama_cli(llama_cli) if local != "off" else None
    if local == "on" and cli is None:
        raise RuntimeError("llama-mtmd-cli not found (install llama.cpp, or pass --llama-cli PATH)")
    processor = _processor(hf_repo)
    size = int(summary.get("veg_image_size") or VEG_IMAGE_SIZE)

    results = []
    for ii, image in enumerate(images):
        stem = f"img{ii}_{Path(image).stem}"
        first = prepare(processor, image, prompts[0], size)
        vision = run_vision_encoder(parts["vision"], first, device_name, out, stem) if parts["vision"] else None
        for prompt in prompts:
            prep = first if prompt == prompts[0] else prepare(processor, image, prompt, size)
            ans = local_answer(cli, image, prompt, gguf, mmproj, max_new_tokens) if cli else None
            m = (ans or {}).get("generated_tokens") or max_new_tokens
            est = estimate(parts, prep["prompt_tokens"], m,
                           vision["metrics"]["inference_time_milliseconds"] if vision else None)
            est["decode_tokens_source"] = "local answer" if ans and ans.get("generated_tokens") else "--max-new-tokens"
            results.append({"image": str(image), "prompt": prompt, "answer": ans, "vision_encoder": vision,
                            "breakdown": est, "image_size_original": prep.get("image_size_original"),
                            "image_size_fed": prep.get("image_size_fed")})
    report = {
        "device": device_name,
        "export_dir": str(export_dir),
        "graphs": {
            "vision_encoder": parts["vision"]["metrics"] if parts["vision"] else None,
            "prefill": {r["graph"]: r["metrics"] for r in parts["prefill"]},
            "decode": {r["graph"]: r["metrics"] for r in parts["decode"]},
        },
        "context_length": parts["context_length"],
        "local_answer_model": gguf if cli else None,
        "results": results,
    }
    write_json(out / "results.json", report)
    md = markdown_report(report)
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md)
    if cli is None and local != "off":
        print("\n(No answer text: llama-mtmd-cli not found. Install llama.cpp to also see Gemma 4's answer.)")
    print(f"\nSaved to {out}/results.json and results.md")
    return report


def markdown_report(r: dict[str, Any]) -> str:
    lines = [f"## Gemma 4 E4B (w4a16, compiled QNN) on {r['device']} via AI Hub", ""]
    for res in r["results"]:
        b = res["breakdown"]
        lines += [f"**Image:** {Path(res['image']).name}", f"**Prompt:** {res['prompt']}", ""]
        ans = res.get("answer")
        if ans:
            lines += ["**Answer** (llama.cpp on this computer, same Q4_0 GGUF as GenieX on the S25):", "",
                      ans["answer"] or f"(failed: {ans.get('error', '')[-300:]})", ""]
        rows = [
            ["Vision encoder (your image, measured)", b["vision_encoder_ms"], "-", "-"],
            [f"Prefill ({b['prefill_chunks']} x {b['prefill_chunk_tokens']}-token graph)", b["prefill_ms"],
             f"{b['prefill_tokens']} (N)", b["prefill_tps"]],
            [f"Decode ({b['decode_ms_per_token']:.2f} ms/token)", b["decode_ms"],
             f"{b['decode_tokens']} (M, {b['decode_tokens_source']})", b["decode_tps"]],
            ["Time to first token", b["ttft_ms"], "-", "-"],
            ["End to end", b["e2e_ms"], "-", "-"],
        ]
        lines += [markdown_table(["Stage (NPU graph time on the S25)", "ms", "Tokens", "tok/s"], rows, 1), ""]
        if res.get("vision_encoder"):
            lines.append(f"Vision encoder job: {res['vision_encoder']['job']['url']}")
        lines.append("")
    g = r["graphs"]
    rows = [[name, m["inference_time_milliseconds"], f"[{m['estimated_peak_memory_range_mb']['min']}, "
             f"{m['estimated_peak_memory_range_mb']['max']}]" if m.get("estimated_peak_memory_range_mb") else "-",
             m["layer_counts"]["npu"], m["layer_counts"]["total"]]
            for name, m in [("vision_encoder", g["vision_encoder"])] + sorted(g["prefill"].items())
            + sorted(g["decode"].items()) if m]
    lines += ["### Profiled graphs (AI Hub metrics)", "",
              markdown_table(["graph", "inference ms", "peak MB", "NPU layers", "layers"], rows, 3)]
    return "\n".join(lines)
