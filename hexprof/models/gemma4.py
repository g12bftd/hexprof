"""Gemma 4 E4B on the phone through GenieX.

Qualcomm AI Hub publishes Gemma-4-E4B-it for the Galaxy S25 only on the GenieX
llama.cpp runtime (``geniex_llamacpp``, Q4_0 GGUF, NPU via the Hexagon HTP
backend of ggml). The compiled-QNN GenieX bundle (``geniex_qairt``, w4a16) is a
developer preview and currently not released for download; it can still be run
here from a bundle you export yourself (``hexprof pipeline gemma4``).

Two commands:

* :func:`infer` – your images + prompts, prints the answers and a per-stage
  latency breakdown (vision encoder, prefill of N tokens, decode of M tokens,
  end to end), from geniex-bench ``--accuracy`` runs.
* :func:`profile_aihub` – the exact sweep AI Hub uses for its published
  ``llm_metrics`` (context 512/4096, 128 generated tokens, 3 measured runs) and
  a side-by-side with the published S25 numbers.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from hexprof.geniex.bench import GenieXPhone
from hexprof.geniex.parse import (
    aihub_llm_metrics,
    parse_accuracy_stdout,
    parse_cell,
    stage_breakdown,
)
from hexprof.io_utils import fmt, markdown_table, write_json, write_yaml
from hexprof.reference import published_gemma4_llm_metric

MODEL_NAME = "Gemma-4-E4B-it"
# AI Hub Models manifest `runtime_technical_details.*.model_infer_id`.
DEFAULT_MODELS = {
    "llama_cpp": "google/gemma-4-E4B-it-qat-q4_0-gguf:Q4_0",
    "qairt": "ai-hub-models/Gemma-4-E4B-it",
}
AIHUB_RUNTIME = {"llama_cpp": "geniex_llamacpp", "qairt": "geniex_qairt"}

# AI Hub's GenieX perf sweep (qai_hub_models.utils.llm.geniex.jobs).
N_GEN = 128
AIHUB_CONTEXTS = (512, 4096)
AIHUB_REPETITIONS = 3


def default_model(plugin: str) -> str:
    return DEFAULT_MODELS[plugin]


def _sanitize_prompt(p: str) -> str:
    # geniex-bench splits --prompt-file on lines that are exactly '---'.
    return "\n".join("- - -" if ln.strip() == "---" else ln for ln in p.strip().splitlines())


@dataclass
class InferConfig:
    plugin: str = "llama_cpp"
    compute: str = "npu"
    model: str | None = None
    mmproj: str | None = None
    max_new_tokens: int = 256
    context_length: int = 4096
    repeats: int = 1
    warmup: int = 1
    system_prompt: str | None = None
    think: bool = False
    temperature: float = 0.0
    seed: int = 42
    power_mode: str | None = None
    max_gen_time_s: int | None = 600
    images_together: bool = False
    extra_args: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- setup


def setup(phone: GenieXPhone, version: str | None, plugin: str, model: str | None, mmproj: str | None,
          pull: bool = True, vlm: bool = True, refresh: bool = False) -> dict[str, Any]:
    """Install geniex-bench on the phone and (optionally) pull the model there."""
    info = phone.check_device()
    pkg = phone.install(version, refresh=refresh)
    out = {"device": info, "geniex_bench": pkg}
    if pull:
        cfg = InferConfig(plugin=plugin, model=model, mmproj=mmproj, max_new_tokens=4, repeats=1, warmup=0,
                          context_length=1024)
        with tempfile.TemporaryDirectory() as td:
            images: list[Path] = []
            if vlm:
                from PIL import Image

                img = Path(td) / "hexprof-probe.png"
                Image.new("RGB", (64, 64), (200, 30, 30)).save(img)
                images = [img]
            print("Pulling the model on the phone through GenieX's model manager (first time: several GB)...")
            res = infer(phone, cfg, images, ["Reply with one word."], Path(td) / "probe", quiet=True)
        if res["results"]:
            out["probe"] = {"answer": res["results"][0]["answer"], "model": res["model"],
                            "tool_versions": res["tool_versions"]}
    return out


# --------------------------------------------------------------------------- inference


def _bench_args(phone: GenieXPhone, cfg: InferConfig, model: str, remote_images: list[str],
                remote_prompt: str, remote_json: str, cell_id: str) -> list[str]:
    args = phone.model_args(cfg.plugin, model, cfg.mmproj)
    args += ["--device", cfg.compute, "--accuracy", "--prompt-file", remote_prompt,
             "-n", str(cfg.max_new_tokens), "-c", str(cfg.context_length),
             "--temperature", str(cfg.temperature), "--seed", str(cfg.seed),
             "--output-json", remote_json, "--cell-id", cell_id,
             "--think" if cfg.think else "--no-think"]
    if remote_images:
        args.append("--vlm")
        for img in remote_images:
            args += ["--image", img]
    if cfg.system_prompt:
        args += ["--system-prompt", cfg.system_prompt]
    if cfg.power_mode:
        args += ["--power-mode", cfg.power_mode]
    if cfg.max_gen_time_s:
        args += ["--max-gen-time-s", str(cfg.max_gen_time_s)]
    args += phone.common_args() + list(cfg.extra_args)
    return args


def infer(
    phone: GenieXPhone,
    cfg: InferConfig,
    images: Sequence[str | Path],
    prompts: Sequence[str],
    out_dir: Path,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run every prompt on every image (or all images together) and time each stage.

    Each (image, prompt) pair runs ``warmup + repeats`` times inside one
    geniex-bench process (one ``---`` segment per run, each on a fresh KV cache);
    warm-up runs are dropped from the statistics.
    """
    if not prompts:
        raise ValueError("at least one prompt is required")
    model = cfg.model or default_model(cfg.plugin)
    out_dir = Path(out_dir)
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    prompts = [_sanitize_prompt(p) for p in prompts]

    if not images:
        groups: list[list[Path]] = [[]]
    elif cfg.images_together:
        groups = [[Path(i) for i in images]]
    else:
        groups = [[Path(i)] for i in images]
    for g in groups:
        for img in g:
            if not img.exists():
                raise FileNotFoundError(img)

    per_run = cfg.warmup + cfg.repeats
    results: list[dict[str, Any]] = []
    tool_versions: dict[str, Any] = {}
    for gi, group in enumerate(groups):
        remote_images = [phone.push_input(i) for i in group]
        segments = [p for p in prompts for _ in range(per_run)]
        local_prompt = raw / f"group{gi}_prompts.txt"
        local_prompt.write_text("\n---\n".join(segments) + "\n", encoding="utf-8")
        remote_prompt = phone.push_input(local_prompt)
        cell = f"{MODEL_NAME}-{cfg.plugin}-{cfg.compute}-vqa{gi}"
        remote_json = f"{phone.results}/{cell}.json"
        remote_out, remote_err = f"{phone.results}/{cell}.stdout", f"{phone.results}/{cell}.stderr"
        phone.adb.shell(f"rm -f {remote_json} {remote_out} {remote_err}", check=False)
        args = _bench_args(phone, cfg, model, remote_images, remote_prompt, remote_json, cell)
        if not quiet:
            label = ", ".join(i.name for i in group) or "(text only)"
            print(f"[{gi + 1}/{len(groups)}] {label}: {len(prompts)} prompt(s) x {per_run} run(s) on {cfg.compute}")
        res = phone.run_bench(args, remote_out, remote_err)
        stdout, stderr = phone.read_remote(remote_out), phone.read_remote(remote_err)
        (raw / f"{cell}.stdout.txt").write_text(stdout, encoding="utf-8")
        (raw / f"{cell}.stderr.txt").write_text(stderr, encoding="utf-8")
        (raw / f"{cell}.cmd.txt").write_text("geniex-bench " + " ".join(args) + "\n", encoding="utf-8")
        if phone.adb.dry_run:
            continue
        if res.returncode != 0:
            raise RuntimeError(
                f"geniex-bench failed (exit {res.returncode}). stderr tail:\n{stderr[-3000:]}\n"
                f"Full logs in {raw}. Re-run `hexprof gemma4 setup` if the package or model is missing."
            )
        phone.adb.pull(remote_json, raw / f"{cell}.json")
        cell_data = parse_cell(raw / f"{cell}.json")
        segs = parse_accuracy_stdout(stdout)
        if len(cell_data.runs) != len(segments):
            print(f"WARNING: expected {len(segments)} runs, geniex-bench reported {len(cell_data.runs)}")
        for pi, prompt in enumerate(prompts):
            idx = list(range(pi * per_run, (pi + 1) * per_run))
            measured = [i for i in idx[cfg.warmup:] if i < len(cell_data.runs)]
            texts = [segs[i]["text"] for i in idx if i < len(segs)]
            answer = segs[idx[cfg.warmup]]["text"] if idx[cfg.warmup] < len(segs) else (texts[0] if texts else "")
            results.append(
                {
                    "images": [str(i) for i in group],
                    "prompt": prompt,
                    "answer": answer,
                    "stop_reason": segs[idx[cfg.warmup]]["stop_reason"] if idx[cfg.warmup] < len(segs) else None,
                    "consistent_across_runs": len(set(texts)) <= 1,
                    "breakdown": stage_breakdown([cell_data.runs[i] for i in measured]),
                    "runs": [cell_data.runs[i].to_dict() | {"warmup": i in idx[: cfg.warmup]} for i in idx
                             if i < len(cell_data.runs)],
                }
            )
        tool_versions = cell_data.versions

    summary = {
        "model": model,
        "plugin": cfg.plugin,
        "aihub_runtime": AIHUB_RUNTIME[cfg.plugin],
        "compute": cfg.compute,
        "device": phone.target.hub_name,
        "config": cfg.__dict__,
        "tool_versions": tool_versions,
        "results": results,
    }
    if not phone.adb.dry_run:
        write_json(out_dir / "results.json", summary)
        (out_dir / "results.md").write_text(infer_markdown(summary), encoding="utf-8")
    if not quiet and results:
        print(infer_markdown(summary))
        print(f"\nSaved to {out_dir}/results.json and results.md (raw geniex-bench logs in {raw})")
    return summary


def infer_markdown(summary: dict[str, Any]) -> str:
    lines = [f"## {MODEL_NAME} on {summary['device']} — GenieX {summary['plugin']} / {summary['compute']}",
             f"Model: `{summary['model']}`  Versions: {summary.get('tool_versions') or '-'}", ""]
    for r in summary["results"]:
        b = r["breakdown"]
        imgs = ", ".join(Path(i).name for i in r["images"]) or "(no image)"
        lines += [f"**Image:** {imgs}", f"**Prompt:** {r['prompt']}", "", "**Answer:**", "", r["answer"] or "(empty)", ""]

        def med(key: str, b: dict[str, Any] = b) -> Any:
            return (b.get(key) or {}).get("median")

        rows = [
            ["Vision encoder", med("vision_encoder_ms"), "-", "-"],
            ["Prefill", med("prefill_ms"), fmt(med("prefill_tokens"), 0) + " tokens (N)", med("prefill_tps")],
            ["Decode", med("decode_ms"), fmt(med("decode_tokens"), 0) + " tokens (M)", med("decode_tps")],
            ["Time to first token", med("ttft_ms"), "-", "-"],
            ["End to end", med("e2e_ms"), "-", "-"],
        ]
        lines.append(markdown_table(["Stage", f"Median ms (n={b['n_runs']})", "Tokens", "tok/s"], rows, digits=1))
        if not r["consistent_across_runs"]:
            lines.append("\n_Note: generated text differed between runs._")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- AI Hub perf sweep


def _perf_prompt_file(ctx: int) -> Path | None:
    """AI Hub's qairt perf prompt (largest shipped size <= ctx), if qai_hub_models is installed."""
    try:
        import qai_hub_models.utils.llm as llm_utils  # type: ignore[import-not-found]
    except ImportError:
        return None
    d = Path(llm_utils.__file__).parent / "perf_prompts"
    sizes = sorted(int(p.stem.rsplit("_", 1)[1]) for p in d.glob("sample_prompt_*.txt"))
    if not sizes:
        return None
    size = max((s for s in sizes if s <= ctx), default=sizes[0])
    return d / f"sample_prompt_{size}.txt"


def profile_aihub(
    phone: GenieXPhone,
    out_dir: Path,
    plugin: str = "llama_cpp",
    model: str | None = None,
    computes: Sequence[str] = ("npu",),
    contexts: Sequence[int] = AIHUB_CONTEXTS,
    repetitions: int = AIHUB_REPETITIONS,
    n_gen: int = N_GEN,
    power_mode: str | None = None,
    qairt_lib: str | None = None,
) -> dict[str, Any]:
    """Reproduce AI Hub's GenieX ``llm_metrics`` sweep on the phone."""
    model = model or default_model(plugin)
    out_dir = Path(out_dir)
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    model_args = phone.model_args(plugin, model)  # pushes host files; ["--plugin", p, "-m", value]
    remote_model = model_args[model_args.index("-m") + 1]
    cells = []
    for ctx in contexts:
        rows = [
            f"{MODEL_NAME}-{plugin}-{c}-c{ctx}\t{plugin}\t{c}\t{remote_model}\t\t\t\t" for c in computes
        ]
        tsv = raw / f"matrix-{ctx}.tsv"
        tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
        remote_tsv = phone.push_input(tsv)
        remote_dir = f"{phone.results}/aihub-c{ctx}"
        phone.adb.shell(f"rm -rf {remote_dir} && mkdir -p {remote_dir}", check=False)
        args = ["--matrix-file", remote_tsv, "--output-json-dir", remote_dir, "-r", str(repetitions)]
        if qairt_lib:
            args += ["--qairt-lib", qairt_lib]
        if plugin == "qairt" and (pf := _perf_prompt_file(ctx)) is not None:
            args += ["--prompt-file", phone.push_input(pf), "--chat-template", "-c", str(ctx + n_gen), "-n", str(n_gen)]
        else:
            args += ["-c", str(ctx + n_gen), "-p", str(ctx - n_gen), "-n", str(n_gen)]
        if power_mode:
            args += ["--power-mode", power_mode]
        args += phone.common_args()
        print(f"context {ctx}: {', '.join(computes)} x {repetitions} runs (+1 warm-up) ...")
        remote_out, remote_err = f"{remote_dir}/stdout.txt", f"{remote_dir}/stderr.txt"
        res = phone.run_bench(args, remote_out, remote_err)
        (raw / f"c{ctx}.cmd.txt").write_text("geniex-bench " + " ".join(args) + "\n", encoding="utf-8")
        if phone.adb.dry_run:
            continue
        stdout, stderr = phone.read_remote(remote_out), phone.read_remote(remote_err)
        (raw / f"c{ctx}.stdout.txt").write_text(stdout, encoding="utf-8")
        (raw / f"c{ctx}.stderr.txt").write_text(stderr, encoding="utf-8")
        listing = phone.adb.shell(f"ls -1 {remote_dir}", check=False).stdout.split()
        got = 0
        for name in listing:
            if name.endswith(".json"):
                local = raw / f"c{ctx}" / name
                if phone.adb.pull(f"{remote_dir}/{name}", local):
                    cells.append(parse_cell(local))
                    got += 1
        if res.returncode != 0 or got == 0:
            print(f"WARNING: geniex-bench exit {res.returncode}, {got} cell(s). stderr tail:\n{stderr[-2000:]}")

    metrics = [aihub_llm_metrics(c) for c in cells]
    metrics.sort(key=lambda m: (m["context_length"] or 0, m["desired_compute_unit"]), reverse=True)
    runtime = AIHUB_RUNTIME[plugin]
    versions = next((m["tool_versions"] for m in metrics if m["tool_versions"]), {})
    perf = {
        phone.target.hub_name: {
            runtime: {
                "llm_metrics": [
                    {k: m[k] for k in ("context_length", "desired_compute_unit", "prefill_tokens_per_second",
                                       "time_to_first_token_range_milliseconds", "tokens_per_second")}
                    for m in metrics
                ],
                "tool_versions": {"geniex": versions.get("geniex_version"),
                                  "qairt": versions.get("qairt_version"),
                                  "llama_cpp": versions.get("llama_cpp_version")},
            }
        }
    }
    summary = {"model": model, "plugin": plugin, "metrics": metrics, "perf_yaml": perf}
    if not phone.adb.dry_run:
        write_yaml(out_dir / "perf.yaml", perf)
        write_json(out_dir / "metrics.json", summary)
        md = profile_markdown(summary, runtime)
        (out_dir / "metrics.md").write_text(md, encoding="utf-8")
        print(md)
        print(f"\nSaved perf.yaml (AI Hub Models format), metrics.json and metrics.md to {out_dir}")
    return summary


def profile_markdown(summary: dict[str, Any], runtime: str) -> str:
    rows = []
    for m in summary["metrics"]:
        ref = published_gemma4_llm_metric(m["context_length"], m["desired_compute_unit"], runtime) or {}
        tr = m["time_to_first_token_range_milliseconds"] or {}
        rtr = ref.get("time_to_first_token_range_milliseconds") or {}
        rows.append([
            m["context_length"], m["desired_compute_unit"],
            m["tokens_per_second"], ref.get("tokens_per_second"),
            m["prefill_tokens_per_second"], ref.get("prefill_tokens_per_second"),
            f"{fmt(tr.get('min'), 0)}–{fmt(tr.get('max'), 0)}", f"{fmt(rtr.get('min'), 0)}–{fmt(rtr.get('max'), 0)}",
        ])
    head = ["ctx", "unit", "decode tok/s", "AI Hub", "prefill tok/s", "AI Hub", "TTFT ms", "AI Hub TTFT ms"]
    return f"## {MODEL_NAME} `{summary['model']}` ({runtime}) vs AI Hub published (S25)\n\n" + markdown_table(head, rows, 1)


def load_results(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
