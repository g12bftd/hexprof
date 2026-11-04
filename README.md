# hexprof

Run, profile and bound vision-language workloads on the **Snapdragon Hexagon NPU of the Samsung Galaxy S25**
(Snapdragon 8 Elite for Galaxy, SM8750-AC, Hexagon v79), using the same runtimes and the same metric
definitions as Qualcomm AI Hub.

The package implements the plan in *"Gemma 4 on the Galaxy S25: results, hardware ceilings and next steps"*:

| Plan section | Command |
|---|---|
| Results: Gemma 4 VQA end to end = vision encoder + prefill (N tokens) + decode (M tokens) | `hexprof gemma4 infer` (phone), `hexprof gemma4 aihub` (no phone) |
| Results: SigLIP 2 image encoder, W8A16, AI Hub's own asset | `hexprof siglip2 infer` / `profile` |
| AI Hub-consistent metrics | `hexprof gemma4 profile`, `hexprof hub metrics` |
| Compute ceiling: `latency = t0 + ops / P` per precision | `hexprof ceilings compute` |
| Bandwidth ceiling: `latency = t0 + bytes / B`, width and depth sweeps | `hexprof ceilings bandwidth` |
| Empty vs. full KV cache check | `hexprof ceilings kv-check` |
| Hardware metrics report: QHAS (QNN ops, HTP ops, HMX/HVX, DRAM/VTCM, critical path), optrace timelines, AI Hub per-layer, as HTML + CSV | `hexprof hw report` |
| Per-op compute-, memory-bound or overhead | `hexprof hw classify` |
| Pipeline: fine-tuned Gemma 4 → QNN context binaries → profile on the S25 | `hexprof pipeline gemma4 quantize` / `export` |

## How each model reaches the NPU

* **Gemma 4 E4B → GenieX.** AI Hub ships Gemma-4-E4B-it for the S25 only on the GenieX llama.cpp
  runtime (`geniex_llamacpp`, Q4_0 GGUF `google/gemma-4-E4B-it-qat-q4_0-gguf`). Its NPU path uses
  ggml's Hexagon HTP backend, not a compiled QNN graph. AI Hub Workbench has no hosted GenieX jobs, so
  hexprof drives Qualcomm's `geniex-bench` on **your phone over adb**, with the same package,
  on-device layout, environment and sweep that AI Hub Models uses for its published S25 numbers.
  The compiled-QNN GenieX bundle (`geniex_qairt`, w4a16) is a developer preview and is not
  downloadable at the moment (Qualcomm disabled it after a scorecard accuracy failure). You can
  build one yourself with `hexprof pipeline gemma4 …` and run it with `--plugin qairt`.
* **SigLIP 2 → AI Hub.** GenieX only serves LLMs and VLMs, so it cannot run a standalone image
  encoder. SigLIP 2 runs the way AI Hub publishes it: the pre-built **W8A16 QAIRT DLC** from AI Hub
  Models (`siglip2-base-patch16-224`), executed on AI Hub's hosted Galaxy S25 (QNN HTP = NPU). No
  phone is needed for this path. AI Hub's published figure for it is **3.319 ms**
  (job `jpx0kk1jp`), the 3.32 ms quoted in the plan.

## No phone? Everything on AI Hub's hosted Galaxy S25

You only need an AI Hub account. Gemma 4 and SigLIP 2 then both run on AI Hub's hosted S25 with
**your own image and prompt**:

| | Gemma 4 E4B (`hexprof gemma4 aihub`) | SigLIP 2 (`hexprof siglip2 infer`) |
|---|---|---|
| Runs on the S25 | vision encoder on your image (real job) + every compiled text graph profiled | image encoder on your image (real job) |
| Model | Qualcomm's w4a16 compiled QNN graphs (their official export) | Qualcomm's W8A16 QAIRT DLC |
| You get | per-stage time on the S25 for your input: vision encoder, prefill of your N tokens, decode per token, TTFT, end to end; AI Hub metrics for every graph | embeddings, image similarity, zero-shot label scores, AI Hub metrics |
| Answer text | generated on your computer by llama.cpp with the same Q4_0 GGUF GenieX runs on the S25 (optional) | n/a |

Why Gemma 4 cannot answer *on* AI Hub: Workbench runs one compiled graph per job with no state between
jobs, and Gemma 4 needs its 4 text parts chained with the KV cache for every generated token (4 cloud
jobs per token). So Workbench measures every graph, and your image and prompt decide how many 128-token
prefill chunks are needed. The text-graph timings do not depend on content; only the token count does.

The first `gemma4 aihub` run (or `--export-only`) compiles Gemma 4 with Qualcomm's recipe and profiles
each graph on the S25. That step downloads Qualcomm's published w4a16 checkpoint (18.7 GB), needs
about 32 GB of RAM (Linux or WSL2 recommended; no GPU) and can take a few hours. Every later run reuses
it and only submits one vision-encoder job per image.

## Where downloads go

Add `--download-dir /path/to/dir` to any command (or `export HEXPROF_DOWNLOAD_DIR=/path/to/dir`):

| What | Location |
|---|---|
| SigLIP 2 model bundle | `<dir>/siglip2-qnn_dlc-w8a16/` |
| Gemma 4 checkpoint and AI Hub Models cache | `<dir>/.qaihm/` |
| Gemma 4 compile/profile results (reused) | `<dir>/gemma4-s25-export/` |
| Hugging Face files (processors, tokenizers) | `<dir>/huggingface/` |
| llama.cpp model for the local answer | `<dir>/llama.cpp/` |
| geniex-bench packages (phone path) | `<dir>/hexprof/` |

Results always go to `--out` (default `runs/<command>-<time>/`).

## Install

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e '.[assets,siglip-text,plots]'
```

```bash
qai-hub configure --api_token <YOUR_AI_HUB_TOKEN>
```

Get the token at workbench.aihub.qualcomm.com → Account → Settings → API Token. Gemma 4 additionally
needs Android platform-tools (`adb`), the S25 connected over USB with USB debugging enabled, and
internet on the phone (GenieX downloads the model there). If the phone has no internet, use
`--model /path/model.gguf --mmproj /path/mmproj.gguf` and hexprof pushes the files. Set `HF_TOKEN` if
your Hugging Face access needs one; it is forwarded to the phone and never logged.

```bash
hexprof doctor
```

## Gemma 4 on AI Hub (no phone)

```bash
pip install -e '.[pipeline]'
```

```bash
qai-hub-models install gemma_4_e4b_it
```

```bash
hexprof gemma4 aihub --download-dir ~/hexprof-data --image photo.jpg --prompt "What is in this picture?"
```

* Prompts: repeat `--prompt`, or use `--prompts-file`. Images: repeat `--image`, or pass a folder.
* The answer text needs `llama-mtmd-cli` from llama.cpp (macOS: `brew install llama.cpp`; others:
  github.com/ggml-org/llama.cpp/releases). Without it you still get the S25 timings; decode is then
  estimated for `--max-new-tokens` tokens. `--local-answer off` skips it.
* Output in `runs/gemma4-aihub-<time>/`: `results.md` with the answer and the stage table, `results.json`,
  and the vision encoder's output embeddings as `.npy`.

### How well each part uses the NPU

```bash
hexprof gemma4 efficiency
```

For every compiled part (after the export): operations and bytes per run (from the model config and
Qualcomm's w4a16 recipe), achieved TOPS and GB/s, operations per byte, the whole-model decode and
prefill rates, and QHAS's graph-level fields ("% of overall timeline HTP utilization", inferences per
second). Those QHAS numbers are time-based: how much of the run the NPU was busy, and how many runs fit
in a second. They are not compute or bandwidth utilisation. Add `--compute-json` / `--bandwidth-json`
from `hexprof ceilings` (or `--peak-tops` / `--peak-gbps`) for % of peak and the roofline bound per part,
and `--measure-binaries` to cross-check bytes against the compiled binaries.

## Gemma 4 on the S25 (GenieX, NPU)

### 1. One-time setup

This pushes `geniex-bench` (the Android release package, v0.10.0 at the time of writing) to
`/data/local/tmp/hexprof` and pulls Gemma 4 E4B Q4_0 with its vision projector onto the phone
(about 5 GB on the first run).

```bash
hexprof gemma4 setup
```

### 2. Inference with your images and prompts

```bash
hexprof gemma4 infer --image photos/street.jpg --image photos/kitchen.jpg --prompt "How many people are in the picture?" --prompt "Describe the scene in one sentence."
```

* `--image` takes files or folders and can be repeated. Each image is asked every prompt. Add
  `--images-together` to send all images in a single turn.
* `--prompts-file examples/prompts.txt` reads one prompt per blank-line-separated paragraph.
* Each (image, prompt) pair runs `--warmup 1` + `--repeats 3` times, each on a fresh KV cache.
  Warm-up runs are excluded from the statistics.
* Other options: `--max-new-tokens 256`, `--context-length 4096`, `--system-prompt`, `--think`,
  `--compute npu|cpu|gpu|hybrid`, `--power-mode sustained_high_performance`, `--temperature`.
  `--dry-run` prints every adb and `geniex-bench` command without running them.

Output on screen and in `runs/gemma4-infer-<time>/results.md|json`: the **generated answer** and the
plan's latency breakdown. The medians come from the SDK's own profile data (`geniex_ProfileData`):

```
**Image:** street.jpg
**Prompt:** How many people are in the picture?
**Answer:**
There are three people in the picture ...

| Stage               | Median ms (n=3) | Tokens          | tok/s |
|---------------------|-----------------|-----------------|-------|
| Vision encoder      | ...             | -               | -     |
| Prefill             | ...             | ... tokens (N)  | ...   |
| Decode              | ...             | ... tokens (M)  | ...   |
| Time to first token | ...             | -               | -     |
| End to end          | ...             | -               | -     |
```

The stages map to the SDK fields as follows: vision encoder = `media_time`, prefill = `prompt_time`
over the text + image tokens (N), decode = `decode_time` for M generated tokens, and
end to end = TTFT + decode. The raw `geniex-bench` stdout, stderr, JSON and exact command line are
kept under `raw/`.

### 3. AI Hub-consistent performance metrics

```bash
hexprof gemma4 profile --compute npu cpu gpu
```

This reproduces AI Hub Models' GenieX scorecard exactly: context lengths 512 and 4096,
`-c ctx+128 -p ctx-128 -n 128 -r 3` (1 warm-up and 3 measured runs, medians), the same matrix file and
environment. It writes `perf.yaml` in AI Hub Models' `llm_metrics` format:

* `tokens_per_second`: median decode rate
* `prefill_tokens_per_second`: median prefill rate
* `time_to_first_token_range_milliseconds`: measured TTFT rescaled from the prompt length to
  [one 128-token chunk, the full context]. This convention reproduces every published S25 TTFT range
  (each range is exactly ctx/128 wide and consistent with the published prefill rate); the tests
  check it against the published values.

`metrics.md` puts each number next to the published S25 value. Use `--plugin qairt --model <bundle>`
for a compiled QNN bundle.

## SigLIP 2 on the S25 (AI Hub hosted device, NPU)

### 1. Fetch Qualcomm's pre-built W8A16 model

```bash
hexprof siglip2 fetch
```

This downloads `assets/siglip2-qnn_dlc-w8a16/{image_encoder.dlc,text_encoder.dlc,metadata.json}`, the
asset AI Hub verified with QAIRT 2.50. The `infer` and `profile` commands fetch it automatically if it
is missing.

### 2. Inference with your images

```bash
hexprof siglip2 infer --image photos/ --labels "a photo of a street" "a photo of a kitchen" "a photo of a dog"
```

* Images are preprocessed exactly like AI Hub Models' SigLIP2App: RGB, bilinear resize to 224×224,
  [0,1], NHWC, quantized to the DLC's uint16 input encoding. They run in one AI Hub inference job on
  "Samsung Galaxy S25" with profiling on.
* Outputs: `image_embeddings.npy` (N×768, L2-normalised), a preview of each embedding, the
  image-image cosine similarity matrix, and the AI Hub metrics of the job.
* With `--labels`, the text encoder also runs on the S25 and you get SigLIP's zero-shot scores
  `sigmoid(scale·cos + bias)` and the best label per image. The tokenizer, scale and bias come from
  `google/siglip2-base-patch16-224`.
* The jobs pin `--qairt_version 2.50`, the version the asset was verified with. Override with
  `--qairt-version default`.

### 3. AI Hub profile (same measurement as the published number)

```bash
hexprof siglip2 profile --components image_encoder text_encoder
```

This runs an AI Hub profile job with AI Hub's default random inputs, the same measurement as AI Hub's
scorecard. It reports the metrics with AI Hub Models' definitions:
`inference_time_milliseconds` (= `estimated_inference_time` / 1000),
`estimated_peak_memory_range_mb`, `layer_counts` per compute unit, `primary_compute_unit`, first and
warm load time, and the spread over all runs. It also gives the top op types by time, the published
value next to each metric, and the optrace / QHAS artifacts with a per-op classification.

## Hardware ceilings (AI Hub)

```bash
hexprof ceilings compute --precisions w8a16 w4a16 fp16 --size 2048 --chain 1 2 4 8 16
```

```bash
hexprof ceilings bandwidth --precision w4a16 --widths 3072 4096 5120 6144 --width-layers 16 --gemma4-layers 4 8 12
```

```bash
hexprof ceilings kv-check --precision w4a16 --layers 4 --context 4096
```

* **Compute**: chained square MatMuls (multiples of 32), quantized with AI Hub quantize jobs
  (weights INT4/INT8, activations INT8/INT16), compiled to QNN context binaries and profiled. The fit
  `latency = t0 + ops/P` (ops = 2·M·N·K) gives P (TOPS) and t0 per precision.
  `fp16` = the Gemma 4 vision encoder's precision.
* **Bandwidth**: the width sweep uses single-token stacks of `--width-layers` linear layers. The depth
  sweep uses the first 4/8/12 decoder layers of a Gemma-4-E4B-shaped model: GQA 8/2 heads, head_dim
  256/512, sliding window 512 with every 6th layer global, q/k/v norms, RoPE, gated GELU MLP,
  per-layer-embedding gate, and an explicit static KV cache. Its dimensions are read from the HF
  config; set `--intermediate-size` when offline. The x-axis is weight bytes per token at the deployed
  precision, including per-channel scales and excluding embeddings. Every point is checked to exceed
  on-chip memory (VTCM + system cache, `--onchip-mb`).
* **KV check**: the same compiled decoder subset, profiled with an empty and with a full cache. If
  latency moves, re-run `bandwidth --include-kv`.
* Outputs: `*_ceiling.json|md|png`. Points are submitted in parallel and job ids are saved to
  `state.json`, so re-running with the same `--out` resumes and retries failed points.
  `--dry-run` only builds the ONNX graphs. Weights stream to disk, so a 2 GB graph needs about
  200 MB of RAM.

## Hardware metrics report (QHAS + optrace)

AI Hub attaches two files to every NPU profile job, and hexprof downloads them next to the profile:
the **QHAS** (QNN HTP Analysis Summary, `<job>_main_qhas_summary.json`) and the **optrace**
(`<job>_main_profile_trace.json`). They hold what Workbench shows in its QHAS and trace views, and they
are far too large to read by hand (hundreds of MB per Gemma 4 part, GBs for a vision encoder).
`hexprof hw report` turns them into self-contained HTML pages with charts, plus CSV tables:

```bash
hexprof hw report $HEXPROF_DOWNLOAD_DIR/gemma4-s25-export
```

```bash
hexprof hw report $HEXPROF_DOWNLOAD_DIR/runs/siglip2-profile-<jobid>
```

It scans the folders for QHAS / optrace files, groups them by graph (`hw/<part>__<graph>/` for a Gemma 4
export, `<component>/` for a SigLIP 2 run), attaches AI Hub's `<graph>.profile.json`, and writes
`hw_report/` (default: inside the first folder given):

* `index.html`: every graph side by side. Run time, HTP utilization, inferences per second, DRAM/VTCM
  bytes and average GB/s, and where the cycles go (HMX / HVX / DMA / sync / reference ops / format changes).
* `<graph>.html`, one page per graph:
  * **Overall summary per HTP core**: Qualcomm's "% of overall timeline HTP utilization", inferences per
    second, time, timeline cycles, DRAM and VTCM read/write.
  * **Resource utilization**: HMX and HVX per HTP thread.
  * **HTP op instances by unit**: share of all op cycles.
  * **QNN op types and nodes**: cycles, % active cycles, HTP ops, DRAM/VTCM traffic, % on the critical path.
  * **HTP op types**: reference ops (scalar code) and format changes flagged.
  * **Critical (dominant) path ops** and the **slowest HTP op instances**, with time and DRAM GB/s.
  * **QHAS timeline**: occupancy of each HTP thread.
  * **Optrace**: busy share and timeline per thread, counter tracks such as memory bandwidth over time,
    and time by op name.
  * **AI Hub per-layer profile**: slowest layers and compute unit.
* `<graph>/*.csv`: every table, for your own plots; `summary.json`: headline numbers of every graph.

Each chart has hover tooltips and a table view, and the pages work offline in light and dark mode.
Open `index.html` in a browser; copy the whole `hw_report/` folder to view it elsewhere.

The QHAS column names and their definitions (quoted on the pages) are Qualcomm's, as used by their
QAIRT Visualizer (QHAS schema 1.x). Files are streamed (`ijson`), so memory stays flat. Parsing runs at
about 50 s per GB of QHAS on one core, so run it on a compute node for multi-GB files. Derived values
are labelled as derived:
* **GB/s** = bytes ÷ time;
* **effective clock** = timeline cycles ÷ time;
* **unit shares**: each op instance is counted once, in the order HMX, HVX, reference op, format change,
  DMA wait, sync, DMA.

The "% utilization" Qualcomm reports is the share of the run the HTP was busy, not compute or bandwidth
efficiency; for those, see `hexprof gemma4 efficiency` and the ceilings.

### Per-op bound classification

```bash
hexprof hw classify --job <profile_job_id> --bandwidth-json runs/ceil-bandwidth-*/bandwidth_ceiling.json
```

`hw classify` labels each operator:
* **compute-bound**: HMX or HVX utilisation ≥ 0.6;
* **memory-bound**: DRAM traffic ≥ 0.7·B;
* **overhead**: both low;
* **unknown**.

`--field-map` remaps fields, and `hexprof hw fields <file>` lists the tables and fields in a file. AI Hub
does **not** report power, energy, temperature, clock frequencies or stall counters, and hexprof does not
invent them.

## Pipeline: fine-tuned Gemma 4 → S25

```bash
pip install -e '.[pipeline]' && qai-hub-models install gemma_4_e4b_it
```

```bash
hexprof pipeline gemma4 quantize --checkpoint /path/to/finetuned-hf --out ckpt/w4a16 --context-length 4096
```

```bash
hexprof pipeline gemma4 export --checkpoint ckpt/w4a16 --out runs/ft-export
```

```bash
hexprof gemma4 infer --plugin qairt --model runs/ft-export/export_assets/<bundle> --image photo.jpg --prompt "What is this?"
```

This starts from Qualcomm's own Gemma 4 export: static shapes (128-token prompt processor and 1-token
generator), explicit KV cache, 4 parts (layers 0–23 | 24–32 | 33–41 | lm_head) and an FP16 vision
encoder. Calibration is Qualcomm's AIMET recipe (SeqMSE on WikiText, needs a CUDA GPU). An AI Hub
quantize job takes one weight dtype and one activation dtype per graph, so it cannot express the mixed
w4/w8/a16/kv8 recipe. `export` then compiles, links and profiles every (part, graph) on AI Hub,
collects the metrics plus optrace/QHAS for each, and builds the GenieX QAIRT bundle.
`--context-length` is the shorter-KV-cache optimisation from the plan.

## Tests

```bash
pip install -e '.[dev]' && pytest
```

The tests run without a phone or an AI Hub token. They cover:

* parsers and metric conventions against AI Hub's published numbers;
* every ceiling graph executed in onnxruntime (a full-size 6-layer Gemma-4-shaped decoder was also
  built and run from disk during development);
* the fits;
* optrace/QHAS parsing and the HTML report, on synthetic files with Qualcomm's schema;
* end-to-end Gemma 4 flows against a simulated phone, plus the real adb wrapper through a POSIX shell;
* SigLIP 2 and sweep flows against a simulated AI Hub.

## Troubleshooting

* If `geniex-bench` cannot find a vision projector for the default GGUF, pass `--mmproj`, or use
  `--model unsloth/gemma-4-E4B-it-GGUF:Q4_0` (the id GenieX's own S25 benchmark list uses).
* If several phones are attached, select one with `--serial <adb serial>`.
* An interrupted sweep resumes when re-run with the same `--out`; AI Hub jobs keep running meanwhile.
