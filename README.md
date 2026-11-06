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
* Another KV-cache length or prompt chunk with Qualcomm's real weights: `--context-length 2048` and/or
  `--prompt-length 256`. These are Qualcomm's own export flags (`--context-lengths`, `--sequence-lengths N 1`),
  and the export goes to its own folder (`gemma4-s25-export-cl2048`). An export folder refuses to be reused
  with other flags. A smaller *hidden size* cannot come from the real weights; see `hexprof dummy`.

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

* `index.html`: every graph side by side.
  * For a split LLM (Gemma 4): end-to-end decode and prefill rates summed from the parts, next to
    Qualcomm's published numbers.
  * AI Hub time, peak memory and published time; QHAS time, HTP utilization, inferences per second,
    DRAM/VTCM bytes and average GB/s.
  * **What sets the run time**: critical-path share by unit (HMX / HVX / DMA / sync / …) and by QNN op type.
  * **Where the DRAM / VTCM traffic goes**: bytes by QNN op type, per graph.
  * **Where the work goes**: op cycles summed over all HTP threads, by unit.
* `<graph>.html`, one page per graph:
  * **Overall summary per HTP core**: Qualcomm's "% of overall timeline HTP utilization", inferences per
    second, time, timeline cycles, DRAM and VTCM read/write.
  * **AI Hub profile summary**: inference time, peak memory, layers per compute unit, first and warm load
    time, run-to-run spread, and the published figure when there is one.
  * **Resource utilization**: HMX and HVX per HTP thread.
  * **By unit**: critical-path share (what sets the run time) and op-cycle share (where the work goes).
  * **QNN op types and nodes**: cycles, % active cycles, HTP ops, DRAM/VTCM traffic, % on the critical path.
  * **HTP op types**: reference ops (scalar code) and format changes flagged.
  * **Memory traffic, QNN ops and HTP ops**: DRAM read / write and VTCM read / write per op type, per QNN
    node and per HTP op instance (bytes, as in Workbench's runtime analysis), each op's DRAM and VTCM GB/s,
    the ops with the most traffic, and a check that the per-op bytes add up to QHAS's totals.
  * **Critical (dominant) path ops** and the **slowest HTP op instances**, with time, DRAM / VTCM bytes and GB/s.
  * **QHAS timeline**: occupancy of each HTP thread.
  * **Optrace**: busy share and timeline per thread, counter tracks such as memory bandwidth over time,
    and time by op name.
  * **AI Hub per-layer profile**: slowest layers and compute unit.
* **Inputs and outputs** of every graph, as AI Hub compiled them (`hw fetch-models` records the specs): image size
  and patches, tokens per call, context length, batch, image tokens produced, and the output (vocabulary scores
  for N positions, hidden states, embedding size). With `--requests <results.json>` from `gemma4 aihub` /
  `siglip2 infer`, the overview adds your actual requests: original and encoder image size, image tokens, text
  tokens, total prompt tokens and chunks, generated tokens, estimated time to first token and end to end.
* `<graph>/*.csv`: every table, for your own plots; `summary.json`: headline numbers of every graph.

Optrace timestamps are not always in µs. The report finds their unit by matching the trace's span
against the QHAS run of the same graph (in HTP cycles, µs or ns), converts to µs, and says which it
used. If nothing matches, times stay in the file's own units, with a warning.

**Efficiency and stalls** (overview and every graph page):

| Metric | How it's computed |
|---|---|
| Work (MACs) | ONNX shapes of the compiled graph (`hw fetch-models --onnx`), else analytic: Gemma 4 text parts from Qualcomm's config and w4a16 recipe, SigLIP 2 from its config |
| Achieved TOPS, % of P | 2 × MACs ÷ QHAS run time; P measured at the graph's precision by `hexprof ceilings compute` |
| MACs per cycle | MACs ÷ timeline cycles (clock-independent) |
| DRAM GB/s, % of B | QHAS DRAM bytes ÷ run time; B from `hexprof ceilings bandwidth`, plus % of the theoretical 84.8 GB/s |
| Arithmetic intensity, roofline | 2 × MACs ÷ DRAM bytes; attainable = min(P, intensity × B); achieved ÷ attainable; memory- or compute-bound |
| DRAM ÷ (weights + KV) | weights = AI Hub's compiled model size (`hw fetch-models`) or analytic; the rest is activations, I/O and spills |
| Stalls | idle %; waiting inside ops (Qualcomm's % active cycles); DMA wait and sync as shares of work and of the critical path; data movement and graph I/O on the critical path; waiting cycles per QNN op type |
| Parallelism | average busy threads, busy HVX threads, HMX ops in flight, Qualcomm's per-thread HMX / HVX utilization |
| Per-op roofline | ONNX MACs joined onto QHAS QNN nodes by name: per-op TOPS, intensity and % of roofline |

A result above its roof is flagged: it means P, B or the MAC count is inconsistent. AI Hub does not
expose hardware stall counters, clocks or power; the report says so instead of estimating them.

```bash
hexprof hw fetch-models $HEXPROF_DOWNLOAD_DIR/gemma4-s25-export --onnx vision
```

```bash
hexprof hw report $HEXPROF_DOWNLOAD_DIR/gemma4-s25-export \
    --compute-json $HEXPROF_DOWNLOAD_DIR/runs/ceil-compute/compute_ceiling.json \
    --bandwidth-json $HEXPROF_DOWNLOAD_DIR/runs/ceil-bandwidth/bandwidth_ceiling.json
```

`hw fetch-models` (needs the AI Hub token) follows each profiled model back through its link and compile
jobs and caches the compiled size, parameter count, input shapes and, with `--onnx`, the source ONNX in
`<results>/models/`. `--onnx all` also fetches the Gemma 4 text parts (several GB each) for per-op MACs.

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

### Perfetto

AI Hub's optrace files are Chrome-trace JSON, which [Perfetto](https://ui.perfetto.dev) opens directly. Their
timestamps are HTP clock cycles (about 2000 per µs), though, and Perfetto reads them as µs, so every op would look
about 2000× longer. `hexprof hw perfetto` writes copies with the timestamps converted to µs, calibrated against the
same graph's QHAS run as in the report. It also renumbers process and thread ids, because Perfetto merges id 0
and "thread id = process id" lanes; names, nesting and counters are unchanged.

```bash
hexprof hw perfetto $HEXPROF_DOWNLOAD_DIR/gemma4-s25-export --gzip
```

This writes `<folder>/perfetto/<graph>.perfetto.json.gz`, one per profiled graph. Open it at ui.perfetto.dev: the UI
processes the trace locally in the browser. Its SQL tab, or the `perfetto` Python package's `TraceProcessor`,
gives per-op queries such as `select name, count(*), sum(dur) from slice group by name order by 3 desc`. Perfetto
shows the HTP op timeline the hardware report summarises. It cannot trace the hosted phone itself: AI Hub gives no
system-level access, and Android system traces would not show the NPU's ops anyway.

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

## Run log: what was deployed, with which settings, on which hardware

Every command that deploys something on AI Hub opens a run log (`hexprof/runlog.py`). That covers
`gemma4 aihub`, `pipeline gemma4 export`, `siglip2 infer|profile`, `study run`, `dummy run`,
`ceilings compute|bandwidth|kv` and `hub profile`. At the start it prints a banner, which lands at the top
of the Slurm `.out` file, and writes `run_manifest.json` and `run_manifest.md` into the results folder:

* **verdict**: one line, e.g. `Qualcomm AI Hub base model, Qualcomm's settings`,
  `Qualcomm AI Hub base model, 1 change(s): context lengths [4096] -> [2048]` or
  `SHRUNK DUMMY (random weights), 2 change(s): hidden 2560 -> 1280, intermediate 10240 -> 5120`;
* **model**: name, weights (Qualcomm's published checkpoint or asset, random dummy weights, your
  checkpoint), the code that built it, and the architecture;
* **settings**: every setting next to Qualcomm's (or the real model's) value, changes marked. For Gemma 4
  exports these are what Qualcomm's own export parser reads from the arguments, against what it reads
  with none;
* **hardware**: AI Hub device, SoC, chipset, Hexagon version (from AI Hub's device attributes when
  reachable), NPU facts (VTCM, HVX / HMX units, theoretical DRAM bandwidth), runtime, QAIRT version
  requested;
* **environment**: hexprof commit, command line, host, Slurm job, package versions.

While the run is open, every AI Hub job is logged, including the ones Qualcomm's export code submits. Each
job gets one stdout line (kind, id, name, device, URL) and a line in `jobs_log.jsonl`. At the end the run
records its status and duration. It also records the final state of every job (failed ones are listed with
AI Hub's message), the device, OS and QAIRT version AI Hub reports for the profile jobs, and the headline
results (per-graph latency, decode / prompt sums, verification). The hardware report shows the
manifest as its first section, "What was deployed".

## Slurm jobs (CSCS Alps / Daint)

`slurm/` holds the three batch jobs for the hosted-S25 workflow:

| Script | Runs | Results |
|---|---|---|
| `slurm/gemma4_profile.sh` | `hexprof gemma4 aihub --export-only [args]`: compile Gemma 4 and profile every part; `--context-length N` / `--prompt-length N` change Qualcomm's 4096 / 128 | `$HEXPROF_DOWNLOAD_DIR/gemma4-s25-export/` (`…-cl<N>` when changed) |
| `slurm/siglip2_profile.sh` | `hexprof siglip2 profile`: image and text encoders | `$HEXPROF_DOWNLOAD_DIR/runs/siglip2-profile-<jobid>/` |
| `slurm/ceilings.sh` | `hexprof ceilings compute` (w8a16, w4a16, fp16) and `ceilings bandwidth` (w4a16); resumable | `$HEXPROF_DOWNLOAD_DIR/runs/ceil-compute/`, `runs/ceil-bandwidth/` |
| `slurm/study.sh` | `hexprof study run …` with the arguments you pass (one sweep, verified and compared) | `$HEXPROF_DOWNLOAD_DIR/studies/<target>__<axis>/comparison/` |
| `slurm/dummy.sh` | `hexprof dummy run …`: one shrunk random-weight model, deployed, profiled, verified | `$HEXPROF_DOWNLOAD_DIR/dummy/<target>__<knobs>/` |
| `slurm/hw_report.sh [dir …]` | `hexprof hw fetch-models` + `hexprof hw report` on every `gemma4-s25-export*` and SigLIP 2 result (or on the folders you pass), with the ceilings when present | `…/hw_report/index.html` in each |

```bash
mkdir -p /capstor/scratch/cscs/$USER/job_logs/hexprof
```

```bash
sbatch slurm/gemma4_profile.sh
```

```bash
sbatch slurm/siglip2_profile.sh
```

```bash
sbatch slurm/ceilings.sh
```

```bash
sbatch slurm/hw_report.sh
```

They assume a conda env `hexprof` under `~/miniforge3` and the repo at `~/repos/hexprof` (override
with `HEXPROF_REPO`). `HEXPROF_DOWNLOAD_DIR` defaults to `/capstor/scratch/cscs/$USER/VLMs`. Logs go
to `/capstor/scratch/cscs/$USER/job_logs/hexprof/`, which must exist before you submit. Override the
project with `sbatch --account=<project> …`. To build the report automatically once both profiling
jobs succeed, submit it with `--dependency=afterok:<gemma_jobid>:<siglip_jobid>`.

## Systematic studies (sweeps verified on the device)

`hexprof study` sweeps one deployment setting over a list of values. For every value it:
1. builds the variant with Qualcomm's own code;
2. compiles it and profiles it on AI Hub's S25;
3. **verifies** it: hexprof reads back from AI Hub what actually ran (graph names, input shapes of the
   compiled graphs, quantize-job dtypes, the model the compiler consumed, the graph selected for profiling,
   QAIRT version, device) and checks it against the request;
4. writes a hardware report per value and a comparison page.

Unverified values are flagged with the failed checks and kept out of the charts.

```bash
hexprof study list
```

| Target | Axis | Default values | How it is built |
|---|---|---|---|
| `gemma4-text` | `context-length` | 512 1024 2048 4096 | `qai-hub-models export gemma_4_e4b_it --context-lengths V` (text parts only) |
| `gemma4-text` | `prompt-length` | 64 128 256 512 | `--sequence-lengths V 1` |
| `gemma4-text` | `decode-tokens` | 1 2 4 8 | `--sequence-lengths V`: cost of checking V drafted tokens at once (speculative decoding) |
| `gemma4-vision` | `soft-tokens` | 70 140 280 560 1120 | the image processor's token budget (`max_soft_tokens=V`) while Qualcomm's `VisionEncoder.from_pretrained` traces |
| `gemma4-vision` | `precision` | float w8a16 | AI Hub quantize job on Qualcomm's float graph (Qualcomm ships float only) |
| `siglip2-image` / `siglip2-text` | `batch` | 1 2 4 8 (16) | Qualcomm's `get_input_spec(image_batch_size / text_batch_size=V)` |
| `siglip2-image` / `siglip2-text` | `precision` | float w8a16 w8a8 w4a16 | Qualcomm's quantize stage (`run_quantize`) |
| all single-graph targets | `qairt` | default latest | `--qairt_version V` on compile and profile (`hexprof hub frameworks` lists versions) |
| all single-graph targets | `mixed-precision` | 0 5 10 25 50 | Workbench Lite-MP: V% most sensitive layers at `--override` (fp16 / int16), the rest at the `--precision` base (w8a16 / w8a8); 0 = base only |

Qualcomm's shipped setting is always among the defaults, so every study has its reference point.

Why `soft-tokens` and not the image size for the Gemma 4 vision encoder: Gemma 4's image processor resizes
every image (keeping its aspect ratio) to fill `soft_tokens × pooling_kernel²` patches and pads the rest, so
any square input gives the same graph (448 px and 224 px both become 768 × 768 at 280 tokens). The token
budget is what changes the work. An earlier `image-size` axis would have produced identical graphs; its
"differs from every other value" check would have flagged that.

```bash
sbatch slurm/study.sh gemma4-text --axis context-length --values 512 1024 2048 4096
```

```bash
hexprof study compare $HEXPROF_DOWNLOAD_DIR/studies/gemma4-text__context-length
```

Results go to `$HEXPROF_DOWNLOAD_DIR/studies/<target>__<axis>/`: one folder per value (jobs, profiles, QHAS
/ optrace, `verification.json`, `hw_report/`) and `comparison/index.html` with every metric against the value,
plus CSV / JSON. Settings that are not swept are flags too (`--context-length`, `--prompt-length`,
`--precision`, `--qairt`, `--batch`, `--soft-tokens`). They are recorded in `study.json`, and a study folder
refuses to mix runs with different fixed settings. Re-running resumes; `hexprof study verify <dir>` re-checks
against AI Hub.

Ablation summary across all studies (each verified change against Qualcomm's setting, with the
hardware metrics behind it):

```bash
hexprof study summary
```

Not available as studies:
* **dropping the audio encoder.** Qualcomm's QAIRT deployment of Gemma 4 has no audio component (image +
  text only), so there is nothing to remove on the device.
* **2-bit weights or quantization-aware training on Workbench.** Workbench quantize jobs are post-training
  only, with INT4 / INT8 / INT16. Qualcomm's Gemma 4 text recipe is AIMET-based and needs an x86 GPU machine.
  Per-layer mixed precision on Workbench is Lite-MP, which is the `mixed-precision` axis.

What the studies do not establish:
* accuracy. Quantized variants are calibrated with Qualcomm's calibration data, or the model's sample inputs
  when that is unavailable; quality must be evaluated separately;
* whether GenieX on a phone accepts a changed graph. AI Hub runs each graph on the S25 on its own.

## Shrunk models with random weights (`hexprof dummy`)

Deploy a smaller Gemma 4 (text or vision) or SigLIP 2 (image or text encoder) on the S25 NPU with full
hardware profiling, choosing its sizes: layers, hidden size, MLP width, heads, KV heads, head dims, sliding
window, KV-shared layers, vocabulary, patch size, image-token budget, sequence lengths, context, batch,
precision. Weights are random. Shapes, sizes and operator structure are what matter, not accuracy. No
custom NPU kernels are involved. The models are built by **Qualcomm's own model code** from a Hugging Face
config with the requested sizes, so they get the same NPU adaptations as the published deployments:

| Target | Built with | What the device runs |
|---|---|---|
| `gemma4-text` | Qualcomm's `Gemma4_E4B_PreSplit` / `Gemma4_E4B_PartBase`, subclassed with the requested sizes; their CPU-only ONNX export and split (no AIMET, no GPU) | the same part / graph layout as Qualcomm's export: `part1_of_N` … lm_head part, each a context binary with a `prompt_ar{P}_cl{C}` and a `token_ar1_cl{C}` graph; single-head Conv attention, external embeddings + per-layer inputs, per-head KV-cache I/O, separate sliding-window / global masks and RoPE, KV sharing |
| `gemma4-vision` | Qualcomm's `VisionEncoder.from_pretrained(checkpoint=…)` | Qualcomm's adapted vision tower (attention / RMSNorm / Linear→Conv rewrites, static pooler) |
| `siglip2-image`, `siglip2-text` | Qualcomm's `SigLIP2ImageEncoder` / `SigLIP2TextEncoder.from_pretrained(weights=…)` | Qualcomm's encoder graphs |

```bash
hexprof dummy knobs
```

```bash
hexprof dummy plan gemma4-text --dims layers=12 hidden=1280 intermediate=5120 heads=4 kv-heads=1
```

```bash
sbatch slurm/dummy.sh gemma4-text --dims layers=12 hidden=1280 intermediate=5120 heads=4 kv-heads=1
```

`plan` needs neither torch nor AI Hub. It prints the real and the requested value of every knob, the
parts and graphs, the KV-cache tensors and the estimated weights, and it rejects layouts Qualcomm's code
cannot build. Examples: heads that are not a multiple of the KV heads, or KV-shared layers that have no
earlier layer of their attention type to read from. It warns when a size is not a multiple of 32 (the HMX
pads it). Knobs you do not give keep the real model's values, read from its `config.json` (Gemma 4: the copy
Qualcomm's export cached under `~/.qaihm`, else Hugging Face; or `--base-config`). When only `layers`
changes, the KV-shared fraction (18 of 42) and the split points scale with it.

`run` writes a random checkpoint, runs Qualcomm's export (Gemma text) or Qualcomm's wrapper (vision,
SigLIP 2), and then on AI Hub: upload → quantize job (random calibration data) → compile per graph
(`qnn_dlc`, Qualcomm's own compile options) → link → profile per graph with QHAS / optrace. It then checks
that what ran is what was asked:
* every input of every compiled graph against shapes computed from the knobs alone: hidden, per-layer
  inputs (= total layers × per-layer width), masks (context, sliding window), RoPE widths (head dims), one
  KV tensor per non-shared layer and KV head, and logits (= vocabulary);
* the layers inside each exported part;
* that every op ran on the NPU. AI Hub's per-layer compute units of each profile job must all be NPU; one
  CPU or GPU op fails the check. A QNN context binary is built for the HTP as a whole, so an unsupported op
  makes the compile fail rather than fall back, but the check reads it back anyway. This check also applies
  to the studies of the real models;
* the job chain (uploaded part → quantized → compiled → linked → profiled), quantize dtypes, device, QAIRT;
* for vision / SigLIP 2: the config Qualcomm's wrapper actually built, and the image-processor budget;
* Gemma 4 vision only, when your real export exists (`$HEXPROF_DOWNLOAD_DIR/gemma4-s25-export`, or
  `--reference`): the dummy is compiled, linked and profiled with **the very options of your real export's
  vision-encoder jobs** (read from AI Hub). The QAIRT version those jobs ran with is pinned explicitly,
  because Qualcomm's code omits the flag when its pinned version is AI Hub's default at that moment.
  Verification then requires the same options and the same QAIRT version for each of the three jobs, and the
  same compiled input shapes when the patch geometry is unchanged. Without a real export, Qualcomm's own
  option builders are used.
  Images are prepared by Gemma 4's own image processor (Google's preprocessor config, no tokenizer needed),
  or by transformers' class defaults if the Hub is unreachable (same shapes and patch positions).

Finally the hardware report is built (per-op MACs from the local ONNX parts). Sweep one knob with
verification and a comparison page:

```bash
sbatch slurm/study.sh dummy-gemma4-text --axis layers --values 6 12 24 42 --dims hidden=1280 intermediate=5120
```

Be aware of what a dummy is and is not:
* **Fidelity check first.** `hexprof dummy run gemma4-text` with no `--dims` builds the full-size model
  through the same path. Its per-graph latency and QHAS numbers against Qualcomm's real export
  (`slurm/gemma4_profile.sh`) show how far the dummy path itself is off before you trust shrunk variants.
* **Quantization differs.** Qualcomm's Gemma text recipe (AIMET: int4, with int8 `down_proj` and lm_head and
  per-channel blocks) needs an x86 GPU. Dummies use an AI Hub quantize job: one weight dtype for the whole
  graph. w4a16 moves fewer weight bytes than Qualcomm's mix, and w8a16 more, so the two bracket it. Random
  calibration data changes the scales, not the work.
* **Random weights do not change the timing.** The HTP has no data-dependent sparsity or early exit in these
  graphs.
* Sizes that are not multiples of 32 compile but run padded. That is part of what you measure, but the
  real models are aligned.
* AI Hub runs each graph on its own. A dummy is not a GenieX bundle; no tokenizer or embedding tables are
  shipped.
* Gemma 4 vision width: Qualcomm's vision attention rewrite needs `hidden = heads × head-dim`. So
  `--dims hidden=384` also sets heads 12 → 6 (head-dim 64 kept, like ViT-Base → ViT-Small). The run manifest
  lists both changes. The vision encoder's output width is the text model's hidden size (`text-hidden`,
  2560), so a vision-only change leaves every text graph as it is.
* Qualcomm's vision pooler hard-codes width 768 for its padding tensor. For other widths hexprof rebuilds
  that tensor at the encoder's width (the real model is unaffected). Gemma 4 dummies drop the audio tower,
  and SigLIP 2 dummies shrink the unused tower; neither reaches the NPU.

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
* SigLIP 2 and sweep flows against a simulated AI Hub;
* shrunk models (`hexprof dummy`): knob resolution and validation, the full-size request reproducing
  Qualcomm's layout, and the deploy → verify chain against a simulated AI Hub. That chain includes a
  compiler that changes a shape (caught) and the fallback to per-graph quantization. During development the
  real path ran on CPU: a random 8-layer Gemma 4 through Qualcomm's own export and split, and every part in
  ONNX Runtime at both graph shapes. SigLIP 2 and Gemma 4 vision dummies were also loaded and traced through
  Qualcomm's wrappers.

## Troubleshooting

* If `geniex-bench` cannot find a vision projector for the default GGUF, pass `--mmproj`, or use
  `--model unsloth/gemma-4-E4B-it-GGUF:Q4_0` (the id GenieX's own S25 benchmark list uses).
* If several phones are attached, select one with `--serial <adb serial>`.
* An interrupted sweep resumes when re-run with the same `--out`; AI Hub jobs keep running meanwhile.
