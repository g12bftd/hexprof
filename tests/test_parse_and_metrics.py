import pytest

from hexprof.aihub.metrics import aihub_metrics, layer_counts, op_type_breakdown, primary_compute_unit
from hexprof.geniex.parse import aihub_llm_metrics, parse_accuracy_stdout, parse_cell, stage_breakdown
from hexprof.reference import published_gemma4_llm_metric, published_siglip2

from fakes import fake_profile


def _cell(ctx: int, compute: str, ttft_ms: float, prompt_tokens: int, prefill: float, decode: float) -> dict:
    return {
        "schema_version": "6",
        "cell_id": f"Gemma-4-E4B-it-llama_cpp-{compute}-c{ctx}",
        "plugin": "llama_cpp",
        "device": compute,
        "geniex_version": "v0.8.0",
        "params": {"n_ctx": ctx + 128},
        "runs": [],
        "agg": {
            "ttft_ms": {"median": ttft_ms},
            "prefill_tps": {"median": prefill},
            "decode_tps": {"median": decode},
            "prompt_tokens": {"median": prompt_tokens},
            "gen_tokens": {"median": 128},
        },
    }


@pytest.mark.parametrize("ctx,compute", [(4096, "npu"), (512, "npu"), (4096, "cpu"), (512, "gpu")])
def test_ttft_range_reproduces_published_s25_numbers(ctx, compute):
    """AI Hub's -p ctx-128 run, rescaled to [128, ctx] tokens, gives exactly the published TTFT range."""
    ref = published_gemma4_llm_metric(ctx, compute)
    prompt_tokens = ctx - 128
    measured_ttft = ref["time_to_first_token_range_milliseconds"]["min"] * prompt_tokens / 128
    m = aihub_llm_metrics(parse_cell(_cell(ctx, compute, measured_ttft, prompt_tokens,
                                           ref["prefill_tokens_per_second"], ref["tokens_per_second"])))
    assert m["context_length"] == ctx
    assert m["tokens_per_second"] == ref["tokens_per_second"]
    assert m["prefill_tokens_per_second"] == ref["prefill_tokens_per_second"]
    assert m["time_to_first_token_range_milliseconds"]["min"] == pytest.approx(
        ref["time_to_first_token_range_milliseconds"]["min"], rel=1e-9)
    assert m["time_to_first_token_range_milliseconds"]["max"] == pytest.approx(
        ref["time_to_first_token_range_milliseconds"]["max"], rel=1e-9)


def test_published_ttft_is_consistent_with_prefill_rate():
    # Sanity check of the convention: TTFT(ctx-128 tokens) ~= (ctx-128) / prefill rate.
    ref = published_gemma4_llm_metric(4096, "cpu")
    ttft = ref["time_to_first_token_range_milliseconds"]["min"] * (4096 - 128) / 128
    assert ttft == pytest.approx((4096 - 128) / ref["prefill_tokens_per_second"] * 1000, rel=0.002)


def test_parse_cell_runs_and_breakdown():
    cell = parse_cell({
        "schema_version": "7", "cell_id": "x-vqa0", "plugin": "llama_cpp", "device": "npu",
        "runs": [
            {"run_idx": 0, "ttft_us": 2030000, "media_us": 1600000, "prompt_time_us": 430000, "decode_time_us": 2100000,
             "prompt_tokens": 300, "gen_tokens": 30, "prefill_tps": 697.7, "decode_tps": 14.3, "stop_reason": "eos"},
            {"run_idx": 1, "ttft_us": 2050000, "media_us": 1610000, "prompt_time_us": 440000, "decode_time_us": 2000000,
             "prompt_tokens": 300, "gen_tokens": 30, "prefill_tps": 681.8, "decode_tps": 15.0, "stop_reason": "eos"},
        ],
    })
    assert cell.runs[0].e2e_ms == pytest.approx(4130.0)
    b = stage_breakdown(cell.runs)
    assert b["vision_encoder_ms"]["median"] == pytest.approx(1605.0)
    assert b["prefill_ms"]["median"] == pytest.approx(435.0)
    assert b["decode_tokens"]["median"] == 30
    assert b["e2e_ms"]["n"] == 2


def test_parse_accuracy_stdout_segments():
    out = "\n".join([
        "[sep ] prompt 1/2", "[info] stop_reason=eos", "[gen ] A red square.", "[gen ] ", "[gen ] Second line.",
        "[sep ] prompt 2/2", "[info] stop_reason=length", "[warn] self-abort: reason=wall-time after=9tok elapsed=1ms",
        "[gen ] Truncated", "[ok  ] cell  plugin=llama_cpp device=npu ngl=-1 ttft=1.0ms prefill=1.0tps decode=1.0tps gen=9 tok",
    ])
    segs = parse_accuracy_stdout(out)
    assert [s["index"] for s in segs] == [0, 1]
    assert segs[0]["text"] == "A red square.\n\nSecond line."
    assert segs[1]["stop_reason"] == "length" and segs[1]["warnings"]
    assert "summary_line" in segs[1]
    single = parse_accuracy_stdout("[info] stop_reason=eos\n[gen ] Hi\n")
    assert len(single) == 1 and single[0]["text"] == "Hi"


def test_aihub_metrics_match_scorecard_conventions():
    prof = fake_profile(3319.0, n_npu=404, peak=(0, 180 * 2**20 + 1000))
    m = aihub_metrics(prof)
    assert m["inference_time_milliseconds"] == pytest.approx(3.319)
    assert m["estimated_peak_memory_range_mb"] == {"min": 0, "max": 180}
    assert m["layer_counts"] == {"total": 404, "npu": 404, "gpu": 0, "cpu": 0}
    assert m["primary_compute_unit"] == "NPU"
    assert m["warm_load_time_milliseconds"] == pytest.approx(90.0)
    assert m["all_inference_times_milliseconds"]["min"] == pytest.approx(3.319)
    pub = published_siglip2("w8a16", "image_encoder", "qnn_dlc")
    assert pub["inference_time_milliseconds"] == 3.319


def test_primary_compute_unit_ties_and_breakdown():
    assert primary_compute_unit({"npu": 2, "gpu": 2, "cpu": 2}) == "NPU"
    assert primary_compute_unit({"npu": 0, "gpu": 0, "cpu": 3}) == "CPU"
    prof = {"execution_summary": {}, "execution_detail": [
        {"name": "a", "type": "MatMul", "compute_unit": "NPU", "execution_time": 30},
        {"name": "b", "type": "MatMul", "compute_unit": "NPU", "execution_time": 10},
        {"name": "c", "type": "Softmax", "compute_unit": "CPU", "execution_time": 60},
    ]}
    assert layer_counts(prof)["cpu"] == 1
    br = op_type_breakdown(prof)
    assert br[0]["type"] == "Softmax" and br[0]["share"] == pytest.approx(0.6)
