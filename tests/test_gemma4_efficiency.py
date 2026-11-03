import json

import pytest

from hexprof.cli import main
from hexprof.models import gemma4_efficiency as ge

CFG = {
    "model_type": "gemma4",
    "text_config": {
        "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "global_head_dim": 8,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        "sliding_window": 4, "num_kv_shared_layers": 2, "use_double_wide_mlp": True,
        "hidden_size_per_layer_input": 2, "vocab_size": 32,
    },
}


def test_layer_counts_by_hand():
    c = ge.TextConfig.load(_write_cfg())
    # layer 0: sliding (hd 4, span min(4, ctx)), own KV, normal MLP
    l0 = ge.layer_counts(c, 0, seq=1, ctx=16)
    q, o, k, v = 8 * 8, 8 * 8, 8 * 4, 8 * 4
    mlp = 8 * 16 * 3
    ple = 8 * 2 * 2
    assert l0["params"] == q + o + k + v + mlp + ple
    assert l0["kv_bytes"] == 2 * 1 * 4 * 4  # K+V, 1 kv head, span 4, hd 4, int8
    assert l0["ops"] == 2 * l0["params"] * 1 + 4 * 2 * 1 * 4 * 4
    # bytes: int4 everywhere except down_proj (int8), plus a 4-byte scale per output channel
    w4 = (q + o + k + v + 8 * 16 * 2 + ple) / 2
    w8 = 16 * 8
    scales = (8 + 8 + 4 + 4 + 16 + 16 + 8 + 2 + 8) * 4
    assert l0["weight_bytes"] == pytest.approx(w4 + w8 + scales)
    # layer 3: global (hd 8, span = ctx), KV-shared (no k/v), double-wide MLP
    l3 = ge.layer_counts(c, 3, seq=1, ctx=16)
    assert l3["params"] == 8 * 16 + 16 * 8 + 8 * 32 * 3 + ple
    assert l3["kv_bytes"] == 2 * 1 * 16 * 8


def _write_cfg(tmp=None):
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text(json.dumps(CFG))
    return d / "config.json"


def test_efficiency_report(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CFG))
    parts = {}
    for i in (1, 2, 3, 4):
        for kind, ar, ms in (("prompt", 4, 2.0), ("token", 1, 1.0)):
            g = f"{kind}_ar{ar}_cl16_{i}_of_4"
            parts[f"part{i}_of_4/{g}"] = {"component": f"part{i}_of_4", "graph": g,
                                          "metrics": {"inference_time_milliseconds": ms}}
    parts["vision_encoder/vision_encoder"] = {"component": "vision_encoder", "graph": "vision_encoder",
                                              "metrics": {"inference_time_milliseconds": 5.0}}
    exp = tmp_path / "export"
    exp.mkdir()
    (exp / "export_profiles.json").write_text(json.dumps({"parts": parts}))
    q = exp / "hw" / "part4_of_4__token_ar1_cl16_4_of_4" / "qhas"
    q.mkdir(parents=True)
    ops = [{"op_name": f"op{i}", "htp_utilization": 10 + i} for i in range(20)]  # per-op table: ignored
    (q / "x_qhas_summary.json").write_text(json.dumps(
        {"data": {"overall": {"timeline_htp_utilization_pct": 87.5, "inferences_per_second": 41.0}, "ops": ops}}))

    r = ge.efficiency(exp, config=cfg, peak_tops=1.0, peak_gbps=10.0, splits=(2, 3))
    rows = {g["graph"]: g for g in r["graphs"]}
    lm = rows["part4_of_4/token_ar1_cl16_4_of_4"]
    assert lm["layers"] == "lm_head" and lm["weight_params"] == 8 * 32
    assert lm["achieved_gbps"] == pytest.approx(lm["bytes"] / 1e-3 / 1e9)
    assert lm["qhas"] == {"$.data.overall.timeline_htp_utilization_pct": 87.5,
                          "$.data.overall.inferences_per_second": 41.0}
    p1 = rows["part1_of_4/token_ar1_cl16_1_of_4"]
    assert p1["layers"] == "0-1" and p1["inferences_per_s"] == pytest.approx(1000.0)
    assert p1["roofline_bound"] in ("memory", "compute") and 0 < p1["roofline_efficiency"]
    assert rows["vision_encoder/vision_encoder"]["kind"] == "vision encoder"
    assert r["model"]["decode_ms_per_token"] == pytest.approx(4.0)
    assert r["model"]["prefill_tokens_per_s"] == pytest.approx(4 * 1e3 / 8.0)
    assert "Whole model" in (exp / "efficiency.md").read_text()
    # Qualcomm's real split points (24, 33) are used when they fit the config
    big = ge.TextConfig.load(cfg)
    big.layers = 42
    assert ge.splits_for(big, 4) == (24, 33)
    assert ge.splits_for(ge.TextConfig.load(cfg), 4) == (1, 3)  # 4-layer toy config: even split
    # CLI path
    assert main(["gemma4", "efficiency", "--export-dir", str(exp), "--config", str(cfg)]) == 0


def test_large_qhas_is_scanned_not_loaded(tmp_path):
    f = tmp_path / "big.json"
    per_op = ",".join(f'{{"op":"o{i}","htp_utilization":{i}}}' for i in range(50))
    f.write_text('{"overall":{"timeline_htp_utilization_pct": 91.2, "inferences_per_second": 3.5},"ops":[' + per_op + "]}")
    fields = ge.qhas_summary_fields(f, max_mb=0)  # force the scanning path
    assert fields == {"~timeline_htp_utilization_pct": 91.2, "~inferences_per_second": 3.5}
