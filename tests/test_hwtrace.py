import json

import pytest

from hexprof.aihub import hwtrace


def test_qhas_like_table_is_found_and_classified(tmp_path):
    qhas = {
        "schema_version": "1.2.0",
        "data": {
            "graphs": [{
                "name": "g",
                "ops": [
                    {"op_name": "conv_1", "op_type": "Conv2d", "hmx_utilization_pct": 85, "hvx_utilization_pct": 10,
                     "ddr_read_bytes": 1e6, "ddr_write_bytes": 1e5, "time_us": 100},
                    {"op_name": "fc_2", "op_type": "FullyConnected", "hmx_utilization_pct": 12, "hvx_utilization_pct": 5,
                     "ddr_read_bytes": 5.0e6, "ddr_write_bytes": 0, "time_us": 100},  # 50 GB/s
                    {"op_name": "transpose_3", "op_type": "Transpose", "hmx_utilization_pct": 0, "hvx_utilization_pct": 8,
                     "ddr_read_bytes": 1e4, "ddr_write_bytes": 1e4, "time_us": 50},
                ],
            }]
        },
    }
    f = tmp_path / "qhas.json"
    f.write_text(json.dumps(qhas))
    desc = hwtrace.describe(f)
    assert desc["record_sets"][0]["path"] == "$.data.graphs[0].ops"
    canon = desc["record_sets"][0]["canonical"]
    assert canon["hmx_util"] == "hmx_utilization_pct" and canon["dram_read_bytes"] == "ddr_read_bytes"
    rs = hwtrace.best_record_set(f)
    rows = hwtrace.classify(hwtrace.normalize(rs), bandwidth_gbps=60.0)
    by = {r["name"]: r for r in rows}
    assert by["conv_1"]["bound"] == "compute" and by["conv_1"]["hmx_util"] == pytest.approx(0.85)
    assert by["fc_2"]["bound"] == "memory" and by["fc_2"]["achieved_gbps"] == pytest.approx(50.0)
    assert by["transpose_3"]["bound"] == "overhead"
    summary = hwtrace.class_summary(rows)
    assert summary["compute"]["time_share"] == pytest.approx(0.4)
    # without a bandwidth ceiling, low-utilisation ops cannot be called overhead
    rows = hwtrace.classify(hwtrace.normalize(rs), bandwidth_gbps=None)
    assert {r["bound"] for r in rows} == {"compute", "low-util (no DRAM data)"}


def test_chrome_trace_events_are_aggregated(tmp_path):
    trace = {"traceEvents": [
        {"name": "MatMul_1", "ph": "X", "ts": 0, "dur": 30, "args": {"hmx_util": 0.9}},
        {"name": "MatMul_1", "ph": "X", "ts": 40, "dur": 20, "args": {"hmx_util": 0.7}},
        {"name": "Softmax_2", "ph": "X", "ts": 70, "dur": 10, "args": {"hvx_util": 0.3}},
        {"name": "meta", "ph": "M", "args": {}},
    ]}
    f = tmp_path / "optrace.json"
    f.write_text(json.dumps(trace))
    rows = {r["name"]: r for r in hwtrace.normalize(hwtrace.best_record_set(f))}
    assert rows["MatMul_1"]["count"] == 2 and rows["MatMul_1"]["time_us"] == 50
    assert rows["MatMul_1"]["hmx_util"] == pytest.approx(0.8)
    assert "meta" not in rows


def test_merge_aihub_timing_and_field_override(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"ops": [{"node": "a", "busy_hmx": 0.9}, {"node": "b", "busy_hmx": 0.1}]}))
    rs = hwtrace.best_record_set(f)
    rows = hwtrace.normalize(rs, {"name": "node", "hmx_util": "busy_hmx"})
    hwtrace.merge_aihub_timing(rows, [{"name": "a", "execution_time_us": 12, "type": "Conv", "compute_unit": "NPU"}])
    by = {r["name"]: r for r in rows}
    assert by["a"]["time_us"] == 12 and by["a"]["op_type"] == "Conv"
    assert by["a"]["hmx_util"] == pytest.approx(0.9)
