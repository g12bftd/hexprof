import json

import numpy as np
import pytest

import hwfakes
from hexprof.hwreport import analytic, macs, models, report
from hexprof.hwreport import efficiency as EF


def _onnx(path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 128, 64])
    z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 64, 1, 128])
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [16, 64])
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in ("S", "C", "G")]
    inits = [numpy_helper.from_array(np.zeros((64, 256), np.int8), "W1q"),
             numpy_helper.from_array(np.array(0.1, np.float32), "W1s"),
             numpy_helper.from_array(np.array(0, np.int8), "W1z"),
             numpy_helper.from_array(np.zeros((32, 64, 1, 1), np.float32), "Wc"),
             numpy_helper.from_array(np.zeros((32, 64), np.float32), "Wg")]
    nodes = [helper.make_node("DequantizeLinear", ["W1q", "W1s", "W1z"], ["W1"], name="/model/dq"),
             helper.make_node("MatMul", ["X", "W1"], ["Y1"], name="/model/mm1/MatMul"),
             helper.make_node("Transpose", ["Y1"], ["T"], perm=[0, 2, 1], name="/model/t"),
             helper.make_node("MatMul", ["Y1", "T"], ["S"], name="/model/mm2/MatMul"),
             helper.make_node("Conv", ["Z", "Wc"], ["C"], name="/model/conv/Conv"),
             helper.make_node("Gemm", ["A", "Wg"], ["G"], transB=1, name="/model/fc/Gemm")]
    m = helper.make_model(helper.make_graph(nodes, "g", [x, z, a], outs, inits), opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(m, str(path))
    return path


def test_onnx_mac_counts_and_qhas_matching(tmp_path):
    p = _onnx(tmp_path / "m.onnx")
    c = macs.count(p, {"X": [1, 128, 64]})
    by = {n["name"]: n for n in c["nodes"]}
    assert by["/model/mm1/MatMul"]["macs"] == 128 * 256 * 64 and by["/model/mm1/MatMul"]["weight"]  # weight behind DQ
    assert by["/model/mm2/MatMul"]["macs"] == 128 * 128 * 256 and not by["/model/mm2/MatMul"]["weight"]
    assert by["/model/conv/Conv"]["macs"] == 32 * 128 * 64
    assert by["/model/fc/Gemm"]["macs"] == 16 * 32 * 64
    assert c["unresolved"] == 0 and c["total_macs"] == sum(n["macs"] for n in c["nodes"])
    unpinned = macs.count(p)  # batch unknown -> the two MatMuls cannot be sized
    assert unpinned["unresolved"] == 2 and unpinned["warnings"]
    qnn = [{"qnn_op": "_model_mm1_MatMul", "cycles": 100, "dram_read": 10, "dram_write": 0},
           {"qnn_op": "_model_mm2_MatMul_reshape_out", "cycles": 50, "dram_read": 1, "dram_write": 1},  # converter suffix
           {"qnn_op": "_model_mm2_MatMul", "cycles": 30, "dram_read": 0, "dram_write": 0},
           {"qnn_op": "_model_softmax", "cycles": 999}]
    j = macs.match(qnn, c)
    rows = {r["name"]: r for r in j["rows"]}
    assert rows["/model/mm2/MatMul"]["cycles"] == 80 and rows["/model/mm2/MatMul"]["qnn_nodes"] == 2
    assert j["coverage"] == pytest.approx((128 * 256 * 64 + 128 * 128 * 256) / c["total_macs"])


def test_siglip2_analytic_counts():
    img = analytic.siglip2("image_encoder", "w8a16")
    assert img["macs"] == pytest.approx(17_708_292_096)  # 196 patches, 12 layers, hidden 768, MAP head
    assert img["weight_bytes"] == pytest.approx(92_752_896)
    txt = analytic.siglip2("text_encoder", "w8a16")
    assert txt["macs"] == pytest.approx(5_511_905_280) and txt["weight_bytes"] == pytest.approx(85_573_632)
    assert analytic.siglip2("text_encoder", "fp16")["weight_bytes"] == pytest.approx(2 * 85_573_632)
    assert analytic.siglip2("other") is None


def test_efficiency_metrics_with_ceilings(tmp_path):
    from hexprof.hwreport import qhas

    p = hwfakes.write_json(tmp_path / "q.json", hwfakes.qhas_doc(400, seed=2))
    rep = qhas.load(p)
    k = qhas.overall_kpis(rep)
    ctx = EF.Ctx(P={"w8a16": 20.0}, B=60.0, B_series="width")
    m = EF.graph_metrics("image_encoder", k, rep.instances, rep.qnn_op_types, rep.qnn_op_instances, "w8a16",
                         {"compiled_file_size": 50e6}, None, ctx, rep.resources)
    t = k["time_us"] / 1e6
    assert m["achieved_tops"] == pytest.approx(2 * 17_708_292_096 / t / 1e12)
    assert m["compute_util_pct"] == pytest.approx(100 * m["achieved_tops"] / 20.0)
    assert m["bw_util_pct"] == pytest.approx(100 * k["dram_gbps"] / 60.0)
    assert m["bw_util_theory_pct"] == pytest.approx(100 * k["dram_gbps"] / 84.8)
    assert m["intensity"] == pytest.approx(2 * 17_708_292_096 / k["total_dram"])
    roof = min(20.0, m["intensity"] * 60.0 / 1e3)
    assert m["attainable_tops"] == pytest.approx(roof) and m["roof_pct"] == pytest.approx(100 * m["achieved_tops"] / roof)
    assert m["weight_bytes"] == 50e6 and m["dram_over_expected"] == pytest.approx(k["total_dram"] / 50e6)
    st = m["stalls"]
    rows = rep.qnn_op_types
    active = sum(r["cycles"] * r["percent_active_cycles"] / 100 for r in rows) / sum(r["cycles"] for r in rows)
    assert st["waiting_pct"] == pytest.approx(100 - 100 * active)
    assert st["avg_busy_threads"] == pytest.approx(rep.instances["total_cycles"] / k["timeline_cycles"])
    assert 0 < st["data_movement_critical_pct"] < 100 and "Transpose" in st["data_movement_ops"]
    assert st["hmx_resource_util_sum"] == pytest.approx(sum(r["utilization"] for r in rep.resources if r["type"] == "HMX"))
    # no ceilings: nothing invented
    m0 = EF.graph_metrics("image_encoder", k, rep.instances, rep.qnn_op_types, [], "w8a16", None, None, EF.Ctx())
    assert "compute_util_pct" not in m0 and "bw_util_pct" not in m0 and "bound" not in m0
    assert m0["bw_util_theory_pct"] == pytest.approx(100 * k["dram_gbps"] / 84.8)
    assert EF.graph_metrics("mystery", k, rep.instances, [], [], None, None, None, ctx).get("macs") is None


class _M:
    def __init__(self, mid, name, size=None, params=None, producer=None, kind="QNN_CONTEXT_BINARY"):
        self.model_id, self.name, self.file_size, self.num_parameters = mid, name, size, params
        self._producer, self.model_type = producer, kind

    def get_producer(self):
        return self._producer

    def download(self, filename):
        from pathlib import Path

        _onnx(Path(filename + ".onnx"))
        return filename + ".onnx"


class CompileJob:
    def __init__(self, src, shapes):
        self.model, self.shapes, self.job_id = src, shapes, "jc"

    def get_target_model(self):
        return None


class _Link:
    def __init__(self, models):
        self.models = models


def test_fetch_models_walks_link_and_compile_jobs(tmp_path, monkeypatch):
    from hexprof.aihub import client as hc

    src_p = _M("ms_prompt", "part1 prompt onnx", 9e9, 1_000, None, "ONNX")
    src_t = _M("ms_token", "part1 token onnx", 9e9, 1_000, None, "ONNX")
    comp_p = _M("mc_p", "x", producer=CompileJob(src_p, {"input_ids": ((1, 128), "int32")}))
    comp_t = _M("mc_t", "y", producer=CompileJob(src_t, {"input_ids": ((1, 1), "int32")}))
    linked = _M("ml", "part1_of_4", size=1.9e9, producer=_Link([comp_t, comp_p]))
    vis_src = _M("mvs", "vision onnx", 1e9, 500, None, "ONNX")
    vis = _M("mv", "vision", size=0.8e9, producer=CompileJob(vis_src, {"pixel_values": ((1, 3, 448, 448), "float32")}))
    by_id = {"ml": linked, "mv": vis}
    monkeypatch.setattr(hc, "get_model", lambda mid: by_id[mid])
    exp = tmp_path / "export"
    parts = {"part1_of_4/prompt_ar128_cl4096_1_of_4": {"graph": "prompt_ar128_cl4096_1_of_4", "model_id": "ml",
                                                       "job": {"job_id": "jp"}},
             "vision_encoder/gemma_4_e4b_it": {"graph": "gemma_4_e4b_it", "model_id": "mv", "job": {"job_id": "jv"}}}
    hwfakes.write_json(exp / "export_profiles.json", {"parts": parts})
    got = models.fetch([exp], onnx="vision")
    p = got["part1_of_4__prompt_ar128_cl4096_1_of_4"]
    assert p["compiled_file_size"] == 1.9e9 and p["source_model_id"] == "ms_prompt"  # picked by sequence length
    assert p["shapes"] == {"input_ids": [1, 128]} and "onnx_path" not in p
    v = got["vision_encoder__gemma_4_e4b_it"]
    assert v["source_num_parameters"] == 500 and v["onnx_path"].endswith(".onnx")
    assert json.loads((exp / "models" / "vision_encoder__gemma_4_e4b_it.model.json").read_text())["onnx_path"]
    infos = models.load_infos([exp])
    assert set(infos) == set(got)


def test_report_with_ceilings_models_and_onnx(tmp_path):
    root = hwfakes.gemma_export_layout(tmp_path / "e", parts=2, instances=150)
    label = "part_1_of_2__prompt_ar128_cl4096_1_of_2"
    qfile = next((root / "hw" / label / "qhas").glob("*.json"))
    q = json.loads(qfile.read_text())
    q["data"]["qnn_op_instances_nodes"]["data"] += [  # QNN keeps the ONNX node names (minus punctuation)
        {"op": "_model_mm1_MatMul", "qnn_op": "_model_mm1_MatMul", "cycles": 40000, "dram_read": 2e5, "dram_write": 0},
        {"op": "_model_conv_Conv", "qnn_op": "_model_conv_Conv", "cycles": 9000, "dram_read": 4e4, "dram_write": 1e3}]
    qfile.write_text(json.dumps(q))
    other = "part_2_of_2__prompt_ar128_cl4096_2_of_2"
    hwfakes.write_json(root / "models" / f"{other}.model.json", {"label": other, "onnx_path": str(_onnx(tmp_path / "n.onnx"))})
    hwfakes.write_json(root / "models" / f"{label}.model.json",
                       {"label": label, "compiled_file_size": 3e6, "onnx_path": str(_onnx(tmp_path / "m.onnx")),
                        "shapes": {"X": [1, 128, 64]}})
    cj = hwfakes.write_json(tmp_path / "compute_ceiling.json", {"series": {"w4a16": {"P_tops": 12.0}, "fp16": {"P_tops": 4.0}}})
    bj = hwfakes.write_json(tmp_path / "bandwidth_ceiling.json", {"series": {"width sweep (16 layers)": {"B_gbps": 65.0}}})
    index = report.build([root], tmp_path / "out", top=5, bins=20, compute_json=cj, bandwidth_json=bj)
    data = lambda p: json.loads(__import__("re").search(r'id="data">(.*?)</script>', p.read_text(), __import__("re").S)
                                .group(1).replace("<\\/", "</"))
    ov = data(index)
    eff = next(s for s in ov["sections"] if s["title"] == "Compute and memory efficiency, per graph")
    roof = next(b for b in eff["blocks"] if b["type"] == "roofline")
    assert roof["P"] == {"w4a16": 12.0} and roof["B"] == 65.0  # only roofs of precisions on the chart; and len(roof["points"]) == 2  # both graphs with an ONNX
    assert any(s["title"] == "Stalls and parallelism, per graph" for s in ov["sections"])
    page = data(index.parent / f"{label}.html")
    titles = [s["title"] for s in page["sections"]]
    i = titles.index("Overall summary per HTP core")
    assert titles[i + 1:i + 3] == ["Compute and memory efficiency", "Stalls and parallelism"]
    assert titles[0] == "Inputs and outputs"
    per_op = next(s for s in page["sections"] if s["title"].startswith("Per-op roofline"))
    assert per_op["blocks"][0]["type"] == "roofline" and len(per_op["blocks"][0]["points"]) == 2
    unmatched = data(index.parent / f"{other}.html")
    msg = next(s for s in unmatched["sections"] if s["title"].startswith("Per-op roofline"))
    assert "none of their names matched" in msg["why"] and "/model/conv/Conv" in msg["why"]
    assert any("without known shapes" in w for w in unmatched["warnings"])  # dynamic batch, no shapes pinned
    summ = json.loads((index.parent / "summary.json").read_text())
    e = summ[label]["efficiency"]
    assert e["macs_source"].startswith("ONNX") and e["weight_bytes"] == 3e6 and e["P_tops"] == 12.0
    assert summ["_ceilings"]["B_gbps"] == 65.0
    assert (index.parent / label / "efficiency.csv").exists()


def test_efficiency_uses_qhas_measured_bytes(tmp_path):
    from hexprof.models import gemma4_efficiency as ge

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"text_config": {
        "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "vocab_size": 32}}))
    exp = tmp_path / "x"
    g = "token_ar1_cl16_4_of_4"
    hwfakes.write_json(exp / "export_profiles.json", {"parts": {f"part4_of_4/{g}": {
        "component": "part4_of_4", "graph": g, "metrics": {"inference_time_milliseconds": 1.0}}}})
    q = hwfakes.qhas_doc(50)
    hwfakes.write_json(exp / "hw" / f"part4_of_4__{g}" / "qhas" / "j_main_qhas_summary.json", q)
    r = ge.efficiency(exp, config=cfg, splits=(2, 3))
    row = r["graphs"][0]
    assert row["bytes_source"] == "QHAS"
    assert row["bytes"] == pytest.approx(q["data"]["htp_overall_summary"]["data"][0]["total_dram"])
    assert row["bytes_estimate"] == pytest.approx(8 * 32 / 2 + 32 * 4)  # 4-bit lm_head + fp32 scale per channel


def test_memory_traffic_sections_and_totals_check(tmp_path):
    import re

    root = hwfakes.gemma_export_layout(tmp_path / "e", parts=1, instances=200)
    label = "part_1_of_1__token_ar1_cl4096_1_of_1"
    doc = json.loads(next((root / "hw" / label / "qhas").glob("*.json")).read_text())
    index = report.build([root], tmp_path / "out", top=4, bins=10)
    page = json.loads(re.search(r'id="data">(.*?)</script>', (index.parent / f"{label}.html").read_text(), re.S)
                      .group(1).replace("<\\/", "</"))
    secs = {s["title"]: s for s in page["sections"]}
    q = secs["Memory traffic: QNN ops (DRAM and VTCM)"]
    check = {r["what"]: r for r in q["blocks"][0]["table"]["rows"]}
    insts = doc["data"]["htp_op_instances"]["data"]
    ov = doc["data"]["htp_overall_summary"]["data"][0]
    for key, what in (("dram_read", "DRAM read"), ("vtcm_write", "VTCM write")):
        assert check[what]["overall"] == pytest.approx(ov[f"total_{key}"])
        assert check[what]["htp_instances"] == pytest.approx(sum(r[key] for r in insts))
        assert check[what]["qnn_types"] == pytest.approx(sum(r[key] for r in doc["data"]["qnn_op_types"]["data"]))
        assert check[what]["qnn_vs_overall"] == pytest.approx(100)
    charts = [b for b in q["blocks"] if b["type"] == "stacked"]
    assert [c["series"] for c in charts][:2] == [["DRAM read", "DRAM write"], ["VTCM read", "VTCM write"]]
    assert all(len(c["rows"]) <= 4 for c in charts)
    types_table = next(b for b in q["blocks"] if b.get("text") == "All QNN op types:")["table"]
    row = types_table["rows"][0]
    t = row["cycles"] * ov["time_us"] / ov["timeline_cycles"] * 1e-6
    assert row["vtcm_gbps"] == pytest.approx((row["vtcm_read"] + row["vtcm_write"]) / t / 1e9)
    h = secs["Memory traffic: HTP ops (DRAM and VTCM)"]
    inst_table = next(b for b in h["blocks"] if b.get("table"))["table"]
    keys = [c["key"] for c in inst_table["cols"]]
    assert {"vtcm_read", "vtcm_write", "vtcm_gbps", "dram_gbps"} <= set(keys)
    dram_top = [r["dram_bytes"] for r in inst_table["rows"]]
    assert dram_top == sorted(dram_top, reverse=True) and dram_top[0] == max(r["dram_read"] + r["dram_write"] for r in insts)
    htp_types = next(b for b in secs["HTP op types"]["blocks"] if b.get("table"))["table"]
    assert "vtcm_read" in [c["key"] for c in htp_types["cols"]]
    overview = json.loads(re.search(r'id="data">(.*?)</script>', index.read_text(), re.S).group(1).replace("<\\/", "</"))
    titles = [s["title"] for s in overview["sections"]]
    assert "Where the DRAM traffic goes: by QNN op type, per graph" in titles
    assert "Where the VTCM traffic goes: by QNN op type, per graph" in titles
    assert (index.parent / label / "memory_totals_check.csv").exists()


def test_io_spec_formats_and_meaning():
    from typing import NamedTuple

    from hexprof.hwreport import io as IO

    class TensorSpec(NamedTuple):  # as qai_hub.public_rest_api.TensorSpec
        name: str
        dtype: str
        shape: tuple | None = None

    linked = {"prompt_ar128_cl4096_4_of_4": [TensorSpec("input_embeds", "uint16", (1, 128, 2560))],
              "token_ar1_cl4096_4_of_4": [TensorSpec("input_embeds", "uint16", (1, 1, 2560))]}
    assert IO.shapes(linked, "token_ar1_cl4096_4_of_4") == {"input_embeds": [1, 1, 2560]}  # AI Hub IOSpec, per graph
    assert IO.shapes({None: [TensorSpec("image", "float32", (1, 3, 224, 224))]}) == {"image": [1, 3, 224, 224]}
    assert IO.shapes({"input_ids": ((4, 64), "int32")}) == {"input_ids": [4, 64]}  # job shapes / InputSpec
    assert IO.shapes({"x": (1, 8)}) == {"x": [1, 8]}
    assert IO.normalize({"x": {"shape": [2, 3], "dtype": "float32"}})["x"]["dtype"] == "float32"

    d = IO.describe("part4_of_4__prompt_ar128_cl4096_4_of_4", IO.normalize(linked, "prompt_ar128_cl4096_4_of_4"),
                    {"logits": {"shape": [1, 128, 262144], "dtype": "uint16"}})
    assert d["tokens_per_call"] == 128 and d["context_length"] == 4096 and d["logit_positions"] == 128
    assert d["vocab"] == 262144 and "vocabulary scores for 128" in d["output"]
    v = IO.describe("vision_encoder__gemma_4_e4b_it",
                    {"pixel_values": {"shape": [1, 784, 768]}, "image_position_ids": {"shape": [1, 784, 2]}},
                    {"vision_embedding": {"shape": [1, 256, 2560]}})
    assert v["image"] == "448×448" and v["patch"] == 16 and v["image_patches"] == 784 and v["image_tokens_out"] == 256
    i = IO.describe("image_encoder", {"image": {"shape": [1, 3, 224, 224]}}, {"image_embeds": {"shape": [1, 768]}})
    assert i["image"] == "224×224" and i["batch"] == 1 and i["embedding_dim"] == 768
    tx = IO.describe("text_encoder", {"input_ids": {"shape": [8, 64]}}, {"text_embeds": {"shape": [8, 768]}})
    assert tx["tokens_per_call"] == 64 and tx["batch"] == 8
    rows = IO.table({"input_embeds": {"shape": [1, 1, 2560], "dtype": "uint16"},
                     **{f"past_key_{k}_in": {"shape": [1, 2, 256, 4095], "dtype": "uint8"} for k in range(3)}}, {})
    assert len(rows) == 2 and "+2 more KV tensors" in rows[1]["name"]


def test_fetch_models_records_io_and_report_shows_it(tmp_path, monkeypatch):
    from typing import NamedTuple

    from hexprof.aihub import client as hc

    class TS(NamedTuple):
        name: str
        dtype: str
        shape: tuple | None = None

    m = _M("mv", "vision", size=0.8e9)
    m.input_spec = {"g": [TS("pixel_values", "float32", (1, 784, 768)), TS("image_position_ids", "int32", (1, 784, 2))]}
    m.output_spec = {"g": [TS("vision_embedding", "float32", (1, 256, 2560))]}
    monkeypatch.setattr(hc, "get_model", lambda mid: m)
    root = tmp_path / "export"
    hwfakes.write_json(root / "export_profiles.json", {"parts": {"vision_encoder/gemma_4_e4b_it": {
        "graph": "gemma_4_e4b_it", "model_id": "mv", "job": {"job_id": "jv"}}}})
    label = "vision_encoder__gemma_4_e4b_it"
    hwfakes.write_json(root / "hw" / label / "qhas" / "j_main_qhas_summary.json", hwfakes.qhas_doc(40))
    info = models.fetch([root])[label]
    assert info["inputs"]["pixel_values"]["shape"] == [1, 784, 768] and info["outputs"]["vision_embedding"]["shape"][1] == 256
    img = tmp_path / "photo.png"
    from PIL import Image

    Image.new("RGB", (1024, 768)).save(img)
    req = hwfakes.write_json(tmp_path / "results.json", {"results": [{
        "image": str(img), "prompt": "What is in the image?", "vision_encoder": {"image_tokens": 256},
        "breakdown": {"prefill_tokens": 290, "prefill_chunks": 3, "prefill_chunk_tokens": 128, "decode_tokens": 41,
                      "decode_tokens_source": "local answer", "ttft_ms": 1400.0, "e2e_ms": 4300.0}}]})
    index = report.build([root], tmp_path / "out", requests=[req])
    data = lambda p: json.loads(__import__("re").search(r'id="data">(.*?)</script>', p.read_text(), __import__("re").S)
                                .group(1).replace("<\\/", "</"))
    ov = data(index)
    io_row = next(s for s in ov["sections"] if s["title"] == "Inputs and outputs per graph")["blocks"][0]["table"]["rows"][0]
    assert io_row["image"] == "448×448" and io_row["image_tokens_out"] == 256
    rq = next(s for s in ov["sections"] if s["title"].startswith("Your requests"))["blocks"][0]["table"]["rows"][0]
    assert rq["original"] == "1024×768" and rq["fed"] == "448×448" and rq["image_tokens"] == 256
    assert rq["text_tokens"] == 34 and rq["generated_tokens"] == 41 and rq["chunks"] == "3 × 128"
    page = data(index.parent / f"{label}.html")
    assert page["sections"][0]["title"] == "Inputs and outputs"
    assert any("256 image tokens" in r["meaning"] for r in page["sections"][0]["blocks"][0]["table"]["rows"])
