import csv
import json
import re

import pytest

from hexprof.hwreport import jsonstream, optrace, qhas, report
import hwfakes


def _page_data(html_text):
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html_text, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_qhas_load_aggregates_match_the_source(tmp_path):
    doc = hwfakes.qhas_doc(500, seed=3)
    p = hwfakes.write_json(tmp_path / "x" / "qhas" / "j_main_qhas_summary.json", doc)
    rep = qhas.load(p, top_n=20, timeline_bins=50)
    insts = doc["data"]["htp_op_instances"]["data"]
    assert rep.version == "1.2.0" and not rep.warnings
    assert rep.instances["count"] == 500
    assert rep.instances["total_cycles"] == pytest.approx(sum(r["cycles"] for r in insts))
    by_kind = rep.instances["by_kind"]
    assert sum(v["count"] for v in by_kind.values()) == 500
    hmx = sum(r["cycles"] for r in insts if r["hmx"])
    assert by_kind["HMX (matrix)"]["cycles"] == pytest.approx(hmx)  # HMX wins even when DMA wait is also set
    assert "HMX, DMA Wait" in rep.instances["by_type"]
    top = rep.instances["top_by_cycles"]
    assert len(top) == 20 and top[0]["cycles"] == max(r["cycles"] for r in insts)
    assert top[0]["time_us"] == pytest.approx(top[0]["cycles"] * rep.cycles_to_us())
    assert rep.qnn_op_types[0]["cycles"] >= rep.qnn_op_types[-1]["cycles"]
    assert rep.qnn_op_types[0]["dominant_path"][0]["core"] == 0
    assert len(rep.resources) == 8 and rep.resources[0]["type"] in ("HMX", "HVX")
    tl = rep.instances["timeline"]
    assert tl["unit"] == "cycles" and len(tl["lanes"]) == 4 and all(len(ln["busy"]) == 50 for ln in tl["lanes"])
    assert all(0 <= b <= 1 for ln in tl["lanes"] for b in ln["busy"])
    k = qhas.overall_kpis(rep)
    assert k["clock_mhz"] == pytest.approx(1100, rel=1e-3)
    assert k["dram_gbps"] == pytest.approx(k["total_dram"] / k["time_us"] / 1e3)


def test_qhas_numbers_as_strings_and_version_warning(tmp_path):
    a = hwfakes.write_json(tmp_path / "a.json", hwfakes.qhas_doc(200, seed=1))
    b = hwfakes.write_json(tmp_path / "b.json", hwfakes.qhas_doc(200, seed=1, strings=True, major=3))
    ra, rb = qhas.load(a), qhas.load(b)
    assert rb.instances["total_cycles"] == pytest.approx(ra.instances["total_cycles"])
    assert qhas.overall_kpis(rb)["percent_utilization"] == pytest.approx(qhas.overall_kpis(ra)["percent_utilization"])
    assert rb.qnn_op_types[0]["op"] == ra.qnn_op_types[0]["op"]
    assert any("outside the 1.x range" in w for w in rb.warnings)


def test_occupancy_is_exact_for_back_to_back_ops():
    import numpy as np

    s, d = np.array([0.0, 10.0, 30.0]), np.array([10.0, 10.0, 10.0])
    tl = qhas._lane_occupancy(s, d, np.zeros(3, dtype=np.int32), {"L": 0}, 4)
    assert tl["lanes"][0]["busy"] == [1.0, 1.0, 0.0, 1.0]
    assert tl["lanes"][0]["busy_share"] == pytest.approx(0.75)


@pytest.mark.parametrize("bare", [False, True])
def test_optrace_load(tmp_path, bare):
    doc = hwfakes.optrace_doc(200, seed=2, bare_list=bare)
    p = hwfakes.write_json(tmp_path / "t.json", doc)
    ot = optrace.load(p, bins=40, max_points=10)
    evs = doc if bare else doc["traceEvents"]
    assert ot.events == len(evs)
    assert [ln["name"] for ln in ot.lanes] == ["HTP · Thread 0", "HTP · Thread 1"]
    top_level = [e for e in evs if e["ph"] == "X" and not e["args"]["stackDepth"]]
    assert sum(ln["busy_us"] for ln in ot.lanes) == pytest.approx(sum(e["dur"] for e in top_level))
    assert all(0 < ln["busy_share"] <= 1 for ln in ot.lanes)
    c = ot.counters[0]
    assert c["name"] == "Memory Bandwidth" and c["process"] == "Memory"
    assert len(c["series"]) <= 10 and all(len(pt) == 3 for pt in c["series"])  # downsampled: mean + max
    assert c["min"] <= c["time_weighted_mean"] <= c["max"]
    assert ot.other_phases == {"i": 1}
    assert {o["name"] for o in ot.ops} >= {"dma_in"}


def test_without_ijson_falls_back_to_json(tmp_path, monkeypatch):
    p = hwfakes.write_json(tmp_path / "q.json", hwfakes.qhas_doc(100, seed=4))
    with_ijson = qhas.load(p).instances["total_cycles"]
    monkeypatch.setattr(jsonstream, "ijson", None)
    assert qhas.load(p).instances["total_cycles"] == pytest.approx(with_ijson)


def test_discover_gemma_and_siglip_layouts(tmp_path):
    hwfakes.gemma_export_layout(tmp_path / "gemma", parts=2, instances=50)
    sig = tmp_path / "siglip" / "image_encoder"
    hwfakes.write_json(sig / "qhas" / "abc_main_qhas_summary.json", hwfakes.qhas_doc(50))
    hwfakes.write_json(sig / "image_encoder.profile.json", hwfakes.profile_doc())
    gs = report.discover([tmp_path / "gemma", tmp_path / "siglip"])
    labels = [g.label for g in gs]
    assert "image_encoder" in labels and len(labels) == 5
    g = next(g for g in gs if g.label.startswith("part_1_of_2__prompt"))
    assert g.qhas and g.optrace and g.profile and g.profile.name == f"{g.label}.profile.json"
    ie = next(g for g in gs if g.label == "image_encoder")
    assert ie.qhas and ie.profile and ie.optrace is None


def test_build_report_end_to_end(tmp_path):
    root = hwfakes.gemma_export_layout(tmp_path / "export", parts=2, instances=300)
    index = report.build([root], tmp_path / "out", top=6, bins=60, title="Gemma 4 on S25")
    out = index.parent
    pages = sorted(p.name for p in out.glob("*.html"))
    assert len(pages) == 5 and "index.html" in pages
    ov = _page_data(index.read_text())
    assert ov["title"] == "Gemma 4 on S25"
    table = next(s for s in ov["sections"] if s["title"] == "Graphs")["blocks"][0]["table"]
    assert len(table["rows"]) == 4 and all(r["util"] is not None and r["href"].endswith(".html") for r in table["rows"])
    stacked = next(s for s in ov["sections"] if s["title"].startswith("Where the work goes"))["blocks"][0]
    assert len(stacked["series"]) <= 8 and stacked["series"][-1] == "Other"
    assert all(sum(r["values"]) == pytest.approx(100) for r in stacked["rows"])

    label = table["rows"][0]["folder"]
    assert table["rows"][0]["label"] == "part 1/2 · prompt (ar128, cl4096)"
    page = _page_data((out / f"{label}.html").read_text())
    titles = [s["title"] for s in page["sections"]]
    for want in ("Overall summary per HTP core", "Resource utilization (HMX, HVX, …)", "QNN op types", "HTP op types",
                 "What sets the run time: critical path by unit", "AI Hub profile summary",
                 "Critical path: ops that set the run time", "QHAS timeline: HTP thread occupancy",
                 "Optrace: busy share per thread", "Optrace: counters (memory tracks)", "AI Hub per-layer profile"):
        assert want in titles, want
    qnn = next(b for s in page["sections"] if s["title"] == "QNN op types" for b in s["blocks"] if b["type"] == "hbar")
    assert len(qnn["rows"]) == 7 and qnn["rows"][-1]["label"].startswith("Other")  # top 6 + Other
    assert sum(r["value"] for r in qnn["rows"]) == pytest.approx(100)
    assert {t["label"] for t in page["tiles"]} >= {"HTP utilization", "Inferences per second", "DRAM traffic"}

    for name in ("qnn_op_types.csv", "htp_op_types.csv", "htp_op_instances_slowest.csv", "qhas_resources.csv",
                 "optrace_threads.csv", "optrace_counters.csv", "aihub_layers.csv"):
        rows = list(csv.DictReader(open(out / label / name)))
        assert rows, name
    summ = json.loads((out / "summary.json").read_text())
    assert summ[label]["qhas"]["op_instances"] == 300


def test_cli_hw_report(tmp_path, capsys):
    from hexprof import cli

    root = hwfakes.gemma_export_layout(tmp_path / "e", parts=1, instances=40)
    assert cli.main(["hw", "report", str(root), "--top", "5", "--bins", "20"]) == 0
    assert (root / "hw_report" / "index.html").exists()
    assert "Report:" in capsys.readouterr().out


def test_build_without_files_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.build([tmp_path], tmp_path / "o")


def test_optrace_unit_is_identified_from_qhas(tmp_path):
    q = hwfakes.qhas_doc(100, seed=5)
    k = q["data"]["htp_overall_summary"]["data"][0]
    cycles, time_us = float(k["timeline_cycles"]), float(k["time_us"])
    cases = {"cycles": (cycles, "HTP clock cycles"), "us": (time_us, "µs"), "ns": (time_us * 1e3, "ns")}
    for name, (span, word) in cases.items():
        p = hwfakes.write_json(tmp_path / f"{name}.json", hwfakes.optrace_doc(150, seed=6, span=span))
        ot = optrace.load(p, bins=20)
        optrace.calibrate(ot, cycles, time_us)
        assert ot.unit == "µs" and word in ot.unit_note, name
        assert ot.span_us == pytest.approx(time_us, rel=1e-6), name
        assert ot.timeline["t1"] - ot.timeline["t0"] == pytest.approx(time_us, rel=1e-6)
        assert sum(ln["busy_share"] for ln in ot.lanes) > 0
    assert ot.declared_unit == "ns"
    p = hwfakes.write_json(tmp_path / "odd.json", hwfakes.optrace_doc(150, seed=6, span=time_us * 2.1e3))
    ot = optrace.load(p, bins=20)
    optrace.calibrate(ot, cycles, time_us)
    assert ot.unit == "trace units" and ot.span_us is None and "not identified" in ot.unit_note


def test_critical_path_cycles_by_unit(tmp_path):
    doc = hwfakes.qhas_doc(300, seed=8)
    p = hwfakes.write_json(tmp_path / "q.json", doc)
    bk = qhas.load(p).instances["by_kind"]
    insts = doc["data"]["htp_op_instances"]["data"]
    assert sum(v["dp_cycles"] for v in bk.values()) == pytest.approx(sum(r["num_dominant_path_cycles"] for r in insts))
    assert bk["HMX (matrix)"]["dp_cycles"] == pytest.approx(sum(r["num_dominant_path_cycles"] for r in insts if r["hmx"]))


def test_report_end_to_end_rollup_units_and_published(tmp_path):
    root = hwfakes.gemma_export_layout(tmp_path / "e", parts=2, instances=120)
    sig = tmp_path / "e" / "siglip" / "image_encoder"
    hwfakes.write_json(sig / "qhas" / "x_main_qhas_summary.json", hwfakes.qhas_doc(80))
    hwfakes.write_json(sig / "image_encoder.profile.json", hwfakes.profile_doc(ms=3.3))
    hwfakes.write_json(sig.parent / "profile.json", {"precision": "w8a16", "runtime": "qnn_dlc", "components": {
        "image_encoder": {"published": {"inference_time_milliseconds": 3.319, "job_id": "jpx0kk1jp",
                                        "estimated_peak_memory_range_mb": {"min": 0, "max": 180},
                                        "layer_counts": {"npu": 404, "total": 404}}}}})
    index = report.build([root], tmp_path / "out", top=5, bins=30)
    ov = _page_data(index.read_text())
    tiles = {t["label"]: t for t in ov["tiles"]}
    assert tiles["Decode (AI Hub)"]["value"] == "50.0 tok/s"  # 2 token parts × 10 ms
    assert tiles["Prefill (AI Hub)"]["value"] == "3200 tok/s"  # 128 tokens in 2 × 20 ms
    e2e = next(s for s in ov["sections"] if s["title"].startswith("End to end"))
    assert "13.36 tok/s" in e2e["blocks"][0]["text"] and "q4_0" in e2e["blocks"][0]["text"]
    rows = {r["folder"]: r for r in next(s for s in ov["sections"] if s["title"] == "Graphs")["blocks"][0]["table"]["rows"]}
    assert rows["image_encoder"]["published_ms"] == 3.319 and rows["image_encoder"]["peak_mb"] == 180
    g = next(r for f, r in rows.items() if f.startswith("part_1_of_2__prompt"))
    assert g["trace_ms"] == pytest.approx(g["time_ms"], rel=1e-6)  # cycles -> µs via QHAS
    titles = [s["title"] for s in ov["sections"]]
    assert "What sets the run time: critical path by unit, per graph" in titles
    assert "What sets the run time: critical path by QNN op type, per graph" in titles
    page = _page_data((index.parent / "image_encoder.html").read_text())
    summ = next(s for s in page["sections"] if s["title"] == "AI Hub profile summary")
    trows = {r["metric"]: r for r in summ["blocks"][0]["table"]["rows"]}
    assert trows["Inference time"]["published"] == "3.319 ms" and trows["Warm load time"]["value"] == "120.000 ms"
    assert any(t["label"] == "AI Hub inference peak memory" for t in page["tiles"])
