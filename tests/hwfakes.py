"""Synthetic QHAS / optrace / AI Hub profile files with the real schemas (for tests and demos)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

QNN_OPS = [("MatMul", "ConvLayer.hmx", "hmx"), ("Conv2d", "ConvLayer.hmx", "hmx"), ("RmsNorm", "RmsNorm.hvx", "hvx"),
           ("Softmax", "Softmax.hvx", "hvx"), ("ElementWiseAdd", "Add.hvx", "hvx"), ("ElementWiseMultiply", "Mul.hvx", "hvx"),
           ("Gather", "Gather.ref", "ref_op"), ("Transpose", "Transpose.fmt", "fmt_change"), ("Reshape", "Reshape.dma", "dma"),
           ("Concat", "Concat.dma", "dma_set"), ("Gelu", "Gelu.hvx", "hvx"), ("Quantize", "Quantize.hvx", "hvx")]
FLAGS = ["hmx", "hvx", "dma", "dma_set", "dma_wait", "sync", "ref_op", "fmt_change"]


def qhas_doc(n_instances: int = 600, seed: int = 0, strings: bool = False, threads: int = 4,
             mhz: float = 1100.0, major: int = 1) -> dict[str, Any]:
    rnd = random.Random(seed)
    S = (lambda v: str(v)) if strings else (lambda v: v)
    insts, cursor = [], [0.0] * threads
    agg: dict[str, dict[str, float]] = {}
    hagg: dict[str, dict[str, float]] = {}
    for i in range(n_instances):
        qop, hop, kind = QNN_OPS[rnd.randrange(len(QNN_OPS)) if i % 3 else rnd.randrange(2)]
        tid = i % threads
        cyc = float(int(rnd.expovariate(1 / (9000 if kind == "hmx" else 2500)) + 50))
        gap = float(int(rnd.expovariate(1 / 400)))
        start = cursor[tid] + gap
        cursor[tid] = start + cyc
        dr, dw = float(int(rnd.random() * cyc * 40)), float(int(rnd.random() * cyc * 10))
        row = {f: False for f in FLAGS}
        row[kind] = True
        if kind == "hmx" and rnd.random() < 0.3:
            row["dma_wait"] = True
        pdp = round(100 * cyc / 2e6 * rnd.random() * 3, 4)
        row.update({"htp": "0", "op_id": f"0x{i:05x}", "htp_op": hop, "qnn_op": f"{qop.lower()}_{i % 40}", "tid": str(tid),
                    "start_cycle": S(start), "cycles": S(cyc), "dram_read": S(dr), "dram_write": S(dw),
                    "vtcm_read": S(dr * 3), "vtcm_write": S(dw * 3), "num_dominant_path_cycles": S(int(cyc * pdp / 100 * 50)),
                    "percent_dominant_path_cycles": S(pdp), "dims": "[1,128,2560]", "dtype": "ufixed16",
                    "input_nodes": "x", "rank": 3, "step_size": 1, "zero_offset": 0})
        insts.append(row)
        a = agg.setdefault(qop, {"cycles": 0.0, "dram_read": 0.0, "dram_write": 0.0, "n": 0})
        a["cycles"] += cyc
        a["dram_read"] += dr
        a["dram_write"] += dw
        a["n"] += 1
        h = hagg.setdefault(hop, {"cycles": 0.0, "dram_read": 0.0, "dram_write": 0.0, "n": 0, "kind": kind})  # type: ignore[dict-item]
        h["cycles"] += cyc
        h["dram_read"] += dr
        h["dram_write"] += dw
        h["n"] += 1
    total_cycles = max(cursor)
    time_us = total_cycles / mhz
    busy = sum(float(r["cycles"]) for r in insts) / threads
    dram_r = sum(a["dram_read"] for a in agg.values())
    dram_w = sum(a["dram_write"] for a in agg.values())

    def typerow(op: str, a: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        p = round(100 * a["cycles"] / sum(x["cycles"] for x in agg.values()), 3)
        return {"op": op, "cycles": S(a["cycles"]), "percent_active_cycles": S(round(60 + 40 * rnd.random(), 2)),
                "dram_read": S(a["dram_read"]), "dram_write": S(a["dram_write"]), "vtcm_read": S(a["dram_read"] * 3),
                "vtcm_write": S(a["dram_write"] * 3), "num_htp_ops": S(a["n"]),
                "num_dominant_path_cycles_htp_0": S(int(a["cycles"] * p / 100)), "percent_dominant_path_cycles_htp_0": S(p), **extra}

    resources = [{"tid": str(t), "type": typ, "utilization": S(round(u, 2)), "timeline_cycles": S(total_cycles),
                  "start_cycle": S(0), "end_cycle": S(total_cycles), "cycles_used": S(int(total_cycles * u / 100))}
                 for t in range(threads) for typ, u in (("HMX", 20 + 50 * rnd.random()), ("HVX", 10 + 40 * rnd.random()))]
    return {
        "header": {"artifact_type": "QHAS", "version": {"major": major, "minor": 2, "patch": 0}},
        "data": {
            "htp_overall_summary": {"data": [{
                "htp": "0", "percent_utilization": S(round(100 * busy / total_cycles, 2)),
                "percent_idle": S(round(100 - 100 * busy / total_cycles, 2)), "inf_per_s": S(round(1e6 / time_us, 3)),
                "time_us": S(round(time_us, 3)), "timeline_cycles": S(total_cycles), "total_dram": S(dram_r + dram_w),
                "total_dram_read": S(dram_r), "total_dram_write": S(dram_w), "total_vtcm": S(3 * (dram_r + dram_w)),
                "total_vtcm_read": S(3 * dram_r), "total_vtcm_write": S(3 * dram_w),
                "htp_resources": {"data": resources}}]},
            "qnn_op_types": {"data": [typerow(k, v, {}) for k, v in agg.items()]},
            "qnn_op_instances_nodes": {"data": [typerow(f"{k.lower()}_{j}", v, {"qnn_op": f"{k.lower()}_{j}"})
                                                for k, v in agg.items() for j in range(3)]},
            "htp_op_types": {"data": [typerow(k, v, {"instances": S(v["n"]), "reference_op": v["kind"] == "ref_op",
                                                     "format_change": v["kind"] == "fmt_change"}) for k, v in hagg.items()]},
            "htp_op_instances": {"data": insts},
        },
    }


def optrace_doc(n: int = 400, seed: int = 0, bare_list: bool = False) -> Any:
    rnd = random.Random(seed)
    ev: list[dict[str, Any]] = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "HTP"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 0, "args": {"name": "Thread 0"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1, "args": {"name": "Thread 1"}},
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Memory"}},
        {"ph": "M", "name": "process_sort_index", "pid": 2, "args": {"sort_index": 1}},
    ]
    t = [0.0, 0.0]
    for i in range(n):
        tid = i % 2
        name = QNN_OPS[rnd.randrange(len(QNN_OPS))][1]
        dur = round(rnd.expovariate(1 / 8.0) + 0.5, 3)
        ts = t[tid] + round(rnd.expovariate(1 / 1.5), 3)
        t[tid] = ts + dur
        ev.append({"ph": "X", "name": name, "pid": 1, "tid": tid, "ts": ts, "dur": dur, "args": {"stackDepth": 0}})
        if i % 4 == 0:
            ev.append({"ph": "X", "name": "dma_in", "pid": 1, "tid": tid, "ts": ts, "dur": dur / 3, "args": {"stackDepth": 1}})
    for k in range(0, int(max(t)), 5):
        ev.append({"ph": "C", "name": "Memory Bandwidth", "pid": 2, "ts": float(k), "args": {"GB/s": round(20 + 30 * rnd.random(), 2)}})
    ev.append({"ph": "i", "name": "marker", "pid": 1, "tid": 0, "ts": 1.0})
    return ev if bare_list else {"traceEvents": ev, "displayTimeUnit": "ns"}


def profile_doc(seed: int = 0, n: int = 30) -> dict[str, Any]:
    rnd = random.Random(seed)
    detail = [{"name": f"/model/layers.{i // 3}/{op}", "type": op, "compute_unit": "NPU" if i % 10 else "CPU",
               "execution_time": round(rnd.expovariate(1 / 300), 1), "execution_cycles": int(rnd.random() * 1e6)}
              for i, op in enumerate(rnd.choice(["MatMul", "RmsNorm", "Softmax", "Add"]) for _ in range(n))]
    return {"execution_summary": {"estimated_inference_time": 21500, "inference_memory_peak_range": [1, 2]},
            "execution_detail": detail}


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def gemma_export_layout(root: Path, parts: int = 2, instances: int = 600, strings: bool = False) -> Path:
    """Mirror `hexprof pipeline gemma4 export`: hw/<label>/{qhas,optrace}/ + profiles/<label>.profile.json."""
    for i in range(1, parts + 1):
        for g, graph in enumerate(("prompt_ar128_cl4096", "token_ar1_cl4096")):
            label = f"part_{i}_of_{parts}__{graph}_{i}_of_{parts}"
            job = f"j{i}{g}abcdef"
            write_json(root / "hw" / label / "qhas" / f"{job}_main_qhas_summary.json",
                       qhas_doc(instances, seed=10 * i + g, strings=strings))
            write_json(root / "hw" / label / "optrace" / f"{job}_main_profile_trace.json", optrace_doc(300, seed=i + g))
            write_json(root / "profiles" / f"{label}.profile.json", profile_doc(seed=i + g))
    return root
