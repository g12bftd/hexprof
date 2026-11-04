"""MACs per node from an ONNX graph's tensor shapes, matched to the QNN ops in QHAS.

The graph is loaded without its weight data (so multi-GB models load in seconds), shapes are
inferred, and multiply-accumulates are counted for the ops that do them:

* MatMul / MatMulInteger / QLinearMatMul: prod(output) x K
* Gemm: prod(output) x K
* Conv / QLinearConv: prod(output) x C_in/groups x kernel area (1x1 convs = linear layers)
* ConvTranspose: prod(input) x C_out/groups x kernel area

Elementwise ops do no MACs. QNN keeps ONNX node names (minus punctuation), so names are
normalised to letters and digits before matching; a QNN node whose name extends an ONNX node's
name (converter suffixes) is matched to the longest such ONNX name.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

PASS_THROUGH = {"DequantizeLinear", "QuantizeLinear", "Cast", "Identity", "Transpose", "Reshape", "Squeeze",
                "Unsqueeze", "Flatten"}
MAC_OPS = {"MatMul", "MatMulInteger", "QLinearMatMul", "Gemm", "Conv", "QLinearConv", "ConvTranspose"}
_MIN_PREFIX = 12


def norm(name: str | None) -> str:
    return re.sub(r"[^0-9a-z]", "", (name or "").lower())


def _prod(d: list[int]) -> float:
    return float(math.prod(d)) if d else 0.0


def _known(d: list[int | None] | None) -> bool:
    return bool(d) and all(isinstance(x, int) and x > 0 for x in d)  # type: ignore[union-attr]


def _attr(node: Any, name: str, default: Any) -> Any:
    for a in node.attribute:
        if a.name == name:
            from onnx import helper

            return helper.get_attribute_value(a)
    return default


def count(path: str | Path, input_shapes: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """MACs per node of an ONNX file (``input_shapes`` pins dynamic graph inputs, e.g. from the compile job)."""
    import onnx
    from onnx import shape_inference

    p = Path(path)
    model = onnx.load(str(p), load_external_data=False)
    warnings: list[str] = []
    for inp in model.graph.input:
        dims = (input_shapes or {}).get(inp.name)
        if dims:
            shp = inp.type.tensor_type.shape
            for d, v in zip(shp.dim, dims):
                d.ClearField("dim_param")
                d.dim_value = int(v)
    try:
        model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception as e:  # noqa: BLE001 - counts what is known
        warnings.append(f"shape inference failed ({type(e).__name__}: {e}); only declared shapes used")
    g = model.graph
    shapes: dict[str, list[int | None]] = {}
    for vi in [*g.input, *g.output, *g.value_info]:
        shapes[vi.name] = [d.dim_value if d.HasField("dim_value") else None for d in vi.type.tensor_type.shape.dim]
    inits = {i.name: list(i.dims) for i in g.initializer}
    shapes.update(inits)  # type: ignore[arg-type]
    consts: set[str] = set()
    for n in g.node:
        if n.op_type == "Constant" and n.output:
            t = _attr(n, "value", None)
            if t is not None and hasattr(t, "dims"):
                shapes[n.output[0]] = list(t.dims)
                consts.add(n.output[0])
    producer = {o: n for n in g.node for o in n.output}

    def is_weight(name: str, depth: int = 0) -> bool:
        if name in inits or name in consts:
            return True
        n = producer.get(name)
        if n is None or depth > 6 or n.op_type not in PASS_THROUGH or not n.input:
            return False
        return is_weight(n.input[0], depth + 1)

    nodes: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for n in g.node:
        op = n.op_type
        if op not in MAC_OPS:
            continue
        out = shapes.get(n.output[0]) if n.output else None
        macs: float | None = None
        weight = False
        if op in ("MatMul", "MatMulInteger", "QLinearMatMul"):
            ia, ib = (0, 3) if op == "QLinearMatMul" else (0, 1)
            a = shapes.get(n.input[ia])
            if _known(out) and a and isinstance(a[-1], int) and a[-1] > 0:
                macs = _prod(out) * a[-1]  # type: ignore[arg-type]
            weight = is_weight(n.input[ib]) or is_weight(n.input[ia])
        elif op == "Gemm":
            a = shapes.get(n.input[0])
            k_axis = 0 if _attr(n, "transA", 0) else 1
            if _known(out) and a and len(a) == 2 and isinstance(a[k_axis], int):
                macs = _prod(out) * a[k_axis]  # type: ignore[arg-type,operator]
            weight = is_weight(n.input[1])
        elif op in ("Conv", "QLinearConv"):
            iw = 3 if op == "QLinearConv" else 1
            w = shapes.get(n.input[iw])
            if _known(out) and _known(w):
                macs = _prod(out) * _prod(w[1:])  # type: ignore[index,arg-type]
            weight = is_weight(n.input[iw])
        else:  # ConvTranspose: every input element is multiplied by C_out/groups x kernel weights
            x, w = shapes.get(n.input[0]), shapes.get(n.input[1])
            if _known(x) and _known(w):
                macs = _prod(x) * _prod(w[1:])  # type: ignore[arg-type,index]
            weight = True
        name = n.name or (n.output[0] if n.output else op)
        if macs is None:
            unresolved.append(name)
            continue
        nodes.append({"name": name, "key": norm(name), "op_type": op, "macs": macs, "weight": weight,
                      "out_shape": out})
    params = sum(_prod(d) for d in inits.values() if d and _prod(d) >= 64)
    if unresolved:
        warnings.append(f"{len(unresolved)} MAC op(s) without known shapes (dynamic inputs?): "
                        + ", ".join(unresolved[:5]) + ("…" if len(unresolved) > 5 else ""))
    return {
        "path": str(p),
        "nodes": nodes,
        "total_macs": sum(n["macs"] for n in nodes),
        "weight_macs": sum(n["macs"] for n in nodes if n["weight"]),
        "activation_macs": sum(n["macs"] for n in nodes if not n["weight"]),
        "params": params,
        "unresolved": len(unresolved),
        "warnings": warnings,
    }


def match(qnn_nodes: list[dict[str, Any]], counted: dict[str, Any]) -> dict[str, Any]:
    """Join QHAS QNN op nodes (cycles, DRAM bytes) onto the ONNX nodes that carry the MACs.

    Several QNN nodes can come from one ONNX node; their cycles and bytes are added up on it.
    """
    idx = {n["key"]: n for n in counted["nodes"] if n["key"]}
    rows: dict[str, dict[str, Any]] = {}
    for q in qnn_nodes:
        k = norm(q.get("qnn_op") or q.get("op"))
        hit = idx.get(k)
        if hit is None and len(k) > _MIN_PREFIX:
            for L in range(len(k) - 1, _MIN_PREFIX - 1, -1):
                hit = idx.get(k[:L])
                if hit is not None:
                    break
        if hit is None:
            continue
        r = rows.setdefault(hit["key"], {"name": hit["name"], "op_type": hit["op_type"], "macs": hit["macs"],
                                         "weight": hit["weight"], "out_shape": hit["out_shape"], "cycles": 0.0,
                                         "dram_bytes": 0.0, "qnn_nodes": 0})
        r["cycles"] += float(q.get("cycles") or 0.0)
        r["dram_bytes"] += float(q.get("dram_read") or 0.0) + float(q.get("dram_write") or 0.0)
        r["qnn_nodes"] += 1
    matched = sum(r["macs"] for r in rows.values())
    total = counted.get("total_macs") or 0.0
    return {"rows": sorted(rows.values(), key=lambda r: -r["cycles"]), "matched_macs": matched,
            "coverage": matched / total if total else 0.0}
