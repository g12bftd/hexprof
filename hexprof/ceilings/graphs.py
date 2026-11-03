"""Synthetic ONNX graphs for the ceiling measurements (built with onnx.helper, no torch).

* :func:`matmul_chain` – L identical square MatMuls ``[M,K] x [K,K]`` in one graph.
  Work grows with L while input/output transfer stays constant, so
  ``latency = t0 + ops / P`` (ops = 2*M*K*N per MatMul).
* :func:`linear_stack` – single-token (decode-like) stack of L ``[1,D] x [D,D]``
  linear layers. Each weight is read once per inference, so
  ``latency = t0 + bytes / B``. Widening D changes bytes but not the op count.
* :func:`gemma4_decoder_stack` – the first N decoder layers of a Gemma-4-E4B-shaped
  model for one decode step with an explicit, static KV cache (inputs
  ``past_key_i`` / ``past_value_i``, outputs the new token's key/value), i.e. the
  depth sweep "Gemma 4 decoder subsets of 4, 8, 12 ... layers".

Weights are random: timing depends on shapes and precision, not values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17
QUANT_SCALE_BYTES_PER_CHANNEL = 8  # per-output-channel scale (fp32) + offset (int32)


@dataclass
class Graph:
    name: str
    model: onnx.ModelProto
    inputs: dict[str, tuple[tuple[int, ...], str]]
    matmul_ops: float  # 2 x MACs of weight MatMuls
    attention_ops: float = 0.0  # 2 x MACs of activation-activation MatMuls (attention)
    weight_params: int = 0  # elements of weight matrices
    weight_out_channels: int = 0  # sum of output channels (for per-channel scales)
    other_params: int = 0  # norms etc.
    kv_elements: int = 0  # KV-cache elements read per step (decoder graphs)
    meta: dict[str, Any] = field(default_factory=dict)
    directory: Path | None = None  # set when weights were streamed to <directory>/model.data

    @property
    def ops(self) -> float:
        return self.matmul_ops + self.attention_ops

    def weight_bytes(self, weight_bits: int | None, include_scales: bool = True) -> float:
        """Weight bytes read per inference at deployed precision (+ quantization scales)."""
        if weight_bits is None:  # fp16
            return self.weight_params * 2.0 + self.other_params * 2.0
        b = self.weight_params * weight_bits / 8.0 + self.other_params * 2.0
        if include_scales:
            b += self.weight_out_channels * QUANT_SCALE_BYTES_PER_CHANNEL
        return b

    def kv_bytes(self, act_bits: int | None) -> float:
        return self.kv_elements * ((act_bits or 16) / 8.0)

    def save(self, path: str | Path) -> Path:
        """Save as an AI Hub ONNX directory: ``<path>/model.onnx`` (+ ``model.data``).

        Weights live in external data so graphs above protobuf's 2 GB limit work.
        Graphs built with ``out_dir`` already streamed their weights there; saving to
        that same directory only writes the (small) graph proto.
        """
        d = Path(path)
        if self.directory is not None:
            if d.resolve() != self.directory.resolve():
                raise ValueError(f"graph was streamed to {self.directory}; save it there")
            onnx.save_model(self.model, str(d / "model.onnx"))
            return d
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
        onnx.save_model(
            self.model,
            str(d / "model.onnx"),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
            size_threshold=1024,
        )
        return d


    def sample_inputs(self, n: int = 1, seed: int = 0, kind: str = "random") -> dict[str, list[np.ndarray]]:
        return make_inputs(self.inputs, n, seed, kind)


def make_inputs(
    input_specs: dict[str, tuple[tuple[int, ...], str]] | dict[str, list[Any]],
    n: int = 1,
    seed: int = 0,
    kind: str = "random",
) -> dict[str, list[np.ndarray]]:
    """Calibration / inference data for a graph. ``kind``: random | kv_empty | kv_full.

    RoPE inputs get real cos/sin values, masks are 0 (attend) or MASK_MIN, and for
    ``kv_empty`` the past KV cache is zero with only the new token visible.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, list[np.ndarray]] = {}
    for name, (shape, dtype) in input_specs.items():
        shape = tuple(shape)
        vals = []
        for _ in range(n):
            if name.startswith(("cos", "sin")):
                pos = int(rng.integers(0, 4096))
                hd = shape[-1]
                inv = 1.0 / (10000 ** (np.arange(0, hd, 2) / hd))
                ang = np.concatenate([pos * inv, pos * inv]).astype(np.float32)
                v = (np.cos(ang) if name.startswith("cos") else np.sin(ang)).reshape(shape)
            elif name.startswith("mask"):
                v = np.zeros(shape, np.float32)
                if kind == "kv_empty":
                    v[..., :-1] = MASK_MIN  # only the new token is visible
            elif name.startswith("past_") and kind == "kv_empty":
                v = np.zeros(shape, np.float32)
            else:
                v = rng.standard_normal(shape).astype(np.float32)
            vals.append(np.ascontiguousarray(v.astype(dtype)))
        out[name] = vals
    return out


_INT8_UNIFORM_STD = math.sqrt((255**2 - 1) / 12.0)  # std of uniform integers in [-127, 127]
MASK_MIN = -50.0  # finite "minus infinity": keeps int16 activation ranges sane


class _B:
    """Tiny graph builder."""

    def __init__(self, seed: int, out_dir: str | Path | None = None) -> None:
        self.dir = Path(out_dir) if out_dir is not None else None
        self._data = None
        self._offset = 0
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            for f in self.dir.iterdir():
                f.unlink()
            self._data = open(self.dir / "model.data", "wb")  # noqa: SIM115 - closed in model()
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.inputs: list[onnx.ValueInfoProto] = []
        self.outputs: list[onnx.ValueInfoProto] = []
        self.input_specs: dict[str, tuple[tuple[int, ...], str]] = {}
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.weight_params = 0
        self.weight_out = 0
        self.other_params = 0
        self.matmul_ops = 0.0
        self.attn_ops = 0.0

    def name(self, p: str) -> str:
        self.n += 1
        return f"{p}_{self.n}"

    def inp(self, name: str, shape: tuple[int, ...]) -> str:
        self.inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape)))
        self.input_specs[name] = (tuple(shape), "float32")
        return name

    def out(self, name: str, shape: tuple[int, ...]) -> None:
        self.outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape)))

    def const(self, arr: np.ndarray, p: str = "c") -> str:
        n = self.name(p)
        if self._data is None or arr.nbytes < 1024:
            self.inits.append(numpy_helper.from_array(arr, n))
            return n
        # Stream to model.data (4 KiB aligned for mmap) instead of holding GBs in protos.
        pad = (-self._offset) % 4096
        self._data.write(b"\0" * pad)
        self._offset += pad
        arr = np.ascontiguousarray(arr)
        arr.tofile(self._data)
        t = TensorProto()
        t.name = n
        t.data_type = helper.np_dtype_to_tensor_dtype(arr.dtype)
        t.dims.extend(arr.shape)
        t.data_location = TensorProto.EXTERNAL
        for k, v in (("location", "model.data"), ("offset", str(self._offset)), ("length", str(arr.nbytes))):
            e = t.external_data.add()
            e.key, e.value = k, v
        self._offset += arr.nbytes
        self.inits.append(t)
        return n

    def i64(self, vals: list[int]) -> str:
        return self.const(np.array(vals, dtype=np.int64), "shape")

    def op(self, op_type: str, inputs: list[str], p: str | None = None, **attrs: Any) -> str:
        o = self.name(p or op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [o], name=self.name(op_type), **attrs))
        return o

    def linear(self, x: str, rows: int, d_in: int, d_out: int, p: str = "W") -> str:
        # Uniform random ints scaled to std 1/sqrt(d_in): ~4x faster to generate than
        # normals for multi-GB sweeps, and still incompressible (no structure the
        # compiler or DMA could exploit).
        w = self.rng.integers(-127, 128, size=(d_in, d_out), dtype=np.int8).astype(np.float32)
        w *= 1.0 / (_INT8_UNIFORM_STD * math.sqrt(d_in))
        self.weight_params += d_in * d_out
        self.weight_out += d_out
        self.matmul_ops += 2.0 * rows * d_in * d_out
        return self.op("MatMul", [x, self.const(w, p)], "mm")

    def rmsnorm(self, x: str, dim: int, eps: float = 1e-6, scale: bool = True) -> str:
        sq = self.op("Mul", [x, x])
        ms = self.op("ReduceMean", [sq], axes=[-1], keepdims=1)
        rs = self.op("Reciprocal", [self.op("Sqrt", [self.op("Add", [ms, self.const(np.array(eps, np.float32))])])])
        y = self.op("Mul", [x, rs])
        if scale:
            w = (1.0 + 0.1 * self.rng.standard_normal(dim)).astype(np.float32)
            self.other_params += dim
            y = self.op("Mul", [y, self.const(w, "norm")])
        return y

    def gelu_tanh(self, x: str) -> str:
        x3 = self.op("Mul", [self.op("Mul", [x, x]), x])
        inner = self.op("Add", [x, self.op("Mul", [x3, self.const(np.array(0.044715, np.float32))])])
        t = self.op("Tanh", [self.op("Mul", [inner, self.const(np.array(math.sqrt(2 / math.pi), np.float32))])])
        one_p = self.op("Add", [t, self.const(np.array(1.0, np.float32))])
        return self.op("Mul", [self.op("Mul", [x, one_p]), self.const(np.array(0.5, np.float32))])

    def rope(self, x: str, cos: str, sin: str) -> str:
        x1, x2 = self.name("x1"), self.name("x2")
        self.nodes.append(helper.make_node("Split", [x], [x1, x2], name=self.name("Split"), axis=-1, num_outputs=2)
                          if OPSET >= 18 else helper.make_node("Split", [x], [x1, x2], name=self.name("Split"), axis=-1))
        rot = self.op("Concat", [self.op("Neg", [x2]), x1], axis=-1)
        return self.op("Add", [self.op("Mul", [x, cos]), self.op("Mul", [rot, sin])])

    def model(self, name: str) -> onnx.ModelProto:
        if self._data is not None:
            self._data.close()
            self._data = None
        g = helper.make_graph(self.nodes, name, self.inputs, self.outputs, self.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="hexprof")
        m.ir_version = 8
        return m


def matmul_chain(size: int, layers: int, rows: int | None = None, seed: int = 0,
                 out_dir: str | Path | None = None) -> Graph:
    """``layers`` chained ``[rows, size] x [size, size]`` MatMuls (rows defaults to size)."""
    if size % 32 or (rows or size) % 32:
        raise ValueError("use multiples of 32 so the matrix unit, not padding, sets the pace")
    rows = rows or size
    b = _B(seed, out_dir)
    x = b.inp("x", (rows, size))
    for _ in range(layers):
        x = b.linear(x, rows, size, size)
    b.nodes.append(helper.make_node("Identity", [x], ["y"], name="out"))
    b.out("y", (rows, size))
    name = f"matmul_s{size}_m{rows}_l{layers}"
    return Graph(name, b.model(name), b.input_specs, b.matmul_ops, weight_params=b.weight_params,
                 weight_out_channels=b.weight_out, meta={"kind": "matmul_chain", "size": size, "rows": rows,
                                                         "layers": layers}, directory=b.dir)


def linear_stack(width: int, layers: int, tokens: int = 1, seed: int = 0,
                 out_dir: str | Path | None = None) -> Graph:
    """Single-token stack of ``layers`` ``[tokens, width] x [width, width]`` linear layers."""
    b = _B(seed, out_dir)
    x = b.inp("x", (tokens, width))
    for _ in range(layers):
        x = b.linear(x, tokens, width, width)
    b.nodes.append(helper.make_node("Identity", [x], ["y"], name="out"))
    b.out("y", (tokens, width))
    name = f"linear_w{width}_l{layers}_t{tokens}"
    return Graph(name, b.model(name), b.input_specs, b.matmul_ops, weight_params=b.weight_params,
                 weight_out_channels=b.weight_out, meta={"kind": "linear_stack", "width": width, "layers": layers,
                                                         "tokens": tokens}, directory=b.dir)


@dataclass(frozen=True)
class Gemma4Dims:
    """Gemma-4-E4B text decoder dimensions (from qai_hub_models gemma_4_e4b_it)."""

    hidden: int = 2560
    intermediate: int = 10240  # read from the HF config when available (see load_gemma4_dims)
    heads: int = 8
    kv_heads: int = 2
    head_dim: int = 256  # sliding-window layers
    global_head_dim: int = 512  # full-attention layers
    sliding_window: int = 512
    pattern: int = 6  # every 6th layer is full attention
    ple_dim: int = 256  # per-layer-embedding width
    num_layers: int = 42
    source: str = "defaults"

    def is_global(self, i: int) -> bool:
        return (i + 1) % self.pattern == 0


def load_gemma4_dims(repo: str = "google/gemma-4-E4B-it") -> Gemma4Dims:
    """Read the real dimensions from the Hugging Face config (falls back to defaults)."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(repo)
        t = getattr(cfg, "text_config", cfg)
        inter = t.intermediate_size
        inter = inter[0] if isinstance(inter, (list, tuple)) else inter
        pattern = Gemma4Dims.pattern
        lt = getattr(t, "layer_types", None)
        if lt and "full_attention" in lt:
            pattern = lt.index("full_attention") + 1
        return Gemma4Dims(
            hidden=t.hidden_size,
            intermediate=int(inter),
            heads=t.num_attention_heads,
            kv_heads=t.num_key_value_heads,
            head_dim=getattr(t, "head_dim", Gemma4Dims.head_dim),
            global_head_dim=getattr(t, "global_head_dim", None) or Gemma4Dims.global_head_dim,
            sliding_window=getattr(t, "sliding_window", None) or Gemma4Dims.sliding_window,
            pattern=pattern,
            ple_dim=getattr(t, "hidden_size_per_layer_input", None) or Gemma4Dims.ple_dim,
            num_layers=t.num_hidden_layers,
            source=f"hf:{repo}",
        )
    except Exception as e:  # noqa: BLE001 - offline / gated / old transformers
        print(f"NOTE: could not read {repo} config ({type(e).__name__}); using built-in Gemma 4 E4B dims "
              f"(intermediate_size={Gemma4Dims.intermediate} is an assumption — pass --intermediate-size to override).")
        return Gemma4Dims()


def gemma4_decoder_stack(n_layers: int, dims: Gemma4Dims = Gemma4Dims(), context: int = 4096, seed: int = 0,
                         out_dir: str | Path | None = None) -> Graph:
    """First ``n_layers`` Gemma-4-shaped decoder layers for one decode step (explicit static KV cache)."""
    b = _B(seed, out_dir)
    H, KV = dims.heads, dims.kv_heads
    G = H // KV
    x = b.inp("hidden_in", (1, dims.hidden))
    kv_elems = 0
    rope_inputs: dict[int, tuple[str, str]] = {}
    masks: dict[int, str] = {}
    for i in range(n_layers):
        glob = dims.is_global(i)
        hd = dims.global_head_dim if glob else dims.head_dim
        span = context if glob else min(dims.sliding_window, context)
        if hd not in rope_inputs:
            tag = "global" if glob else "local"
            rope_inputs[hd] = (b.inp(f"cos_{tag}", (1, 1, hd)), b.inp(f"sin_{tag}", (1, 1, hd)))
        if span not in masks:
            masks[span] = b.inp(f"mask_{span}", (1, 1, span))
        cos, sin = rope_inputs[hd]
        ple = b.inp(f"per_layer_input_{i}", (1, dims.ple_dim))
        past_k = b.inp(f"past_key_{i}", (KV, span - 1, hd))
        past_v = b.inp(f"past_value_{i}", (KV, span - 1, hd))
        kv_elems += 2 * KV * (span - 1) * hd

        # --- attention
        h = b.rmsnorm(x, dims.hidden)
        q = b.op("Reshape", [b.linear(h, 1, dims.hidden, H * hd, "Wq"), b.i64([H, 1, hd])])
        k = b.op("Reshape", [b.linear(h, 1, dims.hidden, KV * hd, "Wk"), b.i64([KV, 1, hd])])
        v = b.op("Reshape", [b.linear(h, 1, dims.hidden, KV * hd, "Wv"), b.i64([KV, 1, hd])])
        q = b.rope(b.rmsnorm(q, hd), cos, sin)
        k = b.rope(b.rmsnorm(k, hd), cos, sin)
        v = b.rmsnorm(v, hd, scale=False)
        b.nodes.append(helper.make_node("Identity", [k], [f"new_key_{i}"], name=b.name("Identity")))
        b.nodes.append(helper.make_node("Identity", [v], [f"new_value_{i}"], name=b.name("Identity")))
        b.out(f"new_key_{i}", (KV, 1, hd))
        b.out(f"new_value_{i}", (KV, 1, hd))
        K = b.op("Concat", [past_k, k], axis=1)  # [KV, span, hd]
        V = b.op("Concat", [past_v, v], axis=1)
        qg = b.op("Reshape", [q, b.i64([KV, G, hd])])
        scores = b.op("MatMul", [qg, b.op("Transpose", [K], perm=[0, 2, 1])])  # [KV, G, span]
        scores = b.op("Add", [scores, masks[span]])
        attn = b.op("MatMul", [b.op("Softmax", [scores], axis=-1), V])  # [KV, G, hd]
        b.attn_ops += 2 * 2.0 * H * span * hd
        o = b.linear(b.op("Reshape", [attn, b.i64([1, H * hd])]), 1, H * hd, dims.hidden, "Wo")
        x = b.op("Add", [x, b.rmsnorm(o, dims.hidden)])
        # --- gated MLP
        h = b.rmsnorm(x, dims.hidden)
        gate = b.gelu_tanh(b.linear(h, 1, dims.hidden, dims.intermediate, "Wgate"))
        up = b.linear(h, 1, dims.hidden, dims.intermediate, "Wup")
        down = b.linear(b.op("Mul", [gate, up]), 1, dims.intermediate, dims.hidden, "Wdown")
        x = b.op("Add", [x, b.rmsnorm(down, dims.hidden)])
        # --- per-layer embedding (PLE) gate
        pg = b.op("Mul", [b.gelu_tanh(b.linear(x, 1, dims.hidden, dims.ple_dim, "Wple_gate")), ple])
        pp = b.linear(pg, 1, dims.ple_dim, dims.hidden, "Wple_proj")
        x = b.op("Add", [x, b.rmsnorm(pp, dims.hidden)])
    b.nodes.append(helper.make_node("Identity", [x], ["hidden_out"], name="out"))
    b.out("hidden_out", (1, dims.hidden))
    name = f"gemma4_dec_n{n_layers}_c{context}"
    return Graph(name, b.model(name), b.input_specs, b.matmul_ops, attention_ops=b.attn_ops,
                 weight_params=b.weight_params, weight_out_channels=b.weight_out, other_params=b.other_params,
                 kv_elements=kv_elems, meta={"kind": "gemma4_decoder", "layers": n_layers, "context": context,
                                             "dims": dims.__dict__}, directory=b.dir)
