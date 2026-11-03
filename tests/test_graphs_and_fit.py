import numpy as np
import onnx
import pytest

from hexprof.ceilings import fit, graphs

ort = pytest.importorskip("onnxruntime")


def _run(g: graphs.Graph, feeds: dict) -> dict:
    sess = ort.InferenceSession(g.model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def test_matmul_chain_accounting_and_numerics():
    g = graphs.matmul_chain(64, 3)
    onnx.checker.check_model(g.model)
    assert g.ops == 3 * 2 * 64 * 64 * 64
    assert g.weight_params == 3 * 64 * 64
    assert g.weight_bytes(8) == 3 * 64 * 64 + 3 * 64 * graphs.QUANT_SCALE_BYTES_PER_CHANNEL
    assert g.weight_bytes(None) == 3 * 64 * 64 * 2
    x = np.random.default_rng(1).standard_normal((64, 64)).astype(np.float32)
    y = _run(g, {"x": x})["y"]
    ws = [onnx.numpy_helper.to_array(t) for t in g.model.graph.initializer]
    ref = x
    for w in ws:
        ref = ref @ w
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)
    assert np.isfinite(y).all() and 0.2 < y.std() < 5  # 1/sqrt(K) scaling keeps activations O(1)
    with pytest.raises(ValueError):
        graphs.matmul_chain(100, 1)


def test_linear_stack_is_single_token():
    g = graphs.linear_stack(128, 4)
    assert g.inputs == {"x": ((1, 128), "float32")}
    assert g.ops == 4 * 2 * 128 * 128
    assert g.weight_bytes(4, include_scales=False) == 4 * 128 * 128 / 2


def test_gemma4_decoder_stack_runs_with_explicit_kv_cache(tmp_path):
    dims = graphs.Gemma4Dims(hidden=64, intermediate=128, heads=4, kv_heads=2, head_dim=16, global_head_dim=32,
                             sliding_window=8, pattern=3, ple_dim=8)
    g = graphs.gemma4_decoder_stack(3, dims, context=16)
    onnx.checker.check_model(g.model)
    # layers 0,1 local (span 8, hd 16), layer 2 global (span 16, hd 32)
    assert g.inputs["past_key_0"][0] == (2, 7, 16)
    assert g.inputs["past_key_2"][0] == (2, 15, 32)
    assert g.kv_elements == 2 * 2 * (7 * 16 * 2 + 15 * 32)
    feeds = {k: v[0] for k, v in g.sample_inputs(1, kind="kv_full").items()}
    out = _run(g, feeds)
    assert out["hidden_out"].shape == (1, 64) and np.isfinite(out["hidden_out"]).all()
    assert out["new_key_2"].shape == (2, 1, 32)
    empty = g.sample_inputs(1, kind="kv_empty")
    assert not empty["past_value_0"][0].any() and empty["mask_8"][0][..., :-1].max() == graphs.MASK_MIN
    # per-layer weight params: q,k,v,o + gate,up,down + PLE gate/proj
    local = 64 * 64 + 2 * 64 * 32 + 64 * 64 + 3 * 64 * 128 + 2 * 64 * 8
    glob = 64 * 128 + 2 * 64 * 64 + 128 * 64 + 3 * 64 * 128 + 2 * 64 * 8
    assert g.weight_params == 2 * local + glob
    d = g.save(tmp_path / "dec.onnx")
    assert sorted(p.name for p in d.iterdir()) == ["model.data", "model.onnx"]
    reloaded = onnx.load(str(d / "model.onnx"))
    onnx.checker.check_model(reloaded)


def test_fits_recover_ceilings():
    ops = [2 * 2048**3 * L for L in (1, 2, 4, 8, 16)]
    lat = [0.8 + o / 9.5e12 * 1e3 for o in ops]  # 0.8 ms + ops / 9.5 TOPS
    c = fit.compute_ceiling(ops, lat)
    assert c["P_tops"] == pytest.approx(9.5, rel=1e-6)
    assert c["t0_ms"] == pytest.approx(0.8, abs=1e-6)
    nbytes = [64e6, 128e6, 256e6, 512e6]
    lat = [0.3 + b / 55e9 * 1e3 + n for b, n in zip(nbytes, (0.01, -0.01, 0.02, -0.02))]
    b = fit.bandwidth_ceiling(nbytes, lat)
    assert b["B_gbps"] == pytest.approx(55, rel=0.01)
    lo, hi = b["B_gbps_2se_range"]
    assert lo < b["B_gbps"] < hi
    with pytest.raises(ValueError):
        fit.fit_line([1, 1], [2, 3])


def test_streamed_build_matches_in_memory_build(tmp_path):
    dims = graphs.Gemma4Dims(hidden=64, intermediate=128, heads=4, kv_heads=2, head_dim=16, global_head_dim=32,
                             sliding_window=8, pattern=3, ple_dim=8)
    mem = graphs.gemma4_decoder_stack(3, dims, context=16)
    d = tmp_path / "streamed.onnx"
    disk = graphs.gemma4_decoder_stack(3, dims, context=16, out_dir=d)
    disk.save(d)
    assert (d / "model.data").stat().st_size > 0
    assert len(disk.model.SerializeToString()) < 100_000  # weights are not in the proto
    onnx.checker.check_model(str(d / "model.onnx"))
    feeds = {k: v[0] for k, v in mem.sample_inputs(1, kind="kv_full").items()}
    a = _run(mem, feeds)
    sess = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"])
    b = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
    np.testing.assert_allclose(a["hidden_out"], b["hidden_out"], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError):
        disk.save(tmp_path / "elsewhere.onnx")
