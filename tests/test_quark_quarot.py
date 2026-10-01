"""Tests for onnxsim.quark_quarot (offline R1 residual-stream rotation)."""

import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_quarot as qr

D, F, V = 16, 24, 20


def _run(model, feed):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feed)[0]


def _inits(model):
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def _name_nodes_by_output(model):
    # The text format has no node-name syntax: name each node after its output.
    for n in model.graph.node:
        n.name = n.output[0]
    return model


def _block(gemm_out_transb=False, outlier=False, seed=0):
    """embed -> RMSNorm(g1) -> up -> relu -> down(+bias, Gemm) -> +residual
    -> RMSNorm(g2) -> lm_head."""
    down = (
        "Gemm<transB=1>(a, w_down, b_down)"
        if gemm_out_transb
        else "Gemm(a, w_down, b_down)"
    )
    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 13]>
        agraph (int64[T] ids) => (float[T,{V}] logits)
        <float eps = {{1e-6}}>
        {{
            x0 = Gather(emb, ids)
            sq1 = Mul(x0, x0)
            ms1 = ReduceMean<axes=[-1], keepdims=1>(sq1)
            e1 = Add(ms1, eps)
            r1 = Sqrt(e1)
            h1 = Div(x0, r1)
            n1 = Mul(h1, g1)
            up = MatMul(n1, w_up)
            a = Relu(up)
            dn = {down}
            x1 = Add(x0, dn)
            sq2 = Mul(x1, x1)
            ms2 = ReduceMean<axes=[-1], keepdims=1>(sq2)
            e2 = Add(ms2, eps)
            r2 = Sqrt(e2)
            h2 = Div(x1, r2)
            n2 = Mul(h2, g2)
            logits = MatMul(n2, w_lm)
        }}
        """
    )
    rng = np.random.default_rng(seed)

    def w(shape, scale=1.0):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    emb = w((V, D))
    if outlier:
        emb[:, 3] *= 60.0  # a massive-activation channel
    down_w = w((D, F), 0.3) if gemm_out_transb else w((F, D), 0.3)
    for name, arr in (
        ("emb", emb),
        ("g1", w((D,)) * 0.5 + 1.0),
        ("g2", w((D,)) * 0.5 + 1.0),
        ("w_up", w((D, F), 0.3)),
        ("w_down", down_w),
        ("b_down", w((D,))),
        ("w_lm", w((D, V), 0.3)),
    ):
        model.graph.initializer.append(numpy_helper.from_array(arr, name))
    return _name_nodes_by_output(model)


CONFIG = {
    "R1_pairs": [
        {"prev_nodes": ["x0", "dn"], "next_nodes": ["up"], "norm_node": "n1"},
        {"prev_nodes": [], "next_nodes": ["logits"], "norm_node": "n2"},
    ]
}
IDS = {"ids": np.array([1, 5, 7, 3, 0, 19, 2, 11], dtype=np.int64)}


# -- make_rotation ---------------------------------------------------------------


@pytest.mark.parametrize("dim", [1, 2, 16, 12, 30])
@pytest.mark.parametrize("random_had", [False, True])
def test_make_rotation_is_orthogonal(dim, random_had):
    r = qr.make_rotation(dim, random_had, seed=1)
    assert r.shape == (dim, dim)
    np.testing.assert_allclose(r.T @ r, np.eye(dim), atol=1e-12)


def test_make_rotation_hadamard_structure_and_seeding():
    h = qr.make_rotation(8)
    np.testing.assert_allclose(np.abs(h), 1 / np.sqrt(8))
    a, b = qr.make_rotation(8, True, seed=1), qr.make_rotation(8, True, seed=2)
    assert not np.allclose(a, b)
    np.testing.assert_array_equal(a, qr.make_rotation(8, True, seed=1))
    with pytest.raises(ValueError):
        qr.make_rotation(0)


# -- exactness -------------------------------------------------------------------


@pytest.mark.parametrize("dim_rot, random_had", [(D, False), (D, True)])
@pytest.mark.parametrize("transb", [False, True])
def test_rotation_preserves_the_function(dim_rot, random_had, transb):
    model = _block(gemm_out_transb=transb)
    out = qr.rotate_model(
        model, CONFIG, r_matrix_dim=dim_rot, use_random_had=random_had, seed=3
    )
    np.testing.assert_allclose(_run(out, IDS), _run(model, IDS), rtol=1e-4, atol=1e-4)
    onnx.checker.check_model(out)


def test_rotation_with_a_non_power_of_two_orthogonal_matrix():
    # d=12 stream, so build a block with D=12
    r = qr.make_rotation(12, seed=4)
    model = onnx.ModelProto()
    model.CopyFrom(_block())
    # shrink every D-sized tensor from 16 to 12 by slicing (still a valid model)
    for t in model.graph.initializer:
        a = numpy_helper.to_array(t)
        idx = tuple(slice(0, 12) if s == D else slice(None) for s in a.shape)
        t.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(a[idx]), t.name))
    out = qr.rotate_model(model, CONFIG, r1=r)
    np.testing.assert_allclose(_run(out, IDS), _run(model, IDS), rtol=1e-4, atol=1e-4)


def test_norm_scales_are_folded_to_ones_and_weights_change():
    model = _block()
    out = qr.rotate_model(model, CONFIG, r_matrix_dim=D, seed=0)
    before, after = _inits(model), _inits(out)
    for g in ("g1", "g2"):
        np.testing.assert_array_equal(after[g], np.ones(D, np.float32))
    for w in ("emb", "w_up", "w_down", "b_down", "w_lm"):
        assert not np.allclose(before[w], after[w]), w
    # writers are rotated, not scaled: their Frobenius norms are unchanged
    for w in ("emb", "w_down"):
        np.testing.assert_allclose(
            np.linalg.norm(before[w]), np.linalg.norm(after[w]), rtol=1e-5
        )


def test_rotation_spreads_outlier_channels_out_of_the_stream():
    model = _block(outlier=True)
    out = qr.rotate_model(model, CONFIG, r_matrix_dim=D, use_random_had=True, seed=1)

    def peak_to_rms(m):
        x0 = _inits(m)["emb"]
        return float(np.abs(x0).max() / np.sqrt(np.mean(x0**2)))

    assert peak_to_rms(model) > 3 * peak_to_rms(out)


def test_layernorm_with_bias_is_folded_into_the_next_bias():
    # With R = I the rotation is a no-op, so only the gamma/beta fold is tested.
    model = _name_nodes_by_output(
        parser.parse_model(
            f"""
            <ir_version: 10, opset_import: ["": 17]>
            agraph (float[N,{D}] x) => (float[N,{V}] y)
            {{
                n = LayerNormalization<axis=-1>(x, g, b)
                y = Gemm(n, w, c)
            }}
            """
        )
    )
    rng = np.random.default_rng(2)
    for name, shape in (("g", (D,)), ("b", (D,)), ("w", (D, V)), ("c", (V,))):
        model.graph.initializer.append(
            numpy_helper.from_array(rng.standard_normal(shape).astype(np.float32), name)
        )
    cfg = {"R1_pairs": [{"prev_nodes": [], "next_nodes": ["y"], "norm_node": "n"}]}
    out = qr.rotate_model(model, cfg, r1=np.eye(D))
    x = rng.standard_normal((5, D)).astype(np.float32)
    np.testing.assert_allclose(
        _run(out, {"x": x}), _run(model, {"x": x}), rtol=1e-4, atol=1e-4
    )
    after = _inits(out)
    np.testing.assert_array_equal(after["g"], np.ones(D, np.float32))
    np.testing.assert_array_equal(after["b"], np.zeros(D, np.float32))


def test_config_may_be_a_json_file(tmp_path):
    path = tmp_path / "rot.json"
    path.write_text(json.dumps(CONFIG))
    model = _block()
    out = qr.rotate_model(model, str(path), r_matrix_dim=D)
    np.testing.assert_allclose(_run(out, IDS), _run(model, IDS), rtol=1e-4, atol=1e-4)


def test_input_model_is_not_mutated():
    model = _block()
    before = model.SerializeToString()
    qr.rotate_model(model, CONFIG, r_matrix_dim=D)
    assert model.SerializeToString() == before


# -- validation ------------------------------------------------------------------


def test_validation_errors():
    model = _block()
    pair = lambda **k: {"R1_pairs": [{"prev_nodes": [], "next_nodes": [], **k}]}  # noqa: E731
    with pytest.raises(ValueError, match="not found"):
        qr.rotate_model(model, pair(next_nodes=["nope"]), r_matrix_dim=D)
    with pytest.raises(ValueError, match="not orthogonal"):
        qr.rotate_model(model, CONFIG, r1=np.ones((D, D)))
    with pytest.raises(ValueError, match="square"):
        qr.rotate_model(model, CONFIG, r1=np.ones((D, 3)))
    with pytest.raises(ValueError, match="rotation size"):
        qr.rotate_model(model, pair(next_nodes=["up"]), r_matrix_dim=8)
    with pytest.raises(ValueError, match="r1 or r_matrix_dim"):
        qr.rotate_model(model, CONFIG)
    twice_w = {
        "R1_pairs": [
            {"prev_nodes": ["dn"], "next_nodes": []},
            {"prev_nodes": ["dn"], "next_nodes": []},
        ]
    }
    with pytest.raises(ValueError, match="more than one pair"):
        qr.rotate_model(model, twice_w, r_matrix_dim=D)
    twice_r = {
        "R1_pairs": [
            {"prev_nodes": [], "next_nodes": ["up"]},
            {"prev_nodes": [], "next_nodes": ["up"]},
        ]
    }
    with pytest.raises(ValueError, match="more than one pair"):
        qr.rotate_model(model, twice_r, r_matrix_dim=D)
