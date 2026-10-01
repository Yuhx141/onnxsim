"""Tests for onnxsim.quark_cle (Gemm/MatMul cross-layer equalization)."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim.quark_cle import equalize_linear_layers


def _model(body, **inits):
    m = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": 17]> g (float[3,6] x) => (float[3,4] y) {{ {body} }}'
    )
    m.graph.initializer.extend(
        numpy_helper.from_array(np.asarray(v, np.float32), k) for k, v in inits.items()
    )
    return m


def _run(m, x):
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    return s.run(None, {"x": x})[0]


def _arr(m, name):
    return numpy_helper.to_array(next(i for i in m.graph.initializer if i.name == name))


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        w1=rng.standard_normal((6, 8)) * 2,
        b1=rng.standard_normal(8),
        w2=rng.standard_normal((8, 4)) * 0.3,
    )


X = np.random.default_rng(1).standard_normal((3, 6)).astype(np.float32)


@pytest.mark.parametrize("act", ["Relu", "LeakyRelu<alpha=0.2>", "PRelu"])
def test_function_is_preserved_and_ranges_balance(act):
    args = "(h0)" if act != "PRelu" else "(h0, slope)"
    extra = {"slope": [0.1]} if act == "PRelu" else {}
    m = _model(
        f"h0 = Gemm(x, w1, b1) h1 = {act}{args} y = MatMul(h1, w2)",
        **_weights(),
        **extra,
    )
    out = equalize_linear_layers(m, steps=-1)
    np.testing.assert_allclose(_run(out, X), _run(m, X), rtol=1e-4, atol=1e-4)
    w1, w2 = _arr(out, "w1"), _arr(out, "w2")
    assert not np.allclose(w1, _arr(m, "w1"))
    # bias is part of the head range, so the weight ranges only agree when the
    # bias does not dominate; the weights ranges must at least have moved closer
    r1, r2 = np.abs(w1).max(0), np.abs(w2).max(1)
    o1, o2 = np.abs(_arr(m, "w1")).max(0), np.abs(_arr(m, "w2")).max(1)
    assert np.abs(np.log(r1 / r2)).max() < np.abs(np.log(o1 / o2)).max()


def test_trans_b_gemm_and_matmul_tail():
    w = _weights()
    m = _model(
        "h0 = Gemm<transB=1>(x, w1t, b1) h1 = Relu(h0) y = MatMul(h1, w2)",
        w1t=w["w1"].T,
        b1=w["b1"],
        w2=w["w2"],
    )
    out = equalize_linear_layers(m)
    np.testing.assert_allclose(_run(out, X), _run(m, X), rtol=1e-4, atol=1e-4)
    assert not np.allclose(_arr(out, "w1t"), _arr(m, "w1t"))


def test_shared_intermediate_or_unsupported_layers_are_left_alone():
    w = _weights()
    # the hidden tensor feeds two consumers
    m = _model(
        "h0 = Gemm(x, w1, b1) h1 = Relu(h0) y = MatMul(h1, w2) z = Add(h1, h1)",
        **w,
    )
    m.graph.output.append(onnx.helper.make_tensor_value_info("z", 1, [3, 8]))
    out = equalize_linear_layers(m)
    assert out.SerializeToString() == m.SerializeToString()
    # a Sigmoid between the layers is not positive-homogeneous
    m = _model("h0 = Gemm(x, w1, b1) h1 = Sigmoid(h0) y = MatMul(h1, w2)", **w)
    assert equalize_linear_layers(m).SerializeToString() == m.SerializeToString()


def test_small_channels_below_the_threshold_are_untouched():
    w = {k: v * 0.01 for k, v in _weights().items()}
    m = _model("h0 = Gemm(x, w1, b1) h1 = Relu(h0) y = MatMul(h1, w2)", **w)
    out = equalize_linear_layers(m)
    np.testing.assert_array_equal(_arr(out, "w1"), _arr(m, "w1"))


def test_input_model_is_not_modified():
    m = _model("h0 = Gemm(x, w1, b1) h1 = Relu(h0) y = MatMul(h1, w2)", **_weights())
    before = m.SerializeToString()
    equalize_linear_layers(m)
    assert m.SerializeToString() == before
