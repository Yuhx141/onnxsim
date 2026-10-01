"""Tests for onnxsim.quark_tools (Quark-named post-quantization graph tools)."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_tools as qt


def _run(model, feed):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feed)


def _ops(model):
    return [n.op_type for n in model.graph.node]


def _qdq_model():
    # weight: int8 per-channel (axis 1) DQ; activation: Q->DQ pair before MatMul
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,2] x) => (float[N,2] y)
        <int8[2,2] wq = {10, -20, 30, 40}, float[2] ws = {0.5, 0.25}, int8[2] wz = {0, 4},
         float xs = {0.1}, uint8 xz = {128}>
        {
            xq = QuantizeLinear(x, xs, xz)
            xd = DequantizeLinear(xq, xs, xz)
            w = DequantizeLinear<axis = 1>(wq, ws, wz)
            y = MatMul(xd, w)
        }
        """
    )
    return model


def test_remove_qdq_strips_pairs_and_folds_weights():
    model = _qdq_model()
    out = qt.remove_qdq(model)
    assert _ops(out) == ["MatMul"]
    assert {i.name for i in out.graph.initializer} == {"w"}
    w = numpy_helper.to_array(out.graph.initializer[0])
    expected = (np.array([[10, -20], [30, 40]], np.float32) - [0, 4]) * [0.5, 0.25]
    np.testing.assert_allclose(w, expected)
    x = np.array([[1.0, 2.0]], np.float32)
    np.testing.assert_allclose(_run(out, {"x": x})[0], x @ expected, rtol=1e-6)
    onnx.checker.check_model(out)


def test_remove_qdq_without_fold_keeps_weight_dq():
    out = qt.remove_qdq(_qdq_model(), fold_weights=False)
    assert _ops(out) == ["DequantizeLinear", "MatMul"]


def test_remove_qdq_keeps_graph_output_name_via_identity():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N] x) => (float[N] y)
        <float s = {0.1}, uint8 z = {0}>
        {
            q = QuantizeLinear(x, s, z)
            y = DequantizeLinear(q, s, z)
        }
        """
    )
    out = qt.remove_qdq(model)
    assert _ops(out) == ["Identity"]
    assert out.graph.output[0].name == "y"


def test_remove_qdq_leaves_q_feeding_non_dq():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N] x) => (uint8[N] y)
        <float s = {0.1}, uint8 z = {0}>
        {
            q = QuantizeLinear(x, s, z)
            y = Identity(q)
        }
        """
    )
    assert _ops(qt.remove_qdq(model)) == ["QuantizeLinear", "Identity"]


def test_remove_qdq_does_not_mutate_input():
    model = _qdq_model()
    before = model.SerializeToString()
    qt.remove_qdq(model)
    assert model.SerializeToString() == before


def test_shared_initializer_made_unique():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[2] x) => (float[2] y)
        <float[2] c = {1.0, 2.0}>
        {
            a = Add(x, c)
            b = Mul(a, c)
            y = Sub(b, c)
        }
        """
    )
    out = qt.convert_shared_initializer_to_unique(model)
    names = [i.name for i in out.graph.initializer]
    assert sorted(names) == ["c", "c_copy1", "c_copy2"]
    used = [n.input[1] for n in out.graph.node]
    assert len(set(used)) == 3
    x = np.array([3.0, 4.0], np.float32)
    np.testing.assert_allclose(_run(out, {"x": x})[0], _run(model, {"x": x})[0])


def test_dynamic_to_fixed_propagates_static_shapes():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[N,4] x) => (float[N,4] y)
        {
            y = Relu(x)
        }
        """
    )
    out = qt.convert_dynamic_to_fixed(model, {"x": [8, 4]})
    dims = [d.dim_value for d in out.graph.output[0].type.tensor_type.shape.dim]
    assert dims == [8, 4]
    with pytest.raises(ValueError, match="not a graph input"):
        qt.convert_dynamic_to_fixed(model, {"nope": [1, 4]})
    with pytest.raises(ValueError, match="rank"):
        qt.convert_dynamic_to_fixed(model, {"x": [4]})


def test_replace_inf_weights():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[3] x) => (float[3] y)
        {
            y = Add(x, c)
        }
        """
    )
    # inf has no text-format literal, so the tensor is attached programmatically.
    arr = np.array([np.inf, -np.inf, 1.0], np.float32)
    model.graph.initializer.append(numpy_helper.from_array(arr, "c"))
    out = qt.replace_inf_weights(model, max_value=1e30)
    got = numpy_helper.to_array(out.graph.initializer[0])
    np.testing.assert_array_equal(got, np.array([1e30, -1e30, 1.0], np.float32))


# -- convert_s8s8_to_u8s8 ---------------------------------------------------------


def _s8_model():
    # int8 activation Q/DQ (zero point -5) feeding a MatMul with int8 weights
    return parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,2] x) => (float[N,2] y)
        <float xs = {0.05}, int8 xz = {-5},
         int8[2,2] wq = {10, -20, 30, 40}, float[2] ws = {0.5, 0.25}, int8[2] wz = {0, 4}>
        {
            xq = QuantizeLinear(x, xs, xz)
            xd = DequantizeLinear(xq, xs, xz)
            w = DequantizeLinear<axis = 1>(wq, ws, wz)
            y = MatMul(xd, w)
        }
        """
    )


def _inits(model):
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def test_s8s8_to_u8s8_is_exact_and_keeps_weights_int8():
    model = _s8_model()
    out = qt.convert_s8s8_to_u8s8(model)
    after = _inits(out)
    assert after["xz"].dtype == np.uint8 and int(after["xz"]) == 123  # -5 + 128
    assert after["wq"].dtype == np.int8 and after["wz"].dtype == np.int8
    x = np.random.default_rng(0).standard_normal((5, 2)).astype(np.float32) * 3
    np.testing.assert_array_equal(_run(out, {"x": x})[0], _run(model, {"x": x})[0])
    onnx.checker.check_model(out)


def test_s8s8_to_u8s8_copies_a_zero_point_shared_with_an_unconverted_node():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,2] x) => (float[N,2] y)
        <float s = {0.05}, int8 z = {-5}, int8[2,2] wq = {10, -20, 30, 40}>
        {
            xq = QuantizeLinear(x, s, z)
            xd = DequantizeLinear(xq, s, z)
            w = DequantizeLinear(wq, s, z)
            y = MatMul(xd, w)
        }
        """
    )
    out = qt.convert_s8s8_to_u8s8(model)
    after = _inits(out)
    assert after["z"].dtype == np.int8  # the weight DQ still uses the int8 original
    assert after["z_u8"].dtype == np.uint8 and int(after["z_u8"]) == 123
    x = np.random.default_rng(1).standard_normal((4, 2)).astype(np.float32)
    np.testing.assert_array_equal(_run(out, {"x": x})[0], _run(model, {"x": x})[0])


def test_s8s8_to_u8s8_leaves_an_int8_graph_output_alone():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N] x) => (int8[N] q, float[N] y)
        <float s = {0.1}, int8 z = {3}>
        {
            q = QuantizeLinear(x, s, z)
            y = DequantizeLinear(q, s, z)
        }
        """
    )
    assert _inits(qt.convert_s8s8_to_u8s8(model))["z"].dtype == np.int8


# -- half <-> float32 -------------------------------------------------------------


def _add_model():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[N,3] x) => (float[N,3] y)
        {
            h = Add(x, c)
            y = Relu(h)
        }
        """
    )
    # 0.1 / 1/3 are not exactly representable in half precision
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([0.1, 1 / 3, -2.5], np.float32), "c")
    )
    return model


@pytest.mark.parametrize(
    "to_half, to_fp32, tol",
    [
        (qt.convert_fp32_to_fp16, qt.convert_fp16_to_fp32, 1e-3),
        (qt.convert_fp32_to_bf16, qt.convert_bf16_to_fp32, 1e-2),
    ],
)
def test_half_round_trip_restores_structure_and_float32_types(to_half, to_fp32, tol):
    model = _add_model()
    half = to_half(model)
    assert "Cast" in _ops(half)  # boundary casts (keep_io_types=True)
    back = to_fp32(half)
    assert _ops(back) == _ops(model)  # the now-no-op casts are gone
    c = numpy_helper.to_array(back.graph.initializer[0])
    assert c.dtype == np.float32
    np.testing.assert_allclose(c, [0.1, 1 / 3, -2.5], rtol=tol)
    x = np.array([[1.0, -1.0, 3.0]], np.float32)
    np.testing.assert_allclose(
        _run(back, {"x": x})[0], _run(model, {"x": x})[0], rtol=tol, atol=tol
    )
    onnx.checker.check_model(back)


def test_half_conversion_without_keep_io_types_round_trips_io_types():
    half = qt.convert_fp32_to_fp16(_add_model(), keep_io_types=False)
    assert half.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    back = qt.convert_fp16_to_fp32(half)
    assert back.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert back.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def test_fp32_to_half_does_not_mutate_its_input():
    model = _add_model()
    before = model.SerializeToString()
    qt.convert_fp32_to_fp16(model)
    qt.convert_fp32_to_bf16(model)
    assert model.SerializeToString() == before


# -- opset / initializer-as-input -------------------------------------------------


def test_convert_opset_version():
    model = _add_model()
    out = qt.convert_opset_version(model, 21)
    assert [o.version for o in out.opset_import if o.domain == ""] == [21]
    assert [o.version for o in model.opset_import if o.domain == ""] == [17]
    with pytest.raises(ValueError, match="cannot convert to opset"):
        qt.convert_opset_version(model, 1000)


def test_remove_initializer_from_input():
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 17]>
        agraph (float[N,2] x, float[2] c) => (float[N,2] y)
        <float[2] c = {1.0, 2.0}>
        {
            y = Add(x, c)
        }
        """
    )
    out = qt.remove_initializer_from_input(model)
    assert [i.name for i in out.graph.input] == ["x"]
    assert [i.name for i in model.graph.input] == ["x", "c"]  # input untouched
