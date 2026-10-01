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


# -- quark_tools_extra ------------------------------------------------------------


def _ort_close(a, b, **kw):
    np.testing.assert_allclose(a, b, **kw)


def _conv_qdq(bias_dtype="int8", bias_scale=0.0007, opset=21):
    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": {opset}]>
        g (float[1,3,8,8] x) => (float[1,4,6,6] y)
        <float xs = {{0.02}}, int8 xz = {{0}}, float ws = {{0.01}}, int8 wz = {{0}},
         float bs = {{{bias_scale}}}, {bias_dtype} bz = {{0}},
         float ys = {{0.05}}, int8 yz = {{0}}>
        {{
            bd = DequantizeLinear(bq, bs, bz)
            wd = DequantizeLinear(wq, ws, wz)
            xq = QuantizeLinear(x, xs, xz)
            xd = DequantizeLinear(xq, xs, xz)
            c = Conv(xd, wd, bd)
            cq = QuantizeLinear(c, ys, yz)
            y = DequantizeLinear(cq, ys, yz)
        }}
        """
    )
    rng = np.random.default_rng(0)
    bq = np.array([10, -20, 30, 40], np.int8 if bias_dtype == "int8" else np.int32)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                rng.integers(-100, 100, (4, 3, 3, 3)).astype(np.int8), "wq"
            ),
            numpy_helper.from_array(bq, "bq"),
        ]
    )
    return model


def _inits(model):
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def test_a8w8_npu_to_cpu_rescales_bias_to_int32():
    model = _conv_qdq()
    out = qt.convert_a8w8_npu_to_a8w8_cpu(model)
    init = _inits(out)
    new_scale = np.float32(0.02) * np.float32(0.01)
    assert init["bq"].dtype == np.int32
    expected = (
        np.array([10, -20, 30, 40], np.float32) * np.float32(0.0007) / new_scale
    ).astype(np.int32)
    np.testing.assert_array_equal(init["bq"], expected)
    np.testing.assert_allclose(init["bs"], new_scale)
    assert init["bz"].dtype == np.int32 and init["bz"] == 0
    # the original is untouched and the dequantized bias is (nearly) unchanged
    assert _inits(model)["bq"].dtype == np.int8
    np.testing.assert_allclose(
        init["bq"] * init["bs"], np.array([10, -20, 30, 40]) * 0.0007, atol=2e-4
    )
    x = np.random.default_rng(1).standard_normal((1, 3, 8, 8)).astype(np.float32)
    _ort_close(_run(out, {"x": x})[0], _run(model, {"x": x})[0], atol=0.2)


def test_a8w8_npu_to_cpu_copies_a_shared_bias():
    model = _conv_qdq()
    # a second consumer of the bias initializers
    # (the text format cannot add a node to an existing model, so use helper)
    model.graph.node.append(
        onnx.helper.make_node("DequantizeLinear", ["bq", "bs", "bz"], ["b2"])
    )
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("b2", onnx.TensorProto.FLOAT, [4])
    )
    out = qt.convert_a8w8_npu_to_a8w8_cpu(model)
    assert _inits(out)["bq"].dtype == np.int8  # still int8 for the other user
    assert _inits(out)["bq_i32"].dtype == np.int32


def test_bias_int32_to_int16_clips_and_upgrades_opset():
    model = _conv_qdq(bias_dtype="int32", opset=13)
    for t in model.graph.initializer:
        if t.name == "bq":
            t.CopyFrom(
                numpy_helper.from_array(
                    np.array([100000, -100000, 5, -5], np.int32), "bq"
                )
            )
    out, changed = qt.convert_bias_int32_to_int16(model)
    assert changed is True
    init = _inits(out)
    assert init["bq"].dtype == np.int16 and init["bz"].dtype == np.int16
    np.testing.assert_array_equal(init["bq"], [32767, -32768, 5, -5])
    assert next(o.version for o in out.opset_import if o.domain == "") == 21
    out2, changed2 = qt.convert_bias_int32_to_int16(_conv_qdq())  # int8 bias
    assert changed2 is False
    assert _inits(out2)["bq"].dtype == np.int8


def test_customqdq_to_qdq_and_custom_ops_roundtrip():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21, "com.amd.quark": 1]>
        g (float[4] x) => (float[4] y)
        <float s = {0.1}, int8 z = {0}, bfloat16 zb = {0}>
        {
            a = com.amd.quark.ExtendedQuantizeLinear(x, s, z)
            b = com.amd.quark.ExtendedDequantizeLinear(a, s, z)
            c = com.amd.quark.ExtendedQuantizeLinear(b, s, zb)
            y = com.amd.quark.ExtendedDequantizeLinear(c, s, zb)
        }
        """
    )
    out = qt.convert_customqdq_to_qdq(model)
    assert [(n.op_type, n.domain) for n in out.graph.node] == [
        ("QuantizeLinear", "com.microsoft"),
        ("DequantizeLinear", "com.microsoft"),
        ("ExtendedQuantizeLinear", "com.amd.quark"),  # bfloat16 stays custom
        ("ExtendedDequantizeLinear", "com.amd.quark"),
    ]
    assert "com.microsoft" in {o.domain for o in out.opset_import}
    old = qt.convert_custom_ops(model)
    assert {(n.op_type, n.domain) for n in old.graph.node} == {
        ("VitisQuantizeLinear", "com.vai.quantize"),
        ("VitisDequantizeLinear", "com.vai.quantize"),
    }
    back = qt.convert_custom_ops(
        old,
        "com.amd.quark",
        {v: k for k, v in qt.CUSTOM_OP_NAME_MAPPING.items()},
    )
    assert [(n.op_type, n.domain) for n in back.graph.node] == [
        (n.op_type, n.domain) for n in model.graph.node
    ]
    # the argument was not modified
    assert model.graph.node[0].op_type == "ExtendedQuantizeLinear"


def test_fp16_to_bf16_rounds_weights_and_keeps_boundary_in_fp16():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float16[4] x) => (float16[4] y)
        { y = Add(x, w) }
        """
    )
    w = np.array([0.1, -2.5, 3.14159, 1000.0], np.float16)
    model.graph.initializer.append(numpy_helper.from_array(w, "w"))
    out = qt.convert_fp16_to_bf16(model)
    assert [n.op_type for n in out.graph.node] == ["Cast", "Add", "Cast"]
    assert out.graph.node[0].attribute[0].i == onnx.TensorProto.BFLOAT16
    assert out.graph.node[2].attribute[0].i == onnx.TensorProto.FLOAT16
    assert out.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    (t,) = out.graph.initializer
    assert t.data_type == onnx.TensorProto.BFLOAT16
    # value == float32 -> bfloat16 nearest-even (keep the top 16 bits)
    bits = w.astype(np.float32).view(np.uint32).astype(np.uint64)
    want = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    np.testing.assert_array_equal(np.frombuffer(t.raw_data, np.uint16), want)
    assert model.graph.initializer[0].data_type == onnx.TensorProto.FLOAT16
    onnx.checker.check_model(out)


def test_nchw_to_nhwc_input_and_output_are_nhwc():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,6] x, float[N,5] side) => (float[1,3,8,6] y, float[N,5] z)
        {
            y = Relu(x)
            z = Neg(side)
        }
        """
    )
    out = qt.convert_nchw_to_nhwc(model)
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    assert dims(out.graph.input[0]) == [1, 8, 6, 3]
    assert [o.name for o in out.graph.output] == ["y_transpose", "z"]
    assert dims(out.graph.output[0]) == [1, 8, 6, 3]
    assert _ops(out).count("Transpose") == 2  # `side` is not 4-D: untouched
    x = np.random.default_rng(0).standard_normal((1, 3, 8, 6)).astype(np.float32)
    got = _run(out, {"x": x.transpose(0, 2, 3, 1), "side": np.ones((2, 5), np.float32)})
    np.testing.assert_array_equal(got[0], np.maximum(x, 0).transpose(0, 2, 3, 1))


def test_nchw_to_nhwc_keeps_quantized_output_quantized():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x) => (float[1,3,8,8] y)
        <float s = {0.1}, int8 z = {0}>
        {
            r = Relu(x)
            q = QuantizeLinear(r, s, z)
            y = DequantizeLinear(q, s, z)
        }
        """
    )
    out = qt.convert_nchw_to_nhwc(model)
    assert _ops(out) == [
        "Transpose",
        "Relu",
        "QuantizeLinear",
        "DequantizeLinear",
        "Transpose",
        "QuantizeLinear",
        "DequantizeLinear",
    ]
    assert out.graph.output[0].name == "y_transpose_DequantizeLinear"
    x = np.random.default_rng(0).standard_normal((1, 8, 8, 3)).astype(np.float32)
    ref = _run(model, {"x": x.transpose(0, 3, 1, 2)})[0].transpose(0, 2, 3, 1)
    np.testing.assert_array_equal(_run(out, {"x": x})[0], ref)


def _qop_model(extra=""):
    return parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 13]>
        g (float[2,4] x, float[2,4] u) => (float[2,4] y)
        <float s = {{0.1}}, uint8 z = {{128}}, float sw = {{0.05}}, uint8 zw = {{120}},
         uint8[4,4] wq = {{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}}>
        {{
            xq = QuantizeLinear(x, s, z)
            xd = DequantizeLinear(xq, s, z)
            uq = QuantizeLinear(u, s, z)
            ud = DequantizeLinear(uq, s, z)
            wd = DequantizeLinear(wq, sw, zw)
            m = MatMul(xd, wd)
            mq = QuantizeLinear(m, s, z)
            md = DequantizeLinear(mq, s, z)
            a = Add(md, ud)
            aq = QuantizeLinear(a, s, z)
            ad = DequantizeLinear(aq, s, z)
            g1 = Sigmoid(ad)
            gq = QuantizeLinear(g1, s, z)
            y = DequantizeLinear(gq, s, z)
            {extra}
        }}
        """
    )


def test_qdq_to_qop_fuses_matmul_add_sigmoid():
    model = _qop_model()
    out = qt.convert_qdq_to_qop(model)
    assert [(n.op_type, n.domain) for n in out.graph.node] == [
        ("QuantizeLinear", ""),
        ("QuantizeLinear", ""),
        ("QLinearMatMul", ""),
        ("QLinearAdd", "com.microsoft"),
        ("QLinearSigmoid", "com.microsoft"),
        ("DequantizeLinear", ""),
    ]
    assert "com.microsoft" in {o.domain for o in out.opset_import}
    assert len(model.graph.node) == 14
    rng = np.random.default_rng(0)
    feed = {
        "x": rng.standard_normal((2, 4)).astype(np.float32),
        "u": rng.standard_normal((2, 4)).astype(np.float32),
    }
    _ort_close(_run(out, feed)[0], _run(model, feed)[0], atol=0.3)


def test_qdq_to_qop_skips_ops_with_other_consumers():
    # the MatMul result is also a graph output: nothing to fuse for it
    model = _qop_model()
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("m", onnx.TensorProto.FLOAT, [2, 4])
    )
    out = qt.convert_qdq_to_qop(model)
    assert "QLinearMatMul" not in _ops(out) and "QLinearAdd" in _ops(out)


def test_resize_fs_to_pof2s_sets_power_of_two_scales():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 13]>
        g (float[1,1,4,4] x) => (float[1,1,8,8] y)
        <float s1 = {0.037}, int8 z1 = {3}, float s2 = {0.0123}, int8 z2 = {-5},
         float[4] scales = {1.0, 1.0, 2.0, 2.0}>
        {
            q1 = QuantizeLinear(x, s1, z1)
            d1 = DequantizeLinear(q1, s1, z1)
            r = Resize<mode = "nearest">(d1, , scales)
            q2 = QuantizeLinear(r, s2, z2)
            y = DequantizeLinear(q2, s2, z2)
        }
        """
    )
    out = qt.convert_resize_fs_to_pof2s(model)
    init = _inits(out)
    # max(|(-128-3)*0.037|, |(127-3)*0.037|) / 128 = 0.03787 -> 2**ceil(log2) = 2**-4
    assert init["s1"] == 2.0**-4 and init["z1"] == 0 and init["z1"].dtype == np.int8
    # max(|(-128+5)*0.0123|, |(127+5)*0.0123|)/128 = 0.012685 -> 2**-6
    assert init["s2"] == 2.0**-6 and init["z2"] == 0
    assert _inits(model)["z1"] == 3  # input untouched


def test_resize_fs_to_pof2s_copies_a_shared_scale():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 13]>
        g (float[1,1,4,4] x) => (float[1,1,8,8] y, float[1,1,4,4] w)
        <float s = {0.037}, int8 z = {3}, float[4] scales = {1.0, 1.0, 2.0, 2.0}>
        {
            q1 = QuantizeLinear(x, s, z)
            d1 = DequantizeLinear(q1, s, z)
            r = Resize<mode = "nearest">(d1, , scales)
            q2 = QuantizeLinear(r, s, z)
            y = DequantizeLinear(q2, s, z)
            q3 = QuantizeLinear(x, s, z)
            w = DequantizeLinear(q3, s, z)
        }
        """
    )
    out = qt.convert_resize_fs_to_pof2s(model)
    init = _inits(out)
    assert init["s"] == np.float32(0.037) and init["z"] == 3  # still used by q3
    assert init["s_pof2"] == 2.0**-4 and init["z_pof2"] == 0


def _u16_model():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,4] x) => (float[2,3] y)
        <float s = {0.0001}, uint16 z = {32768}, float sw = {0.0001}, uint16 zw = {32768},
         float sb = {0.00001}, int32 zb = {0}, int32[3] bq = {5, -7, 100}>
        {
            xq = QuantizeLinear(x, s, z)
            xd = DequantizeLinear(xq, s, z)
            wd = DequantizeLinear(wq, sw, zw)
            m = MatMul(xd, wd)
            bd = DequantizeLinear(bq, sb, zb)
            a = Add(m, bd)
            aq = QuantizeLinear(a, s, z)
            y = DequantizeLinear(aq, s, z)
        }
        """
    )
    k = np.array([-100, 60, 0, 1, -50, 0, 0, 90, -3, 10, 70, -20])
    wq = (32768 + 257 * k).astype(np.uint16).reshape(4, 3)
    model.graph.initializer.append(numpy_helper.from_array(wq, "wq"))
    return model


def test_u16u8_to_u8u8_converts_activations_and_constants():
    model = _u16_model()
    out = qt.convert_u16u8_to_u8u8(model)
    init = _inits(out)
    np.testing.assert_allclose(init["s"], np.float32(0.0001) * 65535 / 255)
    assert init["z"].dtype == np.uint8 and init["z"] == 128
    assert init["wq"].dtype == np.uint8 and init["zw"].dtype == np.uint8
    # 257-step weights map exactly onto uint8 codes
    np.testing.assert_array_equal(
        init["wq"].reshape(-1).astype(int) - 128,
        [-100, 60, 0, 1, -50, 0, 0, 90, -3, 10, 70, -20],
    )
    np.testing.assert_allclose(init["sw"], np.float32(0.0001) * 65535 / 255)
    # bias scale is reset to x_scale * w_scale only for Conv / Gemm: MatMul+Add
    # keeps its int32 bias DQ untouched
    assert init["bq"].dtype == np.int32
    x = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
    _ort_close(_run(out, {"x": x})[0], _run(model, {"x": x})[0], atol=0.5)


def test_u16u8_to_u8u8_resets_conv_bias_scale():
    model = _conv_qdq(bias_dtype="int32")
    out = qt.convert_u16u8_to_u8u8(model)  # int8 -> not touched by the u16 pass
    new = np.float32(0.02) * np.float32(0.01)
    np.testing.assert_allclose(_inits(out)["bs"], new)  # bias scale := sx * sw
    assert _inits(out)["wq"].dtype == np.int8


def test_u16s8_to_s16s8_shifts_activation_zero_points_only():
    model = _u16_model()
    out = qt.convert_u16s8_to_s16s8(model)
    init = _inits(out)
    assert init["int16_zp0"].dtype == np.int16 and init["int16_zp0"] == 0
    assert init["zw"].dtype == np.uint16 and init["zw"] == 32768  # weight DQ
    assert "z" not in init  # fully replaced
    x = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
    _ort_close(_run(out, {"x": x})[0], _run(model, {"x": x})[0], atol=1e-6)


def test_u16s8_to_s16s8_is_exact_for_any_zero_point():
    model = _u16_model()
    for t in model.graph.initializer:
        if t.name == "z":
            t.CopyFrom(numpy_helper.from_array(np.array(32767, np.uint16), "z"))
    out = qt.convert_u16s8_to_s16s8(model)
    assert _inits(out)["int16_zp-1"] == -1
    x = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
    _ort_close(_run(out, {"x": x})[0], _run(model, {"x": x})[0], atol=1e-6)


def test_fix_shapes_makes_dynamic_model_static():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[N,3] x) => (float[N,2] y)
        <float[3,2] w = {1,2,3,4,5,6}>
        {
            a = Relu(x)
            y = MatMul(a, w)
        }
        """
    )
    out = qt.fix_shapes(model, "x:[4,3]")
    dims = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(out.graph.input) + list(out.graph.value_info)
    }
    assert dims["x"] == [4, 3] and dims["a"] == [4, 3]
    assert out.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 4
    assert qt.parse_input_and_output_shapes("a:[1,2]; b:[3]") == {
        "a": [1, 2],
        "b": [3],
    }
    with pytest.raises(ValueError):
        qt.parse_input_and_output_shapes("a=[1]")
    # dict form, and a name that is not a graph input is ignored
    out2 = qt.fix_input_and_output_shapes(model, {"x": [2, 3], "nope": [1]})
    assert out2.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 2


def test_fix_shapes_without_spec_fills_in_value_info_from_a_run():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,3] x) => (float[2,3] y)
        {
            a = Relu(x)
            b = NonZero(a)
            y = Neg(a)
        }
        """
    )
    out = qt.fix_shapes(model)
    names = {v.name for v in out.graph.value_info}
    assert "a" in names
    assert "b" in names  # data dependent: filled from the random-input run
    shape_b = [
        d.dim_value
        for v in out.graph.value_info
        if v.name == "b"
        for d in v.type.tensor_type.shape.dim
    ]
    assert shape_b[0] == 2


def test_a16w8_a8w8_nodes_split_by_activation_width():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x) => (float[1,4,6,6] y, float[1,4,6,6] y2)
        <float s = {0.02}, int16 z16 = {0}, int8 z8 = {0}, uint8 zu = {128},
         float ws = {0.01}, int8 wz = {0}>
        {
            q16 = QuantizeLinear(x, s, z16)
            d16 = DequantizeLinear(q16, s, z16)
            q8 = QuantizeLinear(x, s, z8)
            d8 = DequantizeLinear(q8, s, z8)
            qu = QuantizeLinear(x, s, zu)
            du = DequantizeLinear(qu, s, zu)
            wd = DequantizeLinear(wq, ws, wz)
            y = Conv(d16, wd)
            y2 = Conv(d8, wd)
            y3 = Conv(du, wd)
        }
        """
    )
    model.graph.initializer.append(
        numpy_helper.from_array(np.ones((4, 3, 3, 3), np.int8), "wq")
    )
    convs = [n for n in model.graph.node if n.op_type == "Conv"]
    for n, nm in zip(convs, ("c16", "c8", "cu")):
        n.name = nm
    assert qt.a16w8_a8w8_nodes(model) == (["c8"], ["c16"])


def _bf16_qdq_model():
    return parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21, "com.amd.quark": 1]>
        g (float[2,4] x) => (float[2,4] y)
        <float s = {0.5}, float one = {1.0}, bfloat16 zb = {0}, int8 z8 = {0}>
        {
            a = com.amd.quark.ExtendedQuantizeLinear(x, s, zb)
            b = com.amd.quark.ExtendedDequantizeLinear(a, s, zb)
            c = com.amd.quark.ExtendedQuantizeLinear(b, one, zb)
            d = com.amd.quark.ExtendedDequantizeLinear(c, one, zb)
            e = com.amd.quark.ExtendedQuantizeLinear(d, s, z8)
            y = com.amd.quark.ExtendedDequantizeLinear(e, s, z8)
        }
        """
    )


def test_replace_bfloat16_qdq_cast_uses_casts_and_scale_muls():
    model = _bf16_qdq_model()
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}"
    out = qt.replace_bfloat16_qdq_cast(model)
    assert [n.op_type for n in out.graph.node] == [
        "Mul",
        "Cast",  # a = Cast(x * 2)
        "Cast",
        "Mul",  # b = Cast(a) * 0.5
        "Cast",  # c (scale 1: no Mul)
        "Cast",
        "ExtendedQuantizeLinear",  # int8 zero point stays custom
        "ExtendedDequantizeLinear",
    ]
    init = _inits(out)
    np.testing.assert_allclose(init["n0_scale"], 2.0)
    np.testing.assert_allclose(init["n1_scale"], 0.5)
    # run the plain-ONNX prefix: round trip through bfloat16 is x rounded to bf16
    pre = onnx.ModelProto()
    pre.CopyFrom(out)
    del pre.graph.node[6:]
    pre.graph.output[0].name = "d"
    x = np.array([[1.0, 1.0078125, 3.14159, -1000.1]] * 2, np.float32)
    got = _run(pre, {"x": x})[0]
    bits = x.view(np.uint32).astype(np.uint64)
    r = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint32) << 16
    np.testing.assert_array_equal(got, r.view(np.float32))


def test_insert_clip_bfloat16_qdq_shares_the_clip_per_tensor():
    model = _bf16_qdq_model()
    out = qt.insert_clip_bfloat16_qdq(model)
    assert _ops(out).count("Clip") == 2  # inputs of a and c; e has an int8 zp
    prod = {o: n for n in out.graph.node for o in n.output}
    for q in out.graph.node:
        if q.op_type == "ExtendedQuantizeLinear" and q.output[0] in ("a", "c"):
            assert prod[q.input[0]].op_type == "Clip"
    init = _inits(out)
    np.testing.assert_allclose(init["x_clip_max"], 3.38953139e38)
    np.testing.assert_allclose(init["x_clip_min"], -3.38953139e38)
    onnx.checker.check_model(out, full_check=False)


def _cast_model(names=("a", "a_c1", "a_c2")):
    a, c1, c2 = names
    return parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,4] x) => (float[2,4] y)
        <float[4] w = {{0.1234567, -2.7182818, 3.14159265, 1000.123}}>
        {{
            {a} = Relu(x)
            {c1} = Cast<to = 16>({a})
            {c2} = Cast<to = 1>({c1})
            wb = Cast<to = 16>(w)
            wf = Cast<to = 1>(wb)
            m = Add({c2}, wf)
            n = Mul(m, {c2})
            o1 = Cast<to = 16>(n)
            y = Cast<to = 1>(o1)
        }}
        """
    )


def test_remove_bf16_cast_handles_all_three_patterns():
    model = _cast_model()
    out = qt.remove_bf16_cast(model)
    assert _ops(out) == ["Relu", "Add", "Mul"]
    (t,) = out.graph.initializer
    assert t.name == "w_bf16" and t.data_type == onnx.TensorProto.FLOAT
    w = numpy_helper.to_array(t)
    bits = np.array([0.1234567, -2.7182818, 3.14159265, 1000.123], np.float32)
    bits = bits.view(np.uint32).astype(np.uint64)
    r = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint32) << 16
    np.testing.assert_array_equal(w, r.view(np.float32))
    assert out.graph.output[0].name == "y"
    # unlike Quark, no name-containment condition for rewiring
    out2 = qt.remove_bf16_cast(_cast_model(("p", "q1", "r2")))
    assert _ops(out2) == ["Relu", "Add", "Mul"]
    x = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
    ref = np.maximum(x, 0)
    np.testing.assert_allclose(_run(out2, {"x": x})[0], (ref + w) * ref, rtol=1e-6)


def test_remove_bf16_cast_keeps_casts_with_other_uses():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[4] x) => (float[4] y, float[4] z)
        {
            a = Relu(x)
            b = Cast<to = 16>(a)
            c = Cast<to = 1>(b)
            y = Neg(c)
            z = Abs(a)
        }
        """
    )
    # `a` fans out, so removing the round trip would change `y` but not `z`:
    # the round trip is real (it rounds) and must stay.
    assert _ops(qt.remove_bf16_cast(model)) == _ops(model)


def _between_model():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x, float[1,4,6,6] u) => (float[1,4,6,6] y)
        <float s = {0.1}, int8 z = {0}>
        {
            c = Conv(x, wf)
            cq = QuantizeLinear(c, s, z)
            cd = DequantizeLinear(cq, s, z)
            r = Relu(cd)
            rq = QuantizeLinear(r, s, z)
            rd = DequantizeLinear(rq, s, z)
            mu = Mul(rd, u)
            mq = QuantizeLinear(mu, s, z)
            md = DequantizeLinear(mq, s, z)
            y = Add(md, u)
        }
        """
    )
    model.graph.initializer.append(
        numpy_helper.from_array(np.full((4, 3, 3, 3), 0.5, np.float32), "wf")
    )
    return model


def test_remove_qdq_between_ops_pairs():
    model = _between_model()
    out = qt.remove_qdq_between_ops(model, [("Conv", "Relu")])
    assert _ops(out) == [
        "Conv",
        "Relu",
        "QuantizeLinear",
        "DequantizeLinear",
        "Mul",
        "QuantizeLinear",
        "DequantizeLinear",
        "Add",
    ]
    assert _ops(
        qt.remove_qdq_between_ops(model, [("Relu", "Mul"), ("Mul", "Add")])
    ) == [
        "Conv",
        "QuantizeLinear",
        "DequantizeLinear",
        "Relu",
        "Mul",
        "Add",
    ]
    assert _ops(qt.remove_qdq_mul_add(model)) == _ops(
        qt.remove_qdq_between_ops(model, [("Mul", "Add")])
    )
    assert _ops(qt.remove_qdq_between_ops(model, [("Add", "Conv")])) == _ops(model)
    assert len(model.graph.node) == 10  # input untouched


def test_remove_qdq_between_ops_skips_shared_or_output_tensors():
    model = _between_model()
    # the Relu's Q output is read by a second node: removing it would break that
    model.graph.node.append(onnx.helper.make_node("Neg", ["rq"], ["extra"]))
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("extra", onnx.TensorProto.INT8, [1, 4, 6, 6])
    )
    out = qt.remove_qdq_between_ops(model, [("Relu", "Mul")])
    assert _ops(out) == _ops(model)
    # the DQ is a graph output: keep it
    model2 = _between_model()
    model2.graph.output.append(
        onnx.helper.make_tensor_value_info("md", onnx.TensorProto.FLOAT, [1, 4, 6, 6])
    )
    assert "DequantizeLinear" in _ops(
        qt.remove_qdq_between_ops(model2, [("Mul", "Add")])
    )


def test_onnxtxt_roundtrip():
    model = _qop_model()
    text = qt.convert_onnx_to_onnxtxt(model)
    assert isinstance(text, str) and "QuantizeLinear" in text
    assert qt.convert_onnxtxt_to_onnx(text) == model
    assert qt.convert_onnxtxt_to_onnx(text.encode()) == model


def test_copy_shared_nodes_and_check_shared_initializers():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,4] x) => (float[2,4] y)
        <float s = {0.1}, int8 z = {0}, int8[4,4] wq = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}>
        {
            w = DequantizeLinear(wq, s, z)
            a = MatMul(x, w)
            y = MatMul(a, w)
        }
        """
    )
    assert qt.check_shared_initializers(model) is False  # shared DQ output only
    out = qt.copy_shared_nodes(model)
    assert [n.name for n in out.graph.node] == [
        "DequantizeLinear_1",
        "DequantizeLinear_1_1",
        "MatMul_1",
        "MatMul_2",
    ] or sorted(n.name for n in out.graph.node) == [
        "DequantizeLinear_1",
        "DequantizeLinear_1_1",
        "MatMul_1",
        "MatMul_2",
    ]
    assert _ops(out).count("DequantizeLinear") == 2
    shared = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,2] x) => (float[2,2] y)
        <float[2,2] w = {1,2,3,4}>
        {
            a = MatMul(x, w)
            y = MatMul(a, w)
        }
        """
    )
    assert qt.check_shared_initializers(shared) is True
    out2 = qt.copy_shared_nodes(shared)
    assert not qt.check_shared_initializers(out2)
    assert {t.name for t in out2.graph.initializer} == {"w", "w_1"}
    x = np.eye(2, dtype=np.float32)
    np.testing.assert_array_equal(_run(out2, {"x": x})[0], _run(shared, {"x": x})[0])


def test_clean_initializer_in_input_and_external_data(tmp_path):
    model = parser.parse_model(
        """
        <ir_version: 3, opset_import: ["": 9]>
        g (float[2] x, float[2] w) => (float[2] y)
        <float[2] w = {1.0, 2.0}>
        { y = Add(x, w) }
        """
    )
    out = qt.clean_initializer_in_input(model)
    assert [i.name for i in out.graph.input] == ["x"] and out.ir_version == 4
    assert len(model.graph.input) == 2  # not mutated

    big = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,64] x) => (float[2,64] y)
        { y = MatMul(x, w) }
        """
    )
    w = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    big.graph.initializer.append(numpy_helper.from_array(w, "w"))
    path = str(tmp_path / "m.onnx")
    qt.save_onnx_model_with_external_data(big, path, save_as_external_data=True)
    assert (tmp_path / "m.onnx.data").exists()
    assert not big.graph.initializer[0].raw_data  # tensor now points at the file
    loaded = onnx.load(path)  # also loads the external data
    np.testing.assert_array_equal(numpy_helper.to_array(loaded.graph.initializer[0]), w)
    inline = str(tmp_path / "n.onnx")
    qt.save_onnx_model_with_external_data(loaded, inline, save_as_external_data=False)
    assert not (tmp_path / "n.onnx.data").exists()


@pytest.mark.parametrize(
    "build,call",
    [
        (_conv_qdq, qt.convert_a8w8_npu_to_a8w8_cpu),
        (_u16_model, qt.convert_u16u8_to_u8u8),
        (_u16_model, qt.convert_u16s8_to_s16s8),
        (_qop_model, qt.convert_qdq_to_qop),
        (_between_model, qt.remove_qdq_mul_add),
        (_cast_model, qt.remove_bf16_cast),
        (_bf16_qdq_model, qt.insert_clip_bfloat16_qdq),
        (_bf16_qdq_model, qt.replace_bfloat16_qdq_cast),
        (_bf16_qdq_model, qt.convert_customqdq_to_qdq),
        (_bf16_qdq_model, qt.convert_custom_ops),
        (_qop_model, qt.copy_shared_nodes),
        (_qop_model, qt.convert_nchw_to_nhwc),
        (_qop_model, qt.fix_shapes),
    ],
)
def test_extra_tools_do_not_modify_their_input(build, call):
    model = build()
    before = model.SerializeToString()
    call(model)
    assert model.SerializeToString() == before
