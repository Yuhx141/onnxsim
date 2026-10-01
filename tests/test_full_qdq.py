import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim.full_qdq import (
    quantize_full_qdq,
    quantized_io,
    sampling_coordinate_tensors,
)


def _model(body, initializer=(), opset=17):
    model = parser.parse_model(f'<ir_version: 8, opset_import: ["": {opset}]> {body}')
    model.graph.initializer.extend(initializer)
    return model


def _conv_block():
    rng = np.random.default_rng(0)
    return _model(
        """g (float[1,4,8,8] x) => (float[1,4,4,4] y) {
            c = Conv<pads=[1,1,1,1]>(x, w, b)
            r = Relu(c)
            c2 = Conv<pads=[1,1,1,1]>(r, w, b)
            a = Add(c2, r)
            p = MaxPool<kernel_shape=[2,2], strides=[2,2]>(a)
            y = Mul(p, k)
        }""",
        [
            numpy_helper.from_array(
                rng.standard_normal((4, 4, 3, 3)).astype(np.float32), "w"
            ),
            numpy_helper.from_array(rng.standard_normal(4).astype(np.float32), "b"),
            numpy_helper.from_array(np.array(0.5, np.float32), "k"),
        ],
    )


def _data(n=4, shape=(1, 4, 8, 8)):
    rng = np.random.default_rng(1)
    return [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


def _run(model, feed):
    # No graph optimizations: ORT would otherwise fuse DQ -> Gemm/MatMul -> Q into
    # u8s8 integer kernels, which saturate on x86 CPUs without VNNI (CI runners)
    # and make the result depend on the host CPU rather than on the quantization.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    ).run(None, feed)[0]


def _cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def _producers(model):
    return {o: n for n in model.graph.node for o in n.output}


def test_every_compute_node_is_a_qdq_unit():
    m = _conv_block()
    q = quantize_full_qdq(m, _data())
    onnx.checker.check_model(q)
    prod = _producers(q)
    consumers = {}
    for n in q.graph.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    compute = [
        n
        for n in q.graph.node
        if n.op_type not in ("QuantizeLinear", "DequantizeLinear")
    ]
    # the Relu was folded into the first Conv's output quantization
    assert [n.op_type for n in compute] == ["Conv", "Conv", "Add", "MaxPool", "Mul"]
    for n in compute:
        assert all(prod[x].op_type == "DequantizeLinear" for x in n.input if x), (
            n.op_type
        )
        assert [c.op_type for c in consumers[n.output[0]]] == ["QuantizeLinear"], (
            n.op_type
        )
    # int8 per-channel weights, int32 bias
    conv = compute[0]
    w_dq, b_dq = prod[conv.input[1]], prod[conv.input[2]]
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    assert inits[w_dq.input[0]].dtype == np.int8 and inits[w_dq.input[1]].shape == (4,)
    assert inits[b_dq.input[0]].dtype == np.int32
    # the folded Relu: the first Conv's output Q has zero point 0 (clamps at 0)
    (q_node,) = consumers[conv.output[0]]
    assert int(inits[q_node.input[2]]) == 0
    x = _data(1)[0]
    assert _cos(_run(m, x), _run(q, x)) > 0.999


def test_data_movement_ops_share_input_qparams():
    q = quantize_full_qdq(_conv_block(), _data())
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    prod = _producers(q)
    pool = next(n for n in q.graph.node if n.op_type == "MaxPool")
    q_in = prod[prod[pool.input[0]].input[0]]  # DQ <- Q
    q_out = next(
        n
        for n in q.graph.node
        if n.op_type == "QuantizeLinear" and n.input[0] == pool.output[0]
    )
    for i in (1, 2):
        assert inits[q_in.input[i]] == inits[q_out.input[i]]


def test_op_types_and_exclusions_keep_nodes_float():
    rng = np.random.default_rng(2)
    m = _model(
        """g (float[2,8] x) => (float[2,8] y) {
            h = Gemm<transB=1>(x, w, b)
            n = LayerNormalization<axis=-1>(h, s, t)
            y = MatMul(n, v)
        }""",
        [
            numpy_helper.from_array(
                rng.standard_normal((8, 8)).astype(np.float32), "w"
            ),
            numpy_helper.from_array(rng.standard_normal(8).astype(np.float32), "b"),
            numpy_helper.from_array(np.ones(8, np.float32), "s"),
            numpy_helper.from_array(np.zeros(8, np.float32), "t"),
            numpy_helper.from_array(
                rng.standard_normal((8, 8)).astype(np.float32), "v"
            ),
        ],
    )
    data = _data(4, (2, 8))
    q = quantize_full_qdq(m, data, op_types=["Gemm", "MatMul"])
    prod = _producers(q)
    ln = next(n for n in q.graph.node if n.op_type == "LayerNormalization")
    # LayerNorm reads the Gemm's dequantized output but keeps float scale/bias
    assert prod[ln.input[0]].op_type == "DequantizeLinear"
    assert list(ln.input[1:]) == ["s", "t"]
    gemm = next(n for n in q.graph.node if n.op_type == "Gemm")
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    assert inits[prod[gemm.input[1]].input[1]].shape == (8,)  # transB=1 -> per row
    q2 = quantize_full_qdq(
        m, data, exclude_nodes=[prod[ln.input[0]].input[0][: -len("/q")]]
    )
    gemm2 = next(n for n in q2.graph.node if n.op_type == "Gemm")
    assert all(
        _producers(q2).get(x) is None for x in gemm2.input
    )  # excluded by output name
    assert _cos(_run(m, data[0]), _run(q, data[0])) > 0.99


def test_uint16_activations_use_ms_domain_below_opset_21():
    q = quantize_full_qdq(_conv_block(), _data(), activation_dtype="uint16")
    qs = [n for n in q.graph.node if n.op_type == "QuantizeLinear"]
    assert qs and all(n.domain == "com.microsoft" for n in qs)
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    assert inits[qs[0].input[2]].dtype == np.uint16
    x = _data(1)[0]
    assert _cos(_run(_conv_block(), x), _run(q, x)) > 0.99999


@pytest.mark.parametrize(
    "dt, np_dt, ms", [("int8", np.int8, False), ("int16", np.int16, True)]
)
def test_signed_activations_are_symmetric(dt, np_dt, ms):
    q = quantize_full_qdq(_conv_block(), _data(), activation_dtype=dt)
    qs = [n for n in q.graph.node if n.op_type == "QuantizeLinear"]
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    assert qs and all((n.domain == "com.microsoft") == ms for n in qs)
    for n in qs:
        zp = inits[n.input[2]]
        assert zp.dtype == np_dt and not zp.any()
    assert any(n.op_type == "Relu" for n in q.graph.node)  # not folded
    x = _data(1)[0]
    assert _cos(_run(_conv_block(), x), _run(q, x)) > 0.999


def test_power_of_two_scales_with_centred_unsigned_zero_point():
    q = quantize_full_qdq(
        _conv_block(),
        _data(),
        symmetric_activations=True,
        power_of_two=True,
        per_channel=False,
    )
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    scales = [v for k, v in inits.items() if k.endswith("/scale") and v.ndim == 0]
    assert scales and all(np.log2(v) == round(float(np.log2(v))) for v in scales)
    zps = {
        int(v) for k, v in inits.items() if k.endswith("/zp") and v.dtype == np.uint8
    }
    assert zps == {128}


def test_precomputed_ranges_skip_calibration():
    m = _conv_block()
    with pytest.raises(ValueError):
        quantize_full_qdq(m)
    names = ["x", "c", "r", "c2", "a", "p", "y"]
    ranges = {n: (-4.0, 4.0) for n in names}
    q = quantize_full_qdq(m, ranges=ranges)
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    qx = next(
        n for n in q.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == "x"
    )
    assert float(inits[qx.input[1]]) == pytest.approx(8.0 / 255)


def test_partial_precomputed_ranges_win_over_calibration():
    # calibrate() reports every activation it sees; a range the caller pins (here a fixed [-8, 8]
    # input range, wider than the data's) must not be overwritten by the calibrated one
    m = _conv_block()
    q = quantize_full_qdq(m, _data(), ranges={"x": (-8.0, 8.0)})
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    qx = next(
        n for n in q.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == "x"
    )
    assert float(inits[qx.input[1]]) == pytest.approx(16.0 / 255)
    assert int(inits[qx.input[2]]) == 128


def test_quantized_io_is_lossless_and_nhwc():
    m = _conv_block()
    q = quantize_full_qdq(m, _data())
    qi, info = quantized_io(q, nhwc_inputs=["x"])
    onnx.checker.check_model(qi)
    assert info["x"]["layout"] == "nhwc" and info["y"]["dtype"] == "uint8"
    assert qi.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.UINT8
    assert [d.dim_value for d in qi.graph.input[0].type.tensor_type.shape.dim] == [
        1,
        8,
        8,
        4,
    ]
    x = _data(1)[0]["x"]
    s, zp = info["x"]["scale"], info["x"]["zero_point"]
    xq = (
        np.clip(np.round(x / s) + zp, 0, 255)
        .astype(np.uint8)
        .transpose(0, 2, 3, 1)
        .copy()
    )
    yq = _run(qi, {"x": xq})
    y = (yq.astype(np.float32) - info["y"]["zero_point"]) * np.float32(
        info["y"]["scale"]
    )
    np.testing.assert_allclose(y, _run(q, {"x": x}), atol=1e-6)


def test_excluded_node_between_quantized_nodes_stays_float():
    m = _conv_block()
    q = quantize_full_qdq(m, _data(), exclude_nodes=["a"])  # the Add, by output name
    add = next(n for n in q.graph.node if n.op_type == "Add")
    # with a Q on its output the Add would form a QDQ unit (and run quantized)
    assert not [
        n
        for n in q.graph.node
        if add.output[0] in n.input and n.op_type == "QuantizeLinear"
    ]
    x = _data(1)[0]
    assert _cos(_run(m, x), _run(q, x)) > 0.999


def test_nodes_outside_op_types_between_quantized_nodes_run_quantized():
    rng = np.random.default_rng(3)
    m = _model(
        """g (float[2,8] x) => (float[2,8] y) {
            h = Gemm<transB=1>(x, w, b)
            r = Relu(h)
            y = Gemm<transB=1>(r, w2, b)
        }""",
        [
            numpy_helper.from_array(rng.standard_normal((8, 8)).astype(np.float32), n)
            for n in ("w", "w2")
        ]
        + [numpy_helper.from_array(rng.standard_normal(8).astype(np.float32), "b")],
    )
    q = quantize_full_qdq(m, _data(4, (2, 8)), op_types=["Gemm"], fold_relu=False)
    prod = _producers(q)
    for g in (n for n in q.graph.node if n.op_type == "Gemm"):
        # activation, int8 weight and int32 bias all dequantized: both Gemms are QDQ units
        assert all(prod[x].op_type == "DequantizeLinear" for x in g.input), g.input


def test_uint16_sampling_coordinates_in_a_uint8_graph():
    rng = np.random.default_rng(4)
    m = _model(
        """g (float[1,4,6,6] v, float[1,5,5,2] ref, float[1,50,8] q) => (float[1,4,5,5] y) {
            o = MatMul(q, w)
            o2 = Reshape(o, shape)
            grid = Add(ref, o2)
            y = GridSample(v, grid)
        }""",
        [
            numpy_helper.from_array(
                (0.01 * rng.standard_normal((8, 1))).astype(np.float32), "w"
            ),
            numpy_helper.from_array(np.array([1, 5, 5, 2], np.int64), "shape"),
        ],
    )
    coords = sampling_coordinate_tensors(m)
    assert sorted(coords) == ["grid", "o", "o2", "ref"]

    def batch():
        return {
            "v": rng.standard_normal((1, 4, 6, 6)).astype(np.float32),
            "ref": rng.uniform(-1, 1, (1, 5, 5, 2)).astype(np.float32),
            "q": rng.standard_normal((1, 50, 8)).astype(np.float32),
        }

    data = [batch() for _ in range(4)]
    q = quantize_full_qdq(m, data, tensor_dtypes={t: "uint16" for t in coords})
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    zp = {
        n.input[0]: inits[n.input[2]].dtype
        for n in q.graph.node
        if n.op_type == "QuantizeLinear"
    }
    assert (
        zp["grid/f"] == np.uint16 and zp["o/f"] == np.uint16 and zp["y/f"] == np.uint8
    )
    assert zp["o2/f"] == np.uint16  # the Reshape passes its input's dtype on
    # the MatMul computes in its output's uint16: its uint8 input is converted for it,
    # while GridSample keeps a uint8 value input next to its uint16 grid
    mm = next(n for n in q.graph.node if n.op_type == "MatMul")
    assert mm.input[0] == "q/as_uint16" and zp["q/dq"] == np.uint16
    gs = next(n for n in q.graph.node if n.op_type == "GridSample")
    assert list(gs.input) == ["v/dq", "grid"]
    x = data[0]
    q8 = quantize_full_qdq(m, data)
    assert _cos(_run(m, x), _run(q, x)) > _cos(_run(m, x), _run(q8, x))
