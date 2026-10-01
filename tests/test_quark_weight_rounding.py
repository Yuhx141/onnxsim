"""Tests for onnxsim.quark_weight_rounding (AdaRound / GPTQ on int8 QDQ models)."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_weight_rounding as wr
from onnxsim.full_qdq import quantize_full_qdq


def _run(model, feed):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feed)[0]


def _inits(model):
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def _add_weights(model, shapes, seed):
    # Random weights are attached programmatically (too large for text literals).
    rng = np.random.default_rng(seed)
    model.graph.initializer.extend(
        numpy_helper.from_array(rng.standard_normal(s).astype(np.float32), n)
        for n, s in shapes
    )
    return model


def _batches(shape, n, seed=1):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


@pytest.fixture(scope="module")
def mlp():
    model = _add_weights(
        parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 21]>
            agraph (float[N,32] x) => (float[N,16] y)
            {
                h = MatMul(x, w1)
                t = Tanh(h)
                y = MatMul(t, w2)
            }
            """
        ),
        [("w1", (32, 32)), ("w2", (32, 16))],
        seed=0,
    )
    data = _batches((16, 32), 16)
    return model, quantize_full_qdq(model, calibration_data=data), data


@pytest.fixture(scope="module")
def conv():
    model = _add_weights(
        parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 17]>
            agraph (float[2,3,8,8] x) => (float[2,4,8,8] y)
            {
                h = Conv<pads=[1,1,1,1]>(x, w1, b1)
                r = Relu(h)
                y = Conv<pads=[1,1,1,1]>(r, w2, b2)
            }
            """
        ),
        [("w1", (4, 3, 3, 3)), ("b1", (4,)), ("w2", (4, 4, 3, 3)), ("b2", (4,))],
        seed=2,
    )
    data = _batches((2, 3, 8, 8), 8)
    return model, quantize_full_qdq(model, calibration_data=data), data


ALGOS = {
    "adaround": lambda f, q, d: wr.adaround_int8(f, q, d, num_iterations=200),
    "gptq": lambda f, q, d: wr.gptq_int8(f, q, d),
}


# -- im2col ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel, strides, pads, dilations",
    [
        ((3, 3), (1, 1), (1, 1, 1, 1), (1, 1)),
        ((3, 2), (2, 1), (0, 1, 2, 0), (1, 1)),
        ((2, 2), (1, 2), (1, 0, 0, 1), (2, 2)),
        ((1, 1), (1, 1), (0, 0, 0, 0), (1, 1)),
    ],
)
def test_im2col_matches_onnxruntime_conv(kernel, strides, pads, dilations):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 9, 7)).astype(np.float32)
    w = rng.standard_normal((5, 3, *kernel)).astype(np.float32)
    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[2,3,9,7] x) => (float y)
        {{
            y = Conv<strides={list(strides)}, pads={list(pads)},
                     dilations={list(dilations)}>(x, w)
        }}
        """
    )
    model.graph.initializer.append(numpy_helper.from_array(w, "w"))
    expected = _run(model, {"x": x})
    cols = wr._im2col(x.astype(np.float64), kernel, strides, pads, dilations)
    n, o, oh, ow = expected.shape
    got = (cols @ w.reshape(5, -1).T).reshape(n, oh, ow, o).transpose(0, 3, 1, 2)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


# -- behaviour on QDQ models --------------------------------------------------------


@pytest.mark.parametrize("algo", sorted(ALGOS))
@pytest.mark.parametrize("fixture", ["mlp", "conv"])
def test_only_weight_codes_change_and_layer_error_does_not_grow(request, algo, fixture):
    float_model, quant, data = request.getfixturevalue(fixture)
    out, reports = ALGOS[algo](float_model, quant, data)
    before, after = _inits(quant), _inits(out)
    assert before.keys() == after.keys()
    assert [n.SerializeToString() for n in out.graph.node] == [
        n.SerializeToString() for n in quant.graph.node
    ]
    changed = [k for k in before if not np.array_equal(before[k], after[k])]
    assert changed and all(k.endswith("/int8") for k in changed)  # weight codes only
    for k in changed:
        assert after[k].dtype == np.int8 and np.abs(after[k]).max() <= 127
    assert len(reports) == 2
    for r in reports:
        assert r.accepted and r.error_after <= r.error_before
    assert any(r.error_after < r.error_before for r in reports)
    assert any(r.changed_fraction > 0 for r in reports)
    onnx.checker.check_model(out)


def test_conv_layers_are_found_through_im2col(conv):
    float_model, quant, data = conv
    _, reports = wr.adaround_int8(float_model, quant, data, num_iterations=50)
    assert [r.op for r in reports] == ["Conv", "Conv"]
    assert [r.shape for r in reports] == [(4, 3, 3, 3), (4, 4, 3, 3)]


def test_a_refiner_that_makes_things_worse_is_rejected(mlp):
    float_model, quant, data = mlp
    out, reports = wr._refine(
        float_model,
        quant,
        data,
        lambda ly, w_nk, s, x: np.zeros_like(w_nk),  # all-zero weights: terrible
        wr._TARGET_OPS,
        max_rows=512,
        seed=0,
        providers=None,
    )
    assert reports and not any(r.accepted for r in reports)
    assert all(r.error_after == r.error_before for r in reports)
    assert out.SerializeToString() == quant.SerializeToString()


def test_gemm_transb_layout_is_preserved():
    model = _add_weights(
        parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 17]>
            agraph (float[N,24] x) => (float[N,10] y)
            {
                y = Gemm<transB=1>(x, w, b)
            }
            """
        ),
        [("w", (10, 24)), ("b", (10,))],
        seed=3,
    )
    data = _batches((16, 24), 12)
    quant = quantize_full_qdq(model, calibration_data=data)
    for fn in ALGOS.values():
        out, reports = fn(model, quant, data)
        assert [r.op for r in reports] == ["Gemm"] and reports[0].shape == (10, 24)
        assert reports[0].error_after <= reports[0].error_before
        codes = next(v for k, v in _inits(out).items() if k.endswith("/int8"))
        assert codes.shape == (10, 24)


def test_unsupported_layers_are_left_alone():
    # grouped Conv
    model = _add_weights(
        parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 17]>
            agraph (float[1,4,8,8] x) => (float[1,4,8,8] y)
            {
                y = Conv<group=2, pads=[1,1,1,1]>(x, w)
            }
            """
        ),
        [("w", (4, 2, 3, 3))],
        seed=4,
    )
    data = _batches((1, 4, 8, 8), 4)
    quant = quantize_full_qdq(model, calibration_data=data)
    out, reports = wr.gptq_int8(model, quant, data)
    assert reports == [] and out.SerializeToString() == quant.SerializeToString()


def test_a_weight_shared_by_two_layers_is_left_alone():
    model = _add_weights(
        parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 21]>
            agraph (float[N,8] x) => (float[N,8] y)
            {
                h = MatMul(x, w)
                t = Tanh(h)
                y = MatMul(t, w)
            }
            """
        ),
        [("w", (8, 8))],
        seed=5,
    )
    data = _batches((16, 8), 8)
    quant = quantize_full_qdq(model, calibration_data=data)
    for fn in ALGOS.values():
        out, reports = fn(model, quant, data)
        assert reports == [] and out.SerializeToString() == quant.SerializeToString()


def test_float_model_given_as_quantized_model_is_a_noop(mlp):
    float_model, _, data = mlp
    out, reports = wr.adaround_int8(float_model, float_model, data, num_iterations=5)
    assert reports == [] and out.SerializeToString() == float_model.SerializeToString()


def test_inputs_are_not_mutated(mlp):
    float_model, quant, data = mlp
    f_before, q_before = float_model.SerializeToString(), quant.SerializeToString()
    wr.adaround_int8(float_model, quant, data, num_iterations=20)
    wr.gptq_int8(float_model, quant, data)
    assert float_model.SerializeToString() == f_before
    assert quant.SerializeToString() == q_before


def test_adaround_is_deterministic_for_a_seed(mlp):
    float_model, quant, data = mlp
    a, _ = wr.adaround_int8(float_model, quant, data, num_iterations=50, seed=3)
    b, _ = wr.adaround_int8(float_model, quant, data, num_iterations=50, seed=3)
    assert a.SerializeToString() == b.SerializeToString()


def test_gptq_act_order_and_block_size_run_and_do_not_regress(mlp):
    float_model, quant, data = mlp
    for kwargs in ({"act_order": True}, {"block_size": 8}):
        _, reports = wr.gptq_int8(float_model, quant, data, **kwargs)
        assert reports and all(r.error_after <= r.error_before for r in reports)


def test_calibration_data_is_required(mlp):
    float_model, quant, _ = mlp
    with pytest.raises(ValueError, match="calibration_data"):
        wr.adaround_int8(float_model, quant, [])


@pytest.mark.parametrize("algo", sorted(ALGOS))
def test_conv_layers_with_different_output_channel_counts(algo):
    # Each layer's layout transform must use its own weight shape.
    model = _add_weights(
        parser.parse_model(
            """
            <ir_version: 10, opset_import: ["": 17]>
            agraph (float[2,3,8,8] x) => (float[2,5,8,8] y)
            {
                h = Conv<pads=[1,1,1,1]>(x, w1)
                r = Relu(h)
                y = Conv<pads=[0,0,0,0]>(r, w2)
            }
            """
        ),
        [("w1", (6, 3, 3, 3)), ("w2", (5, 6, 1, 1))],
        seed=6,
    )
    data = _batches((2, 3, 8, 8), 8)
    quant = quantize_full_qdq(model, calibration_data=data)
    out, reports = ALGOS[algo](model, quant, data)
    assert [r.shape for r in reports] == [(6, 3, 3, 3), (5, 6, 1, 1)]
    # A rejected layer (the algorithm lost to round-to-nearest) keeps its error.
    assert all(r.error_after <= r.error_before for r in reports)
    shapes = sorted(v.shape for k, v in _inits(out).items() if k.endswith("/int8"))
    assert shapes == [(5, 6, 1, 1), (6, 3, 3, 3)]
    # layer 1 is the one with the extra freedom to improve; both algorithms do
    assert reports[0].accepted and reports[0].error_after < reports[0].error_before
