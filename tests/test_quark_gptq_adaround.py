"""Quark-style GPTQ (bits / group_size / per_channel / mse / symmetry) and AdaRound
(drop_ratio, selective_update, lr_adjust) options of onnxsim.quark_weight_rounding
and onnxsim.quark_compat. No AMD Quark needed; the Quark parity side is
tests/test_quark_gptq_parity.py."""

import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc
from onnxsim import quark_weight_rounding as wr
from onnxsim.full_qdq import quantize_full_qdq


def _model(body, shapes, opset=21, seed=0, inputs="float[N,32] x", outputs="float y"):
    model = parser.parse_model(
        f'<ir_version: 10, opset_import: ["": {opset}]> '
        f"g ({inputs}) => ({outputs}) {{ {body} }}"
    )
    # Random weights are attached programmatically (too large for text literals).
    rng = np.random.default_rng(seed)
    model.graph.initializer.extend(
        numpy_helper.from_array(rng.standard_normal(s).astype(np.float32), n)
        for n, s in shapes
    )
    return model


def _batches(shape, n=8, seed=1):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


def _inits(model):
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def _run(model, feed):
    opts = ort.SessionOptions()
    # ONNX Runtime's QLinearMatMul fusion cannot take blocked weight scales
    opts.add_session_config_entry("session.disable_quant_qdq", "1")
    sess = ort.InferenceSession(model.SerializeToString(), opts)
    return sess.run(None, feed)[0]


def _dq_node(model, weight):
    return next(
        n
        for n in model.graph.node
        if n.op_type == "DequantizeLinear" and n.input[0].startswith(weight)
    )


def _dequantized(model, weight):
    """The float weight a weight DequantizeLinear produces, via ONNX Runtime."""
    dq = _dq_node(model, weight)
    graph = onnx.helper.make_graph(
        [dq],
        "dq",
        [],
        [
            onnx.helper.make_tensor_value_info(
                dq.output[0], onnx.TensorProto.FLOAT, None
            )
        ],
        [t for t in model.graph.initializer if t.name in dq.input],
    )
    m = onnx.helper.make_model(graph, opset_imports=model.opset_import)
    m.ir_version = 10
    return ort.InferenceSession(m.SerializeToString()).run(None, {})[0]


@pytest.fixture(scope="module")
def mlp():
    model = _model(
        "h = MatMul(x, w1) t = Tanh(h) y = MatMul(t, w2)",
        [("w1", (32, 32)), ("w2", (32, 16))],
        outputs="float[N,16] y",
    )
    data = _batches((16, 32), 12)
    return model, quantize_full_qdq(model, calibration_data=data), data


# -- the numpy core --------------------------------------------------------------------


def _problem(k=48, n=12, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((k, n)) * rng.uniform(0.2, 2.0, (1, n))
    x = rng.standard_normal((256, k)) * rng.uniform(0.3, 2.0, k)
    return w, 2.0 / len(x) * x.T @ x, x


def _dequant(res, k, group_size=-1):
    gs = k if group_size == -1 else group_size
    s = np.repeat(res.scale, gs, axis=0)[:k]
    z = np.repeat(res.zero, gs, axis=0)[:k]
    return (res.q_int - z) * s


@pytest.mark.parametrize("bits", [8, 4, 3])
def test_symmetric_grid_is_quarks(bits):
    w, h, _ = _problem()
    res = wr.quark_gptq(w, h, bits)
    maxq = 2**bits - 1
    assert res.scale.shape == (1, w.shape[1]) and np.ptp(res.scale) == 0
    np.testing.assert_allclose(res.scale[0, 0], 2 * np.abs(w).max() / maxq)
    assert np.all(res.zero == (maxq + 1) / 2)
    assert res.q_int.min() >= 0 and res.q_int.max() <= maxq


def test_per_channel_scales_follow_each_column():
    w, h, _ = _problem()
    res = wr.quark_gptq(w, h, 8, per_channel=True)
    np.testing.assert_allclose(res.scale[0], 2 * np.abs(w).max(axis=0) / 255)


def test_asymmetric_grid_uses_min_max_and_a_zero_point():
    w, h, _ = _problem()
    res = wr.quark_gptq(w + 0.3, h, 4, sym=False, per_channel=True)
    lo = np.minimum((w + 0.3).min(axis=0), 0)
    hi = np.maximum((w + 0.3).max(axis=0), 0)
    np.testing.assert_allclose(res.scale[0], (hi - lo) / 15)
    np.testing.assert_array_equal(res.zero[0], np.round(-lo / res.scale[0]))


def test_group_size_gives_one_scale_per_group():
    w, h, _ = _problem(k=48)
    res = wr.quark_gptq(w, h, 4, group_size=16, block_size=16, per_channel=True)
    assert res.scale.shape == (3, w.shape[1])
    # block == group here, so every group's scale comes from its own rows of
    # the weights as compensated by the earlier groups, and differs per group
    assert not np.allclose(res.scale[0], res.scale[1])
    uneven = wr.quark_gptq(w[:40], h[:40, :40], 4, group_size=16, block_size=16)
    assert uneven.scale.shape == (3, w.shape[1])  # ceil(40 / 16)


def test_compensate_false_is_round_to_nearest_and_true_is_better():
    # Quark 0.13's update step is a no-op; ours really propagates the error.
    w, h, x = _problem()
    quark = wr.quark_gptq(w, h, 4, block_size=16, compensate=False)
    ours = wr.quark_gptq(w, h, 4, block_size=16)
    rtn = np.clip(np.round(w / quark.scale[0]) + quark.zero[0], 0, 15)
    np.testing.assert_array_equal(quark.q_int, rtn)
    err = lambda r: np.mean((x @ _dequant(r, len(w)) - x @ w) ** 2)  # noqa: E731
    assert err(ours) < err(quark)
    np.testing.assert_array_equal(quark.scale, ours.scale)  # same grid


def test_act_order_visits_high_energy_rows_first_but_keeps_row_order():
    w, h, x = _problem()
    res = wr.quark_gptq(w, h, 4, act_order=True, block_size=16)
    plain = wr.quark_gptq(w, h, 4, block_size=16)
    assert res.q_int.shape == w.shape and np.any(res.q_int != plain.q_int)
    err = lambda r: np.mean((x @ _dequant(r, len(w)) - x @ w) ** 2)  # noqa: E731
    assert err(res) < err(wr.quark_gptq(w, h, 4, compensate=False))


def test_dead_rows_are_zeroed_and_do_not_break_the_factorization():
    w, h, _ = _problem()
    h[7, :] = h[:, 7] = 0.0
    res = wr.quark_gptq(w, h, 8)
    assert np.all(res.q_int[7] == res.zero[0])  # code of 0.0


def test_mse_search_never_widens_the_range():
    w, h, x = _problem()
    plain = wr.quark_gptq(w, h, 3, compensate=False)
    mse = wr.quark_gptq(w, h, 3, mse=True, compensate=False)
    assert np.all(mse.scale <= plain.scale + 1e-12)
    err = lambda r: np.mean((x @ _dequant(r, len(w)) - x @ w) ** 2)  # noqa: E731
    assert err(mse) <= err(plain)


def test_bits_out_of_range_is_refused():
    w, h, _ = _problem()
    for bits in (1, 9):
        with pytest.raises(ValueError, match="bits"):
            wr.quark_gptq(w, h, bits)


# -- GPTQ on QDQ models ----------------------------------------------------------------


def _layer_error(float_model, out, weight, data, name="x"):
    x = np.concatenate([d[name] for d in data])
    w = _inits(float_model)[weight]
    return float(np.mean((x @ _dequantized(out, weight) - x @ w) ** 2))


def test_default_gptq_keeps_the_model_scales(mlp):
    float_model, quant, data = mlp
    out, reports = wr.gptq_int8(float_model, quant, data)
    assert reports and all(r.error_after <= r.error_before for r in reports)
    before, after = _inits(quant), _inits(out)
    assert all(
        np.array_equal(before[k], after[k]) for k in before if not k.endswith("/int8")
    )


@pytest.mark.parametrize("bits", [6, 4])
def test_bits_regrid_writes_a_per_tensor_qdq_weight(mlp, bits):
    float_model, quant, data = mlp
    out, reports = wr.gptq_int8(float_model, quant, data, bits=bits)
    onnx.checker.check_model(out)
    codes = next(
        v
        for k, v in _inits(out).items()
        if k.startswith("w1") and v.ndim == 2 and v.dtype == np.int8
    )
    assert codes.min() >= -(2 ** (bits - 1)) and codes.max() <= 2 ** (bits - 1) - 1
    dq = _dq_node(out, "w1")
    assert (
        numpy_helper.to_array(
            next(t for t in out.graph.initializer if t.name == dq.input[1])
        ).shape
        == ()
    )
    # the report's error is the real error of the dequantized weights
    assert reports[0].error_after == pytest.approx(
        _layer_error(float_model, out, "w1", data), rel=1e-6
    )
    assert reports[0].error_after <= reports[0].error_before


def test_fewer_bits_means_more_error(mlp):
    float_model, quant, data = mlp
    errs = [
        wr.gptq_int8(float_model, quant, data, bits=b)[1][0].error_after
        for b in (8, 6, 4, 3)
    ]
    assert errs == sorted(errs) and errs[-1] > 10 * errs[0]


def test_per_channel_regrid_scales_along_the_output_axis(mlp):
    float_model, quant, data = mlp
    out, _ = wr.gptq_int8(float_model, quant, data, bits=4, per_channel=True)
    dq = _dq_node(out, "w2")
    scale = _inits(out)[dq.input[1]]
    assert (
        scale.shape == (16,) and dict((a.name, a.i) for a in dq.attribute)["axis"] == 1
    )
    per_tensor = wr.gptq_int8(float_model, quant, data, bits=4)[1]
    per_channel = wr.gptq_int8(float_model, quant, data, bits=4, per_channel=True)[1]
    assert per_channel[1].error_after < per_tensor[1].error_after


def test_group_size_writes_a_blocked_dequantize_linear(mlp):
    float_model, quant, data = mlp
    out, reports = wr.gptq_int8(
        float_model, quant, data, bits=4, group_size=8, per_channel=True
    )
    onnx.checker.check_model(out)
    dq = _dq_node(out, "w1")
    attrs = {a.name: a.i for a in dq.attribute}
    assert attrs == {"axis": 0, "block_size": 8}
    assert _inits(out)[dq.input[1]].shape == (4, 32)  # K / 8 groups x N
    assert reports[0].error_after == pytest.approx(
        _layer_error(float_model, out, "w1", data), rel=1e-6
    )
    # finer groups follow the weights better
    coarse = wr.gptq_int8(float_model, quant, data, bits=4, group_size=32)[1][0]
    assert reports[0].error_after < coarse.error_after


def test_grouped_model_runs_and_matches_numpy(mlp):
    float_model, quant, data = mlp
    out, _ = wr.gptq_int8(float_model, quant, data, bits=4, group_size=16)
    w1, w2 = _dequantized(out, "w1"), _dequantized(out, "w2")
    x = data[0]["x"]
    got = _run(out, {"x": x})
    # activations are int8 fake-quantized, so compare loosely against the float
    # network that uses the dequantized weights
    want = np.tanh(x @ w1) @ w2
    assert np.mean((got - want) ** 2) < 0.05 * np.mean(want**2)


def test_group_size_that_does_not_divide_k(mlp):
    float_model, quant, data = mlp
    out, _ = wr.gptq_int8(float_model, quant, data, bits=4, group_size=12)
    dq = _dq_node(out, "w1")
    assert _inits(out)[dq.input[1]].shape == (3, 32)  # ceil(32 / 12)
    w = _dequantized(out, "w1")
    assert w.shape == (32, 32)


def test_asymmetric_weights_are_uint8_with_a_zero_point(mlp):
    float_model, quant, data = mlp
    out, reports = wr.gptq_int8(
        float_model, quant, data, bits=4, weight_symmetric=False, per_channel=True
    )
    onnx.checker.check_model(out)
    dq = _dq_node(out, "w1")
    assert len(dq.input) == 3
    inits = _inits(out)
    assert inits[dq.input[0]].dtype == np.uint8 and inits[dq.input[0]].max() <= 15
    assert inits[dq.input[2]].dtype == np.uint8 and inits[dq.input[2]].shape == (32,)
    assert reports[0].error_after == pytest.approx(
        _layer_error(float_model, out, "w1", data), rel=1e-6
    )


def test_mse_regrid_runs(mlp):
    float_model, quant, data = mlp
    out, reports = wr.gptq_int8(float_model, quant, data, bits=3, mse=True)
    onnx.checker.check_model(out)
    plain = wr.gptq_int8(float_model, quant, data, bits=3)[1]
    assert reports[0].error_after <= plain[0].error_after


def test_gemm_transb_blocks_run_along_the_right_axis():
    model = _model(
        "y = Gemm<transB=1>(x, w, b)",
        [("w", (10, 24)), ("b", (10,))],
        inputs="float[N,24] x",
        outputs="float[N,10] y",
        seed=3,
    )
    data = _batches((16, 24), 8)
    quant = quantize_full_qdq(model, calibration_data=data)
    out, reports = wr.gptq_int8(model, quant, data, bits=4, group_size=8)
    dq = _dq_node(out, "w")
    assert {a.name: a.i for a in dq.attribute} == {"axis": 1, "block_size": 8}
    assert _inits(out)[dq.input[1]].shape == (10, 3)
    x = np.concatenate([d["x"] for d in data])
    w = _inits(model)["w"]
    assert reports[0].error_after == pytest.approx(
        float(np.mean((x @ _dequantized(out, "w").T - x @ w.T) ** 2)), rel=1e-6
    )


def _conv_pair():
    model = _model(
        "h = Conv<pads=[1,1,1,1]>(x, w1) r = Relu(h) y = Conv<pads=[1,1,1,1]>(r, w2)",
        [("w1", (4, 3, 3, 3)), ("w2", (4, 4, 3, 3))],
        opset=21,
        seed=2,
        inputs="float[2,3,8,8] x",
        outputs="float[2,4,8,8] y",
    )
    data = _batches((2, 3, 8, 8), 6)
    return model, quantize_full_qdq(model, calibration_data=data), data


def test_conv_weights_regrid_per_tensor_and_skip_groups():
    model, quant, data = _conv_pair()
    out, reports = wr.gptq_int8(model, quant, data, bits=4)
    assert [r.op for r in reports] == ["Conv", "Conv"]
    onnx.checker.check_model(out)
    # a grouped scale has no blockable input-channel axis on a Conv weight
    out, reports = wr.gptq_int8(model, quant, data, bits=4, group_size=9)
    assert reports == [] and out.SerializeToString() == quant.SerializeToString()


def test_group_size_needs_opset_21_and_is_incompatible_with_act_order(mlp):
    float_model, quant, data = mlp
    with pytest.raises(NotImplementedError, match="act_order"):
        wr.gptq_int8(float_model, quant, data, bits=4, group_size=8, act_order=True)
    with pytest.raises(ValueError, match="group_size"):
        wr.gptq_int8(float_model, quant, data, bits=4, group_size=0)
    old = _model("y = MatMul(x, w)", [("w", (32, 8))], opset=17, outputs="float[N,8] y")
    d = _batches((16, 32), 4)
    with pytest.raises(NotImplementedError, match="opset >= 21"):
        wr.gptq_int8(old, quantize_full_qdq(old, calibration_data=d), d, group_size=8)


def test_regrid_inputs_are_not_mutated_and_a_shared_weight_is_skipped(mlp):
    float_model, quant, data = mlp
    f, q = float_model.SerializeToString(), quant.SerializeToString()
    wr.gptq_int8(float_model, quant, data, bits=4, group_size=8)
    assert float_model.SerializeToString() == f and quant.SerializeToString() == q
    shared = _model(
        "h = MatMul(x, w) t = Tanh(h) y = MatMul(t, w)",
        [("w", (32, 32))],
        outputs="float[N,32] y",
    )
    d = _batches((16, 32), 4)
    sq = quantize_full_qdq(shared, calibration_data=d)
    out, reports = wr.gptq_int8(shared, sq, d, bits=4)
    assert reports == [] and out.SerializeToString() == sq.SerializeToString()


# -- AdaRound --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scale, zp, dtype, lo, hi",
    [
        (0.05, 3, np.uint8, 0, 255),
        (0.03, -4, np.int8, -128, 127),
        (0.002, 100, np.uint16, 0, 65535),
    ],
)
def test_activation_fake_quant_matches_onnxruntime(scale, zp, dtype, lo, hi):
    dt = np.dtype(dtype).name
    model = parser.parse_model(
        f'<ir_version: 10, opset_import: ["": 21]> g (float[64] x) => (float[64] y) '
        f"<float s = {{{scale}}}, {dt} z = {{{zp}}}> "
        "{ q = QuantizeLinear(x, s, z) y = DequantizeLinear(q, s, z) }"
    )
    x = np.random.default_rng(0).uniform(-8, 8, 64).astype(np.float32)
    want = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
    got = wr._ActQuant(
        "x", np.float32(scale), float(zp), float(lo), float(hi)
    ).fake_quant(x.astype(np.float64))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=2e-4)


def test_drop_ratio_none_is_the_old_behaviour(mlp):
    float_model, quant, data = mlp
    a, _ = wr.adaround_int8(float_model, quant, data, num_iterations=40)
    b, _ = wr.adaround_int8(
        float_model, quant, data, num_iterations=40, drop_ratio=None
    )
    assert a.SerializeToString() == b.SerializeToString()


@pytest.mark.parametrize("drop_ratio", [0.0, 0.5, 1.0])
def test_drop_ratio_runs_and_never_regresses(mlp, drop_ratio):
    float_model, quant, data = mlp
    out, reports = wr.adaround_int8(
        float_model, quant, data, num_iterations=60, drop_ratio=drop_ratio
    )
    onnx.checker.check_model(out)
    assert reports and all(r.error_after <= r.error_before for r in reports)
    # error is judged on the quantized model's own input, so it includes the
    # activation rounding and is larger than the float-input error
    plain = wr.adaround_int8(float_model, quant, data, num_iterations=60)[1]
    assert reports[0].error_before > plain[0].error_before


def test_drop_ratio_is_deterministic_for_a_seed_and_depends_on_it(mlp):
    float_model, quant, data = mlp
    kw = dict(num_iterations=60, drop_ratio=0.5)
    a, _ = wr.adaround_int8(float_model, quant, data, seed=3, **kw)
    b, _ = wr.adaround_int8(float_model, quant, data, seed=3, **kw)
    c, _ = wr.adaround_int8(float_model, quant, data, seed=4, **kw)
    assert a.SerializeToString() == b.SerializeToString()
    assert a.SerializeToString() != c.SerializeToString()


def test_drop_ratio_extremes_use_the_matching_input():
    x_f = np.zeros((4, 3))
    x_q = np.ones((4, 3))
    rng = np.random.default_rng(0)
    ident = lambda a: a  # noqa: E731
    assert np.all(wr._Drop(x_q, x_f, ident, 1.0, rng).sample() == 1)
    assert np.all(wr._Drop(x_q, x_f, ident, 0.0, rng).sample() == 0)
    mixed = wr._Drop(np.ones((200, 50)), np.zeros((200, 50)), ident, 0.3, rng).sample()
    assert 0.25 < mixed.mean() < 0.35


def test_drop_ratio_on_conv_layers():
    model, quant, data = _conv_pair()
    out, reports = wr.adaround_int8(
        model, quant, data, num_iterations=30, drop_ratio=0.5, max_rows=512
    )
    assert [r.op for r in reports] == ["Conv", "Conv"]
    assert all(r.error_after <= r.error_before for r in reports)
    onnx.checker.check_model(out)


def test_selective_update_keeps_only_layers_that_shrink_the_output_distance(
    mlp, monkeypatch
):
    float_model, quant, data = mlp
    out, reports = wr.adaround_int8(
        float_model, quant, data, num_iterations=60, selective_update=True
    )
    f = wr._model_outputs(float_model, data, None)
    assert wr._avg_l2(f, wr._model_outputs(out, data, None)) <= wr._avg_l2(
        f, wr._model_outputs(quant, data, None)
    )
    # a distance that never improves reverts every layer
    ticks = iter(range(100))
    monkeypatch.setattr(wr, "_avg_l2", lambda a, b: 10.0 if next(ticks) == 0 else 11.0)
    out, reports = wr.adaround_int8(
        float_model, quant, data, num_iterations=30, selective_update=True
    )
    assert out.SerializeToString() == quant.SerializeToString()
    assert reports and not any(r.accepted or r.changed_fraction for r in reports)


def test_lr_adjust_picks_the_learning_rate_by_layer_error(mlp, monkeypatch):
    float_model, quant, data = mlp
    used = []
    import onnxsim.adaround as ar

    real = ar._optimize_rounding

    def spy(*args, **kwargs):
        used.append(args[6])
        return real(*args, **kwargs)

    monkeypatch.setattr(ar, "_optimize_rounding", spy)
    wr.adaround_int8(float_model, quant, data, num_iterations=5, learning_rate=0.1)
    wr.adaround_int8(
        float_model,
        quant,
        data,
        num_iterations=5,
        learning_rate=0.1,
        lr_adjust=(0.0, 0.7),
    )
    wr.adaround_int8(
        float_model,
        quant,
        data,
        num_iterations=5,
        learning_rate=0.1,
        lr_adjust=(1e9, 0.7),
    )
    assert used == [0.1, 0.1, 0.7, 0.7, 0.1, 0.1]


# -- through the compat layer ----------------------------------------------------------


def _quantize(algos, model=None, preset="A8W8"):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.algo_config = algos
    batches = _batches((16, 32), 4)

    class Reader:
        def __init__(self):
            self.it = iter(batches)

        def get_next(self):
            return next(self.it, None)

    q = qc.ModelQuantizer(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = q.quantize_model(
            model
            or _model(
                "h = MatMul(x, w1) t = Relu(h) y = MatMul(t, w2)",
                [("w1", (32, 32)), ("w2", (32, 16))],
                outputs="float[N,16] y",
            ),
            calibration_data_reader=Reader(),
        )
    return out, q


def test_compat_gptq_bits_and_group_size():
    out, q = _quantize([qc.GPTQConfig(bits=4, group_size=8)])
    onnx.checker.check_model(out)
    assert len(q.last_weight_rounding["gptq"]) == 2
    assert any("re-grids" in a for a in q.last_approximations)
    attrs = {a.name: a.i for a in _dq_node(out, "w1").attribute}
    assert attrs == {"axis": 0, "block_size": 8}


@pytest.mark.parametrize(
    "cfg",
    [
        qc.GPTQConfig(per_channel=True),
        qc.GPTQConfig(weight_symmetric=False),
        qc.GPTQConfig(bits=6, mse=True, act_order=True, perc_damp=0.05, block_size=16),
    ],
)
def test_compat_gptq_option_combinations_run(cfg):
    out, q = _quantize([cfg])
    onnx.checker.check_model(out)
    reports = q.last_weight_rounding["gptq"]
    assert reports and all(r.error_after <= r.error_before for r in reports)


def test_compat_gptq_without_grid_options_keeps_model_scales():
    _, q = _quantize([qc.GPTQConfig(act_order=True)])
    assert any("GPTQ keeps" in a for a in q.last_approximations)


def test_compat_gptq_refusals():
    with pytest.raises(NotImplementedError, match="act_order"):
        _quantize([qc.GPTQConfig(group_size=8, act_order=True)])
    with pytest.raises(ValueError, match="bits"):
        _quantize([qc.GPTQConfig(bits=12)])


def test_compat_adaround_options_run():
    cfg = qc.AdaRoundConfig(
        num_iterations=30,
        drop_ratio=0.5,
        selective_update=True,
        lr_adjust=(0.0, 0.3),
        data_size=2,
        update_bias=True,
    )
    out, q = _quantize([cfg])
    onnx.checker.check_model(out)
    assert q.last_weight_rounding["adaround"]


def test_compat_adaround_update_bias_is_ignored_like_in_quark():
    a, _ = _quantize([qc.AdaRoundConfig(num_iterations=30)])
    b, _ = _quantize([qc.AdaRoundConfig(num_iterations=30, update_bias=True)])
    assert a.SerializeToString() == b.SerializeToString()
