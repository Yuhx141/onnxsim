"""Tests for onnxsim.quark_compat -- the Quark-ONNX-API-shaped shim backed by
onnxsim's own quantizers (see that module's docstring for scope)."""

import numpy as np
import onnx
import pytest
from onnx import parser

from onnxsim import quark_compat as qc


def _model():
    return parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,4] x) => (float[N,4] y)
        <float[4,4] w = {1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0}>
        {
            y = MatMul(x, w)
        }
        """
    )


class _Reader:
    """onnxruntime-style CalibrationDataReader."""

    def __init__(self, n=4):
        rng = np.random.default_rng(0)
        self._it = iter(
            [{"x": rng.standard_normal((2, 4)).astype(np.float32)} for _ in range(n)]
        )

    def get_next(self):
        return next(self._it, None)


def _ops(model):
    return {n.op_type for n in model.graph.node}


def test_default_config_lookup_and_unknown():
    assert (
        qc.QConfig.get_default_config("A8W8").global_config.activation.dtype == "int8"
    )
    assert (
        qc.QConfig.get_default_config("U16S8_AAWS").global_config.activation.dtype
        == "uint16"
    )
    with pytest.raises(ValueError, match="unknown preset"):
        qc.QConfig.get_default_config("NOPE")


def test_adaround_preset_carries_algo():
    cfg = qc.QConfig.get_default_config("A8W8_ADAROUND")
    assert [a.name for a in cfg.algo_config] == ["adaround"]


def test_config_wrapper_and_extra_options():
    cfg = qc.QConfig(
        qc.QLayerConfig(qc.Int8Spec(), qc.Int8Spec()),
        extra_options={"FoldRelu": True},
    )
    assert cfg.extra_options == {"FoldRelu": True}
    q = qc.ModelQuantizer(qc.Config(global_quant_config=cfg))
    assert q.config is cfg


def test_u8_preset_quantizes_to_qdq_without_approximation(tmp_path):
    out = tmp_path / "q.onnx"
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("U8S8_AAWS"))
    model = q.quantize_model(_model(), str(out), _Reader())
    assert {"QuantizeLinear", "DequantizeLinear"} <= _ops(model)
    assert q.last_approximations == []
    onnx.checker.check_model(onnx.load(str(out)))


def test_int8_activation_preset_warns_about_approximation():
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("A8W8"))
    with pytest.warns(UserWarning, match="int8 activations mapped to uint8"):
        q.quantize_model(_model(), calibration_data_reader=_Reader())
    assert any("int8 activations" in m for m in q.last_approximations)


def test_a16w8_uses_uint16_activations():
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("A16W8"))
    with pytest.warns(UserWarning):
        model = q.quantize_model(_model(), calibration_data_reader=_Reader())
    types = {i.data_type for i in model.graph.initializer}
    assert onnx.TensorProto.UINT16 in types


@pytest.mark.parametrize("preset", ["FP16", "BF16"])
def test_float_presets_need_no_calibration(preset):
    q = qc.ModelQuantizer(qc.QConfig.get_default_config(preset))
    model = q.quantize_model(_model())
    assert "Cast" in _ops(model)


def _wide_matmul_model():
    rng = np.random.default_rng(5)
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,32] x) => (float[N,8] y)
        {
            y = MatMul(x, w)
        }
        """
    )
    # Random weights are attached programmatically (too large for text literals).
    w = (rng.standard_normal((32, 8)) * np.exp2(rng.integers(-3, 3, (32, 8)))).astype(
        np.float32
    )
    model.graph.initializer.append(onnx.numpy_helper.from_array(w, "w"))
    return model


@pytest.mark.parametrize(
    "preset, fmt",
    [
        ("BFP16", "bfp16"),
        ("MX4", "mx4"),
        ("MX9", "mx9"),
        ("MXINT8", "mxint8"),
        ("MXFP8E4M3", "mxfp8_e4m3"),
        ("MXFP4E2M1", "mxfp4_e2m1"),
    ],
)
def test_block_format_presets_fake_quantize_weights(preset, fmt):
    from onnxsim import quark_block_formats as bf

    q = qc.ModelQuantizer(qc.QConfig.get_default_config(preset))
    model = _wide_matmul_model()
    with pytest.warns(UserWarning, match="activations are not quantized"):
        out = q.quantize_model(model)
    w_in = onnx.numpy_helper.to_array(model.graph.initializer[0])
    w_out = onnx.numpy_helper.to_array(out.graph.initializer[0])
    assert not np.array_equal(w_in, w_out)
    # blocks run along the reduction axis (K = axis 0 of a MatMul weight)
    expected = {
        "bfp16": lambda a: bf.bfp16(a, axis=0),
        "mx4": lambda a: bf.bfp_prime(a, bit_width=11, axis=0),
        "mx9": lambda a: bf.bfp_prime(a, bit_width=16, axis=0),
        "mxint8": lambda a: bf.mx(a, element_dtype="int8", axis=0),
        "mxfp8_e4m3": lambda a: bf.mx(a, element_dtype="fp8_e4m3", axis=0),
        "mxfp4_e2m1": lambda a: bf.mx(a, element_dtype="fp4_e2m1", axis=0),
    }[fmt](w_in)
    np.testing.assert_array_equal(w_out, expected)
    assert out.graph.node[0].op_type == "MatMul"  # graph untouched, no custom ops
    onnx.checker.check_model(out)


def test_block_format_conv_blocks_input_channels():
    from onnxsim import quark_block_formats as bf

    model = _conv_model()
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("BFP16"))
    with pytest.warns(UserWarning, match="activations are not quantized"):
        out = q.quantize_model(model)
    w_in = onnx.numpy_helper.to_array(model.graph.initializer[0])
    w_out = onnx.numpy_helper.to_array(
        next(t for t in out.graph.initializer if t.name == "w1")
    )
    np.testing.assert_array_equal(w_out, bf.bfp16(w_in, axis=1))


def test_block_format_with_algo_config_is_refused():
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("BFP16_ADAQUANT"))
    with pytest.raises(NotImplementedError, match="block formats"):
        q.quantize_model(_wide_matmul_model())


def test_no_adaround_variant_for_block_presets():
    qc.QConfig.get_default_config("MX9_ADAQUANT")
    with pytest.raises(ValueError):
        qc.QConfig.get_default_config("MX9_ADAROUND")


def test_unknown_algo_config_is_not_silently_dropped():
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [qc.AlgoConfig(name="some_new_algo")]
    q = qc.ModelQuantizer(cfg)
    with pytest.raises(NotImplementedError, match="some_new_algo"):
        q.quantize_model(_model(), calibration_data_reader=_Reader())
    q.quantize_model(
        _model(), calibration_data_reader=_Reader(), ignore_unsupported_algos=True
    )


def test_integer_preset_requires_reader():
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("U8S8_AAWS"))
    with pytest.raises(ValueError, match="calibration_data_reader"):
        q.quantize_model(_model())


def _two_layer_model():
    rng = np.random.default_rng(1)
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,8] x) => (float[N,8] y)
        {
            h = MatMul(x, w1)
            r = Relu(h)
            y = MatMul(r, w2)
        }
        """
    )
    # Random weights are attached programmatically (too large for text literals).
    model.graph.initializer.extend(
        onnx.numpy_helper.from_array(rng.standard_normal((8, 8)).astype(np.float32), n)
        for n in ("w1", "w2")
    )
    return model


def _conv_model():
    rng = np.random.default_rng(2)
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[1,3,8,8] x) => (float[1,4,8,8] y)
        {
            h = Conv<pads=[1,1,1,1]>(x, w1, b1)
            r = Relu(h)
            y = Conv<pads=[1,1,1,1]>(r, w2, b2)
        }
        """
    )
    scale = np.array([1.0, 10.0, 0.1, 3.0], dtype=np.float32)  # uneven channel ranges
    for name, shape in (
        ("w1", (4, 3, 3, 3)),
        ("b1", (4,)),
        ("w2", (4, 4, 3, 3)),
        ("b2", (4,)),
    ):
        arr = rng.standard_normal(shape).astype(np.float32)
        arr *= scale.reshape(-1, *[1] * (len(shape) - 1))
        model.graph.initializer.append(onnx.numpy_helper.from_array(arr, name))
    return model


def _quantize(model, algos, batches):
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = algos
    return qc.ModelQuantizer(cfg).quantize_model(model, calibration_data_reader=batches)


def _batches(shape, n=4):
    rng = np.random.default_rng(3)
    return [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


@pytest.mark.parametrize(
    "algo", [qc.SmoothQuantConfig(alpha=0.5), qc.BiasCorrectionConfig()]
)
def test_runnable_algo_changes_the_quantized_model(algo):
    batches = _batches((4, 8))
    base = _quantize(_two_layer_model(), [], batches)
    out = _quantize(_two_layer_model(), [algo], batches)
    assert out.SerializeToString() != base.SerializeToString()


def test_cle_changes_a_conv_relu_conv_model():
    batches = _batches((1, 3, 8, 8))
    base = _quantize(_conv_model(), [], batches)
    out = _quantize(_conv_model(), [qc.CLEConfig()], batches)
    assert out.SerializeToString() != base.SerializeToString()


def test_adaquant_runs_and_changes_the_quantized_model():
    batches = _batches((4, 8))
    base = _quantize(_two_layer_model(), [], batches)
    algo = qc.AdaQuantConfig(num_iterations=20)
    out = _quantize(_two_layer_model(), [algo], batches)
    assert out.SerializeToString() != base.SerializeToString()


def test_adaquant_preset_runs_end_to_end():
    cfg = qc.QConfig.get_default_config("U8S8_AAWS_ADAQUANT")
    cfg.algo_config[0].params["num_iterations"] = 20
    out = qc.ModelQuantizer(cfg).quantize_model(
        _two_layer_model(), calibration_data_reader=_batches((4, 8))
    )
    onnx.checker.check_model(out)


@pytest.mark.parametrize("preset", ["U8S8_AAWS_ADAROUND"])
def test_adaround_preset_runs_and_reports_layers(preset):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.algo_config[0].params["num_iterations"] = 30
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning, match="AdaRound is layer-wise"):
        out = q.quantize_model(
            _two_layer_model(), calibration_data_reader=_batches((4, 8))
        )
    reports = q.last_weight_rounding["adaround"]
    assert reports and all(r.error_after <= r.error_before for r in reports)
    base = _quantize(_two_layer_model(), [], _batches((4, 8)))
    assert out.SerializeToString() != base.SerializeToString()
    onnx.checker.check_model(out)


def test_gptq_runs_through_the_compat_layer():
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [qc.GPTQConfig(act_order=True, perc_damp=0.02)]
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning, match="GPTQ keeps"):
        out = q.quantize_model(
            _two_layer_model(), calibration_data_reader=_batches((4, 8))
        )
    reports = q.last_weight_rounding["gptq"]
    assert reports and all(r.error_after <= r.error_before for r in reports)
    onnx.checker.check_model(out)


@pytest.mark.parametrize(
    "algo, match",
    [
        (qc.AdaRoundConfig(update_bias=True), "update_bias"),
        (qc.AdaRoundConfig(drop_ratio=0.5), "drop_ratio"),
        (qc.GPTQConfig(bits=4), "bits must be 8"),
        (qc.GPTQConfig(group_size=128), "group_size"),
        (qc.GPTQConfig(weight_symmetric=False), "weight_symmetric"),
    ],
)
def test_weight_rounding_options_that_change_the_meaning_are_refused(algo, match):
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [algo]
    with pytest.raises(NotImplementedError, match=match):
        qc.ModelQuantizer(cfg).quantize_model(
            _two_layer_model(), calibration_data_reader=_batches((4, 8))
        )


def test_weight_rounding_state_is_reset_between_runs():
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [qc.GPTQConfig()]
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning):
        q.quantize_model(_two_layer_model(), calibration_data_reader=_batches((4, 8)))
    assert "gptq" in q.last_weight_rounding
    q.config.algo_config = []
    q.quantize_model(_two_layer_model(), calibration_data_reader=_batches((4, 8)))
    assert q.last_weight_rounding == {}


def _amp_config(**params):
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    target = params.pop(
        "target_layer_config", qc.QLayerConfig(qc.UInt16Spec(), qc.Int8Spec())
    )
    cfg.algo_config = [qc.AutoMixprecisionConfig(target_layer_config=target, **params)]
    return cfg


def test_auto_mixprecision_moves_layers_to_the_target_activation_precision():
    q = qc.ModelQuantizer(
        _amp_config(metric_optimize_object="quality", metric_threshold=0)
    )
    with pytest.warns(UserWarning, match="AutoMixprecision"):
        out = q.quantize_model(
            _two_layer_model(), calibration_data_reader=_batches((4, 8))
        )
    res = q.last_auto_mixprecision
    assert res is not None and res.moved  # threshold 0: every candidate moves
    assert res.final_score < res.baseline_score
    # the moved layers' activations really are uint16 now
    assert onnx.TensorProto.UINT16 in {i.data_type for i in out.graph.initializer}
    onnx.checker.check_model(out)


def test_auto_mixprecision_threshold_none_is_sensitivity_only():
    q = qc.ModelQuantizer(_amp_config(metric_threshold=None))
    with pytest.warns(UserWarning, match="AutoMixprecision"):
        out = q.quantize_model(
            _two_layer_model(), calibration_data_reader=_batches((4, 8))
        )
    assert q.last_auto_mixprecision.ranked and not q.last_auto_mixprecision.moved
    assert onnx.TensorProto.UINT16 not in {i.data_type for i in out.graph.initializer}


@pytest.mark.parametrize(
    "params, error, match",
    [
        (
            {
                "target_layer_config": {
                    qc.QLayerConfig(qc.UInt16Spec(), qc.Int8Spec()): []
                }
            },
            NotImplementedError,
            "single QLayerConfig",
        ),
        ({"subgraph_json": "x.json"}, NotImplementedError, "subgraph_json"),
        (
            {"sensitivity_cache_file": "c.json"},
            NotImplementedError,
            "sensitivity_cache_file",
        ),
        (
            {"target_layer_config": qc.QLayerConfig(qc.UInt8Spec(), qc.Int8Spec())},
            ValueError,
            "nothing to mix",
        ),
    ],
)
def test_auto_mixprecision_unsupported_forms_are_refused(params, error, match):
    q = qc.ModelQuantizer(_amp_config(**params))
    with pytest.raises(error, match=match):
        q.quantize_model(_two_layer_model(), calibration_data_reader=_batches((4, 8)))


def test_auto_mixprecision_state_is_reset_between_runs():
    q = qc.ModelQuantizer(_amp_config(metric_threshold=None))
    with pytest.warns(UserWarning):
        q.quantize_model(_two_layer_model(), calibration_data_reader=_batches((4, 8)))
    assert q.last_auto_mixprecision is not None
    q.config.algo_config = []
    q.quantize_model(_two_layer_model(), calibration_data_reader=_batches((4, 8)))
    assert q.last_auto_mixprecision is None


def _stream_model():
    # x -> w_in (writes the stream) -> per-channel scale (the "norm") -> w_out (reads it)
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,8] x) => (float[N,8] y)
        {
            s = MatMul(x, w_in)
            n = Mul(s, g)
            y = MatMul(n, w_out)
        }
        """
    )
    rng = np.random.default_rng(7)
    # Random weights are attached programmatically (too large for text literals).
    for name, shape, scale in (
        ("w_in", (8, 16), 1.0),
        ("g", (16,), 0.5),
        ("w_out", (16, 8), 0.3),
    ):
        arr = rng.standard_normal(shape).astype(np.float32) * scale
        if name == "g":
            arr += 1.0
        model.graph.initializer.append(onnx.numpy_helper.from_array(arr, name))
    for node in model.graph.node:
        node.name = node.output[0]
    return model


def _quarot_cfg(tmp_path, **params):
    import json

    path = tmp_path / "rot.json"
    path.write_text(
        json.dumps(
            {"R1_pairs": [{"prev_nodes": ["s"], "next_nodes": ["y"], "norm_node": "n"}]}
        )
    )
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [
        qc.QuarotConfig(r_matrix_dim=16, r_config_path=str(path), **params)
    ]
    return cfg


def test_quarot_rotates_the_float_weights_before_quantization(tmp_path):
    import onnxruntime as ort

    batches = _batches((6, 8))
    plain = _quantize(_stream_model(), [], batches)
    q = qc.ModelQuantizer(_quarot_cfg(tmp_path))
    with pytest.warns(UserWarning, match="Quarot"):
        out = q.quantize_model(_stream_model(), calibration_data_reader=batches)
    codes = lambda m: {  # noqa: E731
        t.name: onnx.numpy_helper.to_array(t)
        for t in m.graph.initializer
        if t.name.endswith("/int8")
    }
    assert any(not np.array_equal(codes(out)[k], codes(plain)[k]) for k in codes(out))

    def run(m, x):
        sess = ort.InferenceSession(m.SerializeToString())
        return sess.run(None, {"x": x})[0]

    x = batches[0]["x"]
    ref = run(_stream_model(), x)
    err = lambda m: float(np.abs(run(m, x) - ref).max())  # noqa: E731
    assert err(out) < 5 * err(plain) + 0.05  # a different basis, same function


def test_quarot_requires_a_config_path():
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [qc.QuarotConfig()]
    with pytest.raises(ValueError, match="r_config_path"):
        qc.ModelQuantizer(cfg).quantize_model(
            _stream_model(), calibration_data_reader=_batches((6, 8))
        )


@pytest.mark.parametrize("preset", ["FP16", "BF16"])
def test_float_presets_refuse_an_algo_config_instead_of_dropping_it(preset):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.algo_config = [qc.SmoothQuantConfig()]
    q = qc.ModelQuantizer(cfg)
    with pytest.raises(NotImplementedError, match="float presets"):
        q.quantize_model(_model())
    out = q.quantize_model(_model(), ignore_unsupported_algos=True)
    assert "Cast" in _ops(out)
