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


@pytest.mark.parametrize("preset", ["BFP16", "MX4", "MX9"])
def test_unsupported_dtypes_raise(preset):
    q = qc.ModelQuantizer(qc.QConfig.get_default_config(preset))
    with pytest.raises(NotImplementedError):
        q.quantize_model(_model(), calibration_data_reader=_Reader())


def test_algo_config_not_silently_dropped():
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("U8S8_AAWS_ADAROUND"))
    with pytest.raises(NotImplementedError, match="adaround"):
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


def test_adaquant_is_refused_not_run_as_a_noop():
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [qc.AdaQuantConfig()]
    with pytest.raises(NotImplementedError, match="adaquant"):
        qc.ModelQuantizer(cfg).quantize_model(
            _two_layer_model(), calibration_data_reader=_batches((4, 8))
        )
