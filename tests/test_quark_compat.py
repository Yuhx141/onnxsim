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
    assert qc.QConfig.get_default_config("A8W8").global_config.activation.dtype == "int8"
    assert qc.QConfig.get_default_config("U16S8_AAWS").global_config.activation.dtype == "uint16"
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
