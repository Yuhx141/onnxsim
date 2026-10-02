"""Quark-free tests of ``INT{8,16}_TRANSFORMER_{DEFAULT,ACCURATE}`` in
:mod:`onnxsim.quark_compat` (parity against the real Quark lives in
``tests/test_quark_parity.py``)."""

import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc
from onnxsim.calibration import calibrate

D = 8


def _attention_block():
    rng = np.random.default_rng(0)
    m = parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["": 20]>
        g (float[1,4,{D}] x) => (float[1,4,{D}] y) {{
            q = MatMul(x, wq)
            k = MatMul(x, wk)
            kt = Transpose<perm=[0,2,1]>(k)
            s = MatMul(q, kt)
            p = Softmax<axis=-1>(s)
            c = MatMul(p, q)
            o = MatMul(c, wo)
            r = Add(x, o)
            n = LayerNormalization<axis=-1>(r, g1, b1)
            h = MatMul(n, w1)
            a = Relu(h)
            y = MatMul(a, w2)
        }}
        """
    )
    m.graph.initializer.extend(
        numpy_helper.from_array((rng.standard_normal(s) * 0.5).astype(np.float32), n)
        for n, s in [
            ("wq", (D, D)),
            ("wk", (D, D)),
            ("wo", (D, D)),
            ("w1", (D, 2 * D)),
            ("w2", (2 * D, D)),
        ]
    )
    m.graph.initializer.extend(
        [
            numpy_helper.from_array(np.ones(D, np.float32), "g1"),
            numpy_helper.from_array(np.zeros(D, np.float32), "b1"),
        ]
    )
    return m


def _data(n=4, seed=3):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal((1, 4, D)).astype(np.float32)} for _ in range(n)]


def _quantize(model, preset="INT8_TRANSFORMER_DEFAULT", **extra):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model,
            calibration_data_reader=_Reader(_data()),
            ignore_unsupported_algos=True,
        )


class _Reader:
    def __init__(self, data):
        self.it = iter(data)

    def get_next(self):
        return next(self.it, None)


def _run(model, x):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x})[0]


def _quantized_tensors(model):
    """Names of the float tensors that went through a QuantizeLinear."""
    return {
        n.input[0].split("/")[0]
        for n in model.graph.node
        if n.op_type == "QuantizeLinear"
    }


@pytest.mark.parametrize(
    "name", ["INT8_TRANSFORMER_DEFAULT", "INT16_TRANSFORMER_DEFAULT"]
)
def test_only_gemm_and_const_matmul_are_quantized(name):
    q = _quantize(_attention_block(), name)
    wide = "int16" in name.lower()
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    # constant-B MatMuls (wq, wk, wo, w1, w2) carry int8 / int16 weights
    codes = [
        a for k, a in inits.items() if a.ndim == 2 and a.dtype in (np.int8, np.int16)
    ]
    assert len(codes) == 5
    assert {c.dtype for c in codes} == {np.dtype("int16" if wide else "int8")}
    # LayerNorm scale / bias stay float, as does the Softmax / Add / Relu path
    assert {"g1", "b1"} <= set(inits)
    ops = [n.op_type for n in q.graph.node]
    assert ops.count("Softmax") == ops.count("LayerNormalization") == 1
    # every Q/DQ pair is an input or output of a constant-B MatMul
    quantized = _quantized_tensors(q)
    # (``h`` -> Relu is left float, see the next test)
    assert quantized == {"x", "q", "k", "c", "o", "n", "a", "y"}


def test_relu_after_quantized_matmul_keeps_float_input():
    q = _quantize(_attention_block())
    assert "h" not in _quantized_tensors(q)
    # Gemm -> Relu: Quark drops the Q/DQ between them (the Relu's own output
    # is the next MatMul's quantized input)
    relu = next(n for n in q.graph.node if n.op_type == "Relu")
    prod = next(n for n in q.graph.node if relu.input[0] in n.output)
    assert prod.op_type == "MatMul"
    cons = next(n for n in q.graph.node if relu.output[0] in n.input)
    assert cons.op_type == "QuantizeLinear"


def test_matmul_const_b_only_can_be_disabled():
    q = _quantize(_attention_block(), MatMulConstBOnly=False)
    quantized = _quantized_tensors(q)
    assert {"s", "kt", "p"} & quantized  # activation x activation MatMuls too


def test_model_without_gemm_or_matmul_is_returned_unchanged():
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[1,4,8] x) => (float[1,4,8] y) {
            r = Relu(x)
            y = Softmax<axis=-1>(r)
        }
        """
    )
    q = _quantize(m)
    assert q.SerializeToString() == m.SerializeToString()


@pytest.mark.parametrize(
    "name", ["INT8_TRANSFORMER_DEFAULT", "INT16_TRANSFORMER_DEFAULT"]
)
def test_quantized_model_tracks_float_model(name):
    model = _attention_block()
    # (the default's mean-of-extremes calibration clips: use plain min / max
    # and a calibration batch so nothing does)
    q = _quantize(model, name, CalibMovingAverage=False)
    x = _data()[0]["x"]
    ref = _run(model, x)
    err = np.linalg.norm(_run(q, x) - ref) / np.linalg.norm(ref)
    assert err < (0.01 if "16" in name else 0.1)


def test_preset_settings():
    for name, dt, wdt in [
        ("INT8_TRANSFORMER_DEFAULT", "uint8", "int8"),
        ("INT16_TRANSFORMER_DEFAULT", "uint16", "int16"),
        ("INT8_TRANSFORMER_ACCURATE", "uint8", "int8"),
        ("INT16_TRANSFORMER_ACCURATE", "uint16", "int16"),
    ]:
        cfg = qc.QConfig.get_default_config(name)
        g = cfg.global_config
        assert (g.activation.dtype, g.weight.dtype) == (dt, wdt)
        assert g.activation.symmetric is False
        assert cfg.extra_options["NPUTransformer"] is True
        accurate = name.endswith("ACCURATE")
        assert g.activation.calibration_method == (
            "percentile:99.9999" if accurate else "minmax_mean"
        )
        assert [a.name for a in cfg.algo_config] == (["adaround"] if accurate else [])


def test_minmax_mean_is_the_mean_of_batch_extremes():
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[2,3] x) => (float[2,3] y) {
            y = Relu(x)
        }
        """
    )
    batches = [
        {"x": np.array([[-1, 0, 2], [1, 1, 1]], np.float32)},
        {"x": np.array([[-3, 0, 4], [1, 1, 1]], np.float32)},
    ]
    r = calibrate(m, batches, method="minmax_mean", tensor_names=["x"])
    assert r["x"] == (-2.0, 3.0)
    assert calibrate(m, batches, method="minmax", tensor_names=["x"])["x"] == (
        -3.0,
        4.0,
    )


def test_adaround_runs_for_int8_and_needs_ignore_flag_for_int16():
    model = _attention_block()
    cfg = qc.QConfig.get_default_config("INT8_TRANSFORMER_ACCURATE")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(_data())
        )
    onnx.checker.check_model(q)
    cfg = qc.QConfig.get_default_config("INT16_TRANSFORMER_ACCURATE")
    with pytest.raises(NotImplementedError, match="adaround"):
        qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(_data())
        )
