"""Quark-style power-of-two MinMSE calibration (``method="minmse_pof2"``),
int8 biases / constants and the Quark-compat presets built on them. The
parity against real AMD Quark lives in ``test_quark_parity.py``; these tests
need no Quark."""

import warnings

import numpy as np
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc
from onnxsim.calibration import (
    _Pof2Histogram,
    calibrate,
    pof2_minmse_weight_scale,
)
from onnxsim.full_qdq import _pof2, quantize_full_qdq


def _model(body, initializer=(), opset=17):
    model = parser.parse_model(f'<ir_version: 8, opset_import: ["": {opset}]> {body}')
    model.graph.initializer.extend(initializer)
    return model


def _mlp(seed=0, heavy=True):
    rng = np.random.default_rng(seed)

    def w(*shape):
        a = rng.standard_t(2.5, shape) if heavy else rng.standard_normal(shape)
        return (a * 0.3).astype(np.float32)

    return _model(
        """g (float[4,12] x) => (float[4,6] y) {
            h = Gemm(x, w1, b1)
            r = Relu(h)
            s = Mul(r, k)
            y = Gemm(s, w2, b2)
        }""",
        [
            numpy_helper.from_array(w(12, 16), "w1"),
            numpy_helper.from_array(w(16), "b1"),
            numpy_helper.from_array(np.array(0.25, np.float32), "k"),
            numpy_helper.from_array(w(16, 6), "w2"),
            numpy_helper.from_array(w(6), "b2"),
        ],
    )


def _data(n=4, shape=(4, 12), seed=1):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


def _k_dtypes(model):
    """Integer dtype of the quantized Mul constant ``k``."""
    inits = _inits(model)
    return {
        inits[n.input[0]].dtype
        for n in model.graph.node
        if n.op_type == "DequantizeLinear"
        and n.input[0].startswith("k/")
        and n.input[0] in inits
    }


def _is_pof2(x):
    m, _ = np.frexp(np.asarray(x, np.float64))
    return bool(np.all(m == 0.5))


def _inits(model):
    return {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}


# -- the weight search -------------------------------------------------------


def _brute_force_weight_scale(w, qmin=-127, qmax=127):
    """The five candidates around the min/max scale, scored independently."""
    w = w.astype(np.float32).ravel()
    s0 = 2 * np.abs(w).max() / (qmax - qmin)
    p = int(np.rint(-np.log2(np.float32(s0))))
    errs = {}
    for pos in range(p - 1, p + 4):
        s = np.float32(2.0**-pos)
        dq = np.clip(np.round(w / s), qmin, qmax) * s
        errs[s] = float(np.sum((dq.astype(np.float64) - w) ** 2))
    return min(errs, key=errs.get)


@pytest.mark.parametrize("seed", range(6))
def test_weight_scale_is_the_least_error_candidate(seed):
    rng = np.random.default_rng(seed)
    w = (rng.standard_t(2.0, (64, 32)) * 0.2).astype(np.float32)
    s = pof2_minmse_weight_scale(w)
    assert _is_pof2(s)
    assert s == float(_brute_force_weight_scale(w))


def test_weight_scale_clips_outliers_when_that_lowers_the_error():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(4096).astype(np.float32) * 0.05
    w[0] = 1.2  # one outlier: ceil(log2) wastes resolution on it
    s = pof2_minmse_weight_scale(w)
    ceil_scale = _pof2(float(np.abs(w).max()) / 127)
    assert s < ceil_scale

    def err(sc):
        return float(np.sum((np.clip(np.round(w / sc), -127, 127) * sc - w) ** 2))

    assert err(s) < err(ceil_scale)


def test_weight_scale_degenerate_inputs():
    assert pof2_minmse_weight_scale(np.zeros(0, np.float32)) == 1.0
    assert _is_pof2(pof2_minmse_weight_scale(np.zeros(8, np.float32)))
    assert (
        pof2_minmse_weight_scale(np.full(8, 0.5, np.float32)) == 2.0**-7
    )  # the first exact candidate


def test_ceil_pof2_ignores_a_rounding_error_above_a_power_of_two():
    s = 2.0**-7
    assert _pof2(s) == s
    assert _pof2(s * (127 / 127.0 + 1e-12)) == s
    assert _pof2(s * 1.01) == 2.0**-6
    assert _pof2(0.0) == 0.0


# -- the activation histogram ---------------------------------------------------


def test_histogram_accumulates_and_expands_like_streaming():
    rng = np.random.default_rng(0)
    batches = [rng.standard_normal(500) * k for k in (1.0, 3.0, 0.5, 6.0)]
    h = _Pof2Histogram()
    for b in batches:
        h.add(b.astype(np.float32))
    assert int(h.counts.sum()) == 2000
    assert h.edges[0] <= min(b.min() for b in batches)
    assert h.edges[-1] >= max(b.max() for b in batches) - 1e-6
    widths = np.diff(h.edges)
    np.testing.assert_allclose(widths, widths[0], rtol=1e-9)
    assert _is_pof2(h.scale("uint8")) and _is_pof2(h.scale("int8"))


@pytest.mark.parametrize("dtype", ["uint8", "int8", "uint16", "int16"])
def test_histogram_scale_tracks_the_data_range(dtype):
    rng = np.random.default_rng(1)
    h = _Pof2Histogram()
    for _ in range(3):
        h.add(rng.standard_normal(2000).astype(np.float32))
    qmax = {"uint8": 127, "int8": 127, "uint16": 32767, "int16": 32767}[dtype]
    s = h.scale(dtype)
    # a power of two within a factor 4 of absmax / half-range
    ref = float(max(abs(h.rmin), abs(h.rmax))) / qmax
    assert ref / 4 <= s <= ref * 2


def test_empty_histogram_scale():
    assert _Pof2Histogram().scale() == 1.0


# -- calibrate / quantize_full_qdq ---------------------------------------------


def test_calibrate_minmse_pof2_ranges_recover_the_scale():
    model = _mlp()
    ranges = calibrate(
        model,
        _data(),
        method="minmse_pof2",
        activation_type="int8",
        tensor_names=["x", "r", "s", "y"],
    )
    assert len(ranges) == 4
    for lo, hi in ranges.values():
        assert lo == -hi  # symmetric
        assert _is_pof2(hi / 127)  # half-range 127 for int8


def test_calibrate_minmse_pof2_per_tensor_dtype():
    model = _mlp()
    kw = dict(method="minmse_pof2", activation_type="uint8", tensor_names=["x", "s"])
    r8 = calibrate(model, _data(), **kw)
    r16 = calibrate(model, _data(), tensor_dtypes={"s": "uint16"}, **kw)
    assert r16["x"] == r8["x"]
    assert _is_pof2(r16["s"][1] / 32767) and _is_pof2(r8["s"][1] / 127)
    assert r16["s"][1] != r8["s"][1]


def test_quantize_full_qdq_minmse_pof2_activation_scales_are_powers_of_two():
    out = quantize_full_qdq(
        _mlp(),
        _data(),
        activation_dtype="uint8",
        method="minmse_pof2",
        symmetric_activations=True,
        power_of_two=True,
        per_channel=False,
        pof2_mode="minmse",
    )
    inits = _inits(out)
    scales = [
        float(inits[n.input[1]])
        for n in out.graph.node
        if n.op_type == "QuantizeLinear"
    ]
    assert scales and all(_is_pof2(s) for s in scales)
    # symmetric uint8: centred zero point
    assert all(
        int(inits[n.input[2]]) == 128
        for n in out.graph.node
        if n.op_type == "QuantizeLinear"
    )


def test_minmse_pof2_needs_symmetric_activations():
    with pytest.raises(ValueError, match="symmetric"):
        quantize_full_qdq(
            _mlp(),
            _data(),
            activation_dtype="uint8",
            method="minmse_pof2",
            symmetric_activations=False,
        )


def test_pof2_mode_validated():
    with pytest.raises(ValueError, match="pof2_mode"):
        quantize_full_qdq(_mlp(), _data(), pof2_mode="floor")


def _dq_consts(model):
    inits = _inits(model)
    return {
        n.input[0]: (inits[n.input[0]], inits[n.input[1]])
        for n in model.graph.node
        if n.op_type == "DequantizeLinear" and n.input[0] in inits
    }


def test_int8_bias_replaces_int32_and_stays_accurate():
    model = _mlp(heavy=False)
    kw = dict(
        calibration_data=_data(),
        activation_dtype="uint8",
        symmetric_activations=True,
        power_of_two=True,
        per_channel=False,
        pof2_mode="minmse",
    )
    int32 = quantize_full_qdq(model, **kw)
    int8 = quantize_full_qdq(model, int8_bias=True, **kw)
    orig = _inits(model)
    for tag, m, dtype in (("int32", int32, np.int32), ("int8", int8, np.int8)):
        biases = {
            k: v
            for k, v in _dq_consts(m).items()
            if v[0].ndim == 1 and v[0].dtype == dtype
        }
        assert len(biases) == 2, tag
    for q, s in (v for k, v in _dq_consts(int8).items() if v[0].ndim == 1):
        if q.dtype == np.int8 and q.shape == orig["b1"].shape:
            assert s.ndim == 0 and _is_pof2(s)  # per-tensor power of two
            assert np.max(np.abs(q.astype(np.float32) * s - orig["b1"])) <= float(s)


def test_int8_constants_and_eltwise_alignment():
    model = _mlp(heavy=False)
    kw = dict(
        calibration_data=_data(),
        activation_dtype="int16",
        symmetric_activations=True,
        per_channel=False,
    )
    plain = quantize_full_qdq(model, **kw)
    as_weight = quantize_full_qdq(model, int8_constants=True, **kw)
    aligned = quantize_full_qdq(
        model, int8_constants=True, align_eltwise_dtype=True, **kw
    )

    def dtype_of_k(m):
        return _k_dtypes(m)

    assert dtype_of_k(plain) == {np.dtype(np.int16)}
    assert dtype_of_k(as_weight) == {np.dtype(np.int8)}
    assert dtype_of_k(aligned) == {np.dtype(np.int16)}


def test_softmax_unit_range():
    rng = np.random.default_rng(0)
    model = _model(
        """g (float[2,8] x) => (float[2,8] y) {
            h = Gemm(x, w, b)
            y = Softmax<axis=-1>(h)
        }""",
        [
            numpy_helper.from_array(
                (rng.standard_normal((8, 8)) * 0.1).astype(np.float32), "w"
            ),
            numpy_helper.from_array(np.zeros(8, np.float32), "b"),
        ],
    )
    data = [{"x": rng.standard_normal((2, 8)).astype(np.float32)} for _ in range(3)]

    def y_scale(**kw):
        out = quantize_full_qdq(model, data, activation_dtype="uint8", **kw)
        inits = _inits(out)
        q = next(
            n
            for n in out.graph.node
            if n.op_type == "QuantizeLinear" and n.input[0] == "y/f"
        )
        return float(inits[q.input[1]]), int(inits[q.input[2]])

    observed = y_scale()
    fixed = y_scale(softmax_unit_range=True)
    assert fixed == (float(np.float32(1.0 / 255)), 0)
    assert observed[0] < fixed[0]  # near-uniform softmax never reaches 1


# -- quark_compat presets --------------------------------------------------------


def _quantize(preset_or_cfg, model=None, data=None, **extra):
    cfg = (
        qc.QConfig.get_default_config(preset_or_cfg)
        if isinstance(preset_or_cfg, str)
        else preset_or_cfg
    )
    cfg.extra_options.update(extra)

    class Reader:
        def __init__(self, batches):
            self.it = iter(batches)

        def get_next(self):
            return next(self.it, None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model or _mlp(), calibration_data_reader=Reader(data or _data())
        )


def test_xint8_preset_emits_pof2_scales_and_int8_biases():
    out = _quantize("XINT8")
    inits = _inits(out)
    for n in out.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear") and n.input[1] in inits:
            assert _is_pof2(inits[n.input[1]]), n.name
    dtypes = {a.dtype for a in inits.values() if a.ndim == 1}
    assert np.dtype(np.int8) in dtypes and np.dtype(np.int32) not in dtypes


def test_xint8_int32_bias_option():
    out = _quantize("XINT8", Int32Bias=True)
    assert any(a.dtype == np.int32 for a in _inits(out).values())


def test_non_pof2_presets_keep_int32_bias():
    for preset in ("A8W8", "U8S8_AAWS"):
        out = _quantize(preset)
        assert any(a.dtype == np.int32 for a in _inits(out).values()), preset


def test_spec_calibration_defaults_follow_quark():
    assert qc.Int8Spec().calibration_method == "percentile:99.999"
    assert qc.UInt16Spec().calibration_method == "percentile:99.999"
    assert qc.XInt8Spec().calibration_method == "minmse_pof2"
    assert qc.Float16Spec().calibration_method == "minmax"
    p = qc.QConfig.get_default_config
    assert p("A8W8").global_config.activation.calibration_method == "minmax"
    assert p("A16W8").global_config.activation.calibration_method == "minmax"
    assert (
        p("S8S8_AAWS").global_config.activation.calibration_method
        == "percentile:99.9999"
    )
    assert (
        p("S16S8_ASWS").global_config.activation.calibration_method
        == "percentile:99.999"
    )
    assert p("XINT8").global_config.activation.calibration_method == "minmse_pof2"
    assert p("A16W8").extra_options["AlignEltwiseQuantType"] is True


def test_calib_method_enum_maps_to_onnxsim_methods():
    names = {
        qc.CalibMethod.MinMax: "minmax",
        qc.CalibMethod.MinMSE: "minmse_pof2",
        qc.CalibMethod.Percentile: "percentile:99.999",
        qc.CalibMethod.Entropy: "entropy",
    }
    for member, name in names.items():
        assert qc.Int8Spec(calibration_method=member).calibration_method == name
    # Distribution / LayerwisePercentile are Quark's calibrators now
    # (tests/test_quark_calibration_methods.py): they run instead of raising
    for member in (qc.CalibMethod.Distribution, qc.CalibMethod.LayerwisePercentile):
        out = _quantize(
            qc.QConfig(
                qc.QLayerConfig(
                    activation=qc.Int8Spec(calibration_method=member),
                    weight=qc.Int8Spec(),
                )
            )
        )
        assert any(n.op_type == "QuantizeLinear" for n in out.graph.node)


def test_user_built_pof2_config_uses_minmse():
    cfg = qc.QConfig(qc.QLayerConfig(activation=qc.XInt8Spec(), weight=qc.XInt8Spec()))
    out = _quantize(cfg)
    inits = _inits(out)
    assert all(
        _is_pof2(inits[n.input[1]])
        for n in out.graph.node
        if n.op_type == "QuantizeLinear" and n.input[1] in inits
    )


def test_a16w8_eltwise_constant_keeps_activation_dtype():
    assert _k_dtypes(_quantize("A16W8")) == {np.dtype(np.int16)}
    assert _k_dtypes(_quantize("A8W8")) == {np.dtype(np.int8)}


def test_compat_outputs_stay_close_to_float():
    import onnxruntime as ort

    model = _mlp(heavy=False)
    x = _data(1)[0]["x"]
    ref = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"x": x})[0]
    for preset in ("XINT8", "A8W8", "S8S8_AAWS", "S16S8_ASWS"):
        out = _quantize(preset, model)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        got = ort.InferenceSession(
            out.SerializeToString(), so, providers=["CPUExecutionProvider"]
        ).run(None, {"x": x})[0]
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        assert rel < 0.25, (preset, rel)
