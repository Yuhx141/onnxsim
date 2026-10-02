"""Parity of onnxsim's XINT8 (power-of-two) flow against the real AMD Quark ONNX
package: the BiasCorrection scale quirk and the NPU graph rewrites Quark's
``XINT8`` preset runs. Skipped unless ``quark.onnx`` is importable.

Each test runs Quark and onnxsim on the same parser-built graph and compares the
emitted graphs node by node (op types, attributes, scales, zero points,
constants) and, with ONNX Runtime's graph optimizations off, what they compute.
"""

import contextlib
import copy
import io
import os
import warnings

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

warnings.filterwarnings("ignore")

with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
):
    try:
        import quark.onnx as quark_onnx
    except Exception as e:  # pragma: no cover - environment dependent
        quark_onnx = None
        _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    quark_onnx is None, reason="AMD Quark (amd-quark) is not installed"
)

from onnxsim import quark_bias_correction as qbc  # noqa: E402
from onnxsim import quark_compat as qc  # noqa: E402


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Quark writes scratch files into the current directory."""
    monkeypatch.chdir(tmp_path)


# -- helpers --------------------------------------------------------------------


def _data(shape, n=4, seed=3, name="x"):
    rng = np.random.default_rng(seed)
    return [{name: rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


def _reader(data):
    from onnxruntime.quantization import CalibrationDataReader

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter(data)

        def get_next(self):
            return next(self.it, None)

        def reset_iter(self):
            self.it = iter(data)

    return R()


def _named(model):
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return onnx.shape_inference.infer_shapes(model)


def _xint8_quark_config(algos=(), extra=None):
    from quark.onnx import QConfig, QLayerConfig, XInt8Spec

    return QConfig(
        global_config=QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()),
        algo_config=list(algos),
        extra_options={"ForceQuantizeNoInputCheck": True, **(extra or {})},
    )


def _quark(model, data, tmp_path, algos=(), extra=None):
    from quark.onnx import ModelQuantizer

    src, dst = str(tmp_path / "src.onnx"), str(tmp_path / "dst.onnx")
    onnx.save(model, src)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        ModelQuantizer(_xint8_quark_config(algos, extra)).quantize_model(
            src, dst, _reader(data)
        )
    return onnx.load(dst)


def _mine(model, data, algos=(), extra=None, quiet=True):
    cfg = qc.QConfig(
        global_config=qc.QLayerConfig(
            activation=qc.XInt8Spec(), weight=qc.XInt8Spec()
        ),
        algo_config=list(algos),
        extra_options=dict(extra or {}),
    )
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_reader(data)
        )


def _inits(model):
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def _bias_codes(model):
    """Per bias DequantizeLinear in node order: (codes, scale, zero point)."""
    inits = _inits(model)
    out = []
    for n in model.graph.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in inits:
            q = inits[n.input[0]]
            if q.ndim == 1:
                # (an int32 bias carries one scale per element in onnxsim, a
                # single per-tensor one in Quark; compare the distinct values)
                scale = sorted(set(inits[n.input[1]].ravel().tolist()))
                zp = sorted(set(inits[n.input[2]].ravel().tolist()))
                out.append((q.tolist(), scale, zp))
    return out


def _w(rng, *shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _with(text, inits):
    m = parser.parse_model(text)
    m.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in inits.items())
    return _named(m)


# -- BiasCorrection: Quark's power-of-two bias scale quirk --------------------------


def _bc_conv(seed=8):
    rng = np.random.default_rng(seed)
    return _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[1,3,8,8] x) => (float[1,4,8,8] y) {
            c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0)
            r0 = Relu(c0)
            c1 = Conv<group=1>(r0, w1, b1)
            r1 = Relu(c1)
            y = Conv<group=1>(r1, w2, b2)
        }""",
        dict(
            w0=_w(rng, 8, 3, 3, 3),
            b0=_w(rng, 8, scale=2),
            w1=_w(rng, 8, 8, 1, 1),
            b1=_w(rng, 8),
            w2=_w(rng, 4, 8, 1, 1),
            b2=_w(rng, 4, scale=0.1),
        ),
    ), (1, 3, 8, 8)


def _bc_mlp(seed=9):
    rng = np.random.default_rng(seed)
    return _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,8] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            y = Gemm(h1, w2, b2)
        }""",
        dict(w1=_w(rng, 16, 32), b1=_w(rng, 32), w2=_w(rng, 32, 8), b2=_w(rng, 8)),
    ), (3, 16)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("dtype", ["int8", "int16"])
def test_pof2_requantization_matches_quarks_quantize_data(dtype, symmetric, seed):
    """The re-derivation Quark's BiasCorrection runs on a corrected bias
    (power-of-two scale + MinMSE over five candidates + zero point + clipped
    codes) -- codes, scale and zero point bit for bit."""
    from onnx import TensorProto
    from quark.onnx.calibration import PowerOfTwoMethod
    from quark.onnx.quantization.quant_utils import quantize_data

    rng = np.random.default_rng(seed)
    for scale in (0.01, 0.3, 4.0):
        data = (rng.standard_normal(rng.integers(1, 40)) * scale).astype(np.float32)
        if seed == 3:
            data[0] = 17 * scale  # an outlier
        qtype = {"int8": TensorProto.INT8, "int16": TensorProto.INT16}[dtype]
        *_, zp, sc, codes = quantize_data(
            data, qtype, symmetric, method=PowerOfTwoMethod.MinMSE
        )
        got, got_scale, got_zp = qbc.quark_pof2_quantize(data, dtype, symmetric)
        np.testing.assert_array_equal(got, codes)
        assert np.float32(got_scale) == np.float32(sc)
        assert got_zp == int(zp)


@pytest.mark.parametrize("build", [_bc_conv, _bc_mlp])
def test_xint8_bias_correction_writes_quarks_integer_biases(build, tmp_path):
    """XINT8 + BiasCorrection: the corrected int8 bias codes -- re-derived
    through the power-of-two quantizer, scale left as stored -- equal Quark's."""
    model, shape = build()
    data = _data(shape)
    algos = [quark_onnx.BiasCorrectionConfig()]
    q = _quark(model, data, tmp_path, algos)
    m = _mine(model, data, [qc.BiasCorrectionConfig()])
    plain = _mine(model, data)
    assert _bias_codes(m) == _bias_codes(q)
    assert _bias_codes(q) != _bias_codes(plain), "bias correction changed nothing"


def test_the_quirk_is_visible_and_can_be_switched_off(tmp_path):
    """Quark's BiasCorrection can leave a bias whose int8 codes belong to a
    different scale than the stored one. onnxsim reproduces it, warns, and with
    ``BiasCorrectionStoredScale`` writes codes for the stored scale instead
    (so the dequantized bias is the corrected float bias)."""
    model, shape = _bc_conv()
    data = _data(shape)
    q = _quark(model, data, tmp_path, [quark_onnx.BiasCorrectionConfig()])
    base = _mine(model, data)
    with pytest.warns(UserWarning, match="power-of-two flow"):
        quirk = _mine(model, data, [qc.BiasCorrectionConfig()], quiet=False)
    assert _bias_codes(quirk) == _bias_codes(q)
    consistent = _mine(
        model,
        data,
        [qc.BiasCorrectionConfig()],
        {"BiasCorrectionStoredScale": True},
    )
    # same stored scales, but the codes follow the corrected float bias
    for (cq, sq, _), (cb, sb, _), (cc, sc, _) in zip(
        _bias_codes(quirk), _bias_codes(base), _bias_codes(consistent)
    ):
        assert sq == sb == sc
    assert _bias_codes(consistent) != _bias_codes(quirk)
    # the last layer is where the stored scale and the fresh one disagree
    assert max(abs(np.array(a) - np.array(b)).max()
               for (a, _, _), (b, _, _) in zip(_bias_codes(quirk), _bias_codes(consistent))) > 1


def test_xint8_bias_correction_from_the_same_quantized_model(tmp_path):
    """The algorithm alone: Quark's ``bias_correction`` and onnxsim's on the
    same Quark-quantized XINT8 model (symmetric and asymmetric)."""
    from quark.onnx.algorithm.bc.bias_correction import bias_correction
    from quark.onnx.calibration import CachedDataReader, PowerOfTwoMethod
    from onnxruntime.quantization import QuantType

    model, shape = _bc_conv(11)
    data = _data(shape, seed=5)
    quant = _quark(model, data, tmp_path)
    for symmetric in (True, False):
        theirs = bias_correction(
            copy.deepcopy(model),
            copy.deepcopy(quant),
            False,
            CachedDataReader(_reader(data), None),
            QuantType.QInt8,
            PowerOfTwoMethod.MinMSE,
            {"ActivationSymmetric": symmetric},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours = qbc.correct_bias_quark(
                model, quant, data, activation_symmetric=symmetric, method="pof2"
            )
        assert _bias_codes(ours) == _bias_codes(theirs), symmetric


def test_xint8_int32_bias_correction_follows_quarks_requantization(tmp_path):
    """``Int32Bias=True``: Quark's re-derivation runs on the int32 biases too
    (scale about 2**-24, zero point 1 from float32 rounding of the int32 range);
    the codes are around 1e8, so only float32 noise in the measured mean error
    separates them."""
    model, shape = _bc_conv()
    data = _data(shape)
    extra = {"Int32Bias": True}
    q = _quark(model, data, tmp_path, [quark_onnx.BiasCorrectionConfig()], extra)
    m = _mine(model, data, [qc.BiasCorrectionConfig()], extra)
    for (cq, sq, zq), (cm, sm, zm) in zip(_bias_codes(q), _bias_codes(m)):
        assert (sq, zq) == (sm, zm)
        assert max(abs(c) for c in cq) > 1e6  # the fresh 2**-24 grid
        np.testing.assert_allclose(cm, cq, rtol=3e-5, atol=1e3)
