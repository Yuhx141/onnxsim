"""apply_adaquant on quantize_full_qdq output (not just quantize_static's).

Two things used to make it silently do nothing / do harm there:
- full_qdq renames a layer's pre-requantize tensor ``<name>/f``, so the
  float and quantized layers never matched by output name;
- the C++ port read a raw-data UINT8 activation zero point as int8, so a
  zero point >= 128 came back negative and was clamped to 0, clipping the
  negative half of the activation range.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim.adaquant import apply_adaquant
from onnxsim.full_qdq import quantize_full_qdq


def _model():
    rng = np.random.default_rng(0)
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,16] x) => (float[N,16] y)
        {
            h = MatMul(x, w1)
            t = Tanh(h)
            y = MatMul(t, w2)
        }
        """
    )
    # Random weights are attached programmatically (too large for text literals).
    model.graph.initializer.extend(
        numpy_helper.from_array(rng.standard_normal((16, 16)).astype(np.float32), n)
        for n in ("w1", "w2")
    )
    return model


def _run(model, feed):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feed)[0]


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(1)
    calib = [{"x": rng.standard_normal((8, 16)).astype(np.float32)} for _ in range(16)]
    test = {"x": rng.standard_normal((64, 16)).astype(np.float32)}
    float_model = _model()
    quantized = quantize_full_qdq(float_model, calibration_data=calib)
    adaquant = apply_adaquant(
        float_model, quantized, calibration_data=calib, num_iterations=50
    )
    return float_model, quantized, adaquant, test


def test_layers_are_matched_and_changed(setup):
    _, quantized, adaquant, _ = setup
    assert adaquant.SerializeToString() != quantized.SerializeToString()


def test_activation_zero_point_is_not_clobbered(setup):
    _, quantized, adaquant, _ = setup
    before = {t.name: numpy_helper.to_array(t) for t in quantized.graph.initializer}
    after = {t.name: numpy_helper.to_array(t) for t in adaquant.graph.initializer}
    zp_names = [n for n, a in before.items() if a.dtype == np.uint8 and a.size == 1]
    assert zp_names
    for name in zp_names:
        # The learned clip range may nudge the zero point, never reset it.
        assert abs(int(after[name]) - int(before[name])) <= 16, name


def test_accuracy_does_not_regress_badly(setup):
    float_model, quantized, adaquant, test = setup
    ref = _run(float_model, test)
    mse_before = float(((_run(quantized, test) - ref) ** 2).mean())
    mse_after = float(((_run(adaquant, test) - ref) ** 2).mean())
    assert mse_after < 2 * mse_before
    onnx.checker.check_model(adaquant)
