"""Tests for onnxsim.quark_block_formats (BFP16 / MX4-6-9 / MX fake-quantizers).

The expected values below are derived by hand from each format's definition
(shared exponent, element grid, saturation). Separately, the implementations
were cross-checked bit-for-bit against Quark's own compiled CPU kernels on
random and edge-case data (all rounding modes, with and without Inf/NaN); that
reference harness is not vendored here.
"""

import numpy as np
import pytest

from onnxsim import quark_block_formats as qbf


def f32(*v):
    return np.array(v, dtype=np.float32)


# -- bfp16: shared exponent = block max, 7 mantissa bits (scale 2**(e-6)) ------


def test_bfp16_scale_follows_the_block_max():
    # max 1.0 -> shared exponent 0 -> grid step 1/64: 0.1 -> round(6.4)/64
    x = f32(1.0, 0.1, 0, 0, 0, 0, 0, 0)
    assert qbf.bfp16(x, axis=0)[:2].tolist() == [1.0, 0.09375]
    # max 100 -> shared exponent 6 -> grid step 1: small values vanish
    y = f32(100.0, 1.0, 0.1, 0, 0, 0, 0, 0)
    assert qbf.bfp16(y, axis=0)[:3].tolist() == [100.0, 1.0, 0.0]


def test_bfp16_saturates_just_below_the_next_power_of_two():
    x = f32(1.999, 0, 0, 0, 0, 0, 0, 0)  # 1.999 * 64 = 127.9 -> 128 -> clamp
    assert qbf.bfp16(x, axis=0)[0] == np.float32(2.0 - 1.0 / 64)


@pytest.mark.parametrize(
    "mode, expected",
    [("std", [1.0, -1.0]), ("dpu", [1.0, -0.0]), ("py3", [0.0, -0.0])],
)
def test_bfp16_rounding_modes_on_ties(mode, expected):
    # shared exponent 6 (the 64) -> step 1, so 0.5 / -0.5 are exact ties
    x = f32(64.0, 0.5, -0.5, 0, 0, 0, 0, 0)
    out = qbf.bfp16(x, axis=0, rounding=mode)
    np.testing.assert_array_equal(out[1:3] + 0.0, np.array(expected, np.float32) + 0.0)


def test_bfp16_inf_and_nan_pass_through():
    x = f32(1.0, np.inf, -np.inf, np.nan, 0, 0, 0, 0)
    out = qbf.bfp16(x, axis=0)
    assert out[0] == 1.0 and out[1] == np.inf and out[2] == -np.inf
    assert np.isnan(out[3])


# -- bfp_prime (Quark MX4/6/9) ---------------------------------------------------


def test_bfp_prime_exact_values_and_microexponent_shift():
    # bit_width 16 -> 7 mantissa bits. A block of ones is represented exactly;
    # a sub-block two binades smaller gets a 1-bit shift (capped at 1), and
    # 0.25 is still exact.
    x = np.ones(16, np.float32)
    np.testing.assert_array_equal(qbf.bfp_prime(x, bit_width=16, axis=0), x)
    y = np.ones(16, np.float32)
    y[2:4] = 0.25
    np.testing.assert_array_equal(qbf.bfp_prime(y, bit_width=16, axis=0), y)


def test_bfp_prime_inf_poisons_the_whole_block():
    x = np.ones(16, np.float32)
    x[3] = np.inf
    assert np.isnan(qbf.bfp_prime(x, axis=0)).all()


# -- mx -------------------------------------------------------------------------


def test_mx_fp4_e2m1_grid_and_ties_to_even():
    # block max 6.0 -> shared scale 2**(2-2) = 1; e2m1 grid: 0 .5 1 1.5 2 3 4 6
    x = np.zeros(32, np.float32)
    x[:5] = [6.0, 0.7, 5.0, 2.4, -3.4]
    out = qbf.mx(x, element_dtype="fp4_e2m1", axis=0)
    # 5.0 is a tie between 4 and 6 -> even mantissa (4); -3.4 -> -3
    assert out[:5].tolist() == [6.0, 0.5, 4.0, 2.0, -3.0]


def test_mx_int8_grid_and_saturation():
    # block max 1.999 -> shared scale 1, int8 elements with implicit 2**-6
    x = np.zeros(32, np.float32)
    x[:3] = [1.999, 1.5, -0.1]
    out = qbf.mx(x, element_dtype="int8", axis=0)
    assert out[0] == np.float32(127 / 64)  # 127.9 saturates at 127
    assert out[1] == 1.5
    assert out[2] == np.float32(-6 / 64)  # -6.4 -> -6


def test_mx_fp8_e4m3_saturates_at_448_times_scale():
    x = np.zeros(32, np.float32)
    x[0] = 1000.0  # exp 9, emax 8 -> scale 2 -> 500 -> clamp 448 -> 896
    assert qbf.mx(x, element_dtype="fp8_e4m3", axis=0)[0] == 896.0


# -- shared behaviour -------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, kwargs",
    [
        (qbf.bfp16, {}),
        (qbf.bfp_prime, {}),
        (qbf.mx, {"element_dtype": "fp6_e3m2"}),
        (qbf.mx, {"element_dtype": "int8"}),
    ],
)
def test_axis_padding_and_idempotence(fn, kwargs):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 21)).astype(np.float32)  # 21: not a block multiple
    out = fn(x, axis=1, **kwargs)
    assert out.shape == x.shape and out.dtype == np.float32
    # blocking along axis 0 of the transpose is the same computation
    np.testing.assert_array_equal(fn(x.T, axis=0, **kwargs).T, out)
    # a quantized tensor is a fixed point
    np.testing.assert_array_equal(fn(out, axis=1, **kwargs), out)


def test_quantization_error_is_bounded_and_shrinks_with_bits():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 64)).astype(np.float32)
    err = {bw: np.abs(qbf.bfp16(x, bit_width=bw) - x).max() for bw in (11, 13, 16)}
    assert err[16] < err[13] < err[11]
    assert err[16] < 0.05


def test_validation_errors():
    with pytest.raises(ValueError, match="element_dtype"):
        qbf.mx(f32(1, 2), element_dtype="fp3")
    with pytest.raises(ValueError, match="rounding"):
        qbf.bfp16(f32(1, 2), rounding="up")
    with pytest.raises(ValueError, match="multiple"):
        qbf.bfp_prime(f32(1, 2), block_size=16, sub_block_size=3)
    with pytest.raises(ValueError, match="scalar"):
        qbf.bfp16(np.float32(1.0))


# -- fp16 / bf16 rounding -------------------------------------------------------------


def test_fp16_round_values_ties_and_overflow():
    x = np.array([0.1, 1.0, 65504.0, 65520.0, -1e9, 2**-24, 2**-26], np.float32)
    out = qbf.fp16_round(x)
    assert out[0] == np.float32(np.float16(0.1)) and out[0] != np.float32(0.1)
    assert out[1] == 1.0 and out[2] == 65504.0
    assert out[3] == np.inf and out[4] == -np.inf  # overflow -> inf, as a Cast does
    assert out[5] == np.float32(2**-24) and out[6] == 0.0  # subnormal / underflow
    assert out.dtype == np.float32


def test_bf16_round_keeps_float32_range_and_rounds_ties_to_even():
    x = np.array([0.1, 1.0, 3e38, np.inf, -np.inf, np.nan], np.float32)
    out = qbf.bf16_round(x)
    assert out[1] == 1.0 and out[3] == np.inf and out[4] == -np.inf
    assert np.isnan(out[5])
    assert abs(out[0] - 0.1) < 0.1 * 2**-8 and out[0] != np.float32(0.1)
    # 1 + 2**-8 is exactly halfway between two bf16 values: ties to even -> 1.0
    assert qbf.bf16_round(np.float32(1 + 2**-8)) == 1.0
    # 1 + 3 * 2**-8 is halfway as well, and the even neighbour is 1 + 2**-6
    assert qbf.bf16_round(np.float32(1 + 3 * 2**-8)) == np.float32(1 + 2**-6)
    assert qbf.bf16_round(np.float32(3.4e38)) == np.inf  # max float32 rounds up to inf


def test_bf16_round_matches_ml_dtypes_on_random_data():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(5000) * np.exp2(rng.integers(-30, 30, 5000))).astype(
        np.float32
    )
    expected = x.astype(ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_array_equal(qbf.bf16_round(x), expected)


def test_half_rounding_is_idempotent():
    x = np.random.default_rng(1).standard_normal(1000).astype(np.float32)
    for fn in (qbf.fp16_round, qbf.bf16_round):
        np.testing.assert_array_equal(fn(fn(x)), fn(x))
