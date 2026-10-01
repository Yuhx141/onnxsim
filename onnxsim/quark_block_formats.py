"""Fake-quantizers (quantize -> dequantize, float32 in / float32 out) for the
block floating point and microscaling formats of AMD Quark's ONNX flow.

Independent numpy implementations: Quark's CPU kernels (MIT) were read for
the *numeric definition* and our output is cross-checked against them
bit-for-bit in the tests' reference harness (not vendored here).

Formats:

- :func:`bfp16` -- "BFP16": per block, one shared 8-bit exponent (the block's
  maximum biased exponent) and a ``bit_width - 9`` bit signed mantissa per
  element (default: 16 bit, blocks of 8).
- :func:`bfp_prime` -- the shared-microexponent family Quark calls
  MX4 / MX6 / MX9 (``bit_width`` 11 / 13 / 16, blocks of 16, sub-blocks of 2
  with a 1-bit shift each).
- :func:`mx` -- OCP microscaling: blocks of 32 share a power-of-two scale
  ``2**(max_exp - emax)``; elements are int8, fp8 (e5m2 / e4m3), fp6
  (e3m2 / e2m3) or fp4 (e2m1).

All blocks run along ``axis`` (default 1, Quark's default); an axis length
that is not a multiple of the block size is zero-padded for the computation
and cropped afterwards. ``rounding`` is ``"std"`` (half away from zero),
``"dpu"`` (half up) or ``"py3"`` (half to even; Quark's default). The MX
floating-point element formats always round half to even, as in Quark.
Inf/NaN pass through unchanged in :func:`bfp16` and :func:`mx`.

Two deliberate differences from Quark's kernels: an axis length that is not
a block multiple is handled (Quark's kernel silently skips the remainder),
and NaN/Inf inputs are not special-cased in the shared-exponent search of
:func:`bfp_prime` beyond what Quark does (an Inf/NaN in a block yields an
all-NaN block there too).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

_EXP_MASK = 0x7F800000


def _bits(x: np.ndarray) -> np.ndarray:
    return x.view(np.uint32).astype(np.int64)


def _from_bits(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32).view(np.float32)


def _exp(x: np.ndarray) -> np.ndarray:
    """Biased exponent field (0..255) of a float32 array."""
    return (_bits(x) >> 23) & 0xFF


def _round(v: np.ndarray, mode: str) -> np.ndarray:
    if mode == "py3":
        return np.round(v)  # numpy rounds half to even
    if mode == "std":
        return np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))
    if mode == "dpu":
        return np.floor(v + np.float32(0.5))
    raise ValueError(f"unknown rounding mode {mode!r}")


def _blockify(
    x: np.ndarray, axis: int, block_size: int
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Return ``x`` as ``[..., n_blocks, block_size]`` (axis moved last,
    zero-padded) plus the inverse mapping back to ``x``'s shape."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 0:
        raise ValueError("scalar input has no block axis")
    axis = axis % x.ndim
    moved = np.moveaxis(x, axis, -1)
    n = moved.shape[-1]
    pad = (-n) % block_size
    if pad:
        moved = np.concatenate(
            [moved, np.zeros(moved.shape[:-1] + (pad,), np.float32)], axis=-1
        )
    blocks = moved.reshape(moved.shape[:-1] + (-1, block_size))

    def restore(y: np.ndarray) -> np.ndarray:
        y = y.reshape(y.shape[:-2] + (-1,))[..., :n]
        return np.moveaxis(y, -1, axis)

    return blocks, restore


def _shared_exp_finite(blocks: np.ndarray) -> np.ndarray:
    """Per-block max biased exponent, Inf/NaN (exponent 255) counted as 0."""
    e = _exp(blocks)
    return np.where(e == 255, 0, e).max(axis=-1, keepdims=True)


def bfp16(
    x,
    bit_width: int = 16,
    block_size: int = 8,
    axis: int = 1,
    rounding: str = "py3",
) -> np.ndarray:
    """Block floating point fake-quantization (see module docstring)."""
    blocks, restore = _blockify(x, axis, block_size)
    shared = _shared_exp_finite(blocks)
    m_bits = bit_width - 9
    scale = np.exp2((shared - 127 - (m_bits - 1)).astype(np.float64)).astype(np.float32)
    max_v = (np.exp2((shared - 127 + 1).astype(np.float64)) - scale).astype(np.float32)
    with np.errstate(all="ignore"):
        q = _round(blocks / scale, rounding) * scale
        out = np.maximum(-max_v, np.minimum(q, max_v)).astype(np.float32)
    out = np.where(_exp(blocks) == 255, blocks, out)
    return restore(out)


def _round_bits(
    sign_neg: np.ndarray,
    x: np.ndarray,
    tail: np.ndarray,
    upper: int,
    mode: str,
) -> np.ndarray:
    """Vectorized ``round_bits``: drop ``tail`` low bits of ``x`` with
    rounding, never rounding up past ``upper``."""
    t = np.clip(tail, 0, 40)
    ret = x >> np.minimum(t, 62)
    half = np.where(t > 0, np.left_shift(1, np.maximum(t - 1, 0)), 0)
    rem = x & (np.left_shift(1, t) - 1)
    if mode == "std":
        tie = ret + 1
    elif mode == "dpu":
        tie = np.where(sign_neg, ret, ret + 1)
    elif mode == "py3":
        tie = np.where(ret % 2 == 1, ret + 1, ret)
    else:
        raise ValueError(f"unknown rounding mode {mode!r}")
    rounded = np.where(rem < half, ret, np.where(rem > half, ret + 1, tie))
    rounded = np.where(ret == upper, ret, rounded)
    rounded = np.where(tail == 0, x, rounded)  # nothing to drop: unchanged
    return np.where(tail > 25, 0, rounded)


def bfp_prime(
    x,
    bit_width: int = 13,
    block_size: int = 16,
    sub_block_size: int = 2,
    sub_block_shift_bits: int = 1,
    axis: int = 1,
    rounding: str = "py3",
) -> np.ndarray:
    """Shared-microexponent block format (Quark's MX4 / MX6 / MX9)."""
    if block_size % sub_block_size:
        raise ValueError("block_size must be a multiple of sub_block_size")
    blocks, restore = _blockify(x, axis, block_size)
    m_bfp = bit_width - 9
    bits = _bits(blocks)
    exp = (bits >> 23) & 0xFF
    shared = exp.max(axis=-1, keepdims=True)  # raw: Inf/NaN (255) included

    sub = exp.reshape(exp.shape[:-1] + (-1, sub_block_size))
    sub_max = sub.max(axis=-1, keepdims=True)
    bound = (1 << sub_block_shift_bits) - 1
    shift = np.minimum(shared[..., None, :] - sub_max, bound)  # [..., nsub, 1]
    shift = np.broadcast_to(shift, sub.shape).reshape(exp.shape)

    mant = np.where(exp == 0, 0, (bits & 0x7FFFFF) | (1 << 23))
    tail = shared - shift - exp + 23 - m_bfp + 1
    neg = (bits >> 31) == 1
    mant = _round_bits(neg, mant, tail, (1 << m_bfp) - 1, rounding)
    sign = np.where(neg, -1.0, 1.0)
    pow2 = np.exp2((shared - 127 - shift + 1 - m_bfp).astype(np.float64))
    with np.errstate(over="ignore", invalid="ignore"):
        out = (sign * pow2 * mant.astype(np.float64)).astype(np.float32)
    out = np.where(shared == 255, np.float32(np.nan), out)
    return restore(out)


# element format -> (ebits, mbits, emax, max_norm, min_norm)
_MX_FORMATS: Dict[str, Tuple[int, int, int, float, float]] = {
    "fp8_e5m2": (5, 2, 15, 57344.0, -57344.0),
    "fp8_e4m3": (4, 3, 8, 448.0, -448.0),
    "fp6_e3m2": (3, 2, 4, 28.0, -28.0),
    "fp6_e2m3": (2, 3, 2, 7.5, -7.5),
    "fp4_e2m1": (2, 1, 2, 6.0, -6.0),
    "int8": (0, 8, 0, 127.0, -128.0),
}


def _fake_quantize_minifloat(
    e: np.ndarray, max_norm: float, ebits: int, mbits: int
) -> np.ndarray:
    """Round float32 values to a sign/ebits/mbits minifloat (half to even),
    saturating at ``max_norm``; Inf/NaN pass through."""
    bits = _bits(e)
    exp = (bits >> 23) & 0xFF
    new_bias = (1 << (ebits - 1)) - 1
    mant = bits & 0x7FFFFF
    full = exp != 0
    mant = np.where(full, mant | (1 << 23), mant)
    mant_bits = np.where(full, 24, 23)

    new_exp = exp - 127 + new_bias
    exp_shift = np.where(new_exp > 0, 0, 1 - new_exp)
    tail = 23 - mbits + exp_shift

    t = np.clip(tail, 0, 40)
    ret = mant >> np.minimum(t, 62)
    half = np.where(t > 0, np.left_shift(1, np.maximum(t - 1, 0)), 0)
    rem = mant & (np.left_shift(1, t) - 1)
    rounded = np.where(
        rem < half,
        ret,
        np.where(rem > half, ret + 1, np.where(ret % 2 == 1, ret + 1, ret)),
    )
    rounded = np.where(tail == 0, mant, rounded)
    rounded = np.where(tail > 25, 0, rounded)

    zero = rounded == 0
    shifted = rounded << np.clip(tail, 0, 40)
    overflow = shifted >= (1 << mant_bits)
    shifted = np.where(overflow & full, shifted >> 1, shifted)
    exp2 = np.where(overflow, exp + 1, exp)
    mag = _from_bits((exp2 << 23) + (shifted & 0x7FFFFF))
    mag = np.minimum(mag, np.float32(max_norm))
    out = _from_bits(_bits(mag) + ((bits >> 31) << 31))
    out = np.where(zero, np.float32(0.0), out)
    return np.where(exp == 255, e, out).astype(np.float32)


def mx(
    x,
    element_dtype: str = "int8",
    block_size: int = 32,
    axis: int = 1,
    rounding: str = "py3",
) -> np.ndarray:
    """OCP microscaling fake-quantization (see module docstring)."""
    if element_dtype not in _MX_FORMATS:
        raise ValueError(
            f"unknown element_dtype {element_dtype!r}; known: {sorted(_MX_FORMATS)}"
        )
    ebits, mbits, emax, max_norm, min_norm = _MX_FORMATS[element_dtype]
    blocks, restore = _blockify(x, axis, block_size)
    shared = _shared_exp_finite(blocks)
    scale = np.exp2((shared - 127 - emax).astype(np.float64)).astype(np.float32)
    with np.errstate(all="ignore"):
        element = blocks / scale
        if ebits > 0:
            q = _fake_quantize_minifloat(element, max_norm, ebits, mbits)
        else:
            implicit = np.float32(2.0**-6)
            q = _round(element / implicit, rounding)
            q = np.maximum(np.float32(min_norm), np.minimum(q, np.float32(max_norm)))
            q = (q * implicit).astype(np.float32)
        out = (scale * q).astype(np.float32)
    out = np.where(_exp(blocks) == 255, blocks, out)
    return restore(out)


__all__ = ["bfp16", "bfp_prime", "mx"]
