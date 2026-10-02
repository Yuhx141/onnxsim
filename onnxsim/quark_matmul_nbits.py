"""Quark's ``MATMUL_NBITS`` preset: weight-only 4-bit quantization that
rewrites constant-weight ``MatMul`` nodes into ``com.microsoft::MatMulNBits``.

Mirrors ``quark.onnx.quantizers.matmul_nbits_quantizer`` (amd-quark 0.13) and
reproduces its emitted graph and its numbers:

- **Which nodes**: every ``MatMul`` whose input 1 is a constant 2-D ``[K, N]``
  initializer (float32; float16 is computed in float32 and its scales cast
  back), in the main graph and, recursively, in control-flow subgraphs.
  ``Gemm`` is never converted, nor are ``MatMul`` with a 1-D / 3-D+ weight or a
  constant ``A``. The rewritten node keeps its output tensor, is named
  ``<node name>_Q4`` (``""`` when the node has no name) and stays in place.
- **Emission** (``algorithm="DEFAULT"``): inputs ``A, <w>_Q4, <w>_scales[,
  <w>_zero_points]`` with ``<w>_Q4`` uint8 ``[N, ceil(K / block), block / 2]``,
  ``<w>_scales`` float ``[N * ceil(K / block)]`` (flat, column major) and, when
  not symmetric, ``<w>_zero_points`` uint8 ``[N * ceil(ceil(K / block) / 2)]``
  (flat). Attributes ``K``, ``N``, ``bits``, ``block_size`` and
  ``accuracy_level`` (``0`` unless given; the preset sets ``1``); domain
  ``com.microsoft`` (opset 1 is added to the model's imports).
- **Numbers**: the block quantizer is ONNX Runtime's
  ``quantize_matmul_4bits`` (MLAS) re-implemented in numpy and bit-identical to
  it, including its quirks: blocks run along ``K`` per output column; the last
  block is zero padded (padding takes no part in min / max); two codes per byte,
  **even ``k`` in the low nibble**; symmetric: ``scale = max_abs_signed / -8``
  (the signed extreme of larger magnitude, ``vmin`` on ties), zero point 8,
  ``code = floor(w / scale + 8 + 0.5)`` clipped to ``[0, 15]``; asymmetric:
  ``scale = (max(vmax, 0) - min(vmin, 0)) / 15``,
  ``zero point = floor(-vmin / scale + 0.5)``. A weight shared by several
  MatMuls is converted once and the nodes share its tensors.
- ``algorithm="HQQ"`` (half-quadratic quantization): per-block min/max grid,
  20 iterations of Quark's lp-norm shrinkage on the zero point; always emits
  **float** zero points (``[N * blocks]``, unpacked) and no ``accuracy_level``,
  as Quark does. Quark computes it in torch float32; here it is numpy float32,
  so zero points can differ from Quark's by float rounding (last ulp) and, very
  rarely, a code at an exact rounding tie.
- ``algorithm="GPTQ"``: Quark's ``GptqProcessor.apply_matmul4bits``. The Hessian
  is built from the **first calibration batch only** (the MatMul's input,
  captured with ONNX Runtime), the weights are rounded on Quark's GPTQ grid
  (``GPTQParams``: ``GroupSize`` -- default ``-1``, i.e. one block of ``K`` --,
  ``PerChannel`` (default ``False``), ``WeightSymmetric`` (default ``True``; the
  *packing* symmetry too -- ``MatMulNBitsParams.Symmetric`` is ignored),
  ``MSE``, ``ActOrder``, ``BlockSize``, ``PercDamp``), and the dequantized
  values are then **re-quantized on a new block-wise grid** derived from their
  range, which is what gets packed. Scales / zero points are 2-D here
  (``[N, blocks]`` / ``[N, ceil(blocks / 2)]``) and no ``accuracy_level`` is
  written. Quark 0.13's GPTQ error propagation is a no-op (see
  :func:`onnxsim.quark_weight_rounding.quark_gptq`), so by default so is ours,
  and the output matches Quark bit for bit; ``GPTQParams["Compensate"] = True``
  (onnxsim only) really propagates the error with
  :func:`onnxsim.quark_weight_rounding.quark_gptq`.

Deliberate differences from Quark:

- Quark first runs ONNX Runtime graph optimizations over the model
  (``SkipPreprocess=False``, its default), whose ``MatMulAddFusion`` turns
  ``MatMul`` + ``Add`` into a ``Gemm`` that is then *not* quantized. onnxsim
  converts those MatMuls too (the ``Add`` stays); this equals Quark with
  ``extra_options={"SkipPreprocess": True}``, which the parity tests use. The
  other things its pre-processing does (shape inference, opset-import
  bloat, ``Gemm`` renames) are not reproduced.
- ``exclude`` / ``nodes_to_exclude`` is honoured (Quark's preset drops it).
- ``Bits`` other than 4 raise: Quark's DEFAULT path always packs 4-bit codes
  but writes ``bits`` as the attribute, which yields a model ONNX Runtime
  rejects. ``block_size`` must be a power of two >= 16 (the MatMulNBits
  kernel's requirement; Quark would emit the node anyway).
- GPTQ needs every converted weight to feed a single MatMul (Quark crashes on
  shared weights) and only converts main-graph nodes (as Quark does).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

_MS_DOMAIN = "com.microsoft"
_F32 = np.float32


@dataclass
class MatMulNBitsReport:
    """What :func:`quantize_matmul_nbits` did."""

    converted: List[str] = field(default_factory=list)  # new node names
    skipped: List[str] = field(default_factory=list)  # MatMul names left alone


# -- the block quantizer (ONNX Runtime's quantize_matmul_4bits) ------------------


def _check_block_size(block_size: int) -> None:
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError(
            f"block_size must be a power of two >= 16 (MatMulNBits), got {block_size}"
        )


def pack_int4(codes: np.ndarray) -> np.ndarray:
    """Packs unsigned 4-bit ``codes`` along the last axis (even length),
    even index in the low nibble: the MatMulNBits layout."""
    c = np.asarray(codes, dtype=np.uint8)
    if c.shape[-1] % 2:
        raise ValueError("the packed axis must have even length")
    return (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_int4`."""
    p = np.asarray(packed, dtype=np.uint8)
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = p & 0xF
    out[..., 1::2] = p >> 4
    return out


def block_quantize_int4(
    w_kn: np.ndarray, block_size: int, symmetric: bool
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """4-bit block-wise quantization of a ``[K, N]`` weight, bit-identical to
    ONNX Runtime's ``quantize_matmul_4bits`` (what Quark's DEFAULT algorithm
    calls).

    Returns ``(packed, scales, zero_points)``: ``packed`` uint8
    ``[N, blocks, block_size // 2]``, ``scales`` ``[N * blocks]`` and (not
    symmetric) ``zero_points`` uint8 ``[N * ceil(blocks / 2)]``, else ``None``.
    """
    _check_block_size(block_size)
    w = np.asarray(w_kn)
    if w.ndim != 2:
        raise ValueError("expected a 2-D [K, N] weight")
    out_dtype = w.dtype
    k, n = w.shape
    blocks = (k + block_size - 1) // block_size
    padded = np.full((n, blocks * block_size), np.nan, dtype=_F32)
    padded[:, :k] = w.T.astype(_F32)
    b = padded.reshape(n, blocks, block_size)
    vmin = np.nanmin(b, axis=2)
    vmax = np.nanmax(b, axis=2)
    if symmetric:
        extreme = np.where(np.abs(vmax) > np.abs(vmin), vmax, vmin)
        scale = (extreme / _F32(-8)).astype(_F32)
        zp = np.full(scale.shape, 8, dtype=_F32)
    else:
        vmin = np.minimum(vmin, _F32(0))
        vmax = np.maximum(vmax, _F32(0))
        scale = ((vmax - vmin) / _F32(15)).astype(_F32)
        with np.errstate(all="ignore"):
            zp_f = np.where(scale != 0, -vmin / scale, _F32(0)).astype(_F32)
        zp = np.clip(np.floor(zp_f + _F32(0.5)), 0, 15)
    with np.errstate(all="ignore"):
        recip = np.where(scale != 0, _F32(1) / scale, _F32(0)).astype(_F32)
    t = np.nan_to_num(b, nan=0.0) * recip[..., None] + zp[..., None].astype(_F32)
    codes = np.clip(np.floor(t + _F32(0.5)), 0, 15).astype(np.uint8)
    # padding rows quantize to code 0 (the kernel never writes them)
    valid = np.zeros((n, blocks * block_size), dtype=bool)
    valid[:, :k] = True
    codes = np.where(valid.reshape(n, blocks, block_size), codes, 0).astype(np.uint8)
    # The MLAS kernel quantizes rows in pairs; with an odd number of rows left
    # in the last block the unused high nibble keeps the previous pair's second
    # code (scratch the kernel never reads: reproduced for >= 3 rows left, left
    # 0 for a single leftover row, where ORT leaves a value carried over from
    # the previously processed block).
    last = k - (blocks - 1) * block_size
    if last % 2 and last >= 3:
        codes[:, -1, last] = codes[:, -1, last - 2]
    packed = pack_int4(codes)
    zero_points = None
    if not symmetric:
        zero_points = _pack_zero_points(zp.astype(np.uint8), 8).reshape(-1)
    return packed, scale.reshape(-1).astype(out_dtype), zero_points


def _pack_zero_points(zp_nb: np.ndarray, pad_value: int) -> np.ndarray:
    """``[N, blocks]`` codes -> ``[N, ceil(blocks / 2)]`` bytes, low nibble
    first (an odd tail is padded with ``pad_value``)."""
    if zp_nb.shape[1] % 2:
        zp_nb = np.pad(zp_nb, ((0, 0), (0, 1)), constant_values=pad_value)
    return pack_int4(zp_nb).astype(np.uint8)


def dequantize_int4(
    packed: np.ndarray,
    scales: np.ndarray,
    zero_points: Optional[np.ndarray],
    k: int,
    n: int,
    block_size: int,
) -> np.ndarray:
    """Dequantizes the (DEFAULT-layout) tensors of a MatMulNBits node back to
    the float ``[K, N]`` weight: ``scale * (code - zero_point)``."""
    blocks = (k + block_size - 1) // block_size
    codes = unpack_int4(packed.reshape(n, blocks, block_size // 2)).reshape(
        n, blocks, block_size
    )
    s = np.asarray(scales, dtype=np.float64).reshape(n, blocks, 1)
    if zero_points is None:
        zp = np.full((n, blocks, 1), 8.0)
    else:
        zp = (
            unpack_int4(np.asarray(zero_points, np.uint8).reshape(n, -1))[:, :blocks]
            .astype(np.float64)
            .reshape(n, blocks, 1)
        )
    w = (s * (codes.astype(np.float64) - zp)).reshape(n, blocks * block_size)
    return w[:, :k].T.astype(np.float32)


def _row_mean_f32(x: np.ndarray) -> np.ndarray:
    """``torch.mean(x, axis=1, keepdim=True)`` for a contiguous float32
    ``[M, n]`` (``n`` a multiple of 8) with torch's CPU summation order: 8 SIMD
    lanes, four interleaved accumulators over the row, accumulators added in
    sequence, then the 8 lanes added left to right."""
    m, n = x.shape
    accs = [np.zeros((m, 8), _F32) for _ in range(4)]
    for c in range(n // 8):
        accs[c % 4] = (accs[c % 4] + x[:, c * 8 : (c + 1) * 8]).astype(_F32)
    acc = accs[0]
    for a in accs[1:]:
        acc = (acc + a).astype(_F32)
    total = acc[:, 0].copy()
    for j in range(1, 8):
        total = (total + acc[:, j]).astype(_F32)
    return (total / _F32(n)).astype(_F32)[:, None]


# -- HQQ ---------------------------------------------------------------------------


def hqq_quantize(
    w_kn: np.ndarray, block_size: int, bits: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quark's ``HQQWeightOnlyQuantizer`` on a ``[K, N]`` weight, in float32:
    ``(packed [N, blocks, block // 2] uint8, scales [N * blocks],
    zero_points [N * blocks])`` (the zero points are floats, not packed)."""
    _check_block_size(block_size)
    if bits != 4:
        raise NotImplementedError("HQQ packing here is 4-bit only (as Quark's)")
    w = np.asarray(w_kn, dtype=_F32).T  # [N, K]
    n, k = w.shape
    pad = (block_size - k % block_size) % block_size
    w = np.pad(w, ((0, 0), (0, pad)))
    shape = w.shape
    wg = w.reshape(-1, block_size)
    wmin = wg.min(axis=1, keepdims=True)
    wmax = wg.max(axis=1, keepdims=True)
    max_v = 2**bits - 1
    with np.errstate(all="ignore"):
        # torch evaluates ``int / tensor`` as ``reciprocal(tensor) * int``
        scale = np.minimum(
            (_F32(1) / (wmax - wmin)).astype(_F32) * _F32(max_v), _F32(2e4)
        ).astype(_F32)
    zero = np.round(-wmin * scale).astype(_F32)

    lp_norm, beta, kappa, iters = 0.7, 1e1, 1.01, 20
    best = 1e4
    for _ in range(iters):
        w_q = np.clip(np.round(wg * scale + zero), 0, max_v).astype(_F32)
        w_r = ((w_q - zero) / scale).astype(_F32)
        x = (wg - w_r).astype(_F32)
        mag = np.abs(x)
        shrunk = mag - _F32(1.0 / beta) * np.power(mag + _F32(1e-8), _F32(lp_norm - 1))
        w_e = (np.sign(x) * np.maximum(shrunk, _F32(0))).astype(_F32)
        zero = _row_mean_f32((w_q - (wg - w_e) * scale).astype(_F32))
        beta *= kappa
        err = float(np.abs(wg - w_r).astype(_F32).mean(dtype=np.float64))
        if err < best:
            best = err
        else:
            break
    codes = (
        np.clip(np.round(wg * scale + zero), 0, max_v).reshape(shape).astype(np.uint8)
    )
    blocks = (k + block_size - 1) // block_size
    packed = pack_int4(codes).reshape(n, blocks, block_size // 2)
    scales = (_F32(1.0) / scale).astype(_F32).reshape(-1)
    return packed, scales, zero.astype(_F32).reshape(-1)


# -- GPTQ (Quark's GptqProcessor.apply_matmul4bits) --------------------------------


def _find_params(
    x: np.ndarray,
    maxq: np.ndarray,
    per_channel: bool,
    sym: bool,
    mse: bool,
    norm: float = 2.4,
    grid: int = 100,
    max_shrink: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """``GPTQ.find_params`` of ``quark.onnx.algorithm.gptq``, transcribed with
    its dtype behaviour (float32 min / max, float64 scale)."""
    shape = x.shape
    if not per_channel:
        x = np.expand_dims(x.flatten(), axis=1)
    tmp = np.zeros(x.shape[1], dtype=x.dtype)
    xmin = np.minimum(np.min(x, axis=0), tmp)
    xmax = np.maximum(np.max(x, axis=0), tmp)
    if sym:
        xmax = np.maximum(np.abs(xmin), xmax)
        neg = xmin < 0
        if np.any(neg):
            xmin[neg] = -xmax[neg]
    both_zero = (xmin == 0) & (xmax == 0)
    xmin[both_zero] = -1
    xmax[both_zero] = +1
    scale = (xmax - xmin) / maxq
    zero = np.ones(scale.shape) * (maxq + 1) / 2 if sym else np.round(-xmin / scale)
    if mse:
        best = np.full([x.shape[1]], float("inf"))
        for i in range(int(max_shrink * grid)):
            p = 1 - i / grid
            xmin1 = p * xmin
            xmax1 = p * xmax
            scale1 = (xmax1 - xmin1) / maxq
            zero1 = np.round(-xmin1 / scale1) if not sym else zero
            q = np.clip(
                np.round(x / np.expand_dims(scale1, axis=0))
                + np.expand_dims(zero1, axis=0),
                0,
                maxq,
            ).astype(x.dtype)
            q = np.expand_dims(scale1, axis=0) * (q - np.expand_dims(zero1, axis=0))
            q -= x
            q = np.abs(q)
            q = np.power(q, norm)
            err = np.sum(q, 0)
            better = err < best
            if np.any(better):
                best[better] = err[better]
                scale[better] = scale1[better]
                zero[better] = zero1[better]
    if not per_channel:
        scale = np.repeat(scale, shape[1])
        zero = np.repeat(zero, shape[1])
    return scale, zero


def _hessian(act: np.ndarray, k: int) -> np.ndarray:
    """``GPTQ.add_batch`` for one batch: ``2 / n * X^T X`` with ``n`` the
    batch (leading) dimension, the product taken in float32."""
    n = act.shape[0]
    x = np.reshape(act, (-1, k))
    x = math.sqrt(2 / n) * x.astype(np.float32)
    h = np.zeros((k, k))
    h += np.matmul(np.transpose(x), x)
    return h


def _gptq_rtn(
    w: np.ndarray,
    h: np.ndarray,
    *,
    group_size: int,
    per_channel: bool,
    sym: bool,
    mse: bool,
    act_order: bool,
) -> np.ndarray:
    """What Quark 0.13's ``GPTQ.fasterquant(bits=4)`` returns as the dequantized
    weights: the error-propagation step is a no-op there (it slices an upper
    triangular factor by column), so this is round-to-nearest on its grid, with
    its per-group parameters and activation-order permutation."""
    maxq = np.array(15)
    w = w.copy()
    scale, zero = _find_params(w, maxq, per_channel, sym, mse)  # as ``not ready()``
    dead = np.diag(h) == 0
    w[dead, :] = 0
    perm = np.arange(w.shape[0])
    if act_order:
        hd = np.diag(h).copy()
        hd[dead] = 1
        perm = np.argsort(-hd)
        w = w[perm, :]
    q = np.zeros_like(w)
    k = w.shape[0]
    starts = range(0, k, group_size) if group_size != -1 else [0]
    for g0 in starts:
        g1 = min(g0 + group_size, k) if group_size != -1 else k
        if group_size != -1:
            scale, zero = _find_params(w[g0:g1, :], maxq, per_channel, sym, mse)
        rows = w[g0:g1, :]
        q_int = np.clip(np.round(rows / scale) + zero, 0, maxq).astype(rows.dtype)
        q[g0:g1, :] = scale * (q_int - zero)
    out = np.empty_like(q)
    out[perm, :] = q
    return out


def _gptq_compensated(
    w: np.ndarray,
    h: np.ndarray,
    *,
    group_size: int,
    per_channel: bool,
    sym: bool,
    mse: bool,
    act_order: bool,
    block_size: int,
    perc_damp: float,
) -> np.ndarray:
    """The same grid with the error propagation GPTQ is meant to do."""
    from onnxsim.quark_weight_rounding import quark_gptq

    res = quark_gptq(
        w,
        h,
        bits=4,
        group_size=group_size,
        block_size=block_size,
        perc_damp=perc_damp,
        act_order=act_order,
        per_channel=per_channel,
        sym=sym,
        mse=mse,
        compensate=True,
    )
    k = w.shape[0]
    hd = np.diag(h).copy()
    hd[hd == 0] = 1
    perm = np.argsort(-hd) if act_order else np.arange(k)
    pos = np.empty(k, dtype=np.int64)
    pos[perm] = np.arange(k)
    grp = pos // group_size if group_size != -1 else np.zeros(k, dtype=np.int64)
    scale = res.scale[grp]
    zero = res.zero[grp]
    return (scale * (res.q_int - zero)).astype(np.float32)


def _gptq_repack(
    q_real: np.ndarray, group_size: int, sym: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``prepare_matmul4bits_node``: re-derive a block-wise grid from the
    GPTQ-dequantized ``[K, N]`` values and pack it. Returns packed
    ``[N, blocks, g/2]``, scales ``[N, blocks]``, packed zero points
    ``[N, ceil(blocks/2)]``."""
    bits = 4
    rows, cols = q_real.shape
    gs = group_size if group_size != -1 else rows
    blob = gs // 2
    blocks = (rows + gs - 1) // gs
    pad = blocks * gs - rows
    if pad > 0:
        q_real = np.pad(q_real, ((0, pad), (0, 0)), "constant")
    qw = np.reshape(q_real.T, (-1, gs))
    min_q = np.min(qw, axis=1, keepdims=True)
    max_q = np.max(qw, axis=1, keepdims=True)
    range_q = np.maximum(np.abs(min_q), np.abs(max_q))
    mask = range_q > 0
    new_scale = np.ones(max_q.shape)
    if sym:
        new_scale[mask] = (range_q[mask] * 2.0).astype(np.float64) / (2**bits - 1)
        new_zero = np.ones(max_q.shape).astype(np.uint8) * (1 << (bits - 1))
    else:
        new_scale[max_q != min_q] = np.array(
            [
                float(v) / (2**bits - 1)
                for v in (max_q - min_q)[max_q != min_q].flatten().tolist()
            ]
        )
        new_zero = np.maximum(
            0,
            np.minimum(
                2**bits - 1, ((np.zeros(new_scale.shape) - min_q) / new_scale).round()
            ),
        ).astype("uint8")
    new_scale = new_scale.astype(np.float32)
    codes = np.clip(np.round(qw / new_scale + new_zero), 0, 2**bits - 1).astype(
        np.uint8
    )
    pair = codes[:, ::2] | (codes[:, 1::2] << 4)
    packed = np.zeros((codes.shape[0], blob)).astype(np.uint8)
    packed[:, :] = pair[:, :blob]
    packed = np.reshape(packed, (-1, blocks, blob))
    scales = np.reshape(new_scale, (-1, blocks))
    zp = np.reshape(new_zero, (-1, blocks))
    zp = _pack_zero_points(zp, 1 << (bits - 1))
    return packed, scales, zp


# -- graph rewriting ----------------------------------------------------------------


def _initializer(
    name: str, stack: List[onnx.GraphProto]
) -> Tuple[Optional[onnx.TensorProto], Optional[onnx.GraphProto]]:
    for g in reversed(stack):
        for t in g.initializer:
            if t.name == name:
                return t, g
    return None, None


def _make_node(
    node: onnx.NodeProto, inputs: List[str], attrs: Dict[str, Any]
) -> onnx.NodeProto:
    return onnx.helper.make_node(
        "MatMulNBits",
        inputs=inputs,
        outputs=[node.output[0]],
        name=node.name + "_Q4" if node.name else "",
        domain=_MS_DOMAIN,
        **attrs,
    )


def _add_ms_opset(model: onnx.ModelProto) -> None:
    if not any(o.domain == _MS_DOMAIN for o in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid(_MS_DOMAIN, 1))


def _uses(name: str, graph: onnx.GraphProto) -> bool:
    for n in graph.node:
        if name in n.input:
            return True
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH and _uses(name, a.g):
                return True
            if a.type == onnx.AttributeProto.GRAPHS and any(
                _uses(name, g) for g in a.graphs
            ):
                return True
    return any(o.name == name for o in graph.output)


def _tensor(arr: np.ndarray, name: str) -> onnx.TensorProto:
    t = numpy_helper.from_array(np.ascontiguousarray(arr))
    t.name = name
    return t


def quantize_matmul_nbits(
    model: onnx.ModelProto,
    *,
    group_size: int = 128,
    symmetric: bool = True,
    bits: int = 4,
    accuracy_level: Optional[int] = 0,
    algorithm: str = "DEFAULT",
    exclude_nodes: Sequence[str] = (),
    gptq_params: Optional[Dict[str, Any]] = None,
    calibration: Optional[Sequence[Dict[str, np.ndarray]]] = None,
) -> Tuple[onnx.ModelProto, MatMulNBitsReport]:
    """Rewrites ``model``'s constant-weight ``MatMul`` nodes into
    ``MatMulNBits`` the way Quark's ``MATMUL_NBITS`` preset does (see the module
    docstring). ``model`` is not modified. ``calibration`` (a list of input
    dicts) is only read by ``algorithm="GPTQ"``."""
    algorithm = algorithm.upper()
    if algorithm not in ("DEFAULT", "HQQ", "GPTQ"):
        raise ValueError(f"unknown MatMulNBits algorithm {algorithm!r}")
    if bits != 4:
        raise NotImplementedError(
            "only 4-bit MatMulNBits is supported (Quark's DEFAULT / HQQ / GPTQ "
            "paths all pack 4-bit codes whatever 'Bits' says)"
        )
    out = onnx.ModelProto()
    out.CopyFrom(model)
    report = MatMulNBitsReport()
    excluded = set(exclude_nodes)
    if algorithm == "GPTQ":
        _gptq_model(out, report, excluded, gptq_params or {}, calibration)
        return out, report
    _check_block_size(group_size)
    _add_ms_opset(out)
    made: Dict[Tuple[int, str], List[str]] = {}
    replaced: List[Tuple[onnx.GraphProto, str]] = []
    _process_graph(
        out.graph,
        [out.graph],
        report,
        excluded,
        made,
        replaced,
        group_size,
        symmetric,
        accuracy_level,
        algorithm,
    )
    # drop the float weights nothing reads any more
    for owner, name in replaced:
        if not _uses(name, owner):
            for t in list(owner.initializer):
                if t.name == name:
                    owner.initializer.remove(t)
    return out, report


def _process_graph(
    graph: onnx.GraphProto,
    stack: List[onnx.GraphProto],
    report: MatMulNBitsReport,
    excluded: "set[str]",
    made: Dict[Tuple[int, str], List[str]],
    replaced: List[Tuple[onnx.GraphProto, str]],
    group_size: int,
    symmetric: bool,
    accuracy_level: Optional[int],
    algorithm: str,
) -> None:
    new_nodes: List[onnx.NodeProto] = []
    for node in graph.node:
        for a in node.attribute:
            subs = (
                [a.g]
                if a.type == onnx.AttributeProto.GRAPH
                else list(a.graphs)
                if a.type == onnx.AttributeProto.GRAPHS
                else []
            )
            for sub in subs:
                _process_graph(
                    sub,
                    stack + [sub],
                    report,
                    excluded,
                    made,
                    replaced,
                    group_size,
                    symmetric,
                    accuracy_level,
                    algorithm,
                )
        if node.op_type != "MatMul" or node.name in excluded:
            new_nodes.append(node)
            continue
        b, owner = _initializer(node.input[1], stack)
        if b is None or owner is None:
            report.skipped.append(node.name)
            new_nodes.append(node)
            continue
        w = numpy_helper.to_array(b)
        if w.ndim != 2 or w.dtype not in (np.float32, np.float16):
            report.skipped.append(node.name)
            new_nodes.append(node)
            continue
        key = (id(owner), b.name)
        if key not in made:
            if algorithm == "HQQ":
                packed, scales, zps = hqq_quantize(w, group_size)
                scales = scales.astype(w.dtype)
                zps = zps.astype(w.dtype)
            else:
                packed, scales, zps = block_quantize_int4(w, group_size, symmetric)
            names = [b.name + "_Q4", b.name + "_scales"]
            inits = [_tensor(packed, names[0]), _tensor(scales, names[1])]
            if zps is not None:
                names.append(b.name + "_zero_points")
                inits.append(_tensor(zps, names[2]))
            owner.initializer.extend(inits)
            for i in list(owner.input):
                if i.name == b.name:
                    owner.input.remove(i)
            made[key] = names
            replaced.append((owner, b.name))
        attrs: Dict[str, Any] = {
            "K": int(w.shape[0]),
            "N": int(w.shape[1]),
            "bits": 4,
            "block_size": group_size,
        }
        if algorithm != "HQQ" and accuracy_level is not None:
            attrs["accuracy_level"] = accuracy_level
        new_nodes.append(_make_node(node, [node.input[0], *made[key]], attrs))
        report.converted.append(new_nodes[-1].name)
    graph.ClearField("node")
    graph.node.extend(new_nodes)


def _gptq_model(
    model: onnx.ModelProto,
    report: MatMulNBitsReport,
    excluded: "set[str]",
    params: Dict[str, Any],
    calibration: Optional[Sequence[Dict[str, np.ndarray]]],
) -> None:
    import onnxruntime as ort

    from onnxsim.bias_correction import _add_probe_outputs

    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    todo: List[onnx.NodeProto] = []
    todo_idx: "set[int]" = set()
    for i, n in enumerate(graph.node):
        if n.op_type == "MatMul" and n.input[1] in inits and n.name not in excluded:
            w = inits[n.input[1]]
            if len(w.dims) == 2 and w.data_type == onnx.TensorProto.FLOAT:
                todo.append(n)
                todo_idx.add(i)
            else:
                report.skipped.append(n.name)
    weights = [n.input[1] for n in todo]
    if len(set(weights)) != len(weights):
        raise NotImplementedError(
            "GPTQ MatMulNBits: a weight shared by several MatMuls (Quark fails on this too)"
        )
    if not todo:
        return
    if not calibration:
        raise ValueError("GPTQ needs calibration data")
    group_size = int(params.get("GroupSize", -1))
    per_channel = bool(params.get("PerChannel", False))
    sym = bool(params.get("WeightSymmetric", True))
    mse = bool(params.get("MSE", False))
    act_order = bool(params.get("ActOrder", False))
    block_size = int(params.get("BlockSize", 128))
    perc_damp = float(params.get("PercDamp", 0.01))
    compensate = bool(params.get("Compensate", False))
    if group_size != -1:
        _check_block_size(group_size)
    names = list(dict.fromkeys(n.input[0] for n in todo))
    sess = ort.InferenceSession(
        _add_probe_outputs(model, names).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    out_names = [o.name for o in sess.get_outputs()]
    got = dict(zip(out_names, sess.run(None, dict(calibration[0]))))

    new_nodes: Dict[int, onnx.NodeProto] = {}
    add: List[onnx.TensorProto] = []
    for idx, node in enumerate(graph.node):
        if idx not in todo_idx:
            continue
        w_t = inits[node.input[1]]
        w = numpy_helper.to_array(w_t)
        k, n_out = w.shape
        h = _hessian(np.asarray(got[node.input[0]]), k)
        kwargs: Dict[str, Any] = dict(
            group_size=group_size,
            per_channel=per_channel,
            sym=sym,
            mse=mse,
            act_order=act_order,
        )
        if compensate:
            q = _gptq_compensated(
                w.astype(np.float64),
                h,
                block_size=block_size,
                perc_damp=perc_damp,
                **kwargs,
            )
        else:
            q = _gptq_rtn_checked(w, h, block_size, perc_damp, kwargs)
        packed, scales, zp = _gptq_repack(q.astype(np.float32), group_size, sym)
        gs = group_size if group_size != -1 else k
        w_name = node.input[1]
        tensors = [
            _tensor(packed, w_name + "_Q4"),
            _tensor(scales, w_name + "_scales"),
        ]
        inputs = [node.input[0], w_name + "_Q4", w_name + "_scales"]
        if not sym:
            tensors.append(_tensor(zp, w_name + "_zero_points"))
            inputs.append(w_name + "_zero_points")
        add.extend(tensors)
        new_nodes[idx] = _make_node(
            node, inputs, {"K": int(k), "N": int(n_out), "bits": 4, "block_size": gs}
        )
        report.converted.append(new_nodes[idx].name)
    nodes = [new_nodes.get(i, n) for i, n in enumerate(graph.node)]
    graph.ClearField("node")
    graph.node.extend(nodes)
    for w_name in weights:
        for t in list(graph.initializer):
            if t.name == w_name:
                graph.initializer.remove(t)
        for i in list(graph.input):
            if i.name == w_name:
                graph.input.remove(i)
    graph.initializer.extend(add)
    _add_ms_opset(model)


def _gptq_rtn_checked(
    w: np.ndarray,
    h: np.ndarray,
    block_size: int,
    perc_damp: float,
    kwargs: Dict[str, Any],
) -> np.ndarray:
    # Quark's fasterquant factors the damped Hessian before anything else and
    # fails if it is not positive definite; mirror that failure mode.
    hd = h.copy()
    dead = np.diag(hd) == 0
    hd[dead, dead] = 1
    if kwargs["act_order"]:
        perm = np.argsort(-np.diag(hd))
        hd = hd[perm, :][:, perm]
    diag = np.arange(hd.shape[0])
    hd[diag, diag] += perc_damp * np.mean(np.diag(hd))
    np.linalg.cholesky(hd)
    return _gptq_rtn(w, h, **kwargs)
