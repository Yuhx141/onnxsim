"""Layer-wise weight-rounding refinement (AdaRound, GPTQ) for **int8 QDQ**
models, i.e. what Quark's ``AdaRoundConfig`` / ``GPTQConfig`` do on top of a
quantized ONNX model. Built on onnxsim's existing numpy cores
(:func:`onnxsim.adaround._optimize_rounding`,
:func:`onnxsim.gptq._gptq_quantize_columns`), which only knew the int4
weight-only scheme.

Both take the float model and a quantized model produced by
:func:`onnxsim.full_qdq.quantize_full_qdq` and change **only the int8 weight
codes** -- scales, zero points, biases and the graph are untouched, so the
result is still a valid QDQ model with identical structure. For each
MatMul / Gemm / Conv layer with a constant float weight:

1. the layer's float input activation is captured over the calibration data
   (Conv inputs are unfolded with im2col, so a Conv is the same ``Y = X W^T``
   problem as a MatMul);
2. AdaRound learns each weight's rounding direction against the layer's float
   output (``min ||X W^T - X W_hat^T||^2`` + the annealed rounding regularizer),
   or GPTQ rounds columns sequentially, compensating the error with the
   inverse Hessian ``X^T X``;
3. the new codes are accepted **only if** the layer's reconstruction error on
   the captured rows does not get worse than the codes already in the model
   (a guard: both algorithms can occasionally lose to round-to-nearest).

Differences from Quark, on purpose: reconstruction is layer-wise against the
float model's activations (Quark's AdaRound optimizes subgraph blocks inside
its fine-tuning engine, full-batch Adam here rather than random mini-batches).
GPTQ keeps the scales ``quantize_full_qdq`` chose unless asked for Quark's own
grid: :func:`gptq_int8` with ``bits`` / ``group_size`` / ``per_channel`` /
``mse`` / ``weight_symmetric`` re-grids the weights exactly like Quark's GPTQ
(:func:`quark_gptq`, bit-identical to ``quark.onnx.algorithm.gptq`` when asked
to mimic its no-op error propagation) and writes the new scales, zero points
and blocked ``DequantizeLinear`` back. AdaRound's ``drop_ratio`` (QDrop-style
input mixing), ``selective_update`` and ``lr_adjust`` follow Quark's options.

Skipped (left as calibrated): ConvTranspose / grouped / non-2-D Conv, Gemm
with ``transA`` or ``alpha != 1``, constant-``A`` MatMuls, weights shared by
several layers, and anything whose weight is not an int8 ``DequantizeLinear``
over initializers with a zero zero-point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from onnxsim.bias_correction import _activation_rows, _add_probe_outputs

_QMIN, _QMAX = -127.0, 127.0  # quantize_full_qdq's symmetric int8 grid
_TARGET_OPS = ("Conv", "Gemm", "MatMul")


@dataclass
class LayerReport:
    """What happened to one layer."""

    name: str
    op: str
    shape: Tuple[int, ...]
    error_before: float  # mean squared output error vs the float layer, as given
    error_after: float  # ... with the returned codes
    accepted: bool
    changed_fraction: float  # fraction of weight codes that changed


@dataclass
class _ActQuant:
    """The Q/DQ pair in front of a layer in the quantized model."""

    pre: str  # tensor feeding the QuantizeLinear (float, pre-quantization)
    scale: float
    zero_point: float
    qmin: float
    qmax: float

    def fake_quant(self, x: np.ndarray) -> np.ndarray:
        q = np.clip(np.round(x / self.scale) + self.zero_point, self.qmin, self.qmax)
        return (q - self.zero_point) * self.scale


_Q_RANGE = {
    onnx.TensorProto.INT8: (-128.0, 127.0),
    onnx.TensorProto.UINT8: (0.0, 255.0),
    onnx.TensorProto.INT16: (-32768.0, 32767.0),
    onnx.TensorProto.UINT16: (0.0, 65535.0),
}


@dataclass
class _Layer:
    name: str
    op: str
    node: onnx.NodeProto  # the float model's node
    w_float: np.ndarray  # original layout
    wq_name: str
    codes: np.ndarray  # int8 codes, original layout
    scale: np.ndarray  # broadcast to the original layout
    to_nk: Callable[[np.ndarray], np.ndarray]
    from_nk: Callable[[np.ndarray], np.ndarray]
    dq_output: str = ""  # the weight DequantizeLinear's output tensor
    scale_name: str = ""
    zp_name: str = ""
    k_axis: Optional[int] = None  # input-channel axis in the original layout
    out_axis: int = 0  # output-channel axis in the original layout
    act: Optional[_ActQuant] = None
    # set per layer by _refine
    drop: Optional["_Drop"] = None
    err_before: float = 0.0


def _find_act_quant(
    q_by_out: Dict[str, onnx.NodeProto],
    q_inits: Dict[str, onnx.TensorProto],
    tensor: str,
) -> Optional[_ActQuant]:
    """The per-tensor Q/DQ pair that produces layer input ``tensor``, if any."""
    dq = q_by_out.get(tensor)
    if dq is None or dq.op_type != "DequantizeLinear":
        return None
    q = q_by_out.get(dq.input[0])
    if q is None or q.op_type != "QuantizeLinear" or len(q.input) < 2:
        return None
    s = q_inits.get(q.input[1])
    if s is None or numpy_helper.to_array(s).size != 1:
        return None
    zp, dtype = 0.0, onnx.TensorProto.UINT8
    if len(q.input) > 2 and q.input[2]:
        z = q_inits.get(q.input[2])
        if z is None or numpy_helper.to_array(z).size != 1:
            return None
        zp, dtype = float(numpy_helper.to_array(z).reshape(-1)[0]), z.data_type
    if dtype not in _Q_RANGE:
        return None
    lo, hi = _Q_RANGE[dtype]
    return _ActQuant(
        q.input[0], float(numpy_helper.to_array(s).reshape(-1)[0]), zp, lo, hi
    )


# -- locating layers -----------------------------------------------------------------


def _attr(node: onnx.NodeProto, name: str, default):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def _identity(a: np.ndarray) -> np.ndarray:
    return a


def _transpose(a: np.ndarray) -> np.ndarray:
    return a.T


def _flatten_conv(a: np.ndarray) -> np.ndarray:
    """``[O, I, kh, kw]`` -> ``[O, I*kh*kw]`` (the shape comes from ``a``)."""
    return a.reshape(a.shape[0], -1)


def _reshape_to(shape: Tuple[int, ...]) -> Callable[[np.ndarray], np.ndarray]:
    def reshape(a: np.ndarray) -> np.ndarray:
        return a.reshape(shape)

    return reshape


def _find_layers(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    target_ops: Sequence[str],
) -> List[_Layer]:
    f_inits = {t.name: t for t in float_model.graph.initializer}
    q_inits = {t.name: t for t in quant_model.graph.initializer}
    q_by_out = {o: n for n in quant_model.graph.node for o in n.output}
    f_by_weight: Dict[Tuple[str, str], onnx.NodeProto] = {}
    for fn in float_model.graph.node:
        if fn.op_type in target_ops and len(fn.input) >= 2:
            f_by_weight[(fn.op_type, fn.input[1])] = fn

    uses: Dict[str, int] = {}
    for qn in quant_model.graph.node:
        if qn.op_type in target_ops and len(qn.input) >= 2:
            uses[qn.input[1]] = uses.get(qn.input[1], 0) + 1

    layers: List[_Layer] = []
    for qn in quant_model.graph.node:
        if qn.op_type not in target_ops or len(qn.input) < 2 or uses[qn.input[1]] > 1:
            continue
        dq = q_by_out.get(qn.input[1])
        if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) < 2:
            continue
        wq = q_inits.get(dq.input[0])
        ws = q_inits.get(dq.input[1])
        if wq is None or ws is None or wq.data_type != onnx.TensorProto.INT8:
            continue
        if len(dq.input) > 2 and dq.input[2]:
            zp = q_inits.get(dq.input[2])
            if zp is None or bool(np.any(numpy_helper.to_array(zp) != 0)):
                continue
        # quantize_full_qdq names a weight's quantized tensors "<weight>/qdqN/..."
        fw_name = dq.input[0].split("/qdq")[0]
        fn = f_by_weight.get((qn.op_type, fw_name))
        w_init = f_inits.get(fw_name)
        if fn is None or w_init is None or w_init.data_type != onnx.TensorProto.FLOAT:
            continue
        w = numpy_helper.to_array(w_init)
        codes = numpy_helper.to_array(wq)
        if w.shape != codes.shape:
            continue

        axis = int(_attr(dq, "axis", 1))
        s = numpy_helper.to_array(ws).astype(np.float64)
        if s.size == 1:
            scale = np.full(w.shape, float(s.reshape(-1)[0]))
        elif (
            s.ndim == 1 and w.ndim > axis % w.ndim and s.size == w.shape[axis % w.ndim]
        ):
            shape = [1] * w.ndim
            shape[axis % w.ndim] = -1
            scale = np.broadcast_to(s.reshape(shape), w.shape).copy()
        else:
            continue

        to_nk: Callable[[np.ndarray], np.ndarray]
        from_nk: Callable[[np.ndarray], np.ndarray]
        if qn.op_type == "MatMul":
            if w.ndim != 2 or fn.input[1] != fw_name:
                continue
            to_nk, from_nk = _transpose, _transpose
        elif qn.op_type == "Gemm":
            if (
                w.ndim != 2
                or _attr(fn, "transA", 0)
                or float(_attr(fn, "alpha", 1.0)) != 1.0
            ):
                continue
            if _attr(fn, "transB", 0):
                to_nk, from_nk = _identity, _identity
            else:
                to_nk, from_nk = _transpose, _transpose
        else:  # Conv
            if (
                w.ndim != 4
                or int(_attr(fn, "group", 1)) != 1
                or _attr(fn, "auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET")
            ):
                continue
            to_nk, from_nk = _flatten_conv, _reshape_to(w.shape)
        k_axis: Optional[int] = None
        if qn.op_type != "Conv":
            k_axis = 0 if to_nk is _transpose else 1
        layers.append(
            _Layer(
                qn.name or qn.output[0],
                qn.op_type,
                fn,
                w.astype(np.float64),
                wq.name,
                codes,
                scale,
                to_nk,
                from_nk,
                dq_output=dq.output[0],
                scale_name=dq.input[1],
                zp_name=dq.input[2] if len(dq.input) > 2 else "",
                k_axis=k_axis,
                out_axis=0 if k_axis is None else 1 - k_axis,
                act=_find_act_quant(q_by_out, q_inits, qn.input[0]),
            )
        )
    # Two layers resolving to the same int8 initializer (a shared weight behind
    # separate DQ nodes) cannot both be refined: drop them all.
    count: Dict[str, int] = {}
    for ly in layers:
        count[ly.wq_name] = count.get(ly.wq_name, 0) + 1
    return [ly for ly in layers if count[ly.wq_name] == 1]


# -- activations -----------------------------------------------------------------------


def _im2col(
    x: np.ndarray,
    kernel: Tuple[int, int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
) -> np.ndarray:
    """``[N, C, H, W]`` -> ``[N*Ho*Wo, C*kh*kw]`` with ``(c, kh, kw)`` column
    order, i.e. matching ``weight.reshape(O, -1)``."""
    n, c, h, w = x.shape
    kh, kw = kernel
    sh, sw = strides
    dh, dw = dilations
    pt, pl, pb, pr = pads
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    oh = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1
    ow = (w + pl + pr - dw * (kw - 1) - 1) // sw + 1
    cols = np.empty((n, oh, ow, c, kh, kw), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            patch = xp[
                :, :, i * dh : i * dh + sh * oh : sh, j * dw : j * dw + sw * ow : sw
            ]
            cols[:, :, :, :, i, j] = patch.transpose(0, 2, 3, 1)
    return cols.reshape(n * oh * ow, c * kh * kw)


def _rows(layer: _Layer, arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
    if layer.op != "Conv":
        return _activation_rows(arrays)
    node = layer.node
    k = layer.w_float.shape[2:]
    strides = _attr(node, "strides", [1, 1])
    pads = _attr(node, "pads", [0, 0, 0, 0])
    dil = _attr(node, "dilations", [1, 1])
    return [
        _im2col(a.astype(np.float64), (k[0], k[1]), strides, pads, dil)
        for a in arrays
        if a.ndim == 4
    ]


def _capture(
    float_model: onnx.ModelProto,
    names: Sequence[str],
    calibration_data: Sequence[Dict[str, np.ndarray]],
    providers: Optional[Sequence[str]],
) -> Dict[str, List[np.ndarray]]:
    import onnxruntime as ort

    probe = _add_probe_outputs(float_model, names)
    sess = ort.InferenceSession(
        probe.SerializeToString(), providers=list(providers or ["CPUExecutionProvider"])
    )
    out_names = [o.name for o in sess.get_outputs()]
    acts: Dict[str, List[np.ndarray]] = {n: [] for n in names}
    for batch in calibration_data:
        res = dict(zip(out_names, sess.run(None, batch)))
        for n in names:
            acts[n].append(np.asarray(res[n], dtype=np.float64))
    return acts


# -- Quark's GPTQ, mirrored ---------------------------------------------------------


def _quark_find_params(
    x: np.ndarray,
    maxq: float,
    per_channel: bool,
    sym: bool,
    mse: bool,
    norm: float = 2.4,
    grid: int = 100,
    max_shrink: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """``GPTQ.find_params`` of ``quark.onnx.algorithm.gptq``: scale and zero
    point (both ``[N]``) of the ``[rows, N]`` weight slice ``x``. Without
    ``per_channel`` one scale covers the whole slice (repeated ``N`` times)."""
    x = np.asarray(x, dtype=np.float64)
    n_cols = x.shape[1]
    if not per_channel:
        x = x.reshape(-1, 1)
    zeros = np.zeros(x.shape[1])
    xmin = np.minimum(x.min(axis=0), zeros)
    xmax = np.maximum(x.max(axis=0), zeros)
    if sym:
        xmax = np.maximum(np.abs(xmin), xmax)
        xmin = np.where(xmin < 0, -xmax, xmin)
    both_zero = (xmin == 0) & (xmax == 0)
    xmin = np.where(both_zero, -1.0, xmin)
    xmax = np.where(both_zero, 1.0, xmax)

    scale = (xmax - xmin) / maxq
    zero = np.full_like(scale, (maxq + 1) / 2) if sym else np.round(-xmin / scale)
    if mse:
        best = np.full(x.shape[1], np.inf)
        for i in range(int(max_shrink * grid)):
            p = 1 - i / grid
            xmin1, xmax1 = p * xmin, p * xmax
            scale1 = (xmax1 - xmin1) / maxq
            zero1 = zero if sym else np.round(-xmin1 / scale1)
            q = np.clip(np.round(x / scale1) + zero1, 0, maxq)
            err = np.sum(np.abs(scale1 * (q - zero1) - x) ** norm, axis=0)
            better = err < best
            best = np.where(better, err, best)
            scale = np.where(better, scale1, scale)
            zero = np.where(better, zero1, zero)
    if not per_channel:
        scale, zero = np.repeat(scale, n_cols), np.repeat(zero, n_cols)
    return scale, zero


@dataclass
class QuarkGPTQResult:
    """Output of :func:`quark_gptq`, for a ``[K, N]`` weight."""

    q_int: np.ndarray  # [K, N] unsigned codes in [0, 2**bits - 1], original row order
    scale: np.ndarray  # [G, N] (G = ceil(K / group_size), or 1 when ungrouped)
    zero: np.ndarray  # [G, N] integer-valued zero points
    group_size: int  # -1 when ungrouped


def quark_gptq(
    w_kn: np.ndarray,
    h: np.ndarray,
    bits: int = 8,
    group_size: int = -1,
    block_size: int = 128,
    perc_damp: float = 0.01,
    act_order: bool = False,
    per_channel: bool = False,
    sym: bool = True,
    mse: bool = False,
    compensate: bool = True,
) -> QuarkGPTQResult:
    """AMD Quark's GPTQ (``GPTQ.fasterquant`` in ``quark.onnx.algorithm.gptq``)
    on a ``[K, N]`` weight (input channel first, the ONNX MatMul layout) and
    its ``[K, K]`` Hessian ``h``, quirks included:

    - the quantization grid is Quark's own, not the existing QDQ model's:
      unsigned ``bits``-bit codes, ``scale = (max - min) / (2**bits - 1)``
      with ``zero = 2**(bits-1)`` when ``sym`` (so a symmetric code range is
      ``[-2**(bits-1), 2**(bits-1) - 1]``), one scale for the whole tensor
      unless ``per_channel`` (per output column);
    - with ``group_size != -1`` every ``group_size`` rows get their own
      scale, found from the weights *as of the last block update* (rows of the
      block being processed do not yet carry that block's own compensation --
      Quark calls ``find_params`` on the outer ``W``, not the block copy);
    - ``act_order`` visits rows by descending Hessian diagonal; the groups are
      then runs of ``group_size`` rows of that permuted order;
    - dead rows (zero Hessian diagonal) get ``H = 1`` and zero weights.

    **Quark 0.13 never compensates** (checked against its source, and bit for
    bit by ``tests/test_quark_parity.py``): it indexes the upper-triangular
    Cholesky factor of ``H^-1`` by *column* (``Hinv1[i:, i]``,
    ``Hinv[i2:, i1:i2]``), which is zero below the diagonal, so its "GPTQ"
    reduces to round-to-nearest on its grid. ``compensate=False`` reproduces
    that exactly (the codes then match Quark's bit for bit); the default
    ``True`` indexes by row as the GPTQ paper does and really propagates each
    row's rounding error into the not-yet-quantized rows. Scales, zero points
    and the grouping are the same either way for the ungrouped case.

    Returns the unsigned codes and the per-group scales / zero points (Quark
    itself only keeps the last group's, then re-derives scales in its
    MatMulNBits packing; the QDQ path here keeps the real ones).
    """
    if not 2 <= bits <= 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    w0 = np.asarray(w_kn, dtype=np.float64)
    k, n = w0.shape
    maxq = float(2**bits - 1)
    scale, zero = _quark_find_params(w0, maxq, per_channel, sym, mse)

    w = w0.copy()
    h = np.array(h, dtype=np.float64)
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    w[dead, :] = 0.0
    perm = np.arange(k)
    if act_order:
        perm = np.argsort(-np.diag(h))
        w = w[perm, :]
        h = h[perm, :][:, perm]
    diag = np.arange(k)
    h[diag, diag] += perc_damp * np.mean(np.diag(h))
    l_inv = np.linalg.inv(np.linalg.cholesky(h))
    hinv = np.linalg.cholesky(l_inv.T @ l_inv).T

    q_int = np.zeros_like(w)
    scales: List[np.ndarray] = []
    zeros: List[np.ndarray] = []
    for i1 in range(0, k, block_size):
        i2 = min(i1 + block_size, k)
        w1 = w[i1:i2, :].copy()
        err1 = np.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            d = hinv1[i, i]
            if group_size != -1 and (i1 + i) % group_size == 0:
                scale, zero = _quark_find_params(
                    w[i1 + i : i1 + i + group_size, :], maxq, per_channel, sym, mse
                )
                scales.append(scale)
                zeros.append(zero)
            wi = w1[i, :]
            qi = np.clip(np.round(wi / scale) + zero, 0, maxq)
            q_int[i1 + i, :] = qi
            err = (wi - scale * (qi - zero)) / d
            if compensate:
                w1[i:, :] -= np.outer(hinv1[i, i:], err)
            else:  # Quark: column slice of an upper-triangular matrix == 0
                w1[i:, :] -= np.outer(hinv1[i:, i], err)
            err1[i, :] = err
        if compensate:
            w[i2:, :] -= hinv[i1:i2, i2:].T @ err1
        else:
            w[i2:, :] -= hinv[i2:, i1:i2] @ err1

    if group_size == -1:
        scales, zeros = [scale], [zero]
    out = np.empty_like(q_int)
    out[perm, :] = q_int
    return QuarkGPTQResult(out, np.stack(scales), np.stack(zeros), group_size)


# -- applying a re-gridded layer -----------------------------------------------------


@dataclass
class _Regrid:
    """A refiner result that replaces the weight's *scales* too (Quark's GPTQ
    recomputes them): unsigned codes ``[N, K]`` and the grid they sit on."""

    q_nk: np.ndarray  # chosen codes, unsigned
    baseline_nk: np.ndarray  # round-to-nearest codes on the very same grid
    scale: np.ndarray  # [G, N]
    zero: np.ndarray  # [G, N]
    group_size: int  # -1: ungrouped
    per_channel: bool
    sym: bool

    def _expand(self, a: np.ndarray, k: int) -> np.ndarray:
        gs = k if self.group_size == -1 else self.group_size
        return np.repeat(a, gs, axis=0)[:k].T  # [N, K]

    def scale_nk(self, k: int) -> np.ndarray:
        return self._expand(self.scale, k)

    def zero_nk(self, k: int) -> np.ndarray:
        return self._expand(self.zero, k)


def _set_attr(node: onnx.NodeProto, name: str, value: Optional[int]) -> None:
    for i, a in enumerate(node.attribute):
        if a.name == name:
            del node.attribute[i]
            break
    if value is not None:
        node.attribute.append(onnx.helper.make_attribute(name, value))


def _write_regrid(
    out: onnx.ModelProto, ly: _Layer, rg: _Regrid, q_nk: np.ndarray
) -> Callable[[], None]:
    """Rewrites ``ly``'s weight codes, scale, zero point and DequantizeLinear
    ``axis`` / ``block_size`` for the re-gridded layer; returns an undo."""
    graph = out.graph
    dq = next(n for n in graph.node if ly.dq_output in n.output)
    inits = {t.name: t for t in graph.initializer}
    saved_dq = onnx.NodeProto()
    saved_dq.CopyFrom(dq)
    saved = {
        n: _copy_tensor(inits[n]) for n in (ly.wq_name, ly.scale_name, ly.zp_name) if n
    }

    if rg.group_size != -1:
        scale = ly.from_nk(rg.scale.T)  # blocked layout: the K axis shrinks to G
        zero = ly.from_nk(rg.zero.T)
        axis: Optional[int] = ly.k_axis
        block: Optional[int] = rg.group_size
    elif rg.per_channel:
        scale, zero = rg.scale[0], rg.zero[0]
        axis, block = ly.out_axis, None
    else:
        scale, zero = rg.scale[0, :1].reshape(()), rg.zero[0, :1].reshape(())
        axis, block = None, None

    unsigned = not rg.sym
    code_nk = q_nk if unsigned else q_nk - rg.zero_nk(q_nk.shape[1])
    stored = np.ascontiguousarray(ly.from_nk(code_nk)).astype(
        np.uint8 if unsigned else np.int8
    )
    inits[ly.wq_name].CopyFrom(numpy_helper.from_array(stored, ly.wq_name))

    new_scale = f"{ly.wq_name}/scale"
    dq.input[1] = new_scale
    graph.initializer.append(
        numpy_helper.from_array(np.asarray(scale, dtype=np.float32), new_scale)
    )
    added = [new_scale]
    del dq.input[2:]
    if unsigned:
        new_zp = f"{ly.wq_name}/zero_point"
        dq.input.append(new_zp)
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(zero).astype(np.uint8), new_zp)
        )
        added.append(new_zp)
    _set_attr(dq, "axis", axis)
    _set_attr(dq, "block_size", block)

    used = {i for n in graph.node for i in n.input}
    removed = [
        n for n in (ly.scale_name, ly.zp_name) if n and n not in used and n in inits
    ]
    for t in [t for t in graph.initializer if t.name in removed]:
        graph.initializer.remove(t)

    def undo() -> None:
        dq.CopyFrom(saved_dq)
        for t in [t for t in graph.initializer if t.name in added]:
            graph.initializer.remove(t)
        present = {t.name for t in graph.initializer}
        for name, tensor in saved.items():
            if name not in present:
                graph.initializer.append(tensor)
            elif name == ly.wq_name:
                inits[name].CopyFrom(tensor)

    return undo


def _copy_tensor(t: onnx.TensorProto) -> onnx.TensorProto:
    c = onnx.TensorProto()
    c.CopyFrom(t)
    return c


# -- activations of the quantized model (Quark's quantized layer input) ---------------


@dataclass
class _Drop:
    """Quark's ``drop_ratio`` input mixing for one layer: the quantized model's
    pre-quantization input ``x_q`` and the float model's ``x_f`` (same rows),
    mixed element-wise, then through the layer's own input Q/DQ."""

    x_q: np.ndarray
    x_f: np.ndarray
    fake_quant: Callable[[np.ndarray], np.ndarray]
    ratio: float
    rng: np.random.Generator

    def sample(self) -> np.ndarray:
        if self.ratio >= 1:
            mixed = self.x_q
        elif self.ratio <= 0:
            mixed = self.x_f
        else:
            mixed = np.where(
                self.rng.random(self.x_q.shape) < self.ratio, self.x_q, self.x_f
            )
        return self.fake_quant(mixed)


def _model_outputs(
    model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    providers: Optional[Sequence[str]],
) -> List[List[np.ndarray]]:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=list(providers or ["CPUExecutionProvider"])
    )
    return [sess.run(None, b) for b in calibration_data]


def _avg_l2(a: List[List[np.ndarray]], b: List[List[np.ndarray]]) -> float:
    """Quark's ``average_L2``: mean over (batch, output) of the L2 norm of the
    float32 difference."""
    return float(
        np.mean(
            [
                np.linalg.norm(x.astype(np.float32) - y.astype(np.float32))
                for ra, rb in zip(a, b)
                for x, y in zip(ra, rb)
            ]
        )
    )


# -- the algorithms ------------------------------------------------------------------

Refiner = Callable[[_Layer, np.ndarray, np.ndarray, np.ndarray], object]
"""``f(layer, w_nk, scale_nk, x_rows)`` -> new codes ``[N, K]`` (float array of
ints) on the model's existing scales, a :class:`_Regrid` (new scales too), or
``None`` (skip the layer)."""


def _refine(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    refiner: Refiner,
    target_ops: Sequence[str],
    max_rows: int,
    seed: int,
    providers: Optional[Sequence[str]],
    drop_ratio: Optional[float] = None,
    selective_update: bool = False,
) -> Tuple[onnx.ModelProto, List[LayerReport]]:
    if not calibration_data:
        raise ValueError("calibration_data is required")
    layers = _find_layers(float_model, quant_model, target_ops)
    out = onnx.ModelProto()
    out.CopyFrom(quant_model)
    if not layers:
        return out, []

    probe_names = sorted({ly.node.input[0] for ly in layers})
    acts = _capture(float_model, probe_names, calibration_data, providers)
    q_acts: Dict[str, List[np.ndarray]] = {}
    if drop_ratio is not None:
        q_names = sorted({ly.act.pre for ly in layers if ly.act is not None})
        if q_names:
            q_acts = _capture(quant_model, q_names, calibration_data, providers)
    inits = {t.name: t for t in out.graph.initializer}
    rng = np.random.default_rng(seed)
    drop_rng = np.random.default_rng([seed, 1])
    reports: List[LayerReport] = []

    float_out: List[List[np.ndarray]] = []
    l2 = 0.0
    if selective_update:
        float_out = _model_outputs(float_model, calibration_data, providers)
        l2 = _avg_l2(
            float_out, _model_outputs(quant_model, calibration_data, providers)
        )

    for ly in layers:
        parts = _rows(ly, acts[ly.node.input[0]])
        if not parts:
            continue
        x = np.concatenate(parts, axis=0)
        w_nk = ly.to_nk(ly.w_float)
        if x.shape[1] != w_nk.shape[1]:
            continue
        x_q: Optional[np.ndarray] = None
        if drop_ratio is not None and ly.act is not None and ly.act.pre in q_acts:
            q_parts = _rows(ly, q_acts[ly.act.pre])
            if q_parts:
                x_q = np.concatenate(q_parts, axis=0)
                if x_q.shape != x.shape:
                    x_q = None
        if x.shape[0] > max_rows:
            pick = rng.choice(x.shape[0], max_rows, replace=False)
            x = x[pick]
            x_q = None if x_q is None else x_q[pick]
        scale_nk = ly.to_nk(ly.scale)
        y_ref = x @ w_nk.T

        # With drop_ratio, accuracy is judged where the deployed model runs:
        # on the quantized model's own (fake-quantized) layer input.
        x_eval = x
        ly.drop = None
        if x_q is not None and ly.act is not None:
            ly.drop = _Drop(x_q, x, ly.act.fake_quant, float(drop_ratio or 0), drop_rng)
            x_eval = ly.act.fake_quant(x_q)

        def err(codes_nk: np.ndarray) -> float:
            return float(np.mean((x_eval @ (codes_nk * scale_nk).T - y_ref) ** 2))

        cur_nk = ly.to_nk(ly.codes).astype(np.float64)
        before = err(cur_nk)
        ly.err_before = before
        try:
            res = refiner(ly, w_nk, scale_nk, x)
        except np.linalg.LinAlgError:
            res = cur_nk  # Hessian not factorizable: keep the calibrated codes
        if res is None:
            continue

        undo: Optional[Callable[[], None]] = None
        if isinstance(res, _Regrid):
            k = w_nk.shape[1]
            s_nk, z_nk = res.scale_nk(k), res.zero_nk(k)

            def err_rg(q_nk: np.ndarray) -> float:
                w_hat = (q_nk - z_nk) * s_nk
                return float(np.mean((x_eval @ w_hat.T - y_ref) ** 2))

            before, after = err_rg(res.baseline_nk), err_rg(res.q_nk)
            accepted = after <= before
            final_q = res.q_nk if accepted else res.baseline_nk
            undo = _write_regrid(out, ly, res, final_q)
            changed = float(np.mean(final_q != res.baseline_nk))
            reported = after if accepted else before
        else:
            new_nk = np.clip(np.asarray(res, dtype=np.float64), _QMIN, _QMAX)
            after = err(new_nk)
            accepted = after <= before
            final_nk = new_nk if accepted else cur_nk
            if accepted:
                prev = _copy_tensor(inits[ly.wq_name])
                inits[ly.wq_name].CopyFrom(
                    numpy_helper.from_array(
                        np.ascontiguousarray(ly.from_nk(final_nk)).astype(np.int8),
                        ly.wq_name,
                    )
                )

                def undo(prev: onnx.TensorProto = prev, name: str = ly.wq_name) -> None:
                    inits[name].CopyFrom(prev)

            changed = float(np.mean(final_nk != cur_nk))
            reported = after if accepted else before

        if selective_update and undo is not None:
            new_l2 = _avg_l2(
                float_out, _model_outputs(out, calibration_data, providers)
            )
            if new_l2 < l2:
                l2 = new_l2
            else:  # the end-to-end distance did not shrink: drop this layer's update
                undo()
                accepted, reported, changed = False, before, 0.0
        reports.append(
            LayerReport(
                ly.name,
                ly.op,
                tuple(ly.codes.shape),
                before,
                reported,
                accepted,
                changed,
            )
        )
    return out, reports


def adaround_int8(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    num_iterations: int = 1000,
    learning_rate: float = 0.1,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
    target_ops: Sequence[str] = _TARGET_OPS,
    max_rows: int = 2048,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    drop_ratio: Optional[float] = None,
    selective_update: bool = False,
    lr_adjust: Optional[Tuple[float, float]] = None,
) -> Tuple[onnx.ModelProto, List[LayerReport]]:
    """AdaRound over int8 QDQ weights (see the module docstring). Returns the
    refined model and one :class:`LayerReport` per layer it examined.

    Quark's options:

    - ``drop_ratio``: ``None`` (default) optimizes against the float model's
      layer input. A number in ``[0, 1]`` is Quark's ``DropRatio`` (QDrop
      style): every iteration's layer input is the element-wise mix of the
      quantized model's pre-quantization activation (kept with probability
      ``drop_ratio``) and the float model's, then fake-quantized with the
      layer's own input Q/DQ; ``1`` is all-quantized (Quark's default), ``0``
      all-float. The target is always the float layer output. The quantized
      activations come from the *given* ``quant_model`` for every layer at
      once, like Quark's ``parallel=True`` (Quark otherwise re-captures them
      after each layer's update).
    - ``selective_update``: after each layer, keep its new codes only if the
      average L2 distance between the float and quantized *model outputs* over
      ``calibration_data`` shrank (Quark's ``SelectiveUpdate``).
    - ``lr_adjust``: ``(threshold, lr)``: a layer whose output MSE before
      optimization exceeds ``threshold`` uses ``lr`` instead of
      ``learning_rate`` (Quark's ``LRAdjust``).

    Quark's ``update_bias`` has no AdaRound effect there either (only AdaQuant
    reads it), so it is not a parameter here.
    """
    from onnxsim.adaround import _optimize_rounding

    def refine(ly, w_nk, scale_nk, x):
        lr = learning_rate
        if lr_adjust is not None and ly.err_before > lr_adjust[0]:
            lr = float(lr_adjust[1])
        return _optimize_rounding(
            w_nk,
            scale_nk,
            x,
            _QMIN,
            _QMAX,
            num_iterations,
            lr,
            reg_param,
            warm_start,
            tuple(beta_range),
            x_sampler=None if ly.drop is None else ly.drop.sample,
        )

    return _refine(
        float_model,
        quant_model,
        calibration_data,
        refine,
        target_ops,
        max_rows,
        seed,
        providers,
        drop_ratio=drop_ratio,
        selective_update=selective_update,
    )


def gptq_int8(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    perc_damp: float = 0.01,
    block_size: int = 128,
    act_order: bool = False,
    target_ops: Sequence[str] = _TARGET_OPS,
    max_rows: int = 8192,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    bits: int = 8,
    group_size: int = -1,
    per_channel: bool = False,
    weight_symmetric: bool = True,
    mse: bool = False,
    requantize: Optional[bool] = None,
    selective_update: bool = False,
) -> Tuple[onnx.ModelProto, List[LayerReport]]:
    """GPTQ over int8 QDQ weights. ``act_order`` quantizes columns in
    descending order of Hessian diagonal.

    By default the existing per-channel scales are kept (``requantize``
    ``False``). With ``requantize=True`` -- implied by ``bits != 8``,
    ``group_size != -1``, ``weight_symmetric=False`` or ``mse`` -- each layer
    is re-gridded the way Quark's GPTQ does it (:func:`quark_gptq`:
    ``bits``-bit codes, one scale per tensor / per output channel
    (``per_channel``) / per ``group_size`` input channels, ``mse`` clipping
    search), and the scales, zero points and the DequantizeLinear ``axis`` /
    ``block_size`` are rewritten to match. Codes stay in an int8 container
    (uint8 for ``weight_symmetric=False``); ``group_size`` needs a blocked
    ``DequantizeLinear`` (opset >= 21) and a MatMul / Gemm weight, other
    layers are left as calibrated. In that mode a layer's "before" error in
    its :class:`LayerReport` is round-to-nearest on the *new* grid, and the
    GPTQ codes replace round-to-nearest only if they are not worse.
    """
    from onnxsim.gptq import _gptq_quantize_columns

    if requantize is None:
        requantize = bits != 8 or group_size != -1 or not weight_symmetric or bool(mse)
    if requantize:
        if not 2 <= bits <= 8:
            raise ValueError(f"bits must be in [2, 8], got {bits}")
        if group_size != -1:
            if group_size < 1:
                raise ValueError("group_size must be -1 or positive")
            if act_order:
                raise NotImplementedError(
                    "act_order with group_size cannot be expressed as a blocked "
                    "DequantizeLinear (the groups are not contiguous input channels)"
                )
            opset = next(
                (
                    o.version
                    for o in quant_model.opset_import
                    if o.domain in ("", "ai.onnx")
                ),
                0,
            )
            if opset < 21:
                raise NotImplementedError(
                    "group_size needs blocked DequantizeLinear (opset >= 21), "
                    f"model has opset {opset}"
                )

    def refine_regrid(ly, w_nk, scale_nk, x):
        if group_size != -1 and ly.k_axis is None:
            return None  # Conv weights have no blockable K axis
        n, k = w_nk.shape
        h = 2.0 / x.shape[0] * (x.T @ x)
        res = quark_gptq(
            w_nk.T,
            h,
            bits,
            group_size,
            block_size,
            perc_damp,
            act_order,
            per_channel,
            weight_symmetric,
            mse,
        )
        rg = _Regrid(
            res.q_int.T,
            np.zeros((n, k)),
            res.scale,
            res.zero,
            group_size,
            per_channel,
            weight_symmetric,
        )
        s_nk, z_nk = rg.scale_nk(k), rg.zero_nk(k)
        rg.baseline_nk = np.clip(np.round(w_nk / s_nk) + z_nk, 0, 2.0**bits - 1)
        return rg

    def refine(ly, w_nk, scale_nk, x):
        n, k = w_nk.shape
        # GPTQ rounds whole columns with one scale per output channel, so the
        # scale must not vary along K.
        if not np.allclose(scale_nk, scale_nk[:, :1]):
            return ly.to_nk(ly.codes).astype(np.float64)
        h = x.T @ x / x.shape[0]
        perm = np.argsort(-np.diag(h)) if act_order else np.arange(k)
        codes = _gptq_quantize_columns(
            w_nk[:, perm],
            scale_nk[:, :1],
            k,  # one scale group spanning all of K
            h[np.ix_(perm, perm)],
            perc_damp,
            block_size,
            _QMIN,
            _QMAX,
        )
        out = np.empty_like(codes)
        out[:, perm] = codes
        return out

    return _refine(
        float_model,
        quant_model,
        calibration_data,
        refine_regrid if requantize else refine,
        target_ops,
        max_rows,
        seed,
        providers,
        selective_update=selective_update,
    )


__all__ = [
    "LayerReport",
    "QuarkGPTQResult",
    "adaround_int8",
    "gptq_int8",
    "quark_gptq",
]
