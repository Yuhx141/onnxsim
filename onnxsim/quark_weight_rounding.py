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
its fine-tuning engine); GPTQ keeps the scales ``quantize_full_qdq`` chose
rather than recomputing them; only 8-bit, symmetric, ungrouped weights.

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


# -- the two algorithms ----------------------------------------------------------------

Refiner = Callable[[_Layer, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
"""``f(layer, w_nk, scale_nk, x_rows) -> new_codes_nk`` (float array of ints)."""


def _refine(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    refiner: Refiner,
    target_ops: Sequence[str],
    max_rows: int,
    seed: int,
    providers: Optional[Sequence[str]],
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
    inits = {t.name: t for t in out.graph.initializer}
    rng = np.random.default_rng(seed)
    reports: List[LayerReport] = []

    for ly in layers:
        parts = _rows(ly, acts[ly.node.input[0]])
        if not parts:
            continue
        x = np.concatenate(parts, axis=0)
        w_nk = ly.to_nk(ly.w_float)
        if x.shape[1] != w_nk.shape[1]:
            continue
        if x.shape[0] > max_rows:
            x = x[rng.choice(x.shape[0], max_rows, replace=False)]
        scale_nk = ly.to_nk(ly.scale)
        y_ref = x @ w_nk.T

        def err(codes_nk: np.ndarray) -> float:
            return float(np.mean((x @ (codes_nk * scale_nk).T - y_ref) ** 2))

        cur_nk = ly.to_nk(ly.codes).astype(np.float64)
        before = err(cur_nk)
        try:
            new_nk = refiner(ly, w_nk, scale_nk, x)
        except np.linalg.LinAlgError:
            new_nk = cur_nk  # Hessian not factorizable: keep the calibrated codes
        new_nk = np.clip(new_nk, _QMIN, _QMAX)
        after = err(new_nk)
        accepted = after <= before
        final_nk = new_nk if accepted else cur_nk
        if accepted:
            inits[ly.wq_name].CopyFrom(
                numpy_helper.from_array(
                    np.ascontiguousarray(ly.from_nk(final_nk)).astype(np.int8),
                    ly.wq_name,
                )
            )
        reports.append(
            LayerReport(
                ly.name,
                ly.op,
                tuple(ly.codes.shape),
                before,
                after if accepted else before,
                accepted,
                float(np.mean(final_nk != cur_nk)),
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
) -> Tuple[onnx.ModelProto, List[LayerReport]]:
    """AdaRound over int8 QDQ weights (see the module docstring). Returns the
    refined model and one :class:`LayerReport` per layer it examined."""
    from onnxsim.adaround import _optimize_rounding

    def refine(ly, w_nk, scale_nk, x):
        return _optimize_rounding(
            w_nk,
            scale_nk,
            x,
            _QMIN,
            _QMAX,
            num_iterations,
            learning_rate,
            reg_param,
            warm_start,
            tuple(beta_range),
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
) -> Tuple[onnx.ModelProto, List[LayerReport]]:
    """GPTQ over int8 QDQ weights, keeping the existing per-channel scales.
    ``act_order`` quantizes columns in descending order of Hessian diagonal."""
    from onnxsim.gptq import _gptq_quantize_columns

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
        refine,
        target_ops,
        max_rows,
        seed,
        providers,
    )


__all__ = ["LayerReport", "adaround_int8", "gptq_int8"]
