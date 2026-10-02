"""AMD Quark's ``FastFinetune`` (AdaRound / AdaQuant) for QDQ models, in numpy.

This is a port of ``quark.onnx.algorithm.finetuning`` (read from the
``amd-quark`` 0.13 wheel; no code copied): the same *blocks*, the same
*training loop*, the same *loss* and the same option names / defaults, so that
``QConfig`` algo configs (``AdaRoundConfig`` / ``AdaQuantConfig``) mean the
same thing here as there. Quark runs it on torch; here every gradient is
hand-derived numpy (float64), so results agree with Quark's *statistically*
(its mini-batches come from ``torch.randperm``), and **exactly** when given the
same mini-batch indices (``tests/test_quark_finetune_parity.py`` feeds torch's
own ``randperm`` stream through ``perm_fn``).

What a "block" is (Quark's ``Subgraph``): for every ``Conv`` / ``ConvTranspose``
/ ``Gemm`` / ``MatMul`` (constant weight) / ``InstanceNormalization`` /
``LayerNormalization`` in the quantized model, the sub-model from the float
tensor in front of the layer's *input* ``QuantizeLinear`` to the layer's
output: input Q/DQ, weight Q/DQ, the op, the (quantized) bias, an optional
following activation (``Relu``, ``LeakyRelu``, ``Clip``, ``Sigmoid``,
``Tanh``, ``Gelu``, ``Softmax``) and -- with ``output_qdq`` -- the output Q/DQ.
The training target is the *float* model's tensor at the same block output,
the input is the quantized model's pre-quantization tensor (``drop_ratio`` mixes
it element-wise with the float model's) and the loss is Quark's
``(||quant - float||_F over dim 1)^2`` averaged over the rest.

Per layer, in graph order, **sequentially**: the quantized model's input
activation is re-captured after every layer's update (``parallel=True``
captures them all once up front, like Quark's ``Parallel``).

* AdaRound learns one rectified-sigmoid ``alpha`` per weight against that loss
  plus the annealed rounding regularizer (cosine ``beta``, ``reg_param``,
  ``warm_start``); the new codes are ``floor(w / s) + (alpha >= 0)``.
* AdaQuant instead trains the float weight (and, with ``update_bias``, the
  quantized bias) directly through a straight-through quantizer with Adam
  (``learning_rate`` defaults to ``1e-5``, as in Quark) and rewrites the codes
  as the quantization of the trained value.

Mini-batches are ``batch_size`` samples drawn without replacement from all
calibration samples (the rows of every calibration batch's leading axis) each
iteration, ``num_iterations`` times; ``early_stop`` is Quark's rule verbatim,
including its quirks (the window is ``num_batches`` iterations when
``num_batches > 1`` else ``num_iterations / 10``; the window's last
iteration's loss is *not* accumulated; AdaRound compares the mean *rounding*
loss, AdaQuant the reconstruction loss; the break happens before that
iteration's optimizer step).

Not replicated (no effect on the numbers, or out of scope): ``optim_device`` /
``infer_device`` / ``num_workers`` / ``pin_memory`` / ``use_gds`` /
``log_period`` / ``cache_dir`` / ``dynamic_batch`` (all calibration data is
always used as one set of samples) and ``mem_opt_level`` (only chooses
between caching float activations up front or per layer); layers Quark itself
cannot convert (``auto_pad``, ``ConvTranspose`` with ``output_padding`` /
groups, 3-D convolutions, ``Gemm`` with ``transA``, activations such as
``PRelu`` whose Quark module ignores the real parameters) are skipped and left
as calibrated. The one deliberate addition is ``guard`` (default on): a layer's
new codes are kept only if the block's reconstruction error on all samples did
not get worse, which Quark has no equivalent of.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.quark_weight_rounding import (
    LayerReport,
    _attr,
    _avg_l2,
    _copy_tensor,
    _im2col,
)

TARGET_OPS = (
    "Conv",
    "ConvTranspose",
    "InstanceNormalization",
    "LayerNormalization",
    "Gemm",
    "MatMul",
)
_ACT_OPS = (
    "Relu",
    "PRelu",
    "LeakyRelu",
    "Gelu",
    "Tanh",
    "Clip",
    "Sigmoid",
    "Softmax",
)
_GAMMA, _ZETA = -0.1, 1.1  # AdaRound's rectified-sigmoid stretch

_RANGES = {
    TensorProto.INT8: (-128.0, 127.0),
    TensorProto.UINT8: (0.0, 255.0),
    TensorProto.INT16: (-32768.0, 32767.0),
    TensorProto.UINT16: (0.0, 65535.0),
    TensorProto.INT32: (-(2.0**31), 2.0**31 - 1),
}


@dataclass
class FinetuneOptions:
    """Quark's ``FastFinetune`` options (``AdaRoundConfig`` / ``AdaQuantConfig``
    names in snake case). ``learning_rate=None`` picks Quark's per-algorithm
    default (``0.1`` AdaRound, ``1e-5`` AdaQuant)."""

    algorithm: str = "adaround"
    num_iterations: int = 1000
    learning_rate: Optional[float] = None
    batch_size: int = 1
    num_batches: int = 1
    early_stop: bool = False
    reg_param: float = 0.01
    beta_range: Tuple[float, float] = (20.0, 2.0)
    warm_start: float = 0.2
    drop_ratio: float = 1.0
    lr_adjust: Optional[Tuple[float, float]] = None
    selective_update: bool = False
    update_bias: bool = False
    output_qdq: bool = False
    parallel: bool = False
    mem_opt_level: int = 1
    output_index: Optional[int] = None
    select_max_mem_layer: bool = False
    target_ops: Sequence[str] = TARGET_OPS
    seed: int = 1705472343
    guard: bool = True

    def lr(self) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)
        return 1e-5 if self.algorithm == "adaquant" else 0.1


# -- quantized constants & activation quantizers -----------------------------------------


@dataclass
class _QConst:
    """A weight / bias quantized as ``DequantizeLinear(int codes, scale, zp)``."""

    name: str  # the integer initializer
    codes: np.ndarray
    scale: np.ndarray  # broadcast to codes.shape
    zp: np.ndarray
    lo: float
    hi: float

    def dequant(self, codes: Optional[np.ndarray] = None) -> np.ndarray:
        c = self.codes if codes is None else codes
        return (c.astype(np.float64) - self.zp) * self.scale

    def ste(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize-dequantize with Quark's straight-through gradient mask."""
        q = np.round(w / self.scale) + self.zp
        mask = (q >= self.lo) & (q <= self.hi)
        return (np.clip(q, self.lo, self.hi) - self.zp) * self.scale, mask

    def encode(self, w: np.ndarray) -> np.ndarray:
        return np.clip(np.round(w / self.scale) + self.zp, self.lo, self.hi)


@dataclass
class _ActQ:
    scale: float
    zp: float
    lo: float
    hi: float
    pre: str = ""  # float tensor feeding the QuantizeLinear (input quantizers)

    def fq(self, x: np.ndarray) -> np.ndarray:
        q = np.clip(np.round(x / self.scale) + self.zp, self.lo, self.hi)
        return (q - self.zp) * self.scale

    def fq_mask(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q = np.round(x / self.scale) + self.zp
        return (np.clip(q, self.lo, self.hi) - self.zp) * self.scale, (
            (q >= self.lo) & (q <= self.hi)
        )


def _broadcast(
    values: np.ndarray, shape: Tuple[int, ...], axis: int
) -> Optional[np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 1:
        return np.full(shape, float(v.reshape(-1)[0]))
    if v.ndim == 1 and shape and v.size == shape[axis % len(shape)]:
        s = [1] * len(shape)
        s[axis % len(shape)] = -1
        return np.broadcast_to(v.reshape(s), shape).copy()
    return None


def _qconst(
    dq: onnx.NodeProto, inits: Dict[str, onnx.TensorProto]
) -> Optional[_QConst]:
    """The folded-Q ``DequantizeLinear`` over an integer initializer, if ``dq``
    is one (per-tensor / per-axis scales only)."""
    if dq.op_type != "DequantizeLinear" or len(dq.input) < 3 or not dq.input[2]:
        return None
    codes_t, scale_t, zp_t = (inits.get(i) for i in dq.input[:3])
    if (
        codes_t is None
        or scale_t is None
        or zp_t is None
        or codes_t.data_type not in _RANGES
        or zp_t.data_type != codes_t.data_type
    ):
        return None
    codes = numpy_helper.to_array(codes_t)
    axis = int(_attr(dq, "axis", 1))
    scale = _broadcast(numpy_helper.to_array(scale_t), codes.shape, axis)
    zp = _broadcast(numpy_helper.to_array(zp_t), codes.shape, axis)
    if scale is None or zp is None or _attr(dq, "block_size", 0):
        return None
    lo, hi = _RANGES[codes_t.data_type]
    return _QConst(codes_t.name, codes, scale, zp, lo, hi)


def _act_quant(
    dq: onnx.NodeProto, inits: Dict[str, onnx.TensorProto], pre: str = ""
) -> Optional[_ActQ]:
    if dq.op_type != "DequantizeLinear" or len(dq.input) < 3 or not dq.input[2]:
        return None
    s, z = inits.get(dq.input[1]), inits.get(dq.input[2])
    if s is None or z is None or z.data_type not in _RANGES:
        return None
    sv, zv = numpy_helper.to_array(s), numpy_helper.to_array(z)
    if sv.size != 1 or zv.size != 1:
        return None
    lo, hi = _RANGES[z.data_type]
    return _ActQ(float(sv.reshape(-1)[0]), float(zv.reshape(-1)[0]), lo, hi, pre=pre)


# -- the compute ops, in natural layout, with hand-derived weight gradients -----------------


class _Op:
    #: axis of the output the bias runs along (None: the last axis)
    bias_axis: Optional[int] = None

    def forward(self, x: np.ndarray, w: np.ndarray):  # -> (y, ctx)
        raise NotImplementedError

    def backward(self, ctx, dy: np.ndarray) -> np.ndarray:  # -> dw
        raise NotImplementedError

    def add_bias(self, y: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.bias_axis is None:
            return y + b
        shape = [1] * y.ndim
        shape[self.bias_axis] = -1
        return y + b.reshape(shape)

    def bias_grad(self, dy: np.ndarray) -> np.ndarray:
        if self.bias_axis is None:
            return dy.reshape(-1, dy.shape[-1]).sum(axis=0)
        axes = tuple(i for i in range(dy.ndim) if i != self.bias_axis)
        return dy.sum(axis=axes)


class _MatMulOp(_Op):
    def __init__(self, transposed: bool) -> None:
        self.t = transposed  # weight stored [N, K]

    def forward(self, x, w):
        wk = w.T if self.t else w
        return x @ wk, x

    def backward(self, ctx, dy):
        x2 = ctx.reshape(-1, ctx.shape[-1])
        dw = x2.T @ dy.reshape(-1, dy.shape[-1])
        return dw.T if self.t else dw


class _ConvOp(_Op):
    bias_axis = 1

    def __init__(self, node: onnx.NodeProto, w_shape: Tuple[int, ...]) -> None:
        nd = len(w_shape) - 2
        self.one_d = nd == 1
        strides = list(_attr(node, "strides", [1] * nd))
        dil = list(_attr(node, "dilations", [1] * nd))
        pads = list(_attr(node, "pads", [0] * (2 * nd)))
        if self.one_d:
            strides, dil = [1] + strides, [1] + dil
            pads = [0, pads[0], 0, pads[1]]
        self.strides, self.dil, self.pads = strides, dil, pads
        self.group = int(_attr(node, "group", 1))

    def _w4(self, w: np.ndarray) -> np.ndarray:
        return w.reshape(w.shape[0], w.shape[1], 1, -1) if self.one_d else w

    def forward(self, x, w):
        w4 = self._w4(w)
        if self.one_d:
            x = x[:, :, None, :]
        o, ig, kh, kw = w4.shape
        og, g = o // self.group, self.group
        pt, pl, pb, pr = self.pads
        b, _, h, wd = x.shape
        oh = (h + pt + pb - self.dil[0] * (kh - 1) - 1) // self.strides[0] + 1
        ow = (wd + pl + pr - self.dil[1] * (kw - 1) - 1) // self.strides[1] + 1
        cols, ys = [], []
        for gi in range(g):
            c = _im2col(
                x[:, gi * ig : (gi + 1) * ig],
                (kh, kw),
                self.strides,
                self.pads,
                self.dil,
            )
            cols.append(c)
            ys.append(c @ w4[gi * og : (gi + 1) * og].reshape(og, -1).T)
        y = np.concatenate(ys, axis=1).reshape(b, oh, ow, o).transpose(0, 3, 1, 2)
        if self.one_d:
            y = y[:, :, 0, :]
        return y, (cols, w4.shape)

    def backward(self, ctx, dy):
        cols, (o, ig, kh, kw) = ctx
        if self.one_d:
            dy = dy[:, :, None, :]
        og = o // self.group
        dyr = dy.transpose(0, 2, 3, 1).reshape(-1, o)
        parts = [
            (dyr[:, gi * og : (gi + 1) * og].T @ cols[gi]).reshape(og, ig, kh, kw)
            for gi in range(self.group)
        ]
        dw = np.concatenate(parts, axis=0)
        return dw[:, :, 0, :] if self.one_d else dw


class _ConvTransposeOp(_Op):
    bias_axis = 1

    def __init__(self, node: onnx.NodeProto, w_shape: Tuple[int, ...]) -> None:
        nd = len(w_shape) - 2
        self.one_d = nd == 1
        strides = list(_attr(node, "strides", [1] * nd))
        dil = list(_attr(node, "dilations", [1] * nd))
        pads = list(_attr(node, "pads", [0] * (2 * nd)))
        if self.one_d:
            strides, dil = [1] + strides, [1] + dil
            pads = [0, pads[0], 0, pads[1]]
        self.strides, self.dil, self.pads = strides, dil, pads

    def _geometry(self, h: int, w: int, kh: int, kw: int):
        sh, sw = self.strides
        dh, dw = self.dil
        return (
            sh * (h - 1) + dh * (kh - 1) + 1,
            sw * (w - 1) + dw * (kw - 1) + 1,
        )

    def forward(self, x, w):
        if self.one_d:
            x, w = x[:, :, None, :], w[:, :, None, :]
        b, c, h, wd = x.shape
        _, co, kh, kw = w.shape
        hp, wp = self._geometry(h, wd, kh, kw)
        sh, sw = self.strides
        dh, dw_ = self.dil
        cols = np.einsum("bchw,cokl->bhwokl", x, w)
        buf = np.zeros((b, co, hp, wp))
        for i in range(kh):
            for j in range(kw):
                buf[
                    :,
                    :,
                    i * dh : i * dh + sh * h : sh,
                    j * dw_ : j * dw_ + sw * wd : sw,
                ] += cols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        pt, pl, pb, pr = self.pads
        y = buf[:, :, pt : hp - pb, pl : wp - pr]
        if self.one_d:
            y = y[:, :, 0, :]
        return y, (x, w.shape, (hp, wp))

    def backward(self, ctx, dy):
        x, (_, co, kh, kw), (hp, wp) = ctx
        if self.one_d:
            dy = dy[:, :, None, :]
        b, c, h, wd = x.shape
        pt, pl, pb, pr = self.pads
        full = np.zeros((b, co, hp, wp))
        full[:, :, pt : hp - pb, pl : wp - pr] = dy
        sh, sw = self.strides
        dh, dw_ = self.dil
        dcols = np.empty((b, h, wd, co, kh, kw))
        for i in range(kh):
            for j in range(kw):
                dcols[:, :, :, :, i, j] = full[
                    :,
                    :,
                    i * dh : i * dh + sh * h : sh,
                    j * dw_ : j * dw_ + sw * wd : sw,
                ].transpose(0, 2, 3, 1)
        dw = np.einsum("bchw,bhwokl->cokl", x, dcols)
        return dw[:, :, 0, :] if self.one_d else dw


class _LayerNormOp(_Op):
    def __init__(self, eps: float) -> None:
        self.eps = eps

    def forward(self, x, w):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mean) / np.sqrt(var + self.eps)
        return xhat * w, xhat

    def backward(self, ctx, dy):
        return (dy * ctx).reshape(-1, dy.shape[-1]).sum(axis=0)


class _InstanceNormOp(_Op):
    bias_axis = 1

    def __init__(self, eps: float) -> None:
        self.eps = eps

    def forward(self, x, w):
        axes = tuple(range(2, x.ndim))
        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
        xhat = (x - mean) / np.sqrt(var + self.eps)
        shape = [1] * x.ndim
        shape[1] = -1
        return xhat * w.reshape(shape), xhat

    def backward(self, ctx, dy):
        axes = tuple(i for i in range(dy.ndim) if i != 1)
        return (dy * ctx).sum(axis=axes)


# -- activations after the op ----------------------------------------------------------------


def _erf(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf, otypes=[np.float64])(x)


class _Act:
    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, z: np.ndarray, a: np.ndarray, da: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _Relu(_Act):
    def forward(self, z):
        return np.maximum(z, 0.0)

    def backward(self, z, a, da):
        return da * (z > 0)


class _LeakyRelu(_Act):
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def forward(self, z):
        return np.where(z > 0, z, self.alpha * z)

    def backward(self, z, a, da):
        return da * np.where(z > 0, 1.0, self.alpha)


class _Clip(_Act):
    def __init__(self, lo: float, hi: float) -> None:
        self.lo, self.hi = lo, hi

    def forward(self, z):
        return np.clip(z, self.lo, self.hi)

    def backward(self, z, a, da):
        return da * ((z >= self.lo) & (z <= self.hi))


class _Sigmoid(_Act):
    def forward(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, z, a, da):
        return da * a * (1.0 - a)


class _Tanh(_Act):
    def forward(self, z):
        return np.tanh(z)

    def backward(self, z, a, da):
        return da * (1.0 - a * a)


class _Gelu(_Act):
    # Quark's module is torch.nn.GELU() whatever the node's ``approximate``
    def forward(self, z):
        return 0.5 * z * (1.0 + _erf(z / math.sqrt(2.0)))

    def backward(self, z, a, da):
        pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return da * (0.5 * (1.0 + _erf(z / math.sqrt(2.0))) + z * pdf)


class _Softmax(_Act):
    def __init__(self, axis: int) -> None:
        self.axis = axis

    def forward(self, z):
        e = np.exp(z - z.max(axis=self.axis, keepdims=True))
        return e / e.sum(axis=self.axis, keepdims=True)

    def backward(self, z, a, da):
        return a * (da - (da * a).sum(axis=self.axis, keepdims=True))


def _make_act(
    node: onnx.NodeProto, inits: Dict[str, onnx.TensorProto]
) -> Optional[_Act]:
    t = node.op_type
    if t == "Relu":
        return _Relu()
    if t == "LeakyRelu":
        return _LeakyRelu(float(_attr(node, "alpha", 0.01)))
    if t == "Sigmoid":
        return _Sigmoid()
    if t == "Tanh":
        return _Tanh()
    if t == "Gelu":
        return _Gelu()
    if t == "Softmax":
        return _Softmax(int(_attr(node, "axis", -1)))
    if t == "Clip":
        if len(node.input) == 3 and node.input[1] in inits and node.input[2] in inits:
            lo = float(numpy_helper.to_array(inits[node.input[1]]).reshape(-1)[0])
            hi = float(numpy_helper.to_array(inits[node.input[2]]).reshape(-1)[0])
            return _Clip(lo, hi)
        if len(node.input) == 1:
            lo, hi = _attr(node, "min", None), _attr(node, "max", None)
            if lo is None or hi is None:
                return None  # Quark's Clip module is the identity here
            return _Clip(float(lo), float(hi))
    return None  # PRelu etc.: Quark's module ignores the real parameters


# -- finding the blocks ------------------------------------------------------------------------


@dataclass
class _Block:
    name: str
    op_type: str
    op: _Op
    w_float: np.ndarray
    qw: _QConst
    qb: Optional[_QConst]  # quantized bias
    b_float: Optional[np.ndarray]  # the float model's bias (what AdaQuant trains)
    b_plain: Optional[np.ndarray]  # an unquantized bias, used as is
    w_alpha: float
    b_beta: float
    in_q: Optional[_ActQ]
    q_start: str  # quantized-model tensor the block starts from (pre input Q)
    f_start: str
    f_end: str
    act: Optional[_Act]
    out_q: Optional[_ActQ]


def _consumers(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    out: Dict[str, onnx.NodeProto] = {}
    for n in model.graph.node:
        for i in n.input:
            out.setdefault(i, n)  # Quark's converter takes the first consumer
    return out


def _find_blocks(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    opt: FinetuneOptions,
) -> List[_Block]:
    f_inits = {t.name: t for t in float_model.graph.initializer}
    q_inits = {t.name: t for t in quant_model.graph.initializer}
    q_prod = {o: n for n in quant_model.graph.node for o in n.output}
    q_cons = _consumers(quant_model)
    f_cons = _consumers(float_model)
    f_by_name: Dict[str, List[onnx.NodeProto]] = {}
    f_by_weight: Dict[Tuple[str, str], onnx.NodeProto] = {}
    for fn in float_model.graph.node:
        f_by_name.setdefault(fn.name, []).append(fn)
        if len(fn.input) > 1:
            f_by_weight[(fn.op_type, fn.input[1])] = fn

    use_count: Dict[str, int] = {}
    for n in quant_model.graph.node:
        for i in n.input:
            use_count[i] = use_count.get(i, 0) + 1

    blocks: List[_Block] = []
    for qn in quant_model.graph.node:
        if qn.op_type not in opt.target_ops or qn.op_type not in TARGET_OPS:
            continue
        if len(qn.input) < 2 or qn.output[0] not in q_cons:
            continue
        blk = _make_block(
            qn,
            q_prod,
            q_inits,
            q_cons,
            f_inits,
            f_cons,
            f_by_name,
            f_by_weight,
            use_count,
            opt,
        )
        if blk is not None:
            blocks.append(blk)
    seen: Dict[str, int] = {}
    for b in blocks:
        seen[b.qw.name] = seen.get(b.qw.name, 0) + 1
    return [b for b in blocks if seen[b.qw.name] == 1]


def _make_block(
    qn: onnx.NodeProto,
    q_prod: Dict[str, onnx.NodeProto],
    q_inits: Dict[str, onnx.TensorProto],
    q_cons: Dict[str, onnx.NodeProto],
    f_inits: Dict[str, onnx.TensorProto],
    f_cons: Dict[str, onnx.NodeProto],
    f_by_name: Dict[str, List[onnx.NodeProto]],
    f_by_weight: Dict[Tuple[str, str], onnx.NodeProto],
    use_count: Dict[str, int],
    opt: FinetuneOptions,
) -> Optional[_Block]:
    t = qn.op_type
    # -- the weight: DequantizeLinear over an integer initializer ---------------------
    wdq = q_prod.get(qn.input[1])
    qw = None if wdq is None else _qconst(wdq, q_inits)
    if (
        qw is None
        or use_count.get(qw.name, 0) != 1
        or use_count.get(qn.input[1], 0) != 1
    ):
        return None
    # -- the float node: by name, else through the weight's name ------------------------
    fn = None
    cands = f_by_name.get(qn.name, []) if qn.name else []
    if len(cands) == 1 and cands[0].op_type == t:
        fn = cands[0]
    if fn is None:
        fn = f_by_weight.get((t, wdq.input[0].split("/qdq")[0].split("_quantized")[0]))
    if fn is None or len(fn.input) < 2 or fn.input[1] not in f_inits:
        return None
    w_float = numpy_helper.to_array(f_inits[fn.input[1]]).astype(np.float64)
    if (
        w_float.shape != qw.codes.shape
        or f_inits[fn.input[1]].data_type != TensorProto.FLOAT
    ):
        return None

    # -- the input: DequantizeLinear <- QuantizeLinear (float tensor "pre") -----------------
    in_dq = q_prod.get(qn.input[0])
    if in_dq is None:
        return None  # Quark: no producer, no block
    in_q: Optional[_ActQ] = None
    q_start = qn.input[0]
    if in_dq.op_type == "DequantizeLinear":
        q_node = q_prod.get(in_dq.input[0])
        if q_node is not None and q_node.op_type == "QuantizeLinear":
            in_q = _act_quant(in_dq, q_inits, pre=q_node.input[0])
            if in_q is None:
                return None
            q_start = q_node.input[0]

    # -- op, layout, attributes -----------------------------------------------------------------
    op: _Op
    w_alpha, b_beta = 1.0, 1.0
    if t == "MatMul":
        if w_float.ndim != 2:
            return None
        op = _MatMulOp(False)
    elif t == "Gemm":
        if _attr(fn, "transA", 0):
            return None
        if w_float.ndim != 2:
            return None
        op = _MatMulOp(bool(_attr(fn, "transB", 0)))
        w_alpha, b_beta = float(_attr(fn, "alpha", 1.0)), float(_attr(fn, "beta", 1.0))
    elif t in ("Conv", "ConvTranspose"):
        if w_float.ndim not in (3, 4):
            return None
        if _attr(fn, "auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            return None
        if t == "Conv":
            op = _ConvOp(fn, w_float.shape)
        else:
            if (
                int(_attr(fn, "group", 1)) != 1
                or any(_attr(fn, "output_padding", [0]))
                or _attr(fn, "output_shape", None)
            ):
                return None
            op = _ConvTransposeOp(fn, w_float.shape)
    elif t == "LayerNormalization":
        if int(_attr(fn, "axis", -1)) != -1 or w_float.ndim != 1:
            return None
        op = _LayerNormOp(float(_attr(fn, "epsilon", 1e-5)))
    else:  # InstanceNormalization
        if w_float.ndim != 1:
            return None
        op = _InstanceNormOp(float(_attr(fn, "epsilon", 1e-5)))

    # -- bias ------------------------------------------------------------------------------------
    qb: Optional[_QConst] = None
    b_float: Optional[np.ndarray] = None
    b_plain: Optional[np.ndarray] = None
    if len(qn.input) > 2 and qn.input[2]:
        bdq = q_prod.get(qn.input[2])
        if bdq is not None:
            qb = _qconst(bdq, q_inits)
            if qb is None:
                return None
            if len(fn.input) < 3 or fn.input[2] not in f_inits:
                return None
            b_float = numpy_helper.to_array(f_inits[fn.input[2]]).astype(np.float64)
            if b_float.shape != qb.codes.shape:
                return None
        elif qn.input[2] in q_inits:
            b_plain = numpy_helper.to_array(q_inits[qn.input[2]]).astype(np.float64)
        else:
            return None
    elif t == "InstanceNormalization":
        return None

    # -- the end of the block ----------------------------------------------------------------------
    cons = q_cons.get(qn.output[0])
    if cons is None:
        return None
    act: Optional[_Act] = None
    out_q: Optional[_ActQ] = None
    f_end = fn.output[0]
    tail = qn.output[0]
    if cons.op_type in _ACT_OPS:
        act = _make_act(cons, q_inits)
        fcons = f_cons.get(fn.output[0])
        if act is None or fcons is None or fcons.op_type != cons.op_type:
            return None
        f_end, tail = fcons.output[0], cons.output[0]
    if opt.output_qdq:
        nxt = q_cons.get(tail)
        if nxt is not None and nxt.op_type == "QuantizeLinear":
            dqn = q_cons.get(nxt.output[0])
            if dqn is None or dqn.op_type != "DequantizeLinear":
                return None  # Quark: a Q without its DQ is an error, layer skipped
            out_q = _act_quant(dqn, q_inits)
            if out_q is None:
                return None
    return _Block(
        qn.name or qn.output[0],
        t,
        op,
        w_float,
        qw,
        qb,
        b_float,
        b_plain,
        w_alpha,
        b_beta,
        in_q,
        q_start,
        fn.input[0],
        f_end,
        act,
        out_q,
    )


# -- capturing activations -------------------------------------------------------------------------


def _capture(
    model: onnx.ModelProto,
    names: Sequence[str],
    data: Sequence[Dict[str, np.ndarray]],
    providers: Optional[Sequence[str]],
    optimize: bool,
) -> Dict[str, np.ndarray]:
    """``{name: [samples, ...]}``: each tensor over all calibration batches,
    concatenated along the leading axis. ``optimize=False`` runs ORT without
    graph optimizations (Quark does for the quantized model: ORT would
    otherwise fuse DQ -> op -> Q into integer kernels)."""
    import onnxruntime as ort

    names = sorted(set(names))
    probe = _add_probe_outputs(model, names)
    so = ort.SessionOptions()
    if not optimize:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        probe.SerializeToString(),
        so,
        providers=list(providers or ["CPUExecutionProvider"]),
    )
    outs = [o.name for o in sess.get_outputs()]
    acc: Dict[str, List[np.ndarray]] = {n: [] for n in names}
    for batch in data:
        res = dict(zip(outs, sess.run(outs, batch)))
        for n in names:
            acc[n].append(np.asarray(res[n], dtype=np.float64))
    return {n: np.concatenate(v, axis=0) for n, v in acc.items()}


def _model_outputs(
    model: onnx.ModelProto,
    data: Sequence[Dict[str, np.ndarray]],
    providers: Optional[Sequence[str]],
    output_index: Optional[int],
) -> List[List[np.ndarray]]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(),
        so,
        providers=list(providers or ["CPUExecutionProvider"]),
    )
    n_out = len(sess.get_outputs())
    res = [sess.run(None, b) for b in data]
    if output_index is not None and 0 <= output_index < n_out:
        return [[r[output_index]] for r in res]
    return res


# -- training ------------------------------------------------------------------------------------------


def _adam_step(p, g, m, v, t, lr):
    m *= 0.9
    m += 0.1 * g
    v *= 0.999
    v += 0.001 * g * g
    mh = m / (1.0 - 0.9 ** (t + 1))
    vh = v / (1.0 - 0.999 ** (t + 1))
    return p - lr * mh / (np.sqrt(vh) + 1e-8)


def _beta(max_iter: int, it: int, beta_range, warm_start: float) -> float:
    start, end = beta_range
    ws = warm_start * max_iter
    rel = (it - ws) / (max_iter - ws)
    return end + 0.5 * (start - end) * (1 + math.cos(rel * math.pi))


def _block_forward(
    blk: _Block,
    x_in: np.ndarray,
    w_hat: np.ndarray,
    bias: Optional[np.ndarray],
    out_mask: bool = False,
):
    """``x_in`` is already fake-quantized. Returns ``(y, cache)``."""
    z, ctx = blk.op.forward(x_in, w_hat * blk.w_alpha)
    if bias is not None:
        z = blk.op.add_bias(z, bias * blk.b_beta)
    a = z if blk.act is None else blk.act.forward(z)
    mask = None
    y = a
    if blk.out_q is not None:
        y, mask = blk.out_q.fq_mask(a)
    return y, (ctx, z, a, mask)


def _recon_grad(
    blk: _Block, cache, y: np.ndarray, y_ref: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Quark's loss ``mean(sum((y - y_ref)^2, dim=1))`` and the gradients of it
    w.r.t. the quantized-weight tensor and the bias."""
    ctx, z, a, mask = cache
    err = y - y_ref
    denom = err.size / err.shape[1]
    loss = float(np.sum(err * err) / denom)
    dy = 2.0 * err / denom
    if mask is not None:
        dy = dy * mask
    dz = dy if blk.act is None else blk.act.backward(z, a, dy)
    dw = blk.op.backward(ctx, dz) * blk.w_alpha
    db = blk.op.bias_grad(dz) * blk.b_beta
    return loss, dw, db


def _eval_error(
    blk: _Block,
    x_all: np.ndarray,
    y_all: np.ndarray,
    w_hat: np.ndarray,
    bias: Optional[np.ndarray],
) -> float:
    """Quark's ``_calc_recons_metrics``: plain MSE over every element."""
    tot, count = 0.0, 0
    for i in range(0, x_all.shape[0], 64):
        y, _ = _block_forward(blk, x_all[i : i + 64], w_hat, bias)
        d = y - y_all[i : i + 64]
        tot += float(np.sum(d * d))
        count += d.size
    return tot / max(count, 1)


@dataclass
class _Trained:
    codes: np.ndarray
    bias_codes: Optional[np.ndarray] = None
    err_rtn: float = 0.0  # Quark's "initial" error (hard-rounded float weight)
    iterations: int = 0


def _train_block(
    blk: _Block,
    xq: np.ndarray,
    xf: np.ndarray,
    yf: np.ndarray,
    opt: FinetuneOptions,
    perm_fn: Callable[[int], np.ndarray],
    rng: np.random.Generator,
) -> _Trained:
    qw = blk.qw
    s_total = xq.shape[0]
    adaround = opt.algorithm == "adaround"
    in_fq = (lambda x: x) if blk.in_q is None else blk.in_q.fq
    x_eval = in_fq(xq)
    # the bias the layer is trained with: Quark feeds the float bias through
    # the bias quantizer (identical to the model's own codes unless AdaQuant
    # updates it)
    b_float = blk.b_float
    bias_fq = blk.b_plain if blk.qb is None else blk.qb.ste(blk.b_float)[0]  # type: ignore[arg-type]

    num_iter = int(opt.num_iterations)
    lr = opt.lr()
    w = blk.w_float
    scale, zp, lo, hi = qw.scale, qw.zp, qw.lo, qw.hi

    # initial (hard-rounded float weight) error: drives LRAdjust
    w_rtn = (
        np.clip(
            np.floor(w / scale) + (w / scale - np.floor(w / scale) >= 0.5) + zp, lo, hi
        )
        - zp
    ) * scale
    if adaround:
        w_init_hat = w_rtn
    else:
        w_init_hat = qw.ste(w)[0]
    err0 = _eval_error(blk, x_eval, yf, w_init_hat, bias_fq)
    if (
        opt.lr_adjust is not None
        and len(opt.lr_adjust) == 2
        and err0 > opt.lr_adjust[0]
    ):
        lr = float(opt.lr_adjust[1])

    bs = int(opt.batch_size)
    if bs < 1 or bs > s_total:
        bs = 1

    # parameters
    if adaround:
        diff = w / scale - np.floor(w / scale)
        alpha = -np.log((_ZETA - _GAMMA) / (diff - _GAMMA) - 1.0)
        params = [alpha]
    else:
        wv = w.copy()
        bv = (
            b_float.copy()
            if (opt.update_bias and blk.qb is not None and b_float is not None)
            else None
        )
        params = [wv] + ([bv] if bv is not None else [])
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]

    best_loss = float("inf")
    mean_loss = 0.0
    es_window = opt.num_batches if opt.num_batches > 1 else num_iter / 10
    ws_iter = num_iter * opt.warm_start
    pre_mixed = None
    if opt.drop_ratio >= 1:
        pre_mixed = x_eval
    elif opt.drop_ratio <= 0:
        pre_mixed = in_fq(xf)

    done = 0
    for it in range(num_iter):
        idx = perm_fn(s_total)[:bs]
        if pre_mixed is not None:
            x_in = pre_mixed[idx]
        else:
            xqb, xfb = xq[idx], xf[idx]
            x_in = in_fq(np.where(rng.random(xqb.shape) < opt.drop_ratio, xqb, xfb))
        y_ref = yf[idx]

        if adaround:
            sig = 1.0 / (1.0 + np.exp(-params[0]))
            raw_h = sig * (_ZETA - _GAMMA) + _GAMMA
            h = np.clip(raw_h, 0.0, 1.0)
            raw_q = np.floor(w / scale) + h + zp
            w_hat = (np.clip(raw_q, lo, hi) - zp) * scale
            bias = bias_fq
        else:
            w_hat, wmask = qw.ste(params[0])
            if len(params) > 1:
                bias, bmask = blk.qb.ste(params[1])  # type: ignore[union-attr]
            else:
                bias, bmask = bias_fq, None

        y, cache = _block_forward(blk, x_in, w_hat, bias)
        recons, dw_hat, db = _recon_grad(blk, cache, y, y_ref)

        round_loss = 0.0
        grads: List[np.ndarray]
        if adaround:
            dq_mask = (raw_q >= lo) & (raw_q <= hi)
            dh = dw_hat * scale * dq_mask
            h_mask = (raw_h >= 0.0) & (raw_h <= 1.0)
            dh_dalpha = np.where(h_mask, sig * (1.0 - sig) * (_ZETA - _GAMMA), 0.0)
            if it >= ws_iter:
                beta = _beta(num_iter, it, opt.beta_range, opt.warm_start)
                u = 2.0 * h - 1.0
                round_loss = opt.reg_param * float(np.sum(1.0 - np.abs(u) ** beta))
                dreg = (
                    -2.0 * opt.reg_param * beta * np.sign(u) * np.abs(u) ** (beta - 1.0)
                )
                grads = [(dw_hat * scale * dq_mask + dreg) * dh_dalpha]
            else:
                grads = [dh * dh_dalpha]
        else:
            grads = [dw_hat * wmask]
            if len(params) > 1:
                grads.append(db * bmask)  # type: ignore[operator]

        # Quark's early-stop rule, verbatim (it reuses num_batches / warm_start)
        if opt.early_stop and it >= ws_iter:
            if it % es_window == es_window - 1:
                mean_loss = mean_loss / es_window
                if mean_loss < best_loss:
                    best_loss = mean_loss
                else:
                    break
                mean_loss = 0.0
            else:
                mean_loss += round_loss if adaround else recons

        for k, p in enumerate(params):
            params[k] = _adam_step(p, grads[k], ms[k], vs[k], it, lr)
        done = it + 1

    if adaround:
        new_codes = np.clip(np.floor(w / scale) + (params[0] >= 0) + zp, lo, hi)
        return _Trained(new_codes, None, err0, done)
    codes = qw.encode(params[0])
    bcodes = blk.qb.encode(params[1]) if len(params) > 1 else None  # type: ignore[union-attr]
    return _Trained(codes, bcodes, err0, done)


# -- the driver --------------------------------------------------------------------------------------


def _shape_ok(blk: _Block, x: np.ndarray) -> bool:
    t = blk.op_type
    if t == "Gemm":
        return x.ndim == 2
    if t == "MatMul":
        return x.ndim in (2, 3)
    if t in ("Conv", "ConvTranspose"):
        return x.ndim == blk.w_float.ndim
    if t == "InstanceNormalization":
        return x.ndim >= 3
    return x.ndim >= 2  # LayerNormalization


def _estimate_memory(blk: _Block, y_shape: Tuple[int, ...]) -> float:
    """Quark's ``estimate_memory`` for one block (MiB): weights + bias, the
    outputs of its torch children (compute, activation, output Q/DQ) and the
    Adam state (3 x parameters), all float32."""
    params = blk.w_float.size
    if blk.b_float is not None:
        params += blk.b_float.size
    elif blk.b_plain is not None:
        params += blk.b_plain.size
    n_out = int(np.prod(y_shape))
    children = 1 + (blk.act is not None) + (blk.out_q is not None)
    mib = 1024.0**2
    return params * 4 / mib + children * n_out * 4 / mib + 3 * params * 4 / mib


def finetune(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    options: Optional[FinetuneOptions] = None,
    providers: Optional[Sequence[str]] = None,
    perm_fn: Optional[Callable[[int], np.ndarray]] = None,
) -> Tuple[onnx.ModelProto, List[LayerReport]]:
    """Quark ``FastFinetune`` over a QDQ model (see the module docstring).

    ``float_model`` is the reference (its tensors are the training targets),
    ``quant_model`` the QDQ model whose integer weight (and, for AdaQuant with
    ``update_bias``, bias) codes get rewritten. Returns the new model and one
    :class:`~onnxsim.quark_weight_rounding.LayerReport` per block.

    ``perm_fn(n)`` returns a permutation of ``range(n)``; the first
    ``batch_size`` entries are an iteration's mini-batch. The default is a
    seeded numpy generator; pass torch's ``randperm`` stream to replay Quark's
    mini-batches exactly.
    """
    opt = options or FinetuneOptions()
    if opt.algorithm not in ("adaround", "adaquant"):
        raise ValueError(f"unknown algorithm {opt.algorithm!r}")
    if not calibration_data:
        raise ValueError("calibration_data is required")
    out = onnx.ModelProto()
    out.CopyFrom(quant_model)
    blocks = _find_blocks(float_model, quant_model, opt)
    if not blocks:
        return out, []

    if perm_fn is None:
        perm_rng = np.random.default_rng(opt.seed)
        perm_fn = perm_rng.permutation
    mix_rng = np.random.default_rng([opt.seed, 1])

    f_cache: Dict[str, np.ndarray] = {}
    if opt.select_max_mem_layer:
        # Quark estimates every block from one float sample and finetunes the
        # most memory-hungry one only
        probe = _capture(
            float_model,
            sorted({b.f_end for b in blocks}),
            calibration_data[:1],
            providers,
            True,
        )
        mems = [_estimate_memory(b, (1,) + probe[b.f_end].shape[1:]) for b in blocks]
        blocks = [blocks[int(np.argmax(mems))]]
    if opt.mem_opt_level == 0:
        f_cache = _capture(
            float_model,
            [n for b in blocks for n in (b.f_start, b.f_end)],
            calibration_data,
            providers,
            True,
        )
    q_parallel: Dict[str, np.ndarray] = {}
    if opt.parallel:
        q_parallel = _capture(
            quant_model,
            [b.q_start for b in blocks],
            calibration_data,
            providers,
            False,
        )

    inits = {t.name: t for t in out.graph.initializer}
    f_out: List[List[np.ndarray]] = []
    l2 = 0.0
    if opt.selective_update:
        f_out = _model_outputs(
            float_model, calibration_data, providers, opt.output_index
        )
        l2 = _avg_l2(
            f_out,
            _model_outputs(out, calibration_data, providers, opt.output_index),
        )

    reports: List[LayerReport] = []
    for blk in blocks:
        fc = f_cache or _capture(
            float_model, [blk.f_start, blk.f_end], calibration_data, providers, True
        )
        xf, yf = fc[blk.f_start], fc[blk.f_end]
        if blk.q_start in q_parallel:
            xq = q_parallel[blk.q_start]
        else:
            xq = _capture(out, [blk.q_start], calibration_data, providers, False)[
                blk.q_start
            ]
        if xq.shape != xf.shape or not _shape_ok(blk, xq):
            continue
        res = _train_block(blk, xq, xf, yf, opt, perm_fn, mix_rng)

        x_eval = xq if blk.in_q is None else blk.in_q.fq(xq)
        qw, qb = blk.qw, blk.qb
        cur_bias = blk.b_plain if qb is None else qb.dequant()
        before = _eval_error(blk, x_eval, yf, qw.dequant(), cur_bias)
        if qb is not None and res.bias_codes is not None:
            new_bias = qb.dequant(res.bias_codes)
        else:
            new_bias = cur_bias
        after = _eval_error(blk, x_eval, yf, qw.dequant(res.codes), new_bias)
        accepted = after <= before or not opt.guard
        changed = float(np.mean(res.codes != qw.codes))
        undo: List[Tuple[str, onnx.TensorProto]] = []
        if accepted:
            writes = [(qw.name, res.codes)]
            if qb is not None and res.bias_codes is not None:
                writes.append((qb.name, res.bias_codes))
            for name, codes in writes:
                undo.append((name, _copy_tensor(inits[name])))
                dtype = numpy_helper.to_array(inits[name]).dtype
                inits[name].CopyFrom(numpy_helper.from_array(codes.astype(dtype), name))
        if opt.selective_update and undo:
            new_l2 = _avg_l2(
                f_out,
                _model_outputs(out, calibration_data, providers, opt.output_index),
            )
            if new_l2 < l2:
                l2 = new_l2
            else:
                for name, prev in undo:
                    inits[name].CopyFrom(prev)
                accepted, changed = False, 0.0
        reports.append(
            LayerReport(
                blk.name,
                blk.op_type,
                tuple(qw.codes.shape),
                before,
                after if accepted else before,
                accepted,
                changed,
            )
        )
    return out, reports


__all__ = ["FinetuneOptions", "TARGET_OPS", "finetune"]
