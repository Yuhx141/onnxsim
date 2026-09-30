#!/usr/bin/env python3
"""A small pre-LN transformer encoder as a power-of-two QDQ graph the layer engine can compile.

Tokens are pixels: activations are ``[1, hidden, 1, tokens]`` so every Linear is a 1x1 Conv (engine jobs), the
GELU is a pointwise table job between two Q/DQ pairs, and LayerNorm / attention (MatMul, Softmax) stay float host
nodes between engine launches (``layer_engine_host``). Each Linear input is re-quantized from the float residual
stream, so every attention / residual block costs one host round trip.

    python tiny_transformer.py OUT.onnx [--tokens 32] [--hidden 128] [--heads 4] [--layers 2]
    python tiny_transformer.py OUT.onnx --float          # the fp32 reference graph (same weights)

The engine keeps a channel block's tokens in one 512 B core region, so tokens * ceil(channels / 8 / 32) * 8 <= 512
for the widest activation (the 4x FFN width): ``--tokens 32 --hidden 128`` fits exactly.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _pow2_up(value: float) -> float:
    return 2.0 ** math.ceil(math.log2(max(value, 1e-12)))


class Builder:
    """Emits the graph twice: float (calibration) and QDQ (scales taken from the calibration run)."""

    def __init__(self, quant: bool, scales: dict[str, float]):
        self.quant, self.scales = quant, scales
        self.nodes: list = []
        self.inits: list = []
        self.counts: dict[str, int] = {}
        self.taps: list[str] = []  # tensors that get a Q/DQ in the quantized graph
        self.dq_scale: dict[str, float] = {}  # DQ output name -> its scale

    def name(self, prefix: str) -> str:
        self.counts[prefix] = self.counts.get(prefix, 0) + 1  # per prefix: float and QDQ builds name tensors alike
        return f"{prefix}_{self.counts[prefix]}"

    def const(self, array, prefix="c") -> str:
        n = self.name(prefix)
        self.inits.append(numpy_helper.from_array(np.asarray(array), n))
        return n

    def op(self, kind: str, inputs: list[str], **attrs) -> str:
        out = self.name(kind.lower())
        self.nodes.append(helper.make_node(kind, inputs, [out], name=out, **attrs))
        return out

    def qdq(self, tensor: str, signed: bool = True) -> str:
        """Q/DQ point (uint8, zero point 128, power-of-two scale). Float graph: a tap for calibration only."""
        self.taps.append(tensor)
        if not self.quant:
            return tensor
        scale = self.scales[tensor]
        s, z = self.const(np.float32(scale), "s"), self.const(np.uint8(128), "z")
        q = self.op("QuantizeLinear", [tensor, s, z])
        dq = self.op("DequantizeLinear", [q, s, z])
        self.dq_scale[dq] = scale
        return dq

    def linear(self, x: str, weight: np.ndarray, bias: np.ndarray, in_scale: float | None = None, w_scale: float | None = None) -> str:
        """1x1 Conv over [1, C, 1, T]; weights int8 per tensor (power of two), bias int8 (QDQ graph)."""
        if not self.quant:
            w = self.const(weight.reshape(*weight.shape, 1, 1), "w")
            return self.op("Conv", [x, w, self.const(bias, "b")], kernel_shape=[1, 1])
        in_scale = self.dq_scale.get(x, 1.0) if in_scale is None else in_scale
        w_scale = w_scale or _pow2_up(float(np.abs(weight).max()) / 127.0)
        wq = np.clip(np.rint(weight / w_scale), -128, 127).astype(np.int8).reshape(*weight.shape, 1, 1)
        bias_scale = max(_pow2_up(float(np.abs(bias).max()) / 127.0), in_scale * w_scale)
        bq = np.clip(np.rint(bias / bias_scale), -128, 127).astype(np.int8)
        wd = self.op("DequantizeLinear", [self.const(wq, "wq"), self.const(np.float32(w_scale), "ws"), self.const(np.int8(0), "wz")])
        bd = self.op("DequantizeLinear", [self.const(bq, "bq"), self.const(np.float32(bias_scale), "bs"), self.const(np.int8(0), "bz")])
        return self.op("Conv", [x, wd, bd], kernel_shape=[1, 1])


def layer_norm(b: Builder, x: str, gamma: np.ndarray, beta: np.ndarray) -> str:
    """LayerNorm over the channel axis of [1, C, 1, T]."""
    mean = b.op("ReduceMean", [x], axes=[1], keepdims=1)
    centred = b.op("Sub", [x, mean])
    var = b.op("ReduceMean", [b.op("Mul", [centred, centred])], axes=[1], keepdims=1)
    inv = b.op("Sqrt", [b.op("Add", [var, b.const(np.float32(1e-5))])])
    normed = b.op("Div", [centred, inv])
    scaled = b.op("Mul", [normed, b.const(gamma.reshape(1, -1, 1, 1).astype(np.float32))])
    return b.op("Add", [scaled, b.const(beta.reshape(1, -1, 1, 1).astype(np.float32))])


def gelu(b: Builder, x: str) -> str:
    half = b.op("Mul", [x, b.const(np.float32(0.5))])
    erf = b.op("Erf", [b.op("Mul", [x, b.const(np.float32(1 / math.sqrt(2)))])])
    return b.op("Mul", [half, b.op("Add", [erf, b.const(np.float32(1.0))])])


def attention(b: Builder, q: str, k: str, v: str, heads: int, hidden: int, tokens: int) -> str:
    dh = hidden // heads

    def split(t):  # [1, C, 1, T] -> [heads, dh, T]
        return b.op("Reshape", [t, b.const(np.array([heads, dh, tokens], dtype=np.int64))])

    qh, kh, vh = split(q), split(k), split(v)
    scores = b.op("MatMul", [b.op("Transpose", [qh], perm=[0, 2, 1]), kh])  # [heads, T, T]
    scores = b.op("Mul", [scores, b.const(np.float32(1 / math.sqrt(dh)))])
    probs = b.op("Softmax", [scores], axis=-1)
    ctx = b.op("MatMul", [vh, b.op("Transpose", [probs], perm=[0, 2, 1])])  # [heads, dh, T]
    return b.op("Reshape", [ctx, b.const(np.array([1, hidden, 1, tokens], dtype=np.int64))])


def attention_engine(b: Builder, q: str, k: str, v: str, heads: int, hidden: int, tokens: int) -> str:
    """Multi-head attention built only from engine ops on the quantized maps (the 1/sqrt(d) is folded into Wq).

    scores = K^T Q is one activation-matmul job per head set (channel = key, pixel = query); softmax is an exp table,
    a per-head sum (a block-diagonal 1x1 conv of ones), a reciprocal table and an elementwise product; the context
    V P is a second activation-matmul job.
    """
    dh = hidden // heads

    def reshape(t, dims):
        return b.op("Reshape", [t, b.const(np.array(dims, dtype=np.int64))])

    qh, kh, vh = (reshape(t, [heads, dh, tokens]) for t in (q, k, v))
    scores = reshape(b.op("MatMul", [b.op("Transpose", [kh], perm=[0, 2, 1]), qh]), [1, heads * tokens, 1, tokens])
    e = b.qdq(b.op("Exp", [b.qdq(scores)]))
    per_head = np.kron(np.eye(heads, dtype=np.float32), np.ones((tokens, tokens), np.float32))
    total = b.qdq(b.linear(e, per_head, np.zeros(heads * tokens, np.float32), w_scale=1.0))
    p = b.qdq(b.op("Mul", [e, b.qdq(b.op("Reciprocal", [total]))]))
    ctx = reshape(b.op("MatMul", [vh, reshape(p, [heads, tokens, tokens])]), [1, hidden, 1, tokens])
    return b.qdq(ctx)


def layer_norm_engine(b: Builder, x: str, h: int) -> str:
    """LayerNorm (gamma/beta are folded into the next Linear) built only from engine ops on the quantized stream.

    d = x - mean(x) is a dense 1x1 conv (I - 1/C); var = mean(d * d) a conv with 1/C weights after an elementwise
    product job; rsqrt(var + eps) one table job; y = d * rsqrt an elementwise product job. C a power of two keeps
    the 1/C weights exact in int8.
    """
    zeros = np.zeros(h, np.float32)
    w_scale = _pow2_up(1.0 / h)
    d = b.qdq(b.linear(x, np.eye(h, dtype=np.float32) - 1.0 / h, zeros, w_scale=w_scale))
    sq = b.qdq(b.op("Mul", [d, d]))
    var = b.qdq(b.linear(sq, np.full((h, h), 1.0 / h, np.float32), zeros, w_scale=w_scale))
    inv = b.op("Reciprocal", [b.op("Sqrt", [b.op("Add", [var, b.const(np.float32(1e-5))])])])
    return b.qdq(b.op("Mul", [d, b.qdq(inv)]))


def build(quant: bool, scales: dict[str, float], args) -> tuple[onnx.ModelProto, list[str]]:
    rng = np.random.default_rng(args.seed)
    h, t, f = args.hidden, args.tokens, args.hidden * 4
    b = Builder(quant, scales)
    engine_ln = args.ln == "engine"
    w = lambda o, i: (rng.standard_normal((o, i)) / math.sqrt(i)).astype(np.float32)  # noqa: E731
    bias = lambda n: (0.02 * rng.standard_normal(n)).astype(np.float32)  # noqa: E731
    gamma = lambda: (1 + 0.1 * rng.standard_normal(h)).astype(np.float32)  # noqa: E731

    def folded(weight, g, beta, bvec):  # LayerNorm's gamma/beta folded into the Linear that follows it
        return weight * g[None, :], bvec + weight @ beta

    x = b.qdq("input") if engine_ln else "input"
    for layer in range(args.layers):
        if engine_ln:
            y = layer_norm_engine(b, x, h)
            g1, b1 = gamma(), bias(h)
            scale_q = 1.0 / math.sqrt(h // args.heads) if args.attn == "engine" else 1.0  # folded into Wq / bq
            wq, bq = folded(w(h, h), g1, b1, bias(h))
            q = b.qdq(b.linear(y, wq * scale_q, bq * scale_q))
            k, v = (b.qdq(b.linear(y, *folded(w(h, h), g1, b1, bias(h)))) for _ in range(2))
        else:
            xn = layer_norm(b, x, gamma(), bias(h))
            q, k, v = (b.qdq(b.linear(b.qdq(xn), w(h, h), bias(h))) for _ in range(3))
        if engine_ln and args.attn == "engine":
            ctx_q = attention_engine(b, q, k, v, args.heads, h, t)
        else:
            ctx_q = b.qdq(attention(b, q, k, v, args.heads, h, t))
        o = b.qdq(b.linear(ctx_q, w(h, h), bias(h)))
        x = b.qdq(b.op("Add", [x, o])) if engine_ln else b.op("Add", [x, o])  # engine: fused into the o-proj conv
        if engine_ln:
            y2 = layer_norm_engine(b, x, h)
            g2, b2 = gamma(), bias(h)
            up = b.qdq(b.linear(y2, *folded(w(f, h), np.resize(g2, h), np.resize(b2, h), bias(f))))
        else:
            xn2 = layer_norm(b, x, gamma(), bias(h))
            up = b.qdq(b.linear(b.qdq(xn2), w(f, h), bias(f)))
        act = b.qdq(gelu(b, up))
        down = b.qdq(b.linear(act, w(h, f), bias(h)))
        x = b.qdq(b.op("Add", [x, down])) if engine_ln else b.op("Add", [x, down])
    out = layer_norm_engine(b, x, h) if engine_ln else layer_norm(b, x, np.ones(h, np.float32), np.zeros(h, np.float32))
    b.nodes.append(helper.make_node("Identity", [out], ["output"], name="output"))
    graph = helper.make_graph(
        b.nodes,
        "tiny_transformer",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, h, 1, t])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, h, 1, t])],
        initializer=b.inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), b.taps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ln", choices=["engine", "host"], default="engine", help="LayerNorm as engine jobs on the quantized residual stream, or float host nodes")
    parser.add_argument("--attn", choices=["engine", "host"], default="engine", help="attention as engine jobs (needs --ln engine), or float host nodes")
    parser.add_argument("--float", action="store_true")
    args = parser.parse_args()
    fmodel, taps = build(False, {}, args)
    if args.float:
        onnx.save(fmodel, str(args.out))
        return 0
    import onnxruntime as ort

    probe = onnx.ModelProto()
    probe.CopyFrom(fmodel)
    for tap in dict.fromkeys(taps):
        probe.graph.output.append(helper.make_tensor_value_info(tap, TensorProto.FLOAT, None))
    session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    absmax: dict[str, float] = {}
    names = [o.name for o in probe.graph.output][1:]
    for _ in range(16):
        outs = session.run(names, {"input": rng.standard_normal((1, args.hidden, 1, args.tokens)).astype(np.float32)})
        for n, v in zip(names, outs):
            absmax[n] = max(absmax.get(n, 0.0), float(np.abs(v).max()))
    scales = {n: _pow2_up(m / 127.0) for n, m in absmax.items()}
    qmodel, _ = build(True, scales, args)
    qmodel = onnx.shape_inference.infer_shapes(qmodel)
    onnx.save(qmodel, str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
