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
        return self.op("DequantizeLinear", [q, s, z])

    def linear(self, x: str, weight: np.ndarray, bias: np.ndarray, in_scale: float) -> str:
        """1x1 Conv over [1, C, 1, T]; weights int8 per tensor (power of two), bias int8 (QDQ graph)."""
        if not self.quant:
            w = self.const(weight.reshape(*weight.shape, 1, 1), "w")
            return self.op("Conv", [x, w, self.const(bias, "b")], kernel_shape=[1, 1])
        w_scale = _pow2_up(float(np.abs(weight).max()) / 127.0)
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


def build(quant: bool, scales: dict[str, float], args) -> tuple[onnx.ModelProto, list[str]]:
    rng = np.random.default_rng(args.seed)
    h, t, f = args.hidden, args.tokens, args.hidden * 4
    b = Builder(quant, scales)
    x = "input"
    for layer in range(args.layers):
        w = lambda o, i: (rng.standard_normal((o, i)) / math.sqrt(i)).astype(np.float32)  # noqa: E731
        bias = lambda n: (0.02 * rng.standard_normal(n)).astype(np.float32)  # noqa: E731
        gamma = lambda: (1 + 0.1 * rng.standard_normal(h)).astype(np.float32)  # noqa: E731
        xn = layer_norm(b, x, gamma(), bias(h))
        xq = b.qdq(xn)
        s_in = scales.get(xn, 1.0)
        q, k, v = (b.qdq(b.linear(xq, w(h, h), bias(h), s_in)) for _ in range(3))
        ctx = attention(b, q, k, v, args.heads, h, t)
        o = b.qdq(b.linear(b.qdq(ctx), w(h, h), bias(h), scales.get(ctx, 1.0)))
        x = b.op("Add", [x, o])
        xn2 = layer_norm(b, x, gamma(), bias(h))
        xq2 = b.qdq(xn2)
        up_raw = b.linear(xq2, w(f, h), bias(f), scales.get(xn2, 1.0))
        up = b.qdq(up_raw)
        g = gelu(b, up)
        act = b.qdq(g)
        down = b.qdq(b.linear(act, w(h, f), bias(h), scales.get(g, 1.0)))
        x = b.op("Add", [x, down])
    out = layer_norm(b, x, np.ones(h, np.float32), np.zeros(h, np.float32))
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
