"""Use tinygrad's ONNX frontend to lower operators the layer engine has no kernel for.

The engine's activations are 8-bit, so any *unary elementwise* operator (SiLU, HardSwish, Sigmoid, GELU,
Tanh, Erf, Softplus, Mish, ...) reduces to one 256-entry byte table, run by the engine's ``lut`` job. This
module builds that table from the operator's own float definition by executing the single-node ONNX model
through tinygrad (``tinygrad.nn.onnx.OnnxRunner``), i.e. tinygrad supplies the semantics for every op it
implements instead of one hand-written table generator per op. It also *detects* which nodes are pointwise
unary by execution (a permutation of the input permutes the output), and classifies a whole model.

tinygrad is optional: point ``ONNXSIM_TINYGRAD`` (or PYTHONPATH) at a checkout; without it these helpers raise
``TinygradUnavailable`` and the engine falls back to the ops it supports natively.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


class TinygradUnavailable(RuntimeError):
    pass


def _tinygrad():
    # The pure-Python device is enough for 256-element tables and needs no host compiler.
    os.environ.setdefault("DEV", "PYTHON")
    try:
        root = os.environ.get("ONNXSIM_TINYGRAD")
        if root and root not in sys.path:
            sys.path.insert(0, root)
        from tinygrad import Tensor
        from tinygrad.nn.onnx import OnnxRunner
    except Exception as exc:  # pragma: no cover - environment dependent
        raise TinygradUnavailable(str(exc)) from exc
    return Tensor, OnnxRunner


def run_node(op_type: str, x: np.ndarray, attrs: dict[str, Any] | None = None, opset: int = 17, extra: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Execute ONE ONNX node on ``x`` (float32) through tinygrad and return its output."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    Tensor, OnnxRunner = _tinygrad()
    extra = extra or {}
    node = helper.make_node(op_type, ["x", *extra], ["y"], **(attrs or {}))
    graph = helper.make_graph(
        [node], "single", [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializer=[numpy_helper.from_array(v, k) for k, v in extra.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "node.onnx"
        onnx.save(model, str(path))
        return OnnxRunner(str(path))({"x": Tensor(x.astype(np.float32))})["y"].numpy()


def is_pointwise_unary(op_type: str, attrs: dict[str, Any] | None = None, extra: dict[str, np.ndarray] | None = None) -> bool:
    """True when the node computes ``y[i] = f(x[i])``: permuting the input permutes the output identically."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, (1, 8, 4, 4)).astype(np.float32)
    try:
        y = run_node(op_type, x, attrs, extra=extra)
    except Exception:
        return False
    if y.shape != x.shape:
        return False
    perm = rng.permutation(x.size)
    yp = run_node(op_type, x.reshape(-1)[perm].reshape(x.shape), attrs, extra=extra)
    return bool(np.array_equal(yp.reshape(-1), y.reshape(-1)[perm]))


def unary_table(op_type: str, in_scale: float, in_zero: int, in_signed: bool, out_scale: float, out_zero: int, out_signed: bool,
                attrs: dict[str, Any] | None = None, extra: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """256 output bytes: ``table[b]`` = quantize(f(dequantize(b))) with the input/output byte encodings given."""
    raw = np.arange(256, dtype=np.uint8)
    values = raw.view(np.int8).astype(np.int64) if in_signed else raw.astype(np.int64)
    x = ((values - in_zero) * in_scale).astype(np.float32)
    y = run_node(op_type, x.reshape(1, 1, 16, 16), attrs, extra=extra).reshape(-1).astype(np.float64)
    lo, hi = (-128, 127) if out_signed else (0, 255)
    q = np.clip(np.rint(y / out_scale) + out_zero, lo, hi).astype(np.int64)
    return (q.astype(np.int8).view(np.uint8) if out_signed else q.astype(np.uint8))


ENGINE_NATIVE = frozenset({"Conv", "Relu", "Clip", "Add", "MaxPool", "GlobalAveragePool", "Flatten", "Gemm", "QuantizeLinear", "DequantizeLinear", "Constant", "Identity"})


def classify(model: Any) -> dict[str, Any]:
    """Which ONNX nodes the layer engine can run: natively, as a depthwise job, as a tinygrad-built table, or not."""
    counts: Counter[str] = Counter()
    detail: dict[str, str] = {}
    cache: dict[str, bool] = {}
    for node in model.graph.node:
        op = node.op_type
        if op == "Conv":
            group = next((a.i for a in node.attribute if a.name == "group"), 1)
            counts["conv_depthwise" if group > 1 else "conv_dense"] += 1
            if group > 1:
                detail.setdefault("Conv(group)", "depthwise job if group == channels and 3x3, else needs grouped reduction")
            continue
        if op in ENGINE_NATIVE:
            counts["native"] += 1
            continue
        if op not in cache:
            try:
                cache[op] = is_pointwise_unary(op)
            except TinygradUnavailable:
                cache[op] = False
        if cache[op]:
            counts["table"] += 1
            detail[op] = "pointwise unary (found by executing it in tinygrad): 256-entry table job"
        else:
            counts["unsupported"] += 1
            detail[op] = "not a pointwise unary op: needs a kernel"
    return {"counts": dict(counts), "detail": detail}
