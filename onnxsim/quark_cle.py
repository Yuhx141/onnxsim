"""Cross-layer equalization for fully-connected chains (``Gemm`` / constant-weight
``MatMul``), the part of Quark's ``CLEConfig`` that onnxsim's Conv-only
:func:`onnxsim.onnx_simplifier.cross_layer_equalize` does not cover.

For ``L1 -> [Relu | LeakyRelu | PRelu] -> L2`` where the intermediate tensor
has no other consumer, every shared channel ``c`` is rescaled by
``S[c] = sqrt(r1[c] / r2[c])`` (``r1`` / ``r2``: the channel's weight range in
``L1`` -- bias included -- / ``L2``): ``L1``'s output channel ``c`` (weight column and bias) is
divided by ``S[c]`` and ``L2``'s input channel ``c`` (weight row) multiplied by
it. The activations are positive-homogeneous, so the composed function is
unchanged, and both layers end up with the same per-channel range. One sweep
over the pairs is Quark's default; a layer shared by two pairs of a chain is
only fully balanced after repeated sweeps (``steps=-1``).

Independent implementation; the scale rule was checked against Quark's output
(``tests/test_quark_parity.py::test_cle_matches_quarks_equalization``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

_HOMOGENEOUS = {"Relu", "LeakyRelu", "PRelu"}


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for a in node.attribute:
        if a.name == name:
            return int(a.i)
    return default


def _weight_axes(node: onnx.NodeProto) -> Optional[Tuple[int, int]]:
    """``(in_axis, out_axis)`` of a layer's constant weight, or None."""
    if node.op_type == "MatMul":
        return 0, 1
    if node.op_type == "Gemm":
        if _attr_int(node, "transA", 0):
            return None
        return (1, 0) if _attr_int(node, "transB", 0) else (0, 1)
    return None


def _head_ranges(w: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    """Per-output-channel range of the head layer, ``w`` being ``[out, in]``.

    The bias takes part, shrunk by a heuristic factor (2 / 5 / 10 / 20) that
    depends on how large it is next to the weights, the same rule Quark uses
    so the resulting scales agree."""
    if bias is None:
        return np.abs(w).max(axis=1)
    bias_col = bias.astype(np.float64).reshape(-1, 1).copy()
    wc = w.astype(np.float64).copy()
    if np.count_nonzero(wc) != wc.size:
        for c in range(wc.shape[0]):
            nz = wc[c] != 0
            if not nz.any():
                bias_col[c] = 0.0
                wc[c] = 1e-7
            elif not nz.all():
                wc[c] = np.where(nz, wc[c], -np.abs(wc[c][nz]).min())
    wc = np.where(np.abs(wc) < 1e-7, 1e-7, wc)
    factor = np.abs(bias_col) / np.abs(wc)
    bmax = np.abs(bias).max()
    if bmax < 10 and bmax / np.abs(wc).max() < 20:
        shrink = 5 if (np.median(factor) > 100 or factor.mean() > 1000) else 2
    elif np.median(factor) > 30 or factor.mean() > 500:
        shrink = 20
    elif np.median(factor) > 15 or factor.mean() > 100:
        shrink = 10
    else:
        shrink = 5
    return np.abs(np.concatenate([w, bias_col / shrink], axis=1)).max(axis=1)


def equalize_linear_layers(
    model: onnx.ModelProto,
    steps: int = 1,
    tolerance: float = 1e-7,
    weight_threshold: float = 0.5,
) -> onnx.ModelProto:
    """Return ``model`` with its Gemm/MatMul chains equalized (see the module
    docstring). Constant float32 weights only; other patterns are left alone.

    :param steps: sweeps over all pairs; 1 is Quark's default (``CLESteps``),
            a negative value repeats until the scales settle (at most 20)
    :param weight_threshold: channels whose two ranges sum to less than this
            are left alone (Quark's ``CLEWeightThreshold``, default 0.5)"""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in g.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    graph_outputs = {o.name for o in g.output}

    def weight_of(n: onnx.NodeProto) -> Optional[str]:
        if (
            _weight_axes(n) is None
            or len(n.input) < 2
            or n.input[1] not in inits
            or inits[n.input[1]].data_type != onnx.TensorProto.FLOAT
        ):
            return None
        if len(inits[n.input[1]].dims) != 2:
            return None
        return n.input[1]

    def bias_of(n: onnx.NodeProto) -> Optional[str]:
        if n.op_type == "Gemm" and len(n.input) > 2 and n.input[2]:
            return n.input[2]
        return None

    # (L1, L2) pairs, only when nothing else reads the intermediate tensor and
    # a bias (when present) is a constant vector we can rescale.
    pairs: List[Tuple[onnx.NodeProto, onnx.NodeProto]] = []
    for n1 in g.node:
        if weight_of(n1) is None or not n1.output:
            continue
        b = bias_of(n1)
        if b is not None and (
            b not in inits
            or inits[b].data_type != onnx.TensorProto.FLOAT
            or len(inits[b].dims) != 1
        ):
            continue
        cur = n1
        while True:
            nxt = consumers.get(cur.output[0], [])
            if len(nxt) != 1 or cur.output[0] in graph_outputs:
                break
            c = nxt[0]
            if c.op_type in _HOMOGENEOUS and c.input[0] == cur.output[0]:
                if c.op_type == "PRelu":  # slope must be channel-independent
                    s = inits.get(c.input[1])
                    if s is None or s.data_type != onnx.TensorProto.FLOAT:
                        break
                    if numpy_helper.to_array(s).size != 1:
                        break
                cur = c
                continue
            if weight_of(c) is not None and c.input[0] == cur.output[0]:
                pairs.append((n1, c))
            break
    if not pairs:
        return m

    def get(name: str) -> np.ndarray:
        return numpy_helper.to_array(inits[name]).astype(np.float64)

    def put(name: str, arr: np.ndarray) -> None:
        inits[name].CopyFrom(numpy_helper.from_array(arr.astype(np.float32), name))

    # Layers shared with another consumer would be rescaled twice over
    # incompatible bases; keep only pairs whose weight tensors are used once.
    uses: Dict[str, int] = {}
    for n in g.node:
        for x in n.input:
            uses[x] = uses.get(x, 0) + 1
    pairs = [
        (a, b)
        for a, b in pairs
        if uses[weight_of(a)] == 1  # type: ignore[index]
        and uses[weight_of(b)] == 1  # type: ignore[index]
    ]

    # Quark visits the pairs in a quirky order (the first pair twice, later
    # ones inserted ahead of the previous one); the fixed point of a chain
    # depends on it, so reproduce it.
    order: List[Tuple[onnx.NodeProto, onnx.NodeProto]] = []
    for pair in pairs:
        if not order:
            order.append(pair)
        order.insert(-1, pair)

    limit = steps if steps >= 0 else 20
    for _ in range(limit):
        worst = 0.0
        for n1, n2 in order:
            w1n, w2n = weight_of(n1), weight_of(n2)
            assert w1n is not None and w2n is not None
            w1, w2 = get(w1n), get(w2n)
            in1, out1 = _weight_axes(n1)  # type: ignore[misc]
            in2, _ = _weight_axes(n2)  # type: ignore[misc]
            if w1.shape[out1] != w2.shape[in2]:
                continue
            b = bias_of(n1)
            if b is not None and get(b).size != w1.shape[out1]:
                continue
            r1 = _head_ranges(np.moveaxis(w1, out1, 0), None if b is None else get(b))
            r2 = np.abs(w2).max(axis=1 - in2)
            ok = (r1 > 0) & (r2 > 0) & (r1 + r2 >= weight_threshold)
            s = np.ones_like(r1)
            s[ok] = np.sqrt(r1[ok] / r2[ok])
            worst = max(worst, float(np.abs(s - 1).max()))
            shape1 = [1, 1]
            shape1[out1] = -1
            shape2 = [1, 1]
            shape2[in2] = -1
            put(w1n, w1 / s.reshape(shape1))
            put(w2n, w2 * s.reshape(shape2))
            if b is not None:
                put(b, get(b) / s)
        if steps < 0 and worst < tolerance:
            break
    return m


__all__ = ["equalize_linear_layers"]
