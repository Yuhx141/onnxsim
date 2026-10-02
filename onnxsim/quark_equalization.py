"""Quark-faithful cross-layer equalization (``CLEConfig``) and stem-BN
equalization for ONNX graphs.

Where :mod:`onnxsim.quark_cle` equalizes Gemm / MatMul chains generically, this
module reproduces what ``quark.onnx.algorithm.cle`` does -- patterns, options
and arithmetic -- so a ``CLEConfig`` run through :mod:`onnxsim.quark_compat`
gives the weights Quark gives (``tests/test_quark_algo_parity.py`` compares
them against the installed ``amd-quark``):

- *pairs*: ``Conv|Gemm -> [Relu | LeakyRelu | Pad | ReduceMean]* -> Conv|Gemm``
  where every intermediate tensor has one consumer. The head's output channels
  are multiplied by ``s = sqrt(r_tail / r_head)`` and the tail's matching input
  channels divided by it (``r``: per-channel max-abs weight; the head's range
  optionally includes its bias, see ``scale_append_bias``);
- *triples* ``Conv -> [Relu] -> depthwise Conv -> [Relu] -> pointwise Conv``:
  one cube-root balance across the three layers;
- options ``steps`` (``CLESteps``; ``-1`` = until the summed weight change falls
  below ``total_layer_diff_threshold``), ``weight_threshold``,
  ``scale_append_bias``, ``scale_use_threshold``; ``balance_method`` is
  accepted and, as in Quark, only ``"max"`` exists;
- ``replace_clip6`` (``ReplaceClip6Relu``): ``Clip(0, 6)`` -> ``Relu`` first;
- :func:`stem_equalize`: scale the weak output channels of the first Conv up
  (clamped to 16x) and fold ``1/s`` into every BatchNormalization behind it.

Only float32 constant weights are touched. Differences from Quark, all in
cases where Quark would emit a model that computes something else (or crash):
an intermediate tensor that is also a graph output ends a pattern, and a pair
whose channel counts disagree is skipped.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

_PASS_THROUGH = ("Relu", "ReduceMean", "Pad", "LeakyRelu")
_STEM_HOMOGENEOUS = {"Relu", "MaxPool", "AveragePool", "GlobalAveragePool", "Identity"}
_F32 = onnx.TensorProto.FLOAT

#: Quark's static ``op_types_to_quantize`` contain these two; CLE only looks at them
DEFAULT_OP_TYPES = ("Conv", "Gemm")


def _attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


class _Eq:
    """The graph plus the node-selection filters Quark's ``Optimizer`` has."""

    def __init__(
        self,
        model: onnx.ModelProto,
        op_types: Sequence[str],
        nodes_to_quantize: Sequence[str],
        nodes_to_exclude: Sequence[str],
    ) -> None:
        self.model = model
        self.op_types = set(op_types)
        self.only = set(nodes_to_quantize)
        self.exclude = set(nodes_to_exclude)

    def should(self, node: onnx.NodeProto) -> bool:
        if self.only and node.name not in self.only:
            return False
        if node.op_type not in self.op_types:
            return False
        return node.name not in self.exclude

    def inits(self) -> Dict[str, onnx.TensorProto]:
        return {t.name: t for t in self.model.graph.initializer}

    def consumers(self, node: onnx.NodeProto) -> List[onnx.NodeProto]:
        outs = {o for o in node.output if o}
        return [n for n in self.model.graph.node if outs & set(n.input)]


def _weights_of(
    node: onnx.NodeProto, inits: Dict[str, onnx.TensorProto]
) -> List[onnx.TensorProto]:
    """Initializer inputs of ``node`` in input order (Quark's
    ``get_weights_node_of_node``): ``[weight]`` or ``[weight, bias]``."""
    return [inits[i] for i in node.input if i in inits]


# -- pattern detection --------------------------------------------------------


def _dw_ok(node: onnx.NodeProto, inits: Dict[str, onnx.TensorProto]) -> bool:
    w = _weights_of(node, inits)
    g = _attr(node, "group")
    return bool(w) and g is not None and w[0].dims[1] == 1 and g == w[0].dims[0]


def _pair_supported(
    pair: Sequence[onnx.NodeProto], inits: Dict[str, onnx.TensorProto]
) -> bool:
    """Quark's ``check_conv_layers_support``. Its ``break`` leaves only the
    attribute loop, so a later layer's ``group`` attribute overwrites an
    earlier verdict; reproduced so the same pairs enter the (order sensitive)
    pattern list."""
    ok = True
    for n in pair:
        if n.op_type == "Conv":
            for a in n.attribute:
                if a.name == "group":
                    if a.i == 1:
                        ok = True
                    elif _dw_ok(n, inits):
                        ok = True
                    else:
                        ok = False
                        break
        elif n.op_type == "Gemm":
            ok = True
        else:
            return False
    return ok


def _find_patterns(eq: _Eq) -> List[Tuple[Any, ...]]:
    g = eq.model.graph
    inits = eq.inits()
    graph_out = {o.name for o in g.output}
    found: List[Tuple[Any, ...]] = []
    targets = ("Conv", "Gemm")

    def single(node: onnx.NodeProto) -> List[onnx.NodeProto]:
        nxt = eq.consumers(node)
        if any(o in graph_out for o in node.output):
            return []
        return nxt

    for node in g.node:
        if node.op_type not in targets or not eq.should(node):
            continue
        nxt = single(node)
        while nxt and len(nxt) == 1:
            c = nxt[0]
            if c.op_type in _PASS_THROUGH:
                nxt = single(c)
            elif c.op_type in targets:
                if eq.should(c) and _pair_supported([node, c], inits):
                    found.append(("pair", node, c))
                break
            else:
                break

    for node in g.node:
        if node.op_type != "Conv" or not eq.should(node):
            continue
        chain = [node]
        nxt = single(node)
        while nxt and len(nxt) == 1:
            c = nxt[0]
            if c.op_type == "Relu":
                nxt = single(c)
            elif c.op_type == "Conv":
                if not eq.should(c):
                    break
                chain.append(c)
                nxt = single(c)
                if len(chain) == 3:
                    c0, c1, c2 = chain
                    w1 = _weights_of(c1, inits)
                    g0, g1, g2 = (_attr(n, "group") for n in chain)
                    if (
                        g0 == 1
                        and g2 == 1
                        and g1 is not None
                        and g1 > 1
                        and w1
                        and w1[0].dims[0] == w1[0].dims[1] * g1
                    ):
                        found.append(("triple", c0, c1, c2))
                    break
            else:
                break

    # Quark's visiting order (the first pattern is listed twice, later ones are
    # inserted in front of the previous one unless it is a triple)
    order: List[Tuple[Any, ...]] = []
    for node in g.node:
        for p in found:
            if p[1] is node:
                if not order:
                    order.append(p)
                if len(order[-1]) > len(p):
                    order.append(p)
                else:
                    order.insert(-1, p)
    return order


# -- the two equalization kernels ----------------------------------------------


def _combine_bias(w: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    """Head weights ``[out, k]`` with the (shrunk) bias as an extra column."""
    if bias is None:
        return w.astype(np.float32)
    b = bias.copy().reshape(-1, 1)
    if np.count_nonzero(w) != w.size:
        wc = w.copy()
        for c in range(wc.shape[0]):
            nz = np.count_nonzero(wc[c])
            if nz == 0:
                b[c] = 0.0
                wc[c] = 1e-7
            elif nz != wc[c].size:
                m = np.fabs(np.ma.masked_where(wc[c] == 0.0, wc[c])).min()
                wc[c] = np.where(wc[c] == 0.0, -m, wc[c])
    else:
        wc = w.copy()
    wc = np.where(np.fabs(wc) < 1e-7, 1e-7, wc)
    factor = np.fabs(b) / np.fabs(wc)
    if np.fabs(bias).max() < 10 and np.fabs(bias).max() / np.fabs(wc).max() < 20:
        shrink = 5 if (np.median(factor) > 100 or factor.mean() > 1000) else 2
    elif np.median(factor) > 30 or factor.mean() > 500:
        shrink = 20
    elif np.median(factor) > 15 or factor.mean() > 100:
        shrink = 10
    else:
        shrink = 5
    return np.concatenate((w, b / shrink), axis=1).astype(np.float32)


def _scale(
    head: np.ndarray, tail: np.ndarray, threshold: float, use_threshold: bool
) -> np.ndarray:
    r0 = np.max(np.fabs(head), axis=1)
    r1 = np.max(np.fabs(tail), axis=1)
    root = np.sqrt(r0 * r1)
    scale = np.ones_like(r1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(root != 0, r1 / root, scale)
    if use_threshold:
        scale = np.where((r0 + r1) < threshold, 1, scale)
    return scale


def _pair_step(
    head: onnx.NodeProto,
    tail: onnx.NodeProto,
    arr: Dict[str, np.ndarray],
    inits: Dict[str, onnx.TensorProto],
    threshold: float,
    append_bias: bool,
    use_threshold: bool,
) -> None:
    def ok(n: onnx.NodeProto) -> bool:
        if n.op_type == "Gemm":
            return True
        g = _attr(n, "group")
        return g == 1 or (g is not None and _dw_ok(n, inits))

    # the head must be group 1 / depthwise with an explicit ``group``
    # attribute; Quark reads an absent attribute as "unsupported"
    if head.op_type == "Conv" and _attr(head, "group") is None:
        return
    if tail.op_type == "Conv" and _attr(tail, "group") is None:
        return
    if not (ok(head) and ok(tail)):
        return
    hw = [i for i in head.input if i in arr]
    tw = [i for i in tail.input if i in arr]
    if not hw or not tw or hw[0] != head.input[1] or tw[0] != tail.input[1]:
        return
    w_h = arr[hw[0]]
    b_h = arr[hw[1]] if len(hw) > 1 else None
    w_t = arr[tw[0]]
    if w_h.dtype != np.float32 or w_t.dtype != np.float32:
        return
    oc = w_h.shape[0]
    head_w = w_h.reshape(oc, -1)
    if head.op_type == "Gemm" and not _attr(head, "transB", 0):
        head_w = head_w.T
    if b_h is not None and b_h.size != head_w.shape[0]:
        return
    head_eff = _combine_bias(head_w, b_h) if append_bias else head_w

    tgroup = _attr(tail, "group", 1)
    if tail.op_type == "Conv":
        if tgroup == 1:
            if w_t.ndim not in (4, 5):
                return
            tail_w = np.swapaxes(w_t, 0, 1).reshape(w_t.shape[1], -1)
        else:
            tail_w = w_t.reshape(w_t.shape[0], -1)
    else:
        tail_w = w_t if not _attr(tail, "transB", 0) else w_t.T
    if head_eff.shape[0] != tail_w.shape[0]:
        return

    s = _scale(head_eff, tail_w, threshold, use_threshold)

    if head.op_type == "Conv":
        w_h = w_h * s.reshape((s.size,) + (1,) * (w_h.ndim - 1))
    elif not _attr(head, "transB", 0):
        w_h = w_h * s.reshape(1, -1)
    else:
        w_h = w_h * s.reshape(-1, 1)
    arr[hw[0]] = w_h.astype(w_h.dtype, copy=False)
    if b_h is not None:
        arr[hw[1]] = b_h * s
    if tail.op_type == "Conv":
        if tgroup == 1:
            w_t = w_t * (1 / s.reshape((1, -1) + (1,) * (w_t.ndim - 2)))
        else:
            w_t = w_t * (1 / s.reshape(-1, 1, 1, 1))
    elif not _attr(tail, "transB", 0):
        w_t = w_t * (1 / s.reshape(-1, 1))
    else:
        w_t = w_t * (1 / s.reshape(1, -1))
    arr[tw[0]] = w_t


def _triple_step(
    conv: onnx.NodeProto,
    dw: onnx.NodeProto,
    pw: onnx.NodeProto,
    arr: Dict[str, np.ndarray],
) -> None:
    def parts(n: onnx.NodeProto) -> Optional[Tuple[str, Optional[str]]]:
        w = [i for i in n.input if i in arr]
        if not w or w[0] != n.input[1] or arr[w[0]].dtype != np.float32:
            return None
        if arr[w[0]].ndim != 4:
            return None
        return w[0], (w[1] if len(w) > 1 else None)

    pc, pd, pp = parts(conv), parts(dw), parts(pw)
    if pc is None or pd is None or pp is None:
        return
    w0, w1, w2 = arr[pc[0]], arr[pd[0]], arr[pp[0]]
    if not (w0.shape[0] == w1.shape[0] == w2.shape[1]):
        return
    m0 = np.max(np.fabs(w0), axis=(1, 2, 3))
    m1 = np.max(np.fabs(w1), axis=(1, 2, 3))
    m2 = np.max(np.fabs(w2), axis=(0, 2, 3))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cube = np.power(m0 * m1 * m2, 1.0 / 3)
        s12 = m0 / cube
        s23 = cube / m2
    s12 = np.nan_to_num(s12, nan=1.0, posinf=1.0)
    s23 = np.nan_to_num(s23, nan=1.0, posinf=1.0)
    s12[s12 == 0.0] = 1.0
    s23[s23 == 0.0] = 1.0
    b0 = arr[pc[1]] if pc[1] else None
    b1 = arr[pd[1]] if pd[1] else None
    arr[pc[0]] = w0 * (1.0 / s12.reshape(-1, 1, 1, 1))
    arr[pd[0]] = w1 * s12.reshape(-1, 1, 1, 1) * (1.0 / s23.reshape(-1, 1, 1, 1))
    arr[pp[0]] = w2 * s23.reshape(1, -1, 1, 1)
    if pc[1] and b0 is not None:
        arr[pc[1]] = b0 * (1.0 / s12)
    if pd[1] and b1 is not None:
        arr[pd[1]] = b1 * (1.0 / s23)


# -- driver ----------------------------------------------------------------------


def replace_clip6_with_relu(model: onnx.ModelProto) -> onnx.ModelProto:
    """``Clip(0, 6)`` with constant bounds -> ``Relu`` (in place)."""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    drop: List[str] = []
    for i, n in enumerate(list(g.node)):
        if n.op_type != "Clip" or len(n.input) < 3:
            continue
        lo, hi = inits.get(n.input[1]), inits.get(n.input[2])
        if lo is None or hi is None:
            continue
        if np.allclose(numpy_helper.to_array(lo), 0.0) and np.allclose(
            numpy_helper.to_array(hi), 6.0
        ):
            relu = onnx.helper.make_node(
                "Relu", [n.input[0]], list(n.output), name=n.name
            )
            g.node[i].CopyFrom(relu)
            drop += [lo.name, hi.name]
    used = {x for n in g.node for x in n.input}
    for name in dict.fromkeys(drop):
        if name not in used:
            for t in list(g.initializer):
                if t.name == name:
                    g.initializer.remove(t)
    return model


def equalize(
    model: onnx.ModelProto,
    steps: int = 1,
    balance_method: str = "max",
    weight_threshold: float = 0.5,
    scale_append_bias: bool = True,
    scale_use_threshold: bool = True,
    total_layer_diff_threshold: float = 1.9e-7,
    op_types: Iterable[str] = DEFAULT_OP_TYPES,
    nodes_to_quantize: Sequence[str] = (),
    nodes_to_exclude: Sequence[str] = (),
    replace_clip6: bool = False,
) -> onnx.ModelProto:
    """Quark's ``cle_transforms`` (see the module docstring). Returns a
    modified copy of ``model``."""
    if balance_method != "max":
        raise ValueError(f"unknown CLE balance method {balance_method!r}")
    m = onnx.ModelProto()
    m.CopyFrom(model)
    if replace_clip6:
        replace_clip6_with_relu(m)
    eq = _Eq(m, tuple(op_types), nodes_to_quantize, nodes_to_exclude)
    patterns = _find_patterns(eq)
    inits = eq.inits()
    arr: Dict[str, np.ndarray] = {
        k: numpy_helper.to_array(v) for k, v in inits.items() if v.data_type == _F32
    }
    nodes = [n for n in m.graph.node if n.op_type in ("Conv", "Gemm")]

    diff, count, done = 10.0, 0, 0
    while diff > total_layer_diff_threshold and count < 20:
        if steps >= 0 and done >= steps:
            break
        prev = dict(arr)
        for p in patterns:
            if p[0] == "pair":
                _pair_step(
                    p[1],
                    p[2],
                    arr,
                    inits,
                    weight_threshold,
                    scale_append_bias,
                    scale_use_threshold,
                )
            else:
                _triple_step(p[1], p[2], p[3], arr)
        tmp = 0.0
        for n in nodes:
            names = [i for i in n.input if i in inits]
            if names and names[0] in arr and names[0] in prev:
                d = prev[names[0]] - arr[names[0]]
                tmp += float(np.mean(np.abs(np.float64(d))))
        if abs(diff - tmp) > 1e-9:
            count, diff = 0, tmp
        else:
            count += 1
        done += 1

    for name, a in arr.items():
        if a is not None and name in inits:
            old = numpy_helper.to_array(inits[name])
            if old.shape != a.shape or not np.array_equal(old, a):
                inits[name].CopyFrom(numpy_helper.from_array(a, name))
    return m


# -- stem equalization -------------------------------------------------------------


def _channels(graph: onnx.GraphProto, name: str) -> int:
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == name:
            dims = vi.type.tensor_type.shape.dim
            if len(dims) >= 2 and dims[1].dim_value > 0:
                return int(dims[1].dim_value)
    return -1


def _pad_is_zero(node: onnx.NodeProto, inits: Dict[str, onnx.TensorProto]) -> bool:
    mode = _attr(node, "mode", b"constant")
    mode = mode.decode() if isinstance(mode, bytes) else str(mode)
    if mode != "constant":
        return False
    value = _attr(node, "value")
    if value is not None and value != 0.0:
        return False
    if len(node.input) >= 3 and node.input[2]:
        c = inits.get(node.input[2])
        if c is None or np.any(numpy_helper.to_array(c) != 0):
            return False
    return True


def _concat_offset(graph: onnx.GraphProto, node: onnx.NodeProto, tensor: str) -> int:
    if _attr(node, "axis", 1) != 1:
        return -1
    offset = 0
    for x in node.input:
        if x == tensor:
            return offset
        ch = _channels(graph, x)
        if ch < 0:
            return -1
        offset += ch
    return -1


def _stem_groups(
    eq: _Eq,
) -> Tuple[Optional[onnx.NodeProto], int, List[Tuple[onnx.NodeProto, int]]]:
    g = eq.model.graph
    inits = eq.inits()
    graph_in = {i.name for i in g.input}
    stem = None
    for n in g.node:
        if n.op_type == "Conv" and eq.should(n) and any(x in graph_in for x in n.input):
            if _attr(n, "group", 1) == 1:
                stem = n
                break
    if stem is None:
        return None, 0, []
    wname = next((x for x in stem.input if x in inits), None)
    if wname is None:
        return None, 0, []
    cout = int(inits[wname].dims[0])
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in g.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    groups: List[Tuple[onnx.NodeProto, int]] = []
    seen = set()
    queue = deque((o, 0) for o in stem.output)
    while queue:
        tensor, off = queue.popleft()
        for n in consumers.get(tensor, []):
            key = (id(n), tensor, off)
            if key in seen:
                continue
            seen.add(key)
            if n.op_type == "BatchNormalization":
                gamma = inits.get(n.input[1])
                if gamma is None or gamma.dims[0] < off + cout:
                    return None, 0, []
                groups.append((n, off))
                continue
            delta = 0
            if n.op_type in _STEM_HOMOGENEOUS:
                pass
            elif n.op_type == "Pad":
                if not _pad_is_zero(n, inits):
                    return None, 0, []
            elif n.op_type == "Concat":
                delta = _concat_offset(g, n, tensor)
                if delta < 0:
                    return None, 0, []
            else:
                return None, 0, []
            for o in n.output:
                queue.append((o, off + delta))
    if not groups:
        return None, 0, []
    return stem, cout, groups


def stem_equalize(
    model: onnx.ModelProto,
    op_types: Iterable[str] = DEFAULT_OP_TYPES + ("MatMul",),
    nodes_to_quantize: Sequence[str] = (),
    nodes_to_exclude: Sequence[str] = (),
) -> onnx.ModelProto:
    """Quark's ``stem_equalize_transforms``: weak channels of the first Conv
    are scaled up by ``clip(max_absmax / absmax_c, 1, 16)`` and the inverse is
    folded into the BatchNormalizations downstream (``gamma /= s``,
    ``mean *= s``), which keeps the float function exact. No-op unless the stem
    Conv's output reaches BatchNormalizations through Relu / pooling /
    Identity / zero Pad / Concat only. Returns a modified copy."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    eq = _Eq(m, tuple(op_types), nodes_to_quantize, nodes_to_exclude)
    stem, cout, groups = _stem_groups(eq)
    if stem is None:
        return m
    inits = eq.inits()
    given = [x for x in stem.input if x in inits]
    w_name = given[0]
    b_name = given[1] if len(given) > 1 else None
    w = numpy_helper.to_array(inits[w_name]).astype(np.float32)
    absmax = np.abs(w).reshape(cout, -1).max(axis=1)
    live = absmax > 1e-6
    if live.sum() == 0:
        return m
    target = absmax[live].max()
    s = np.ones(cout, dtype=np.float32)
    s[live] = np.clip(target / absmax[live], 1.0, 16.0)

    def put(name: str, a: np.ndarray) -> None:
        inits[name].CopyFrom(numpy_helper.from_array(a.astype(np.float32), name))

    put(w_name, w * s[:, None, None, None])
    if b_name is not None:
        put(b_name, numpy_helper.to_array(inits[b_name]).astype(np.float32) * s)
    for bn, off in groups:
        gamma = numpy_helper.to_array(inits[bn.input[1]]).astype(np.float32).copy()
        mean = numpy_helper.to_array(inits[bn.input[3]]).astype(np.float32).copy()
        sl = slice(off, off + cout)
        gamma[sl] = gamma[sl] / s
        mean[sl] = mean[sl] * s
        put(bn.input[1], gamma)
        put(bn.input[3], mean)
    return m


def apply_cle_config(
    model: onnx.ModelProto,
    params: Dict[str, Any],
    extra_options: Dict[str, Any],
    exclude: Sequence[str] = (),
    op_types: Iterable[str] = DEFAULT_OP_TYPES + ("MatMul",),
) -> onnx.ModelProto:
    """Quark's pre-quantization CLE stage: stem equalization, then
    ``cle_transforms``. ``params`` are a ``CLEConfig``'s keyword arguments;
    ``extra_options`` (``CLESteps`` ...) take precedence over them, as in
    Quark's ``CLEConfig.get_options``."""

    def opt(key: str, name: str, default: Any) -> Any:
        return extra_options.get(key, params.get(name, default))

    try:  # the stem search reads static channel counts from value_info
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:  # pragma: no cover - best effort
        pass
    op_types = tuple(op_types)
    out = stem_equalize(model, op_types, (), exclude)
    return equalize(
        out,
        steps=opt("CLESteps", "cle_steps", 1),
        balance_method=opt("CLEBalanceMethod", "cle_balance_method", "max"),
        weight_threshold=opt("CLEWeightThreshold", "cle_weight_threshold", 0.5),
        scale_append_bias=opt("CLEScaleAppendBias", "cle_scale_append_bias", True),
        scale_use_threshold=opt(
            "CLEScaleUseThreshold", "cle_scale_use_threshold", True
        ),
        total_layer_diff_threshold=opt(
            "CLETotalLayerDiffThreshold", "cle_total_layer_diff_threshold", 1.9e-7
        ),
        op_types=op_types,
        nodes_to_exclude=exclude,
        replace_clip6=bool(extra_options.get("ReplaceClip6Relu", False)),
    )


__all__ = [
    "apply_cle_config",
    "equalize",
    "replace_clip6_with_relu",
    "stem_equalize",
]
