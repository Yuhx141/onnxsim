"""More post-quantization graph tools modelled on ``quark.onnx.tools`` and the
graph utilities in ``quark.onnx.utils.model_utils``. Independent
implementations (Quark's source was read for names and intent only);
``tests/test_quark_parity.py`` re-checks them against the installed
``amd-quark``. They are re-exported from :mod:`onnxsim.quark_tools`, whose
module docstring carries the correspondence table.

Every function takes an ``onnx.ModelProto`` and returns a new one (the input
is never modified) unless noted, and rewrites the **top-level graph** only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_QDQ_DOMAINS = ("", "ai.onnx", "com.microsoft")
MS_DOMAIN = "com.microsoft"
QUARK_DOMAIN = "com.amd.quark"
VITIS_DOMAIN = "com.vai.quantize"

_Q_NAMES = ("QuantizeLinear", "ExtendedQuantizeLinear")
_DQ_NAMES = ("DequantizeLinear", "ExtendedDequantizeLinear")

# Quark's extended custom ops <-> the deprecated Vitis ones.
CUSTOM_OP_NAME_MAPPING = {
    "ExtendedQuantizeLinear": "VitisQuantizeLinear",
    "ExtendedDequantizeLinear": "VitisDequantizeLinear",
    "ExtendedInstanceNormalization": "VitisInstanceNormalization",
    "ExtendedLSTM": "VitisLSTM",
    "BFPQuantizeDequantize": "BFPFixNeuron",
    "MXQuantizeDequantize": "MXFixNeuron",
}

_INT_RANGE = {
    TensorProto.UINT8: (0, 255),
    TensorProto.INT8: (-128, 127),
    TensorProto.UINT16: (0, 65535),
    TensorProto.INT16: (-32768, 32767),
    TensorProto.UINT32: (0, 2**32 - 1),
    TensorProto.INT32: (-(2**31), 2**31 - 1),
}
_NP_INT = {
    TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8,
    TensorProto.UINT16: np.uint16,
    TensorProto.INT16: np.int16,
    TensorProto.UINT32: np.uint32,
    TensorProto.INT32: np.int32,
}


# -- helpers --------------------------------------------------------------------


def _copy(model: onnx.ModelProto) -> onnx.ModelProto:
    out = onnx.ModelProto()
    out.CopyFrom(model)
    return out


def _is_std(node: onnx.NodeProto, op: str) -> bool:
    return node.op_type == op and node.domain in ("", "ai.onnx")


def _producers(g: onnx.GraphProto) -> Dict[str, onnx.NodeProto]:
    return {o: n for n in g.node for o in n.output if o}


def _consumers(g: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    """Tensor name -> distinct consuming nodes (graph order)."""
    out: Dict[str, List[onnx.NodeProto]] = defaultdict(list)
    for n in g.node:
        for x in dict.fromkeys(n.input):
            if x:
                out[x].append(n)
    return out


def _subgraph_names(attrs: Iterable[onnx.AttributeProto]) -> Set[str]:
    used: Set[str] = set()
    for a in attrs:
        graphs = []
        if a.type == onnx.AttributeProto.GRAPH:
            graphs = [a.g]
        elif a.type == onnx.AttributeProto.GRAPHS:
            graphs = list(a.graphs)
        for sg in graphs:
            for n in sg.node:
                used.update(x for x in n.input if x)
                used |= _subgraph_names(n.attribute)
            used.update(o.name for o in sg.output)
    return used


def _prune_initializers(g: onnx.GraphProto) -> None:
    """Drop initializers nothing refers to (inputs mirroring them stay)."""
    used = {x for n in g.node for x in n.input if x}
    for n in g.node:
        used |= _subgraph_names(n.attribute)
    used |= {o.name for o in g.output}
    keep = [t for t in g.initializer if t.name in used]
    if len(keep) != len(g.initializer):
        del g.initializer[:]
        g.initializer.extend(keep)


def _replace_input(
    g: onnx.GraphProto, old: str, new: str, skip: Sequence[onnx.NodeProto] = ()
) -> None:
    for n in g.node:
        if any(n is s for s in skip):
            continue
        for i, x in enumerate(n.input):
            if x == old:
                n.input[i] = new


def _unique_name(taken: Set[str], base: str) -> str:
    name, k = base, 1
    while name in taken:
        name = f"{base}_{k}"
        k += 1
    taken.add(name)
    return name


def _all_names(g: onnx.GraphProto) -> Set[str]:
    names = {t.name for t in g.initializer}
    names |= {v.name for v in list(g.input) + list(g.output) + list(g.value_info)}
    for n in g.node:
        names.update(n.input)
        names.update(n.output)
        names.add(n.name)
    return names


def _set_opset(model: onnx.ModelProto, domain: str, version: int) -> None:
    for o in model.opset_import:
        if o.domain == domain:
            o.version = version
            return
    model.opset_import.add(domain=domain, version=version)


def _has_opset(model: onnx.ModelProto, domain: str) -> bool:
    return any(o.domain == domain for o in model.opset_import)


def _f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns (uint16), round-to-nearest-even."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    return np.where(nan, np.uint16(0x7FC0), rounded).astype(np.uint16)


def _bf16_tensor(arr: np.ndarray, name: str) -> onnx.TensorProto:
    t = onnx.TensorProto()
    t.name = name
    t.data_type = TensorProto.BFLOAT16
    t.dims.extend(arr.shape)
    t.raw_data = _f32_to_bf16_bits(arr).tobytes()
    return t


def _round_to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 values snapped to the nearest bfloat16, kept as float32."""
    bits = _f32_to_bf16_bits(x).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(np.shape(x))


# -- NCHW <-> NHWC --------------------------------------------------------------


def convert_nchw_to_nhwc(model: onnx.ModelProto) -> onnx.ModelProto:
    """Make a 4-D NCHW model take / return NHWC.

    Every 4-D graph input with static ``C, H, W`` becomes ``[N, H, W, C]`` and
    feeds the graph through a ``Transpose(perm=[0,3,1,2])``; every such output
    is followed by a ``Transpose(perm=[0,2,3,1])`` and the graph output is
    **renamed** to that transpose's output (``<out>_transpose``). When the
    output was produced by a ``Q -> DQ`` pair the transpose is re-quantized
    (``DQ -> Transpose -> Q -> DQ``, reusing the pair's scale / zero point)
    and the output is the new DQ's (``<out>_transpose_DequantizeLinear``).
    Inputs / outputs that are not 4-D or not static are left alone. Initializers
    mirrored in the graph inputs are dropped first.
    """
    from onnxsim.quark_tools import remove_initializer_from_input

    m = remove_initializer_from_input(model)
    g = m.graph
    taken = _all_names(g)

    def static_chw(vi: onnx.ValueInfoProto) -> Optional[Tuple[int, int, int]]:
        if not vi.type.HasField("tensor_type"):
            return None
        dims = vi.type.tensor_type.shape.dim
        if len(dims) != 4:
            return None
        vals = []
        for d in list(dims)[1:]:
            if d.WhichOneof("value") != "dim_value":
                return None
            vals.append(int(d.dim_value))
        return vals[0], vals[1], vals[2]

    def set_nhwc(vi: onnx.ValueInfoProto, chw: Tuple[int, int, int]) -> None:
        c, h, w = chw
        dims = vi.type.tensor_type.shape.dim
        dims[1].dim_value, dims[2].dim_value, dims[3].dim_value = h, w, c

    new_nodes: List[onnx.NodeProto] = []
    for inp in g.input:
        chw = static_chw(inp)
        if chw is None:
            continue
        set_nhwc(inp, chw)
        name = _unique_name(taken, inp.name + "_transpose")
        _replace_input(g, inp.name, name)
        new_nodes.append(
            helper.make_node(
                "Transpose", [inp.name], [name], name=name, perm=[0, 3, 1, 2]
            )
        )
    head = list(new_nodes)

    tail: List[onnx.NodeProto] = []
    prod = _producers(g)
    for out in g.output:
        chw = static_chw(out)
        if chw is None:
            continue
        set_nhwc(out, chw)
        name = _unique_name(taken, out.name + "_transpose")
        tr = helper.make_node(
            "Transpose", [out.name], [name], name=name, perm=[0, 2, 3, 1]
        )
        tail.append(tr)
        last = prod.get(out.name)
        before = prod.get(last.input[0]) if last is not None and last.input else None
        if (
            last is not None
            and before is not None
            and _is_std(last, "DequantizeLinear")
            and _is_std(before, "QuantizeLinear")
        ):
            qn = _unique_name(taken, name + "_QuantizeLinear")
            dn = _unique_name(taken, name + "_DequantizeLinear")
            q = helper.make_node(
                "QuantizeLinear",
                [name, *before.input[1:3]],
                [qn],
                name=qn,
                domain=before.domain,
            )
            dq = helper.make_node(
                "DequantizeLinear",
                [qn, *last.input[1:3]],
                [dn],
                name=dn,
                domain=last.domain,
            )
            tail.extend([q, dq])
            out.name = dn
        else:
            out.name = name

    old = list(g.node)
    del g.node[:]
    g.node.extend(head + old + tail)
    # Shape info of the interior is now stale at the boundary; drop it.
    del g.value_info[:]
    return m


# -- Q/DQ rewrites --------------------------------------------------------------


def convert_qdq_to_qop(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fuse ``DQ -> {MatMul, Add, Mul, Sigmoid} -> Q`` into the QLinear
    operator (``QLinearMatMul`` in the default domain; ``QLinearAdd`` /
    ``QLinearMul`` / ``QLinearSigmoid`` in ``com.microsoft``), as Quark's
    ``ConvertQDQToQOPTransformsPipeline`` does (``Conv`` is not converted
    there either). The fused node keeps the original node's name and takes
    the Q node's output. DQ nodes whose output is no longer used are dropped.

    Conditions: every data input comes from a ``DequantizeLinear`` with a zero
    point, the op's output is used only by one ``QuantizeLinear`` (with a zero
    point), and scales are per-tensor (a 1-D per-column scale on the B input
    of ``MatMul`` is accepted too). ``Sigmoid`` additionally needs its DQ to
    feed nothing else, as in Quark.
    """
    m = _copy(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    prod = _producers(g)
    cons = _consumers(g)
    graph_out = {o.name for o in g.output}
    arity = {"MatMul": 2, "Add": 2, "Mul": 2, "Sigmoid": 1}
    domain = {"MatMul": "", "Add": MS_DOMAIN, "Mul": MS_DOMAIN, "Sigmoid": MS_DOMAIN}
    fused: Dict[int, onnx.NodeProto] = {}  # id(Q node) -> replacement
    dead_ops: Set[int] = set()

    def scale_ok(dq: onnx.NodeProto, is_b: bool) -> bool:
        s = inits.get(dq.input[1])
        if s is None:
            return False
        sd = list(s.dims)
        if int(np.prod(sd or [1])) == 1:
            return True
        if is_b and len(sd) == 1:
            x = inits.get(dq.input[0])
            axis = next((a.i for a in dq.attribute if a.name == "axis"), 1)
            return x is not None and len(x.dims) == 2 and axis % 2 == 1
        return False

    for q in g.node:
        if not (_is_std(q, "QuantizeLinear") and len(q.input) >= 3):
            continue
        op = prod.get(q.input[0])
        if (
            op is None
            or op.op_type not in arity
            or not _is_std(op, op.op_type)
            or len(op.input) != arity[op.op_type]
            or op.output[0] in graph_out
            or [c for c in cons[op.output[0]]] != [q]
        ):
            continue
        if not scale_ok(q, False) or q.input[1] not in inits:
            continue
        dqs = [prod.get(x) for x in op.input]
        if any(d is None or not _is_std(d, "DequantizeLinear") for d in dqs):
            continue
        if any(len(d.input) < 3 for d in dqs):  # type: ignore[union-attr]
            continue
        if not all(
            scale_ok(d, op.op_type == "MatMul" and i == 1)  # type: ignore[arg-type]
            for i, d in enumerate(dqs)
        ):
            continue
        if op.op_type == "Sigmoid" and len(cons[dqs[0].output[0]]) > 1:  # type: ignore[union-attr]
            continue
        ins: List[str] = []
        for d in dqs:
            ins += [d.input[0], d.input[1], d.input[2]]  # type: ignore[union-attr]
        ins += [q.input[1], q.input[2]]
        fused[id(q)] = helper.make_node(
            "QLinear" + op.op_type,
            ins,
            [q.output[0]],
            name=op.name,
            domain=domain[op.op_type],
        )
        dead_ops.add(id(op))

    if not fused:
        return m
    kept: List[onnx.NodeProto] = []
    for n in g.node:
        if id(n) in dead_ops:
            continue
        kept.append(fused.get(id(n), n))
    del g.node[:]
    g.node.extend(kept)
    # DQ nodes (and constants) that lost their last consumer.
    changed = True
    while changed:
        changed = False
        used = {x for n in g.node for x in n.input} | {o.name for o in g.output}
        for n in list(g.node):
            if _is_std(n, "DequantizeLinear") and n.output[0] not in used:
                g.node.remove(n)
                changed = True
                break
    _prune_initializers(g)
    if any(n.domain == MS_DOMAIN for n in g.node) and not _has_opset(m, MS_DOMAIN):
        _set_opset(m, MS_DOMAIN, 1)
    return m


def convert_customqdq_to_qdq(model: onnx.ModelProto) -> onnx.ModelProto:
    """``ExtendedQuantizeLinear`` / ``ExtendedDequantizeLinear`` (Quark's
    custom ops) -> ``com.microsoft`` ``QuantizeLinear`` / ``DequantizeLinear``,
    which support 16-bit. Only nodes whose zero point is an initializer of
    int8 / uint8 / int16 / uint16 / int32 are converted (bfloat16 and
    friends stay custom). Unlike Quark, a ``com.microsoft`` opset import is
    added when needed so the result loads."""
    m = _copy(model)
    inits = {t.name: t for t in m.graph.initializer}
    ok = (
        TensorProto.INT8,
        TensorProto.UINT8,
        TensorProto.INT16,
        TensorProto.UINT16,
        TensorProto.INT32,
    )
    mapping = {
        "ExtendedQuantizeLinear": "QuantizeLinear",
        "ExtendedDequantizeLinear": "DequantizeLinear",
    }
    changed = False
    for n in m.graph.node:
        if n.op_type not in mapping or len(n.input) < 3:
            continue
        zp = inits.get(n.input[2])
        if zp is not None and zp.data_type in ok:
            n.op_type = mapping[n.op_type]
            n.domain = MS_DOMAIN
            changed = True
    if changed and not _has_opset(m, MS_DOMAIN):
        _set_opset(m, MS_DOMAIN, 1)
    return m


def convert_custom_ops(
    model: onnx.ModelProto,
    domain: str = VITIS_DOMAIN,
    mapping: Optional[Dict[str, str]] = None,
) -> onnx.ModelProto:
    """Rename custom ops and move them to ``domain``. The defaults convert
    Quark's ``Extended*`` ops to the deprecated Vitis ones
    (:data:`CUSTOM_OP_NAME_MAPPING`, domain ``com.vai.quantize``); for the
    reverse pass ``domain="com.amd.quark"`` and the inverted mapping."""
    mapping = CUSTOM_OP_NAME_MAPPING if mapping is None else mapping
    m = _copy(model)
    n_changed = 0
    for n in m.graph.node:
        if n.op_type in mapping:
            n.domain = domain
            n.op_type = mapping[n.op_type]
            n_changed += 1
    if n_changed:
        _set_opset(m, domain, 1)
    return m


def remove_qdq_between_ops(
    model: onnx.ModelProto, between_ops: Sequence[Tuple[str, str]]
) -> onnx.ModelProto:
    """For each ``(upper_op, lower_op)`` pair, delete the ``Q -> DQ`` that sits
    between an ``upper_op`` node and a ``lower_op`` node (``upper -> Q -> DQ ->
    lower`` becomes ``upper -> lower``). The DQ must feed only that lower
    node. Unlike Quark, the Q must also feed only that DQ (and neither may
    produce a graph output), otherwise the pattern is skipped instead of
    leaving a dangling consumer."""
    m = _copy(model)
    g = m.graph
    graph_out = {o.name for o in g.output}
    for upper_op, lower_op in between_ops:
        prod = _producers(g)
        cons = _consumers(g)
        drop: List[onnx.NodeProto] = []
        rewires: List[Tuple[onnx.NodeProto, int, str]] = []
        for lower in g.node:
            if lower.op_type != lower_op:
                continue
            for i, x in enumerate(lower.input):
                dq = prod.get(x)
                if dq is None or dq.op_type != "DequantizeLinear":
                    continue
                if len(cons[dq.output[0]]) > 1 or dq.output[0] in graph_out:
                    continue
                q = prod.get(dq.input[0])
                if q is None or q.op_type != "QuantizeLinear":
                    continue
                if len(cons[q.output[0]]) > 1 or q.output[0] in graph_out:
                    continue
                up = prod.get(q.input[0])
                if up is None or up.op_type != upper_op:
                    continue
                if any(q is d or dq is d for d in drop):
                    continue
                drop += [q, dq]
                rewires.append((lower, i, q.input[0]))
        if not drop:
            continue
        for lower, i, name in rewires:
            lower.input[i] = name
        keep = [n for n in g.node if not any(n is d for d in drop)]
        del g.node[:]
        g.node.extend(keep)
    _prune_initializers(g)
    return m


def remove_qdq_mul_add(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove the ``Q -> DQ`` in ``Mul -> Q -> DQ -> Add`` (see
    :func:`remove_qdq_between_ops`)."""
    return remove_qdq_between_ops(model, [("Mul", "Add")])


# -- integer-width converters ---------------------------------------------------


def _qdq_scale_zp_inits(
    node: onnx.NodeProto, inits: Dict[str, onnx.TensorProto]
) -> Tuple[Optional[onnx.TensorProto], Optional[onnx.TensorProto]]:
    s = inits.get(node.input[1]) if len(node.input) > 1 else None
    z = inits.get(node.input[2]) if len(node.input) > 2 else None
    return s, z


def convert_u16u8_to_u8u8(model: onnx.ModelProto) -> onnx.ModelProto:
    """Re-express uint16 activation Q/DQ as uint8.

    For each ``Q -> DQ`` pair on an *activation* (the Q's input is a graph
    input or a node output) whose zero point is a uint16 initializer: the
    scale is multiplied by ``65535 / 255`` and the zero point mapped with
    ``round(zp * 255 / 65535)`` to uint8. Constants quantized with uint16
    (a DQ fed directly by an initializer) are re-quantized to uint8 with the
    new scale / zero point. Finally, for ``Conv`` / ``ConvTranspose`` /
    ``Gemm`` with an int32 bias DQ, the bias scale is reset to
    ``x_scale * w_scale``. Initializers shared between pairs are updated once.

    Differences from Quark: requantized constants are rounded to nearest
    (Quark truncates toward zero); nodes without a zero-point input are
    skipped instead of raising.
    """
    m = _copy(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    prod = _producers(g)
    graph_in = {i.name for i in g.input}
    src, dst = TensorProto.UINT16, TensorProto.UINT8
    qrange = lambda t: _INT_RANGE[t][1] - _INT_RANGE[t][0]  # noqa: E731
    scaled: Set[str] = set()
    moved: Set[str] = set()

    def is_dq(n: Optional[onnx.NodeProto]) -> bool:
        return n is not None and n.op_type in _DQ_NAMES

    def is_q(n: Optional[onnx.NodeProto]) -> bool:
        return n is not None and n.op_type in _Q_NAMES

    def rescale(node: onnx.NodeProto) -> None:
        s, z = _qdq_scale_zp_inits(node, inits)
        if s is None or z is None or z.data_type != src or s.name in scaled:
            return
        arr = numpy_helper.to_array(s)
        new = (arr * qrange(src) / qrange(dst)).astype(np.float32)
        s.CopyFrom(numpy_helper.from_array(new, s.name))
        scaled.add(s.name)

    def remap_zp(node: onnx.NodeProto) -> None:
        s, z = _qdq_scale_zp_inits(node, inits)
        if z is None or z.name in moved or z.data_type != src:
            return
        zp = numpy_helper.to_array(z).astype(np.float32)
        new = np.rint(zp * np.float32(qrange(dst) / qrange(src)))
        z.CopyFrom(numpy_helper.from_array(new.astype(np.uint8), z.name))
        moved.add(z.name)

    pairs = []
    for dq in g.node:
        if not is_dq(dq):
            continue
        q = prod.get(dq.input[0])
        if not is_q(q):
            continue
        if q.input[0] not in graph_in and q.input[0] not in prod:  # type: ignore[union-attr]
            continue  # a constant: handled below
        pairs.append((q, dq))
    for q, dq in pairs:
        rescale(dq)
        if q.input[1] != dq.input[1]:  # type: ignore[union-attr]
            rescale(q)  # type: ignore[arg-type]
    for q, dq in pairs:
        remap_zp(dq)
        if q.input[2:3] != dq.input[2:3]:  # type: ignore[union-attr]
            remap_zp(q)  # type: ignore[arg-type]

    # constants stored as uint16 behind a DQ
    for dq in g.node:
        if not is_dq(dq) or len(dq.input) < 3:
            continue
        s, z = _qdq_scale_zp_inits(dq, inits)
        w = inits.get(dq.input[0])
        if s is None or z is None or w is None or z.data_type != src:
            continue
        old_s, old_z = numpy_helper.to_array(s), numpy_helper.to_array(z)
        rescale(dq)
        remap_zp(dq)
        new_s, new_z = numpy_helper.to_array(s), numpy_helper.to_array(z)
        real = (numpy_helper.to_array(w).astype(np.float32) - old_z) * old_s
        lo, hi = _INT_RANGE[dst]
        q = np.clip(np.rint(real / new_s + new_z), lo, hi).astype(np.uint8)
        w.CopyFrom(numpy_helper.from_array(q, w.name))

    for n in g.node:
        if n.op_type not in ("Conv", "ConvTranspose", "Gemm") or len(n.input) < 3:
            continue
        dqs = [prod.get(x) for x in n.input[:3]]
        if not all(is_dq(d) for d in dqs):
            continue
        sx, sw, sb = (inits.get(d.input[1]) for d in dqs)  # type: ignore[union-attr]
        zb = inits.get(dqs[2].input[2]) if len(dqs[2].input) > 2 else None  # type: ignore[union-attr]
        if sx is None or sw is None or sb is None:
            continue
        if zb is None or zb.data_type != TensorProto.INT32:
            continue
        new = numpy_helper.to_array(sx) * numpy_helper.to_array(sw)
        sb.CopyFrom(numpy_helper.from_array(new, sb.name))
    _prune_initializers(g)
    return m


def convert_u16s8_to_s16s8(model: onnx.ModelProto) -> onnx.ModelProto:
    """Re-express uint16 *activation* Q/DQ as int16 by shifting the zero
    point by 32768 (``q' = q - 32768``; the represented values are identical).
    Q nodes with a non-constant input and DQ nodes whose input is not a
    constant or a weight-quantizer output are converted; weights are left
    alone. Quark only handles zero points 32767 / 32768 and maps both to int16
    0; here every scalar uint16 zero point is shifted exactly, which agrees
    with Quark for 32768."""
    m = _copy(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    prod = _producers(g)
    taken = _all_names(g)
    new_inits: Dict[int, str] = {}

    def int16_zp(name: str) -> Optional[str]:
        z = inits.get(name)
        if z is None or z.data_type != TensorProto.UINT16 or z.dims:
            return None
        v = int(numpy_helper.to_array(z)) - 32768
        if v not in new_inits:
            nm = _unique_name(taken, "int16_zp0" if v == 0 else f"int16_zp{v}")
            g.initializer.append(numpy_helper.from_array(np.array(v, np.int16), nm))
            new_inits[v] = nm
        return new_inits[v]

    for n in g.node:
        if len(n.input) < 3:
            continue
        if n.op_type == "QuantizeLinear" and n.input[0] not in inits:
            new = int16_zp(n.input[2])
            if new:
                n.input[2] = new
        elif n.op_type == "DequantizeLinear" and n.input[0] not in inits:
            q = prod.get(n.input[0])
            if q is not None and q.op_type == "QuantizeLinear" and q.input[0] in inits:
                continue  # weight quantizer
            new = int16_zp(n.input[2])
            if new:
                n.input[2] = new
    _prune_initializers(g)
    return m


def convert_bias_int32_to_int16(
    model: onnx.ModelProto,
) -> Tuple[onnx.ModelProto, bool]:
    """int32 bias -> int16 bias in a QDQ model; returns ``(model, changed)``.

    The default-domain opset is raised to 21 first (the first opset with
    16-bit DQ) if it is lower. For ``Conv`` / ``ConvTranspose`` / ``Gemm`` /
    ``LayerNormalization`` / ``InstanceNormalization`` /
    ``BatchNormalization`` whose bias (input 2) comes from a 3-input
    ``DequantizeLinear``, the DQ's int32 data and int32 zero point are
    clipped to int16 range and stored as int16."""
    from onnxsim.quark_tools import convert_opset_version

    m = _copy(model)
    cur = next((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), 0)
    if cur < 21:
        m = convert_opset_version(m, 21)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    prod = _producers(g)
    ops = {
        "Conv",
        "ConvTranspose",
        "Gemm",
        "LayerNormalization",
        "InstanceNormalization",
        "BatchNormalization",
    }
    flag = False
    for n in g.node:
        if n.op_type not in ops or len(n.input) <= 2:
            continue
        dq = prod.get(n.input[2])
        if dq is None or len(dq.input) != 3:
            continue
        for name in (dq.input[0], dq.input[2]):
            t = inits.get(name)
            if t is not None and t.data_type == TensorProto.INT32:
                arr = np.clip(numpy_helper.to_array(t), -32768, 32767)
                t.CopyFrom(numpy_helper.from_array(arr.astype(np.int16), name))
                flag = True
    return m, flag


def convert_a8w8_npu_to_a8w8_cpu(model: onnx.ModelProto) -> onnx.ModelProto:
    """NPU-style A8W8 (int8 bias) -> CPU-style A8W8 (int32 bias).

    For each ``Conv`` / ``ConvTranspose`` / ``Gemm`` with a bias whose
    activation, weight and bias come from ``DequantizeLinear`` nodes: the bias
    scale becomes ``act_scale * weight_scale`` (float32), the stored bias
    becomes ``int32(bias_q * old_scale / new_scale)`` (truncating, as Quark
    does) and its zero point an int32 zero. The initializers keep their names.
    A bias initializer shared with another node is copied under a new name
    instead (Quark overwrites it)."""
    m = _copy(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    prod = _producers(g)
    taken = _all_names(g)
    uses: Dict[str, int] = defaultdict(int)
    for n in g.node:
        for x in set(n.input):
            uses[x] += 1

    for n in g.node:
        if n.op_type not in ("Conv", "ConvTranspose", "Gemm") or len(n.input) != 3:
            continue
        a, w, b = (prod.get(x) for x in n.input)
        if not all(
            d is not None and d.op_type == "DequantizeLinear" for d in (a, w, b)
        ):
            continue
        sa, sw, sb = (inits.get(d.input[1]) for d in (a, w, b))  # type: ignore[union-attr]
        bq = inits.get(b.input[0])  # type: ignore[union-attr]
        if sa is None or sw is None or sb is None or bq is None:
            continue
        zpb = inits.get(b.input[2]) if len(b.input) > 2 else None  # type: ignore[union-attr]
        old_scale = numpy_helper.to_array(sb).astype(np.float32)
        new_scale = numpy_helper.to_array(sa).astype(
            np.float32
        ) * numpy_helper.to_array(sw).astype(np.float32)
        zp = 0 if zpb is None else numpy_helper.to_array(zpb).astype(np.float32)
        real = (numpy_helper.to_array(bq).astype(np.float32) - zp) * old_scale
        q32 = (real / new_scale).astype(np.int32)
        names = [bq.name, sb.name, zpb.name if zpb is not None else bq.name + "_zp"]
        if uses[bq.name] > 1 or uses[sb.name] > 1:
            names = [_unique_name(taken, x + "_i32") for x in names]
        drop = set(names)
        keep = [t for t in g.initializer if t.name not in drop]
        del g.initializer[:]
        g.initializer.extend(keep)
        g.initializer.extend(
            [
                numpy_helper.from_array(q32, names[0]),
                numpy_helper.from_array(new_scale.astype(np.float32), names[1]),
                numpy_helper.from_array(np.zeros((), np.int32), names[2]),
            ]
        )
        inits = {t.name: t for t in g.initializer}
        b.input[:] = names  # type: ignore[union-attr]
    _prune_initializers(g)
    return m


def convert_resize_fs_to_pof2s(model: onnx.ModelProto) -> onnx.ModelProto:
    """Make the Q/DQ scale pairs around every ``Resize`` power-of-two.

    For the ``Q -> DQ`` feeding a Resize and the ``Q -> DQ`` consuming its
    output, each node's int8 scale / zero point is replaced by
    ``2 ** -floor(-log2(max(|(-128-zp)*s|, |(127-zp)*s|) / 128))`` and an int8
    zero. Only per-tensor int8 nodes are touched (other nodes are skipped,
    where Quark would raise or force int8). Scale / zero-point initializers
    also used by other nodes are copied rather than changed."""
    m = _copy(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    prod = _producers(g)
    cons = _consumers(g)
    taken = _all_names(g)
    targets: List[onnx.NodeProto] = []

    def add(n: Optional[onnx.NodeProto]) -> None:
        if n is not None and not any(n is t for t in targets):
            targets.append(n)

    for r in g.node:
        if r.op_type != "Resize":
            continue
        dq = prod.get(r.input[0])
        if dq is not None and dq.op_type == "DequantizeLinear":
            q = prod.get(dq.input[0])
            if q is not None and q.op_type == "QuantizeLinear":
                add(q)
                add(dq)
        for q in cons.get(r.output[0], []):
            if q.op_type != "QuantizeLinear":
                continue
            for dq in cons.get(q.output[0], []):
                if dq.op_type == "DequantizeLinear" and dq.input[0] == q.output[0]:
                    add(q)
                    add(dq)

    tset = {id(t) for t in targets}
    for n in targets:
        if len(n.input) < 3:
            continue
        s, z = inits.get(n.input[1]), inits.get(n.input[2])
        if s is None or z is None or z.data_type != TensorProto.INT8:
            continue
        sv = numpy_helper.to_array(s).astype(np.float64)
        zv = numpy_helper.to_array(z).astype(np.float64)
        if sv.size != 1 or zv.size != 1:
            continue
        f_min, f_max = (-128 - zv) * sv, (127 - zv) * sv
        new = float(np.maximum(np.abs(f_max), np.abs(f_min)).item() / 128)
        new = min(max(new, 2.0**-127), 2.0**127)
        pos = int(np.floor(-np.log2(new)))
        pof2 = np.array(2.0**-pos, np.float32).reshape(sv.shape)
        zero = np.zeros(zv.shape, np.int8)
        for slot, name, arr in ((1, s.name, pof2), (2, z.name, zero)):
            shared = any(id(c) not in tset for c in cons.get(name, []))
            tgt = _unique_name(taken, name + "_pof2") if shared else name
            t = numpy_helper.from_array(arr, tgt)
            if shared:
                g.initializer.append(t)
                n.input[slot] = tgt
            else:
                inits[name].CopyFrom(t)
    _prune_initializers(g)
    return m


# -- bfloat16 ------------------------------------------------------------------


def convert_fp16_to_bf16(model: onnx.ModelProto) -> onnx.ModelProto:
    """float16 -> bfloat16, keeping float16 at the graph boundary (Quark's
    ``convert_fp16_to_bf16`` ``format="bf16"``): float16 initializers and
    ``Constant`` tensors are rounded to bfloat16 (nearest-even), ``Cast(to=
    float16)`` becomes ``Cast(to=bfloat16)``, every float16 graph input is
    consumed through a ``Cast(to=bfloat16)`` named ``<in>_cast`` and every
    float16 output is produced through ``<out>_cast`` and a trailing
    ``Cast(to=float16)``."""
    m = _copy(model)
    g = m.graph
    f16, bf16 = TensorProto.FLOAT16, TensorProto.BFLOAT16

    def conv(t: onnx.TensorProto) -> None:
        arr = numpy_helper.to_array(t).astype(np.float32)
        t.CopyFrom(_bf16_tensor(arr, t.name))

    for t in g.initializer:
        if t.data_type == f16:
            conv(t)
    for n in g.node:
        for a in n.attribute:
            if n.op_type == "Constant" and a.name == "value":
                if a.t.data_type == f16:
                    conv(a.t)
            elif n.op_type == "Cast" and a.name == "to" and a.i == f16:
                a.i = bf16

    pre: List[onnx.NodeProto] = []
    ins = [
        i.name
        for i in g.input
        if i.type.HasField("tensor_type") and i.type.tensor_type.elem_type == f16
    ]
    for n in g.node:
        for k, x in enumerate(n.input):
            if x in ins:
                n.input[k] = x + "_cast"
    for x in ins:
        pre.append(helper.make_node("Cast", [x], [x + "_cast"], to=bf16))
    outs = {
        o.name
        for o in g.output
        if o.type.HasField("tensor_type") and o.type.tensor_type.elem_type == f16
    }
    post: List[onnx.NodeProto] = []
    for n in g.node:
        for k, y in enumerate(n.output):
            if y in outs:
                n.output[k] = y + "_cast"
                post.append(helper.make_node("Cast", [y + "_cast"], [y], to=f16))
    for n in g.node:  # consumers of an output read the pre-cast tensor
        for k, x in enumerate(n.input):
            if x in outs:
                n.input[k] = x + "_cast"
    del g.value_info[:]
    old = list(g.node)
    del g.node[:]
    g.node.extend(pre + old + post)
    return m


def _cast_to(n: onnx.NodeProto) -> Optional[int]:
    if not _is_std(n, "Cast"):
        return None
    for a in n.attribute:
        if a.name == "to":
            return int(a.i)
    return None


def remove_bf16_cast(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove ``Cast(to=bfloat16) -> Cast(to=float32)`` round trips (the
    "simulated bfloat16" pattern):

    - between two nodes: both casts are removed and the consumers read the
      original tensor (the first cast must be the only consumer of its
      producer's output);
    - on a constant: the float32 initializer is replaced by its bfloat16-rounded
      values (still float32, named ``<init>_bf16``);
    - at a graph output: both casts are removed and the producer writes the
      output directly.
    """
    m = _copy(model)
    g = m.graph
    outputs = {o.name for o in g.output}

    changed = True
    while changed:
        changed = False
        cons = _consumers(g)
        prod = _producers(g)
        inits = {t.name: t for t in g.initializer}
        for c1 in list(g.node):
            if _cast_to(c1) != TensorProto.BFLOAT16 or c1.output[0] in outputs:
                continue
            nxt = cons.get(c1.output[0], [])
            if len(nxt) != 1 or _cast_to(nxt[0]) != TensorProto.FLOAT:
                continue
            c2 = nxt[0]
            src = c1.input[0]
            if src in inits:  # constant folding
                t = inits[src]
                if t.data_type != TensorProto.FLOAT:
                    continue
                new_name = src + "_bf16"
                rounded = _round_to_bf16(numpy_helper.to_array(t))
                g.initializer.append(numpy_helper.from_array(rounded, new_name))
                _replace_input(g, c2.output[0], new_name)
                if c2.output[0] in outputs:
                    continue
            elif c2.output[0] in outputs:  # graph output
                p = prod.get(src)
                if p is None or len(cons.get(src, [])) != 1 or src in outputs:
                    continue
                p.output[list(p.output).index(src)] = c2.output[0]
            else:
                if len(cons.get(src, [])) != 1:
                    continue  # the producer fans out; Quark requires one consumer
                _replace_input(g, c2.output[0], src)
            g.node.remove(c1)
            g.node.remove(c2)
            changed = True
            break
    _prune_initializers(g)
    return m


def replace_bfloat16_qdq_cast(model: onnx.ModelProto) -> onnx.ModelProto:
    """Turn bfloat16 ``ExtendedQuantizeLinear`` / ``ExtendedDequantizeLinear``
    (zero point an all-zero bfloat16 initializer) into plain casts:
    ``Q(x) -> Cast(bf16)(x * (1/scale))`` and ``DQ(x) -> Cast(float)(x) *
    scale``; the ``Mul`` is omitted when the scale is exactly 1. (Quark skips
    the ``Mul`` only when *every* scale element is 1 as well -- its test
    ``np.all(scale != 1)`` also skips it for mixed scales, which this does
    not.) Constants are named ``<node>_scale`` / ``<node>_mul_out`` /
    ``<node>_cast_out``."""
    m = _copy(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    new_nodes: List[onnx.NodeProto] = []
    for n in g.node:
        if (
            n.op_type
            not in (
                "ExtendedQuantizeLinear",
                "ExtendedDequantizeLinear",
            )
            or len(n.input) < 3
        ):
            new_nodes.append(n)
            continue
        scale_t, zp_t = inits.get(n.input[1]), inits.get(n.input[2])
        if (
            scale_t is None
            or zp_t is None
            or zp_t.data_type != TensorProto.BFLOAT16
            or numpy_helper.to_array(zp_t).astype(np.float32).any()
        ):
            new_nodes.append(n)
            continue
        scale = numpy_helper.to_array(scale_t)
        need_mul = bool(np.any(scale != 1))
        is_q = n.op_type == "ExtendedQuantizeLinear"
        if need_mul:
            sname = f"{n.name}_scale"
            factor = (1.0 / scale) if is_q else scale
            g.initializer.append(
                numpy_helper.from_array(np.asarray(factor, np.float32), sname)
            )
        if is_q:
            src = n.input[0]
            if need_mul:
                new_nodes.append(
                    helper.make_node("Mul", [src, sname], [f"{n.name}_mul_out"])
                )
                src = f"{n.name}_mul_out"
            new_nodes.append(
                helper.make_node("Cast", [src], list(n.output), to=TensorProto.BFLOAT16)
            )
        else:
            out = f"{n.name}_cast_out" if need_mul else n.output[0]
            new_nodes.append(
                helper.make_node("Cast", [n.input[0]], [out], to=TensorProto.FLOAT)
            )
            if need_mul:
                new_nodes.append(helper.make_node("Mul", [out, sname], list(n.output)))
    del g.node[:]
    g.node.extend(new_nodes)
    _prune_initializers(g)
    return m


def insert_clip_bfloat16_qdq(model: onnx.ModelProto) -> onnx.ModelProto:
    """Put a ``Clip(-bf16_max, +bf16_max)`` in front of every bfloat16
    activation ``ExtendedQuantizeLinear`` (non-constant input, bfloat16 zero
    point) so overflowing values saturate instead of becoming inf. The clip
    bounds are float32 scalars named ``<input>_clip_min`` / ``_clip_max`` and
    the clipped tensor ``<input>_clip_output`` (shared when several Q nodes
    read the same tensor)."""
    m = _copy(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    bf16_max = 3.38953139e38
    made: Dict[str, str] = {}
    out_nodes: List[onnx.NodeProto] = []
    for n in g.node:
        if (
            n.op_type == "ExtendedQuantizeLinear"
            and len(n.input) >= 3
            and n.input[0] not in inits
            and n.input[2] in inits
            and inits[n.input[2]].data_type == TensorProto.BFLOAT16
        ):
            x = n.input[0]
            if x not in made:
                lo, hi = x + "_clip_min", x + "_clip_max"
                g.initializer.append(
                    numpy_helper.from_array(np.array(-bf16_max, np.float32), lo)
                )
                g.initializer.append(
                    numpy_helper.from_array(np.array(bf16_max, np.float32), hi)
                )
                made[x] = x + "_clip_output"
                out_nodes.append(
                    helper.make_node("Clip", [x, lo, hi], [made[x]], name=x + "_clip")
                )
            n.input[0] = made[x]
        out_nodes.append(n)
    del g.node[:]
    g.node.extend(out_nodes)
    return m


# -- shapes ---------------------------------------------------------------------


def parse_input_and_output_shapes(spec: str) -> Dict[str, List[int]]:
    """``"a:[1,3];b:[2]"`` -> ``{"a": [1, 3], "b": [2]}`` (Quark's
    ``--fix_shapes`` syntax)."""
    out: Dict[str, List[int]] = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        name, _, dims = item.rpartition(":")
        dims = dims.strip()
        if not (dims.startswith("[") and dims.endswith("]")):
            raise ValueError(f"bad shape spec {item!r}; expected name:[d0,d1,...]")
        out[name.strip()] = [int(d) for d in dims[1:-1].split(",") if d.strip()]
    return out


def fix_input_and_output_shapes(
    model: onnx.ModelProto, shapes: Union[str, Dict[str, Sequence[int]]]
) -> onnx.ModelProto:
    """Set the dimensions of the named graph inputs / outputs to the given
    static values. Names that are not graph inputs / outputs, and inputs /
    outputs not mentioned, are left alone (Quark raises ``KeyError`` for an
    unmentioned one). Dimensions are written as ``dim_value``; a symbolic
    ``dim_param`` is cleared."""
    spec = parse_input_and_output_shapes(shapes) if isinstance(shapes, str) else shapes
    m = _copy(model)
    for vi in list(m.graph.input) + list(m.graph.output):
        if vi.name not in spec:
            continue
        dims = list(spec[vi.name])
        tshape = vi.type.tensor_type.shape
        if len(tshape.dim) != len(dims):
            del tshape.dim[:]
            tshape.dim.extend(onnx.TensorShapeProto.Dimension() for _ in dims)
        for d, v in zip(tshape.dim, dims):
            d.ClearField("dim_param")
            d.dim_value = int(v)
    return m


def fix_shapes(
    model: onnx.ModelProto,
    shapes: Union[None, str, Dict[str, Sequence[int]]] = None,
) -> onnx.ModelProto:
    """Give every tensor a static shape.

    Optionally fixes input / output shapes first (:func:`fix_input_and_output_
    shapes`), then records the shape of each intermediate tensor in
    ``value_info``. Shapes come from ONNX shape inference, and, for tensors it
    cannot resolve, from running ONNX Runtime on random inputs of the (now
    static) input shapes -- the way Quark gets all of them. Without
    onnxruntime, or when the inputs are still dynamic, only inference is used.
    Existing static ``value_info`` entries are kept. Nothing is changed for
    tensors whose shape is data dependent and cannot be inferred."""
    m = fix_input_and_output_shapes(model, shapes) if shapes else _copy(model)
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception:  # noqa: BLE001
        pass
    g = m.graph

    def static(vi: onnx.ValueInfoProto) -> bool:
        t = vi.type.tensor_type
        return t.HasField("shape") and all(
            d.WhichOneof("value") == "dim_value" for d in t.shape.dim
        )

    known = {v.name for v in g.value_info if static(v)}
    known |= {v.name for v in list(g.input) + list(g.output) if static(v)}
    known |= {t.name for t in g.initializer}
    missing = [o for n in g.node for o in n.output if o and o not in known]
    if missing and all(
        static(i) for i in g.input if i.name not in {t.name for t in g.initializer}
    ):
        try:
            import onnxruntime as ort

            probe = _copy(m)
            existing = {o.name for o in probe.graph.output}
            for name in missing:
                if name not in existing:
                    probe.graph.output.append(onnx.ValueInfoProto(name=name))
            so = ort.SessionOptions()
            so.log_severity_level = 4
            sess = ort.InferenceSession(
                probe.SerializeToString(), so, providers=["CPUExecutionProvider"]
            )
            rng = np.random.default_rng(42)
            feed = {}
            for i in sess.get_inputs():
                dt = np.dtype(
                    {
                        "tensor(float)": np.float32,
                        "tensor(double)": np.float64,
                        "tensor(float16)": np.float16,
                        "tensor(int64)": np.int64,
                        "tensor(int32)": np.int32,
                        "tensor(int8)": np.int8,
                        "tensor(uint8)": np.uint8,
                        "tensor(bool)": np.bool_,
                    }[i.type]
                )
                feed[i.name] = rng.random(i.shape).astype(dt)
            names = [o.name for o in sess.get_outputs()]
            vals = sess.run(None, feed)
            by_name = {v.name: v for v in g.value_info}
            elem = {}
            for vi in probe.graph.value_info:
                elem[vi.name] = vi.type.tensor_type.elem_type
            for name, val in zip(names, vals):
                if name not in set(missing):
                    continue
                et = elem.get(name) or helper.np_dtype_to_tensor_dtype(val.dtype)
                new = helper.make_tensor_value_info(name, et, list(val.shape))
                if name in by_name:
                    by_name[name].CopyFrom(new)
                else:
                    g.value_info.append(new)
        except Exception:  # noqa: BLE001 - ORT absent / model not runnable
            pass
    return m


# -- inspection ----------------------------------------------------------------


def a16w8_a8w8_nodes(model: onnx.ModelProto) -> Tuple[List[str], List[str]]:
    """Names of the quantized ``Conv`` / ``ConvTranspose`` / ``Gemm`` nodes
    whose activation input is dequantized with an **int8** (first list) or
    **int16** (second list) zero point. A node counts as quantized when its
    first input comes from a ``DequantizeLinear`` or its first output feeds a
    ``QuantizeLinear``; 8-bit unsigned and other zero-point types appear in
    neither list. (Quark's ``print_a16w8_a8w8_nodes`` tool takes a path and
    prints this.)"""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    prod = _producers(g)
    cons = _consumers(g)
    int8_names: List[str] = []
    int16_names: List[str] = []
    for n in g.node:
        if n.op_type not in ("Conv", "ConvTranspose", "Gemm"):
            continue
        if not n.input or not n.output:
            continue
        dq = prod.get(n.input[0])
        feeds_q = any(
            c.op_type == "QuantizeLinear" and c.input and c.input[0] == n.output[0]
            for c in cons.get(n.output[0], [])
        )
        if not ((dq is not None and dq.op_type == "DequantizeLinear") or feeds_q):
            continue
        if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) < 3:
            continue
        zp = inits.get(dq.input[2])
        if zp is None:
            continue
        if zp.data_type == TensorProto.INT8:
            int8_names.append(n.name)
        elif zp.data_type == TensorProto.INT16:
            int16_names.append(n.name)
    return int8_names, int16_names


# -- text format ---------------------------------------------------------------


def convert_onnx_to_onnxtxt(model: onnx.ModelProto) -> str:
    """The protobuf text-format dump of a model (``.onnxtxt``)."""
    from google.protobuf import text_format

    return str(text_format.MessageToString(model))


def convert_onnxtxt_to_onnx(text: Union[str, bytes]) -> onnx.ModelProto:
    """Parse a protobuf text-format dump back into a model."""
    from google.protobuf import text_format

    m = onnx.ModelProto()
    text_format.Parse(text, m)
    return m


# -- shared nodes / initializers -----------------------------------------------


def check_shared_initializers(model: onnx.ModelProto) -> bool:
    """True if some initializer is read by more than one input slot (Quark's
    ``check_shared_initializers``)."""
    names = {t.name for t in model.graph.initializer}
    seen: Set[str] = set()
    for n in model.graph.node:
        for x in n.input:
            if x in names:
                if x in seen:
                    return True
                seen.add(x)
    return False


def copy_shared_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Give every consumer its own copy of a shared initializer or shared
    ``DequantizeLinear`` (repeated until nothing is shared, so a duplicated DQ
    also gets its own scale / zero point). Like Quark, **all nodes are first
    renamed** ``<OpType>_<k>`` (k counting per op type from 1); copies are named
    ``<original>_<j>`` and a copied DQ outputs ``<name>_out``. Tensors without
    a node / initializer producer (graph inputs) and other ops are not
    duplicated."""
    m = _copy(model)
    g = m.graph
    counter: Dict[str, int] = defaultdict(int)
    for n in g.node:
        counter[n.op_type] += 1
        n.name = f"{n.op_type}_{counter[n.op_type]}"
    modified = True
    while modified:
        modified = False
        inits = {t.name: t for t in g.initializer}
        prod = _producers(g)
        seen: Dict[str, int] = defaultdict(int)
        renames: List[Tuple[onnx.NodeProto, int, str]] = []
        add_inits: List[onnx.TensorProto] = []
        add_nodes: List[onnx.NodeProto] = []
        for n in g.node:
            for k, x in enumerate(n.input):
                if not x:
                    continue
                if seen[x] == 0:
                    seen[x] = 1
                    continue
                idx = seen[x]
                if x in inits:
                    t = onnx.TensorProto()
                    t.CopyFrom(inits[x])
                    t.name = f"{x}_{idx}"
                    add_inits.append(t)
                    renames.append((n, k, t.name))
                elif x in prod and prod[x].op_type == "DequantizeLinear":
                    p = onnx.NodeProto()
                    p.CopyFrom(prod[x])
                    p.name = f"{p.name}_{idx}"
                    p.output[0] = p.name + "_out"
                    add_nodes.append(p)
                    renames.append((n, k, p.output[0]))
                else:
                    continue
                seen[x] += 1
        if not renames:
            break
        existing = {t.name for t in g.initializer}
        for t in add_inits:
            if t.name not in existing:
                g.initializer.append(t)
                existing.add(t.name)
        have = {n.output[0] for n in g.node}
        for p in add_nodes:
            if p.output[0] not in have:
                g.node.append(p)
                have.add(p.output[0])
        for n, k, name in renames:
            n.input[k] = name
        modified = True
    # copies were appended at the end; restore a topological order
    return _toposort(m)


def _toposort(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    avail = {i.name for i in g.input} | {t.name for t in g.initializer} | {""}
    pending = list(g.node)
    ordered: List[onnx.NodeProto] = []
    while pending:
        progressed = False
        rest = []
        for n in pending:
            deps = set(n.input) | _subgraph_names(n.attribute)
            local = {o for sn in [n] for o in sn.output}
            deps -= local
            outer_unknown = {
                d
                for d in deps
                if d not in avail and d in {o for p in pending for o in p.output}
            }
            if outer_unknown:
                rest.append(n)
            else:
                ordered.append(n)
                avail.update(n.output)
                progressed = True
        if not progressed:
            ordered.extend(rest)
            break
        pending = rest
    del g.node[:]
    g.node.extend(ordered)
    return model


def save_onnx_model_with_external_data(
    model: onnx.ModelProto, path: str, save_as_external_data: bool = False
) -> None:
    """Save ``model`` to ``path``. With ``save_as_external_data`` all tensors
    (and tensor attributes) go to one ``<path>.data`` file next to it (an old
    file of that name is removed first). Mutates ``model``'s tensors to refer
    to the external file, like ``onnx.save(save_as_external_data=True)``."""
    import os

    if save_as_external_data:
        location = os.path.basename(path) + ".data"
        full = os.path.join(os.path.dirname(os.path.abspath(path)), location)
        if os.path.exists(full):
            os.remove(full)
        onnx.external_data_helper.convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=location,
            convert_attribute=True,
        )
    onnx.save(model, path)


def clean_initializer_in_input(model: onnx.ModelProto) -> onnx.ModelProto:
    """Quark's name for :func:`onnxsim.quark_tools.remove_initializer_from_input`
    (returns a copy)."""
    from onnxsim.quark_tools import remove_initializer_from_input

    return remove_initializer_from_input(model)


__all__ = [
    "CUSTOM_OP_NAME_MAPPING",
    "a16w8_a8w8_nodes",
    "check_shared_initializers",
    "clean_initializer_in_input",
    "convert_a8w8_npu_to_a8w8_cpu",
    "convert_bias_int32_to_int16",
    "convert_custom_ops",
    "convert_customqdq_to_qdq",
    "convert_fp16_to_bf16",
    "convert_nchw_to_nhwc",
    "convert_onnx_to_onnxtxt",
    "convert_onnxtxt_to_onnx",
    "convert_qdq_to_qop",
    "convert_resize_fs_to_pof2s",
    "convert_u16s8_to_s16s8",
    "convert_u16u8_to_u8u8",
    "copy_shared_nodes",
    "fix_input_and_output_shapes",
    "fix_shapes",
    "insert_clip_bfloat16_qdq",
    "parse_input_and_output_shapes",
    "remove_bf16_cast",
    "remove_qdq_between_ops",
    "remove_qdq_mul_add",
    "replace_bfloat16_qdq_cast",
    "save_onnx_model_with_external_data",
]
