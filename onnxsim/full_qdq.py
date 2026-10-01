"""Whole-graph static QDQ quantization for NPU backends (QNN HTP, ...).

:func:`onnxsim.quantize_static` wraps only the *inputs* of MatMul/Gemm/Conv in
QuantizeLinear/DequantizeLinear. That is the right shape for a CPU runtime
that fuses ``DQ -> MatMul`` into an integer kernel, but an NPU compiler such
as Qualcomm's QNN HTP backend only runs a node in integer arithmetic when it
forms a complete *QDQ node unit*: every float input comes from a
``DequantizeLinear`` and every output goes straight into a
``QuantizeLinear``. A Conv whose output stays float (and every Relu/Add/
MaxPool in between) runs in fp16 instead, with conversions on both sides.

:func:`quantize_full_qdq` produces that whole-graph form:

- every float activation touching a quantized node gets a calibrated
  ``Q -> DQ`` pair (uint8 by default, uint16 for ``activation_dtype="uint16"``);
- Conv/ConvTranspose/Gemm/MatMul constant weights become int8, symmetric,
  per output channel; Conv/Gemm biases become int32 with scale
  ``input_scale * weight_scale`` (what integer accumulators need);
- any other constant input of a quantized node is quantized per tensor with
  the activation dtype;
- data-movement ops (Reshape, Transpose, MaxPool, Resize, GridSample, ...)
  reuse their input's quantization parameters for their output, so they are
  exact and need no requantization;
- a Relu right after a quantized producer is folded into that producer's
  output quantization (a uint8 Q with zero point 0 already clamps at 0);
- nodes can be left in float (``op_types`` include list, ``exclude_op_types``,
  ``exclude_nodes``): mixed precision, e.g. keeping LayerNorm/Softmax and
  sampling coordinates in fp16 while the Linear layers run in int8.

:func:`quantized_io` then optionally drops the float boundary: a graph input
becomes the uint8 tensor its ``QuantizeLinear`` produced (the caller
quantizes on the host, typically for free in preprocessing), a graph output
becomes the quantized tensor before its ``DequantizeLinear``.

Calibration reuses :func:`onnxsim.calibration.calibrate` (every calibration
method it supports is available through ``method``).
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnxsim.calibration import Tensors, calibrate

__all__ = ["quantize_full_qdq", "quantized_io", "sampling_coordinate_tensors"]

# Output values are a subset / convex combination of the data input's values
# (with 0 for padding, and every range here contains 0), so the output can
# share the input's quantization parameters exactly.
_SHARED_QPARAM_OPS = {
    "Reshape",
    "Transpose",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
    "MaxPool",
    "Slice",
    "Split",
    "Expand",
    "Tile",
    "Gather",
    "DepthToSpace",
    "SpaceToDepth",
    "Identity",
    "GridSample",
    "Resize",
}

# Inputs that carry tensor data; the rest (shapes, indices, scales, axes, ...)
# are never quantized. Ops not listed: every input is data.
_DATA_INPUTS = {
    "Reshape": (0,),
    "Expand": (0,),
    "Tile": (0,),
    "Slice": (0,),
    "Split": (0,),
    "Squeeze": (0,),
    "Unsqueeze": (0,),
    "Gather": (0,),
    "Resize": (0,),
    "Pad": (0,),
    "Clip": (0,),
    "TopK": (0,),
    "ReduceMean": (0,),
    "ReduceSum": (0,),
    "ReduceMax": (0,),
}

# Never quantized: shape arithmetic, existing Q/DQ, control flow.
_NEVER_QUANTIZED = {
    "QuantizeLinear",
    "DequantizeLinear",
    "Shape",
    "Size",
    "Constant",
    "ConstantOfShape",
    "Range",
    "NonZero",
    "Cast",
    "If",
    "Loop",
    "Scan",
    "ArgMax",
    "ArgMin",
}

_WEIGHT_AXIS_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}

_DTYPES = {
    "uint8": (TensorProto.UINT8, np.uint8, 0, 255),
    "uint16": (TensorProto.UINT16, np.uint16, 0, 65535),
    "int8": (TensorProto.INT8, np.int8, -128, 127),
    "int16": (TensorProto.INT16, np.int16, -32768, 32767),
}
_WIDE_DTYPES = ("uint16", "int16")  # need com.microsoft Q/DQ below opset 21


def _pof2(scale: float) -> float:
    """The smallest power of two >= ``scale`` (never clips more than ``scale``)."""
    return float(2.0 ** np.ceil(np.log2(scale))) if scale > 0 else scale


def _qparams(
    lo: float,
    hi: float,
    qmin: int,
    qmax: int,
    symmetric: bool = False,
    power_of_two: bool = False,
) -> Tuple[float, int]:
    """``(scale, zero_point)`` for the range ``[lo, hi]`` (always including 0).

    ``symmetric``: ``scale = absmax / half-range`` with the zero point in the
    middle of the integer range -- 0 for signed types, 128 / 32768 for unsigned
    ones. ``power_of_two`` rounds the scale up to a power of two."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if symmetric:
        absmax = max(-lo, hi)
        half = qmax if qmin < 0 else (qmax - qmin) // 2
        scale = absmax / half
        zp = 0 if qmin < 0 else half + 1
        if not scale > 0:
            return 1.0, zp
        return (_pof2(scale) if power_of_two else scale), zp
    scale = (hi - lo) / (qmax - qmin)
    if not scale > 0:
        return 1.0, qmin
    if power_of_two:
        scale = _pof2(scale)
    zp = int(np.clip(round(qmin - lo / scale), qmin, qmax))
    return scale, zp


def _weight_axis(node: onnx.NodeProto, rank: int) -> Optional[int]:
    if node.op_type == "Conv":
        return 0
    if node.op_type == "ConvTranspose":
        return 1
    if rank != 2:
        return None
    if node.op_type == "Gemm":
        trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
        return 0 if trans_b else 1
    return 1  # MatMul [K, N]


def _float_tensor_names(model: onnx.ModelProto) -> set:
    g = model.graph
    floats = set()
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        if vi.type.tensor_type.elem_type in (TensorProto.FLOAT, TensorProto.FLOAT16):
            floats.add(vi.name)
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            floats.add(init.name)
    return floats


def _constants_to_initializers(model: onnx.ModelProto) -> None:
    g = model.graph
    keep = []
    for n in g.node:
        if (
            n.op_type == "Constant"
            and len(n.attribute) == 1
            and n.attribute[0].name == "value"
        ):
            t = onnx.TensorProto()
            t.CopyFrom(n.attribute[0].t)
            t.name = n.output[0]
            g.initializer.append(t)
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)


def _is_quantized_node(
    n: onnx.NodeProto,
    op_types: Optional[set],
    exclude_op_types: set,
    exclude_nodes: set,
) -> bool:
    if n.domain not in ("", "ai.onnx") or n.op_type in _NEVER_QUANTIZED:
        return False
    if op_types is not None and n.op_type not in op_types:
        return False
    return (
        n.op_type not in exclude_op_types
        and n.name not in exclude_nodes
        and n.output[0] not in exclude_nodes
    )


def _data_inputs(n: onnx.NodeProto) -> List[str]:
    idx = _DATA_INPUTS.get(n.op_type)
    return [x for i, x in enumerate(n.input) if x and (idx is None or i in idx)]


def quantize_full_qdq(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    activation_dtype: str = "uint8",
    per_channel: bool = True,
    op_types: Optional[Iterable[str]] = None,
    exclude_op_types: Iterable[str] = (),
    exclude_nodes: Iterable[str] = (),
    fold_relu: bool = True,
    method: str = "minmax",
    providers: Optional[Sequence[str]] = None,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    tensor_dtypes: Optional[Dict[str, str]] = None,
    symmetric_activations: Optional[bool] = None,
    power_of_two: bool = False,
    weight_dtype: str = "int8",
    convert_inputs: bool = True,
) -> onnx.ModelProto:
    """
    Quantize the whole graph to QDQ form for an NPU backend (see the module
    docstring for the exact rules).

    :param model: onnx ModelProto object or file path (float32)
    :param calibration_data: representative input batches (``{name: array}``),
            run through ONNX Runtime by :func:`onnxsim.calibration.calibrate`.
            Not needed when ``ranges`` is given.
    :param activation_dtype: ``"uint8"`` (default), ``"uint16"`` (W8A16;
            opset < 21 models get ``com.microsoft`` Q/DQ, which ONNX Runtime
            and its QNN execution provider accept), or the signed ``"int8"`` /
            ``"int16"`` (zero point 0)
    :param per_channel: int8 weights per output channel (default) or per tensor
    :param op_types: only quantize nodes of these op types (default: all)
    :param exclude_op_types: never quantize nodes of these op types
    :param exclude_nodes: node names (or first-output names) to keep in float
    :param fold_relu: fold a Relu into its quantized producer's output Q (not
            done with symmetric activations: their zero point is centred)
    :param method: calibration method, passed to
            :func:`onnxsim.calibration.calibrate`
    :param providers: onnxruntime providers for calibration
    :param ranges: precomputed ``{tensor: (min, max)}`` (e.g. calibrated on a
            batch-1 twin of the model); skips calibration for those tensors
    :param tensor_dtypes: per-activation overrides of ``activation_dtype``
            (``{tensor: "uint16"}``), e.g. 16-bit sampling coordinates in an
            otherwise 8-bit graph. Data-movement ops pass their input's dtype
            on unless their output is overridden too.
    :param symmetric_activations: ``scale = absmax / half-range`` with a
            centred zero point (0 for signed types, 128 / 32768 for unsigned).
            Default: on for the signed dtypes, off for the unsigned ones.
    :param power_of_two: round every activation and weight scale up to a
            power of two (fixed-point friendly, as Quark's ``XINT8``)
    :param weight_dtype: ``"int8"`` (default) or ``"int16"`` for the symmetric
            Conv / ConvTranspose / Gemm / MatMul weights (int16 weights use
            ``com.microsoft`` Q/DQ below opset 21, like the 16-bit activations)
    :param convert_inputs: re-quantize an input whose dtype differs from its
            node's output dtype (the 8/16-bit "convert" described above);
            False leaves such nodes consuming one dtype and producing the
            other, as Quark's mixed-precision presets do
    :returns: the quantized ModelProto
    """
    if weight_dtype not in ("int8", "int16"):
        raise ValueError(f"unsupported weight_dtype: {weight_dtype!r}")
    if activation_dtype not in _DTYPES:
        raise ValueError(f"unsupported activation_dtype: {activation_dtype!r}")
    act_type, act_np, qmin, qmax = _DTYPES[activation_dtype]
    if symmetric_activations is None:
        symmetric_activations = qmin < 0
    sym, p2 = symmetric_activations, power_of_two
    if isinstance(model, str):
        model = onnx.load(model)
    m = onnx.ModelProto()
    m.CopyFrom(model)
    _constants_to_initializers(m)
    m = onnx.shape_inference.infer_shapes(m)
    g = m.graph
    op_types = set(op_types) if op_types is not None else None
    exclude_op_types, exclude_nodes = set(exclude_op_types), set(exclude_nodes)

    floats = _float_tensor_names(m)
    inits = {i.name: i for i in g.initializer}
    graph_inputs = {i.name for i in g.input}
    graph_outputs = {o.name for o in g.output}
    consumers = defaultdict(list)
    for n in g.node:
        for x in n.input:
            consumers[x].append(n)

    qnodes = [
        n
        for n in g.node
        if _is_quantized_node(n, op_types, exclude_op_types, exclude_nodes)
    ]
    qnode_ids = {id(n) for n in qnodes}

    # Activations to quantize: float, non-constant data inputs and outputs of quantized nodes.
    acts = []
    seen = set()
    for n in qnodes:
        for x in _data_inputs(n) + [o for o in n.output if o]:
            if x in floats and x not in inits and x not in seen:
                seen.add(x)
                acts.append(x)
    # An explicitly excluded node whose every data input is dequantized and every output
    # quantized would itself form a QDQ node unit (and run quantized). Keep it float by leaving
    # its outputs unquantized; its consumers then read the float value (and run float too).
    # (A node merely outside ``op_types`` is left alone: sandwiched between quantized nodes it
    # runs quantized, which is what an op_types list asks for everywhere else.)
    for n in g.node:
        if id(n) in qnode_ids or not _is_quantized_node(n, op_types, set(), set()):
            continue
        ins = [x for x in _data_inputs(n) if x in floats and x not in inits]
        outs = [o for o in n.output if o and o in floats]
        if (
            ins
            and outs
            and all(x in seen for x in ins)
            and all(o in seen for o in outs)
        ):
            for o in outs:
                seen.discard(o)
            acts = [a for a in acts if a not in outs]

    ranges = dict(ranges or {})
    missing = [a for a in acts if a not in ranges]
    if missing:
        if calibration_data is None:
            raise ValueError(
                f"no calibration data and no ranges for {len(missing)} tensors, e.g. {missing[:3]}"
            )
        # calibrate() reports every quantizable activation, not just the missing ones: the
        # caller's precomputed ranges must win over it
        calibrated = calibrate(
            m,
            calibration_data,
            providers=providers,
            method=method,
            extra_tensor_names=missing,
        )
        ranges = {**calibrated, **ranges}

    # Relu folding: producer -> Relu becomes producer -> Q(range of the Relu output, lo = 0).
    removed = set()
    if fold_relu and not sym:
        producer = {o: n for n in g.node for o in n.output}
        for r in qnodes:
            if r.op_type != "Relu":
                continue
            src = r.input[0]
            p = producer.get(src)
            if (
                p is None
                or id(p) not in qnode_ids
                or len(consumers[src]) != 1
                or src in graph_outputs
                or r.output[0] not in ranges
            ):
                continue
            for k, o in enumerate(p.output):
                if o == src:
                    p.output[k] = r.output[0]
            ranges[r.output[0]] = (0.0, max(ranges[r.output[0]][1], 0.0))
            removed.add(id(r))
        acts = [
            a
            for a in acts
            if not any(id(r) in removed and r.input[0] == a for r in consumers[a])
        ]

    elif fold_relu:
        # A centred zero point cannot clamp at zero, so the Relu stays; instead
        # the tensor between a quantized producer and its Relu is left float
        # (Quark's "remove Q/DQ between conv and relu").
        producer = {o: n for n in g.node for o in n.output}
        skip = set()
        for r in qnodes:
            src = r.input[0] if r.op_type == "Relu" else None
            p = producer.get(src) if src else None
            if (
                p is not None
                and id(p) in qnode_ids
                and len(consumers[src]) == 1
                and src not in graph_outputs
            ):
                skip.add(src)
        acts = [a for a in acts if a not in skip]

    # Quantization parameters, propagating through data-movement ops in topological order.
    tensor_dtypes = dict(tensor_dtypes or {})
    for t in set(tensor_dtypes.values()) - set(_DTYPES):
        raise ValueError(f"unsupported dtype in tensor_dtypes: {t!r}")
    qp: Dict[str, Tuple[float, int]] = {}
    qdt: Dict[str, str] = {}

    def set_qp(x: str) -> None:
        dt = tensor_dtypes.get(x, activation_dtype)
        qp[x] = _qparams(*ranges[x], *_DTYPES[dt][2:], sym, p2)
        qdt[x] = dt

    for x in graph_inputs:
        if x in seen and x in ranges:
            set_qp(x)
    for n in g.node:
        if id(n) in removed:
            continue
        if n.op_type in _SHARED_QPARAM_OPS and id(n) in qnode_ids and n.input[0] in qp:
            for o in n.output:
                if o and o in floats and o not in tensor_dtypes:
                    qp[o], qdt[o] = qp[n.input[0]], qdt[n.input[0]]
        for x in list(n.input) + list(n.output):
            if x in seen and x not in qp and x in ranges and x not in inits:
                set_qp(x)

    opset = next((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), 0)
    if opset < 13:
        raise ValueError(
            "full-graph QDQ needs opset >= 13 (per-channel DequantizeLinear)"
        )

    def domain_of(dt: str) -> str:
        return "com.microsoft" if dt in _WIDE_DTYPES and opset < 21 else ""

    qdq_domain = domain_of(activation_dtype)
    if any(
        domain_of(d) for d in list(qdt.values()) + [activation_dtype, weight_dtype]
    ) and not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    new_inits: List[TensorProto] = []
    uid = [0]

    def fresh(base: str) -> str:
        uid[0] += 1
        return f"{base}/qdq{uid[0]}"

    def add_init(name: str, arr: np.ndarray) -> str:
        new_inits.append(numpy_helper.from_array(arr, name))
        return name

    # Activation Q -> DQ pairs. The producer's output is renamed and the DQ takes over the
    # original name, so every consumer (and a graph output) reads the dequantized value.
    act_nodes: List[onnx.NodeProto] = []
    rename: Dict[str, str] = {}
    for a in acts:
        if a not in qp:
            continue
        s, zp = qp[a]
        dom = domain_of(qdt[a])
        sn = add_init(fresh(a) + "/scale", np.array(s, np.float32))
        zn = add_init(fresh(a) + "/zp", np.array(zp, _DTYPES[qdt[a]][1]))
        q_out = a + "/q"
        if a in graph_inputs:
            dq_out = a + "/dq"
            for c in consumers[a]:
                for k, x in enumerate(c.input):
                    if x == a:
                        c.input[k] = dq_out
            act_nodes += [
                helper.make_node(
                    "QuantizeLinear",
                    [a, sn, zn],
                    [q_out],
                    name=a + "/Q",
                    domain=dom,
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [q_out, sn, zn],
                    [dq_out],
                    name=a + "/DQ",
                    domain=dom,
                ),
            ]
        else:
            pre = a + "/f"
            rename[a] = pre
            act_nodes += [
                helper.make_node(
                    "QuantizeLinear",
                    [pre, sn, zn],
                    [q_out],
                    name=a + "/Q",
                    domain=dom,
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [q_out, sn, zn],
                    [a],
                    name=a + "/DQ",
                    domain=dom,
                ),
            ]

    # Mixed 8/16-bit activations: a quantized node computes in its output's dtype, so an input
    # of the other dtype is re-quantized for it (DQ -> Q' -> DQ', a "convert" the QNN EP maps
    # to its Convert op). GridSample's grid is exempt: its coordinates are the reason to mix.
    converted: Dict[Tuple[str, str], str] = {}
    act_extra: List[str] = []
    for n in qnodes if convert_inputs else ():
        if id(n) in removed:
            continue
        outs = [o for o in n.output if o in qdt]
        if not outs:
            continue
        node_dt = qdt[outs[0]]
        for k, x in enumerate(n.input):
            src = (
                x[: -len("/dq")]
                if x.endswith("/dq") and x[: -len("/dq")] in graph_inputs
                else x
            )
            if (
                src not in qdt
                or qdt[src] == node_dt
                or (n.op_type == "GridSample" and k == 1)
            ):
                continue
            ckey = (src, node_dt)
            if ckey not in converted:
                c = f"{src}/as_{node_dt}"
                sc, zc = _qparams(*ranges[src], *_DTYPES[node_dt][2:], sym, p2)
                qp[c], qdt[c] = (sc, zc), node_dt
                sn = add_init(fresh(c) + "/scale", np.array(sc, np.float32))
                zn = add_init(fresh(c) + "/zp", np.array(zc, _DTYPES[node_dt][1]))
                dom = domain_of(node_dt)
                act_nodes += [
                    helper.make_node(
                        "QuantizeLinear",
                        [x, sn, zn],
                        [c + "/q"],
                        name=c + "/Q",
                        domain=dom,
                    ),
                    helper.make_node(
                        "DequantizeLinear",
                        [c + "/q", sn, zn],
                        [c],
                        name=c + "/DQ",
                        domain=dom,
                    ),
                ]
                converted[ckey] = c
                act_extra.append(c)
            n.input[k] = converted[ckey]

    def act_scale(x: str) -> Optional[float]:
        if x not in qp and x.endswith("/dq"):
            x = x[: -len("/dq")]  # a graph input, rewired to its DQ above
        return qp[x][0] if x in qp else None

    # Constant inputs of quantized nodes that form a real QDQ unit (every float activation
    # input dequantized): a lone DQ on a weight of a float node would strand it on the CPU.
    cache: Dict[Tuple, str] = {}
    act_set = set(acts) | set(act_extra)
    for n in qnodes:
        if id(n) in removed:
            continue
        if not all(
            x in act_set and x in qp
            for x in _data_inputs(n)
            if x in floats and x not in inits
        ):
            continue
        data = set(_data_inputs(n))
        for k, x in enumerate(list(n.input)):
            if (
                x not in inits
                or x not in data
                or inits[x].data_type != TensorProto.FLOAT
            ):
                continue
            w = numpy_helper.to_array(inits[x]).astype(np.float32)
            if n.op_type in _WEIGHT_AXIS_OPS and k == 1:
                axis = _weight_axis(n, w.ndim) if per_channel else None
                key = ("w", x, axis)
                if key not in cache:
                    w_np = _DTYPES[weight_dtype][1]
                    w_max = _DTYPES[weight_dtype][3]
                    if axis is None:
                        s = np.array(max(np.abs(w).max(), 1e-12) / w_max, np.float32)
                        if p2:
                            s = np.array(_pof2(float(s)), np.float32)
                        q = np.clip(np.round(w / s), -w_max, w_max).astype(w_np)
                        zp = np.array(0, w_np)
                    else:
                        red = tuple(i for i in range(w.ndim) if i != axis)
                        s = (np.maximum(np.abs(w).max(axis=red), 1e-12) / w_max).astype(
                            np.float32
                        )
                        if p2:
                            s = (2.0 ** np.ceil(np.log2(s))).astype(np.float32)
                        shape = [1] * w.ndim
                        shape[axis] = -1
                        q = np.clip(
                            np.round(w / s.reshape(shape)), -w_max, w_max
                        ).astype(w_np)
                        zp = np.zeros(s.shape, w_np)
                    base = fresh(x)
                    add_init(base + f"/{weight_dtype}", q)
                    add_init(base + "/scale", s)
                    add_init(base + "/zp", zp)
                    out = base + "/dq"
                    attrs = {"axis": axis} if axis is not None else {}
                    act_nodes.append(
                        helper.make_node(
                            "DequantizeLinear",
                            [base + f"/{weight_dtype}", base + "/scale", base + "/zp"],
                            [out],
                            name=out,
                            domain=domain_of(weight_dtype),
                            **attrs,
                        )
                    )
                    cache[key] = out
                n.input[k] = cache[key]
            elif (
                n.op_type in ("Conv", "ConvTranspose", "Gemm")
                and k == 2
                and w.ndim == 1
            ):
                w_dq = n.input[1]
                sx = act_scale(n.input[0])
                w_scale_name = (
                    w_dq[: -len("/dq")] + "/scale" if w_dq.endswith("/dq") else None
                )
                ws = next(
                    (
                        numpy_helper.to_array(t)
                        for t in new_inits
                        if t.name == w_scale_name
                    ),
                    None,
                )
                if sx is None or ws is None:
                    continue  # weight or input not quantized: keep a float bias
                s = (sx * np.broadcast_to(ws, w.shape)).astype(np.float32)
                s = np.maximum(s, 1e-30)
                q = np.clip(np.round(w / s), -(2**31) + 1, 2**31 - 1).astype(np.int32)
                base = fresh(x)
                add_init(base + "/int32", q)
                add_init(base + "/scale", s)
                add_init(base + "/zp", np.zeros(s.shape, np.int32))
                out = base + "/dq"
                act_nodes.append(
                    helper.make_node(
                        "DequantizeLinear",
                        [base + "/int32", base + "/scale", base + "/zp"],
                        [out],
                        name=out,
                        axis=0,
                    )
                )
                n.input[k] = out
            else:
                key = ("c", x, None)
                if key not in cache:
                    s, zp = _qparams(w.min(), w.max(), qmin, qmax, sym, p2)
                    q = np.clip(np.round(w / s) + zp, qmin, qmax).astype(act_np)
                    base = fresh(x)
                    add_init(base + "/q", q)
                    add_init(base + "/scale", np.array(s, np.float32))
                    add_init(base + "/zp", np.array(zp, act_np))
                    out = base + "/dq"
                    act_nodes.append(
                        helper.make_node(
                            "DequantizeLinear",
                            [base + "/q", base + "/scale", base + "/zp"],
                            [out],
                            name=out,
                            domain=qdq_domain,
                        )
                    )
                    cache[key] = out
                n.input[k] = cache[key]

    for n in g.node:
        for k, o in enumerate(n.output):
            if o in rename:
                n.output[k] = rename[o]
    nodes = [n for n in g.node if id(n) not in removed]
    del g.node[:]
    # Q/DQ first is fine topologically for constants; activation pairs must follow their
    # producer, so sort once at the end.
    g.node.extend(nodes + act_nodes)
    g.initializer.extend(new_inits)
    used = {x for n in g.node for x in n.input} | graph_outputs
    kept = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(kept)
    del g.value_info[:]
    _toposort(g)
    return m


def _toposort(g: onnx.GraphProto) -> None:
    avail = {i.name for i in g.input} | {i.name for i in g.initializer} | {""}
    pending = list(g.node)
    order = []
    while pending:
        rest = []
        for n in pending:
            if all(x in avail for x in n.input):
                order.append(n)
                avail.update(n.output)
            else:
                rest.append(n)
        if len(rest) == len(pending):
            raise ValueError(
                f"graph has a cycle or dangling input at {rest[0].name}: {list(rest[0].input)}"
            )
        pending = rest
    del g.node[:]
    g.node.extend(order)


def quantized_io(
    model: onnx.ModelProto,
    inputs: Optional[Iterable[str]] = None,
    outputs: Optional[Iterable[str]] = None,
    nhwc_inputs: Iterable[str] = (),
) -> Tuple[onnx.ModelProto, Dict[str, Dict[str, Union[float, str]]]]:
    """
    Make a :func:`quantize_full_qdq` model take/return quantized tensors.

    A float graph input ``x`` whose only consumer is its ``QuantizeLinear``
    becomes the integer tensor ``x`` itself (the host quantizes it:
    ``round(x / scale) + zero_point``); a graph output ``y`` produced by a
    ``DequantizeLinear`` becomes the integer tensor before it (the host
    dequantizes, or feeds it to the next quantized model as-is). Both are
    lossless: the graph quantized/dequantized there anyway. ``nhwc_inputs``
    additionally take the input channels-last (``[N, H, W, C]``), transposed
    in the graph before the DequantizeLinear.

    :param inputs: input names to convert (default: every quantized input)
    :param outputs: output names to convert (default: every quantized output)
    :returns: ``(model, {tensor: {"scale": s, "zero_point": zp, "dtype": ...}})``
            (``"layout": "nhwc"`` added for ``nhwc_inputs``)
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    consumers = defaultdict(list)
    for n in g.node:
        for x in n.input:
            consumers[x].append(n)
    producer = {o: n for n in g.node for o in n.output}
    nhwc_inputs = set(nhwc_inputs)
    info: Dict[str, Dict[str, Union[float, str]]] = {}
    remove = set()
    extra_nodes = []
    for vi in g.input:
        if inputs is not None and vi.name not in inputs:
            continue
        cs = consumers[vi.name]
        if len(cs) != 1 or cs[0].op_type != "QuantizeLinear":
            continue
        q = cs[0]
        zp = inits[q.input[2]]
        info[vi.name] = {
            "scale": float(inits[q.input[1]]),
            "zero_point": int(zp),
            "dtype": str(zp.dtype),
        }
        remove.add(id(q))
        elem = helper.np_dtype_to_tensor_dtype(zp.dtype)
        dims = [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in vi.type.tensor_type.shape.dim
        ]
        if vi.name in nhwc_inputs:
            assert len(dims) == 4, vi.name
            info[vi.name]["layout"] = "nhwc"
            dims = [dims[0], dims[2], dims[3], dims[1]]
            extra_nodes.append(
                helper.make_node(
                    "Transpose",
                    [vi.name],
                    [q.output[0]],
                    name=vi.name + "/to_nchw",
                    perm=[0, 3, 1, 2],
                )
            )
        else:
            for c in consumers[q.output[0]]:
                for k, x in enumerate(c.input):
                    if x == q.output[0]:
                        c.input[k] = vi.name
        vi.type.CopyFrom(helper.make_tensor_type_proto(elem, dims))
    for vi in g.output:
        if outputs is not None and vi.name not in outputs:
            continue
        dq = producer.get(vi.name)
        if dq is None or dq.op_type != "DequantizeLinear" or consumers[vi.name]:
            continue
        zp = inits[dq.input[2]]
        if zp.ndim != 0:
            continue
        info[vi.name] = {
            "scale": float(inits[dq.input[1]]),
            "zero_point": int(zp),
            "dtype": str(zp.dtype),
        }
        remove.add(id(dq))
        # The Q's output takes the graph output's name.
        q_out = dq.input[0]
        for n in g.node:
            for k, o in enumerate(n.output):
                if o == q_out:
                    n.output[k] = vi.name
            for k, x in enumerate(n.input):
                if x == q_out and id(n) != id(dq):
                    n.input[k] = vi.name
        dims = [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in vi.type.tensor_type.shape.dim
        ]
        vi.type.CopyFrom(
            helper.make_tensor_type_proto(
                helper.np_dtype_to_tensor_dtype(zp.dtype), dims
            )
        )
    nodes = [n for n in g.node if id(n) not in remove] + extra_nodes
    del g.node[:]
    g.node.extend(nodes)
    _toposort(g)
    return m, info


def sampling_coordinate_tensors(
    model: onnx.ModelProto, stop_op_types: Iterable[str] = ("Gemm", "MatMul", "Conv")
) -> List[str]:
    """
    The tensors that compute ``GridSample``'s sampling grid: a backward slice
    from every GridSample's grid input through elementwise/data-movement ops,
    stopping at (and including) the outputs of ``stop_op_types`` (the offset
    projection) and graph inputs (host-computed reference points).

    Sampling coordinates need far more resolution than 8 bits over their
    range (a uint8 step of a [-1, 1] grid is ~0.4% of the feature map, i.e.
    most of a pixel on a 25-wide map), so these are the natural
    ``tensor_dtypes={t: "uint16" ...}`` of :func:`quantize_full_qdq` in a
    deformable-attention model.
    """
    producer = {o: n for n in model.graph.node for o in n.output}
    inits = {i.name for i in model.graph.initializer}
    stop = set(stop_op_types)
    out: List[str] = []
    todo = [n.input[1] for n in model.graph.node if n.op_type == "GridSample"]
    while todo:
        t = todo.pop()
        if t in out or t in inits or not t:
            continue
        out.append(t)
        p = producer.get(t)
        if p is None or p.op_type in stop or p.op_type in _NEVER_QUANTIZED:
            continue
        todo.extend(_data_inputs(p))
    return out
