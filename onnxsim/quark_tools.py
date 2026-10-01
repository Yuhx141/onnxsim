"""Post-quantization graph utilities modelled on the scripts in
``quark.onnx.tools``. Independent implementations: Quark's source was read for
the names and intent only, and ``tests/test_quark_parity.py`` re-checks them
against the installed ``amd-quark`` in CI.

Correspondence with Quark (verified on the parity models):

- :func:`remove_qdq` ~ Quark's ``convert_quant_to_float`` (strip every Q/DQ and
  fold quantized weights back to float; identical outputs). Quark's own
  ``remove_qdq()`` is a different, narrower transform pipeline -- on the models
  probed it returned the model unchanged -- so it is *not* what this does;
  :func:`convert_quant_to_float` is the same function under the matching name.
- :func:`convert_s8s8_to_u8s8` is a superset of Quark's, which only converts
  activation zero points equal to 0 (to uint8 128); this handles any int8 zero
  point (``zp + 128``) and gives the same result for zero.
- :func:`convert_opset_version`, :func:`convert_shared_initializer_to_unique`,
  :func:`convert_dynamic_to_fixed`, :func:`replace_inf_weights`,
  :func:`convert_fp32_to_fp16` / ``bf16`` and :func:`convert_fp16_to_fp32` /
  ``bf16_to_fp32`` follow the Quark tool of the same name.

The remaining tools of ``quark.onnx.tools`` (and the graph helpers of
``quark.onnx.utils.model_utils``) live in :mod:`onnxsim.quark_tools_extra` and
are re-exported here; that module's docstrings note each deliberate difference:

- ``convert_a8w8_npu_to_a8w8_cpu``, ``convert_bias_int32_to_int16`` (returns
  ``(model, changed)`` like Quark), ``convert_custom_ops``,
  ``convert_customqdq_to_qdq``, ``convert_nchw_to_nhwc``, ``convert_qdq_to_qop``,
  ``convert_resize_fs_to_pof2s``, ``convert_u16s8_to_s16s8``,
  ``convert_u16u8_to_u8u8``, ``convert_fp16_to_bf16`` (Quark's ``bf16`` format),
  ``fix_shapes`` (+ ``fix_input_and_output_shapes``), ``insert_clip_bfloat16_qdq``,
  ``remove_bf16_cast``, ``remove_qdq_between_ops``, ``remove_qdq_mul_add``,
  ``replace_bfloat16_qdq_cast``, ``convert_onnx_to_onnxtxt`` /
  ``convert_onnxtxt_to_onnx`` (model <-> text; Quark's are CLI-only) and
  ``a16w8_a8w8_nodes`` (Quark's ``print_a16w8_a8w8_nodes`` takes a path).
- From ``model_utils``: ``copy_shared_nodes``, ``check_shared_initializers``,
  ``clean_initializer_in_input``, ``save_onnx_model_with_external_data``.
- Not implemented: ``convert_lstm_to_customlstm`` (needs Quark's ``ExtendedLSTM``
  custom op), ``convert_fp16_to_bfp16`` / ``convert_fp32_to_bfp16`` and
  ``random_quantize`` (thin drivers around Quark's whole quantizer; onnxsim's
  BFP16 / MX fake-quantizers are in ``quark_compat``), ``evaluate`` and
  ``save_tensor_hist`` / ``save_weights_hist`` (image metrics / matplotlib
  reports, not graph edits).

Every function takes and returns an ``onnx.ModelProto`` (the input is not
modified) and only rewrites the **top-level graph** -- nodes inside
control-flow subgraphs are left alone.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import onnx
from onnx import numpy_helper

_Q_OPS = {"QuantizeLinear"}
_DQ_OPS = {"DequantizeLinear"}
_QDQ_DOMAINS = ("", "ai.onnx", "com.microsoft")


def _copy(model: onnx.ModelProto) -> onnx.ModelProto:
    out = onnx.ModelProto()
    out.CopyFrom(model)
    return out


def _is(node: onnx.NodeProto, ops: set) -> bool:
    return node.op_type in ops and node.domain in _QDQ_DOMAINS


def _dequantize(
    q: np.ndarray, scale: np.ndarray, zp: Optional[np.ndarray], axis: int
) -> np.ndarray:
    x = q.astype(np.float32)
    z = np.zeros((), np.float32) if zp is None else zp.astype(np.float32)
    s = scale.astype(np.float32)
    if s.ndim == 1 and s.size > 1:
        shape = [1] * x.ndim
        shape[axis % x.ndim] = -1
        s = s.reshape(shape)
        if z.ndim == 1 and z.size > 1:
            z = z.reshape(shape)
    return ((x - z) * s).astype(np.float32)


def remove_qdq(model: onnx.ModelProto, fold_weights: bool = True) -> onnx.ModelProto:
    """Strip quantization from a QDQ model, returning a float32 graph.

    - ``QuantizeLinear -> DequantizeLinear`` pairs are removed and their
      consumers wired to the pair's float input (a pair whose DQ output is a
      graph output keeps that name through an ``Identity``).
    - With ``fold_weights``, a ``DequantizeLinear`` over constant
      initializers is evaluated and replaced by a float32 initializer of the
      same name as its output; the now-unused quantized tensor, scale and
      zero-point initializers are dropped.

    A Q whose output feeds anything other than ``DequantizeLinear``, and
    blocked (``block_size``) dequantization, are left untouched.
    """
    m = _copy(model)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    graph_outputs = {o.name for o in g.output}
    removed_inputs: List[str] = []
    drop = set()  # ids of nodes to remove
    replace_with: Dict[str, str] = {}  # DQ output name -> float source name
    identities: Dict[int, onnx.NodeProto] = {}  # id(DQ node) -> Identity to emit
    new_inits: List[onnx.TensorProto] = []

    consumers = defaultdict(list)
    for n in g.node:
        for x in n.input:
            consumers[x].append(n)

    if fold_weights:
        for n in g.node:
            if not _is(n, _DQ_OPS) or n.output[0] in graph_outputs:
                continue
            ins = list(n.input) + [""] * (3 - len(n.input))
            x, s, z = ins[:3]
            if x not in inits or s not in inits or (z and z not in inits):
                continue
            if any(a.name == "block_size" and a.i > 0 for a in n.attribute):
                continue
            axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
            w = _dequantize(
                numpy_helper.to_array(inits[x]),
                numpy_helper.to_array(inits[s]),
                numpy_helper.to_array(inits[z]) if z else None,
                axis,
            )
            new_inits.append(numpy_helper.from_array(w, n.output[0]))
            drop.add(id(n))
            removed_inputs += [i for i in (x, s, z) if i]

    for q in g.node:
        if not _is(q, _Q_OPS) or q.output[0] in graph_outputs:
            continue
        users = consumers[q.output[0]]
        if not users or not all(_is(u, _DQ_OPS) and id(u) not in drop for u in users):
            continue
        drop.add(id(q))
        removed_inputs += [i for i in q.input[1:] if i]
        for dq in users:
            drop.add(id(dq))
            removed_inputs += [i for i in dq.input[1:] if i]
            if dq.output[0] in graph_outputs:
                identities[id(dq)] = onnx.helper.make_node(
                    "Identity", [q.input[0]], [dq.output[0]]
                )
            else:
                replace_with[dq.output[0]] = q.input[0]

    def resolve(name: str) -> str:
        while name in replace_with:
            name = replace_with[name]
        return name

    kept: List[onnx.NodeProto] = []
    for n in g.node:
        if id(n) in identities:
            kept.append(identities[id(n)])
        elif id(n) not in drop:
            for i, x in enumerate(n.input):
                if x in replace_with:
                    n.input[i] = resolve(x)
            kept.append(n)
    del g.node[:]
    g.node.extend(kept)
    g.initializer.extend(new_inits)

    referenced = {x for n in g.node for x in n.input} | graph_outputs
    unused = {i for i in removed_inputs if i not in referenced}
    keep_inits = [i for i in g.initializer if i.name not in unused]
    del g.initializer[:]
    g.initializer.extend(keep_inits)
    return m


def convert_quant_to_float(
    model: onnx.ModelProto, fold_weights: bool = True
) -> onnx.ModelProto:
    """Quark's name for :func:`remove_qdq`: strip the quantization from a QDQ
    model, returning the float32 graph."""
    return remove_qdq(model, fold_weights=fold_weights)


def convert_shared_initializer_to_unique(model: onnx.ModelProto) -> onnx.ModelProto:
    """Give each node its own copy of an initializer that several nodes use
    (the first consumer keeps the original name; the others get
    ``<name>_copy<k>``)."""
    m = _copy(model)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    seen: Dict[str, int] = defaultdict(int)
    for n in g.node:
        renamed: Dict[str, str] = {}
        for i, x in enumerate(n.input):
            if x not in inits:
                continue
            if x in renamed:  # same tensor twice in one node: share one copy
                n.input[i] = renamed[x]
                continue
            k = seen[x]
            seen[x] += 1
            if k == 0:
                renamed[x] = x
                continue
            new_name = f"{x}_copy{k}"
            dup = onnx.TensorProto()
            dup.CopyFrom(inits[x])
            dup.name = new_name
            g.initializer.append(dup)
            renamed[x] = new_name
            n.input[i] = new_name
    return m


def convert_dynamic_to_fixed(
    model: onnx.ModelProto, input_shapes: Dict[str, Sequence[int]]
) -> onnx.ModelProto:
    """Fix graph input shapes (``{name: [dims]}``), drop stale intermediate
    and output shape info, and re-run shape inference so static shapes
    propagate."""
    m = _copy(model)
    g = m.graph
    by_name = {i.name: i for i in g.input}
    for name, dims in input_shapes.items():
        if name not in by_name:
            raise ValueError(f"{name!r} is not a graph input")
        shape = by_name[name].type.tensor_type.shape
        if len(shape.dim) != len(dims):
            raise ValueError(
                f"{name!r} has rank {len(shape.dim)}, got {len(dims)} dims"
            )
        for d, v in zip(shape.dim, dims):
            d.ClearField("dim_param")
            d.dim_value = int(v)
    del g.value_info[:]
    for o in g.output:
        for d in o.type.tensor_type.shape.dim:
            d.Clear()
    return onnx.shape_inference.infer_shapes(m)


def replace_inf_weights(
    model: onnx.ModelProto, max_value: float = 3.0e38
) -> onnx.ModelProto:
    """Clamp +-inf in float initializers to +-``max_value`` (NaN is kept)."""
    m = _copy(model)
    for t in m.graph.initializer:
        if t.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE):
            continue
        arr = numpy_helper.to_array(t)
        if np.isfinite(arr).all() or not np.isinf(arr).any():
            continue
        fixed = np.where(np.isposinf(arr), max_value, arr)
        fixed = np.where(np.isneginf(arr), -max_value, fixed).astype(arr.dtype)
        t.CopyFrom(numpy_helper.from_array(fixed, t.name))
    return m


def convert_s8s8_to_u8s8(model: onnx.ModelProto) -> onnx.ModelProto:
    """Re-express int8 *activation* Q/DQ pairs as uint8 (weights stay int8).

    Exact: ``(q - zp) * scale`` is unchanged by ``q' = q + 128``,
    ``zp' = zp + 128``, and the int8 / uint8 clamp ranges map onto each
    other. A ``QuantizeLinear`` is converted when its input is not an
    initializer and its zero point is an int8 initializer; the
    ``DequantizeLinear`` nodes consuming it follow (so Q and DQ agree). A
    zero-point initializer also used by a node that is *not* converted is
    copied rather than changed. Q nodes with no zero point, or a non-constant
    one, are left untouched.
    """
    m = _copy(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    graph_outputs = {o.name for o in g.output}
    consumers = defaultdict(list)
    for n in g.node:
        for x in n.input:
            consumers[x].append(n)

    def is_int8_const(name: str) -> bool:
        return (
            bool(name)
            and name in inits
            and inits[name].data_type == onnx.TensorProto.INT8
        )

    convert: Dict[int, onnx.NodeProto] = {}  # id(node) -> node, zp at input[2]
    for q in g.node:
        if not _is(q, _Q_OPS) or len(q.input) < 3 or q.input[0] in inits:
            continue
        if not is_int8_const(q.input[2]):
            continue
        users = consumers[q.output[0]]
        dqs = [u for u in users if _is(u, _DQ_OPS) and u.input[0] == q.output[0]]
        if not users or len(dqs) != len(users) or q.output[0] in graph_outputs:
            continue  # the int8 tensor is used as int8 elsewhere: its type is observable
        if any(len(d.input) < 3 or not is_int8_const(d.input[2]) for d in dqs):
            continue  # a DQ we cannot convert consistently: leave the whole chain
        convert[id(q)] = q
        for d in dqs:
            convert[id(d)] = d

    users_of_zp = defaultdict(list)
    for n in g.node:
        if len(n.input) >= 3 and n.input[2] in inits:
            users_of_zp[n.input[2]].append(n)

    shifted: Dict[str, str] = {}
    for n in g.node:
        if id(n) not in convert:
            continue
        zp = n.input[2]
        if zp not in shifted:
            arr = numpy_helper.to_array(inits[zp])
            new = (arr.astype(np.int16) + 128).astype(np.uint8)
            all_converted = all(id(u) in convert for u in users_of_zp[zp])
            name = zp if all_converted else f"{zp}_u8"
            tensor = numpy_helper.from_array(new, name)
            if all_converted:
                inits[zp].CopyFrom(tensor)
            else:
                g.initializer.append(tensor)
            shifted[zp] = name
        n.input[2] = shifted[zp]
        if _is(n, _Q_OPS):
            for a in n.attribute:
                if a.name == "output_dtype":
                    a.i = onnx.TensorProto.UINT8
    return m


_HALF_TYPES = (onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16)


def _convert_half_to_fp32(model: onnx.ModelProto, half: int) -> onnx.ModelProto:
    m = _copy(model)
    g = m.graph
    f32 = onnx.TensorProto.FLOAT

    def fix_tensor(t: onnx.TensorProto) -> None:
        if t.data_type == half:
            t.CopyFrom(
                numpy_helper.from_array(
                    numpy_helper.to_array(t).astype(np.float32), t.name
                )
            )

    def fix_value_info(vi: onnx.ValueInfoProto) -> None:
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == half:
            vi.type.tensor_type.elem_type = f32

    for t in g.initializer:
        fix_tensor(t)
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        fix_value_info(vi)
    for n in g.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                fix_tensor(a.t)
            elif a.name == "to" and a.i == half:
                a.i = f32  # Cast(..., to=half) -> Cast(..., to=float)

    # A Cast whose input is already float32 is now a no-op: drop it unless it
    # produces a graph output.
    inferred = onnx.shape_inference.infer_shapes(m)
    elem = {
        vi.name: vi.type.tensor_type.elem_type
        for vi in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    elem.update({t.name: t.data_type for t in g.initializer})
    outputs = {o.name for o in g.output}
    producer = {o: n for n in g.node for o in n.output}
    uses: Dict[str, int] = defaultdict(int)
    for n in g.node:
        for x in n.input:
            uses[x] += 1
    rename: Dict[str, str] = {}
    kept: List[onnx.NodeProto] = []
    for n in g.node:
        for i, x in enumerate(n.input):
            while x in rename:
                x = rename[x]
            n.input[i] = x
        noop_cast = (
            n.op_type == "Cast"
            and n.domain in ("", "ai.onnx")
            and any(a.name == "to" and a.i == f32 for a in n.attribute)
            and elem.get(n.input[0]) == f32
        )
        if noop_cast and n.output[0] not in outputs:
            rename[n.output[0]] = n.input[0]
            continue
        if noop_cast:
            # Produces a graph output: let the node feeding it produce that
            # name directly, if nothing else uses the intermediate tensor.
            src = n.input[0]
            prod = producer.get(src)
            if (
                prod is not None
                and uses[src] == 1
                and src not in outputs
                and any(prod is k for k in kept)
            ):
                prod.output[list(prod.output).index(src)] = n.output[0]
                continue
        kept.append(n)
    del g.node[:]
    g.node.extend(kept)
    return m


def convert_fp16_to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
    """float16 -> float32: initializers, ``Constant`` values, graph and
    intermediate types and ``Cast`` targets; casts that become float32 ->
    float32 are removed (so ``quantize_fp16(..., keep_io_types=True)``
    round-trips to the original structure)."""
    return _convert_half_to_fp32(model, onnx.TensorProto.FLOAT16)


def convert_bf16_to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
    """bfloat16 -> float32; see :func:`convert_fp16_to_fp32`."""
    return _convert_half_to_fp32(model, onnx.TensorProto.BFLOAT16)


def convert_fp32_to_fp16(
    model: onnx.ModelProto, keep_io_types: bool = True
) -> onnx.ModelProto:
    """float32 -> float16 (onnxsim's own :func:`onnxsim.quantize_fp16`)."""
    from onnxsim.onnx_simplifier import quantize_fp16

    return quantize_fp16(_copy(model), keep_io_types=keep_io_types)


def convert_fp32_to_bf16(
    model: onnx.ModelProto, keep_io_types: bool = True
) -> onnx.ModelProto:
    """float32 -> bfloat16 (onnxsim's own :func:`onnxsim.quantize_bf16`)."""
    from onnxsim.onnx_simplifier import quantize_bf16

    return quantize_bf16(_copy(model), keep_io_types=keep_io_types)


def convert_opset_version(model: onnx.ModelProto, target: int) -> onnx.ModelProto:
    """Convert the default-domain opset with ``onnx.version_converter``;
    ``ValueError`` if the converter cannot do it."""
    try:
        return onnx.version_converter.convert_version(_copy(model), target)
    except Exception as e:  # the converter raises assorted types
        raise ValueError(f"cannot convert to opset {target}: {e}") from e


def remove_initializer_from_input(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop graph inputs that merely mirror an initializer (old exporters
    list every weight as an input). Same behavior as onnxsim's internal pass:
    a model with IR version < 4 is bumped to IR 4 (unless its opset is too
    old for that to be safe, in which case it is returned unchanged)."""
    from onnxsim.onnx_simplifier import remove_initializer_from_input as _impl

    return _impl(_copy(model))


from onnxsim.quark_tools_extra import (  # noqa: E402
    CUSTOM_OP_NAME_MAPPING,
    a16w8_a8w8_nodes,
    check_shared_initializers,
    clean_initializer_in_input,
    convert_a8w8_npu_to_a8w8_cpu,
    convert_bias_int32_to_int16,
    convert_custom_ops,
    convert_customqdq_to_qdq,
    convert_fp16_to_bf16,
    convert_nchw_to_nhwc,
    convert_onnx_to_onnxtxt,
    convert_onnxtxt_to_onnx,
    convert_qdq_to_qop,
    convert_resize_fs_to_pof2s,
    convert_u16s8_to_s16s8,
    convert_u16u8_to_u8u8,
    copy_shared_nodes,
    fix_input_and_output_shapes,
    fix_shapes,
    insert_clip_bfloat16_qdq,
    parse_input_and_output_shapes,
    remove_bf16_cast,
    remove_qdq_between_ops,
    remove_qdq_mul_add,
    replace_bfloat16_qdq_cast,
    save_onnx_model_with_external_data,
)

__all__ = [
    "convert_bf16_to_fp32",
    "convert_dynamic_to_fixed",
    "convert_fp16_to_fp32",
    "convert_fp32_to_bf16",
    "convert_fp32_to_fp16",
    "convert_opset_version",
    "convert_quant_to_float",
    "convert_s8s8_to_u8s8",
    "convert_shared_initializer_to_unique",
    "remove_initializer_from_input",
    "remove_qdq",
    "replace_inf_weights",
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
