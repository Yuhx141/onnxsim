"""Static INT8 Conv lowering for the XDNA ResNet codegen path.

The first implementation lowers NCHW convolution to one or more GEMMs over
an im2col view.  It is deliberately a planning layer: the eventual emitter
can choose a direct-convolution kernel when that becomes faster, while this
contract keeps shapes, padding, groups, and fused activation explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    from .xdna_backend import select_matmul_tile
except ImportError:  # direct script-directory imports used by lightweight tests
    from xdna_backend import select_matmul_tile


@dataclass(frozen=True)
class ConvGemmPlan:
    node_index: int
    node_name: str
    input_shape: Tuple[int, int, int, int]
    weight_shape: Tuple[int, int, int, int]
    output_shape: Tuple[int, int, int, int]
    stride: Tuple[int, int]
    dilation: Tuple[int, int]
    pads: Tuple[int, int, int, int]
    groups: int
    gemm_shape: Tuple[int, int, int]
    group_count: int
    has_bias: bool
    fused_relu: bool
    tile: Tuple[int, int, int]
    kernel: str

    @property
    def output_elements(self) -> int:
        return self.output_shape[0] * self.output_shape[1] * self.output_shape[2] * self.output_shape[3]


def _attrs(node: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for attr in getattr(node, "attribute", ()):
        value = getattr(attr, "ints", None)
        if value:
            converted = tuple(int(v) for v in value)
            # Some lightweight graph shims represent scalar integer attrs
            # (notably Conv's `group`) through `ints=(value,)`.
            result[str(attr.name)] = (
                converted[0]
                if str(attr.name) in {"group"} and len(converted) == 1
                else converted
            )
            continue
        if hasattr(attr, "i"):
            result[str(attr.name)] = int(attr.i)
        elif hasattr(attr, "s") and attr.s:
            result[str(attr.name)] = attr.s.decode() if isinstance(attr.s, bytes) else str(attr.s)
    return result


def _shape(value: Any) -> Optional[Tuple[int, ...]]:
    tensor_type = getattr(getattr(value, "type", None), "tensor_type", None)
    shape = getattr(tensor_type, "shape", None)
    dims = getattr(shape, "dim", None)
    if not dims:
        return None
    result = []
    for axis, dim in enumerate(dims):
        value = int(getattr(dim, "dim_value", 0))
        if value < 0:
            return None
        # The Vitis quicktest graph uses dim_value=0 for its dynamic batch.
        # The current XDNA lowering contract is batch-1, so specialize that
        # dimension while retaining strict static checks for spatial axes.
        if value == 0:
            if axis != 0:
                return None
            value = 1
        result.append(value)
    return tuple(result)


def _value_shapes(model: Any) -> Dict[str, Tuple[int, ...]]:
    graph = model.graph
    result: Dict[str, Tuple[int, ...]] = {}
    for value in (*getattr(graph, "input", ()), *getattr(graph, "value_info", ()), *getattr(graph, "output", ())):
        shape = _shape(value)
        if shape is not None:
            result[str(value.name)] = shape
    # Quantized exports commonly omit value_info for every QDQ intermediate.
    # Seed initializer shapes as well, then propagate the shape-preserving QDQ
    # edges and infer Conv outputs in graph order.
    for value in getattr(graph, "initializer", ()):
        result.setdefault(str(value.name), tuple(int(v) for v in value.dims))
    nodes = list(getattr(graph, "node", ()))
    for _ in range(len(nodes) + 1):
        changed = False
        for node in nodes:
            op = str(node.op_type)
            inputs = [str(value) for value in getattr(node, "input", ())]
            outputs = [str(value) for value in getattr(node, "output", ())]
            if not inputs or not outputs:
                continue
            if op in {"QuantizeLinear", "DequantizeLinear"}:
                shape = result.get(inputs[0])
                if shape is not None:
                    for output in outputs:
                        if result.get(output) != shape:
                            result[output] = shape
                            changed = True
            elif op == "Conv" and len(inputs) >= 2:
                input_shape = result.get(inputs[0])
                weight_shape = result.get(inputs[1])
                if input_shape is None or weight_shape is None or len(input_shape) != 4 or len(weight_shape) != 4:
                    continue
                attrs = _attrs(node)
                stride = tuple(attrs.get("strides", (1, 1)))
                dilation = tuple(attrs.get("dilations", (1, 1)))
                pads = tuple(attrs.get("pads", (0, 0, 0, 0)))
                groups = attrs.get("group", 1)
                groups = int(groups[0] if isinstance(groups, (tuple, list)) else groups)
                if attrs.get("auto_pad", "NOTSET") == "VALID":
                    pads = (0, 0, 0, 0)
                if len(pads) != 4 or len(stride) != 2 or len(dilation) != 2 or groups <= 0:
                    continue
                out_hw = []
                for axis in range(2):
                    kernel = dilation[axis] * (weight_shape[2 + axis] - 1) + 1
                    numerator = input_shape[2 + axis] + pads[axis] + pads[axis + 2] - kernel
                    out_hw.append(numerator // stride[axis] + 1)
                shape = (input_shape[0], weight_shape[0], out_hw[0], out_hw[1])
                for output in outputs:
                    if result.get(output) != shape:
                        result[output] = shape
                        changed = True
        if not changed:
            break
    return result


def _initializer_shapes(model: Any) -> Dict[str, Tuple[int, ...]]:
    return {str(value.name): tuple(int(v) for v in value.dims) for value in getattr(model.graph, "initializer", ())}


def _next_semantic(index: int, nodes: Sequence[Any], consumers: Mapping[str, Sequence[int]]) -> Tuple[int, ...]:
    frontier = list(getattr(nodes[index], "output", ()))
    seen = set()
    result = []
    while frontier:
        value = str(frontier.pop())
        if value in seen:
            continue
        seen.add(value)
        for consumer_index in consumers.get(value, ()):
            consumer = nodes[consumer_index]
            if str(consumer.op_type) in {"QuantizeLinear", "DequantizeLinear"}:
                frontier.extend(getattr(consumer, "output", ()))
            else:
                result.append(consumer_index)
    return tuple(dict.fromkeys(result))


def plan_conv_gemm(
    model: Any,
    node_index: int,
    *,
    dtype: str = "i8",
    columns: int = 8,
    profile: Optional[Mapping[str, Any]] = None,
    optimize_small_m: bool = False,
) -> ConvGemmPlan:
    """Lower one static NCHW Conv node to grouped im2col GEMM metadata."""
    nodes = list(model.graph.node)
    node = nodes[node_index]
    if str(node.op_type) != "Conv":
        raise ValueError("node_index must refer to an ONNX Conv node")
    shapes = _value_shapes(model)
    inputs = [str(value) for value in getattr(node, "input", ())]
    outputs = [str(value) for value in getattr(node, "output", ())]
    input_shape = shapes.get(inputs[0])
    weight_shape = shapes.get(inputs[1])
    output_shape = shapes.get(outputs[0])
    if input_shape is None or weight_shape is None or output_shape is None:
        raise ValueError(f"Conv {node_index} requires static input, weight, and output shapes")
    if len(input_shape) != 4 or len(weight_shape) != 4 or len(output_shape) != 4:
        raise ValueError("XDNA Conv lowering currently requires rank-4 NCHW tensors")
    attrs = _attrs(node)
    stride = tuple(attrs.get("strides", (1, 1)))
    dilation = tuple(attrs.get("dilations", (1, 1)))
    groups = int(attrs.get("group", 1))
    auto_pad = attrs.get("auto_pad", "NOTSET")
    if auto_pad not in ("NOTSET", "VALID"):
        raise ValueError(f"Conv auto_pad {auto_pad!r} needs explicit padding before lowering")
    if "pads" in attrs:
        pads = tuple(attrs["pads"])
        if len(pads) != 4:
            raise ValueError("Conv pads must contain four values")
    else:
        pads = (0, 0, 0, 0)
    if groups <= 0 or weight_shape[0] % groups != 0 or input_shape[1] % groups != 0:
        raise ValueError("Conv groups are incompatible with channel dimensions")
    if weight_shape[1] * groups != input_shape[1]:
        raise ValueError("Conv weight channels do not match grouped input channels")
    m = input_shape[0] * output_shape[2] * output_shape[3]
    k = weight_shape[1] * weight_shape[2] * weight_shape[3]
    n = weight_shape[0] // groups
    tile = select_matmul_tile(dtype, m, k, n, columns, profile)
    # The AIE2P vectorized int8 MAC kernel has a 16-row minimum. A matching tile cuts the
    # padded M dimension sharply for small ResNet feature maps.
    if optimize_small_m and m <= 64:
        tile = (16, tile[1], tile[2])
    kernel = "conv_im2col_gemm_int8"

    # A fused Relu is legal only when the Conv's semantic output has exactly
    # one consumer and that consumer is Relu (QDQ edges are transparent).
    consumers: Dict[str, list[int]] = {}
    for i, candidate in enumerate(nodes):
        for value in getattr(candidate, "input", ()):
            consumers.setdefault(str(value), []).append(i)
    following = _next_semantic(node_index, nodes, consumers)
    fused_relu = len(following) == 1 and str(nodes[following[0]].op_type) == "Relu"
    if fused_relu:
        kernel += "_relu"
    return ConvGemmPlan(
        node_index=node_index,
        node_name=str(getattr(node, "name", "")) or f"Conv_{node_index}",
        input_shape=tuple(input_shape),
        weight_shape=tuple(weight_shape),
        output_shape=tuple(output_shape),
        stride=tuple(stride),
        dilation=tuple(dilation),
        pads=tuple(pads),
        groups=groups,
        gemm_shape=(m, k, n),
        group_count=groups,
        has_bias=len(inputs) >= 3 and bool(inputs[2]),
        fused_relu=fused_relu,
        tile=tile,
        kernel=kernel,
    )


def plan_all_convs(model: Any, **kwargs: Any) -> Tuple[ConvGemmPlan, ...]:
    """Return lowering metadata for every static Conv in graph order."""
    return tuple(
        plan_conv_gemm(model, index, **kwargs)
        for index, node in enumerate(model.graph.node)
        if str(node.op_type) == "Conv"
    )
