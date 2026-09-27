"""Dependency-light artifact specification for the XDNA ResNet schedule.

The emitter deliberately stops at an explicit artifact contract.  The same
contract can be consumed by the checked-in IRON whole-array GEMM example or
by a native Conv kernel once that kernel is available; no MIGraphX or ORT
dependency is introduced here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

try:
    from .resnet_codegen import ResNetCodegenPlan, codegen_plan_to_dict
except ImportError:  # direct script-directory imports used by tooling/tests
    from resnet_codegen import ResNetCodegenPlan, codegen_plan_to_dict
try:
    from .resnet_bottleneck import BottleneckBlockPlan, plan_bottleneck_blocks
except ImportError:  # direct script-directory imports
    from resnet_bottleneck import BottleneckBlockPlan, plan_bottleneck_blocks
try:
    from .graph_fusion import _attributes, _value_metadata
except ImportError:  # direct script-directory imports
    from graph_fusion import _attributes, _value_metadata
try:
    from .qdq_runtime import extract_qdq_edges
except ImportError:  # direct script-directory imports
    from qdq_runtime import extract_qdq_edges


@dataclass(frozen=True)
class KernelArtifactSpec:
    """One offline-buildable XDNA kernel artifact description."""

    key: str
    kernel_kind: str
    gemm_shape: Tuple[int, int, int]
    compiled_shape: Tuple[int, int, int]
    tile: Tuple[int, int, int]
    columns: int
    padding: Tuple[int, int, int]
    strategy: str
    buildable_with_whole_array: bool
    source: str
    entrypoint: str
    requires_im2col: bool
    fused_relu: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kernel_kind": self.kernel_kind,
            "gemm_shape": list(self.gemm_shape),
            "compiled_shape": list(self.compiled_shape),
            "tile": list(self.tile),
            "columns": self.columns,
            "padding": list(self.padding),
            "strategy": self.strategy,
            "buildable_with_whole_array": self.buildable_with_whole_array,
            "source": self.source,
            "entrypoint": self.entrypoint,
            "requires_im2col": self.requires_im2col,
            "fused_relu": self.fused_relu,
        }


@dataclass(frozen=True)
class BottleneckArtifactSpec:
    prefix: str
    input_shape: Tuple[int, int, int, int]
    has_downsample: bool
    source: str
    entrypoint: str = "bottleneck"

    @property
    def tensor_height(self) -> int:
        return self.input_shape[2]

    @property
    def tensor_width(self) -> int:
        return self.input_shape[3]

    @property
    def input_channels(self) -> int:
        return self.input_shape[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix": self.prefix,
            "input_shape": list(self.input_shape),
            "tensor_height": self.tensor_height,
            "tensor_width": self.tensor_width,
            "input_channels": self.input_channels,
            "has_downsample": self.has_downsample,
            "source": self.source,
            "entrypoint": self.entrypoint,
        }


@dataclass(frozen=True)
class OperationArtifactSpec:
    """Serializable lowering record for non-Conv/Gemm graph operations.

    These records describe each op lowering or reference a shape-specialized
    native artifact when the compiler has an executable kernel for it.
    """

    node_index: int
    node_name: str
    op_type: str
    lowering: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    attributes: Mapping[str, Any]
    input_shapes: Tuple[Optional[Tuple[int, ...]], ...]
    output_shapes: Tuple[Optional[Tuple[int, ...]], ...]
    status: str = "descriptor_only_native_kernel_required"
    quantization: Optional[Mapping[str, Any]] = None
    parameters: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_index": self.node_index,
            "node_name": self.node_name,
            "op_type": self.op_type,
            "lowering": self.lowering,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attributes": dict(self.attributes),
            "input_shapes": [list(shape) if shape is not None else None for shape in self.input_shapes],
            "output_shapes": [list(shape) if shape is not None else None for shape in self.output_shapes],
            "status": self.status,
            "quantization": dict(self.quantization) if self.quantization is not None else None,
            "parameters": dict(self.parameters) if self.parameters is not None else None,
        }


_OP_LOWERINGS = {
    "Gemm": "dense_gemm_int8",
    "Add": "broadcast_binary_int8",
    "Mul": "broadcast_binary_quantized",
    "Relu": "elementwise_relu_int8",
    "MaxPool": "nchw_max_pool",
    "AveragePool": "nchw_average_pool",
    "GlobalAveragePool": "nchw_global_average_pool",
    "Flatten": "contiguous_tensor_view",
    "Reshape": "contiguous_tensor_view",
    "Transpose": "tensor_permutation",
    "Concat": "tensor_concatenation",
    "QuantizeLinear": "quantize_linear_edge",
    "DequantizeLinear": "dequantize_linear_edge",
}


def emit_operation_specs(plan: ResNetCodegenPlan, model: Any) -> Tuple[OperationArtifactSpec, ...]:
    """Emit operation-level lowering descriptors for graph ops outside Conv kernels."""
    graph = model.graph
    nodes = list(graph.node)
    shapes, _ = _value_metadata(model)
    graph_outputs = {str(value.name) for value in graph.output}
    scalar_constants: dict[str, float] = {}
    float32_values: set[str] = set()
    try:
        from onnx import numpy_helper
        for initializer in graph.initializer:
            arr = numpy_helper.to_array(initializer)
            if arr.size == 1:
                scalar_constants[str(initializer.name)] = float(arr.reshape(-1)[0])
            if int(initializer.data_type) == 1:
                float32_values.add(str(initializer.name))
        for value in (*graph.input, *graph.value_info, *graph.output):
            tensor = getattr(getattr(value.type, "tensor_type", None), "elem_type", 0)
            if int(tensor) == 1:
                float32_values.add(str(value.name))
        for constant in nodes:
            if str(constant.op_type) != "Constant" or not constant.output:
                continue
            value_attr = next((attr for attr in constant.attribute if str(attr.name) == "value"), None)
            if value_attr is not None:
                arr = numpy_helper.to_array(value_attr.t)
                if arr.size == 1:
                    scalar_constants[str(constant.output[0])] = float(arr.reshape(-1)[0])
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    float_propagating_ops = {
        "Conv", "Gemm", "Relu", "Add", "Mul", "MaxPool", "AveragePool",
        "GlobalAveragePool", "Flatten", "Reshape", "Transpose", "Concat",
    }
    for _ in range(len(nodes) + 1):
        changed = False
        for candidate in nodes:
            if str(candidate.op_type) == "DequantizeLinear":
                for output in candidate.output:
                    if str(output) not in float32_values:
                        float32_values.add(str(output))
                        changed = True
            elif str(candidate.op_type) in float_propagating_ops and any(
                str(value) in float32_values for value in candidate.input
            ):
                for output in candidate.output:
                    if str(output) not in float32_values:
                        float32_values.add(str(output))
                        changed = True
        if not changed:
            break
    consumers: dict[str, list[int]] = {}
    for consumer_index, consumer in enumerate(nodes):
        for value in consumer.input:
            consumers.setdefault(str(value), []).append(consumer_index)
    try:
        qdq = extract_qdq_edges(model)
    except ValueError:
        # Lightweight planner fixtures may intentionally omit QDQ constants;
        # in that case they remain descriptors without a compiled fusion.
        qdq = ()
    dequantized = {edge.output_name: edge for edge in qdq if edge.op_type == "DequantizeLinear"}
    quantized_by_input = {edge.input_name: edge for edge in qdq if edge.op_type == "QuantizeLinear"}
    result = []
    for dispatch in plan.dispatches:
        node_index = dispatch.node_indices[0]
        node = nodes[node_index]
        op = str(node.op_type)
        if op == "Conv":
            continue
        qadd = None
        mul_scalar = None
        mul_identity = False
        gap = None
        gap_identity = False
        qadd_inputs = tuple(str(value) for value in node.input if value)
        qadd_outputs = tuple(str(value) for value in node.output if value)
        qadd_node_indices: Tuple[int, ...] = (node_index,)
        if op == "Add" and len(node.input) == 2 and node.output:
            input_edges = [dequantized.get(str(value)) for value in node.input]
            post_ops = consumers.get(str(node.output[0]), ())
            relu_index = (
                post_ops[0]
                if len(post_ops) == 1
                and str(nodes[post_ops[0]].op_type) == "Relu"
                and str(node.output[0]) not in graph_outputs
                else None
            )
            output_quant = None
            if relu_index is not None:
                relu = nodes[relu_index]
                relu_consumers = consumers.get(str(relu.output[0]), ()) if len(relu.output) == 1 else ()
                if (
                    len(relu.output) == 1
                    and len(relu_consumers) == 1
                    and str(relu.output[0]) not in graph_outputs
                ):
                    output_quant = quantized_by_input.get(str(relu.output[0]))
            if all(input_edges) and output_quant is not None:
                params = [edge.params for edge in input_edges] + [output_quant.params]
                output_shape = shapes.get(str(output_quant.output_name))
                ratios = (params[0].scale[0] / params[2].scale[0], params[1].scale[0] / params[2].scale[0])
                power_of_two_ratios = all(
                    math.frexp(value)[0] == 0.5 and -30 <= math.frexp(value)[1] - 1 <= 24
                    for value in ratios
                )
                compatible_shape = (
                    output_shape is not None
                    and all(shapes.get(edge.input_name) == output_shape for edge in input_edges)
                )
                if (
                    all(param.scalar and param.dtype == "u8" for param in params)
                    and compatible_shape
                    and power_of_two_ratios
                ):
                    multipliers = [round(value * (1 << 30)) for value in ratios]
                    qadd = {
                        "input_scales": [param.scale[0] for param in params[:2]],
                        "input_zero_points": [param.zero_point[0] for param in params[:2]],
                        "output_scale": params[2].scale[0],
                        "output_zero_point": params[2].zero_point[0],
                        "multipliers_q30": multipliers,
                        "dtype": "u8",
                        "relu": True,
                        "raw_inputs": [edge.input_name for edge in input_edges],
                        "raw_output": output_quant.output_name,
                    }
                    qadd_inputs = tuple(qadd["raw_inputs"])
                    qadd_outputs = (str(qadd["raw_output"]),)
                    qadd_node_indices = (node_index, relu_index, output_quant.node_index)
        if op == "Mul" and len(node.input) == 2 and node.output:
            scalar_slot = next((slot for slot, value in enumerate(node.input) if str(value) in scalar_constants), None)
            if scalar_slot is not None:
                data_slot = 1 - scalar_slot
                data_name = str(node.input[data_slot])
                out_name = str(node.output[0])
                scalar = scalar_constants[str(node.input[scalar_slot])]
                if (
                    math.isfinite(scalar)
                    and data_name in float32_values
                    and shapes.get(data_name) is not None
                    and shapes.get(data_name) == shapes.get(out_name)
                ):
                    mul_scalar = {"scalar": scalar, "dtype": "float32", "input_name": data_name}
                    mul_identity = scalar == 1.0
                    qadd_inputs = (data_name,)
                    qadd_outputs = (out_name,)
        if op == "GlobalAveragePool" and len(node.input) == 1 and node.output:
            input_name, output_name = str(node.input[0]), str(node.output[0])
            input_shape = shapes.get(input_name)
            output_shape = shapes.get(output_name)
            if (
                input_name in float32_values
                and input_shape is not None
                and len(input_shape) == 4
                and input_shape[0] == 1
                and output_shape == (1, input_shape[1], 1, 1)
            ):
                channels, spatial = input_shape[1], input_shape[2] * input_shape[3]
                gap_identity = spatial == 1
                tile_channels = next(
                    (tile for tile in (64, 32, 16, 8, 4, 2, 1) if channels % tile == 0),
                    1,
                )
                if not gap_identity:
                    gap = {
                        "channels": channels,
                        "spatial": spatial,
                        "tile_channels": tile_channels,
                        "dtype": "float32",
                    }
        result.append(OperationArtifactSpec(
            node_index=node_index,
            node_name=dispatch.node_names[0],
            op_type=op,
            lowering=(
                "quantized_add_relu_u8" if qadd is not None
                else "global_avgpool_nchw_f32" if gap is not None
                else "identity_device_view" if gap_identity
                else "identity_device_view" if mul_identity
                else "mul_scalar_f32" if mul_scalar is not None
                else _OP_LOWERINGS.get(op, "unsupported_native_lowering")
            ),
            inputs=qadd_inputs,
            outputs=qadd_outputs,
            attributes=_attributes(node),
            input_shapes=tuple(shapes.get(str(value)) for value in qadd_inputs),
            output_shapes=tuple(shapes.get(str(value)) for value in qadd_outputs),
            status=(
                "compilable_quantized_add_relu" if qadd is not None
                else "compilable_global_avgpool_f32" if gap is not None
                else "zero_copy_device_view" if gap_identity
                else "zero_copy_device_view" if mul_identity
                else "compilable_mul_scalar_f32" if mul_scalar is not None
                else "compilable_iron_kernel" if op == "Relu"
                else "zero_copy_device_view" if op in {"Flatten", "Reshape"}
                else "descriptor_only_native_kernel_required" if op in _OP_LOWERINGS
                else "unsupported"
            ),
            quantization=(
                {**qadd, "fused_node_indices": list(qadd_node_indices)} if qadd is not None else None
            ),
            parameters=(
                {**gap, "input_name": str(node.input[0])} if gap is not None
                else mul_scalar
            ),
        ))
    # Q/DQ nodes are graph-edge semantics rather than dispatches. Emit them
    # too, including graph-boundary conversions that are outside a region.
    for node_index, node in enumerate(nodes):
        op = str(node.op_type)
        if op not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        result.append(OperationArtifactSpec(
            node_index=node_index,
            node_name=str(getattr(node, "name", "") or f"{op}_{node_index}"),
            op_type=op,
            lowering=_OP_LOWERINGS[op],
            inputs=tuple(str(value) for value in node.input if value),
            outputs=tuple(str(value) for value in node.output if value),
            attributes=_attributes(node),
            input_shapes=tuple(shapes.get(str(value)) for value in node.input if value),
            output_shapes=tuple(shapes.get(str(value)) for value in node.output if value),
        ))
    result.sort(key=lambda spec: spec.node_index)
    return tuple(result)


def emit_bottleneck_specs(
    model: Any,
    *,
    source: str = "mlir-aie/programming_examples/ml/bottleneck/bottleneck.py",
    columns: int = 8,
) -> Tuple[BottleneckArtifactSpec, ...]:
    """Describe each detected ResNet block for the real IRON bottleneck kernel."""
    result = []
    for block in plan_bottleneck_blocks(model, columns=columns):
        first = block.conv_plans[0]
        result.append(
            BottleneckArtifactSpec(
                prefix=block.prefix,
                input_shape=first.input_shape,
                has_downsample=block.skip_conv_index is not None,
                source=source,
            )
        )
    return tuple(result)


def _artifact_key(shape: Sequence[int], tile: Sequence[int], fused_relu: bool) -> str:
    m, k, n = (int(value) for value in shape)
    tm, tk, tn = (int(value) for value in tile)
    suffix = "_relu" if fused_relu else ""
    return f"conv_im2col_i8_m{m}k{k}n{n}_tm{tm}tk{tk}tn{tn}{suffix}"


def _compiled_shape(
    shape: Sequence[int], tile: Sequence[int], requested_columns: int
) -> Tuple[Tuple[int, int, int], int, Tuple[int, int, int]]:
    """Pad a logical GEMM to the whole-array example's legal dimensions."""
    m, k, n = (int(value) for value in shape)
    tm, tk, tn = (int(value) for value in tile)
    # Prefer the widest legal placement, but reduce columns for narrow N.
    required_columns = max(1, (n + tn - 1) // tn)
    columns = min(requested_columns, required_columns)
    # M must be a multiple of m*4 and contain an even number of transfer
    # blocks in the current whole-array design: M >= m*4*2.
    m_block = tm * 4 * 2
    cm = max(m_block, ((m + m_block - 1) // m_block) * m_block)
    ck = ((k + tk - 1) // tk) * tk
    cn = ((n + tn * columns - 1) // (tn * columns)) * (tn * columns)
    return (cm, ck, cn), columns, (cm - m, ck - k, cn - n)


def emit_kernel_specs(
    plan: ResNetCodegenPlan,
    *,
    columns: int = 8,
    source: str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py",
    entrypoint: str = "whole_array",
) -> Tuple[KernelArtifactSpec, ...]:
    """Deduplicate Conv GEMM artifacts required by a graph schedule.

    These are intentionally build specifications, not fake executable
    artifacts.  ``requires_im2col`` makes the current boundary explicit:
    the GEMM kernel is ready for XDNA, while packing and post-op handling are
    supplied by the graph runtime until a fused Conv kernel is emitted.
    """
    specs: dict[str, KernelArtifactSpec] = {}
    for dispatch in plan.conv_dispatches:
        conv = dispatch.conv
        assert conv is not None
        compiled_shape, columns, padding = _compiled_shape(conv.gemm_shape, conv.tile, columns)
        logical_work = conv.gemm_shape[0] * conv.gemm_shape[1] * conv.gemm_shape[2]
        compiled_work = compiled_shape[0] * compiled_shape[1] * compiled_shape[2]
        # Whole-array padding is acceptable for the large layers, but becomes
        # counterproductive for small spatial tails.  Keep those explicit so
        # the next native direct-Conv emitter can replace them safely.
        buildable = compiled_work <= logical_work * 4
        strategy = "whole_array_gemm" if buildable else "native_conv_required"
        key = _artifact_key(compiled_shape, conv.tile, conv.fused_relu) + ("_native" if not buildable else "")
        specs.setdefault(
            key,
            KernelArtifactSpec(
                key=key,
                kernel_kind=dispatch.kernel_kind,
                gemm_shape=conv.gemm_shape,
                compiled_shape=compiled_shape,
                tile=conv.tile,
                columns=columns,
                padding=padding,
                strategy=strategy,
                buildable_with_whole_array=buildable,
                source=source,
                entrypoint=entrypoint,
                requires_im2col=True,
                fused_relu=conv.fused_relu,
            ),
        )
    return tuple(specs.values())


def render_build_manifest(
    plan: ResNetCodegenPlan,
    *,
    model: Optional[Any] = None,
    columns: int = 8,
    source: str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py",
    entrypoint: str = "whole_array",
    bottleneck_source: str = "mlir-aie/programming_examples/ml/bottleneck/bottleneck.py",
) -> Mapping[str, Any]:
    """Render a stable JSON-ready offline build manifest."""
    specs = emit_kernel_specs(plan, columns=columns, source=source, entrypoint=entrypoint)
    bottleneck_specs = emit_bottleneck_specs(model, source=bottleneck_source, columns=columns) if model is not None else ()
    operation_specs = emit_operation_specs(plan, model) if model is not None else ()
    schedule = codegen_plan_to_dict(plan)
    return {
        "format": "xdna-resnet-build-v1",
        "runtime": "iron-xrt",
        "execution": "build_manifest_only",
        "migraphx": False,
        "ort": False,
        "graph_dispatches": plan.estimated_dispatches,
        "unsupported_ops": list(plan.unsupported_ops),
        "schedule": schedule,
        "graph_programs": schedule["graph_regions"],
        "kernels": [spec.to_dict() for spec in specs],
        "operation_kernels": [spec.to_dict() for spec in operation_specs],
        "bottlenecks": [spec.to_dict() for spec in bottleneck_specs],
    }


def write_build_manifest(
    plan: ResNetCodegenPlan,
    path: str | Path,
    *,
    model: Optional[Any] = None,
    columns: int = 8,
    source: str = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py",
    entrypoint: str = "whole_array",
) -> Path:
    """Write the build manifest for an explicit user-requested output path."""
    output = Path(path)
    output.write_text(json.dumps(render_build_manifest(plan, model=model, columns=columns, source=source, entrypoint=entrypoint), indent=2) + "\n", encoding="utf-8")
    return output
