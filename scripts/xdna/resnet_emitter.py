"""Dependency-light artifact specification for the XDNA ResNet schedule.

The emitter deliberately stops at an explicit artifact contract.  The same
contract can be consumed by the checked-in IRON whole-array GEMM example or
by a native Conv kernel once that kernel is available; no MIGraphX or ORT
dependency is introduced here.
"""

from __future__ import annotations

import json
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
    from .graph_fusion import _attributes
except ImportError:  # direct script-directory imports
    from graph_fusion import _attributes


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

    These records are emitted into the graph program so a later IRON kernel
    builder can consume them. They are deliberately marked as descriptors:
    this repository does not yet ship executable kernels for these ops.
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
        }


_OP_LOWERINGS = {
    "Gemm": "dense_gemm_int8",
    "Add": "broadcast_binary_int8",
    "Mul": "broadcast_binary_quantized",
    "Relu": "elementwise_relu_int8",
    "MaxPool": "nchw_max_pool",
    "AveragePool": "nchw_average_pool",
    "GlobalAveragePool": "nchw_global_average_pool",
    "Flatten": "view_or_reorder",
    "Reshape": "view_or_reorder",
    "Transpose": "tensor_permutation",
    "Concat": "tensor_concatenation",
    "QuantizeLinear": "quantize_linear_edge",
    "DequantizeLinear": "dequantize_linear_edge",
}


def emit_operation_specs(plan: ResNetCodegenPlan, model: Any) -> Tuple[OperationArtifactSpec, ...]:
    """Emit operation-level lowering descriptors for graph ops outside Conv kernels."""
    graph = model.graph
    nodes = list(graph.node)
    shapes: dict[str, Tuple[int, ...]] = {}
    for value in (*getattr(graph, "input", ()), *getattr(graph, "value_info", ()), *getattr(graph, "output", ())):
        dims = getattr(getattr(getattr(value, "type", None), "tensor_type", None), "shape", None)
        if dims is not None:
            shape = tuple(int(dim.dim_value) for dim in dims.dim)
            if all(shape):
                shapes[str(value.name)] = shape
    for value in getattr(graph, "initializer", ()):
        shapes[str(value.name)] = tuple(int(dim) for dim in value.dims)
    result = []
    for dispatch in plan.dispatches:
        node_index = dispatch.node_indices[0]
        node = nodes[node_index]
        op = str(node.op_type)
        if op == "Conv":
            continue
        result.append(OperationArtifactSpec(
            node_index=node_index,
            node_name=dispatch.node_names[0],
            op_type=op,
            lowering=_OP_LOWERINGS.get(op, "unsupported_native_lowering"),
            inputs=tuple(str(value) for value in node.input if value),
            outputs=tuple(str(value) for value in node.output if value),
            attributes=_attributes(node),
            input_shapes=tuple(shapes.get(str(value)) for value in node.input if value),
            output_shapes=tuple(shapes.get(str(value)) for value in node.output if value),
            status=("descriptor_only_native_kernel_required" if op in _OP_LOWERINGS else "unsupported"),
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
