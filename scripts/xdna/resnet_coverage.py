"""Complete semantic-node coverage map for the QDQ ResNet execution path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

try:
    from .qdq_runtime import QDQEdge, qdq_edge_map
    from .resnet_codegen import ResNetCodegenPlan, build_codegen_plan
except ImportError:  # direct script-directory imports used by tooling/tests
    from qdq_runtime import QDQEdge, qdq_edge_map
    from resnet_codegen import ResNetCodegenPlan, build_codegen_plan


_LOWERED_OPS = frozenset({"Conv", "Relu", "Add", "Mul", "MaxPool", "GlobalAveragePool", "Flatten", "Gemm"})


@dataclass(frozen=True)
class CoverageEntry:
    node_index: int
    node_name: str
    op_type: str
    kernel_kind: str
    fused_group: Tuple[int, ...]
    status: str
    input_edges: Tuple[QDQEdge, ...]
    output_edge: Optional[QDQEdge]

    @property
    def covered(self) -> bool:
        return self.status in {"lowered", "fused"}


@dataclass(frozen=True)
class ResNetCoverage:
    codegen: ResNetCodegenPlan
    entries: Tuple[CoverageEntry, ...]

    @property
    def semantic_nodes(self) -> int:
        return len(self.entries)

    @property
    def covered_nodes(self) -> int:
        return sum(entry.covered for entry in self.entries)

    @property
    def coverage_percent(self) -> float:
        return 100.0 * self.covered_nodes / self.semantic_nodes if self.semantic_nodes else 100.0

    @property
    def uncovered_ops(self) -> Tuple[str, ...]:
        return tuple(sorted({entry.op_type for entry in self.entries if not entry.covered}))


def build_resnet_coverage(model: Any, **kwargs: Any) -> ResNetCoverage:
    """Attach QDQ metadata and a status to every non-QDQ/non-Constant node."""
    codegen = build_codegen_plan(model, **kwargs)
    nodes = list(model.graph.node)
    edges = qdq_edge_map(model)
    group_by_node = {
        index: dispatch
        for dispatch in codegen.dispatches
        for index in dispatch.node_indices
    }
    entries = []
    for index, node in enumerate(nodes):
        op_type = str(node.op_type)
        if op_type in {"QuantizeLinear", "DequantizeLinear", "Constant"}:
            continue
        dispatch = group_by_node.get(index)
        if dispatch is None:
            status = "unsupported"
            kernel_kind = op_type.lower()
            group = (index,)
        else:
            status = "fused" if len(dispatch.node_indices) > 1 else "lowered"
            kernel_kind = dispatch.kernel_kind
            group = dispatch.node_indices
        input_edges = tuple(edge for edge in edges.values() if edge.output_name in {str(value) for value in node.input})
        output_edge = next((edges.get(str(value)) for value in node.output if str(value) in edges), None)
        entries.append(
            CoverageEntry(
                node_index=index,
                node_name=str(getattr(node, "name", "")) or f"{op_type}_{index}",
                op_type=op_type,
                kernel_kind=kernel_kind,
                fused_group=tuple(group),
                status=status,
                input_edges=input_edges,
                output_edge=output_edge,
            )
        )
    return ResNetCoverage(codegen=codegen, entries=tuple(entries))


def coverage_to_dict(coverage: ResNetCoverage) -> Mapping[str, Any]:
    """Serialize static lowering coverage without implying device execution.

    ``lowered`` and ``fused`` describe the planner's operator inventory. They
    do not establish that an artifact exists, that a runner selected it, or
    that the node ran on hardware. Device execution must be reported from a
    runtime result that has those facts.
    """
    return {
        "coverage_scope": "static_codegen_plan",
        "semantic_nodes": coverage.semantic_nodes,
        "covered_nodes": coverage.covered_nodes,
        "coverage_percent": coverage.coverage_percent,
        "hardware_execution_coverage": {
            "status": "not_measured_by_static_planner",
            "covered_nodes": None,
            "coverage_percent": None,
        },
        "uncovered_ops": list(coverage.uncovered_ops),
        "entries": [
            {
                "node_index": entry.node_index,
                "node_name": entry.node_name,
                "op_type": entry.op_type,
                "kernel_kind": entry.kernel_kind,
                "fused_group": list(entry.fused_group),
                "status": entry.status,
                "execution_status": "not_assessed",
                "input_qdq_edges": [edge.output_name for edge in entry.input_edges],
                "output_qdq_edge": entry.output_edge.output_name if entry.output_edge else None,
            }
            for entry in coverage.entries
        ],
    }
