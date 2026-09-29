"""Graph-level XDNA codegen plan for QDQ ResNet graphs.

This module is the bridge between the graph optimizer and a future IRON
emitter.  It does not launch hardware: it produces a complete, serializable
dispatch schedule and attaches the concrete Conv lowering metadata to each
Conv-containing fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    from .conv_lowering import ConvGemmPlan, plan_all_convs
    from .graph_fusion import GraphRegion, graph_regions_to_dict, plan_graph_regions
    from .resnet_plan import ResNetPlan, plan_qdq_resnet
except ImportError:  # direct script-directory imports used by tooling/tests
    from conv_lowering import ConvGemmPlan, plan_all_convs
    from graph_fusion import GraphRegion, graph_regions_to_dict, plan_graph_regions
    from resnet_plan import ResNetPlan, plan_qdq_resnet


@dataclass(frozen=True)
class DispatchSpec:
    """One semantic graph group that the eventual emitter will dispatch."""

    node_indices: Tuple[int, ...]
    node_names: Tuple[str, ...]
    kernel_kind: str
    op_type: str
    conv: Optional[ConvGemmPlan] = None

    @property
    def gemm_shape(self) -> Optional[Tuple[int, int, int]]:
        return self.conv.gemm_shape if self.conv is not None else None

    @property
    def tile(self) -> Optional[Tuple[int, int, int]]:
        return self.conv.tile if self.conv is not None else None


@dataclass(frozen=True)
class ResNetCodegenPlan:
    """Complete static schedule for a batch-1 QDQ graph."""

    graph: ResNetPlan
    dispatches: Tuple[DispatchSpec, ...]
    boundary_qdq_nodes: Tuple[int, ...]
    unsupported_ops: Tuple[str, ...]
    graph_regions: Tuple[GraphRegion, ...] = ()

    @property
    def conv_dispatches(self) -> Tuple[DispatchSpec, ...]:
        return tuple(dispatch for dispatch in self.dispatches if dispatch.conv is not None)

    @property
    def estimated_dispatches(self) -> int:
        return len(self.dispatches)


def _node_name(node: Any, index: int) -> str:
    return str(getattr(node, "name", "")) or f"{node.op_type}_{index}"


def build_codegen_plan(
    model: Any,
    *,
    dtype: str = "i8",
    columns: int = 8,
    profile: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
    optimize_small_m: bool = False,
) -> ResNetCodegenPlan:
    """Bind graph fusions to concrete XDNA dispatch metadata.

    ``strict`` is useful for an emitter: it rejects unsupported semantic ops
    before any artifact is compiled.  The default keeps them in the report so
    a benchmark can still measure the supported subset and identify fallback
    work.
    """
    graph = plan_qdq_resnet(model)
    if strict and graph.unsupported_ops:
        raise ValueError(f"unsupported XDNA graph operators: {graph.unsupported_ops}")
    nodes = list(model.graph.node)
    convs = {
        plan.node_index: plan
        for plan in plan_all_convs(
            model,
            dtype=dtype,
            columns=columns,
            profile=profile,
            optimize_small_m=optimize_small_m,
        )
    }
    dispatches = []
    for fusion in graph.fusions:
        first = fusion.node_indices[0]
        node = nodes[first]
        dispatches.append(
            DispatchSpec(
                node_indices=fusion.node_indices,
                node_names=tuple(_node_name(nodes[index], index) for index in fusion.node_indices),
                kernel_kind=fusion.kernel_kind,
                op_type=str(node.op_type),
                conv=convs.get(first),
            )
        )
    return ResNetCodegenPlan(
        graph=graph,
        dispatches=tuple(dispatches),
        boundary_qdq_nodes=graph.boundary_qdq_nodes,
        unsupported_ops=graph.unsupported_ops,
        graph_regions=plan_graph_regions(model),
    )


def codegen_plan_to_dict(plan: ResNetCodegenPlan) -> Dict[str, Any]:
    """Return a JSON-compatible schedule for tooling and benchmark reports."""
    return {
        "estimated_dispatches": plan.estimated_dispatches,
        "boundary_qdq_nodes": list(plan.boundary_qdq_nodes),
        "unsupported_ops": list(plan.unsupported_ops),
        "graph_regions": graph_regions_to_dict(plan.graph_regions),
        "dispatches": [
            {
                "node_indices": list(dispatch.node_indices),
                "node_names": list(dispatch.node_names),
                "kernel_kind": dispatch.kernel_kind,
                "op_type": dispatch.op_type,
                "gemm_shape": list(dispatch.gemm_shape) if dispatch.gemm_shape else None,
                "tile": list(dispatch.tile) if dispatch.tile else None,
            }
            for dispatch in plan.dispatches
        ],
    }
