"""Static planning for AMD Ryzen AI QDQ ResNet-style graphs.

This module does not execute a model.  It turns the structure that Vitis AI
receives into an explicit codegen plan: QDQ nodes are treated as edges,
convolution/activation pairs are identified as fusion candidates, and every
remaining operator is reported instead of silently being called NPU work.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


QDQ_OPS = frozenset({"QuantizeLinear", "DequantizeLinear"})
STATIC_OPS = frozenset({"Constant"})
BODY_OPS = frozenset(
    {
        "Conv",
        "Gemm",
        "Relu",
        "Add",
        "Mul",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "Flatten",
        "Reshape",
        "Transpose",
        "Concat",
    }
)
FUSIBLE_PRODUCERS = frozenset({"Conv", "Gemm"})
FUSIBLE_POST_OPS = frozenset({"Relu"})


@dataclass(frozen=True)
class ResNetFusion:
    """A semantic group that can become one XDNA dispatch."""

    node_indices: Tuple[int, ...]
    op_types: Tuple[str, ...]
    kernel_kind: str


@dataclass(frozen=True)
class ResNetPlan:
    """The static lowering inventory for one QDQ graph."""

    total_nodes: int
    op_counts: Mapping[str, int]
    semantic_nodes: int
    boundary_qdq_nodes: Tuple[int, ...]
    internal_qdq_nodes: int
    fusions: Tuple[ResNetFusion, ...]
    unsupported_ops: Tuple[str, ...]
    estimated_dispatches: int

    @property
    def fusion_reduction_percent(self) -> float:
        if not self.semantic_nodes:
            return 0.0
        return 100.0 * (self.semantic_nodes - self.estimated_dispatches) / self.semantic_nodes


def _nodes(model: Any) -> Sequence[Any]:
    try:
        return model.graph.node
    except AttributeError as exc:
        raise TypeError("expected an ONNX ModelProto-like object") from exc


def _consumers(nodes: Sequence[Any]) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = {}
    for index, node in enumerate(nodes):
        for value in getattr(node, "input", ()):
            result.setdefault(str(value), []).append(index)
    return result


def _graph_outputs(model: Any) -> set[str]:
    return {str(value.name) for value in getattr(model.graph, "output", ())}


def _next_semantic(
    node_index: int,
    nodes: Sequence[Any],
    consumers: Mapping[str, Sequence[int]],
) -> Tuple[int, ...]:
    """Follow one output through QDQ nodes to its semantic consumers."""
    frontier = list(getattr(nodes[node_index], "output", ()))
    seen_values: set[str] = set()
    result: List[int] = []
    while frontier:
        value = str(frontier.pop())
        if value in seen_values:
            continue
        seen_values.add(value)
        for consumer_index in consumers.get(value, ()):
            consumer = nodes[consumer_index]
            if str(consumer.op_type) in QDQ_OPS:
                frontier.extend(getattr(consumer, "output", ()))
            else:
                result.append(consumer_index)
    return tuple(dict.fromkeys(result))


def plan_qdq_resnet(model: Any) -> ResNetPlan:
    """Build a conservative fusion plan for a static QDQ ResNet graph.

    QDQ operations are not counted as standalone compute dispatches.  Only a
    QDQ node touching a graph input or output is marked as a boundary; the
    remaining QDQ nodes are internal quantized edges that the eventual
    fused-kernel codegen must preserve in its scale/zero-point metadata.
    """
    nodes = list(_nodes(model))
    consumers = _consumers(nodes)
    graph_outputs = _graph_outputs(model)
    counts = Counter(str(node.op_type) for node in nodes)
    semantic = [
        i
        for i, node in enumerate(nodes)
        if str(node.op_type) not in QDQ_OPS and str(node.op_type) not in STATIC_OPS
    ]

    boundary: set[int] = set()
    graph_inputs = {str(value.name) for value in getattr(model.graph, "input", ())}
    for index, node in enumerate(nodes):
        op_type = str(node.op_type)
        if op_type == "QuantizeLinear" and any(str(value) in graph_inputs for value in getattr(node, "input", ())):
            boundary.add(index)
        if op_type == "DequantizeLinear" and any(str(value) in graph_outputs for value in getattr(node, "output", ())):
            boundary.add(index)

    unsupported = tuple(sorted({str(nodes[i].op_type) for i in semantic if str(nodes[i].op_type) not in BODY_OPS}))
    fusions: List[ResNetFusion] = []
    consumed: set[int] = set()
    for index in semantic:
        if index in consumed:
            continue
        node = nodes[index]
        op_type = str(node.op_type)
        following = _next_semantic(index, nodes, consumers)
        if op_type in FUSIBLE_PRODUCERS and len(following) == 1:
            post_index = following[0]
            post_type = str(nodes[post_index].op_type)
            if post_type in FUSIBLE_POST_OPS:
                kind = "conv_relu_int8" if op_type == "Conv" else "gemm_relu_int8"
                fusions.append(ResNetFusion((index, post_index), (op_type, post_type), kind))
                consumed.update((index, post_index))
                continue
        kind = {
            "Conv": "conv_int8",
            "Gemm": "gemm_int8",
            "Add": "residual_add_int8",
            "Relu": "relu_int8",
        }.get(op_type, op_type.lower())
        fusions.append(ResNetFusion((index,), (op_type,), kind))
        consumed.add(index)

    return ResNetPlan(
        total_nodes=len(nodes),
        op_counts=dict(counts),
        semantic_nodes=len(semantic),
        boundary_qdq_nodes=tuple(sorted(boundary)),
        internal_qdq_nodes=sum(counts.get(op, 0) for op in QDQ_OPS) - len(boundary),
        fusions=tuple(fusions),
        unsupported_ops=unsupported,
        estimated_dispatches=len(fusions),
    )


def plan_to_dict(plan: ResNetPlan) -> Dict[str, Any]:
    """Return a JSON-serializable report for benchmark/planner tooling."""
    return {
        "total_nodes": plan.total_nodes,
        "op_counts": dict(plan.op_counts),
        "semantic_nodes": plan.semantic_nodes,
        "boundary_qdq_nodes": list(plan.boundary_qdq_nodes),
        "internal_qdq_nodes": plan.internal_qdq_nodes,
        "unsupported_ops": list(plan.unsupported_ops),
        "estimated_dispatches": plan.estimated_dispatches,
        "fusion_reduction_percent": plan.fusion_reduction_percent,
        "fusions": [
            {
                "node_indices": list(group.node_indices),
                "op_types": list(group.op_types),
                "kernel_kind": group.kernel_kind,
            }
            for group in plan.fusions
        ],
    }
