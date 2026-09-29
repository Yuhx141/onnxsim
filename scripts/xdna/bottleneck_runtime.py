"""Runtime binding checks for the existing IRON INT8 bottleneck primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    from .qdq_runtime import qdq_edge_map
    from .resnet_bottleneck import BottleneckBlockPlan
except ImportError:  # direct script-directory imports used by tests/tooling
    from qdq_runtime import qdq_edge_map
    from resnet_bottleneck import BottleneckBlockPlan


@dataclass(frozen=True)
class BottleneckBinding:
    prefix: str
    weights: np.ndarray
    weight_shapes: Tuple[Tuple[int, ...], ...]
    biases: Tuple[np.ndarray, ...]
    skip_weight: Optional[np.ndarray]
    has_bias: bool
    has_downsample: bool
    executable_with_existing_primitive: bool
    reason: Optional[str]

    @property
    def required_postops(self) -> Tuple[str, ...]:
        postops = ["bias_requantize"] if self.has_bias else []
        if self.has_downsample:
            postops.append("downsample_conv")
        postops.append("residual_add")
        return tuple(postops)


def _initializers(model: Any) -> Dict[str, np.ndarray]:
    from onnx import numpy_helper
    return {str(value.name): numpy_helper.to_array(value) for value in model.graph.initializer}


def _resolve_initializer(name: str, model: Any, initializers: Dict[str, np.ndarray], edges: Mapping[str, Any]) -> np.ndarray:
    if name in initializers:
        return initializers[name]
    edge = edges.get(name)
    if edge is None:
        raise ValueError(f"tensor {name!r} is not backed by a static initializer")
    if edge.input_name in initializers:
        return initializers[edge.input_name]
    return _resolve_initializer(edge.input_name, model, initializers, edges)


def bind_bottleneck_block(model: Any, block: BottleneckBlockPlan) -> BottleneckBinding:
    """Pack the three main Conv weights and report primitive incompatibilities."""
    nodes = list(model.graph.node)
    initializers = _initializers(model)
    edges = qdq_edge_map(model)
    weights = []
    shapes = []
    biases = []
    has_bias = False
    for index in block.main_conv_indices:
        node = nodes[index]
        inputs = [str(value) for value in node.input]
        array = _resolve_initializer(inputs[1], model, initializers, edges).astype(np.int8)
        weights.append(array.reshape(-1))
        shapes.append(tuple(int(value) for value in array.shape))
        if len(inputs) >= 3 and bool(inputs[2]):
            has_bias = True
            biases.append(_resolve_initializer(inputs[2], model, initializers, edges).astype(np.int32).reshape(-1))
        else:
            biases.append(np.empty((0,), dtype=np.int32))
    skip_weight = None
    if block.skip_conv_index is not None:
        skip_inputs = [str(value) for value in nodes[block.skip_conv_index].input]
        skip_weight = _resolve_initializer(skip_inputs[1], model, initializers, edges).astype(np.int8).reshape(-1)
    reason = None
    executable = True
    if has_bias:
        executable = False
        reason = "existing IRON bottleneck primitive has no Conv bias inputs"
    elif block.skip_conv_index is not None:
        executable = False
        reason = "downsample residual path needs a separate skip Conv kernel"
    return BottleneckBinding(
        prefix=block.prefix,
        weights=np.concatenate(weights),
        weight_shapes=tuple(shapes),
        biases=tuple(biases),
        skip_weight=skip_weight,
        has_bias=has_bias,
        has_downsample=block.skip_conv_index is not None,
        executable_with_existing_primitive=executable,
        reason=reason,
    )
