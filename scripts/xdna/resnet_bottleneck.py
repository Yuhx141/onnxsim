"""Map ONNX ResNet bottleneck blocks to the installed IRON primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

try:
    from .conv_lowering import ConvGemmPlan, plan_all_convs
except ImportError:  # direct script-directory imports used by tests/tooling
    from conv_lowering import ConvGemmPlan, plan_all_convs


@dataclass(frozen=True)
class BottleneckBlockPlan:
    prefix: str
    main_conv_indices: Tuple[int, int, int]
    skip_conv_index: int | None
    add_index: int
    conv_plans: Tuple[ConvGemmPlan, ...]
    kernel: str = "bottleneck_int8"

    @property
    def node_indices(self) -> Tuple[int, ...]:
        values = list(self.main_conv_indices)
        if self.skip_conv_index is not None:
            values.append(self.skip_conv_index)
        values.append(self.add_index)
        return tuple(values)


def _prefix(name: str) -> str | None:
    for marker in ("/conv1/Conv", "/conv2/Conv", "/conv3/Conv", "/downsample/downsample.0/Conv"):
        if marker in name:
            return name.split(marker, 1)[0]
    marker = "/Add"
    if marker in name:
        return name.split(marker, 1)[0]
    return None


def plan_bottleneck_blocks(model: Any, *, columns: int = 8) -> Tuple[BottleneckBlockPlan, ...]:
    """Detect the standard 3-conv-plus-residual structure in graph order."""
    nodes = list(model.graph.node)
    conv_plans = {plan.node_index: plan for plan in plan_all_convs(model, columns=columns)}
    groups: dict[str, dict[str, int]] = {}
    for index, node in enumerate(nodes):
        name = str(getattr(node, "name", ""))
        prefix = _prefix(name)
        if prefix is None:
            continue
        group = groups.setdefault(prefix, {})
        if str(node.op_type) == "Conv" and name.endswith("/conv1/Conv"):
            group["conv1"] = index
        elif str(node.op_type) == "Conv" and name.endswith("/conv2/Conv"):
            group["conv2"] = index
        elif str(node.op_type) == "Conv" and name.endswith("/conv3/Conv"):
            group["conv3"] = index
        elif str(node.op_type) == "Conv" and "/downsample/downsample.0/Conv" in name:
            group["skip"] = index
        elif name.endswith("/Add"):
            group["add"] = index
    result = []
    for prefix, group in groups.items():
        required = {"conv1", "conv2", "conv3", "add"}
        if not required.issubset(group) or not all(index in conv_plans for index in group.values() if index != group["add"]):
            continue
        conv_indices = (group["conv1"], group["conv2"], group["conv3"])
        skip = group.get("skip")
        result.append(
            BottleneckBlockPlan(
                prefix=prefix,
                main_conv_indices=conv_indices,
                skip_conv_index=skip,
                add_index=group["add"],
                conv_plans=tuple(conv_plans[index] for index in (*conv_indices, *(() if skip is None else (skip,)))),
            )
        )
    return tuple(result)
