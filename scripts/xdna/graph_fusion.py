"""Connected graph-region and tensor-lifetime planning for XDNA codegen.

This is a planning contract, not an executable fused kernel. It treats Q/DQ
nodes as quantized edges and records the graph boundaries and intermediate
buffers a device-resident graph program must preserve.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, Sequence

try:
    from .conv_lowering import plan_all_convs
    from .resnet_plan import BODY_OPS, QDQ_OPS, STATIC_OPS
except ImportError:  # direct script-directory imports
    from conv_lowering import plan_all_convs
    from resnet_plan import BODY_OPS, QDQ_OPS, STATIC_OPS


@dataclass(frozen=True)
class GraphRegion:
    region_id: int
    node_indices: tuple[int, ...]
    op_types: tuple[str, ...]
    input_values: tuple[str, ...]
    constant_values: tuple[str, ...]
    output_values: tuple[str, ...]
    internal_qdq_nodes: tuple[int, ...]
    instructions: tuple[Mapping[str, Any], ...]
    internal_tensors: tuple[Mapping[str, Any], ...]
    peak_live_bytes: int | None
    device_lowering_gaps: tuple[str, ...]


def _consumers(nodes: Sequence[Any]) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for input_index, value in enumerate(getattr(node, "input", ())):
            if value:
                result[str(value)].append((index, input_index))
    return result


def _semantic_source(value: str, producer: Mapping[str, int], nodes: Sequence[Any]) -> int | None:
    seen: set[str] = set()
    while value and value not in seen:
        seen.add(value)
        index = producer.get(value)
        if index is None:
            return None
        if str(nodes[index].op_type) not in QDQ_OPS:
            return index
        value = str(nodes[index].input[0])
    return None


def _qdq_root(value: str, producer: Mapping[str, int], nodes: Sequence[Any]) -> str:
    seen: set[str] = set()
    while value and value not in seen:
        seen.add(value)
        index = producer.get(value)
        if index is None or str(nodes[index].op_type) not in QDQ_OPS:
            return value
        value = str(nodes[index].input[0])
    return value


def _qdq_path_to_root(value: str, producer: Mapping[str, int], nodes: Sequence[Any]) -> tuple[int, ...]:
    path: list[int] = []
    seen: set[str] = set()
    while value and value not in seen:
        seen.add(value)
        index = producer.get(value)
        if index is None or str(nodes[index].op_type) not in QDQ_OPS:
            break
        path.append(index)
        value = str(nodes[index].input[0])
    return tuple(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # TensorProto-valued Constant attributes are serialized with a small,
    # dependency-light descriptor; initializer contents remain named inputs.
    if hasattr(value, "dims") and hasattr(value, "data_type"):
        return {"tensor_dims": [int(dim) for dim in value.dims], "tensor_data_type": int(value.data_type)}
    return str(value)


def _attributes(node: Any) -> dict[str, Any]:
    try:
        from onnx import helper, numpy_helper
    except ImportError:
        helper = numpy_helper = None
    result: dict[str, Any] = {}
    for attr in getattr(node, "attribute", ()):
        try:
            value = helper.get_attribute_value(attr) if helper is not None else None
        except Exception:
            value = None
        if value is None:
            if getattr(attr, "ints", ()):
                value = tuple(attr.ints)
            elif getattr(attr, "floats", ()):
                value = tuple(attr.floats)
            elif getattr(attr, "s", b""):
                value = attr.s
            elif hasattr(attr, "i"):
                value = attr.i
            elif hasattr(attr, "f"):
                value = attr.f
        if numpy_helper is not None and hasattr(value, "dims") and hasattr(value, "data_type"):
            try:
                value = numpy_helper.to_array(value)
            except Exception:
                pass
        result[str(attr.name)] = _json_value(value)
    return result


def _semantic_edges(nodes: Sequence[Any], consumers: Mapping[str, Sequence[tuple[int, int]]]):
    producer = {
        str(output): index
        for index, node in enumerate(nodes)
        for output in getattr(node, "output", ())
        if output
    }
    edges: list[tuple[int, int, str, tuple[int, ...]]] = []
    for source, node in enumerate(nodes):
        if str(node.op_type) in QDQ_OPS or str(node.op_type) in STATIC_OPS:
            continue
        for output in getattr(node, "output", ()):
            queue = deque([(str(output), ())])
            seen: set[str] = set()
            while queue:
                value, qdq_path = queue.popleft()
                if value in seen:
                    continue
                seen.add(value)
                for target, _ in consumers.get(value, ()):
                    target_op = str(nodes[target].op_type)
                    if target_op in QDQ_OPS:
                        queue.append((str(nodes[target].output[0]), qdq_path + (target,)))
                    elif target != source:
                        edges.append((source, target, str(output), qdq_path))
    return producer, tuple(dict.fromkeys(edges))


def _value_metadata(model: Any) -> tuple[dict[str, tuple[int, ...]], dict[str, int]]:
    shapes: dict[str, tuple[int, ...]] = {}
    sizes: dict[str, int] = {}
    # ONNX TensorProto element type byte widths. Unknown types remain unpriced.
    type_bytes = {
        1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 8: 1,
        9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2,
    }
    graph = model.graph
    for value in (*getattr(graph, "input", ()), *getattr(graph, "value_info", ()), *getattr(graph, "output", ())):
        tensor = getattr(getattr(value, "type", None), "tensor_type", None)
        dims = getattr(getattr(tensor, "shape", None), "dim", ())
        shape = tuple(
            1 if axis == 0 and int(getattr(dim, "dim_value", 0)) == 0
            else int(getattr(dim, "dim_value", 0))
            for axis, dim in enumerate(dims)
        )
        if dims and all(dim >= 0 for dim in shape):
            shapes[str(value.name)] = shape
        elem_type = int(getattr(tensor, "elem_type", 0))
        if elem_type in type_bytes:
            sizes[str(value.name)] = type_bytes[elem_type]
    for value in getattr(graph, "initializer", ()):
        shapes[str(value.name)] = tuple(int(d) for d in value.dims)
        sizes[str(value.name)] = type_bytes.get(int(getattr(value, "data_type", 0)), 0)
    for plan in plan_all_convs(model):
        output = str(graph.node[plan.node_index].output[0])
        shapes.setdefault(output, plan.output_shape)
        sizes.setdefault(output, 4)
    for node in graph.node:
        op = str(node.op_type)
        if op == "QuantizeLinear" and node.output:
            sizes.setdefault(str(node.output[0]), 1)
        elif op in QDQ_OPS | {"Conv", "Gemm", "Relu", "Add", "Mul", "MaxPool", "AveragePool", "GlobalAveragePool", "Flatten", "Reshape", "Transpose", "Concat"}:
            for output in node.output:
                sizes.setdefault(str(output), 4)
    # Infer common ResNet shapes so the planner can estimate live buffers
    # even when an exporter omits intermediate value_info records.
    for _ in range(len(graph.node) + 1):
        changed = False
        for node in graph.node:
            if not node.input or not node.output:
                continue
            op = str(node.op_type)
            source_shape = shapes.get(str(node.input[0]))
            shape = None
            if op in QDQ_OPS | {"Relu", "Add", "Mul"}:
                shape = source_shape
            elif op in {"MaxPool", "AveragePool"} and source_shape is not None and len(source_shape) == 4:
                attrs = {a.name: a for a in node.attribute}
                kernel = tuple(int(x) for x in attrs["kernel_shape"].ints)
                strides = tuple(int(x) for x in attrs["strides"].ints) if "strides" in attrs else (1, 1)
                pads = tuple(int(x) for x in attrs["pads"].ints) if "pads" in attrs else (0, 0, 0, 0)
                shape = (source_shape[0], source_shape[1],
                    (source_shape[2] + pads[0] + pads[2] - kernel[0]) // strides[0] + 1,
                    (source_shape[3] + pads[1] + pads[3] - kernel[1]) // strides[1] + 1)
            elif op == "GlobalAveragePool" and source_shape is not None and len(source_shape) >= 3:
                shape = (*source_shape[:2], *((1,) * (len(source_shape) - 2)))
            elif op == "Flatten" and source_shape is not None:
                attrs = {a.name: int(a.i) for a in node.attribute if getattr(a, "name", "") == "axis"}
                axis = attrs.get("axis", 1)
                shape = (prod(source_shape[:axis]), prod(source_shape[axis:]))
            elif op == "Gemm" and len(node.input) > 1:
                left, right = shapes.get(str(node.input[0])), shapes.get(str(node.input[1]))
                if left and right and len(left) == len(right) == 2:
                    attrs = {a.name: int(a.i) for a in node.attribute if getattr(a, "name", "") in {"transA", "transB"}}
                    m = left[1] if attrs.get("transA", 0) else left[0]
                    n = right[0] if attrs.get("transB", 0) else right[1]
                    shape = (m, n)
            if shape is not None:
                if shape is not None:
                    for output in node.output:
                        if str(output) not in shapes:
                            shapes[str(output)] = shape
                            changed = True
        if not changed:
            break
    return shapes, sizes


def plan_graph_regions(model: Any) -> tuple[GraphRegion, ...]:
    """Partition supported semantic nodes into maximal connected graph regions."""
    nodes = list(model.graph.node)
    consumers = _consumers(nodes)
    producer, edges = _semantic_edges(nodes, consumers)
    semantic = {i for i, n in enumerate(nodes) if str(n.op_type) not in QDQ_OPS | STATIC_OPS}
    candidates = {i for i in semantic if str(nodes[i].op_type) in BODY_OPS}
    adjacency: dict[int, set[int]] = {i: set() for i in candidates}
    for source, target, _, _ in edges:
        if source in candidates and target in candidates:
            adjacency[source].add(target)
            adjacency[target].add(source)

    components: list[tuple[int, ...]] = []
    unseen = set(candidates)
    while unseen:
        first = min(unseen)
        stack = [first]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            unseen.discard(current)
            stack.extend(adjacency[current] - component)
        components.append(tuple(sorted(component)))
    components.sort(key=lambda group: group[0])

    graph_outputs = {str(value.name) for value in getattr(model.graph, "output", ())}
    initializer_names = {str(value.name) for value in getattr(model.graph, "initializer", ())}
    constant_outputs = {
        str(output)
        for node in nodes
        if str(node.op_type) == "Constant"
        for output in getattr(node, "output", ())
    }
    shapes, sizes = _value_metadata(model)
    regions: list[GraphRegion] = []
    for region_id, group in enumerate(components):
        members = set(group)
        input_values: set[str] = set()
        constant_values: set[str] = set()
        output_values: set[str] = set()
        qdq_nodes: set[int] = set()
        for index in group:
            node = nodes[index]
            for value in getattr(node, "input", ()):
                if not value:
                    continue
                qdq_nodes.update(_qdq_path_to_root(str(value), producer, nodes))
                source = _semantic_source(str(value), producer, nodes)
                if source not in members:
                    root = _qdq_root(str(value), producer, nodes)
                    if root in initializer_names or root in constant_outputs:
                        constant_values.add(root)
                    else:
                        input_values.add(root)
            for output in getattr(node, "output", ()):
                consumers_after_qdq: set[int] = set()
                reaches_graph_output = str(output) in graph_outputs
                queue = deque([(str(output), ())])
                visited: set[str] = set()
                while queue:
                    value, path = queue.popleft()
                    if value in visited:
                        continue
                    visited.add(value)
                    for target, _ in consumers.get(value, ()):
                        if str(nodes[target].op_type) in QDQ_OPS:
                            qdq_nodes.add(target)
                            qdq_output = str(nodes[target].output[0])
                            reaches_graph_output = reaches_graph_output or qdq_output in graph_outputs
                            queue.append((qdq_output, path + (target,)))
                        else:
                            consumers_after_qdq.add(target)
                if reaches_graph_output or consumers_after_qdq - members:
                    output_values.add(str(output))
        internal_tensors = []
        positions = {node_index: position for position, node_index in enumerate(group)}
        for source, target, value, qpath in edges:
            if source not in members or target not in members:
                continue
            qdq_nodes.update(qpath)
            shape = shapes.get(value)
            item: dict[str, Any] = {
                "value": value,
                "producer_node": source,
                "consumer_node": target,
                "first_use": positions[source],
                "last_use": positions[target],
                "shape": list(shape) if shape is not None else None,
            }
            elem_bytes = sizes.get(value)
            if elem_bytes is not None and shape is not None:
                item["nbytes"] = prod(shape) * elem_bytes
            internal_tensors.append(item)
        constant_node_indices = {
            producer[value]
            for index in group
            for value in getattr(nodes[index], "input", ())
            if value and value in producer and str(nodes[producer[value]].op_type) in STATIC_OPS
        }
        program_indices = sorted(members | qdq_nodes | constant_node_indices)
        instructions = tuple(
            {
                "node_index": index,
                "op_type": str(nodes[index].op_type),
                "inputs": [str(value) for value in getattr(nodes[index], "input", ()) if value],
                "outputs": [str(value) for value in getattr(nodes[index], "output", ()) if value],
                "attributes": _attributes(nodes[index]),
                "quantization_edge": str(nodes[index].op_type) in QDQ_OPS,
            }
            for index in program_indices
        )
        program_outputs = {
            value for instruction in instructions for value in instruction["outputs"]
        }
        graph_input_names = {str(value.name) for value in getattr(model.graph, "input", ())}
        for instruction in instructions:
            for value in instruction["inputs"]:
                if value in program_outputs:
                    continue
                root = _qdq_root(value, producer, nodes)
                if root in initializer_names or root in constant_outputs:
                    constant_values.add(root)
                elif root in graph_input_names:
                    input_values.add(root)
                else:
                    input_values.add(value)
        program_outputs = {
            value for instruction in instructions for value in instruction["outputs"]
        }
        graph_input_names = {str(value.name) for value in getattr(model.graph, "input", ())}
        for instruction in instructions:
            for value in instruction["inputs"]:
                if value in program_outputs:
                    continue
                root = _qdq_root(value, producer, nodes)
                if root in initializer_names or root in constant_outputs:
                    constant_values.add(root)
                elif root in graph_input_names:
                    input_values.add(root)
                else:
                    input_values.add(value)
        tensor_lifetimes: dict[str, dict[str, Any]] = {}
        for tensor in internal_tensors:
            value = str(tensor["value"])
            lifetime = tensor_lifetimes.setdefault(value, dict(tensor))
            lifetime["first_use"] = min(lifetime["first_use"], tensor["first_use"])
            lifetime["last_use"] = max(lifetime["last_use"], tensor["last_use"])
        events: list[tuple[int, int]] = []
        for tensor in tensor_lifetimes.values():
            if "nbytes" in tensor:
                events.append((int(tensor["first_use"]), int(tensor["nbytes"])))
                events.append((int(tensor["last_use"]) + 1, -int(tensor["nbytes"])))
        live = peak = 0
        for _, delta in sorted(events):
            live += delta
            peak = max(peak, live)
        gaps = set()
        for index in group:
            op = str(nodes[index].op_type)
            if op == "Conv":
                gaps.update(("device_resident_conv_outputs", "fused_conv_epilogue"))
            elif op == "Gemm":
                gaps.add("device_resident_gemm_outputs")
            else:
                gaps.add(f"device_lowering:{op}")
        regions.append(GraphRegion(
            region_id=region_id,
            node_indices=group,
            op_types=tuple(str(nodes[index].op_type) for index in group),
            input_values=tuple(sorted(input_values)),
            constant_values=tuple(sorted(constant_values)),
            output_values=tuple(sorted(output_values)),
            internal_qdq_nodes=tuple(sorted(qdq_nodes)),
            instructions=instructions,
            internal_tensors=tuple(internal_tensors),
            peak_live_bytes=peak if events else None,
            device_lowering_gaps=tuple(sorted(gaps)),
        ))
    return tuple(regions)


def graph_regions_to_dict(regions: Sequence[GraphRegion]) -> list[dict[str, Any]]:
    return [
        {
            "region_id": region.region_id,
            "status": "planning_only_not_executable",
            "node_indices": list(region.node_indices),
            "op_types": list(region.op_types),
            "input_values": list(region.input_values),
            "constant_values": list(region.constant_values),
            "output_values": list(region.output_values),
            "internal_qdq_nodes": list(region.internal_qdq_nodes),
            "instructions": [dict(instruction) for instruction in region.instructions],
            "internal_tensors": [dict(tensor) for tensor in region.internal_tensors],
            "peak_live_bytes_known": region.peak_live_bytes,
            "device_lowering_gaps": list(region.device_lowering_gaps),
        }
        for region in regions
    ]
