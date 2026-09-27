"""Runtime metadata extraction for ONNX QuantizeLinear/DequantizeLinear edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class QuantParams:
    scale: Tuple[float, ...]
    zero_point: Tuple[int, ...]
    axis: Optional[int]
    dtype: str

    @property
    def scalar(self) -> bool:
        return len(self.scale) == 1


@dataclass(frozen=True)
class QDQEdge:
    node_index: int
    op_type: str
    input_name: str
    output_name: str
    params: QuantParams


def _attribute(node: Any, name: str, default: Any = None) -> Any:
    for attr in getattr(node, "attribute", ()):
        if str(getattr(attr, "name", "")) != name:
            continue
        if getattr(attr, "ints", None):
            return tuple(int(value) for value in attr.ints)
        if hasattr(attr, "i"):
            return int(attr.i)
        if getattr(attr, "s", None):
            return attr.s.decode() if isinstance(attr.s, bytes) else str(attr.s)
    return default


def _initializer_values(model: Any) -> Dict[str, Tuple[Tuple[Any, ...], str]]:
    result: Dict[str, Tuple[Tuple[Any, ...], str]] = {}
    for initializer in getattr(model.graph, "initializer", ()):
        # Avoid importing ONNX at module import time.  TensorProto-like test
        # doubles may expose values directly; real models use numpy_helper.
        if hasattr(initializer, "values"):
            values = tuple(initializer.values)
            dtype = str(getattr(initializer, "dtype", "u8"))
        else:
            try:
                from onnx import numpy_helper
                array = numpy_helper.to_array(initializer)
                values = tuple(array.reshape(-1).tolist())
                dtype = str(array.dtype)
            except Exception as exc:
                raise ValueError(f"cannot read initializer {initializer.name!r}") from exc
        result[str(initializer.name)] = (values, dtype)
    return result


def _dtype(dtype: str, zero_point: Sequence[int]) -> str:
    normalized = str(dtype).lower()
    if "uint8" in normalized or normalized in {"u8", "uint8"}:
        return "u8"
    if "int8" in normalized or normalized in {"i8", "int8"}:
        return "i8"
    # Lightweight graph shims may omit dtype; retain the conservative prior.
    return "u8" if zero_point and min(zero_point) >= 0 else "i8"


def extract_qdq_edges(model: Any) -> Tuple[QDQEdge, ...]:
    """Extract static quantization parameters for every Q/DQ node.

    Per-channel parameters are retained as vectors and their ONNX axis is
    preserved.  The XDNA emitter can then decide whether to fold them into a
    kernel or schedule a vector requantization operation.
    """
    values = _initializer_values(model)
    result = []
    for index, node in enumerate(getattr(model.graph, "node", ())):
        op_type = str(node.op_type)
        if op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        inputs = [str(value) for value in getattr(node, "input", ())]
        outputs = [str(value) for value in getattr(node, "output", ())]
        if len(inputs) < 3 or not outputs or inputs[1] not in values or inputs[2] not in values:
            raise ValueError(f"QDQ node {index} requires static scale and zero-point initializers")
        scale_values, _ = values[inputs[1]]
        zero_values, zero_dtype = values[inputs[2]]
        scales = tuple(float(value) for value in scale_values)
        zero_point = tuple(int(value) for value in zero_values)
        if not scales or len(scales) != len(zero_point) or any(value <= 0 for value in scales):
            raise ValueError(f"invalid quantization parameters on node {index}")
        result.append(
            QDQEdge(
                node_index=index,
                op_type=op_type,
                input_name=inputs[0],
                output_name=outputs[0],
                params=QuantParams(scales, zero_point, _attribute(node, "axis"), _dtype(zero_dtype, zero_point)),
            )
        )
    return tuple(result)


def qdq_edge_map(model: Any) -> Mapping[str, QDQEdge]:
    """Index extracted QDQ metadata by produced tensor name."""
    return {edge.output_name: edge for edge in extract_qdq_edges(model)}
