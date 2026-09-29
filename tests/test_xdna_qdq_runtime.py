import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

from qdq_runtime import extract_qdq_edges, qdq_edge_map  # noqa: E402


def _node(op, inputs, outputs, axis=None):
    attrs = (
        [] if axis is None else [SimpleNamespace(name="axis", ints=(), i=axis, s=b"")]
    )
    return SimpleNamespace(op_type=op, input=inputs, output=outputs, attribute=attrs)


def test_extracts_scalar_and_per_channel_qdq_metadata():
    model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[
                _node("QuantizeLinear", ["x", "s", "z"], ["xq"]),
                _node("DequantizeLinear", ["wq", "ws", "wz"], ["w"], axis=0),
            ],
            initializer=[
                SimpleNamespace(name="s", values=(0.25,), dtype="float32"),
                SimpleNamespace(name="z", values=(128,), dtype="uint8"),
                SimpleNamespace(name="ws", values=(0.1, 0.2), dtype="float32"),
                SimpleNamespace(name="wz", values=(0, 0), dtype="int8"),
            ],
        )
    )
    edges = extract_qdq_edges(model)
    assert edges[0].params.scalar is True
    assert edges[0].params.dtype == "u8"
    assert edges[1].params.scalar is False
    assert edges[1].params.axis == 0
    assert qdq_edge_map(model)["w"].params.scale == (0.1, 0.2)
