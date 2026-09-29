import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

from resnet_plan import plan_qdq_resnet, plan_to_dict  # noqa: E402


def _node(op, inputs, outputs, name=""):
    return SimpleNamespace(op_type=op, input=inputs, output=outputs, name=name)


def _model(nodes):
    return SimpleNamespace(
        graph=SimpleNamespace(
            node=nodes,
            input=[SimpleNamespace(name="x")],
            output=[SimpleNamespace(name="y")],
        )
    )


def test_qdq_edges_are_not_standalone_dispatches_and_conv_relu_fuses():
    model = _model(
        [
            _node("QuantizeLinear", ["x", "xs", "xz"], ["xq"]),
            _node("DequantizeLinear", ["xq", "xs", "xz"], ["xdq"]),
            _node("Conv", ["xdq", "w", "b"], ["co"]),
            _node("QuantizeLinear", ["co", "ys", "yz"], ["cq"]),
            _node("DequantizeLinear", ["cq", "ys", "yz"], ["cdq"]),
            _node("Relu", ["cdq"], ["ro"]),
            _node("QuantizeLinear", ["ro", "os", "oz"], ["oq"]),
            _node("DequantizeLinear", ["oq", "os", "oz"], ["y"]),
        ]
    )
    plan = plan_qdq_resnet(model)
    assert plan.op_counts["Conv"] == 1
    assert plan.internal_qdq_nodes == 4
    assert len(plan.boundary_qdq_nodes) == 2
    assert any(group.kernel_kind == "conv_relu_int8" for group in plan.fusions)
    assert plan.unsupported_ops == ()
    assert plan_to_dict(plan)["estimated_dispatches"] == plan.estimated_dispatches


def test_unknown_semantic_op_is_reported():
    model = _model(
        [_node("QuantizeLinear", ["x", "s", "z"], ["q"]), _node("Foo", ["q"], ["y"])]
    )
    plan = plan_qdq_resnet(model)
    assert plan.unsupported_ops == ("Foo",)
