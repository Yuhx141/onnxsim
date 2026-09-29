import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

from resnet_codegen import build_codegen_plan, codegen_plan_to_dict  # noqa: E402


def _value(name, shape):
    dims = [SimpleNamespace(dim_value=value) for value in shape]
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dims))
        ),
    )


def _node(op, inputs, outputs, attrs=(), name=""):
    return SimpleNamespace(
        op_type=op,
        input=inputs,
        output=outputs,
        attribute=[
            SimpleNamespace(name=key, ints=tuple(value), i=0, s=b"")
            for key, value in attrs
        ],
        name=name,
    )


def test_codegen_binds_conv_fusion_to_gemm_metadata():
    model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[
                _node("QuantizeLinear", ["x", "s", "z"], ["xq"]),
                _node("DequantizeLinear", ["xq", "s", "z"], ["xdq"]),
                _node(
                    "Conv",
                    ["xdq", "w", "b"],
                    ["co"],
                    (("pads", (1, 1, 1, 1)),),
                    "conv1",
                ),
                _node("QuantizeLinear", ["co", "s", "z"], ["cq"]),
                _node("DequantizeLinear", ["cq", "s", "z"], ["cdq"]),
                _node("Relu", ["cdq"], ["y"], name="relu1"),
            ],
            input=[_value("x", (1, 64, 8, 8))],
            value_info=[_value("co", (1, 128, 8, 8))],
            output=[_value("y", (1, 128, 8, 8))],
            initializer=[
                SimpleNamespace(name="w", dims=(128, 64, 3, 3)),
                SimpleNamespace(name="b", dims=(128,)),
            ],
        )
    )
    plan = build_codegen_plan(model, columns=1, strict=True)
    assert plan.estimated_dispatches == 1
    conv = plan.conv_dispatches[0]
    assert conv.kernel_kind == "conv_relu_int8"
    assert conv.gemm_shape == (64, 576, 128)
    assert conv.tile == (32, 32, 32)
    assert codegen_plan_to_dict(plan)["dispatches"][0]["gemm_shape"] == [64, 576, 128]


def test_codegen_strict_mode_reports_unsupported_ops():
    model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[_node("Foo", ["x"], ["y"])],
            input=[SimpleNamespace(name="x")],
            output=[SimpleNamespace(name="y")],
        )
    )
    try:
        build_codegen_plan(model, strict=True)
    except ValueError as exc:
        assert "Foo" in str(exc)
    else:
        raise AssertionError(
            "strict codegen planning must reject unsupported operators"
        )
