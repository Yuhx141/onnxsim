import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

from conv_lowering import plan_conv_gemm  # noqa: E402


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


def test_conv_lowering_extracts_im2col_gemm_and_relu_fusion():
    conv = _node(
        "Conv", ["x", "w", "b"], ["co"], (("strides", (1, 1)), ("pads", (1, 1, 1, 1)))
    )
    relu = _node("Relu", ["co"], ["y"])
    model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[conv, relu],
            input=[_value("x", (1, 64, 56, 56))],
            value_info=[_value("co", (1, 128, 56, 56))],
            output=[_value("y", (1, 128, 56, 56))],
            initializer=[
                SimpleNamespace(name="w", dims=(128, 64, 3, 3)),
                SimpleNamespace(name="b", dims=(128,)),
            ],
        )
    )
    plan = plan_conv_gemm(model, 0, columns=8)
    assert plan.gemm_shape == (3136, 576, 128)
    assert plan.groups == 1
    assert plan.has_bias is True
    assert plan.fused_relu is True
    assert plan.kernel == "conv_im2col_gemm_int8_relu"


def test_grouped_conv_reduces_gemm_n_per_group():
    conv = _node("Conv", ["x", "w"], ["y"], (("group", (4,)),))
    model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[conv],
            input=[_value("x", (1, 32, 8, 8))],
            value_info=[],
            output=[_value("y", (1, 64, 8, 8))],
            initializer=[SimpleNamespace(name="w", dims=(64, 8, 1, 1))],
        )
    )
    plan = plan_conv_gemm(model, 0, columns=1)
    assert plan.groups == 4
    assert plan.gemm_shape == (64, 8, 16)
