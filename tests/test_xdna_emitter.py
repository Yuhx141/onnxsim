import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

from resnet_codegen import build_codegen_plan  # noqa: E402
from resnet_emitter import emit_kernel_specs, render_build_manifest  # noqa: E402


def _value(name, shape):
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(
                shape=SimpleNamespace(
                    dim=[SimpleNamespace(dim_value=value) for value in shape]
                )
            )
        ),
    )


def _node(op, inputs, outputs, attrs=()):
    return SimpleNamespace(
        op_type=op,
        input=inputs,
        output=outputs,
        attribute=[
            SimpleNamespace(name=k, ints=tuple(v), i=0, s=b"") for k, v in attrs
        ],
        name=op,
    )


def test_emitter_deduplicates_and_classifies_conv_artifacts():
    nodes = [
        _node("QuantizeLinear", ["x", "s", "z"], ["xq"]),
        _node("DequantizeLinear", ["xq", "s", "z"], ["xdq"]),
        _node("Conv", ["xdq", "w", "b"], ["co"], (("pads", (1, 1, 1, 1)),)),
        _node("Relu", ["co"], ["y"]),
    ]
    model = SimpleNamespace(
        graph=SimpleNamespace(
            node=nodes,
            input=[_value("x", (1, 16, 8, 8))],
            value_info=[_value("co", (1, 32, 8, 8)), _value("y", (1, 32, 8, 8))],
            output=[_value("y", (1, 32, 8, 8))],
            initializer=[
                SimpleNamespace(name="w", dims=(32, 16, 3, 3)),
                SimpleNamespace(name="b", dims=(32,)),
            ],
        )
    )
    plan = build_codegen_plan(model, columns=1, strict=True)
    specs = emit_kernel_specs(plan, columns=1, source="whole_array.py")
    assert len(specs) == 1
    assert specs[0].gemm_shape == (64, 144, 32)
    assert specs[0].compiled_shape == (256, 160, 32)
    assert specs[0].padding == (192, 16, 0)
    assert specs[0].strategy == "native_conv_required"
    assert specs[0].buildable_with_whole_array is False
    assert specs[0].requires_im2col is True
    manifest = render_build_manifest(plan, source="whole_array.py")
    assert manifest["migraphx"] is False
    assert manifest["ort"] is False
    assert manifest["kernels"][0]["fused_relu"] is True
