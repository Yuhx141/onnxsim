import json
import os
import sys

import numpy as np
import onnx
import pytest

_AXERA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "axera")
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import graph_generator  # noqa: E402


def _model(nodes, inputs, outputs, initializers=()):
    graph = onnx.helper.make_graph(nodes, "source", inputs, outputs, initializers)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )


def _full_model():
    f = onnx.helper.make_tensor("indices", onnx.TensorProto.INT64, [8], list(range(8)))
    nodes = [
        onnx.helper.make_node("Gather", ["x", "indices"], ["g"], axis=3),
        onnx.helper.make_node("Reshape", ["g", "shape"], ["r"]),
        onnx.helper.make_node("MatMul", ["r", "w"], ["m"]),
        onnx.helper.make_node("Transpose", ["m"], ["t"], perm=[0, 1, 3, 2]),
        onnx.helper.make_node("Add", ["t", "b"], ["y"]),
    ]
    shape = onnx.helper.make_tensor("shape", onnx.TensorProto.INT64, [4], [1, 1, 8, 4])
    inputs = [
        onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 4, 16]),
        onnx.helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, [4, 6]),
        onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [1, 1, 6, 8]),
    ]
    return _model(
        nodes,
        inputs,
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1, 6, 8])],
        [f, shape],
    )


def test_schedule_fuses_measured_chain():
    plan = graph_generator.schedule_graph(_full_model())
    assert plan.chain == "gather_reshape_matmul_transpose_add"
    assert plan.segments[0].inputs == ("x", "w", "b")


def test_schedule_arbitrary_matmul_carries_both_operand_shapes():
    model = _model(
        [onnx.helper.make_node("MatMul", ["x", "z"], ["y"])],
        [
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3, 4]),
            onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [4, 5]),
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3, 5])],
    )
    plan = graph_generator.schedule_graph(model)
    assert plan.chain == "matmul"
    assert plan.segments[0].input_shape == (2, 3, 4)
    assert plan.segments[0].operand_shape == (4, 5)


def test_generate_normalizes_initializer_first_commutative_binary(tmp_path):
    shape = (1, 64)
    constant = onnx.numpy_helper.from_array(np.array(0.2, dtype="float32"), "z")
    model = _model(
        [onnx.helper.make_node("Mul", ["z", "x"], ["y"])],
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        [constant],
    )
    _, meta = graph_generator.binary_op_scale_emit.load_template(
        "Mul", shape, {"x": 0, "y": 0, "z": 0}
    )
    source = tmp_path / "constant_first.onnx"
    output = tmp_path / "constant_first.axmodel"
    schedule = tmp_path / "constant_first.schedule.json"
    onnx.save(model, source)
    plan = graph_generator.generate(
        str(source),
        str(output),
        schedule_path=str(schedule),
        calibration={"scales": meta["scales"], "zero_points": meta["zero_points"]},
    )
    assert plan.chain == "mul"
    scheduled = json.loads(schedule.read_text())
    assert [item["name"] for item in scheduled["inputs"]] == ["x", "z"]
    assert scheduled["kernels"][0]["inputs"] == ["z", "x"]


def test_generate_uses_one_fused_template(tmp_path):
    source = tmp_path / "source.onnx"
    output = tmp_path / "generated.axmodel"
    schedule = tmp_path / "generated.schedule.json"
    onnx.save(_full_model(), source)
    plan = graph_generator.generate(
        str(source),
        str(output),
        indices=[15, 0, 7, 7, 3, 12, 1, 14],
        schedule_path=str(schedule),
    )
    assert plan.chain == "gather_reshape_matmul_transpose_add"
    generated = onnx.load(str(output), load_external_data=False)
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert [item.name for item in generated.graph.input] == ["x", "w", "b"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == plan.chain


def test_schedule_and_generate_fused_reshape_relu(tmp_path):
    shape = onnx.helper.make_tensor("shape", onnx.TensorProto.INT64, [4], [1, 1, 8, 16])
    nodes = [
        onnx.helper.make_node("Reshape", ["x", "shape"], ["r"]),
        onnx.helper.make_node("Relu", ["r"], ["y"]),
    ]
    source_model = _model(
        nodes,
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 8, 4, 4])],
        [
            onnx.helper.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, [1, 1, 8, 16]
            )
        ],
        [shape],
    )
    source = tmp_path / "reshape.onnx"
    output = tmp_path / "reshape.axmodel"
    onnx.save(source_model, source)
    plan = graph_generator.generate(str(source), str(output))
    assert plan.chain == "reshape_relu"
    generated = onnx.load(str(output), load_external_data=False)
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert [
        dim.dim_value for dim in generated.graph.input[0].type.tensor_type.shape.dim
    ] == [1, 8, 4, 4]


def test_schedule_refuses_unmeasured_transpose():
    model = _full_model()
    model.graph.node[3].attribute[0].ints[:] = [0, 2, 1, 3]
    with pytest.raises(ValueError, match="Transpose perm"):
        graph_generator.schedule_graph(model)
