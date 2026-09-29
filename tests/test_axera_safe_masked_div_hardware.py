"""AX8850 check for the zero-count-safe masked normalization template."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.join(HERE, "..", "scripts", "axera")
sys.path.insert(0, AXERA)

import axcl_session  # noqa: E402
import pulsar2_docker  # noqa: E402

IMAGE = "pulsar2:7.0-lite"
SHAPES = ((1024, 9, 3136), (1024, 1, 3136))
pytestmark = pytest.mark.skipif(
    not pulsar2_docker.docker_image_available(IMAGE)
    or not pulsar2_docker.axcl_available(),
    reason="needs Pulsar2 7.0-lite and an AXCL device",
)


def test_safe_masked_div_template_handles_empty_rows(tmp_path):
    work = str(tmp_path)
    os.makedirs(os.path.join(work, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(work, "config"), exist_ok=True)
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPES[0])
    d_info = helper.make_tensor_value_info("count", TensorProto.FLOAT, SHAPES[1])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, SHAPES[0])
    one = numpy_helper.from_array(np.array([1.0], np.float32), "one")
    full_shape = numpy_helper.from_array(np.array(SHAPES[0], np.int64), "full_shape")
    graph = helper.make_graph(
        [
            helper.make_node("Expand", ["count", "full_shape"], ["expanded_count"]),
            helper.make_node("Max", ["expanded_count", "one"], ["safe_count"]),
            helper.make_node("Div", ["x", "safe_count"], ["y"]),
        ],
        "safe_masked_div",
        [x_info, d_info],
        [y_info],
        [one, full_shape],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(work, "safe_div.onnx"))

    x_calib = np.zeros(SHAPES[0], np.float32)
    d_calib = np.ones(SHAPES[1], np.float32)
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work, "dataset", "x.tar"), [x_calib]
    )
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work, "dataset", "count.tar"), [d_calib]
    )
    config = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": name,
                    "calibration_dataset": f"./dataset/{name}.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 1,
                }
                for name in ("x", "count")
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
            "layer_configs": [
                {"op_types": ["Expand", "Max", "Div"], "data_type": "FP32"}
            ],
        },
        "compiler": {"check": 0},
    }
    with open(os.path.join(work, "config", "safe_div.json"), "w") as stream:
        json.dump(config, stream)
    built = pulsar2_docker.build(
        work,
        "safe_div.onnx",
        "out",
        config_path="config/safe_div.json",
        target_hardware="AX650",
        image=IMAGE,
        timeout=1200,
    )
    assert built.success, built.error

    rng = np.random.default_rng(8850)
    x = rng.choice(np.array([0.0, 1.0], np.float32), size=SHAPES[0])
    count = rng.integers(0, 10, size=SHAPES[1]).astype(np.float32)
    expected = x / np.maximum(count, np.float32(1.0))
    with axcl_session.AXSession(subdir="safe_masked_div") as session:
        device_model = session.load(built.axmodel_path)
        try:
            assert [value.dtype for value in device_model.inputs] == [
                np.dtype(np.float32),
                np.dtype(np.float32),
            ]
            (actual,) = session.run(device_model, [x, count])
        finally:
            session.unload(device_model)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
