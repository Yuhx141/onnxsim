#!/usr/bin/env python3
"""Build small, RV1106-friendly ONNX -> RKNN benchmark models.

The models deliberately use static NCHW shapes, Conv/BatchNormalization/Relu,
depthwise Conv, pointwise Conv, and global average pooling: these are common
RV1106 vision kernels and make a useful first benchmark without downloading a
large application model.  ``onnxsim`` is run before RKNN compilation so the
benchmark exercises the same deployment path users will use for real models.
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np
import onnx
import cv2
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify

from rv1106_compat import check_rv1106


def _install_onnx_mapping_shim() -> None:
    """Keep RKNN-Toolkit2 2.3.2 compatible with ONNX 1.22+."""
    if hasattr(onnx, "mapping"):
        return
    mapping = types.ModuleType("onnx.mapping")
    table = {
        dtype: onnx.helper.tensor_dtype_to_np_dtype(dtype)
        for dtype in onnx.TensorProto.DataType.values()
        if dtype != onnx.TensorProto.UNDEFINED
    }
    mapping.TENSOR_TYPE_TO_NP_TYPE = table
    mapping.NP_TYPE_TO_TENSOR_TYPE = {value: key for key, value in table.items()}
    onnx.mapping = mapping
    sys.modules["onnx.mapping"] = mapping


_install_onnx_mapping_shim()
from rknn.api import RKNN


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value.astype(np.float32), name=name)


def _model(name: str, shape: tuple[int, int, int, int], depthwise: bool) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [shape[0], 32, 1, 1])
    nodes = []
    initializers = []
    if depthwise:
        w0 = np.zeros((16, 3, 3, 3), np.float32)
        w0[:, :, 1, 1] = 0.25
        nodes.append(helper.make_node("Conv", ["input", "w0"], ["stem"],
                                      name="stem", pads=[1, 1, 1, 1], strides=[2, 2]))
        initializers.append(_initializer("w0", w0))
        wd = np.zeros((16, 1, 3, 3), np.float32)
        wd[:, :, 1, 1] = 1.0
        nodes.append(helper.make_node("Conv", ["stem", "wd"], ["dw"], name="depthwise",
                                      pads=[1, 1, 1, 1], group=16))
        initializers.append(_initializer("wd", wd))
        wp = np.zeros((32, 16, 1, 1), np.float32)
        for i in range(32):
            wp[i, i % 16, 0, 0] = 0.5
        nodes.append(helper.make_node("Conv", ["dw", "wp"], ["features"], name="pointwise"))
        initializers.append(_initializer("wp", wp))
    else:
        w0 = np.zeros((16, 3, 3, 3), np.float32)
        w0[:, :, 1, 1] = 0.25
        nodes.append(helper.make_node("Conv", ["input", "w0"], ["conv"], name="conv",
                                      pads=[1, 1, 1, 1]))
        initializers.append(_initializer("w0", w0))
        initializers.extend([
            _initializer("bn_scale", np.ones(16)),
            _initializer("bn_bias", np.zeros(16)),
            _initializer("bn_mean", np.zeros(16)),
            _initializer("bn_var", np.ones(16)),
        ])
        nodes.append(helper.make_node("BatchNormalization",
                                      ["conv", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
                                      ["features"], name="batchnorm", epsilon=1e-5))
        # Add a second convolution to keep the benchmark representative of a
        # small feature extractor rather than a single-kernel microbenchmark.
        w1 = np.zeros((32, 16, 3, 3), np.float32)
        w1[:, :, 1, 1] = 0.125
        initializers.append(_initializer("w1", w1))
        nodes.append(helper.make_node("Conv", ["features", "w1"], ["features2"],
                                      name="conv2", pads=[1, 1, 1, 1], strides=[2, 2]))
        nodes.append(helper.make_node("Relu", ["features2"], ["relu"], name="relu"))
        nodes.append(helper.make_node("GlobalAveragePool", ["relu"], ["output"], name="gap"))
        graph = helper.make_graph(nodes, name, [x], [y], initializer=initializers)
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    nodes.append(helper.make_node("Relu", ["features"], ["relu"], name="relu"))
    nodes.append(helper.make_node("GlobalAveragePool", ["relu"], ["output"], name="gap"))
    graph = helper.make_graph(nodes, name, [x], [y], initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _build(model: onnx.ModelProto, stem: str, output_dir: str,
           onnxsim_quantize: bool = False) -> None:
    raw_path = os.path.join(output_dir, stem + ".onnx")
    simp_path = os.path.join(output_dir, stem + ".simplified.onnx")
    rknn_path = os.path.join(output_dir, stem + ".rv1106.rknn")
    onnx.save(model, raw_path)
    simplified, ok = simplify(model, check_n=0)
    if not ok:
        raise RuntimeError(f"onnxsim could not simplify {stem}")
    onnx.save(simplified, simp_path)
    findings = check_rv1106(simplified)
    errors = [finding for finding in findings if finding.level in {"error", "unknown"}]
    for finding in findings:
        print(f"{finding.level}: {finding.node} ({finding.op_type}): {finding.message}")
    if errors:
        raise RuntimeError(f"RV1106 compatibility check failed for {stem}")
    load_path = simp_path
    if onnxsim_quantize:
        from onnxsim import quantize_static

        q_path = os.path.join(output_dir, stem + ".onnxsim-int8.onnx")
        shape = tuple(dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim)
        calibration_data = [{"input": np.full(shape, (index + 1) / 8.0, np.float32)}
                            for index in range(8)]
        quantized = quantize_static(simplified, calibration_data=calibration_data,
                                    method="minmax", full_graph=True,
                                    op_types_to_exclude=["Relu", "GlobalAveragePool"])
        onnx.save(quantized, q_path)
        load_path = q_path
    rknn = RKNN(verbose=False)
    try:
        config = dict(target_platform="rv1106", mean_values=[[0, 0, 0]],
                      std_values=[[1, 1, 1]])
        if onnxsim_quantize:
            config["optimization_level"] = 3
        _check(rknn.config(**config), "config")
        _check(rknn.load_onnx(model=load_path), "load_onnx")
        dataset = _calibration_dataset(output_dir, stem, tuple(
            dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim
        ))
        _check(rknn.build(do_quantization=not onnxsim_quantize, dataset=dataset), "build")
        _check(rknn.export_rknn(rknn_path), "export_rknn")
    finally:
        rknn.release()
    print(f"{stem}: raw={len(model.graph.node)} nodes, "
          f"simplified={len(simplified.graph.node)} nodes, "
          f"rknn={os.path.getsize(rknn_path) // 1024} KiB")


def _calibration_dataset(
    output_dir: str, stem: str, shape: tuple[int, int, int, int]
) -> str:
    calibration_dir = os.path.join(output_dir, stem + ".calibration")
    os.makedirs(calibration_dir, exist_ok=True)
    paths = []
    for index in range(8):
        # Deterministic low-contrast calibration images keep this smoke model
        # representative without pulling a dataset into the repository.
        image = np.full((shape[2], shape[3], 3), 8 + index * 24, dtype=np.uint8)
        path = os.path.join(calibration_dir, f"{index:02d}.png")
        cv2.imwrite(path, image)
        paths.append(path)
    dataset = os.path.join(output_dir, stem + ".dataset.txt")
    with open(dataset, "w", encoding="utf-8") as f:
        f.write("\n".join(paths) + "\n")
    return dataset


def _check(ret: int, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"rknn {operation} failed: {ret}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="bench/luckfox-rv1106")
    ap.add_argument("--onnxsim-quantize", action="store_true",
                    help="emit QDQ INT8 ONNX with onnxsim before RKNN conversion")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    _build(_model("conv_bn_relu_224", (1, 3, 224, 224), False),
           "conv_bn_relu_224", args.output_dir, args.onnxsim_quantize)
    _build(_model("depthwise_pointwise_112", (1, 3, 112, 112), True),
           "depthwise_pointwise_112", args.output_dir, args.onnxsim_quantize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
