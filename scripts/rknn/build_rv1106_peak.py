#!/usr/bin/env python3
"""Build dense INT8 convolution workloads for RV1106 throughput measurement."""

from __future__ import annotations

import argparse
import os
import sys
import types

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify


def _install_onnx_mapping_shim() -> None:
    if hasattr(onnx, "mapping"):
        return
    mapping = types.ModuleType("onnx.mapping")
    table = {dtype: onnx.helper.tensor_dtype_to_np_dtype(dtype)
             for dtype in onnx.TensorProto.DataType.values()
             if dtype != onnx.TensorProto.UNDEFINED}
    mapping.TENSOR_TYPE_TO_NP_TYPE = table
    mapping.NP_TYPE_TO_TENSOR_TYPE = {value: key for key, value in table.items()}
    onnx.mapping = mapping
    sys.modules["onnx.mapping"] = mapping


_install_onnx_mapping_shim()
from rknn.api import RKNN


def _weight(name: str, shape: tuple[int, ...], seed: int) -> onnx.TensorProto:
    rng = np.random.default_rng(seed)
    value = rng.normal(0.0, 0.03, size=shape).astype(np.float32)
    return numpy_helper.from_array(value, name=name)


def _model(channels: int, layers: int, size: int) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                      [1, channels, size, size])
    nodes = []
    initializers = []
    current = "input"
    for index in range(layers):
        in_channels = 3 if index == 0 else channels
        out = f"conv_{index}"
        weight_name = f"weight_{index}"
        nodes.append(helper.make_node(
            "Conv", [current, weight_name], [out], name=f"conv_{index}",
            pads=[1, 1, 1, 1], strides=[1, 1], dilations=[1, 1], group=1))
        initializers.append(_weight(weight_name, (channels, in_channels, 3, 3), index))
        current = out
    nodes.append(helper.make_node("Relu", [current], ["output"], name="relu"))
    graph = helper.make_graph(nodes, f"dense_c{channels}_l{layers}_{size}", [x], [y],
                              initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _dataset(directory: str, name: str, size: int) -> str:
    paths = []
    calibration = os.path.join(directory, name + ".calibration")
    os.makedirs(calibration, exist_ok=True)
    for index in range(8):
        image = np.full((size, size, 3), 8 + index * 24, dtype=np.uint8)
        path = os.path.join(calibration, f"{index:02d}.png")
        cv2.imwrite(path, image)
        paths.append(path)
    dataset = os.path.join(directory, name + ".dataset.txt")
    with open(dataset, "w", encoding="utf-8") as stream:
        stream.write("\n".join(paths) + "\n")
    return dataset


def _check(value: int, operation: str) -> None:
    if value != 0:
        raise RuntimeError(f"rknn {operation} failed: {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/luckfox-rv1106-peak")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--quantized-dtype", default="w8a8",
                        choices=("w8a8", "w4a16"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    name = f"dense_c{args.channels}_l{args.layers}_{args.size}"
    model = _model(args.channels, args.layers, args.size)
    raw = os.path.join(args.output_dir, name + ".onnx")
    simplified_path = os.path.join(args.output_dir, name + ".simplified.onnx")
    rknn_path = os.path.join(args.output_dir, name + ".rv1106.rknn")
    onnx.save(model, raw)
    simplified, ok = simplify(model, check_n=0)
    if not ok:
        raise RuntimeError("onnxsim simplification failed")
    onnx.save(simplified, simplified_path)
    rknn = RKNN(verbose=False)
    try:
        _check(rknn.config(target_platform="rv1106", mean_values=[[0, 0, 0]],
                           std_values=[[1, 1, 1]],
                           quantized_dtype=args.quantized_dtype), "config")
        _check(rknn.load_onnx(model=simplified_path), "load_onnx")
        _check(rknn.build(do_quantization=True,
                          dataset=_dataset(args.output_dir, name, args.size)), "build")
        _check(rknn.export_rknn(rknn_path), "export_rknn")
    finally:
        rknn.release()
    macs = (3 * args.channels * 9 * args.size * args.size +
            max(args.layers - 1, 0) * args.channels * args.channels * 9 * args.size * args.size)
    print(f"model={rknn_path} layers={args.layers} channels={args.channels} "
          f"size={args.size} macs={macs} ops={macs * 2} "
          f"rknn_bytes={os.path.getsize(rknn_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
