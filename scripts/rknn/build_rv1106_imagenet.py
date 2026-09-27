#!/usr/bin/env python3
"""Compile a standard ImageNet ONNX model for the RV1106 benchmark."""

from __future__ import annotations

import argparse
import os
import sys
import types

import cv2
import numpy as np
import onnx
from onnxsim import simplify

from rv1106_compat import check_rv1106


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


def _check(ret: int, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"rknn {operation} failed: {ret}")


def _data_input(model: onnx.ModelProto) -> onnx.ValueInfoProto:
    initializers = {item.name for item in model.graph.initializer}
    for value in model.graph.input:
        if value.name not in initializers:
            return value
    raise RuntimeError("model has no non-initializer input")


def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(dim.dim_value or 1)
    return tuple(dims)


def _dataset(directory: str, name: str, height: int, width: int) -> str:
    calibration = os.path.join(directory, name + ".calibration")
    os.makedirs(calibration, exist_ok=True)
    paths = []
    for index in range(8):
        image = np.full((height, width, 3), 8 + index * 24, dtype=np.uint8)
        path = os.path.join(calibration, f"{index:02d}.png")
        cv2.imwrite(path, image)
        paths.append(path)
    dataset = os.path.join(directory, name + ".dataset.txt")
    with open(dataset, "w", encoding="utf-8") as stream:
        stream.write("\n".join(paths) + "\n")
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output-dir", default="/tmp/luckfox-rv1106-imagenet")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.model))[0]
    model = onnx.load(args.model)
    data_input = _data_input(model)
    input_shape = _shape(data_input)
    if len(input_shape) != 4:
        raise RuntimeError(f"expected NCHW input, got {input_shape}")
    simplified, ok = simplify(model, check_n=0)
    if not ok:
        raise RuntimeError("onnxsim simplification failed")
    findings = check_rv1106(simplified)
    for finding in findings:
        print(f"{finding.level}: {finding.node} ({finding.op_type}): {finding.message}")
    if any(finding.level in {"error", "unknown"} for finding in findings):
        raise RuntimeError("RV1106 compatibility check failed")
    simplified_path = os.path.join(args.output_dir, stem + ".simplified.onnx")
    rknn_path = os.path.join(args.output_dir, stem + ".rv1106.rknn")
    onnx.save(simplified, simplified_path)
    rknn = RKNN(verbose=False)
    try:
        _check(rknn.config(target_platform="rv1106", mean_values=[[0, 0, 0]],
                           std_values=[[1, 1, 1]]), "config")
        _check(rknn.load_onnx(model=simplified_path, inputs=[data_input.name],
                              input_size_list=[list(input_shape)]), "load_onnx")
        _check(rknn.build(do_quantization=True,
                          dataset=_dataset(args.output_dir, stem,
                                           input_shape[-2], input_shape[-1])), "build")
        _check(rknn.export_rknn(rknn_path), "export_rknn")
    finally:
        rknn.release()
    print(f"model={rknn_path} input={input_shape} nodes={len(simplified.graph.node)} "
          f"rknn_bytes={os.path.getsize(rknn_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
