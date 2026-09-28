#!/usr/bin/env python3
"""Benchmark a projection bottleneck with the skip branch on a second column."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnxruntime import InferenceSession

from benchmark_fused_bottleneck import bind_fused_bottleneck
from resnet_bottleneck import plan_bottleneck_blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("block", help="projection block prefix, e.g. /layer1/layer1.0")
    parser.add_argument("xclbin", type=Path)
    parser.add_argument("insts", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    model = onnx.load(args.model)
    block = next((item for item in plan_bottleneck_blocks(model) if item.prefix == args.block), None)
    if block is None or block.skip_conv_index is None:
        parser.error(f"{args.block!r} is not a supported projection bottleneck")
    if not args.xclbin.is_file() or not args.insts.is_file():
        parser.error("compiled xclbin and instruction stream must both exist")
    binding = bind_fused_bottleneck(model, block)

    ref_model = onnx.ModelProto()
    ref_model.CopyFrom(model)
    for name, shape in (
        (binding["input_raw_name"], binding["input_shape"]),
        (binding["output_raw_name"], binding["output_shape"]),
    ):
        ref_model.graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.UINT8, list(shape))
        )
    input_info = model.graph.input[0]
    input_shape = [int(dim.dim_value) or 1 for dim in input_info.type.tensor_type.shape.dim]
    sample = np.random.default_rng(0).random(input_shape, dtype=np.float32)
    expected_session = InferenceSession(ref_model.SerializeToString(), providers=["CPUExecutionProvider"])
    expected_input, expected_output = expected_session.run(
        [binding["input_raw_name"], binding["output_raw_name"]],
        {input_info.name: sample},
    )

    import aie.iron as iron
    from aie.utils import NPUKernel

    input_data = expected_input.transpose(0, 2, 3, 1).copy().view(np.int8).reshape(-1)
    activation = iron.tensor(input_data, dtype=np.int8, device="npu")
    main_parameters = iron.tensor(binding["main_params"], dtype=np.uint8, device="npu")
    skip_parameters = iron.tensor(binding["skip_params"], dtype=np.uint8, device="npu")
    output = iron.zeros(int(np.prod(binding["output_shape"])), dtype=np.int8, device="npu")
    kernel = NPUKernel(str(args.xclbin), str(args.insts))

    def execute():
        kernel(activation, main_parameters, skip_parameters, output)
        raw = output.numpy().view(np.uint8)
        n, c, h, w = binding["output_shape"]
        return raw.reshape(n, h, w, c).transpose(0, 3, 1, 2).copy()

    actual = execute()
    for _ in range(args.warmup):
        actual = execute()
    started = time.perf_counter()
    for _ in range(args.iters):
        actual = execute()
    avg_ms = (time.perf_counter() - started) * 1000.0 / args.iters
    expected = expected_output.astype(np.uint8)
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    print(json.dumps({
        "block": args.block,
        "schedule": "conv1_and_projection_on_separate_columns",
        "xclbin": str(args.xclbin),
        "warmup": args.warmup,
        "iters": args.iters,
        "avg_ms": avg_ms,
        "max_abs_quantized_error": int(delta.max(initial=0)),
        "mismatched_elements": int(np.count_nonzero(delta)),
        "output_shape": list(actual.shape),
    }, indent=2))
    return 0 if not np.any(delta) else 1


if __name__ == "__main__":
    raise SystemExit(main())
