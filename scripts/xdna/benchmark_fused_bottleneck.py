#!/usr/bin/env python3
"""Compile-artifact validation and benchmark for an identity ResNet block."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from qdq_runtime import QDQEdge, qdq_edge_map
from resnet_bottleneck import BottleneckBlockPlan, plan_bottleneck_blocks


def _single_q_params(edge: QDQEdge, label: str) -> tuple[float, int]:
    if not edge.params.scalar:
        raise ValueError(f"{label}: per-channel scales are not supported by this fused kernel")
    return edge.params.scale[0], edge.params.zero_point[0]


def _power_of_two_shift(ratio: float, label: str) -> int:
    if ratio < 1:
        raise ValueError(f"{label}: output scale must be >= accumulator scale")
    shift = round(math.log2(ratio))
    if shift < 0 or shift > 30 or not math.isclose(ratio, 2.0**shift, rel_tol=1e-6):
        raise ValueError(f"{label}: scale ratio {ratio} is not an integer power of two")
    return shift


def _power_of_two_exponent(ratio: float, label: str) -> int:
    exponent = round(math.log2(ratio))
    if not math.isclose(ratio, 2.0**exponent, rel_tol=1e-6) or not -30 <= exponent <= 30:
        raise ValueError(f"{label}: ratio {ratio} is not a supported power of two")
    return exponent


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _quantizer_after(value: str, nodes: list[Any], consumers: dict[str, list[int]], edges: dict[str, QDQEdge]) -> QDQEdge:
    pending = [value]
    visited: set[str] = set()
    while pending:
        current = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for index in consumers.get(current, ()):
            node = nodes[index]
            if node.op_type == "QuantizeLinear":
                return edges[str(node.output[0])]
            if node.op_type == "Relu":
                pending.extend(str(output) for output in node.output)
    raise ValueError(f"no QuantizeLinear follows {value!r} through Relu")


def bind_fused_bottleneck(model: Any, block: BottleneckBlockPlan) -> dict[str, Any]:
    nodes = list(model.graph.node)
    edges = dict(qdq_edge_map(model))
    initializers = {str(item.name): numpy_helper.to_array(item) for item in model.graph.initializer}
    conv_indices = block.main_conv_indices
    conv_nodes = [nodes[index] for index in conv_indices]
    projection = block.skip_conv_index is not None
    input_shape = tuple(int(v) for v in block.conv_plans[0].input_shape)
    output_shape = tuple(int(v) for v in block.conv_plans[2].output_shape)
    batch, channels, height, width = input_shape
    out_batch, output_channels, output_height, output_width = output_shape
    mid_channels = int(block.conv_plans[1].weight_shape[0])
    if batch != 1 or out_batch != 1:
        raise ValueError("fused bottleneck blocks require batch one")
    if not projection and output_shape != input_shape:
        raise ValueError("identity bottleneck blocks require equal input/output shapes")
    if height < 1 or width < 1 or mid_channels < 2 or mid_channels % 2:
        raise ValueError("fused block dimensions require positive H/W and an even inner channel count")
    if (block.conv_plans[0].weight_shape[2:] != (1, 1)
            or block.conv_plans[1].weight_shape[2:] != (3, 3)
            or block.conv_plans[2].weight_shape[2:] != (1, 1)):
        raise ValueError("supported bottleneck kernels are 1x1, 3x3, 1x1")
    conv2_stride = tuple(block.conv_plans[1].stride)
    if (tuple(block.conv_plans[0].stride) != (1, 1)
            or tuple(block.conv_plans[2].stride) != (1, 1)
            or conv2_stride not in {(1, 1), (2, 2)}
            or any(tuple(plan.dilation) != (1, 1) or plan.groups != 1 for plan in block.conv_plans[:3])):
        raise ValueError("supported main path requires unit conv1/conv3 stride, conv2 stride one or two, unit dilation, and group one")
    expected_spatial = tuple(
        (input_shape[2 + axis] + block.conv_plans[1].pads[axis] + block.conv_plans[1].pads[axis + 2] - 3)
        // conv2_stride[axis] + 1
        for axis in range(2)
    )
    if (output_height, output_width) != expected_spatial:
        raise ValueError("conv2 output shape does not match its 3x3 stride/padding formula")
    if tuple(block.conv_plans[1].pads) != (1, 1, 1, 1):
        raise ValueError("the 3x3 bottleneck convolution requires symmetric one-pixel padding")
    if any(tuple(plan.pads) != (0, 0, 0, 0) for plan in (block.conv_plans[0], block.conv_plans[2])):
        raise ValueError("the 1x1 bottleneck convolutions must not use padding")

    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for name in node.input:
            if name:
                consumers.setdefault(str(name), []).append(index)
    quantizers = [
        _quantizer_after(str(node.output[0]), nodes, consumers, edges)
        for node in conv_nodes
    ]
    add_node = nodes[block.add_index]
    final_quantizer = _quantizer_after(str(add_node.output[0]), nodes, consumers, edges)

    input_edge = edges.get(str(conv_nodes[0].input[0]))
    if input_edge is None or input_edge.op_type != "DequantizeLinear":
        raise ValueError("block activation must have a static DequantizeLinear input")
    add_inputs = [edges.get(str(name)) for name in add_node.input]
    skip_node = nodes[block.skip_conv_index] if projection else None
    skip_quantizer = (
        _quantizer_after(str(skip_node.output[0]), nodes, consumers, edges) if skip_node is not None else None
    )
    if projection:
        skip_plan = block.conv_plans[3]
        if (skip_plan.node_index != block.skip_conv_index or skip_plan.input_shape != input_shape
                or skip_plan.output_shape != output_shape or tuple(skip_plan.stride) != conv2_stride
                or tuple(skip_plan.dilation) != (1, 1) or skip_plan.groups != 1
                or skip_plan.weight_shape[2:] != (1, 1) or tuple(skip_plan.pads) != (0, 0, 0, 0)):
            raise ValueError("projection path requires a matching unpadded 1x1 Conv with conv2 stride")
        if not any(edge is not None and edge.op_type == "DequantizeLinear"
                   and edge.input_name == skip_quantizer.output_name for edge in add_inputs):
            raise ValueError("projection residual Add must consume the skip Conv's quantized output")
    elif not any(
        edge is not None and edge.op_type == "DequantizeLinear" and edge.input_name == input_edge.input_name
        for edge in add_inputs
    ):
        raise ValueError("identity residual Add must consume the original block activation")
    if not any(
        edge is not None and edge.op_type == "DequantizeLinear"
        and edge.input_name == quantizers[2].output_name
        for edge in add_inputs
    ):
        raise ValueError("residual Add must consume the third Conv's quantized output")
    input_scale, input_zero = _single_q_params(input_edge, "block input")
    if input_zero != 128 or input_edge.params.dtype != "u8":
        raise ValueError("the fused block currently requires uint8 activations with zero point 128")

    activation_scales = [input_scale]
    weight_arrays = []
    bias_accumulators = []
    shifts = []
    for stage, node in enumerate(conv_nodes):
        activation_edge = edges.get(str(node.input[0]))
        weight_edge = edges.get(str(node.input[1]))
        bias_edge = edges.get(str(node.input[2])) if len(node.input) > 2 else None
        if activation_edge is None or weight_edge is None or bias_edge is None:
            raise ValueError(f"{node.name}: expected static QDQ activation, weight, and bias")
        act_scale, act_zero = _single_q_params(activation_edge, f"{node.name} activation")
        weight_scale, weight_zero = _single_q_params(weight_edge, f"{node.name} weight")
        bias_scale, bias_zero = _single_q_params(bias_edge, f"{node.name} bias")
        output_scale, output_zero = _single_q_params(quantizers[stage], f"{node.name} output")
        if act_zero != 128 or weight_zero != 0 or bias_zero != 0 or output_zero != 128:
            raise ValueError(f"{node.name}: unsupported activation/weight/bias zero points")
        if weight_edge.params.dtype != "i8" or bias_edge.params.dtype != "i8":
            raise ValueError(f"{node.name}: weight and bias tensors must be signed int8")
        if len(node.input) < 3 or bias_edge.input_name not in initializers:
            raise ValueError(f"{node.name}: bias must be a static quantized initializer")
        if weight_edge.input_name not in initializers:
            raise ValueError(f"{node.name}: weight must be a static quantized initializer")
        weight = np.asarray(initializers[weight_edge.input_name], dtype=np.int8)
        bias_raw = np.asarray(initializers[bias_edge.input_name], dtype=np.int32).reshape(-1)
        if bias_raw.size != weight.shape[0]:
            raise ValueError(f"{node.name}: bias length does not match output channels")
        product_scale = act_scale * weight_scale
        accum_bias = bias_raw.astype(np.float64) * bias_scale / product_scale
        rounded_bias = np.rint(accum_bias)
        if not np.allclose(accum_bias, rounded_bias, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{node.name}: bias cannot be represented as an exact integer accumulator")
        if np.any(rounded_bias < np.iinfo(np.int32).min) or np.any(rounded_bias > np.iinfo(np.int32).max):
            raise ValueError(f"{node.name}: bias overflows int32 accumulator")
        shifts.append(_power_of_two_shift(output_scale / product_scale, f"{node.name} requantization"))
        weight_arrays.append(weight)
        bias_accumulators.append(rounded_bias.astype(np.int32))
        activation_scales.append(output_scale)

    skip_weight = skip_bias = None
    skip_shift = None
    skip_output_scale = None
    if projection:
        node = skip_node
        activation_edge = edges.get(str(node.input[0]))
        weight_edge = edges.get(str(node.input[1]))
        bias_edge = edges.get(str(node.input[2])) if len(node.input) > 2 else None
        if activation_edge is None or weight_edge is None or bias_edge is None:
            raise ValueError(f"{node.name}: expected static projection QDQ activation, weight, and bias")
        act_scale, act_zero = _single_q_params(activation_edge, f"{node.name} activation")
        weight_scale, weight_zero = _single_q_params(weight_edge, f"{node.name} weight")
        bias_scale, bias_zero = _single_q_params(bias_edge, f"{node.name} bias")
        skip_output_scale, skip_output_zero = _single_q_params(skip_quantizer, f"{node.name} output")
        if (act_zero != input_zero or not math.isclose(act_scale, input_scale, rel_tol=1e-6)
                or activation_edge.input_name != input_edge.input_name
                or weight_zero != 0 or bias_zero != 0 or skip_output_zero != 128
                or activation_edge.params.dtype != "u8" or weight_edge.params.dtype != "i8"
                or bias_edge.params.dtype != "i8"):
            raise ValueError(f"{node.name}: unsupported projection QDQ zero points or dtypes")
        if bias_edge.input_name not in initializers or weight_edge.input_name not in initializers:
            raise ValueError(f"{node.name}: projection weights and bias must be static initializers")
        skip_weight = np.asarray(initializers[weight_edge.input_name], dtype=np.int8)
        bias_raw = np.asarray(initializers[bias_edge.input_name], dtype=np.int32).reshape(-1)
        if bias_raw.size != skip_weight.shape[0] or skip_weight.shape != tuple(skip_plan.weight_shape):
            raise ValueError(f"{node.name}: projection weight/bias shape mismatch")
        product_scale = act_scale * weight_scale
        accum_bias = bias_raw.astype(np.float64) * bias_scale / product_scale
        rounded_bias = np.rint(accum_bias)
        if not np.allclose(accum_bias, rounded_bias, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{node.name}: projection bias cannot be represented as an exact integer accumulator")
        if np.any(rounded_bias < np.iinfo(np.int32).min) or np.any(rounded_bias > np.iinfo(np.int32).max):
            raise ValueError(f"{node.name}: projection bias overflows int32 accumulator")
        skip_bias = rounded_bias.astype(np.int32)
        skip_shift = _power_of_two_shift(skip_output_scale / product_scale, f"{node.name} requantization")

    final_scale, final_zero = _single_q_params(final_quantizer, "residual output")
    conv3_scale, conv3_zero = _single_q_params(quantizers[2], "conv3 output")
    if final_zero != 128 or conv3_zero != 128:
        raise ValueError("the fused residual requires uint8 output zero point 128")
    residual_shift = _power_of_two_exponent(conv3_scale / final_scale, "residual branch scale ratio")
    input_shift = _power_of_two_exponent(input_scale / final_scale, "identity branch scale ratio")
    skip_residual_shift = (
        _power_of_two_exponent(skip_output_scale / final_scale, "projection branch scale ratio")
        if projection else input_shift
    )
    if max(abs(residual_shift), abs(skip_residual_shift), abs(input_shift)) > 8:
        raise ValueError("residual scale exponents outside [-8, 8] are not supported")
    final_dequantizer = next(
        (
            edge for edge in edges.values()
            if edge.op_type == "DequantizeLinear" and edge.input_name == final_quantizer.output_name
        ),
        None,
    )
    if final_dequantizer is None:
        raise ValueError("fused block output has no matching DequantizeLinear boundary")

    # Pack output-channel chunks independently so each transfer fits an NPU2
    # DMA descriptor. Each worker reuses one weight FIFO for its whole stage.
    w1, w2, w3 = weight_arrays
    b1, b2, b3 = bias_accumulators
    # Keep per-core working sets below the 64 KiB AIE tile memory while
    # leaving enough room under the shim's 16 simultaneously live BDs.
    # The producer tile owns the input, Conv1 staging output, skip output,
    # and one weight slot. Bound the slot by the remaining 64 KiB tile memory
    # after reserving the default 4 KiB worker stack.
    tile_memory_bytes = 65536
    worker_stack_bytes = 4096
    live_tensor_bytes = (
        int(np.prod(input_shape))
        + int(np.prod((1, mid_channels, height, width)))
        + (int(np.prod(output_shape)) if projection else 0)
    )
    max_chunk_bytes = min(36864, tile_memory_bytes - worker_stack_bytes - live_tensor_bytes)
    if max_chunk_bytes <= 0:
        raise ValueError(
            f"{block.prefix}: full-tensor producer buffers need {live_tensor_bytes} bytes; "
            f"the 64 KiB tile has no remaining space for its stack and weight slot"
        )

    def choose_chunks(outputs: int, weight_bytes_per_output: int) -> int:
        for count in range(1, outputs + 1):
            if outputs % count == 0:
                rows = outputs // count
                if _align4(rows * weight_bytes_per_output) + rows * 4 <= max_chunk_bytes:
                    return count
        raise ValueError("could not divide weight stage into DMA-sized output-channel chunks")

    c1_chunks = choose_chunks(w1.shape[0], w1.shape[1])
    skip_chunks = choose_chunks(skip_weight.shape[0], int(np.prod(skip_weight.shape[1:]))) if projection else 0
    c2_worker_outputs = mid_channels // 2
    c2_chunks = choose_chunks(c2_worker_outputs, w2.shape[1] * w2.shape[2] * w2.shape[3])
    c3_chunks = choose_chunks(w3.shape[0], w3.shape[1])
    c1_rows, c2_rows, c3_rows = w1.shape[0] // c1_chunks, c2_worker_outputs // c2_chunks, w3.shape[0] // c3_chunks

    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(c1_chunks):
        sl = slice(index * c1_rows, (index + 1) * c1_rows)
        chunks.append((w1[sl], b1[sl]))
    skip_chunk_start = len(chunks)
    if projection:
        skip_rows = skip_weight.shape[0] // skip_chunks
        for index in range(skip_chunks):
            sl = slice(index * skip_rows, (index + 1) * skip_rows)
            chunks.append((skip_weight[sl], skip_bias[sl]))
    else:
        skip_rows = 0
    for worker in range(2):
        for index in range(c2_chunks):
            start = worker * c2_worker_outputs + index * c2_rows
            sl = slice(start, start + c2_rows)
            chunks.append((w2[sl], b2[sl]))
    for index in range(c3_chunks):
        sl = slice(index * c3_rows, (index + 1) * c3_rows)
        chunks.append((w3[sl], b3[sl]))
    chunk_sizes = []
    offsets = []
    packed_chunks = []
    for weight, bias in chunks:
        raw_weight = np.ascontiguousarray(weight).view(np.uint8).reshape(-1)
        bias_offset = _align4(raw_weight.size)
        packed = np.zeros(bias_offset + bias.nbytes, dtype=np.uint8)
        packed[:raw_weight.size] = raw_weight
        packed[bias_offset:] = bias.view(np.uint8)
        chunk_sizes.append(packed.size)
        packed_chunks.append(packed)
    slot_bytes = max(chunk_sizes)
    # ObjectFifo fills transfer a complete object. Pad every packed chunk to
    # the common slot size while keeping its bias at the unpadded weight end.
    params = np.zeros(len(packed_chunks) * slot_bytes, dtype=np.uint8)
    offsets = []
    for index, packed in enumerate(packed_chunks):
        offset = index * slot_bytes
        offsets.append(offset)
        params[offset : offset + packed.size] = packed

    q_type = "u8"
    q_nodes = [edges[edge.output_name] for edge in (*quantizers, final_quantizer)]
    if any(edge.params.dtype != q_type for edge in q_nodes):
        raise ValueError("all fused activation boundaries must use uint8")
    input_raw_name = input_edge.input_name
    output_raw_name = final_quantizer.output_name
    covered = tuple(
        index for index, node in enumerate(nodes)
        if block.prefix in str(node.name) and min(conv_indices) <= index <= block.add_index + 3
    )
    return {
        "block": block,
        "params": params,
        "shifts": tuple(shifts),
        "input_raw_name": input_raw_name,
        "input_zero_point": input_zero,
        "output_raw_name": output_raw_name,
        "output_dequant_name": final_dequantizer.output_name,
        "output_scale": final_scale,
        "output_zero_point": final_zero,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "covered_nodes": covered,
        "quantizers": quantizers,
        "final_quantizer": final_quantizer,
        "chunk_counts": (c1_chunks, c2_chunks, c3_chunks),
        "chunk_sizes": tuple(chunk_sizes),
        "chunk_offsets": tuple(offsets),
        "chunk_slot_bytes": slot_bytes,
        "chunk_output_counts": (c1_rows, c2_rows, c3_rows),
        "residual_shift": residual_shift,
        "input_shift": input_shift,
        "projection": projection,
        "skip_chunk_count": skip_chunks,
        "skip_chunk_rows": skip_rows,
        "skip_chunk_sizes": tuple(chunk_sizes[skip_chunk_start:skip_chunk_start + skip_chunks]),
        "skip_chunk_offsets": tuple(offsets[skip_chunk_start:skip_chunk_start + skip_chunks]),
        "skip_output_shift": skip_shift,
        "main_residual_shift": residual_shift,
        "skip_residual_shift": skip_residual_shift,
        "input_height": height,
        "input_width": width,
        "input_channels": channels,
        "output_height": output_height,
        "output_width": output_width,
        "output_channels": output_channels,
        "conv2_stride": conv2_stride,
        "skip_output_scale": skip_output_scale,
        "skip_output_zero_point": 128 if projection else None,
        "parameter_stage_order": ("conv1", "skip", "conv2a", "conv2b", "conv3") if projection else ("conv1", "conv2a", "conv2b", "conv3"),
    }


def _resolve_type(dtype: str) -> int:
    from onnx import TensorProto
    return TensorProto.UINT8 if dtype == "u8" else TensorProto.INT8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("block", help="ONNX node-name prefix, e.g. /layer1/layer1.1")
    parser.add_argument("xclbin", type=Path)
    parser.add_argument("insts", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    model = onnx.load(args.model)
    block = next((item for item in plan_bottleneck_blocks(model) if item.prefix == args.block), None)
    if block is None:
        parser.error(f"no ResNet bottleneck found for prefix {args.block!r}")
    binding = bind_fused_bottleneck(model, block)
    if not args.xclbin.is_file() or not args.insts.is_file():
        parser.error("compiled xclbin and instruction stream must both exist")

    from aie.utils import NPUKernel
    import aie.iron as iron

    input_shape = tuple(int(v) for v in binding["input_shape"])
    output_shape = tuple(int(v) for v in binding["output_shape"])
    input_channels = input_shape[1]
    params = binding["params"]
    parameters = iron.tensor(params, dtype=np.uint8, device="npu")
    result = iron.zeros(input_shape[0] * output_shape[1] * output_shape[2] * output_shape[3], dtype=np.int8, device="npu")
    kernel = NPUKernel(str(args.xclbin), str(args.insts))

    # Add the raw quantized tensors at the block boundary to an ORT copy so
    # both the NPU input and expected output come from the same graph input.
    from onnx import TensorProto
    from onnxruntime import InferenceSession

    ref_model = onnx.ModelProto()
    ref_model.CopyFrom(model)
    for name, shape in ((binding["input_raw_name"], input_shape), (binding["output_raw_name"], output_shape)):
        ref_model.graph.output.append(
            helper.make_tensor_value_info(name, _resolve_type("u8"), list(shape))
        )
    sample_shape = [int(dim.dim_value) or 1 for dim in model.graph.input[0].type.tensor_type.shape.dim]
    sample = np.random.default_rng(0).random(sample_shape, dtype=np.float32)
    expected_session = InferenceSession(ref_model.SerializeToString(), providers=["CPUExecutionProvider"])
    expected_input, expected_output = expected_session.run(
        [binding["input_raw_name"], binding["output_raw_name"]],
        {model.graph.input[0].name: sample},
    )
    edge = qdq_edge_map(model)[next(
        str(node.input[0]) for node in model.graph.node
        if node.op_type == "Conv" and node.name == f"{args.block}/conv1/Conv"
    )]
    zero = edge.params.zero_point[0]
    x_hwc = (expected_input.astype(np.int16) - zero).astype(np.int8).transpose(0, 2, 3, 1).copy()
    x_tensor = iron.tensor(x_hwc.reshape(-1), dtype=np.int8, device="npu")

    def execute() -> np.ndarray:
        kernel(x_tensor, parameters, result)
        return result.numpy().view(np.uint8).reshape(output_shape[0], output_shape[2], output_shape[3], output_shape[1]).transpose(0, 3, 1, 2).copy()

    actual = execute()
    for _ in range(args.warmup):
        actual = execute()
    start = time.perf_counter()
    for _ in range(args.iters):
        actual = execute()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / args.iters
    delta = np.abs(actual.astype(np.int16) - expected_output.astype(np.int16))
    report = {
        "block": args.block,
        "mode": "one_xdna_dispatch_three_convs_qdq_residual_relu",
        "covered_node_indices": list(binding["covered_nodes"]),
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "requantization_shifts": list(binding["shifts"]),
        "latency_ms": elapsed_ms,
        "iterations": args.iters,
        "max_abs_quantized_error": int(delta.max(initial=0)),
        "exact_match": bool(np.array_equal(actual, expected_output)),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["exact_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
