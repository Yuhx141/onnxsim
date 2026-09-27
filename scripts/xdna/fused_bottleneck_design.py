#!/usr/bin/env python3
"""Compile a chunk-streamed fused INT8 identity ResNet bottleneck for XDNA."""
from __future__ import annotations

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifoLink
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL = Path(__file__).with_name("kernels") / "fused_identity_bottleneck.cc"
_SKIP_KERNEL = Path(__file__).with_name("kernels") / "fused_bottleneck_skip.cc"
_IDENTITY_SKIP_KERNEL = Path(__file__).with_name("kernels") / "fused_bottleneck_identity_skip.cc"


def _align4(value: int) -> int:
    return (value + 3) & ~3


@iron.jit
def fused_identity_bottleneck(
    activation: In,
    parameters: In,
    result: Out,
    *,
    width: CompileTime[int] = 8,
    height: CompileTime[int] = 8,
    channels: CompileTime[int] = 256,
    mid_channels: CompileTime[int] = 64,
    shift1: CompileTime[int] = 5,
    shift2: CompileTime[int] = 8,
    shift3: CompileTime[int] = 7,
    residual_shift: CompileTime[int] = 2,
    input_shift: CompileTime[int] = 0,
    chunks1: CompileTime[int] = 1,
    chunks2: CompileTime[int] = 1,
    chunks3: CompileTime[int] = 1,
    output_width: CompileTime[int] = 8,
    output_height: CompileTime[int] = 8,
    output_channels: CompileTime[int] = 256,
    conv2_stride: CompileTime[int] = 1,
    skip_stride: CompileTime[int] = 1,
    skip_chunks: CompileTime[int] = 0,
    skip_shift: CompileTime[int] = 0,
    residual_main_shift: CompileTime[int] = 2,
    residual_skip_shift: CompileTime[int] = 0,
):
    pixels = width * height
    output_pixels = output_width * output_height
    outputs1 = mid_channels // chunks1
    outputs2 = (mid_channels // 2) // chunks2
    outputs3 = output_channels // chunks3
    skip_outputs = output_channels // (skip_chunks if skip_chunks > 0 else 1)
    bytes1 = _align4(outputs1 * channels) + outputs1 * 4
    bytes2 = _align4(outputs2 * mid_channels * 9) + outputs2 * 4
    bytes3 = _align4(outputs3 * mid_channels) + outputs3 * 4
    bytes_skip = _align4(skip_outputs * channels) + skip_outputs * 4 if skip_chunks > 0 else 0

    activation_ty = np.ndarray[(pixels * channels,), np.dtype[np.int8]]
    max_weight_bytes = max(bytes1, bytes2, bytes3, bytes_skip)
    parameter_chunks = chunks1 + skip_chunks + 2 * chunks2 + chunks3
    params_ty = np.ndarray[(parameter_chunks * max_weight_bytes,), np.dtype[np.uint8]]
    weight_ty = np.ndarray[(max_weight_bytes,), np.dtype[np.uint8]]
    stage1_ty = np.ndarray[(pixels * mid_channels,), np.dtype[np.int8]]
    skip_ty = np.ndarray[(output_pixels * output_channels,), np.dtype[np.int8]]
    stage2a_ty = np.ndarray[(output_pixels * (mid_channels // 2),), np.dtype[np.int8]]
    stage2b_ty = np.ndarray[(output_pixels * (mid_channels // 2),), np.dtype[np.int8]]
    stage2_ty = np.ndarray[(output_pixels * (mid_channels + output_channels),), np.dtype[np.int8]]
    output_ty = np.ndarray[(output_pixels * output_channels,), np.dtype[np.int8]]

    flags = [
        f"-DFUSED_W={width}", f"-DFUSED_H={height}", f"-DFUSED_C={channels}",
        f"-DFUSED_OUT_W={output_width}", f"-DFUSED_OUT_H={output_height}", f"-DFUSED_OUT_C={output_channels}",
        f"-DFUSED_CONV2_STRIDE={conv2_stride}", f"-DFUSED_SKIP_STRIDE={skip_stride}",
        f"-DFUSED_PROJECTION={1 if skip_chunks > 0 else 0}", f"-DFUSED_SKIP_CHUNKS={max(skip_chunks, 1)}",
        f"-DFUSED_SKIP_OUTPUTS={skip_outputs}", f"-DFUSED_SKIP_SHIFT={skip_shift}",
        f"-DFUSED_SKIP_BIAS_OFFSET={_align4(skip_outputs * channels)}",
        f"-DFUSED_MID={mid_channels}", f"-DFUSED_SHIFT1={shift1}",
        f"-DFUSED_SHIFT2={shift2}", f"-DFUSED_SHIFT3={shift3}",
        f"-DFUSED_RESIDUAL_SHIFT={residual_shift}", f"-DFUSED_INPUT_SHIFT={input_shift}",
        f"-DFUSED_C1_CHUNKS={chunks1}", f"-DFUSED_C2_CHUNKS={chunks2}",
        f"-DFUSED_C3_CHUNKS={chunks3}", f"-DFUSED_C1_OUTPUTS={outputs1}",
        f"-DFUSED_C2_OUTPUTS={outputs2}", f"-DFUSED_C3_OUTPUTS={outputs3}",
        f"-DFUSED_BIAS1_OFFSET={_align4(outputs1 * channels)}",
        f"-DFUSED_BIAS2_OFFSET={_align4(outputs2 * mid_channels * 9)}",
        f"-DFUSED_BIAS3_OFFSET={_align4(outputs3 * mid_channels)}",
        f"-DFUSED_MAIN_RESIDUAL_SHIFT={residual_main_shift}", f"-DFUSED_SKIP_RESIDUAL_SHIFT={residual_skip_shift}",
    ]
    k1 = ExternalFunction("fused_bottleneck_conv1_chunk", source_file=str(_KERNEL), arg_types=[activation_ty, weight_ty, stage1_ty, np.int32], compile_flags=flags)
    kskip = ExternalFunction("fused_bottleneck_skip_chunk", source_file=str(_SKIP_KERNEL), arg_types=[activation_ty, weight_ty, skip_ty, np.int32], compile_flags=flags)
    kidentity = ExternalFunction("fused_bottleneck_identity_skip", source_file=str(_IDENTITY_SKIP_KERNEL), arg_types=[activation_ty, skip_ty], compile_flags=flags)
    k2 = ExternalFunction("fused_bottleneck_conv2_chunk", source_file=str(_KERNEL), arg_types=[stage1_ty, weight_ty, stage2a_ty, np.int32, np.int32], compile_flags=flags)
    k2b = ExternalFunction("fused_bottleneck_conv2_chunk_b", source_file=str(_KERNEL), arg_types=[stage1_ty, weight_ty, stage2b_ty, np.int32, np.int32], compile_flags=flags)
    k3 = ExternalFunction("fused_bottleneck_conv3_chunk", source_file=str(_KERNEL), arg_types=[stage2_ty, weight_ty, output_ty, np.int32], compile_flags=flags)

    activation_fifo = ObjectFifo(activation_ty, depth=1, name="bottleneck_activation")
    weights_fifo = ObjectFifo(weight_ty, depth=1, name="bottleneck_weight_chunks")
    stage1_fifo = ObjectFifo(stage1_ty, depth=1, name="bottleneck_stage1_bundle")
    skip_fifo = ObjectFifo(skip_ty, depth=1, name="bottleneck_skip")
    stage2a_fifo = ObjectFifo(stage2a_ty, depth=1, name="bottleneck_stage2a")
    stage2b_fifo = ObjectFifo(stage2b_ty, depth=1, name="bottleneck_stage2b")
    stage2_fifo = ObjectFifo(stage2_ty, depth=1, name="bottleneck_stage2_bundle")
    output_fifo = ObjectFifo(output_ty, depth=1, name="bottleneck_output")

    def discard(weights, count):
        for _ in range_(count):
            weights.acquire(1)
            weights.release(1)

    def conv1_worker(inp, weights, out, skip_out, kernel, skip_kernel, identity_kernel):
        x = inp.acquire(1)
        bundle = out.acquire(1)
        for i in range_(chunks1):
            w = weights.acquire(1)
            kernel(x, w, bundle, i)
            weights.release(1)
        residual = skip_out.acquire(1)
        if skip_chunks > 0:
            for i in range_(skip_chunks):
                w = weights.acquire(1)
                skip_kernel(x, w, residual, i)
                weights.release(1)
        else:
            identity_kernel(x, residual)
        skip_out.release(1)
        out.release(1)
        inp.release(1)
        discard(weights, 2 * chunks2 + chunks3)

    def conv2_worker(inp, weights, out, kernel, channel_offset, is_a):
        discard(weights, chunks1 + skip_chunks + (0 if is_a else chunks2))
        bundle = inp.acquire(1)
        output = out.acquire(1)
        for i in range_(chunks2):
            w = weights.acquire(1)
            kernel(bundle, w, output, i, channel_offset)
            weights.release(1)
        out.release(1)
        inp.release(1)
        # Release the stage2 bundle before discarding later chunks: conv3
        # needs this bundle before it can consume its own weight chunks.
        discard(weights, (chunks2 if is_a else 0) + chunks3)

    def conv3_worker(inp, weights, out, kernel):
        discard(weights, chunks1 + skip_chunks + 2 * chunks2)
        bundle = inp.acquire(1)
        output = out.acquire(1)
        for i in range_(chunks3):
            w = weights.acquire(1)
            kernel(bundle, w, output, i)
            weights.release(1)
        out.release(1)
        inp.release(1)

    workers = [
        Worker(conv1_worker, fn_args=[activation_fifo.cons(), weights_fifo.cons(), stage1_fifo.prod(), skip_fifo.prod(), k1, kskip, kidentity], tile=Tile(0, 2), stack_size=0x1000),
        Worker(conv2_worker, fn_args=[stage1_fifo.cons(), weights_fifo.cons(), stage2a_fifo.prod(), k2, 0, True], tile=Tile(0, 3), stack_size=0x1000),
        Worker(conv2_worker, fn_args=[stage1_fifo.cons(), weights_fifo.cons(), stage2b_fifo.prod(), k2b, mid_channels // 2, False], tile=Tile(0, 5), stack_size=0x1000),
        Worker(conv3_worker, fn_args=[stage2_fifo.cons(), weights_fifo.cons(), output_fifo.prod(), k3], tile=Tile(0, 4), stack_size=0x1000),
    ]
    ObjectFifoLink([stage2a_fifo.cons(), stage2b_fifo.cons(), skip_fifo.cons()], stage2_fifo.prod(), src_offsets=[0, output_pixels * (mid_channels // 2), output_pixels * mid_channels])

    def sequence(x, w, y, xprod, weights_prod, ycons):
        group = TaskGroup()
        xprod.fill(x, wait=True, group=group)
        group.finish()
        offset = 0
        for _ in range(parameter_chunks):
            # Complete and free each transfer before configuring the next BD.
            group = TaskGroup()
            weights_prod.fill(w, wait=True, sizes=[max_weight_bytes], strides=[1], offset=offset, transfer_len=max_weight_bytes, group=group)
            group.finish()
            offset += max_weight_bytes
        group = TaskGroup()
        ycons.drain(y, wait=True, group=group)
        group.finish()

    runtime = Runtime(sequence, [activation_ty, params_ty, output_ty, activation_fifo.prod(), weights_fifo.prod(), output_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--mid-channels", type=int, default=64)
    parser.add_argument("--shift1", type=int, default=5)
    parser.add_argument("--shift2", type=int, default=8)
    parser.add_argument("--shift3", type=int, default=7)
    parser.add_argument("--residual-shift", type=int, default=2)
    parser.add_argument("--input-shift", type=int, default=0)
    parser.add_argument("--output-width", type=int, default=8)
    parser.add_argument("--output-height", type=int, default=8)
    parser.add_argument("--output-channels", type=int, default=256)
    parser.add_argument("--conv2-stride", type=int, default=1)
    parser.add_argument("--skip-stride", type=int, default=1)
    parser.add_argument("--skip-chunks", type=int, default=0)
    parser.add_argument("--skip-shift", type=int, default=0)
    parser.add_argument("--chunks1", type=int, default=1)
    parser.add_argument("--chunks2", type=int, default=1)
    parser.add_argument("--chunks3", type=int, default=1)
    parser.add_argument("--model", type=Path, help="derive compile parameters from an ONNX model and block")
    parser.add_argument("--block", help="bottleneck node-name prefix to compile from --model")
    return parser


def _compile_kwargs(opts):
    if opts.model is not None or opts.block is not None:
        if opts.model is None or opts.block is None:
            raise ValueError("--model and --block must be used together")
        import onnx
        try:
            from .benchmark_fused_bottleneck import bind_fused_bottleneck
            from .resnet_bottleneck import plan_bottleneck_blocks
        except ImportError:
            from benchmark_fused_bottleneck import bind_fused_bottleneck
            from resnet_bottleneck import plan_bottleneck_blocks
        model = onnx.load(opts.model)
        block = next((item for item in plan_bottleneck_blocks(model) if item.prefix == opts.block), None)
        if block is None:
            raise ValueError(f"no bottleneck block found for {opts.block!r}")
        binding = bind_fused_bottleneck(model, block)
        height, width = binding["input_shape"][2:]
        return {"width": width, "height": height, "channels": binding["input_shape"][1], "mid_channels": block.conv_plans[1].weight_shape[0],
                "shift1": binding["shifts"][0], "shift2": binding["shifts"][1], "shift3": binding["shifts"][2],
                "residual_shift": binding["main_residual_shift"], "input_shift": binding["skip_residual_shift"],
                "chunks1": binding["chunk_counts"][0], "chunks2": binding["chunk_counts"][1], "chunks3": binding["chunk_counts"][2],
                "output_width": binding["output_width"], "output_height": binding["output_height"], "output_channels": binding["output_channels"],
                "conv2_stride": binding["conv2_stride"][0], "skip_stride": binding["conv2_stride"][0],
                "skip_chunks": binding["skip_chunk_count"], "skip_shift": binding["skip_output_shift"] or 0,
                "residual_main_shift": binding["main_residual_shift"],
                "residual_skip_shift": binding["skip_residual_shift"]}
    return {"width": opts.width, "height": opts.height, "channels": opts.channels, "mid_channels": opts.mid_channels,
            "shift1": opts.shift1, "shift2": opts.shift2, "shift3": opts.shift3,
            "residual_shift": opts.residual_shift, "input_shift": opts.input_shift,
            "chunks1": opts.chunks1, "chunks2": opts.chunks2, "chunks3": opts.chunks3,
            "output_width": opts.output_width, "output_height": opts.output_height, "output_channels": opts.output_channels,
            "conv2_stride": opts.conv2_stride, "skip_stride": opts.skip_stride, "skip_chunks": opts.skip_chunks,
            "skip_shift": opts.skip_shift,
            "residual_main_shift": opts.residual_shift, "residual_skip_shift": opts.input_shift}


def main() -> None:
    opts = _parser().parse_args()
    run_design_cli(fused_identity_bottleneck, opts, compile_kwargs=_compile_kwargs, device=lambda value: device_from_args(value, n_cols=1))


if __name__ == "__main__":
    main()
