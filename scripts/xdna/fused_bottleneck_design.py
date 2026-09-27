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
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL = Path(__file__).with_name("kernels") / "fused_identity_bottleneck.cc"


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
):
    pixels = width * height
    outputs1 = mid_channels // chunks1
    outputs2 = (mid_channels // 2) // chunks2
    outputs3 = channels // chunks3
    bytes1 = _align4(outputs1 * channels) + outputs1 * 4
    bytes2 = _align4(outputs2 * mid_channels * 9) + outputs2 * 4
    bytes3 = _align4(outputs3 * mid_channels) + outputs3 * 4

    activation_ty = np.ndarray[(pixels * channels,), np.dtype[np.int8]]
    params_ty = np.ndarray[(chunks1 * bytes1 + 2 * chunks2 * bytes2 + chunks3 * bytes3,), np.dtype[np.uint8]]
    max_weight_bytes = max(bytes1, bytes2, bytes3)
    weight_ty = np.ndarray[(max_weight_bytes,), np.dtype[np.uint8]]
    stage1_ty = np.ndarray[(pixels * (mid_channels + channels),), np.dtype[np.int8]]
    stage2a_ty = np.ndarray[(pixels * (mid_channels // 2 + channels),), np.dtype[np.int8]]
    stage2b_ty = np.ndarray[(pixels * (mid_channels // 2),), np.dtype[np.int8]]
    stage2_ty = np.ndarray[(pixels * (mid_channels + channels),), np.dtype[np.int8]]
    output_ty = np.ndarray[(pixels * channels,), np.dtype[np.int8]]

    flags = [
        f"-DFUSED_W={width}", f"-DFUSED_H={height}", f"-DFUSED_C={channels}",
        f"-DFUSED_MID={mid_channels}", f"-DFUSED_SHIFT1={shift1}",
        f"-DFUSED_SHIFT2={shift2}", f"-DFUSED_SHIFT3={shift3}",
        f"-DFUSED_RESIDUAL_SHIFT={residual_shift}", f"-DFUSED_INPUT_SHIFT={input_shift}",
        f"-DFUSED_C1_CHUNKS={chunks1}", f"-DFUSED_C2_CHUNKS={chunks2}",
        f"-DFUSED_C3_CHUNKS={chunks3}", f"-DFUSED_C1_OUTPUTS={outputs1}",
        f"-DFUSED_C2_OUTPUTS={outputs2}", f"-DFUSED_C3_OUTPUTS={outputs3}",
        f"-DFUSED_BIAS1_OFFSET={_align4(outputs1 * channels)}",
        f"-DFUSED_BIAS2_OFFSET={_align4(outputs2 * mid_channels * 9)}",
        f"-DFUSED_BIAS3_OFFSET={_align4(outputs3 * mid_channels)}",
    ]
    k1 = ExternalFunction("fused_bottleneck_conv1_chunk", source_file=str(_KERNEL), arg_types=[activation_ty, weight_ty, stage1_ty, np.int32], compile_flags=flags)
    k2 = ExternalFunction("fused_bottleneck_conv2_chunk", source_file=str(_KERNEL), arg_types=[stage1_ty, weight_ty, stage2a_ty, np.int32, np.int32], compile_flags=flags)
    k2b = ExternalFunction("fused_bottleneck_conv2_chunk_b", source_file=str(_KERNEL), arg_types=[stage1_ty, weight_ty, stage2b_ty, np.int32, np.int32], compile_flags=flags)
    k3 = ExternalFunction("fused_bottleneck_conv3_chunk", source_file=str(_KERNEL), arg_types=[stage2_ty, weight_ty, output_ty, np.int32], compile_flags=flags)

    activation_fifo = ObjectFifo(activation_ty, depth=1, name="bottleneck_activation")
    weights_fifo = ObjectFifo(weight_ty, depth=1, name="bottleneck_weight_chunks")
    stage1_fifo = ObjectFifo(stage1_ty, depth=1, name="bottleneck_stage1_bundle")
    stage2a_fifo = ObjectFifo(stage2a_ty, depth=1, name="bottleneck_stage2a")
    stage2b_fifo = ObjectFifo(stage2b_ty, depth=1, name="bottleneck_stage2b")
    stage2_fifo = ObjectFifo(stage2_ty, depth=1, name="bottleneck_stage2_bundle")
    output_fifo = ObjectFifo(output_ty, depth=1, name="bottleneck_output")

    def discard(weights, count):
        for _ in range_(count):
            weights.acquire(1)
            weights.release(1)

    def conv1_worker(inp, weights, out, kernel):
        x = inp.acquire(1)
        bundle = out.acquire(1)
        for i in range_(chunks1):
            w = weights.acquire(1)
            kernel(x, w, bundle, i)
            weights.release(1)
        # Append centered identity branch after q1 tensor.
        for i in range_(pixels * channels):
            bundle[pixels * mid_channels + i] = x[i]
        out.release(1)
        inp.release(1)
        discard(weights, 2 * chunks2 + chunks3)

    def conv2_worker(inp, weights, out, kernel, channel_offset, is_a):
        discard(weights, chunks1 + (0 if is_a else chunks2))
        bundle = inp.acquire(1)
        output = out.acquire(1)
        for i in range_(chunks2):
            w = weights.acquire(1)
            kernel(bundle, w, output, i, channel_offset)
            weights.release(1)
        discard(weights, (chunks2 if is_a else 0) + chunks3)
        if is_a:
            for i in range_(pixels * channels):
                output[pixels * (mid_channels // 2) + i] = bundle[pixels * mid_channels + i]
        out.release(1)
        inp.release(1)

    def conv3_worker(inp, weights, out, kernel):
        discard(weights, chunks1 + 2 * chunks2)
        bundle = inp.acquire(1)
        output = out.acquire(1)
        for i in range_(chunks3):
            w = weights.acquire(1)
            kernel(bundle, w, output, i)
            weights.release(1)
        out.release(1)
        inp.release(1)

    workers = [
        Worker(conv1_worker, fn_args=[activation_fifo.cons(), weights_fifo.cons(), stage1_fifo.prod(), k1], tile=Tile(0, 2), stack_size=0x1000),
        Worker(conv2_worker, fn_args=[stage1_fifo.cons(), weights_fifo.cons(), stage2a_fifo.prod(), k2, 0, True], tile=Tile(0, 3), stack_size=0x1000),
        Worker(conv2_worker, fn_args=[stage1_fifo.cons(), weights_fifo.cons(), stage2b_fifo.prod(), k2b, mid_channels // 2, False], tile=Tile(0, 5), stack_size=0x1000),
        Worker(conv3_worker, fn_args=[stage2_fifo.cons(), weights_fifo.cons(), output_fifo.prod(), k3], tile=Tile(0, 4), stack_size=0x1000),
    ]
    ObjectFifoLink([stage2a_fifo.cons(), stage2b_fifo.cons()], stage2_fifo.prod(), src_offsets=[0, pixels * (mid_channels // 2 + channels)])

    def sequence(x, w, y, xprod, weights_prod, ycons):
        xprod.fill(x)
        sizes = [bytes1] * chunks1 + [bytes2] * (2 * chunks2) + [bytes3] * chunks3
        offset = 0
        for size in sizes:
            weights_prod.fill(w, wait=True, sizes=[size], strides=[1], offset=offset, transfer_len=size)
            offset += size
        ycons.drain(y, wait=True)

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
                "residual_shift": binding["residual_shift"], "input_shift": binding["input_shift"],
                "chunks1": binding["chunk_counts"][0], "chunks2": binding["chunk_counts"][1], "chunks3": binding["chunk_counts"][2]}
    return {"width": opts.width, "height": opts.height, "channels": opts.channels, "mid_channels": opts.mid_channels,
            "shift1": opts.shift1, "shift2": opts.shift2, "shift3": opts.shift3,
            "residual_shift": opts.residual_shift, "input_shift": opts.input_shift}


def main() -> None:
    opts = _parser().parse_args()
    run_design_cli(fused_identity_bottleneck, opts, compile_kwargs=_compile_kwargs, device=lambda value: device_from_args(value, n_cols=1))


if __name__ == "__main__":
    main()
