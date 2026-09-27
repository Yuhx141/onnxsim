#!/usr/bin/env python3
"""Compile one fused INT8 identity ResNet bottleneck for an XDNA NPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import AnyMemTile, Tile
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
):
    pixels = width * height
    weight1_bytes = channels * mid_channels
    bias1_bytes = mid_channels * 4
    weight2_bytes = mid_channels * mid_channels * 9
    bias2_bytes = mid_channels * 4
    weight3_bytes = mid_channels * channels
    bias3_bytes = channels * 4
    bias1_offset = _align4(weight1_bytes)
    bias2_offset = _align4(weight2_bytes)
    bias3_offset = _align4(weight3_bytes)
    parameter1_bytes = bias1_offset + bias1_bytes
    parameter2_bytes = bias2_offset + bias2_bytes
    parameter3_bytes = bias3_offset + bias3_bytes
    total_parameter_bytes = parameter1_bytes + parameter2_bytes + parameter3_bytes

    activation_full_ty = np.ndarray[(pixels * channels,), np.dtype[np.int8]]
    activation_row_ty = np.ndarray[(width * channels,), np.dtype[np.int8]]
    mid_row_ty = np.ndarray[(width * mid_channels,), np.dtype[np.uint8]]
    mid_half_row_ty = np.ndarray[(width * (mid_channels // 2),), np.dtype[np.uint8]]
    output_row_ty = np.ndarray[(width * channels,), np.dtype[np.uint8]]
    output_full_ty = np.ndarray[(pixels * channels,), np.dtype[np.int8]]
    parameters_full_ty = np.ndarray[(total_parameter_bytes,), np.dtype[np.uint8]]
    params1_ty = np.ndarray[(parameter1_bytes,), np.dtype[np.uint8]]
    params2_ty = np.ndarray[(parameter2_bytes,), np.dtype[np.uint8]]
    params3_ty = np.ndarray[(parameter3_bytes,), np.dtype[np.uint8]]

    common_flags = [
        f"-DFUSED_W={width}",
        f"-DFUSED_H={height}",
        f"-DFUSED_C={channels}",
        f"-DFUSED_MID={mid_channels}",
        f"-DFUSED_SHIFT1={shift1}",
        f"-DFUSED_SHIFT2={shift2}",
        f"-DFUSED_SHIFT3={shift3}",
        f"-DFUSED_RESIDUAL_SHIFT={residual_shift}",
        f"-DFUSED_INPUT_SHIFT={input_shift}",
        f"-DFUSED_BIAS1_OFFSET={bias1_offset}",
        f"-DFUSED_BIAS2_OFFSET={bias2_offset}",
        f"-DFUSED_BIAS3_OFFSET={bias3_offset}",
    ]
    conv1_kernel = ExternalFunction(
        "fused_bottleneck_conv1_row",
        source_file=str(_KERNEL),
        arg_types=[activation_row_ty, params1_ty, mid_row_ty],
        compile_flags=common_flags,
    )
    conv2_kernel = ExternalFunction(
        "fused_bottleneck_conv2_row",
        source_file=str(_KERNEL),
        arg_types=[
            mid_row_ty,
            mid_row_ty,
            mid_row_ty,
            params2_ty,
            mid_half_row_ty,
            np.int32,
            np.int32,
        ],
        compile_flags=common_flags,
    )
    conv3_kernel = ExternalFunction(
        "fused_bottleneck_conv3_residual_row",
        source_file=str(_KERNEL),
        arg_types=[mid_half_row_ty, mid_half_row_ty, params3_ty, activation_row_ty, output_row_ty],
        compile_flags=common_flags,
    )

    activation_fifo = ObjectFifo(activation_row_ty, name="bottleneck_activation")
    skip_fifo = activation_fifo.cons(4).forward(
        depth=2, tile=AnyMemTile, name="bottleneck_skip_buffer"
    )
    weights_fifo = ObjectFifo(parameters_full_ty, depth=1, name="bottleneck_weights")
    weight1_fifo, weight2_fifo, weight3_fifo = weights_fifo.cons().split(
        [0, parameter1_bytes, parameter1_bytes + parameter2_bytes],
        obj_types=[params1_ty, params2_ty, params3_ty],
        names=["bottleneck_conv1_weights", "bottleneck_conv2_weights", "bottleneck_conv3_weights"],
    )
    stage1_fifo = ObjectFifo(mid_row_ty, name="bottleneck_stage1")
    stage2a_fifo = ObjectFifo(mid_half_row_ty, name="bottleneck_stage2a")
    stage2b_fifo = ObjectFifo(mid_half_row_ty, name="bottleneck_stage2b")
    output_fifo = ObjectFifo(output_row_ty, name="bottleneck_output")

    def conv1_worker(input_fifo, weights_fifo, output_fifo, kernel):
        weights = weights_fifo.acquire(1)
        for _ in range_(height):
            input_row = input_fifo.acquire(1)
            output_row = output_fifo.acquire(1)
            kernel(input_row, weights, output_row)
            input_fifo.release(1)
            output_fifo.release(1)
        weights_fifo.release(1)

    def conv2_worker(input_fifo, weights_fifo, output_fifo, kernel, channel_offset):
        weights = weights_fifo.acquire(1)
        rows = input_fifo.acquire(2)
        output = output_fifo.acquire(1)
        kernel(rows[0], rows[0], rows[1], weights, output, 0, channel_offset)
        output_fifo.release(1)
        for _ in range_(height - 2):
            rows = input_fifo.acquire(3)
            output = output_fifo.acquire(1)
            kernel(rows[0], rows[1], rows[2], weights, output, 1, channel_offset)
            output_fifo.release(1)
            input_fifo.release(1)
        rows = input_fifo.acquire(2)
        output = output_fifo.acquire(1)
        kernel(rows[0], rows[1], rows[1], weights, output, height - 1, channel_offset)
        output_fifo.release(1)
        input_fifo.release(1)
        input_fifo.release(1)
        weights_fifo.release(1)

    def conv3_worker(main0_fifo, main1_fifo, weights_fifo, skip_fifo, output_fifo, kernel):
        weights = weights_fifo.acquire(1)
        for _ in range_(height):
            main0_row = main0_fifo.acquire(1)
            main1_row = main1_fifo.acquire(1)
            skip_row = skip_fifo.acquire(1)
            output_row = output_fifo.acquire(1)
            kernel(main0_row, main1_row, weights, skip_row, output_row)
            main0_fifo.release(1)
            main1_fifo.release(1)
            skip_fifo.release(1)
            output_fifo.release(1)
        weights_fifo.release(1)

    workers = [
        Worker(
            conv1_worker,
            fn_args=[activation_fifo.cons(), weight1_fifo.cons(), stage1_fifo.prod(), conv1_kernel],
            tile=Tile(0, 2),
            stack_size=0x1000,
        ),
        Worker(
            conv2_worker,
            fn_args=[stage1_fifo.cons(4), weight2_fifo.cons(), stage2a_fifo.prod(), conv2_kernel, 0],
            tile=Tile(0, 3),
            stack_size=4736,
        ),
        Worker(
            conv2_worker,
            fn_args=[stage1_fifo.cons(4), weight2_fifo.cons(), stage2b_fifo.prod(), conv2_kernel, mid_channels // 2],
            tile=Tile(0, 5),
            stack_size=4736,
        ),
        Worker(
            conv3_worker,
            fn_args=[
                stage2a_fifo.cons(),
                stage2b_fifo.cons(),
                weight3_fifo.cons(),
                skip_fifo.cons(),
                output_fifo.prod(),
                conv3_kernel,
            ],
            tile=Tile(0, 4),
            stack_size=0x1000,
        ),
    ]

    def sequence(x, w, y, activation_prod, weights_prod, output_cons):
        activation_prod.fill(x)
        weights_prod.fill(w)
        output_cons.drain(y, wait=True)

    runtime = Runtime(
        sequence,
        [
            activation_full_ty,
            parameters_full_ty,
            output_full_ty,
            activation_fifo.prod(),
            weights_fifo.prod(),
            output_fifo.cons(),
        ],
    )
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
        channels = binding["input_shape"][1]
        mid_channels = binding["block"].conv_plans[1].weight_shape[0]
        return {
            "width": width,
            "height": height,
            "channels": channels,
            "mid_channels": mid_channels,
            "shift1": binding["shifts"][0],
            "shift2": binding["shifts"][1],
            "shift3": binding["shifts"][2],
            "residual_shift": binding["residual_shift"],
            "input_shift": binding["input_shift"],
        }
    return {
        "width": opts.width,
        "height": opts.height,
        "channels": opts.channels,
        "mid_channels": opts.mid_channels,
        "shift1": opts.shift1,
        "shift2": opts.shift2,
        "shift3": opts.shift3,
        "residual_shift": opts.residual_shift,
        "input_shift": opts.input_shift,
    }


def main() -> None:
    opts = _parser().parse_args()
    run_design_cli(
        fused_identity_bottleneck,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda value: device_from_args(value, n_cols=1),
    )


if __name__ == "__main__":
    main()
