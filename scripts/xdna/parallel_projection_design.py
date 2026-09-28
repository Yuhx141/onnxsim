#!/usr/bin/env python3
"""Compile a bottleneck whose projection branch runs beside Conv1 on XDNA."""
from __future__ import annotations

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


def _align4(value: int) -> int:
    return (value + 3) & ~3


@iron.jit
def parallel_projection_bottleneck(
    activation: In,
    main_parameters: In,
    skip_parameters: In,
    result: Out,
    *,
    width: CompileTime[int] = 8,
    height: CompileTime[int] = 8,
    channels: CompileTime[int] = 64,
    mid_channels: CompileTime[int] = 64,
    shift1: CompileTime[int] = 0,
    shift2: CompileTime[int] = 0,
    shift3: CompileTime[int] = 0,
    residual_shift: CompileTime[int] = 0,
    input_shift: CompileTime[int] = 0,
    skip_shift: CompileTime[int] = 0,
    chunks1: CompileTime[int] = 1,
    chunks2: CompileTime[int] = 1,
    chunks3: CompileTime[int] = 1,
    skip_chunks: CompileTime[int] = 1,
    output_width: CompileTime[int] = 8,
    output_height: CompileTime[int] = 8,
    output_channels: CompileTime[int] = 256,
    conv2_stride: CompileTime[int] = 1,
    skip_stride: CompileTime[int] = 1,
    residual_main_shift: CompileTime[int] = 0,
    residual_skip_shift: CompileTime[int] = 0,
):
    """Fork the input DMA and execute the projection on a second NPU column."""
    pixels = width * height
    output_pixels = output_width * output_height
    outputs1 = mid_channels // chunks1
    outputs2 = (mid_channels // 2) // chunks2
    outputs3 = output_channels // chunks3
    skip_outputs = output_channels // skip_chunks
    bytes1 = _align4(outputs1 * channels) + outputs1 * 4
    bytes2 = _align4(outputs2 * mid_channels * 9) + outputs2 * 4
    bytes3 = _align4(outputs3 * mid_channels) + outputs3 * 4
    bytes_skip = _align4(skip_outputs * channels) + skip_outputs * 4
    max_weight_bytes = max(bytes1, bytes2, bytes3, bytes_skip)

    activation_ty = np.ndarray[(pixels * channels,), np.dtype[np.int8]]
    main_chunk_count = chunks1 + 2 * chunks2 + chunks3
    main_params_ty = np.ndarray[(main_chunk_count * max_weight_bytes,), np.dtype[np.uint8]]
    skip_params_ty = np.ndarray[(skip_chunks * max_weight_bytes,), np.dtype[np.uint8]]
    weight_ty = np.ndarray[(max_weight_bytes,), np.dtype[np.uint8]]
    stage1_ty = np.ndarray[(pixels * mid_channels,), np.dtype[np.int8]]
    skip_ty = np.ndarray[(output_pixels * output_channels,), np.dtype[np.int8]]
    stage2a_ty = np.ndarray[(output_pixels * (mid_channels // 2),), np.dtype[np.int8]]
    stage2b_ty = np.ndarray[(output_pixels * (mid_channels // 2),), np.dtype[np.int8]]
    stage2_ty = np.ndarray[(output_pixels * (mid_channels + output_channels),), np.dtype[np.int8]]
    output_ty = np.ndarray[(output_pixels * output_channels,), np.dtype[np.int8]]

    flags = [
        f"-DFUSED_W={width}", f"-DFUSED_H={height}", f"-DFUSED_C={channels}",
        f"-DFUSED_OUT_W={output_width}", f"-DFUSED_OUT_H={output_height}",
        f"-DFUSED_OUT_C={output_channels}", f"-DFUSED_CONV2_STRIDE={conv2_stride}",
        f"-DFUSED_SKIP_STRIDE={skip_stride}", "-DFUSED_PROJECTION=1",
        f"-DFUSED_SKIP_CHUNKS={skip_chunks}", f"-DFUSED_SKIP_OUTPUTS={skip_outputs}",
        f"-DFUSED_SKIP_SHIFT={skip_shift}",
        f"-DFUSED_SKIP_BIAS_OFFSET={_align4(skip_outputs * channels)}",
        f"-DFUSED_MID={mid_channels}", f"-DFUSED_SHIFT1={shift1}",
        f"-DFUSED_SHIFT2={shift2}", f"-DFUSED_SHIFT3={shift3}",
        f"-DFUSED_C1_CHUNKS={chunks1}", f"-DFUSED_C2_CHUNKS={chunks2}",
        f"-DFUSED_C3_CHUNKS={chunks3}", f"-DFUSED_C1_OUTPUTS={outputs1}",
        f"-DFUSED_C2_OUTPUTS={outputs2}", f"-DFUSED_C3_OUTPUTS={outputs3}",
        f"-DFUSED_BIAS1_OFFSET={_align4(outputs1 * channels)}",
        f"-DFUSED_BIAS2_OFFSET={_align4(outputs2 * mid_channels * 9)}",
        f"-DFUSED_BIAS3_OFFSET={_align4(outputs3 * mid_channels)}",
        f"-DFUSED_MAIN_RESIDUAL_SHIFT={residual_main_shift}",
        f"-DFUSED_SKIP_RESIDUAL_SHIFT={residual_skip_shift}",
    ]
    k1 = ExternalFunction(
        "fused_bottleneck_conv1_chunk", source_file=str(_KERNEL),
        arg_types=[activation_ty, weight_ty, stage1_ty, np.int32], compile_flags=flags,
    )
    kskip = ExternalFunction(
        "fused_bottleneck_skip_chunk", source_file=str(_SKIP_KERNEL),
        arg_types=[activation_ty, weight_ty, skip_ty, np.int32], compile_flags=flags,
    )
    k2 = ExternalFunction(
        "fused_bottleneck_conv2_chunk", source_file=str(_KERNEL),
        arg_types=[stage1_ty, weight_ty, stage2a_ty, np.int32, np.int32], compile_flags=flags,
    )
    k2b = ExternalFunction(
        "fused_bottleneck_conv2_chunk_b", source_file=str(_KERNEL),
        arg_types=[stage1_ty, weight_ty, stage2b_ty, np.int32, np.int32], compile_flags=flags,
    )
    k3 = ExternalFunction(
        "fused_bottleneck_conv3_chunk", source_file=str(_KERNEL),
        arg_types=[stage2_ty, weight_ty, output_ty, np.int32], compile_flags=flags,
    )

    main_activation_fifo = ObjectFifo(activation_ty, depth=1, name="main_activation")
    skip_activation_fifo = ObjectFifo(activation_ty, depth=1, name="skip_activation")
    main_weights_fifo = ObjectFifo(weight_ty, depth=1, name="main_weights")
    skip_weights_fifo = ObjectFifo(weight_ty, depth=1, name="skip_weights")
    stage1_fifo = ObjectFifo(stage1_ty, depth=1, name="main_stage1")
    skip_fifo = ObjectFifo(skip_ty, depth=1, name="projection_output")
    stage2a_fifo = ObjectFifo(stage2a_ty, depth=1, name="main_stage2a")
    stage2b_fifo = ObjectFifo(stage2b_ty, depth=1, name="main_stage2b")
    stage2_fifo = ObjectFifo(stage2_ty, depth=1, name="residual_join")
    output_fifo = ObjectFifo(output_ty, depth=1, name="block_output")

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
        out.release(1)
        inp.release(1)
        discard(weights, 2 * chunks2 + chunks3)

    def projection_worker(inp, weights, out, kernel):
        x = inp.acquire(1)
        residual = out.acquire(1)
        for i in range_(skip_chunks):
            w = weights.acquire(1)
            kernel(x, w, residual, i)
            weights.release(1)
        out.release(1)
        inp.release(1)

    def conv2_worker(inp, weights, out, kernel, channel_offset, is_a):
        discard(weights, chunks1 + (0 if is_a else chunks2))
        bundle = inp.acquire(1)
        output = out.acquire(1)
        for i in range_(chunks2):
            w = weights.acquire(1)
            kernel(bundle, w, output, i, channel_offset)
            weights.release(1)
        out.release(1)
        inp.release(1)
        discard(weights, (chunks2 if is_a else 0) + chunks3)

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
        Worker(conv1_worker, fn_args=[main_activation_fifo.cons(), main_weights_fifo.cons(), stage1_fifo.prod(), k1], tile=Tile(0, 2), stack_size=0x1000),
        Worker(projection_worker, fn_args=[skip_activation_fifo.cons(), skip_weights_fifo.cons(), skip_fifo.prod(), kskip], tile=Tile(1, 2), stack_size=0x1000),
        Worker(conv2_worker, fn_args=[stage1_fifo.cons(), main_weights_fifo.cons(), stage2a_fifo.prod(), k2, 0, True], tile=Tile(0, 3), stack_size=0x1000),
        Worker(conv2_worker, fn_args=[stage1_fifo.cons(), main_weights_fifo.cons(), stage2b_fifo.prod(), k2b, mid_channels // 2, False], tile=Tile(0, 5), stack_size=0x1000),
        Worker(conv3_worker, fn_args=[stage2_fifo.cons(), main_weights_fifo.cons(), output_fifo.prod(), k3], tile=Tile(0, 4), stack_size=0x1000),
    ]
    ObjectFifoLink(
        [stage2a_fifo.cons(), stage2b_fifo.cons(), skip_fifo.cons()], stage2_fifo.prod(),
        src_offsets=[0, output_pixels * (mid_channels // 2), output_pixels * mid_channels],
    )

    def sequence(x, main_w, skip_w, y, main_x_prod, skip_x_prod, main_w_prod, skip_w_prod, y_cons):
        inputs = TaskGroup()
        main_x_prod.fill(x, wait=True, group=inputs)
        skip_x_prod.fill(x, wait=True, group=inputs)
        inputs.finish()

        transfers = TaskGroup()
        for i in range(main_chunk_count):
            main_w_prod.fill(
                main_w, wait=True, sizes=[max_weight_bytes], strides=[1],
                offset=i * max_weight_bytes, transfer_len=max_weight_bytes, group=transfers,
            )
        for i in range(skip_chunks):
            skip_w_prod.fill(
                skip_w, wait=True, sizes=[max_weight_bytes], strides=[1],
                offset=i * max_weight_bytes, transfer_len=max_weight_bytes, group=transfers,
            )
        transfers.finish()

        output = TaskGroup()
        y_cons.drain(y, wait=True, group=output)
        output.finish()

    runtime = Runtime(
        sequence,
        [activation_ty, main_params_ty, skip_params_ty, output_ty,
         main_activation_fifo.prod(), skip_activation_fifo.prod(),
         main_weights_fifo.prod(), skip_weights_fifo.prod(), output_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


def _compile_kwargs(opts):
    try:
        from .fused_bottleneck_design import _compile_kwargs as fused_compile_kwargs
    except ImportError:
        from fused_bottleneck_design import _compile_kwargs as fused_compile_kwargs
    kwargs = fused_compile_kwargs(opts)
    if not kwargs.get("skip_chunks", 0):
        raise ValueError("parallel projection scheduling requires a projection bottleneck")
    if (kwargs["chunks1"], kwargs["chunks2"], kwargs["chunks3"], kwargs["skip_chunks"]) != (1, 1, 1, 1):
        raise ValueError(
            "parallel projection scheduling currently requires one weight chunk per Conv; "
            "larger blocks need per-stage weight-stream routing to avoid FIFO contention"
        )
    return kwargs


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--block", required=True)
    run_design_cli(
        parallel_projection_bottleneck,
        parser.parse_args(),
        compile_kwargs=_compile_kwargs,
        device=lambda value: device_from_args(value, n_cols=2),
    )


if __name__ == "__main__":
    main()
