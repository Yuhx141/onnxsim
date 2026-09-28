#!/usr/bin/env python3
"""Compile or verify a row-streamed float32 NCHW 2D MaxPool kernel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.iron.runtime import TaskGroup
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL = Path(__file__).with_name("kernels") / "maxpool2d_nchw_f32.cc"
_U8_KERNEL = Path(__file__).with_name("kernels") / "maxpool2d_nchw_u8.cc"


@iron.jit
def maxpool2d(
    inp: In,
    out: Out,
    *,
    channels: CompileTime[int] = 64,
    input_height: CompileTime[int] = 114,
    input_width: CompileTime[int] = 114,
    output_height: CompileTime[int] = 56,
    output_width: CompileTime[int] = 56,
    kernel_height: CompileTime[int] = 3,
    kernel_width: CompileTime[int] = 3,
    stride_height: CompileTime[int] = 2,
    stride_width: CompileTime[int] = 2,
    tile_output_rows: CompileTime[int] = 8,
    tile_channels: CompileTime[int] = 1,
    uint8: CompileTime[bool] = False,
):
    if output_height % tile_output_rows:
        raise ValueError("output height must be divisible by tile_output_rows")
    if input_height < (output_height - 1) * stride_height + kernel_height:
        raise ValueError("padded input height is too small for the pooling window")
    if channels % tile_channels:
        raise ValueError("channels must be divisible by tile_channels")
    output_row_groups = output_height // tile_output_rows
    input_rows_per_tile = (tile_output_rows - 1) * stride_height + kernel_height
    channel_groups = channels // tile_channels
    chunks = channel_groups * output_row_groups
    data_dtype = np.uint8 if uint8 else np.float32
    input_type = np.ndarray[(channels * input_height * input_width,), np.dtype[data_dtype]]
    output_type = np.ndarray[(channels * output_height * output_width,), np.dtype[data_dtype]]
    input_tile = np.ndarray[(tile_channels * input_rows_per_tile * input_width,), np.dtype[data_dtype]]
    output_tile = np.ndarray[(tile_channels * tile_output_rows * output_width,), np.dtype[data_dtype]]
    kernel = ExternalFunction(
        "maxpool2d_nchw_u8" if uint8 else "maxpool2d_nchw_f32",
        source_file=str(_U8_KERNEL if uint8 else _KERNEL),
        arg_types=[input_tile, output_tile, np.int32, np.int32, np.int32, np.int32,
                   np.int32, np.int32, np.int32, np.int32],
    )
    input_fifo = ObjectFifo(input_tile, depth=1, name="maxpool_input")
    output_fifo = ObjectFifo(output_tile, depth=1, name="maxpool_output")

    def worker_fn(input_cons, output_prod, kernel_fn):
        for _ in range_(chunks):
            source = input_cons.acquire(1)
            result = output_prod.acquire(1)
            kernel_fn(
                source, result, input_width, output_width,
                kernel_height, kernel_width, stride_height, stride_width,
                tile_output_rows, tile_channels,
            )
            output_prod.release(1)
            input_cons.release(1)

    worker = Worker(
        worker_fn,
        fn_args=[input_fifo.cons(), output_fifo.prod(), kernel],
        tile=Tile(0, 2),
        stack_size=0x1000,
    )

    def sequence(source, result, source_prod, result_cons):
        for chunk in range(chunks):
            channel = (chunk // output_row_groups) * tile_channels
            row_group = chunk % output_row_groups
            input_offset = channel * input_height * input_width + row_group * tile_output_rows * stride_height * input_width
            output_offset = channel * output_height * output_width + row_group * tile_output_rows * output_width
            group = TaskGroup()
            source_prod.fill(
                source, wait=True,
                sizes=[tile_channels, input_rows_per_tile * input_width],
                strides=[input_height * input_width, 1],
                offset=input_offset, transfer_len=tile_channels * input_rows_per_tile * input_width,
                group=group,
            )
            group.finish()
            group = TaskGroup()
            result_cons.drain(
                result, wait=True,
                sizes=[tile_channels, tile_output_rows * output_width],
                strides=[output_height * output_width, 1],
                offset=output_offset, transfer_len=tile_channels * tile_output_rows * output_width,
                group=group,
            )
            group.finish()

    runtime = Runtime(
        sequence,
        [input_type, output_type, input_fifo.prod(), output_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


def _pool_reference(x, *, kernel, stride, pads):
    channels, height, width = x.shape
    pt, pl, pb, pr = pads
    kh, kw = kernel
    sh, sw = stride
    padded = np.pad(x, ((0, 0), (pt, pb), (pl, pr)), constant_values=-np.inf)
    oh = (height + pt + pb - kh) // sh + 1
    ow = (width + pl + pr - kw) // sw + 1
    out = np.empty((channels, oh, ow), dtype=np.float32)
    for y in range(oh):
        for x_index in range(ow):
            out[:, y, x_index] = np.max(
                padded[:, y * sh : y * sh + kh, x_index * sw : x_index * sw + kw],
                axis=(1, 2),
            )
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--input-height", type=int, default=114, help="padded input height")
    parser.add_argument("--input-width", type=int, default=114, help="padded input width")
    parser.add_argument("--output-height", type=int, default=56)
    parser.add_argument("--output-width", type=int, default=56)
    parser.add_argument("--kernel-height", type=int, default=3)
    parser.add_argument("--kernel-width", type=int, default=3)
    parser.add_argument("--stride-height", type=int, default=2)
    parser.add_argument("--stride-width", type=int, default=2)
    parser.add_argument("--tile-output-rows", type=int, default=8)
    parser.add_argument("--tile-channels", type=int, default=1)
    parser.add_argument("--uint8", action="store_true", help="compile a quantized uint8 MaxPool kernel")
    parser.add_argument("--pad-top", type=int, default=1)
    parser.add_argument("--pad-left", type=int, default=1)
    parser.add_argument("--pad-bottom", type=int, default=1)
    parser.add_argument("--pad-right", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    opts = _parser().parse_args(argv)
    if min(opts.channels, opts.input_height, opts.input_width, opts.output_height,
           opts.output_width, opts.kernel_height, opts.kernel_width,
           opts.stride_height, opts.stride_width, opts.tile_output_rows, opts.tile_channels) < 1:
        _parser().error("all pooling dimensions and strides must be positive")
    if opts.output_height % opts.tile_output_rows:
        _parser().error("output height must be divisible by tile output rows")
    if opts.channels % opts.tile_channels:
        _parser().error("channels must be divisible by tile_channels")
    kwargs = {
        "channels": opts.channels, "input_height": opts.input_height,
        "input_width": opts.input_width, "output_height": opts.output_height,
        "output_width": opts.output_width, "kernel_height": opts.kernel_height,
        "kernel_width": opts.kernel_width, "stride_height": opts.stride_height,
        "stride_width": opts.stride_width, "tile_output_rows": opts.tile_output_rows,
        "tile_channels": opts.tile_channels, "uint8": opts.uint8,
    }

    def verify(run_opts: Any) -> None:
        iron.set_current_device(from_name(run_opts.dev, n_cols=None))
        in_h = ((run_opts.output_height - 1) * run_opts.stride_height
                + run_opts.kernel_height - run_opts.pad_top)
        in_w = ((run_opts.output_width - 1) * run_opts.stride_width
                + run_opts.kernel_width - run_opts.pad_left)
        semantic_height = in_h + run_opts.pad_top + run_opts.pad_bottom
        semantic_width = in_w + run_opts.pad_left + run_opts.pad_right
        extra_bottom = run_opts.input_height - semantic_height
        extra_right = run_opts.input_width - semantic_width
        if extra_bottom < 0 or extra_right < 0:
            raise ValueError("compiled input storage is too small for the pooling shape and padding")
        dtype = np.uint8 if run_opts.uint8 else np.float32
        values = np.arange(run_opts.channels * in_h * in_w, dtype=np.int32)
        source = ((values % 251) if run_opts.uint8 else (values % 997) / 997.0)
        source = source.reshape(run_opts.channels, in_h, in_w).astype(dtype)
        padded = np.pad(
            source,
            ((0, 0), (run_opts.pad_top, run_opts.pad_bottom + extra_bottom),
             (run_opts.pad_left, run_opts.pad_right + extra_right)),
            constant_values=0 if run_opts.uint8 else -np.inf,
        ).astype(dtype)
        output = np.zeros((run_opts.channels * run_opts.output_height * run_opts.output_width,), dtype=dtype)
        input_tensor = iron.tensor(padded.reshape(-1), dtype=dtype, device="npu")
        output_tensor = iron.tensor(output, dtype=dtype, device="npu")
        maxpool2d(input_tensor, output_tensor, **kwargs)
        actual = output_tensor.numpy().reshape(run_opts.channels, run_opts.output_height, run_opts.output_width)
        expected = _pool_reference(
            source.astype(np.float32),
            kernel=(run_opts.kernel_height, run_opts.kernel_width),
            stride=(run_opts.stride_height, run_opts.stride_width),
            pads=(run_opts.pad_top, run_opts.pad_left, run_opts.pad_bottom, run_opts.pad_right),
        )
        if not np.array_equal(actual, expected):
            raise RuntimeError("XDNA MaxPool result differs from NumPy reference")

    run_design_cli(
        maxpool2d,
        opts,
        compile_kwargs=kwargs,
        run_and_verify=verify,
        device=lambda run_opts: from_name(run_opts.dev, n_cols=None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
