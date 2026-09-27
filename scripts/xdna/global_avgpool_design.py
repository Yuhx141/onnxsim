#!/usr/bin/env python3
"""Compile or verify a tiled float32 GlobalAveragePool kernel on XDNA."""

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

_KERNEL = Path(__file__).with_name("kernels") / "global_avgpool_f32.cc"


@iron.jit
def global_avgpool(
    inp: In,
    out: Out,
    *,
    channels: CompileTime[int] = 2048,
    spatial: CompileTime[int] = 49,
    tile_channels: CompileTime[int] = 64,
):
    if channels % tile_channels:
        raise ValueError("channels must be divisible by tile_channels")
    chunks = channels // tile_channels
    input_type = np.ndarray[(channels * spatial,), np.dtype[np.float32]]
    output_type = np.ndarray[(channels,), np.dtype[np.float32]]
    input_tile = np.ndarray[(tile_channels * spatial,), np.dtype[np.float32]]
    output_tile = np.ndarray[(tile_channels,), np.dtype[np.float32]]
    kernel = ExternalFunction(
        "global_avgpool_f32",
        source_file=str(_KERNEL),
        arg_types=[input_tile, output_tile, np.int32, np.int32],
    )
    input_fifo = ObjectFifo(input_tile, depth=1, name="gap_input")
    output_fifo = ObjectFifo(output_tile, depth=1, name="gap_output")

    def worker_fn(input_cons, output_prod, kernel_fn):
        for _ in range_(chunks):
            source = input_cons.acquire(1)
            result = output_prod.acquire(1)
            kernel_fn(source, result, spatial, tile_channels)
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
            source_offset = chunk * tile_channels * spatial
            result_offset = chunk * tile_channels
            group = TaskGroup()
            source_prod.fill(
                source,
                wait=True,
                sizes=[tile_channels * spatial],
                strides=[1],
                offset=source_offset,
                transfer_len=tile_channels * spatial,
                group=group,
            )
            group.finish()
            group = TaskGroup()
            result_cons.drain(
                result,
                wait=True,
                sizes=[tile_channels],
                strides=[1],
                offset=result_offset,
                transfer_len=tile_channels,
                group=group,
            )
            group.finish()

    runtime = Runtime(
        sequence,
        [input_type, output_type, input_fifo.prod(), output_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--channels", type=int, default=2048)
    parser.add_argument("--spatial", type=int, default=49)
    parser.add_argument("--tile-channels", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    opts = _parser().parse_args(argv)
    if min(opts.channels, opts.spatial, opts.tile_channels) < 1:
        _parser().error("channel and spatial sizes must be positive")
    if opts.channels % opts.tile_channels:
        _parser().error("channels must be divisible by tile channels")
    kwargs = {
        "channels": opts.channels,
        "spatial": opts.spatial,
        "tile_channels": opts.tile_channels,
    }

    def verify(run_opts: Any) -> None:
        iron.set_current_device(from_name(run_opts.dev, n_cols=None))
        source = np.linspace(-2.0, 3.0, run_opts.channels * run_opts.spatial, dtype=np.float32)
        inp = iron.tensor(source, dtype=np.float32, device="npu")
        out = iron.tensor(np.zeros((run_opts.channels,), dtype=np.float32), dtype=np.float32, device="npu")
        global_avgpool(inp, out, **kwargs)
        actual = out.numpy()
        expected = source.reshape(run_opts.channels, run_opts.spatial).mean(axis=1, dtype=np.float32)
        if not np.allclose(actual, expected, rtol=1e-6, atol=2e-6):
            raise RuntimeError("XDNA GlobalAveragePool differs from NumPy reference")

    run_design_cli(
        global_avgpool,
        opts,
        compile_kwargs=kwargs,
        run_and_verify=verify,
        device=lambda run_opts: from_name(run_opts.dev, n_cols=None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
