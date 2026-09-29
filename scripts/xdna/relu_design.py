#!/usr/bin/env python3
"""Compile or run a standalone signed INT8 ReLU XDNA kernel."""

from __future__ import annotations

import argparse
from typing import Any

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, Out
from aie.iron.algorithms import transform
from aie.iron.device import from_name
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL = __file__.replace("relu_design.py", "kernels/relu_int8.cc")


@iron.jit
def relu_int8(
    inp: In,
    out: Out,
    *,
    elements: CompileTime[int] = 1024,
    tile_width: CompileTime[int] = 128,
):
    tensor_type = np.ndarray[(elements,), np.dtype[np.int8]]
    tile_type = np.ndarray[(tile_width,), np.dtype[np.int8]]
    kernel = ExternalFunction(
        "relu_int8_kernel",
        source_file=_KERNEL,
        arg_types=[tile_type, tile_type, np.int32, np.int32],
    )
    return transform(kernel, tensor_type, tile_width, tile_size=tile_width)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--tile-width", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    opts = _parser().parse_args(argv)
    if opts.elements < 1 or opts.tile_width < 1:
        _parser().error("elements and tile width must be positive")

    def verify(run_opts: Any) -> None:
        iron.set_current_device(from_name(run_opts.dev, n_cols=None))
        source = (np.arange(run_opts.elements, dtype=np.int16) % 255 - 127).astype(np.int8)
        inp = iron.tensor(source, dtype=np.int8, device="npu")
        out = iron.zeros_like(inp)
        relu_int8(inp, out, elements=run_opts.elements, tile_width=run_opts.tile_width)
        actual = out.numpy()
        expected = np.maximum(source, 0).astype(np.int8)
        if not np.array_equal(actual, expected):
            raise RuntimeError("XDNA INT8 ReLU result differs from NumPy reference")

    run_design_cli(
        relu_int8,
        opts,
        compile_kwargs={"elements": opts.elements, "tile_width": opts.tile_width},
        run_and_verify=verify,
        device=lambda run_opts: from_name(run_opts.dev, n_cols=None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
