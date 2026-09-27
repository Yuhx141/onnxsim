#!/usr/bin/env python3
"""Compile or verify a scalar float32 multiplication kernel on XDNA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, Out
from aie.iron.algorithms import transform
from aie.iron.device import from_name
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL = Path(__file__).with_name("kernels") / "mul_scalar_f32.cc"


@iron.jit
def mul_scalar(
    inp: In,
    out: Out,
    *,
    elements: CompileTime[int] = 1024,
    tile_width: CompileTime[int] = 128,
    scalar: CompileTime[float] = 1.0,
):
    tensor_type = np.ndarray[(elements,), np.dtype[np.float32]]
    tile_type = np.ndarray[(tile_width,), np.dtype[np.float32]]
    kernel = ExternalFunction(
        "mul_scalar_f32",
        source_file=str(_KERNEL),
        arg_types=[tile_type, tile_type, np.int32],
        compile_flags=[f"-DMUL_SCALAR={scalar:.9g}f", "-ffp-contract=off"],
    )
    return transform(kernel, tensor_type, tile_size=tile_width)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--tile-width", type=int, default=128)
    parser.add_argument("--scalar", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    opts = _parser().parse_args(argv)
    if opts.elements < 1 or opts.tile_width < 1 or opts.elements % opts.tile_width:
        _parser().error("elements must be positive and divisible by tile width")
    kwargs = {"elements": opts.elements, "tile_width": opts.tile_width, "scalar": opts.scalar}

    def verify(run_opts: Any) -> None:
        iron.set_current_device(from_name(run_opts.dev, n_cols=None))
        source = np.linspace(-3.0, 5.0, run_opts.elements, dtype=np.float32)
        inp = iron.tensor(source, dtype=np.float32, device="npu")
        out = iron.zeros_like(inp)
        mul_scalar(inp, out, **kwargs)
        actual = out.numpy()
        expected = np.multiply(source, np.float32(run_opts.scalar), dtype=np.float32)
        if not np.array_equal(actual, expected):
            raise RuntimeError("XDNA float32 scalar Mul differs from NumPy reference")

    run_design_cli(
        mul_scalar,
        opts,
        compile_kwargs=kwargs,
        run_and_verify=verify,
        device=lambda run_opts: from_name(run_opts.dev, n_cols=None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
