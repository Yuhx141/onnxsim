#!/usr/bin/env python3
"""Compile or verify a quantized uint8 residual Add + ReLU XDNA kernel."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, Out
from aie.iron.algorithms import transform_binary
from aie.iron.device import from_name
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL = Path(__file__).with_name("kernels") / "quantized_add_relu_u8.cc"


@iron.jit
def quantized_add_relu(
    lhs: In,
    rhs: In,
    output: Out,
    *,
    elements: CompileTime[int] = 1024,
    tile_width: CompileTime[int] = 128,
    scale_a: CompileTime[float] = 1.0,
    scale_b: CompileTime[float] = 1.0,
    scale_out: CompileTime[float] = 1.0,
    zero_a: CompileTime[int] = 0,
    zero_b: CompileTime[int] = 0,
    zero_out: CompileTime[int] = 0,
):
    tensor_type = np.ndarray[(elements,), np.dtype[np.uint8]]
    tile_type = np.ndarray[(tile_width,), np.dtype[np.uint8]]
    mult_a = round((scale_a / scale_out) * (1 << 30))
    mult_b = round((scale_b / scale_out) * (1 << 30))
    flags = [
        f"-DADD_MULT_A={mult_a}LL", f"-DADD_MULT_B={mult_b}LL",
        f"-DADD_ZERO_A={zero_a}",
        f"-DADD_ZERO_B={zero_b}", f"-DADD_ZERO_OUT={zero_out}",
    ]
    kernel = ExternalFunction(
        "quantized_add_relu_u8",
        source_file=str(_KERNEL),
        arg_types=[tile_type, tile_type, tile_type, np.int32],
        compile_flags=flags,
    )
    return transform_binary(kernel, tensor_type, tile_size=tile_width)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--tile-width", type=int, default=128)
    parser.add_argument("--scale-a", type=float, default=1.0)
    parser.add_argument("--scale-b", type=float, default=1.0)
    parser.add_argument("--scale-out", type=float, default=1.0)
    parser.add_argument("--zero-a", type=int, default=0)
    parser.add_argument("--zero-b", type=int, default=0)
    parser.add_argument("--zero-out", type=int, default=0)
    return parser


def _reference(a, b, *, scale_a, scale_b, scale_out, zero_a, zero_b, zero_out):
    real = (a.astype(np.float32) - np.float32(zero_a)) * np.float32(scale_a)
    real += (b.astype(np.float32) - np.float32(zero_b)) * np.float32(scale_b)
    quantized = np.rint(np.maximum(real, 0.0) / np.float32(scale_out) + np.float32(zero_out))
    return np.clip(quantized, 0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    opts = _parser().parse_args(argv)
    if opts.elements < 1 or opts.tile_width < 1 or opts.elements % opts.tile_width:
        _parser().error("elements must be positive and divisible by tile width")
    if min(opts.scale_a, opts.scale_b, opts.scale_out) <= 0:
        _parser().error("quantization scales must be positive")
    for scale in (opts.scale_a / opts.scale_out, opts.scale_b / opts.scale_out):
        mantissa, _ = math.frexp(scale)
        if mantissa != 0.5:
            _parser().error("this kernel requires input/output scale ratios to be powers of two")
    if any(value < 0 or value > 255 for value in (opts.zero_a, opts.zero_b, opts.zero_out)):
        _parser().error("uint8 zero points must be in [0, 255]")

    kwargs = {
        "elements": opts.elements, "tile_width": opts.tile_width,
        "scale_a": opts.scale_a, "scale_b": opts.scale_b, "scale_out": opts.scale_out,
        "zero_a": opts.zero_a, "zero_b": opts.zero_b, "zero_out": opts.zero_out,
    }

    def verify(run_opts: Any) -> None:
        iron.set_current_device(from_name(run_opts.dev, n_cols=None))
        indices = np.arange(run_opts.elements, dtype=np.int32)
        a = ((indices * 37 + 11) % 256).astype(np.uint8)
        b = ((indices * 19 + 239) % 256).astype(np.uint8)
        a_tensor = iron.tensor(a, dtype=np.uint8, device="npu")
        b_tensor = iron.tensor(b, dtype=np.uint8, device="npu")
        out_tensor = iron.zeros_like(a_tensor)
        quantized_add_relu(a_tensor, b_tensor, out_tensor, **kwargs)
        actual = out_tensor.numpy()
        expected = _reference(
            a, b, scale_a=run_opts.scale_a, scale_b=run_opts.scale_b,
            scale_out=run_opts.scale_out, zero_a=run_opts.zero_a,
            zero_b=run_opts.zero_b, zero_out=run_opts.zero_out,
        )
        if not np.array_equal(actual, expected):
            mismatch = int(np.flatnonzero(actual != expected)[0])
            raise RuntimeError(
                f"XDNA quantized Add+ReLU mismatch at {mismatch}: "
                f"actual={actual[mismatch]}, expected={expected[mismatch]}"
            )

    run_design_cli(
        quantized_add_relu,
        opts,
        compile_kwargs=kwargs,
        run_and_verify=verify,
        device=lambda run_opts: from_name(run_opts.dev, n_cols=None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
