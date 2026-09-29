#!/usr/bin/env python3
"""Compile three sequential ResNet bottlenecks linked on the XDNA device."""

from __future__ import annotations

import argparse
import json
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
_BLOCKED_KERNEL = Path(__file__).with_name("kernels") / "fused_bottleneck_blocked.cc"


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _block_workers(chunks1, skip_chunks, chunks2, chunks3, nocompute=0):
    """Bind per-block chunk counts now; IRON traces worker bodies after the block loop ends."""
    def discard(weights, count):
        for _ in range_(count):
            weights.acquire(1)
            weights.release(1)

    def conv1_worker(inp, weights, out, skip_out, kernel, skip_kernel, identity_kernel):
        x = inp.acquire(1)
        bundle = out.acquire(1)
        for i in range_(chunks1):
            w = weights.acquire(1)
            if not nocompute & 1:
                kernel(x, w, bundle, i)
            weights.release(1)
        residual = skip_out.acquire(1)
        if skip_chunks:
            for i in range_(skip_chunks):
                w = weights.acquire(1)
                if not nocompute & 2:
                    skip_kernel(x, w, residual, i)
                weights.release(1)
        else:
            if not nocompute & 2:
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
            if not nocompute & 4:
                kernel(bundle, w, output, i, channel_offset)
            weights.release(1)
        out.release(1)
        inp.release(1)
        discard(weights, (chunks2 if is_a else 0) + chunks3)

    def conv3_worker(inp, weights, out, kernel):
        discard(weights, chunks1 + skip_chunks + 2 * chunks2)
        bundle = inp.acquire(1)
        output = out.acquire(1)
        for i in range_(chunks3):
            w = weights.acquire(1)
            if not nocompute & 8:
                kernel(bundle, w, output, i)
            weights.release(1)
        out.release(1)
        inp.release(1)

    return conv1_worker, conv2_worker, conv3_worker


@iron.jit
def linked_bottleneck_stage(
    activation: In,
    parameters: In,
    result: Out,
    *,
    stage_specs: CompileTime[str],
    tap: CompileTime[int] = 0,
    nocompute: CompileTime[int] = 0,
    blocked: CompileTime[int] = 0,
    dbg: CompileTime[int] = 0,
):
    """Run a fixed three-block stage with inter-block FIFOs on the device."""
    specs = json.loads(stage_specs)
    if not 1 <= len(specs) <= 3:
        raise ValueError("linked stage supports one to three bottleneck blocks")

    input_fifos = []
    weight_fifos = []
    output_fifos = []
    workers = []

    for block_index, spec in enumerate(specs):
        width, height, channels = spec["width"], spec["height"], spec["channels"]
        output_width, output_height = spec["output_width"], spec["output_height"]
        output_channels, mid_channels = spec["output_channels"], spec["mid_channels"]
        chunks1, chunks2, chunks3 = spec["chunks1"], spec["chunks2"], spec["chunks3"]
        skip_chunks = spec["skip_chunks"]
        pixels, output_pixels = width * height, output_width * output_height
        outputs1 = mid_channels // chunks1
        outputs2 = (mid_channels // 2) // chunks2
        outputs3 = output_channels // chunks3
        skip_outputs = output_channels // (skip_chunks if skip_chunks else 1)
        bytes1 = _align4(outputs1 * channels) + outputs1 * 4
        bytes2 = _align4(outputs2 * mid_channels * 9) + outputs2 * 4
        bytes3 = _align4(outputs3 * mid_channels) + outputs3 * 4
        bytes_skip = _align4(skip_outputs * channels) + skip_outputs * 4 if skip_chunks else 0
        slot_bytes = max(bytes1, bytes2, bytes3, bytes_skip)
        parameter_chunks = chunks1 + skip_chunks + 2 * chunks2 + chunks3
        if slot_bytes != spec["slot_bytes"] or parameter_chunks * slot_bytes != spec["params_len"]:
            raise ValueError(f"{spec['prefix']}: packed parameter layout does not match compiled stage")

        activation_ty = np.ndarray[(pixels * channels,), np.dtype[np.int8]]
        weight_ty = np.ndarray[(slot_bytes,), np.dtype[np.uint8]]
        # Blocked kernels keep conv1's output in a zero-padded [C/8][H+2][W+2][8] layout.
        stage1_elems = (mid_channels // 8) * (height + 2) * (width + 2) * 8 if blocked else pixels * mid_channels
        stage1_ty = np.ndarray[(stage1_elems,), np.dtype[np.int8]]
        skip_ty = np.ndarray[(output_pixels * output_channels,), np.dtype[np.int8]]
        stage2a_ty = np.ndarray[(output_pixels * (mid_channels // 2),), np.dtype[np.int8]]
        stage2b_ty = np.ndarray[(output_pixels * (mid_channels // 2),), np.dtype[np.int8]]
        stage2_ty = np.ndarray[(output_pixels * (mid_channels + output_channels),), np.dtype[np.int8]]
        output_ty = np.ndarray[(output_pixels * output_channels,), np.dtype[np.int8]]

        conv2_stride = spec["conv2_stride"]
        flags = [
            f"-DFUSED_W={width}", f"-DFUSED_H={height}", f"-DFUSED_C={channels}",
            f"-DFUSED_OUT_W={output_width}", f"-DFUSED_OUT_H={output_height}", f"-DFUSED_OUT_C={output_channels}",
            f"-DFUSED_CONV2_STRIDE={conv2_stride}", f"-DFUSED_SKIP_STRIDE={conv2_stride}",
            f"-DFUSED_PROJECTION={1 if skip_chunks else 0}", f"-DFUSED_SKIP_CHUNKS={max(skip_chunks, 1)}",
            f"-DFUSED_SKIP_OUTPUTS={skip_outputs}", f"-DFUSED_SKIP_SHIFT={spec['skip_shift']}",
            f"-DFUSED_SKIP_BIAS_OFFSET={_align4(skip_outputs * channels)}",
            f"-DFUSED_MID={mid_channels}", f"-DFUSED_SHIFT1={spec['shift1']}",
            f"-DFUSED_SHIFT2={spec['shift2']}", f"-DFUSED_SHIFT3={spec['shift3']}",
            f"-DFUSED_RESIDUAL_SHIFT={spec['residual_shift']}", f"-DFUSED_INPUT_SHIFT={spec['input_shift']}",
            f"-DFUSED_C1_CHUNKS={chunks1}", f"-DFUSED_C2_CHUNKS={chunks2}", f"-DFUSED_C3_CHUNKS={chunks3}",
            f"-DFUSED_C1_OUTPUTS={outputs1}", f"-DFUSED_C2_OUTPUTS={outputs2}", f"-DFUSED_C3_OUTPUTS={outputs3}",
            f"-DFUSED_BIAS1_OFFSET={_align4(outputs1 * channels)}",
            f"-DFUSED_BIAS2_OFFSET={_align4(outputs2 * mid_channels * 9)}",
            f"-DFUSED_BIAS3_OFFSET={_align4(outputs3 * mid_channels)}",
            f"-DFUSED_C1_MMUL={1 if spec['conv1_mmul'] else 0}",
            f"-DFUSED_MAIN_RESIDUAL_SHIFT={spec['residual_main_shift']}",
            f"-DFUSED_SKIP_RESIDUAL_SHIFT={spec['residual_skip_shift']}",
        ]
        if dbg:
            flags = flags + [f"-DFUSED_DBG={dbg}"]
        symbol_prefix = f"stage{block_index}"
        _KERNEL_SRC = _SKIP_SRC = _IDENT_SRC = str(_BLOCKED_KERNEL) if blocked else None
        if not blocked:
            _KERNEL_SRC, _SKIP_SRC, _IDENT_SRC = str(_KERNEL), str(_SKIP_KERNEL), str(_IDENTITY_SKIP_KERNEL)
        k1 = ExternalFunction("fused_bottleneck_conv1_chunk", source_file=_KERNEL_SRC, arg_types=[activation_ty, weight_ty, stage1_ty, np.int32], compile_flags=flags + ["-DBLK_CONV1"], symbol_prefix=symbol_prefix)
        kskip = ExternalFunction("fused_bottleneck_skip_chunk", source_file=_SKIP_SRC, arg_types=[activation_ty, weight_ty, skip_ty, np.int32], compile_flags=flags + ["-DBLK_SKIP"], symbol_prefix=symbol_prefix)
        kidentity = ExternalFunction("fused_bottleneck_identity_skip", source_file=_IDENT_SRC, arg_types=[activation_ty, skip_ty], compile_flags=flags + ["-DBLK_IDENTITY"], symbol_prefix=symbol_prefix)
        k2a = ExternalFunction("fused_bottleneck_conv2_chunk", source_file=_KERNEL_SRC, arg_types=[stage1_ty, weight_ty, stage2a_ty, np.int32, np.int32], compile_flags=flags + ["-DBLK_CONV2A"], symbol_prefix=symbol_prefix + "a")
        k2b = ExternalFunction("fused_bottleneck_conv2_chunk_b", source_file=_KERNEL_SRC, arg_types=[stage1_ty, weight_ty, stage2b_ty, np.int32, np.int32], compile_flags=flags + ["-DBLK_CONV2B"], symbol_prefix=symbol_prefix + "b")
        k3 = ExternalFunction("fused_bottleneck_conv3_chunk", source_file=_KERNEL_SRC, arg_types=[stage2_ty, weight_ty, output_ty, np.int32], compile_flags=flags + ["-DBLK_CONV3"], symbol_prefix=symbol_prefix)

        input_fifo = ObjectFifo(activation_ty, depth=1, name=f"stage{block_index}_activation")
        weights_fifo = ObjectFifo(weight_ty, depth=1, name=f"stage{block_index}_weights")
        stage1_fifo = ObjectFifo(stage1_ty, depth=1, name=f"stage{block_index}_conv1_out")
        skip_fifo = ObjectFifo(skip_ty, depth=1, name=f"stage{block_index}_skip")
        stage2a_fifo = ObjectFifo(stage2a_ty, depth=1, name=f"stage{block_index}_conv2a")
        stage2b_fifo = ObjectFifo(stage2b_ty, depth=1, name=f"stage{block_index}_conv2b")
        stage2_fifo = ObjectFifo(stage2_ty, depth=1, name=f"stage{block_index}_residual_join")
        output_fifo = ObjectFifo(output_ty, depth=1, name=f"stage{block_index}_output")

        conv1_worker, conv2_worker, conv3_worker = _block_workers(chunks1, skip_chunks, chunks2, chunks3, int(nocompute))

        column = block_index
        workers.extend([
            Worker(conv1_worker, fn_args=[input_fifo.cons(), weights_fifo.cons(), stage1_fifo.prod(), skip_fifo.prod(), k1, kskip, kidentity], tile=Tile(column, 2), stack_size=0x1000),
            Worker(conv2_worker, fn_args=[stage1_fifo.cons(), weights_fifo.cons(), stage2a_fifo.prod(), k2a, 0, True], tile=Tile(column, 3), stack_size=0x1000),
            Worker(conv2_worker, fn_args=[stage1_fifo.cons(), weights_fifo.cons(), stage2b_fifo.prod(), k2b, mid_channels // 2, False], tile=Tile(column, 5), stack_size=0x1000),
            Worker(conv3_worker, fn_args=[stage2_fifo.cons(), weights_fifo.cons(), output_fifo.prod(), k3], tile=Tile(column, 4), stack_size=0x1000),
        ])
        ObjectFifoLink(
            [stage2a_fifo.cons(), stage2b_fifo.cons(), skip_fifo.cons()], stage2_fifo.prod(),
            src_offsets=[0, output_pixels * (mid_channels // 2), output_pixels * mid_channels],
        )
        input_fifos.append(input_fifo)
        weight_fifos.append(weights_fifo)
        output_fifos.append(output_fifo)

    # Link block outputs directly to the next block's activation input. Only
    # the first input and final output remain host-visible.
    for index in range(len(specs) - 1):
        previous = specs[index]
        following = specs[index + 1]
        if previous["output_width"] * previous["output_height"] * previous["output_channels"] != following["width"] * following["height"] * following["channels"]:
            raise ValueError("adjacent stage blocks must have the same flattened activation size")
        ObjectFifoLink(output_fifos[index].cons(), input_fifos[index + 1].prod())

    activation_ty = np.ndarray[(specs[0]["width"] * specs[0]["height"] * specs[0]["channels"],), np.dtype[np.int8]]
    output_ty = np.ndarray[(specs[-1]["output_width"] * specs[-1]["output_height"] * specs[-1]["output_channels"],), np.dtype[np.int8]]
    parameter_bytes = sum(spec["params_len"] for spec in specs)
    parameters_ty = np.ndarray[(parameter_bytes,), np.dtype[np.uint8]]

    tap_cons = output_fifos[0].cons() if tap else None
    tap_ty = np.ndarray[(specs[0]["output_width"] * specs[0]["output_height"] * specs[0]["output_channels"],), np.dtype[np.int8]]

    def sequence(x, packed, y, *rest):
        if tap:
            tapbuf, xprod, ycons, tapcons, *wprods = rest
        else:
            xprod, ycons, *wprods = rest
        group = TaskGroup()
        if blocked:
            first = specs[0]
            xprod.fill(
                x, wait=True, group=group,
                sizes=[first["channels"] // 8, first["width"] * first["height"], 8],
                strides=[8, first["channels"], 1],
                transfer_len=first["width"] * first["height"] * first["channels"],
            )
        else:
            xprod.fill(x, wait=True, group=group)
        group.finish()
        parameter_offset = 0
        for index, wprod in enumerate(wprods):
            spec = specs[index]
            offset = parameter_offset
            for chunk in range(spec["parameter_chunks"]):
                group = TaskGroup()
                wprod.fill(
                    packed, wait=True, sizes=[spec["slot_bytes"]], strides=[1],
                    offset=offset, transfer_len=spec["slot_bytes"], group=group,
                )
                group.finish()
                offset += spec["slot_bytes"]
            parameter_offset += spec["params_len"]
        group = TaskGroup()
        if blocked:
            last = specs[-1]
            ycons.drain(
                y, wait=True, group=group,
                sizes=[last["output_channels"] // 8, last["output_width"] * last["output_height"], 8],
                strides=[8, last["output_channels"], 1],
                transfer_len=last["output_width"] * last["output_height"] * last["output_channels"],
            )
        else:
            ycons.drain(y, wait=True, group=group)
        if tap:
            tapcons.drain(tapbuf, wait=True, group=group)
        group.finish()

    runtime = Runtime(sequence, [
        activation_ty, parameters_ty, output_ty,
        *([tap_ty] if tap else []),
        input_fifos[0].prod(), output_fifos[-1].cons(),
        *([tap_cons] if tap else []),
        *[fifo.prod() for fifo in weight_fifos],
    ])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--nocompute", type=int, default=0, help="debug: bitmask of kernels to skip (1 conv1, 2 skip, 4 conv2, 8 conv3)")
    parser.add_argument("--dbg", type=int, default=0, help="debug output mode for blocked conv3 (1 skip, 2 pre-residual main)")
    parser.add_argument("--blocked", action="store_true", help="vectorized blocked-layout kernels (needs blocked_stage packing)")
    parser.add_argument("--tap", action="store_true", help="debug: also drain block 0 output to a 4th host buffer")
    parser.add_argument("--blocks", nargs="+", required=True)
    return parser


def _compile_kwargs(opts):
    import onnx
    try:
        from .benchmark_fused_bottleneck import bind_fused_bottleneck
        from .resnet_bottleneck import plan_bottleneck_blocks
    except ImportError:
        from benchmark_fused_bottleneck import bind_fused_bottleneck
        from resnet_bottleneck import plan_bottleneck_blocks
    model = onnx.load(opts.model)
    plans = {block.prefix: block for block in plan_bottleneck_blocks(model)}
    specs = []
    previous_output = None
    for prefix in opts.blocks:
        block = plans.get(prefix)
        if block is None:
            raise ValueError(f"no bottleneck block found for {prefix!r}")
        binding = bind_fused_bottleneck(model, block)
        if opts.blocked:
            try:
                from .blocked_stage import blocked_supported
            except ImportError:
                from blocked_stage import blocked_supported
            if not blocked_supported(binding):
                raise ValueError(f"{prefix}: block shape is not supported by the blocked kernels")
        if previous_output is not None and previous_output != binding["input_shape"]:
            raise ValueError(f"{prefix}: its input shape does not match the previous block output")
        previous_output = binding["output_shape"]
        specs.append({
            "prefix": prefix,
            "width": binding["input_shape"][3], "height": binding["input_shape"][2],
            "channels": binding["input_shape"][1], "mid_channels": block.conv_plans[1].weight_shape[0],
            "output_width": binding["output_shape"][3], "output_height": binding["output_shape"][2],
            "output_channels": binding["output_shape"][1], "shift1": binding["shifts"][0],
            "shift2": binding["shifts"][1], "shift3": binding["shifts"][2],
            "residual_shift": binding["main_residual_shift"], "input_shift": binding["skip_residual_shift"],
            "chunks1": binding["chunk_counts"][0], "chunks2": binding["chunk_counts"][1],
            "chunks3": binding["chunk_counts"][2], "conv2_stride": binding["conv2_stride"][0],
            "skip_chunks": binding["skip_chunk_count"], "skip_shift": binding["skip_output_shift"] or 0,
            "residual_main_shift": binding["main_residual_shift"],
            "residual_skip_shift": binding["skip_residual_shift"],
            "conv1_mmul": True, "slot_bytes": binding["chunk_slot_bytes"],
            "parameter_chunks": len(binding["chunk_sizes"]), "params_len": binding["params"].size,
        })
    return {"stage_specs": json.dumps(specs, separators=(",", ":")), "tap": int(opts.tap), "nocompute": int(opts.nocompute), "blocked": int(opts.blocked), "dbg": int(opts.dbg)}


def main() -> None:
    opts = _parser().parse_args()
    run_design_cli(
        linked_bottleneck_stage, opts, compile_kwargs=_compile_kwargs,
        device=lambda value: device_from_args(value, n_cols=3),
    )


if __name__ == "__main__":
    main()
