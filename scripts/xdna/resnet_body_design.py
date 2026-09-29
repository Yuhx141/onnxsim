#!/usr/bin/env python3
"""Compile a whole ResNet bottleneck body as ONE xclbin: one core column per block *kind*.

A ResNet stage is one projection block plus a run of identical identity blocks. The
identity blocks of a stage share their compiled kernels, weight-slot layout and
tiling, and differ only in weights and requantization shifts, so one column of four
cores serves all of them in turn: each iteration is fed a fresh activation and that
block's weight stream, and the shifts come from a per-chunk header at run time
(``FUSED_RT_SHIFTS``). ResNet-50's 16 blocks are 8 kinds = 8 columns = the whole
NPU2 array, so the body runs in one launch with no hardware-context or PDI switches
(each costs ~0.75 ms+ on this device).

Activations between iterations round-trip through a DDR scratch buffer via the shim
DMAs (which also do the NHWC <-> 8-channel-blocked layout conversion); the runtime
sequence issues each iteration's input fill and output drain in order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Runtime, Worker, Program
from aie.iron.dataflow import ObjectFifoLink
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

try:
    from .linked_bottleneck_stage_design import _BLOCKED_KERNEL, _align4, _block_workers
except ImportError:  # run as a script
    from linked_bottleneck_stage_design import _BLOCKED_KERNEL, _align4, _block_workers

HEADER_BYTES = 64


@iron.jit
def resnet_body(
    activation: In,
    parameters: In,
    result: Out,
    scratch: In,
    *,
    body_specs: CompileTime[str],
    weight_depths: CompileTime[str] = "",
    nocompute: CompileTime[int] = 0,
    seg_gather: CompileTime[str] = "",
):
    groups = json.loads(body_specs)
    if not 1 <= len(groups) <= 8:
        raise ValueError("the body needs one to eight block-kind groups")
    input_fifos, weight_fifos, output_fifos, workers = [], [], [], []
    depths = [int(v) for v in weight_depths.split(",")] if weight_depths else [1] * len(groups)
    seg_modes = [int(v) for v in seg_gather.split(",")] if seg_gather else [0] * len(groups)
    if len(depths) != len(groups) or len(seg_modes) != len(groups):
        raise ValueError("weight_depths needs one entry per group")

    for index, spec in enumerate(groups):
        width, height, channels = spec["width"], spec["height"], spec["channels"]
        output_width, output_height = spec["output_width"], spec["output_height"]
        output_channels, mid_channels = spec["output_channels"], spec["mid_channels"]
        chunks1, chunks2, chunks3, skip_chunks = spec["chunks1"], spec["chunks2"], spec["chunks3"], spec["skip_chunks"]
        pixels, output_pixels = width * height, output_width * output_height
        taps = spec["taps"]
        outputs1 = mid_channels // chunks1
        outputs2 = (mid_channels // 2) // chunks2
        outputs3 = output_channels // chunks3
        skip_outputs = output_channels // (skip_chunks or 1)
        bytes1 = _align4(outputs1 * channels) + outputs1 * 4
        bytes2 = _align4(outputs2 * mid_channels * taps) + outputs2 * 4
        bytes3 = _align4(outputs3 * mid_channels) + outputs3 * 4
        bytes_skip = _align4(skip_outputs * channels) + skip_outputs * 4 if skip_chunks else 0
        payload = max(bytes1, bytes2, bytes3, bytes_skip)
        if payload != spec["slot_bytes"]:
            raise ValueError(f"{spec['prefix']}: packed slot does not match the compiled kernels")
        slot_bytes = payload + HEADER_BYTES
        chunk_count = chunks1 + skip_chunks + 2 * chunks2 + chunks3

        activation_ty = np.ndarray[(pixels * channels,), np.dtype[np.int8]]
        weight_ty = np.ndarray[(slot_bytes,), np.dtype[np.uint8]]
        stage1_ty = np.ndarray[((mid_channels // 8) * (height + 2) * (width + 2) * 8,), np.dtype[np.int8]]
        skip_ty = np.ndarray[(output_pixels * output_channels,), np.dtype[np.int8]]
        stage2h_ty = np.ndarray[(output_pixels * (mid_channels // 2),), np.dtype[np.int8]]
        stage2_ty = np.ndarray[(output_pixels * (mid_channels + output_channels),), np.dtype[np.int8]]
        output_ty = np.ndarray[(output_pixels * output_channels,), np.dtype[np.int8]]

        conv2_stride = spec["conv2_stride"]
        flags = [
            f"-DFUSED_W={width}", f"-DFUSED_H={height}", f"-DFUSED_C={channels}",
            f"-DFUSED_OUT_W={output_width}", f"-DFUSED_OUT_H={output_height}", f"-DFUSED_OUT_C={output_channels}",
            f"-DFUSED_CONV2_STRIDE={conv2_stride}", f"-DFUSED_SKIP_STRIDE={conv2_stride}",
            f"-DFUSED_SKIP_CHUNKS={max(skip_chunks, 1)}", f"-DFUSED_MID={mid_channels}",
            f"-DFUSED_SKIP_BIAS_OFFSET={_align4(skip_outputs * channels)}",
            f"-DFUSED_C1_CHUNKS={chunks1}", f"-DFUSED_C2_CHUNKS={chunks2}", f"-DFUSED_C3_CHUNKS={chunks3}",
            f"-DFUSED_BIAS1_OFFSET={_align4(outputs1 * channels)}",
            f"-DFUSED_BIAS2_OFFSET={_align4(outputs2 * mid_channels * taps)}",
            f"-DFUSED_BIAS3_OFFSET={_align4(outputs3 * mid_channels)}",
            "-DFUSED_RT_SHIFTS", f"-DFUSED_HDR_OFFSET={payload}", f"-DFUSED_SEG_GATHER={seg_modes[index]}",
        ]
        prefix = f"g{index}"
        src = str(_BLOCKED_KERNEL)
        k1 = ExternalFunction("fused_bottleneck_conv1_chunk", source_file=src, arg_types=[activation_ty, weight_ty, stage1_ty, np.int32], compile_flags=flags + ["-DBLK_CONV1"], symbol_prefix=prefix)
        kskip = ExternalFunction("fused_bottleneck_skip_chunk", source_file=src, arg_types=[activation_ty, weight_ty, skip_ty, np.int32], compile_flags=flags + ["-DBLK_SKIP"], symbol_prefix=prefix)
        kidentity = ExternalFunction("fused_bottleneck_identity_skip", source_file=src, arg_types=[activation_ty, skip_ty], compile_flags=flags + ["-DBLK_IDENTITY"], symbol_prefix=prefix)
        k2a = ExternalFunction("fused_bottleneck_conv2_chunk", source_file=src, arg_types=[stage1_ty, weight_ty, stage2h_ty, np.int32, np.int32], compile_flags=flags + ["-DBLK_CONV2A"], symbol_prefix=prefix + "a")
        k2b = ExternalFunction("fused_bottleneck_conv2_chunk_b", source_file=src, arg_types=[stage1_ty, weight_ty, stage2h_ty, np.int32, np.int32], compile_flags=flags + ["-DBLK_CONV2B"], symbol_prefix=prefix + "b")
        k3 = ExternalFunction("fused_bottleneck_conv3_chunk", source_file=src, arg_types=[stage2_ty, weight_ty, output_ty, np.int32], compile_flags=flags + ["-DBLK_CONV3"], symbol_prefix=prefix)

        input_fifo = ObjectFifo(activation_ty, depth=1, name=f"g{index}_activation")
        weights_fifo = ObjectFifo(weight_ty, depth=depths[index], name=f"g{index}_weights")
        stage1_fifo = ObjectFifo(stage1_ty, depth=1, name=f"g{index}_conv1_out")
        skip_fifo = ObjectFifo(skip_ty, depth=1, name=f"g{index}_skip")
        stage2a_fifo = ObjectFifo(stage2h_ty, depth=1, name=f"g{index}_conv2a")
        stage2b_fifo = ObjectFifo(stage2h_ty, depth=1, name=f"g{index}_conv2b")
        stage2_fifo = ObjectFifo(stage2_ty, depth=1, name=f"g{index}_residual_join")
        output_fifo = ObjectFifo(output_ty, depth=1, name=f"g{index}_output")
        conv1_worker, conv2_worker, conv3_worker = _block_workers(chunks1, skip_chunks, chunks2, chunks3, int(nocompute))

        out_tiles = (output_pixels + 7) // 8
        row_tiles2 = width % 8 == 0 and conv2_stride == 1 and output_width == width
        use_seg = seg_modes[index] and conv2_stride == 1 and not row_tiles2 and (output_width in (2, 4) or (output_width == 1 and output_height == 1))
        conv2_data = None if (row_tiles2 or use_seg) else taps * (mid_channels // 8) * out_tiles * 64 + 256
        conv1_data = (channels // 8) * out_tiles * 64 + 256 if skip_chunks and not (conv2_stride == 1 and output_pixels == pixels) else None
        column = index
        workers.extend([
            Worker(conv1_worker, fn_args=[input_fifo.cons(), weights_fifo.cons(), stage1_fifo.prod(), skip_fifo.prod(), k1, kskip, kidentity], tile=Tile(column, 2), stack_size=0x1000, data_size=conv1_data),
            Worker(conv2_worker, fn_args=[stage1_fifo.cons(), weights_fifo.cons(), stage2a_fifo.prod(), k2a, 0, True], tile=Tile(column, 3), stack_size=0x1000, data_size=conv2_data),
            Worker(conv2_worker, fn_args=[stage1_fifo.cons(), weights_fifo.cons(), stage2b_fifo.prod(), k2b, mid_channels // 2, False], tile=Tile(column, 5), stack_size=0x1000, data_size=conv2_data),
            Worker(conv3_worker, fn_args=[stage2_fifo.cons(), weights_fifo.cons(), output_fifo.prod(), k3], tile=Tile(column, 4), stack_size=0x1000),
        ])
        ObjectFifoLink(
            [stage2a_fifo.cons(), stage2b_fifo.cons(), skip_fifo.cons()], stage2_fifo.prod(),
            src_offsets=[0, output_pixels * (mid_channels // 2), output_pixels * mid_channels],
        )
        input_fifos.append(input_fifo)
        weight_fifos.append(weights_fifo)
        output_fifos.append(output_fifo)
        spec["_slot"], spec["_chunks"] = slot_bytes, chunk_count

    first, last = groups[0], groups[-1]
    activation_ty = np.ndarray[(first["width"] * first["height"] * first["channels"],), np.dtype[np.int8]]
    output_ty = np.ndarray[(last["output_width"] * last["output_height"] * last["output_channels"],), np.dtype[np.int8]]
    parameters_ty = np.ndarray[(sum(g["repeat"] * g["_chunks"] * g["_slot"] for g in groups),), np.dtype[np.uint8]]
    scratch_bytes = sum(g["repeat"] * g["output_width"] * g["output_height"] * g["output_channels"] for g in groups[:-1]) or 1
    scratch_ty = np.ndarray[(scratch_bytes,), np.dtype[np.int8]]

    def sequence(x, packed, y, mid, *handles):
        n = len(groups)
        xprods, ycons, wprods = handles[:n], handles[n : 2 * n], handles[2 * n :]
        weights = TaskGroup()
        offset = 0
        for g, wprod in zip(groups, wprods):
            total = g["repeat"] * g["_chunks"]
            wprod.fill(packed, group=weights, sizes=[1, 1, total, g["_slot"]], strides=[0, 0, g["_slot"], 1],
                       offset=offset, transfer_len=total * g["_slot"])
            offset += total * g["_slot"]
        source, source_offset = x, 0
        mid_offset = 0
        for gi, g in enumerate(groups):
            in_elems = g["width"] * g["height"] * g["channels"]
            out_elems = g["output_width"] * g["output_height"] * g["output_channels"]
            for rep in range(g["repeat"]):
                last_iter = gi == n - 1 and rep == g["repeat"] - 1
                step = TaskGroup()
                xprods[gi].fill(
                    source, group=step, offset=source_offset,
                    sizes=[g["channels"] // 8, g["width"] * g["height"], 8],
                    strides=[8, g["channels"], 1], transfer_len=in_elems,
                )
                if last_iter:
                    dest, dest_offset = y, 0
                else:
                    dest, dest_offset = mid, mid_offset
                ycons[gi].drain(
                    dest, wait=True, group=step, offset=dest_offset,
                    sizes=[g["output_channels"] // 8, g["output_width"] * g["output_height"], 8],
                    strides=[8, g["output_channels"], 1], transfer_len=out_elems,
                )
                step.finish()
                source, source_offset = dest, dest_offset
                mid_offset += out_elems
        weights.finish()

    runtime = Runtime(sequence, [
        activation_ty, parameters_ty, output_ty, scratch_ty,
        *[f.prod() for f in input_fifos], *[f.cons() for f in output_fifos], *[f.prod() for f in weight_fifos],
    ])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--nocompute", type=int, default=0, help="debug: bitmask of kernels to skip (1 conv1, 2 skip, 4 conv2, 8 conv3) to measure the streaming floor")
    parser.add_argument("--seg-gather", default="", help="comma-separated 0/1 per group: build small-map 3x3 tiles from row segments (no static im2col, frees tile memory) instead of a static im2col buffer (default 0: the static buffer is faster where it fits)")
    parser.add_argument("--chunk-caps", default="", help="comma-separated weight-chunk byte cap per group (0 = default); smaller chunks leave room for deeper weight FIFOs")
    parser.add_argument("--weight-depths", default="", help="comma-separated weight FIFO depth per group (2 double-buffers the weight DMA where tile memory allows; layer4 groups fit)")
    parser.add_argument("--group", nargs="+", action="append", required=True,
                        help="block prefixes served by one column, in execution order; "
                             "all blocks of a group must share shapes and chunking")
    return parser


def normalize_groups(groups):
    """Accept prefix lists or {"blocks": [...], "chunk_cap": N, "depth": D} entries."""
    out = []
    for group in groups:
        if isinstance(group, dict):
            out.append((list(group["blocks"]), group.get("chunk_cap"), int(group.get("depth", 1))))
        else:
            out.append((list(group), None, 1))
    return out


def group_specs(model, groups, caps=None):
    """Bind every group's blocks and build the compile-time spec (shared by tests/runners).

    ``caps`` optionally gives a per-group weight-chunk byte cap; a group's caps must match
    between compile and run because the packed weight layout depends on it.
    """
    try:
        from .benchmark_fused_bottleneck import bind_fused_bottleneck
        from .blocked_stage import blocked_supported
        from .resnet_bottleneck import plan_bottleneck_blocks
    except ImportError:
        from benchmark_fused_bottleneck import bind_fused_bottleneck
        from blocked_stage import blocked_supported
        from resnet_bottleneck import plan_bottleneck_blocks
    plans = {block.prefix: block for block in plan_bottleneck_blocks(model)}
    specs, bindings, previous_output = [], [], None
    for group_index, prefixes in enumerate(groups):
        binds = []
        for prefix in prefixes:
            cap = caps[group_index] if caps else None
            binding = bind_fused_bottleneck(model, plans[prefix], blocked=True, max_chunk=cap or None)
            if not blocked_supported(binding):
                raise ValueError(f"{prefix}: shape not supported by the blocked kernels")
            binds.append(binding)
        first = binds[0]
        signature = lambda b: (b["input_shape"], b["output_shape"], b["chunk_counts"], b["skip_chunk_count"],
                               b["chunk_slot_bytes"], b["conv2_stride"], b["conv2_taps"], b["projection"])
        if any(signature(b) != signature(first) for b in binds):
            raise ValueError(f"group {prefixes}: blocks differ in shape or chunking")
        if len(binds) > 1 and first["input_shape"] != first["output_shape"]:
            raise ValueError(f"group {prefixes}: repeated blocks must map their output back to their input shape")
        if previous_output is not None and previous_output != first["input_shape"]:
            raise ValueError(f"group {prefixes}: input shape does not match the previous group's output")
        previous_output = binds[-1]["output_shape"]
        plan = plans[prefixes[0]]
        specs.append({
            "prefix": prefixes[0], "repeat": len(binds),
            "width": first["input_shape"][3], "height": first["input_shape"][2], "channels": first["input_shape"][1],
            "mid_channels": plan.conv_plans[1].weight_shape[0],
            "output_width": first["output_shape"][3], "output_height": first["output_shape"][2],
            "output_channels": first["output_shape"][1],
            "chunks1": first["chunk_counts"][0], "chunks2": first["chunk_counts"][1], "chunks3": first["chunk_counts"][2],
            "skip_chunks": first["skip_chunk_count"], "conv2_stride": first["conv2_stride"][0],
            "slot_bytes": first["chunk_slot_bytes"], "taps": len(first["conv2_taps"]),
        })
        bindings.append(binds)
    return specs, bindings


def _compile_kwargs(opts):
    import onnx
    caps = [int(v) for v in opts.chunk_caps.split(",")] if opts.chunk_caps else None
    specs, _ = group_specs(onnx.load(opts.model), opts.group, caps)
    return {"body_specs": json.dumps(specs, separators=(",", ":")), "weight_depths": opts.weight_depths, "nocompute": opts.nocompute, "seg_gather": opts.seg_gather}


def main() -> None:
    opts = _parser().parse_args()
    run_design_cli(resnet_body, opts, compile_kwargs=_compile_kwargs, device=lambda value: device_from_args(value, n_cols=max(3, len(value.group))))


if __name__ == "__main__":
    main()
