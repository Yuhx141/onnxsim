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
from aie.iron.controlflow import range_
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
RT_DESC_BYTES = 192
_RT_KERNEL = Path(__file__).with_name("kernels") / "fused_bottleneck_rt.cc"


def _split_workers(chunks1, skip_chunks, chunks2, chunks3, nocompute=0, identity_bytes=None):
    """Workers for per-worker weight FIFOs: each core sees only its own chunks (no discards)."""

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
                if identity_bytes is None:
                    identity_kernel(x, residual)
                else:
                    identity_kernel(x, residual, identity_bytes)
        skip_out.release(1)
        out.release(1)
        inp.release(1)

    def conv2_worker(inp, weights, out, kernel, channel_offset, is_a):
        bundle = inp.acquire(1)
        output = out.acquire(1)
        for i in range_(chunks2):
            w = weights.acquire(1)
            if not nocompute & 4:
                kernel(bundle, w, output, i, channel_offset)
            weights.release(1)
        out.release(1)
        inp.release(1)

    def conv3_worker(inp, weights, out, kernel):
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
    split_weights: CompileTime[str] = "",
    l2_depths: CompileTime[str] = "",
    rt: CompileTime[int] = 0,
    kflags: CompileTime[str] = "",
):
    groups = json.loads(body_specs)
    if not 1 <= len(groups) <= 8:
        raise ValueError("the body needs one to eight block-kind groups")
    input_fifos, weight_fifos, output_fifos, workers = [], [], [], []
    depths = [int(v) for v in weight_depths.split(",")] if weight_depths else [1] * len(groups)
    seg_modes = [int(v) for v in seg_gather.split(",")] if seg_gather else [0] * len(groups)
    splits = [int(v) for v in split_weights.split(",")] if split_weights else [0] * len(groups)
    l2 = [int(v) for v in l2_depths.split(",")] if l2_depths else [0] * len(groups)
    if len(l2) != len(groups):
        raise ValueError("l2_depths needs one entry per group")
    if len(depths) != len(groups) or len(seg_modes) != len(groups) or len(splits) != len(groups):
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
        slot_bytes = payload + (RT_DESC_BYTES if rt else HEADER_BYTES)
        chunk_count = chunks1 + skip_chunks + 2 * chunks2 + chunks3

        activation_ty = np.ndarray[(pixels * channels,), np.dtype[np.int8]]
        weight_ty = np.ndarray[(slot_bytes,), np.dtype[np.uint8]]
        stage1_ty = np.ndarray[((mid_channels // 8) * (height + 2) * (width + 2) * 8,), np.dtype[np.int8]]
        skip_ty = np.ndarray[(output_pixels * output_channels,), np.dtype[np.int8]]
        stage2h_ty = np.ndarray[(output_pixels * (mid_channels // 2),), np.dtype[np.int8]]
        stage2_ty = np.ndarray[(output_pixels * (mid_channels + output_channels),), np.dtype[np.int8]]
        output_ty = np.ndarray[(output_pixels * output_channels,), np.dtype[np.int8]]

        conv2_stride = spec["conv2_stride"]
        out_tiles = (output_pixels + 7) // 8
        row_tiles2 = width % 8 == 0 and conv2_stride == 1 and output_width == width
        strided_skip = bool(skip_chunks) and not (conv2_stride == 1 and output_pixels == pixels)
        if rt:
            # Runtime-shaped kernels: geometry comes from the per-chunk descriptor; only buffer
            # capacities are compile-time.
            col_bytes = 64 if row_tiles2 else max(64, taps * (mid_channels // 8) * out_tiles * 64)
            skipx_bytes = max(64, (channels // 8) * out_tiles * 64) if strided_skip else 64
            flags = [f"-DRT_COL_BYTES={col_bytes}", f"-DRT_SKIPX_BYTES={skipx_bytes}"]
            src = str(_RT_KERNEL)
            identity_types = [activation_ty, skip_ty, np.int32]
        else:
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
            src = str(_BLOCKED_KERNEL)
            identity_types = [activation_ty, skip_ty]
        flags = flags + [f for f in kflags.split() if f]
        prefix = f"g{index}"
        k1 = ExternalFunction("fused_bottleneck_conv1_chunk", source_file=src, arg_types=[activation_ty, weight_ty, stage1_ty, np.int32], compile_flags=flags + ["-DBLK_CONV1"], symbol_prefix=prefix)
        kskip = ExternalFunction("fused_bottleneck_skip_chunk", source_file=src, arg_types=[activation_ty, weight_ty, skip_ty, np.int32], compile_flags=flags + ["-DBLK_SKIP"], symbol_prefix=prefix)
        kidentity = ExternalFunction("fused_bottleneck_identity_skip", source_file=src, arg_types=identity_types, compile_flags=flags + ["-DBLK_IDENTITY"], symbol_prefix=prefix)
        k2a = ExternalFunction("fused_bottleneck_conv2_chunk", source_file=src, arg_types=[stage1_ty, weight_ty, stage2h_ty, np.int32, np.int32], compile_flags=flags + ["-DBLK_CONV2A"], symbol_prefix=prefix + "a")
        k2b = ExternalFunction("fused_bottleneck_conv2_chunk_b", source_file=src, arg_types=[stage1_ty, weight_ty, stage2h_ty, np.int32, np.int32], compile_flags=flags + ["-DBLK_CONV2B"], symbol_prefix=prefix + "b")
        k3 = ExternalFunction("fused_bottleneck_conv3_chunk", source_file=src, arg_types=[stage2_ty, weight_ty, output_ty, np.int32], compile_flags=flags + ["-DBLK_CONV3"], symbol_prefix=prefix)

        input_fifo = ObjectFifo(activation_ty, depth=1, name=f"g{index}_activation")
        split = bool(splits[index])
        if split:
            # Per-worker weight FIFOs: conv1 (+skip), conv2a, conv2b, conv3 each get their own
            # shim stream and see only their own chunks.
            w_fifos = [ObjectFifo(weight_ty, depth=depths[index], name=f"g{index}_w{tag}") for tag in ("1", "2a", "2b", "3")]
            weights_fifo = None
        else:
            if l2[index]:
                # Stage weights in the column's memtile: the shim fills a deep L2 FIFO ahead of
                # compute and the memtile forwards chunks to the four cores as they need them.
                l2_fifo = ObjectFifo(weight_ty, depth=l2[index], name=f"g{index}_weights_l2")
                weights_fifo = l2_fifo.cons().forward(depth=depths[index], name=f"g{index}_weights")
            else:
                l2_fifo = None
                weights_fifo = ObjectFifo(weight_ty, depth=depths[index], name=f"g{index}_weights")
            w_fifos = None
        stage1_fifo = ObjectFifo(stage1_ty, depth=1, name=f"g{index}_conv1_out")
        skip_fifo = ObjectFifo(skip_ty, depth=1, name=f"g{index}_skip")
        stage2a_fifo = ObjectFifo(stage2h_ty, depth=1, name=f"g{index}_conv2a")
        stage2b_fifo = ObjectFifo(stage2h_ty, depth=1, name=f"g{index}_conv2b")
        stage2_fifo = ObjectFifo(stage2_ty, depth=1, name=f"g{index}_residual_join")
        output_fifo = ObjectFifo(output_ty, depth=1, name=f"g{index}_output")
        make_workers = _split_workers if split else _block_workers
        conv1_worker, conv2_worker, conv3_worker = make_workers(chunks1, skip_chunks, chunks2, chunks3, int(nocompute), pixels * channels if rt else None)
        wc1, wc2a, wc2b, wc3 = ([f.cons() for f in w_fifos] if split else [weights_fifo.cons() for _ in range(4)])

        out_tiles = (output_pixels + 7) // 8
        row_tiles2 = width % 8 == 0 and conv2_stride == 1 and output_width == width
        use_seg = seg_modes[index] and conv2_stride == 1 and not row_tiles2 and (output_width in (2, 4) or (output_width == 1 and output_height == 1))
        conv2_data = None if (row_tiles2 or use_seg) else taps * (mid_channels // 8) * out_tiles * 64 + 256
        conv1_data = (channels // 8) * out_tiles * 64 + 256 if skip_chunks and not (conv2_stride == 1 and output_pixels == pixels) else None
        column = index
        workers.extend([
            Worker(conv1_worker, fn_args=[input_fifo.cons(), wc1, stage1_fifo.prod(), skip_fifo.prod(), k1, kskip if skip_chunks else k1, k1 if skip_chunks else kidentity], tile=Tile(column, 2), stack_size=0x1000, data_size=conv1_data),
            Worker(conv2_worker, fn_args=[stage1_fifo.cons(), wc2a, stage2a_fifo.prod(), k2a, 0, True], tile=Tile(column, 3), stack_size=0x1000, data_size=conv2_data),
            Worker(conv2_worker, fn_args=[stage1_fifo.cons(), wc2b, stage2b_fifo.prod(), k2b, mid_channels // 2, False], tile=Tile(column, 5), stack_size=0x1000, data_size=conv2_data),
            Worker(conv3_worker, fn_args=[stage2_fifo.cons(), wc3, output_fifo.prod(), k3], tile=Tile(column, 4), stack_size=0x1000),
        ])
        ObjectFifoLink(
            [stage2a_fifo.cons(), stage2b_fifo.cons(), skip_fifo.cons()], stage2_fifo.prod(),
            src_offsets=[0, output_pixels * (mid_channels // 2), output_pixels * mid_channels],
        )
        input_fifos.append(input_fifo)
        weight_fifos.append(w_fifos if split else [l2_fifo or weights_fifo])
        output_fifos.append(output_fifo)
        spec["_slot"], spec["_chunks"] = slot_bytes, chunk_count

    first, last = groups[0], groups[-1]
    activation_ty = np.ndarray[(first["width"] * first["height"] * first["channels"],), np.dtype[np.int8]]
    output_ty = np.ndarray[(last["output_width"] * last["output_height"] * last["output_channels"],), np.dtype[np.int8]]
    parameters_ty = np.ndarray[(sum(g["repeat"] * g["_chunks"] * g["_slot"] for g in groups),), np.dtype[np.uint8]]
    # Every iteration except the very last drains its output to scratch (the last group's
    # non-final iterations included).
    scratch_bytes = sum(g["repeat"] * g["output_width"] * g["output_height"] * g["output_channels"] for g in groups)
    scratch_bytes -= last["output_width"] * last["output_height"] * last["output_channels"]
    scratch_bytes = max(scratch_bytes, 1)
    scratch_ty = np.ndarray[(scratch_bytes,), np.dtype[np.int8]]

    def sequence(x, packed, y, mid, *handles):
        n = len(groups)
        xprods, ycons, wprods = handles[:n], handles[n : 2 * n], handles[2 * n :]
        weights = TaskGroup()
        offset = 0
        cursor = 0
        for g, is_split in zip(groups, splits):
            total = g["repeat"] * g["_chunks"]
            slot = g["_slot"]
            if is_split:
                # Slice each block's [conv1, skip, conv2a, conv2b, conv3] chunk run into the four
                # per-worker streams (same repeat stride), each on its own shim channel.
                bounds = [0, g["chunks1"] + g["skip_chunks"], g["chunks2"], g["chunks2"], g["chunks3"]]
                start = 0
                for part in range(4):
                    n_chunks = bounds[part + 1]
                    wprods[cursor].fill(
                        packed, group=weights, sizes=[g["repeat"], 1, n_chunks, slot],
                        strides=[g["_chunks"] * slot, 0, slot, 1],
                        offset=offset + start * slot, transfer_len=n_chunks * slot,
                    )
                    start += n_chunks
                    cursor += 1
            else:
                wprods[cursor].fill(
                    packed, group=weights, sizes=[1, 1, total, slot], strides=[0, 0, slot, 1],
                    offset=offset, transfer_len=total * slot,
                )
                cursor += 1
            offset += total * slot
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
        *[f.prod() for f in input_fifos], *[f.cons() for f in output_fifos], *[f.prod() for fifos in weight_fifos for f in fifos],
    ])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--nocompute", type=int, default=0, help="debug: bitmask of kernels to skip (1 conv1, 2 skip, 4 conv2, 8 conv3) to measure the streaming floor")
    parser.add_argument("--seg-gather", default="", help="comma-separated 0/1 per group: build small-map 3x3 tiles from row segments (no static im2col, frees tile memory) instead of a static im2col buffer (default 0: the static buffer is faster where it fits)")
    parser.add_argument("--split-weights", default="", help="comma-separated 0/1 per group: give each of the four cores its own weight FIFO/shim stream (4 shim MM2S channels instead of 1; total channels are limited to 16)")
    parser.add_argument("--cols", type=int, default=0, help="array width to compile for (default: one column per group, min 3); widen it to get more shim DMA channels")
    parser.add_argument("--l2-depths", default="", help="comma-separated memtile weight-staging depth per group (0 = stream shim -> cores directly); the shim prefetches this many weight slots into the memtile ahead of compute")
    parser.add_argument("--rt", action="store_true", help="use the runtime-shaped kernels (fused_bottleneck_rt.cc): geometry from a per-chunk descriptor instead of compile-time flags; needs blocked_stage.pack_rt_params packing")
    parser.add_argument("--kflags", default="", help="extra compiler flags for the kernels (debug/profiling, e.g. -DRT_SKIP_GATHER)")
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
    return {"body_specs": json.dumps(specs, separators=(",", ":")), "weight_depths": opts.weight_depths, "nocompute": opts.nocompute, "seg_gather": opts.seg_gather, "split_weights": opts.split_weights, "l2_depths": opts.l2_depths, "rt": 1 if opts.rt else 0, "kflags": opts.kflags}


def main() -> None:
    opts = _parser().parse_args()
    run_design_cli(resnet_body, opts, compile_kwargs=_compile_kwargs, device=lambda value: device_from_args(value, n_cols=value.cols or max(3, len(value.group))))


if __name__ == "__main__":
    main()
