#!/usr/bin/env python3
"""ResNet bottleneck body with ONE core column per stage (projection block + identity blocks).

With the runtime-shaped kernels (``kernels/fused_bottleneck_rt.cc``) a column is not tied to a
block shape: the geometry lives in a descriptor at the start of every weight slot. A ResNet stage
(one projection block followed by identical identity blocks) therefore fits in a single column of
four cores that runs the projection block once and then the identity blocks in turn. ResNet-50's
four stages use 4 of the 8 columns, leaving columns (and shim channels) for other work, e.g. the
stem Conv and MaxPool.

Every FIFO object is sized to the largest block of the stage; DDR transfers are always whole
objects (padded), and activations between blocks are kept in the linear 8-channel-blocked layout
(``[C/8][pixel][8]``) so no layout conversion happens on the DMAs; the host converts NHWC <-> blocked
once at the network input and output (a few KB).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifoLink
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

RT_DESC_BYTES = 192
_RT_KERNEL = Path(__file__).with_name("kernels") / "fused_bottleneck_rt.cc"


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _stage_workers(kinds, repeat, nocompute=0):
    """Worker bodies for a column: block ``kinds[0]`` once, then ``kinds[1]`` ``repeat`` times.

    Each kind is ``{"c1", "skip", "c2", "c3", "ident_bytes"}``. The weight FIFO is broadcast to the
    column's four cores, so every core discards the chunks that belong to the others.
    """

    def discard(weights, count):
        if count:
            for _ in range_(count):
                weights.acquire(1)
                weights.release(1)

    def run_conv1(kind, inp, weights, out, skip_out, k1, kskip, kident):
        x = inp.acquire(1)
        bundle = out.acquire(1)
        for i in range_(kind["c1"]):
            w = weights.acquire(1)
            if not nocompute & 1:
                k1(x, w, bundle, i)
            weights.release(1)
        residual = skip_out.acquire(1)
        if kind["skip"]:
            for i in range_(kind["skip"]):
                w = weights.acquire(1)
                if not nocompute & 2:
                    kskip(x, w, residual, i)
                weights.release(1)
        elif not nocompute & 2:
            kident(x, residual, kind["ident_bytes"])
        skip_out.release(1)
        out.release(1)
        inp.release(1)
        discard(weights, 2 * kind["c2"] + kind["c3"])

    def run_conv2(kind, inp, weights, out, kern, is_a):
        discard(weights, kind["c1"] + kind["skip"] + (0 if is_a else kind["c2"]))
        bundle = inp.acquire(1)
        output = out.acquire(1)
        for i in range_(kind["c2"]):
            w = weights.acquire(1)
            if not nocompute & 4:
                kern(bundle, w, output, i, 0)
            weights.release(1)
        out.release(1)
        inp.release(1)
        discard(weights, (kind["c2"] if is_a else 0) + kind["c3"])

    def run_conv3(kind, inp, weights, out, kern):
        discard(weights, kind["c1"] + kind["skip"] + 2 * kind["c2"])
        bundle = inp.acquire(1)
        output = out.acquire(1)
        for i in range_(kind["c3"]):
            w = weights.acquire(1)
            if not nocompute & 8:
                kern(bundle, w, output, i)
            weights.release(1)
        out.release(1)
        inp.release(1)

    def loop(run):
        run(kinds[0])
        if repeat:
            for _ in range_(repeat):
                run(kinds[1])

    def conv1_worker(inp, weights, out, skip_out, k1, kskip, kident):
        loop(lambda kind: run_conv1(kind, inp, weights, out, skip_out, k1, kskip, kident))

    def conv2_worker(inp, weights, out, kern, is_a):
        loop(lambda kind: run_conv2(kind, inp, weights, out, kern, is_a))

    def conv3_worker(inp, weights, out, kern):
        loop(lambda kind: run_conv3(kind, inp, weights, out, kern))

    return conv1_worker, conv2_worker, conv3_worker


def _split_stage_workers(kinds, repeat, nocompute=0):
    """Worker bodies when each core has its own weight FIFO (no broadcast, so nothing to discard)."""

    def run_conv1(kind, inp, weights, out, skip_out, k1, kskip, kident):
        x = inp.acquire(1)
        bundle = out.acquire(1)
        for i in range_(kind["c1"]):
            w = weights.acquire(1)
            if not nocompute & 1:
                k1(x, w, bundle, i)
            weights.release(1)
        residual = skip_out.acquire(1)
        if kind["skip"]:
            for i in range_(kind["skip"]):
                w = weights.acquire(1)
                if not nocompute & 2:
                    kskip(x, w, residual, i)
                weights.release(1)
        elif not nocompute & 2:
            kident(x, residual, kind["ident_bytes"])
        skip_out.release(1)
        out.release(1)
        inp.release(1)

    def run_conv2(kind, inp, weights, out, kern, is_a):
        bundle = inp.acquire(1)
        output = out.acquire(1)
        for i in range_(kind["c2"]):
            w = weights.acquire(1)
            if not nocompute & 4:
                kern(bundle, w, output, i, 0)
            weights.release(1)
        out.release(1)
        inp.release(1)

    def run_conv3(kind, inp, weights, out, kern):
        bundle = inp.acquire(1)
        output = out.acquire(1)
        for i in range_(kind["c3"]):
            w = weights.acquire(1)
            if not nocompute & 8:
                kern(bundle, w, output, i)
            weights.release(1)
        out.release(1)
        inp.release(1)

    def loop(run):
        run(kinds[0])
        if repeat:
            for _ in range_(repeat):
                run(kinds[1])

    def conv1_worker(inp, weights, out, skip_out, k1, kskip, kident):
        loop(lambda kind: run_conv1(kind, inp, weights, out, skip_out, k1, kskip, kident))

    def conv2_worker(inp, weights, out, kern, is_a):
        loop(lambda kind: run_conv2(kind, inp, weights, out, kern, is_a))

    def conv3_worker(inp, weights, out, kern):
        loop(lambda kind: run_conv3(kind, inp, weights, out, kern))

    return conv1_worker, conv2_worker, conv3_worker


@iron.jit
def resnet_stages(
    activation: In,
    parameters: In,
    result: Out,
    scratch: In,
    *,
    stage_specs: CompileTime[str],
    weight_depths: CompileTime[str] = "",
    nocompute: CompileTime[int] = 0,
    stem_spec: CompileTime[str] = "",
    split_weights: CompileTime[str] = "",
    kflags: CompileTime[str] = "",
):
    cols = json.loads(stage_specs)
    n = len(cols)
    depths = [int(v) for v in weight_depths.split(",")] if weight_depths else [1] * n
    input_fifos, weight_fifos, output_fifos, workers = [], [], [], []
    splits = [int(v) for v in split_weights.split(",")] if split_weights else [0] * n
    stem = json.loads(stem_spec) if stem_spec else None
    stem_handles = []
    stem_bytes = 0
    if stem:
        # On-device stem Conv (as an im2col GEMM) + MaxPool in one extra column: core 0 runs the GEMM chunk
        # by chunk, core 1 assembles the map and pools it. Its own three shim streams: image chunks and
        # weights in, pooled map out.
        sc, chunks = stem["column"], stem["chunks"]
        stem_bytes = stem["slot"]
        in_ty = np.ndarray[(stem["chunk_in"],), np.dtype[np.int8]]
        sw_ty = np.ndarray[(stem["slot"],), np.dtype[np.uint8]]
        so_ty = np.ndarray[(stem["chunk_out"],), np.dtype[np.uint8]]
        po_ty = np.ndarray[(stem["pool_out"],), np.dtype[np.uint8]]
        pool_flags = [f"-DPOOL_H={stem['map_h']}", f"-DPOOL_W={stem['map_w']}", f"-DPOOL_OB={stem['out_blocks']}", f"-DPOOL_CHUNK_PX={stem['chunk_px']}", f"-DPOOL_CHUNKS={stem['chunks']}"]
        src = str(_RT_KERNEL)
        kstem = ExternalFunction("fused_stem_chunk", source_file=src, arg_types=[in_ty, sw_ty, so_ty, np.int32], compile_flags=["-DBLK_STEM"], symbol_prefix="stem")
        kpool = ExternalFunction("fused_pool_step", source_file=src, arg_types=[so_ty, np.int32, po_ty], compile_flags=pool_flags + ["-DBLK_POOL"], symbol_prefix="pool")
        stem_in = ObjectFifo(in_ty, depth=1, name="stem_image")
        stem_w = ObjectFifo(sw_ty, depth=1, name="stem_weights")
        stem_out = ObjectFifo(so_ty, depth=1, name="stem_out")
        pool_out = ObjectFifo(po_ty, depth=1, name="pool_out")

        def stem_worker(inp, weights, out, kern):
            wc = weights.acquire(1)
            for _ in range_(chunks):
                a = inp.acquire(1)
                o = out.acquire(1)
                kern(a, wc, o, 0)
                out.release(1)
                inp.release(1)
            weights.release(1)

        def pool_worker(inp, out, step):
            r = out.acquire(1)  # held while the chunks arrive; the last chunk triggers the pooling
            for c in range(chunks):
                o = inp.acquire(1)
                step(o, c, r)
                inp.release(1)
            out.release(1)

        workers.append(Worker(stem_worker, fn_args=[stem_in.cons(), stem_w.cons(), stem_out.prod(), kstem], tile=Tile(sc, 2), stack_size=0x1000))
        workers.append(Worker(pool_worker, fn_args=[stem_out.cons(), pool_out.prod(), kpool], tile=Tile(sc, 3), stack_size=0x800, data_size=stem["map_bytes"] + 256))
        stem_handles = [stem_in, stem_w, pool_out]

    for index, col in enumerate(cols):
        act_obj, out_obj = col["act_obj"], col["out_obj"]
        slot = col["slot"] + RT_DESC_BYTES
        act_ty = np.ndarray[(act_obj,), np.dtype[np.int8]]
        w_ty = np.ndarray[(slot,), np.dtype[np.uint8]]
        stage1_ty = np.ndarray[(col["stage1_obj"],), np.dtype[np.int8]]
        skip_ty = np.ndarray[(out_obj,), np.dtype[np.int8]]
        half_ty = np.ndarray[(col["half_obj"],), np.dtype[np.int8]]
        join_ty = np.ndarray[(col["half_obj"] * 2 + out_obj,), np.dtype[np.int8]]
        out_ty = np.ndarray[(out_obj,), np.dtype[np.int8]]
        flags = [f"-DRT_COL_BYTES={col['col_bytes']}", f"-DRT_SKIPX_BYTES={col['skipx_bytes']}"] + [f for f in kflags.split() if f]
        src = str(_RT_KERNEL)
        prefix = f"s{index}"
        ident_ty = [act_ty, skip_ty, np.int32]
        k1 = ExternalFunction("fused_bottleneck_conv1_chunk", source_file=src, arg_types=[act_ty, w_ty, stage1_ty, np.int32], compile_flags=flags + ["-DBLK_CONV1"], symbol_prefix=prefix)
        kskip = ExternalFunction("fused_bottleneck_skip_chunk", source_file=src, arg_types=[act_ty, w_ty, skip_ty, np.int32], compile_flags=flags + ["-DBLK_SKIP"], symbol_prefix=prefix)
        kident = ExternalFunction("fused_bottleneck_identity_skip", source_file=src, arg_types=ident_ty, compile_flags=flags + ["-DBLK_IDENTITY"], symbol_prefix=prefix)
        k2a = ExternalFunction("fused_bottleneck_conv2_chunk", source_file=src, arg_types=[stage1_ty, w_ty, half_ty, np.int32, np.int32], compile_flags=flags + ["-DBLK_CONV2A"], symbol_prefix=prefix + "a")
        k2b = ExternalFunction("fused_bottleneck_conv2_chunk_b", source_file=src, arg_types=[stage1_ty, w_ty, half_ty, np.int32, np.int32], compile_flags=flags + ["-DBLK_CONV2B"], symbol_prefix=prefix + "b")
        k3 = ExternalFunction("fused_bottleneck_conv3_chunk", source_file=src, arg_types=[join_ty, w_ty, out_ty, np.int32], compile_flags=flags + ["-DBLK_CONV3"], symbol_prefix=prefix)

        input_fifo = ObjectFifo(act_ty, depth=1, name=f"s{index}_activation")
        split = bool(splits[index])
        if split:
            w_fifos = [ObjectFifo(w_ty, depth=depths[index], name=f"s{index}_w{tag}") for tag in ("1", "2a", "2b", "3")]
        else:
            w_fifos = [ObjectFifo(w_ty, depth=depths[index], name=f"s{index}_weights")]
        stage1_fifo = ObjectFifo(stage1_ty, depth=1, name=f"s{index}_conv1_out")
        skip_fifo = ObjectFifo(skip_ty, depth=1, name=f"s{index}_skip")
        half_a = ObjectFifo(half_ty, depth=1, name=f"s{index}_conv2a")
        half_b = ObjectFifo(half_ty, depth=1, name=f"s{index}_conv2b")
        join_fifo = ObjectFifo(join_ty, depth=1, name=f"s{index}_residual_join")
        output_fifo = ObjectFifo(out_ty, depth=1, name=f"s{index}_output")

        kinds = col["kinds"]
        conv1_worker, conv2_worker, conv3_worker = (_split_stage_workers if split else _stage_workers)(kinds, col["repeat"], int(nocompute))
        wc = [f.cons() for f in w_fifos] if split else [w_fifos[0].cons() for _ in range(4)]
        # A kernel is only linked into a core if some worker references it.
        has_proj = any(k["skip"] for k in kinds)
        has_id = any(not k["skip"] for k in kinds)
        kskip_arg = kskip if has_proj else k1
        kident_arg = kident if has_id else k1
        column = col["column"]
        workers.extend([
            Worker(conv1_worker, fn_args=[input_fifo.cons(), wc[0], stage1_fifo.prod(), skip_fifo.prod(), k1, kskip_arg, kident_arg],
                   tile=Tile(column, 2), stack_size=0x1000, data_size=col["conv1_data"]),
            Worker(conv2_worker, fn_args=[stage1_fifo.cons(), wc[1], half_a.prod(), k2a, True],
                   tile=Tile(column, 3), stack_size=0x1000, data_size=col["conv2_data"]),
            Worker(conv2_worker, fn_args=[stage1_fifo.cons(), wc[2], half_b.prod(), k2b, False],
                   tile=Tile(column, 5), stack_size=0x1000, data_size=col["conv2_data"]),
            Worker(conv3_worker, fn_args=[join_fifo.cons(), wc[3], output_fifo.prod(), k3],
                   tile=Tile(column, 4), stack_size=0x1000),
        ])
        ObjectFifoLink([half_a.cons(), half_b.cons(), skip_fifo.cons()], join_fifo.prod(),
                       src_offsets=[0, col["half_obj"], 2 * col["half_obj"]])
        input_fifos.append(input_fifo)
        weight_fifos.append(w_fifos)
        output_fifos.append(output_fifo)

    act_ty = np.ndarray[((stem["chunks"] * stem["chunk_in"]) if stem else cols[0]["act_obj"],), np.dtype[np.int8]]
    out_ty = np.ndarray[(cols[-1]["out_obj"],), np.dtype[np.int8]]
    params_ty = np.ndarray[(stem_bytes + sum(c["stream_chunks"] * (c["slot"] + RT_DESC_BYTES) for c in cols),), np.dtype[np.uint8]]
    scratch_bytes = 0
    for col in cols:
        scratch_bytes += col["repeat"] * col["out_obj"] + col["out_obj"]
    scratch_bytes = scratch_bytes - cols[-1]["out_obj"] + max(c["act_obj"] for c in cols)
    if stem:
        scratch_bytes += cols[0]["act_obj"]  # slot for the pooled map feeding the first stage
    scratch_ty = np.ndarray[(scratch_bytes,), np.dtype[np.int8]]

    def sequence(x, packed, y, mid, *handles):
        xprods, ycons = handles[:n], handles[n : 2 * n]
        weight_handles = handles[2 * n : 2 * n + sum(4 if sp else 1 for sp in splits)]
        weights = TaskGroup()
        offset = stem_bytes
        cursor = 0
        for col, is_split in zip(cols, splits):
            slot_total = col["slot"] + RT_DESC_BYTES
            if is_split:
                # Per-core streams: each core's slice of the first block once, then its slice of every
                # following block at the block stride (the blocks after the first share one kind).
                first, rest = col["kinds"][0], (col["kinds"][1] if col["repeat"] else None)
                first_chunks = first["c1"] + first["skip"] + 2 * first["c2"] + first["c3"]
                for kind, start, reps, base in (
                    (first, offset, 1, 0),
                    (rest, offset + first_chunks * slot_total, col["repeat"], first_chunks),
                ):
                    if not kind or not reps:
                        continue
                    block_chunks = kind["c1"] + kind["skip"] + 2 * kind["c2"] + kind["c3"]
                    counts = [kind["c1"] + kind["skip"], kind["c2"], kind["c2"], kind["c3"]]
                    begin = 0
                    for part in range(4):
                        weight_handles[cursor + part].fill(
                            packed, group=weights, offset=start + begin * slot_total,
                            sizes=[reps, 1, counts[part], slot_total], strides=[block_chunks * slot_total, 0, slot_total, 1],
                            transfer_len=counts[part] * slot_total,
                        )
                        begin += counts[part]
                cursor += 4
            else:
                total = col["stream_chunks"]
                weight_handles[cursor].fill(packed, group=weights, sizes=[1, 1, total, slot_total], strides=[0, 0, slot_total, 1],
                                            offset=offset, transfer_len=total * slot_total)
                cursor += 1
            offset += col["stream_chunks"] * slot_total
        source, source_offset, mid_offset = x, 0, 0
        if stem:
            image_prod, stem_w_prod, pool_cons = handles[2 * n + len(weight_handles) :]
            pre = TaskGroup()
            stem_w_prod.fill(packed, group=pre, offset=0, sizes=[1, 1, 1, stem["slot"]], strides=[0, 0, 0, 1],
                             transfer_len=stem["slot"])
            image_prod.fill(x, group=pre, offset=0, sizes=[1, 1, stem["chunks"], stem["chunk_in"]],
                            strides=[0, 0, stem["chunk_in"], 1], transfer_len=stem["chunks"] * stem["chunk_in"])
            pool_cons.drain(mid, wait=True, group=pre, offset=0, sizes=[1, 1, 1, stem["pool_out"]],
                            strides=[0, 0, 0, 1], transfer_len=stem["pool_out"])
            pre.finish()
            source, source_offset, mid_offset = mid, 0, cols[0]["act_obj"]
        for ci, col in enumerate(cols):
            for it in range(col["repeat"] + 1):
                final = ci == n - 1 and it == col["repeat"]
                step = TaskGroup()
                xprods[ci].fill(source, group=step, offset=source_offset,
                                sizes=[1, 1, 1, col["act_obj"]], strides=[0, 0, 0, 1], transfer_len=col["act_obj"])
                if final:
                    dest, dest_offset = y, 0
                else:
                    dest, dest_offset = mid, mid_offset
                ycons[ci].drain(dest, wait=True, group=step, offset=dest_offset,
                                sizes=[1, 1, 1, col["out_obj"]], strides=[0, 0, 0, 1], transfer_len=col["out_obj"])
                step.finish()
                source, source_offset = dest, dest_offset
                mid_offset += col["out_obj"]
        weights.finish()

    runtime = Runtime(sequence, [
        act_ty, params_ty, out_ty, scratch_ty,
        *[f.prod() for f in input_fifos], *[f.cons() for f in output_fifos],
        *[f.prod() for fifos in weight_fifos for f in fifos],
        *([stem_handles[0].prod(), stem_handles[1].prod(), stem_handles[2].cons()] if stem else []),
    ])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


def stage_specs(model, stages, caps=None, first_column=0):
    """Bind every stage's blocks and build the per-column compile-time spec (shared with the runner)."""
    try:
        from .benchmark_fused_bottleneck import bind_fused_bottleneck
        from .blocked_stage import blocked_supported
        from .resnet_bottleneck import plan_bottleneck_blocks
    except ImportError:
        from benchmark_fused_bottleneck import bind_fused_bottleneck
        from blocked_stage import blocked_supported
        from resnet_bottleneck import plan_bottleneck_blocks
    plans = {block.prefix: block for block in plan_bottleneck_blocks(model)}
    cols, bindings, previous_output = [], [], None
    for ci, prefixes in enumerate(stages):
        binds = []
        cap = caps[ci] if caps and caps[ci] else None
        for j, prefix in enumerate(prefixes):
            binding = bind_fused_bottleneck(model, plans[prefix], blocked=True, max_chunk=cap)
            if j == 0 and len(prefixes) > 1 and not cap:
                # identity blocks adopt the first block's slot as their chunk cap, so the stage's
                # common slot is not larger than what its first block needs
                cap = int(binding["chunk_slot_bytes"])
            if not blocked_supported(binding):
                raise ValueError(f"{prefix}: shape not supported by the blocked kernels")
            binds.append(binding)
        outs = {tuple(b["output_shape"]) for b in binds}
        if len(outs) != 1:
            raise ValueError(f"stage {prefixes}: blocks must share one output shape")
        if len(binds) > 2 and any(_kind(b) != _kind(binds[1]) for b in binds[1:]):
            raise ValueError(f"stage {prefixes}: blocks after the first must share one kind")
        if previous_output is not None and previous_output != binds[0]["input_shape"]:
            raise ValueError(f"stage {prefixes}: input shape does not match the previous stage's output")
        previous_output = binds[-1]["output_shape"]
        mid = int(binds[0]["raw_weights"]["w1"].shape[0])
        out_c, oh, ow = binds[0]["output_shape"][1:4]
        kinds, act_bytes, stage1, col_bytes, skipx, slot = [], [], 0, 64, 64, 0
        for b in binds:
            _, ch, h, w = b["input_shape"]
            op = oh * ow
            tiles = (op + 7) // 8
            taps = len(b["conv2_taps"])
            row_tiles2 = w % 8 == 0 and tuple(b["conv2_stride"]) == (1, 1) and ow == w
            strided = b["skip_chunk_count"] > 0 and not (tuple(b["conv2_stride"]) == (1, 1) and op == h * w)
            act_bytes.append(ch * h * w)
            stage1 = max(stage1, (mid // 8) * (h + 2) * (w + 2) * 8)
            col_bytes = max(col_bytes, 64 if row_tiles2 else taps * (mid // 8) * tiles * 64)
            skipx = max(skipx, (ch // 8) * tiles * 64 if strided else 64)
            slot = max(slot, int(b["chunk_slot_bytes"]))
            c1, c2, c3 = b["chunk_counts"]
            kinds.append({"c1": c1, "skip": int(b["skip_chunk_count"]), "c2": c2, "c3": c3, "ident_bytes": ch * h * w})
        stream_chunks = sum(
            k["c1"] + k["skip"] + 2 * k["c2"] + k["c3"] for k in kinds
        )
        first_kind = kinds[0]
        rest = kinds[1] if len(kinds) > 1 else None
        conv2_data = max(64, col_bytes) + 256
        conv1_data = skipx + 256 if skipx > 64 else None
        cols.append({
            "column": first_column + ci, "prefix": prefixes[0], "blocks": list(prefixes), "repeat": len(binds) - 1,
            "mid": mid, "slot": slot, "act_obj": max(act_bytes), "out_obj": out_c * oh * ow,
            "stage1_obj": stage1, "half_obj": oh * ow * (mid // 2),
            "col_bytes": col_bytes, "skipx_bytes": skipx,
            "conv1_data": conv1_data, "conv2_data": None if col_bytes <= 64 else conv2_data,
            "kinds": [first_kind] + ([rest] if rest else []), "stream_chunks": stream_chunks,
            "block_chunks": [k["c1"] + k["skip"] + 2 * k["c2"] + k["c3"] for k in kinds],
            "input_bytes": act_bytes[0],
        })
        bindings.append(binds)
    return cols, bindings


def _kind(binding):
    return (binding["chunk_counts"], binding["skip_chunk_count"], tuple(binding["conv2_taps"]),
            tuple(binding["input_shape"]), tuple(binding["conv2_stride"]))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stage", nargs="+", action="append", required=True,
                        help="block prefixes of one stage in execution order (projection block first)")
    parser.add_argument("--chunk-caps", default="", help="comma-separated weight-chunk byte cap per stage (0 = default)")
    parser.add_argument("--weight-depths", default="")
    parser.add_argument("--nocompute", type=int, default=0)
    parser.add_argument("--kflags", default="", help="extra kernel compiler flags (profiling, e.g. -DRT_SKIP_GATHER, -DRT_REPEAT_GEMM=4)")
    parser.add_argument("--split-weights", default="", help="comma-separated 0/1 per stage: give each of the four cores its own weight stream (4 shim MM2S channels instead of 1)")
    parser.add_argument("--cols", type=int, default=0, help="array width to compile for (default: stages + stem column, min 3); use 8 to reach all shim DMA channels")
    parser.add_argument("--stem", action="store_true", help="also run the stem Conv + MaxPool on the device (extra column after the stages)")
    return parser


def stem_spec(model, column):
    """Compile-time description of the on-device stem/pool column."""
    try:
        from . import stem_pool
    except ImportError:
        import stem_pool
    stem = stem_pool.extract_stem(model)
    g = stem_pool.geometry(stem)
    out_blocks = g["out_channels"] // 8
    return {
        "column": column, "chunks": g["chunks"], "chunk_in": g["k_pad"] * stem_pool.CHUNK_PIXELS,
        "chunk_out": g["out_channels"] * stem_pool.CHUNK_PIXELS, "slot": stem_pool.stem_slot_bytes(stem),
        "pool_out": g["out_channels"] * (g["oh"] // 2) * (g["ow"] // 2), "map_h": g["oh"], "map_w": g["ow"],
        "out_blocks": out_blocks, "chunk_px": stem_pool.CHUNK_PIXELS, "map_bytes": g["out_channels"] * g["pixels"],
    }


def _compile_kwargs(opts):
    import onnx
    model = onnx.load(opts.model)
    caps = [int(v) for v in opts.chunk_caps.split(",")] if opts.chunk_caps else None
    cols, _ = stage_specs(model, opts.stage, caps)
    kwargs = {"stage_specs": json.dumps(cols, separators=(",", ":")), "weight_depths": opts.weight_depths, "nocompute": opts.nocompute, "split_weights": opts.split_weights, "kflags": opts.kflags}
    if opts.stem:
        kwargs["stem_spec"] = json.dumps(stem_spec(model, len(cols)), separators=(",", ":"))
    return kwargs


def main() -> None:
    opts = _parser().parse_args()
    run_design_cli(resnet_stages, opts, compile_kwargs=_compile_kwargs,
                   device=lambda value: device_from_args(value, n_cols=value.cols or max(3, len(value.stage) + (1 if value.stem else 0))))


if __name__ == "__main__":
    main()
