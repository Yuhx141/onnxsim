#!/usr/bin/env python3
"""Feasibility probe for a Vitis-style layer engine: one 1x1 conv layer spread over 8 columns.

Every layer's output channels are split across all 32 cores (8 columns x 4 cores); each
column has its own shim weight stream (broadcast to its four cores, each keeping its own
slice) and activation stream, and a join drains the column's outputs to DDR. The layer is
repeated ``--layers`` times with fresh weights (sequenced in the runtime sequence, one
wait per layer) to mimic a layer-sequential network and to amortize the launch.

Weights per layer: K x N int8 in blocked tiles; per core N/32 output channels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.dataflow import ObjectFifoLink
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL = Path(__file__).with_name("kernels") / "fused_bottleneck_blocked.cc"
COLS, CORES = 8, 4


def _align4(v: int) -> int:
    return (v + 3) & ~3


def slot_bytes(k: int, n_core: int) -> int:
    return _align4(n_core * k) + n_core * 4


@iron.jit
def engine_probe(
    activation: In,
    parameters: In,
    result: Out,
    *,
    k: CompileTime[int],
    n: CompileTime[int],
    p: CompileTime[int],
    layers: CompileTime[int],
    compute: CompileTime[int] = 1,
    reuse: CompileTime[int] = 0,
    cols: CompileTime[int] = 8,
    once: CompileTime[int] = 0,
    bcast: CompileTime[int] = 0,
):
    COLS = cols
    n_core = n // (COLS * CORES)
    slot = slot_bytes(k, n_core)
    act_ty = np.ndarray[(k * p,), np.dtype[np.int8]]
    w_ty = np.ndarray[(slot,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(n_core * p,), np.dtype[np.int8]]
    col_out_ty = np.ndarray[(CORES * n_core * p,), np.dtype[np.int8]]
    flags = [
        f"-DFUSED_W={p}", "-DFUSED_H=1", f"-DFUSED_C={k}", f"-DFUSED_OUT_C={n_core}", f"-DFUSED_MID=16",
        f"-DFUSED_OUT_W={p}", "-DFUSED_OUT_H=1", "-DFUSED_SKIP_CHUNKS=1",
        f"-DFUSED_SKIP_BIAS_OFFSET={_align4(n_core * k)}", "-DFUSED_SKIP_SHIFT=8", "-DBLK_SKIP",
    ]
    kernel = ExternalFunction("fused_bottleneck_skip_chunk", source_file=str(_KERNEL),
                              arg_types=[act_ty, w_ty, out_ty, np.int32], compile_flags=flags, symbol_prefix="eng")

    def core_fn(act, w, out, kern, index):
        a = act.acquire(1)
        o = out.acquire(1)
        for c in range(CORES):  # the column stream carries one slice per core; keep ours
            wc = w.acquire(1)
            if c == index and compute:
                kern(a, wc, o, 0)
            w.release(1)
        out.release(1)
        act.release(1)

    acts, weights, outs, workers = [], [], [], []
    shared_act = ObjectFifo(act_ty, depth=1, name="act_all") if bcast else None
    for col in range(COLS):
        act = shared_act if bcast else ObjectFifo(act_ty, depth=1, name=f"c{col}_act")
        wf = ObjectFifo(w_ty, depth=1, name=f"c{col}_w")
        core_outs = [ObjectFifo(out_ty, depth=1, name=f"c{col}_o{i}") for i in range(CORES)]
        col_out = ObjectFifo(col_out_ty, depth=1, name=f"c{col}_out")
        for i in range(CORES):
            workers.append(Worker(core_fn, fn_args=[act.cons(), wf.cons(), core_outs[i].prod(), kernel, i],
                                  tile=Tile(col, 2 + i), stack_size=0x800))
        ObjectFifoLink([o.cons() for o in core_outs], col_out.prod(),
                       src_offsets=[i * n_core * p for i in range(CORES)])
        acts.append(act)
        weights.append(wf)
        outs.append(col_out)
    if bcast:
        acts = [shared_act]  # one activation fill feeds every core; outputs stay one drain per column

    per_layer = COLS * CORES * slot
    params_ty = np.ndarray[(layers * per_layer,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(n * p,), np.dtype[np.int8]]

    def sequence(x, packed, y, *handles):
        n_act, n_out = len(acts), len(outs)
        aprods, wprods, ocons = handles[:n_act], handles[n_act : n_act + COLS], handles[n_act + COLS :]
        weights_group = TaskGroup()
        if once:
            for col in range(COLS):
                wprods[col].fill(packed, group=weights_group, offset=col * CORES * slot,
                                 sizes=[layers, 1, 1, CORES * slot], strides=[per_layer, 0, 0, 1],
                                 transfer_len=CORES * slot)
        out_span = (COLS // n_out) * CORES * n_core * p
        for layer in range(layers):
            group = TaskGroup()
            for a in aprods:
                a.fill(x, group=group, sizes=[k // 8, p, 8], strides=[8 * p, 8, 1], transfer_len=k * p)
            if not once:
                for col in range(COLS):
                    wprods[col].fill(packed, group=group, offset=(0 if reuse else layer * per_layer) + col * CORES * slot,
                                     sizes=[1, 1, 1, CORES * slot], strides=[0, 0, 0, 1], transfer_len=CORES * slot)
            for i, oc in enumerate(ocons):
                oc.drain(y, wait=True, group=group, offset=i * out_span,
                         sizes=[1, 1, 1, out_span], strides=[0, 0, 0, 1], transfer_len=out_span)
            group.finish()
        weights_group.finish()

    runtime = Runtime(sequence, [
        np.ndarray[(k * p,), np.dtype[np.int8]], params_ty, y_ty,
        *[f.prod() for f in acts], *[f.prod() for f in weights], *[f.cons() for f in outs],
    ])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("-k", type=int, required=True, help="input channels")
    parser.add_argument("-n", type=int, required=True, help="output channels (multiple of 256)")
    parser.add_argument("-p", type=int, required=True, help="pixels (multiple of 8 or <8)")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--nocompute", action="store_true")
    parser.add_argument("--bcast", action="store_true", help="one shared activation stream, outputs joined into 2 drains")
    parser.add_argument("--once", action="store_true", help="one weight transfer per column for all layers")
    parser.add_argument("--cols", type=int, default=8, help="columns used (fewer columns = fewer DMA tasks per layer)")
    parser.add_argument("--reuse", action="store_true", help="every layer re-reads layer 0's weights (small memory footprint)")
    return parser


def main() -> None:
    opts = _parser().parse_args()
    run_design_cli(
        engine_probe, opts,
        compile_kwargs=lambda o: {"k": o.k, "n": o.n, "p": o.p, "layers": o.layers, "compute": 0 if o.nocompute else 1, "reuse": 1 if o.reuse else 0, "cols": o.cols, "once": 1 if o.once else 0, "bcast": 1 if o.bcast else 0},
        device=lambda value: device_from_args(value, n_cols=8),
    )


if __name__ == "__main__":
    main()
