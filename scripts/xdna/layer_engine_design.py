#!/usr/bin/env python3
"""Layer-sequential engine: every conv layer runs on all 32 cores (8 columns x 4 rows).

One job per conv layer. The layer's input map (one arena slot) is broadcast to every core; each
column streams its four cores' weight chunks (each core keeps its own, see ``kernels/layer_engine.cc``);
the four cores of a column join their outputs into one drain to the next arena slot. An optional
second broadcast carries a residual map. See ``layer_engine.py`` for the arena layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, InOut, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifoLink
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

from layer_engine import COLS, REGION_BYTES, ROWS, SLOT_BYTES, n_chunks
import layer_engine_nets

_KERNEL = Path(__file__).with_name("kernels") / "layer_engine.cc"


@iron.jit
def engine(
    arena_in: In,
    packed: In,
    arena_out: Out,
    *,
    net: CompileTime[str],
    slot: CompileTime[int],
    depth: CompileTime[int] = 2,
    compute: CompileTime[int] = 0xFFFFFF,
):
    jobs, _ = layer_engine_nets.build(net)
    nch = [n_chunks(j, slot - 192) for j in jobs]
    slots_used = 1 + max(j.out_slot for j in jobs)
    has_res = any(j.res_slot is not None for j in jobs)

    act_ty = np.ndarray[(SLOT_BYTES,), np.dtype[np.int8]]
    w_ty = np.ndarray[(slot,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(REGION_BYTES,), np.dtype[np.int8]]
    col_out_ty = np.ndarray[(ROWS * REGION_BYTES,), np.dtype[np.int8]]
    kernel = ExternalFunction(
        "layer_chunk", source_file=str(_KERNEL),
        arg_types=[act_ty, w_ty, out_ty, act_ty], compile_flags=[f"-DENG_REGION_BYTES={REGION_BYTES}"],
    )

    def core_fn(act, w, out, kern, index):
        for j, job in enumerate(jobs):
            if job.res_slot is not None:  # residual map = a second broadcast object right after the input
                both = act.acquire(2)
                a, r = both[0], both[1]
            else:
                a = act.acquire(1)
                r = a
            o = out.acquire(1)
            for _ in range_(nch[j]):
                for c in range(ROWS):
                    wc = w.acquire(1)
                    if c == index and (compute >> j) & 1:
                        kern(a, wc, o, r)
                    w.release(1)
            out.release(1)
            act.release(2 if job.res_slot is not None else 1)

    act_all = ObjectFifo(act_ty, depth=2 if has_res else 1, name="act_all")
    wfs, outs, workers = [], [], []
    for col in range(COLS):
        wf = ObjectFifo(w_ty, depth=depth, name=f"c{col}_w")
        core_outs = [ObjectFifo(out_ty, depth=1, name=f"c{col}_o{i}") for i in range(ROWS)]
        col_out = ObjectFifo(col_out_ty, depth=1, name=f"c{col}_out")
        for i in range(ROWS):
            workers.append(Worker(
                core_fn, fn_args=[act_all.cons(), wf.cons(), core_outs[i].prod(), kernel, i],
                tile=Tile(col, 2 + i), stack_size=0x1500))
        ObjectFifoLink([o.cons() for o in core_outs], col_out.prod(), src_offsets=[i * REGION_BYTES for i in range(ROWS)])
        wfs.append(wf)
        outs.append(col_out)

    per_col = sum(nch) * ROWS * slot
    arena_ty = np.ndarray[(slots_used * SLOT_BYTES,), np.dtype[np.int8]]
    params_ty = np.ndarray[(COLS * per_col,), np.dtype[np.uint8]]
    handles_ty = [f.prod() for f in wfs] + [f.cons() for f in outs]

    def sequence(a_in, params, a_out, aprod, *handles):
        wprods, ocons = handles[:COLS], handles[COLS:]
        weights_group = TaskGroup()
        for col in range(COLS):
            wprods[col].fill(params, group=weights_group, offset=col * per_col, sizes=[1, 1, 1, per_col],
                             strides=[0, 0, 0, 1], transfer_len=per_col)
        for j, job in enumerate(jobs):
            group = TaskGroup()
            aprod.fill(a_in, group=group, offset=job.in_slot * SLOT_BYTES, sizes=[1, 1, 1, SLOT_BYTES],
                       strides=[0, 0, 0, 1], transfer_len=SLOT_BYTES)
            if job.res_slot is not None:
                aprod.fill(a_in, group=group, offset=job.res_slot * SLOT_BYTES, sizes=[1, 1, 1, SLOT_BYTES],
                           strides=[0, 0, 0, 1], transfer_len=SLOT_BYTES)
            for col, oc in enumerate(ocons):
                oc.drain(a_out, wait=True, group=group, offset=job.out_slot * SLOT_BYTES + col * ROWS * REGION_BYTES,
                         sizes=[1, 1, 1, ROWS * REGION_BYTES], strides=[0, 0, 0, 1], transfer_len=ROWS * REGION_BYTES)
            group.finish()
        weights_group.finish()

    args = [arena_ty, params_ty, arena_ty, act_all.prod()] + handles_ty
    runtime = Runtime(sequence, args)
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument("--net", default="l1proj")
    parser.add_argument("--slot", type=int, default=8192)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--nocompute", action="store_true")
    parser.add_argument("--compute", type=lambda v: int(v, 0), default=0xFFFFFF, help="bitmask of jobs that run their kernel (profiling)")
    return parser


def main() -> None:
    opts = _parser().parse_args()
    run_design_cli(
        engine, opts,
        compile_kwargs=lambda o: {"net": o.net, "slot": o.slot, "depth": o.depth, "compute": 0 if o.nocompute else o.compute},
        device=lambda value: device_from_args(value, n_cols=8),
    )


if __name__ == "__main__":
    main()
