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
from aie.iron import (
    Buffer,
    CompileTime,
    ExternalFunction,
    In,
    InOut,
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

import layer_engine
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
    looped: CompileTime[int] = 0,
    l2: CompileTime[int] = 0,
    kflags: CompileTime[str] = "",
    arch: CompileTime[str] = "3,4,6,3",
):
    jobs, _ = layer_engine_nets.build(net, 0, arch)
    segments = layer_engine_nets.segments_for(net, jobs, arch)
    resnet_like = net in ("full", "body", "bodyr", "l1proj", "l1id", "l2proj", "l3id", "l4id")  # no table/movement/depthwise code: 16 KB program memory
    generic = net.startswith(
        ("onnx:", "gen:")
    )  # graph-compiled jobs: one table-driven loop, every job takes two act objects
    basic = arch.startswith("basic:")
    stem = (
        net == "full"
    )  # jobs[:5] are the stem GEMM chunks + pool; the rest is the looped body
    nch = [n_chunks(j, slot - 192) for j in jobs]
    slots_used = layer_engine.arena_slots(jobs)
    has_res = generic or any(j.res_slot is not None for j in jobs)

    act_ty = np.ndarray[(SLOT_BYTES,), np.dtype[np.int8]]
    w_ty = np.ndarray[(slot,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(REGION_BYTES,), np.dtype[np.int8]]
    col_out_ty = np.ndarray[(ROWS * REGION_BYTES,), np.dtype[np.int8]]
    kernel = ExternalFunction(
        "layer_chunk",
        source_file=str(_KERNEL),
        arg_types=[act_ty, w_ty, out_ty, act_ty],
        compile_flags=[f"-DENG_REGION_BYTES={REGION_BYTES}"]
        + (["-DENG_NO_G4"] if generic else (["-DENG_NO_MOVE", "-DENG_NO_LUT", "-DENG_NO_DW"] if resnet_like else []))
        + kflags.split(),
    )

    def core_fn(act, w, out, kern, index):
        def run_job(j, job):
            if (
                job.res_slot is not None
            ):  # residual map = a second broadcast object right after the input
                both = act.acquire(2)
                r, a = both[0], both[1]  # the residual is filled first (a job early)
            else:
                a = act.acquire(1)
                r = a
            o = out.acquire(1)
            for _ in range_(nch[j]):
                if l2:  # memtile distributes every chunk round, so only this core's slice arrives
                    wc = w.acquire(1)
                    if (compute >> j) & 1:
                        kern(a, wc, o, r)
                    w.release(1)
                else:
                    for c in range(ROWS):
                        wc = w.acquire(1)
                        if c == index and (compute >> j) & 1:
                            kern(a, wc, o, r)
                        w.release(1)
            out.release(1)
            act.release(2 if job.res_slot is not None else 1)

        for first, count, repeat in segments:

            def body():
                for j in range(first, first + count):
                    run_job(j, jobs[j])

            if repeat > 1:
                for _ in range_(repeat):
                    body()
            else:
                body()

    def core_looped(act, w, out, kern, index, sched):
        """Whole-network core program: 4 stages x (projection block + `nid[s]` identity blocks).

        The per-job chunk counts come from a small table in core memory (indexed by the stage loop
        variable), so the program is 7 job bodies instead of one per layer.
        """

        def run(kind, res, stage, count=None):
            a = act.acquire(2 if res else 1)
            if res:
                r, a = a[0], a[1]  # the residual is filled first (a job early)
            else:
                r = a
            o = out.acquire(1)
            for _ in range_(sched[kind, stage] if count is None else count):
                if l2:
                    wc = w.acquire(1)
                    if (compute >> kind) & 1:
                        kern(a, wc, o, r)
                    w.release(1)
                else:
                    for c in range(ROWS):
                        wc = w.acquire(1)
                        if c == index and (compute >> kind) & 1:
                            kern(a, wc, o, r)
                        w.release(1)
            out.release(1)
            act.release(2 if res else 1)

        if stem:  # four stem GEMM chunks (same shape) then the pool job
            for _ in range_(4):
                run(7, False, None, count=nch[0])
            run(8, False, None, count=nch[4])
        for stage in range_(4):
            run(0, False, stage)
            run(1, False, stage)
            run(2, False, stage)
            run(3, True, stage)
            for _ in range_(sched[7, stage]):
                run(4, False, stage)
                run(5, False, stage)
                run(6, True, stage)

    def core_basic(act, w, out, kern, index, sched):
        """Basic-block (ResNet-18/34) core program: stage 1 is identity blocks only; stages 2-4 each start
        with a projection block [skip, conv a, conv b + residual] followed by `nid` identity blocks [a, b + residual]."""

        def run(kind, res, stage, count=None):
            a = act.acquire(2 if res else 1)
            if res:
                r, a = a[0], a[1]
            else:
                r = a
            o = out.acquire(1)
            for _ in range_(sched[kind, stage] if count is None else count):
                if l2:
                    wc = w.acquire(1)
                    if (compute >> kind) & 1:
                        kern(a, wc, o, r)
                    w.release(1)
                else:
                    for c in range(ROWS):
                        wc = w.acquire(1)
                        if c == index and (compute >> kind) & 1:
                            kern(a, wc, o, r)
                        w.release(1)
            out.release(1)
            act.release(2 if res else 1)

        if stem:
            for _ in range_(4):
                run(7, False, None, count=nch[0])
            run(8, False, None, count=nch[4])
        first, count, repeat = segments[0]  # stage 1: identity blocks [a, b]
        for _ in range_(repeat):
            run(9, False, None, count=nch[first])
            run(10, True, None, count=nch[first + 1])
        for stage in range_(3):
            run(0, False, stage)
            run(1, False, stage)
            run(2, True, stage)
            for _ in range_(sched[7, stage]):
                run(4, False, stage)
                run(5, True, stage)

    def core_generic(act, w, out, kern, index, sched):
        for j in range_(len(jobs)):
            both = act.acquire(2)
            r, a = both[0], both[1]
            o = out.acquire(1)
            for _ in range_(sched[0, j]):
                if l2:
                    wc = w.acquire(1)
                    kern(a, wc, o, r)
                    w.release(1)
                else:
                    for c in range(ROWS):
                        wc = w.acquire(1)
                        if c == index:
                            kern(a, wc, o, r)
                        w.release(1)
            out.release(1)
            act.release(2)

    act_all = ObjectFifo(act_ty, depth=2 if has_res else 1, name="act_all")
    sched_tab = None
    if generic:
        sched_tab = np.array([nch], dtype=np.int16)
    elif looped and basic:
        sched_tab = np.zeros((8, 3), dtype=np.int32)
        rest = segments[1:]
        for st in range(3):
            (pf, _pc, _pr), (idf, _ic, ir) = rest[2 * st], rest[2 * st + 1]
            for k in range(3):
                sched_tab[k, st] = nch[pf + k]
            for k in range(2):
                sched_tab[4 + k, st] = nch[idf + k]
            sched_tab[7, st] = ir
    elif looped:
        sched_tab = np.zeros((8, 4), dtype=np.int32)
        for st, (first, count, repeat) in enumerate(
            [sg for sg in segments if sg[1] == 4]
        ):
            for k in range(4):
                sched_tab[k, st] = nch[first + k]
        for st, (first, count, repeat) in enumerate(
            [sg for sg in segments if sg[1] == 3]
        ):
            for k in range(3):
                sched_tab[4 + k, st] = nch[first + k]
            sched_tab[7, st] = repeat
    wfs, outs, workers = [], [], []
    for col in range(COLS):
        if l2:
            wf = ObjectFifo(
                np.ndarray[(ROWS * slot,), np.dtype[np.uint8]],
                depth=l2,
                name=f"c{col}_w",
            )
            core_ws = [
                ObjectFifo(w_ty, depth=depth, name=f"c{col}_w{i}") for i in range(ROWS)
            ]
            ObjectFifoLink(
                wf.cons(),
                [f.prod() for f in core_ws],
                dst_offsets=[i * slot for i in range(ROWS)],
            )
        else:
            wf = ObjectFifo(w_ty, depth=depth, name=f"c{col}_w")
            core_ws = None
        core_outs = [
            ObjectFifo(out_ty, depth=1, name=f"c{col}_o{i}") for i in range(ROWS)
        ]
        col_out = ObjectFifo(col_out_ty, depth=1, name=f"c{col}_out")
        for i in range(ROWS):
            workers.append(
                Worker(
                    core_generic
                    if generic
                    else (
                        (core_basic if basic else core_looped) if looped else core_fn
                    ),
                    fn_args=[
                        act_all.cons(),
                        core_ws[i].cons() if l2 else wf.cons(),
                        core_outs[i].prod(),
                        kernel,
                        i,
                    ]
                    + (
                        [
                            Buffer(
                                np.ndarray[
                                    sched_tab.shape,
                                    np.dtype[np.int16 if generic else np.int32],
                                ],
                                initial_value=sched_tab,
                            )
                        ]
                        if (looped or generic)
                        else []
                    ),
                    tile=Tile(col, 2 + i),
                    stack_size=0x1300,
                )
            )
        ObjectFifoLink(
            [o.cons() for o in core_outs],
            col_out.prod(),
            src_offsets=[i * REGION_BYTES for i in range(ROWS)],
        )
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
            wprods[col].fill(
                params,
                group=weights_group,
                offset=col * per_col,
                sizes=[1, 1, 1, per_col],
                strides=[0, 0, 0, 1],
                transfer_len=per_col,
            )
        res_of = (
            (
                lambda k: (
                    jobs[k].res_slot
                    if jobs[k].res_slot is not None
                    else jobs[k].in_slot
                )
            )
            if generic
            else (lambda k: jobs[k].res_slot)
        )
        # A residual is prefetched during the previous job unless that very job produces it.
        prefetchable = lambda k: (
            k > 0 and res_of(k) is not None and jobs[k - 1].out_slot != res_of(k)
        )
        for j, job in enumerate(jobs):
            group = TaskGroup()
            if (generic and j == 0) or (
                j > 0 and res_of(j) is not None and not prefetchable(j)
            ):
                # first job, or its residual is the previous job's output: queue it now, ahead of the input
                aprod.fill(
                    a_in,
                    group=group,
                    offset=res_of(j) * SLOT_BYTES,
                    sizes=[1, 1, 1, SLOT_BYTES],
                    strides=[0, 0, 0, 1],
                    transfer_len=SLOT_BYTES,
                )
            aprod.fill(
                a_in,
                group=group,
                offset=job.in_slot * SLOT_BYTES,
                sizes=[1, 1, 1, SLOT_BYTES],
                strides=[0, 0, 0, 1],
                transfer_len=SLOT_BYTES,
            )
            if j + 1 < len(jobs) and prefetchable(j + 1):
                # The next job's residual map already exists: queue it behind this job's input now so its
                # transfer overlaps this job's compute (the core takes it first, then the input).
                aprod.fill(
                    a_in,
                    group=group,
                    offset=res_of(j + 1) * SLOT_BYTES,
                    sizes=[1, 1, 1, SLOT_BYTES],
                    strides=[0, 0, 0, 1],
                    transfer_len=SLOT_BYTES,
                )
            for col, oc in enumerate(ocons):
                if job.stem_chunk is not None:
                    # chunk c of block s (core s) lands at s*2048 + c*512 of the dense stem map (blocks 2048 apart);
                    # columns past the 8 real blocks write into the map's unused tail
                    oc.drain(
                        a_out,
                        wait=True,
                        group=group,
                        offset=job.out_slot * SLOT_BYTES
                        + col * ROWS * 2048
                        + job.stem_chunk * REGION_BYTES,
                        sizes=[1, 1, ROWS, REGION_BYTES],
                        strides=[0, 0, 2048, 1],
                        transfer_len=ROWS * REGION_BYTES,
                    )
                else:
                    oc.drain(
                        a_out,
                        wait=True,
                        group=group,
                        offset=job.out_slot * SLOT_BYTES + col * ROWS * REGION_BYTES,
                        sizes=[1, 1, 1, ROWS * REGION_BYTES],
                        strides=[0, 0, 0, 1],
                        transfer_len=ROWS * REGION_BYTES,
                    )
            group.finish()
        weights_group.finish()

    args = [arena_ty, params_ty, arena_ty, act_all.prod()] + handles_ty
    runtime = Runtime(sequence, args)
    return Program(
        iron.get_current_device(), runtime, workers=workers
    ).resolve_program()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_compile_args(parser)
    parser.add_argument(
        "--net",
        default="l1proj",
        help="l1proj|l2proj|l3id|l4id|body|bodyr|full (full = stem + pool + body, reused slots)",
    )
    parser.add_argument("--slot", type=int, default=8192)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--nocompute", action="store_true")
    parser.add_argument(
        "--kflags", default="", help="extra kernel compile flags (profiling)"
    )
    parser.add_argument(
        "--l2",
        type=int,
        default=0,
        help="stage weights in memtile L2 (this many 4-slice objects deep) and distribute one slice per core",
    )
    parser.add_argument(
        "--arch",
        default="3,4,6,3",
        help="bottleneck counts per stage, optional ':W' 3x3 width multiplier (full net)",
    )
    parser.add_argument(
        "--looped", action="store_true", help="stage-looped core program (body net)"
    )
    parser.add_argument(
        "--compute",
        type=lambda v: int(v, 0),
        default=0xFFFFFF,
        help="bitmask of jobs that run their kernel (profiling)",
    )
    return parser


def main() -> None:
    opts = _parser().parse_args()
    run_design_cli(
        engine,
        opts,
        compile_kwargs=lambda o: {
            "net": o.net,
            "slot": o.slot,
            "depth": o.depth,
            "compute": 0 if o.nocompute else o.compute,
            "looped": 1 if o.looped else 0,
            "kflags": o.kflags,
            "l2": o.l2,
            "arch": o.arch,
        },
        device=lambda value: device_from_args(value, n_cols=8),
    )


if __name__ == "__main__":
    main()
