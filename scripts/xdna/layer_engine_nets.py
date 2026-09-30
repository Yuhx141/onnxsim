"""Synthetic job lists for exercising the layer engine (random weights, numpy reference)."""

from __future__ import annotations

import numpy as np

from layer_engine import SLOT_BYTES, Job, assign_slots, layout_for, reference, to_arena, from_arena


def bottleneck(name: str, cin: int, mid: int, out: int, w: int, h: int, stride: int, rng, *, project: bool):
    """c1 (1x1) -> c2 (3x3, stride) -> c3 (1x1 + residual); slot 0 holds the block input (uint8)."""
    lay_in = layout_for(cin, w, h)
    rw = lambda o, i, k: rng.integers(-24, 24, (o, i, k, k), dtype=np.int8)
    rb = lambda n: rng.integers(-800, 800, n, dtype=np.int32)
    jobs = [Job(f"{name}.c1", rw(mid, cin, 1), rb(mid), 0, 1, lay_in, in_flip=True, shift=9)]
    jobs.append(Job(f"{name}.c2", rw(mid, mid, 3), rb(mid), 1, 2, jobs[0].out_layout, stride=stride, shift=9))
    nxt = 3
    res_slot, res_mode = 0, 2
    if project:
        jobs.append(Job(f"{name}.sk", rw(out, cin, 1), rb(out), 0, 3, lay_in, stride=stride, in_flip=True, shift=8, relu=False))
        res_slot, res_mode, nxt = 3, 1, 4
    jobs.append(Job(f"{name}.c3", rw(out, mid, 1), rb(out), 2, nxt, jobs[1].out_layout, shift=8, res_slot=res_slot,
                    res_mode=res_mode, out_flip=True, ea=-1, eb=0))
    return jobs


def body(seed: int = 0, stages=((3, 64, 256, 1), (4, 128, 512, 2), (6, 256, 1024, 2), (3, 512, 2048, 2)), hw: int = 8):
    """ResNet-50 layer1..layer4 (random weights). Returns (jobs, segments, input dense map).

    ``segments`` = [(first_job, jobs_per_iteration, repeat)]: consecutive identical identity blocks
    are one segment so the core program loops over them instead of unrolling every job.
    """
    rng = np.random.default_rng(seed)
    jobs, segments = [], []
    cin, w = 64, hw
    lay_in = layout_for(cin, w, w)
    slot = 1  # slot 0 = network input
    block_in = 0
    for n, mid, out, stride in stages:
        for b in range(n):
            st = stride if b == 0 else 1
            proj = b == 0
            first = len(jobs)
            cur_w = jobs[-1].out_layout.w if jobs else w
            lay = jobs[-1].out_layout if jobs else lay_in
            rw = lambda o, i, k: rng.integers(-24, 24, (o, i, k, k), dtype=np.int8)
            rb = lambda c: rng.integers(-800, 800, c, dtype=np.int32)
            c1 = Job(f"c1", rw(mid, cin, 1), rb(mid), block_in, slot, lay, in_flip=True, shift=9); slot += 1
            blk = [c1]
            res_slot, res_mode = block_in, 2
            if proj:  # the skip conv runs before conv2 so its output (the residual) is ready a job early
                sk = Job("sk", rw(out, cin, 1), rb(out), block_in, slot, lay, stride=st, in_flip=True, shift=8, relu=False); slot += 1
                blk.append(sk)
                res_slot, res_mode = sk.out_slot, 1
            c2 = Job(f"c2", rw(mid, mid, 3), rb(mid), c1.out_slot, slot, c1.out_layout, stride=st, shift=9); slot += 1
            c3 = Job("c3", rw(out, mid, 1), rb(out), c2.out_slot, slot, c2.out_layout, shift=8, res_slot=res_slot,
                     res_mode=res_mode, out_flip=True, ea=-1, eb=0); slot += 1
            blk += [c2, c3]
            jobs += blk
            block_in = c3.out_slot
            cin = out
            if proj:
                segments.append((first, len(blk), 1))
            elif segments and segments[-1][0] + segments[-1][1] * segments[-1][2] == first and segments[-1][1] == len(blk) and b > 1:
                f, c, r = segments[-1]
                segments[-1] = (f, c, r + 1)
            else:
                segments.append((first, len(blk), 1))
    x = rng.integers(128, 256, (w * w, 64), dtype=np.uint8)
    return jobs, segments, x


def run_reference(jobs, x_dense: np.ndarray):
    """Returns {slot: dense uint8 map} after running every job."""
    maps = {0: x_dense}
    layouts = {0: jobs[0].in_layout}
    for job in jobs:
        res = maps[job.res_slot] if job.res_slot is not None else None
        maps[job.out_slot] = reference(job, maps[job.in_slot], res)
        layouts[job.out_slot] = job.out_layout
    return maps, layouts


def build(name: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    if name in ("body", "bodyr"):
        jobs, _, x = body(seed)
        if name == "bodyr":  # reused arena slots (production layout)
            assign_slots(jobs)
        return jobs, x
    if name == "l1proj":
        jobs = bottleneck("l1", 64, 64, 256, 8, 8, 1, rng, project=True)
    elif name == "l1id":
        jobs = bottleneck("l1", 256, 64, 256, 8, 8, 1, rng, project=False)
    elif name == "l2proj":
        jobs = bottleneck("l2", 256, 128, 512, 8, 8, 2, rng, project=True)
    elif name == "l3id":
        jobs = bottleneck("l3", 1024, 256, 1024, 2, 2, 1, rng, project=False)
    elif name == "l4id":
        jobs = bottleneck("l4", 2048, 512, 2048, 1, 1, 1, rng, project=False)
    else:
        raise ValueError(name)
    lay0 = jobs[0].in_layout
    x = rng.integers(128, 256, (lay0.pixels, lay0.nb * 8), dtype=np.uint8)
    return jobs, x


def segments_for(name: str, jobs):
    """Core-program segments (first_job, jobs_per_iteration, repeat) for a named net."""
    if name in ("body", "bodyr"):
        return body()[1]
    return [(0, len(jobs), 1)]
