"""Synthetic job lists for exercising the layer engine (random weights, numpy reference)."""

from __future__ import annotations

import numpy as np

from layer_engine import SLOT_BYTES, Job, layout_for, reference, to_arena, from_arena


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
