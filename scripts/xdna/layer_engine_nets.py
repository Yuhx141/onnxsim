"""Synthetic job lists for exercising the layer engine (random weights, numpy reference)."""

from __future__ import annotations

import os

import numpy as np

from layer_engine import (
    SLOT_BYTES,
    Job,
    assemble_full,
    assign_slots,
    layout_for,
    reference,
    to_arena,
    from_arena,
)


def bottleneck(
    name: str,
    cin: int,
    mid: int,
    out: int,
    w: int,
    h: int,
    stride: int,
    rng,
    *,
    project: bool,
):
    """c1 (1x1) -> c2 (3x3, stride) -> c3 (1x1 + residual); slot 0 holds the block input (uint8)."""
    lay_in = layout_for(cin, w, h)
    rw = lambda o, i, k: rng.integers(-24, 24, (o, i, k, k), dtype=np.int8)
    rb = lambda n: rng.integers(-800, 800, n, dtype=np.int32)
    jobs = [
        Job(f"{name}.c1", rw(mid, cin, 1), rb(mid), 0, 1, lay_in, in_flip=True, shift=9)
    ]
    jobs.append(
        Job(
            f"{name}.c2",
            rw(mid, mid, 3),
            rb(mid),
            1,
            2,
            jobs[0].out_layout,
            stride=stride,
            shift=9,
        )
    )
    nxt = 3
    res_slot, res_mode = 0, 2
    if project:
        jobs.append(
            Job(
                f"{name}.sk",
                rw(out, cin, 1),
                rb(out),
                0,
                3,
                lay_in,
                stride=stride,
                in_flip=True,
                shift=8,
                relu=False,
            )
        )
        res_slot, res_mode, nxt = 3, 1, 4
    jobs.append(
        Job(
            f"{name}.c3",
            rw(out, mid, 1),
            rb(out),
            2,
            nxt,
            jobs[1].out_layout,
            shift=8,
            res_slot=res_slot,
            res_mode=res_mode,
            out_flip=True,
            ea=-1,
            eb=0,
        )
    )
    return jobs


def body(
    seed: int = 0,
    stages=((3, 64, 256, 1), (4, 128, 512, 2), (6, 256, 1024, 2), (3, 512, 2048, 2)),
    hw: int = 8,
):
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
            c1 = Job(
                f"c1",
                rw(mid, cin, 1),
                rb(mid),
                block_in,
                slot,
                lay,
                in_flip=True,
                shift=9,
            )
            slot += 1
            blk = [c1]
            res_slot, res_mode = block_in, 2
            if proj:  # the skip conv runs before conv2 so its output (the residual) is ready a job early
                sk = Job(
                    "sk",
                    rw(out, cin, 1),
                    rb(out),
                    block_in,
                    slot,
                    lay,
                    stride=st,
                    in_flip=True,
                    shift=8,
                    relu=False,
                )
                slot += 1
                blk.append(sk)
                res_slot, res_mode = sk.out_slot, 1
            c2 = Job(
                f"c2",
                rw(mid, mid, 3),
                rb(mid),
                c1.out_slot,
                slot,
                c1.out_layout,
                stride=st,
                shift=9,
            )
            slot += 1
            c3 = Job(
                "c3",
                rw(out, mid, 1),
                rb(out),
                c2.out_slot,
                slot,
                c2.out_layout,
                shift=8,
                res_slot=res_slot,
                res_mode=res_mode,
                out_flip=True,
                ea=-1,
                eb=0,
            )
            slot += 1
            blk += [c2, c3]
            jobs += blk
            block_in = c3.out_slot
            cin = out
            if proj:
                segments.append((first, len(blk), 1))
            elif (
                segments
                and segments[-1][0] + segments[-1][1] * segments[-1][2] == first
                and segments[-1][1] == len(blk)
                and b > 1
            ):
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


def build(name: str, seed: int = 0, arch: str = "3,4,6,3"):
    rng = np.random.default_rng(seed)
    if name.startswith(
        "gen:"
    ):  # a named synthetic net run through the table-driven generic core loop
        return build(name[4:], seed, arch)
    if name.startswith(
        "onnx:"
    ):  # any QDQ graph the graph compiler accepts (structure only matters to the design)
        import onnx

        from layer_engine_graph import compile_graph

        jobs = compile_graph(onnx.load(name[5:])).jobs
        limit = int(os.environ.get("ENGINE_JOB_LIMIT", "0"))
        return (jobs[:limit] if limit else jobs), None
    if name == "full":
        jobs, _segments, _stem = full(seed, arch)
        return jobs, None
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
    elif name == "dwchain":
        return dwchain(rng)
    elif name == "yolodata":
        return yolodata(rng)
    elif name == "l4id":
        jobs = bottleneck("l4", 2048, 512, 2048, 1, 1, 1, rng, project=False)
    else:
        raise ValueError(name)
    lay0 = jobs[0].in_layout
    x = rng.integers(128, 256, (lay0.pixels, lay0.nb * 8), dtype=np.uint8)
    return jobs, x


def segments_for(name: str, jobs, arch: str = "3,4,6,3"):
    """Core-program segments (first_job, jobs_per_iteration, repeat) for a named net."""
    if name == "full":
        return full(0, arch)[1]
    if name in ("body", "bodyr"):
        return body()[1]
    return [(0, len(jobs), 1)]


def synthetic_stem(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "weights": rng.integers(-24, 24, (64, 3, 7, 7), dtype=np.int8),
        "bias": rng.integers(-800, 800, 64, dtype=np.int32),
        "shift": 9,
        "in_scale": 1.0 / 64,
        "in_zero": 128,
        "strides": [2, 2],
        "pads": [3, 3, 3, 3],
    }


def arch_stages(arch: str = "3,4,6,3"):
    """``"3,4,6,3"`` or ``"3,4,6,3:2"`` (bottleneck counts per stage, optional width multiplier of the 3x3 width)."""
    counts, _, width = arch.partition(":")
    mult = int(width or 1)
    n1, n2, n3, n4 = (int(v) for v in counts.split(","))
    return (
        (n1, 64 * mult, 256, 1),
        (n2, 128 * mult, 512, 2),
        (n3, 256 * mult, 1024, 2),
        (n4, 512 * mult, 2048, 2),
    )


def full(seed: int = 0, arch: str = "3,4,6,3"):
    """Stem + pool + layer1..4 (random weights) in one arena. Returns (jobs, segments, stem).

    ``arch`` is ``"3,4,6,3[:W]"`` (bottleneck) or ``"basic:2,2,2,2"`` (basic blocks).
    """
    if arch.startswith("basic:"):
        body_jobs, segments, _ = basic_body(
            seed, tuple(int(v) for v in arch[6:].split(","))
        )
    else:
        body_jobs, segments, _ = body(seed, stages=arch_stages(arch))
    assign_slots(body_jobs)
    stem = synthetic_stem(seed)
    jobs = assemble_full(stem, body_jobs)
    return jobs, [(first + 5, count, repeat) for first, count, repeat in segments], stem


def basic_body(seed: int = 0, counts=(2, 2, 2, 2), hw: int = 8):
    """ResNet-18/34 style layer1..4 (basic blocks, random weights). Returns (jobs, segments, x)."""
    from layer_engine_net import basic_block_jobs

    rng = np.random.default_rng(seed)
    jobs, segments = [], []
    lay, cin, slot, block_in = layout_for(64, hw, hw), 64, 1, 0
    widths, strides = (64, 128, 256, 512), (1, 2, 2, 2)
    for stage, (n, width, stride) in enumerate(zip(counts, widths, strides)):
        for b in range(n):
            proj = b == 0 and stage > 0
            st = stride if b == 0 else 1
            raw = {
                "w1": rng.integers(-24, 24, (width, cin, 3, 3), dtype=np.int8),
                "b1": rng.integers(-800, 800, width, dtype=np.int32),
                "w2": rng.integers(-24, 24, (width, width, 3, 3), dtype=np.int8),
                "b2": rng.integers(-800, 800, width, dtype=np.int32),
                "skip_weight": rng.integers(-24, 24, (width, cin, 1, 1), dtype=np.int8)
                if proj
                else None,
                "skip_bias": rng.integers(-800, 800, width, dtype=np.int32)
                if proj
                else None,
            }
            first = len(jobs)
            block, slot, lay = basic_block_jobs(
                raw, st, (9, 8), 8, -1, 0, block_in, slot, lay
            )
            jobs += block
            block_in, cin = block[-1].out_slot, width
            if proj or (stage == 0 and b == 0):
                segments.append((first, len(block), 1))
            else:
                last = segments[-1]
                if last[1] == len(block) and last[0] + last[1] * last[2] == first:
                    segments[-1] = (last[0], last[1], last[2] + 1)
                else:
                    segments.append((first, len(block), 1))
    x = rng.integers(128, 256, (hw * hw, 64), dtype=np.uint8)
    return jobs, segments, x


def silu_table(scale_in: float = 1 / 16, scale_out: float = 1 / 16) -> np.ndarray:
    """Byte table of an int8 -> int8 SiLU (the reference definition of a lookup-table job)."""
    x = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float64) * scale_in
    y = x / (1.0 + np.exp(-x))
    q = np.clip(np.rint(y / scale_out), -128, 127).astype(np.int8)
    return q.view(np.uint8)


def dwchain(rng):
    """MobileNet-style chain: pointwise (ReLU6) -> depthwise -> SiLU table -> depthwise stride 2 -> pointwise (linear)."""
    lay = layout_for(32, 8, 8)
    rw = lambda o, i, k: rng.integers(-24, 24, (o, i, k, k), dtype=np.int8)
    rb = lambda n: rng.integers(-800, 800, n, dtype=np.int32)
    j0 = Job("pw", rw(96, 32, 1), rb(96), 0, 1, lay, in_flip=True, shift=8, clamp=95)
    j1 = Job(
        "dw", rw(96, 1, 3), rb(96), 1, 2, j0.out_layout, shift=7, clamp=95, kind="dw"
    )
    j2 = Job(
        "silu",
        np.zeros((96, 1, 1, 1), dtype=np.int8),
        np.zeros(96, dtype=np.int32),
        2,
        3,
        j1.out_layout,
        kind="lut",
        table=silu_table(),
    )
    j3 = Job(
        "dw2", rw(96, 1, 3), rb(96), 3, 4, j2.out_layout, stride=2, shift=7, kind="dw"
    )
    j4 = Job("pwl", rw(32, 96, 1), rb(32), 4, 5, j3.out_layout, shift=8, relu=False)
    jobs = [j0, j1, j2, j3, j4]
    x = rng.integers(128, 256, (lay.pixels, lay.nb * 8), dtype=np.uint8)
    return jobs, x


def yolodata(rng):
    """Data-movement jobs of a YOLO C2f/SPPF/neck: pointwise, split, max pool (k=5), concat with re-scaling, upsample."""
    lay = layout_for(32, 4, 4)
    rw = lambda o, i, k: rng.integers(-24, 24, (o, i, k, k), dtype=np.int8)
    rb = lambda n: rng.integers(-800, 800, n, dtype=np.int32)
    zero = lambda c: (
        np.zeros((c, 1, 1, 1), dtype=np.int8),
        np.zeros(c, dtype=np.int32),
    )
    j0 = Job(
        "pw", rw(64, 32, 1), rb(64), 0, 1, lay, in_flip=True, out_flip=True, shift=8
    )
    a = layout_for(32, 4, 4)
    w0, b0 = zero(32)
    j1 = Job(
        "split0",
        w0,
        b0,
        1,
        2,
        j0.out_layout,
        kind="copy",
        copy_spec=[(0, g, 0) for g in range(4)],
    )
    j2 = Job(
        "split1",
        w0,
        b0,
        1,
        3,
        j0.out_layout,
        kind="copy",
        copy_spec=[(0, 4 + g, 0) for g in range(4)],
    )
    j3 = Job(
        "sppf", w0, b0, 3, 4, j2.out_layout, kind="maxpool", factor=5, stride=1, exp=-1
    )
    w1, b1 = zero(64)
    j4 = Job(
        "concat",
        w1,
        b1,
        2,
        5,
        j1.out_layout,
        kind="copy",
        res_slot=4,
        b_layout=j3.out_layout,
        copy_spec=[(0, g, 0) for g in range(4)] + [(1, g, 1) for g in range(4)],
    )
    j5 = Job("up", w1, b1, 5, 6, j4.out_layout, kind="up", factor=2, exp=0)
    j6 = Job(
        "pw2",
        rw(32, 64, 1),
        rb(32),
        6,
        7,
        j5.out_layout,
        in_flip=True,
        out_flip=True,
        shift=8,
    )
    jobs = [j0, j1, j2, j3, j4, j5, j6]
    x = rng.integers(128, 256, (a.pixels, a.nb * 8), dtype=np.uint8)
    return jobs, x
