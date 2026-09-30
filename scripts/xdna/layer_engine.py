"""Host side of the layer-sequential engine (``kernels/layer_engine.cc``).

Every conv layer is one *job* spread over the 32 cores (8 columns x 4). Activations live in a DDR
arena of fixed-size slots (``SLOT_BYTES`` = 32 regions x ``REGION_BYTES``): a layer's output block
``g`` (8 channels) is computed by core ``g // nbc`` and stored at ``region[core] + local*P*8``. This
module builds the per-core weight chunks (descriptor + tiles + bias), converts between dense
``[pixel][channel]`` maps and that layout, and evaluates a job list in numpy (the reference).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

REGION_BYTES = 512
CORES = 32
COLS, ROWS = 8, 4
SLOT_BYTES = CORES * REGION_BYTES
ENGINE_SLOT_BYTES = 4096  # weight object size of the shipped engine artifact (layer_engine_design.py --slot)
DESC_BYTES = 192
TILE = 8

# Descriptor word indices: keep in sync with the enum in kernels/layer_engine.cc.
D_NBP, D_NCP, D_W, D_H, D_OW, D_OH, D_S, D_MODE, D_NTAPS, D_TAP0 = range(10)
D_NB = D_TAP0 + 9
(D_TT0, D_TTN, D_FIRST, D_LAST, D_SHIFT, D_RELU, D_IN_FLIP, D_OUT_FLIP, D_RES, D_EA, D_EB, D_BIAS, D_CORE, D_TI0, D_CP0) = range(
    D_NB + 1, D_NB + 16
)


@dataclass(frozen=True)
class Layout:
    """Where a map's channel blocks live in an arena slot."""

    nb: int  # channel blocks
    nbc: int  # blocks per producing core
    w: int
    h: int

    @property
    def ncp(self) -> int:
        return math.ceil(self.nb / self.nbc)

    @property
    def pixels(self) -> int:
        return self.w * self.h


def layout_for(channels: int, w: int, h: int) -> Layout:
    nb = channels // TILE
    return Layout(nb, math.ceil(nb / CORES), w, h)


def to_arena(dense: np.ndarray, layout: Layout) -> np.ndarray:
    """dense [pixel][channel] (any 1-byte dtype) -> one arena slot."""
    slot = np.zeros(SLOT_BYTES, dtype=np.uint8)
    p = layout.pixels
    raw = np.ascontiguousarray(dense).view(np.uint8).reshape(p, layout.nb, TILE)
    for g in range(layout.nb):
        core, local = divmod(g, layout.nbc)
        off = core * REGION_BYTES + local * p * TILE
        slot[off : off + p * TILE] = raw[:, g, :].reshape(-1)
    return slot


def from_arena(slot: np.ndarray, layout: Layout) -> np.ndarray:
    p = layout.pixels
    out = np.zeros((p, layout.nb, TILE), dtype=np.uint8)
    for g in range(layout.nb):
        core, local = divmod(g, layout.nbc)
        off = core * REGION_BYTES + local * p * TILE
        out[:, g, :] = slot[off : off + p * TILE].reshape(p, TILE)
    return out.reshape(p, layout.nb * TILE)


@dataclass
class Job:
    name: str
    weight: np.ndarray  # [oc][ic][ky][kx] int8 (ky=kx=1 for 1x1)
    bias: np.ndarray  # int32 [oc]
    in_slot: int
    out_slot: int
    in_layout: Layout
    stride: int = 1
    shift: int = 8
    relu: bool = True
    in_flip: bool = False
    out_flip: bool = False
    res_slot: int | None = None
    res_mode: int = 0  # 1 int8 residual, 2 uint8 residual
    ea: int = 0
    eb: int = 0
    out_layout: Layout = field(init=False)
    taps: list[int] = field(init=False)

    def __post_init__(self):
        oc, _, kh, _ = self.weight.shape
        s = self.stride
        ow = (self.in_layout.w - 1) // s + 1
        oh = (self.in_layout.h - 1) // s + 1
        self.out_layout = layout_for(oc, ow, oh)
        if kh == 3:
            self.taps = valid_taps(self.in_layout.h, self.in_layout.w, oh, ow, s)
        else:
            self.taps = [4] if s > 1 else [0]  # 1x1: direct mode ignores the tap id; strided uses the centre tap

    @property
    def gather(self) -> bool:
        """Spatial (non-direct) kernel path. A 3x3 over a 1x1 map only ever uses the centre tap: direct."""
        if self.weight.shape[2] == 3 and self.stride == 1 and self.taps == [4]:
            return False
        return self.weight.shape[2] == 3 or self.stride > 1


def valid_taps(h: int, w: int, oh: int, ow: int, stride: int) -> list[int]:
    taps = []
    for tap in range(9):
        ky, kx = divmod(tap, 3)
        if any(0 <= y * stride + ky - 1 < h and 0 <= x * stride + kx - 1 < w for y in range(oh) for x in range(ow)):
            taps.append(tap)
    return taps


def _blocks(job: Job, core: int) -> range:
    lo = core * job.out_layout.nbc
    return range(lo, min(lo + job.out_layout.nbc, job.out_layout.nb)) if lo < job.out_layout.nb else range(0)


def plan_chunks(job: Job, payload: int) -> int:
    """Reduction steps (tap x region) per chunk so that every core's tiles + bias fit ``payload``."""
    nbp = job.in_layout.nbc
    step_bytes = job.out_layout.nbc * nbp * 64
    bias = job.out_layout.nbc * TILE * 4
    per_chunk = (payload - bias) // step_bytes
    if per_chunk < 1:
        raise ValueError(f"{job.name}: one reduction step ({step_bytes} B + bias) does not fit a {payload} B slot")
    return per_chunk


def n_chunks(job: Job, payload: int) -> int:
    steps = len(job.taps) * job.in_layout.ncp
    return math.ceil(steps / plan_chunks(job, payload))


def pack_job(job: Job, slot_bytes: int) -> np.ndarray:
    """Weight stream of one job: uint8 [column][chunk][row][slot_bytes]."""
    payload = slot_bytes - DESC_BYTES
    lay_in, lay_out = job.in_layout, job.out_layout
    nbp, ncp = lay_in.nbc, lay_in.ncp
    ntaps = len(job.taps)
    steps = ntaps * ncp
    per = plan_chunks(job, payload)
    nch = math.ceil(steps / per)
    oc, ic, kh, kw = job.weight.shape
    out = np.zeros((COLS, nch, ROWS, slot_bytes), dtype=np.uint8)
    for core in range(CORES):
        col, row = divmod(core, ROWS)
        blocks = _blocks(job, core)
        nb = len(blocks)
        for chunk in range(nch):
            tt0, ttn = chunk * per, min(per, steps - chunk * per)
            desc = np.zeros(DESC_BYTES // 4, dtype=np.int32)
            desc[D_NBP], desc[D_NCP] = nbp, ncp
            desc[D_W], desc[D_H] = lay_in.w, lay_in.h
            desc[D_OW], desc[D_OH] = lay_out.w, lay_out.h
            desc[D_S], desc[D_MODE] = job.stride, 1 if job.gather else 0
            desc[D_NTAPS] = ntaps
            desc[D_TAP0 : D_TAP0 + len(job.taps)] = job.taps
            desc[D_NB] = nb
            desc[D_TT0], desc[D_TTN] = tt0, ttn
            desc[D_FIRST], desc[D_LAST] = int(chunk == 0), int(chunk == nch - 1)
            desc[D_SHIFT], desc[D_RELU] = job.shift, int(job.relu)
            desc[D_IN_FLIP], desc[D_OUT_FLIP] = int(job.in_flip), int(job.out_flip)
            desc[D_RES], desc[D_EA], desc[D_EB] = job.res_mode, job.ea, job.eb
            tiles = nb * ttn * nbp * 64
            desc[D_BIAS] = (tiles + 3) & ~3
            desc[D_CORE] = core
            desc[D_TI0], desc[D_CP0] = divmod(tt0, ncp)
            slot = out[col, chunk, row]
            slot[:DESC_BYTES] = desc.view(np.uint8)
            body = np.zeros(desc[D_BIAS] + nb * TILE * 4, dtype=np.uint8)
            tile_view = np.zeros((nb, ttn, nbp, TILE, TILE), dtype=np.int8)  # [ocl][tt][l][k][n]
            for ol, g in enumerate(blocks):
                for i in range(ttn):
                    tt = tt0 + i
                    ti, cp = divmod(tt, ncp)
                    ky, kx = divmod(job.taps[ti], 3) if kh == 3 else (0, 0)
                    for l in range(nbp):
                        icb = cp * nbp + l
                        # B[k][n] = W[oc = g*8+n][ic = icb*8+k]
                        blk = job.weight[g * TILE : (g + 1) * TILE, icb * TILE : (icb + 1) * TILE, ky, kx]
                        tile_view[ol, i, l] = blk.T
            if nb:
                body[: tiles] = tile_view.view(np.uint8).reshape(-1)
                body[desc[D_BIAS] :] = job.bias[blocks[0] * TILE : (blocks[-1] + 1) * TILE].astype(np.int32).view(np.uint8)
            slot[DESC_BYTES : DESC_BYTES + body.size] = body
    return out


def _rse(v: np.ndarray, s: int) -> np.ndarray:
    a = np.abs(v)
    return np.sign(v) * np.rint(a / float(1 << s))


def reference(job: Job, act: np.ndarray, resid: np.ndarray | None) -> np.ndarray:
    """dense [pixel][cin] uint8/int8 -> dense [pixel][cout] (uint8 view when out_flip else int8 view)."""
    lay = job.in_layout
    x = act.astype(np.int64)
    if job.in_flip:
        x = x - 128
    else:
        x = act.view(np.int8).astype(np.int64)
    oc, ic, kh, kw = job.weight.shape
    x = x.reshape(lay.h, lay.w, ic)
    s = job.stride
    pad = 1 if kh == 3 else 0
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    oh, ow = job.out_layout.h, job.out_layout.w
    acc = np.zeros((oh, ow, oc), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            sub = xp[ky : ky + (oh - 1) * s + 1 : s, kx : kx + (ow - 1) * s + 1 : s, :]
            acc += sub @ job.weight[:, :, ky, kx].astype(np.int64).T
    acc += job.bias.astype(np.int64)
    q = np.clip(_rse(acc, job.shift), -128, 127).astype(np.int64).reshape(oh * ow, oc)
    if job.res_mode:
        r = resid.astype(np.int64) - 128 if job.res_mode == 2 else resid.view(np.int8).astype(np.int64)
        common = max(-job.ea, 0) if job.ea < job.eb else max(-job.eb, 0)
        total = (q << (job.ea + common)) + (r << (job.eb + common))
        q = np.clip(_rse(total, common), -128, 127).astype(np.int64)
    if job.relu:
        q = np.maximum(q, 0)
    if job.out_flip:
        return (q + 128).astype(np.uint8)
    return q.astype(np.int8).view(np.uint8)


def assign_slots(jobs: list[Job], pinned: dict[int, int] | None = None) -> int:
    """Rewrite the jobs' logical slot ids to a small set of reused arena slots; returns the slot count.

    Slot 0 (the network input) stays 0. A slot is live from the job that writes it until the last job
    that reads it (a residual is read one job early: its fill is queued during the previous job).
    """
    last_use: dict[int, int] = {}
    for index, job in enumerate(jobs):
        last_use[job.in_slot] = max(last_use.get(job.in_slot, -1), index)
        if job.res_slot is not None:
            last_use[job.res_slot] = max(last_use.get(job.res_slot, -1), index)
    physical: dict[int, int] = {0: 0}
    free: list[int] = []
    count = 1
    busy_until: dict[int, int] = {0: last_use.get(0, -1)}  # physical slot -> last job that reads it
    for index, job in enumerate(jobs):
        for logical in (job.in_slot, job.res_slot):
            if logical is not None and logical not in physical:
                raise ValueError("slot read before it is written")
        # a slot is reusable once its last reader is strictly before this job (writes land after job start)
        for p, until in list(busy_until.items()):
            if until < index and p not in free and p != 0:
                free.append(p)
        if free:
            p = free.pop(0)
        else:
            p = count
            count += 1
        physical[job.out_slot] = p
        busy_until[p] = last_use.get(job.out_slot, index)
    for job in jobs:
        job.in_slot = physical[job.in_slot]
        job.out_slot = physical[job.out_slot]
        if job.res_slot is not None:
            job.res_slot = physical[job.res_slot]
    return count
