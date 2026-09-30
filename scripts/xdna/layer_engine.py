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
(
    D_TT0,
    D_TTN,
    D_FIRST,
    D_LAST,
    D_SHIFT,
    D_RELU,
    D_IN_FLIP,
    D_OUT_FLIP,
    D_RES,
    D_EA,
    D_EB,
    D_BIAS,
    D_CORE,
    D_TI0,
    D_CP0,
    D_REG,
    D_CLAMP,
    D_KSZ,
) = range(D_NB + 1, D_NB + 19)


@dataclass(frozen=True)
class Layout:
    """Where a map's channel blocks live in an arena slot."""

    nb: int  # channel blocks
    nbc: int  # blocks per producing core
    w: int
    h: int
    region_bytes: int = REGION_BYTES  # distance between producing cores' regions

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
        off = core * layout.region_bytes + local * p * TILE
        slot[off : off + p * TILE] = raw[:, g, :].reshape(-1)
    return slot


def from_arena(slot: np.ndarray, layout: Layout) -> np.ndarray:
    p = layout.pixels
    out = np.zeros((p, layout.nb, TILE), dtype=np.uint8)
    for g in range(layout.nb):
        core, local = divmod(g, layout.nbc)
        off = core * layout.region_bytes + local * p * TILE
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
    kind: str = "conv"  # "conv" | "pool" | "dw" (depthwise 3x3) | "lut" (unary table) | "copy" | "up" | "maxpool" | "add" | "gap" | "bmul" | "avgpool" | "d2s"
    copy_spec: list | None = (
        None  # "copy": per output block (source 0/1, source block, scale exponent)
    )
    b_layout: Layout | None = (
        None  # "copy": layout of the second source (object B = res_slot)
    )
    pad: int | None = (
        None  # conv padding when it is not kernel // 2 (host jobs only: e.g. YOLOv5's 6x6 stride-2 stem, pad 2)
    )
    factor: int = (
        1  # "up": nearest upsample factor; "maxpool": window; stride is `stride`
    )
    exp: int = 0  # "up"/"maxpool": result scaled by 2^exp (the Q after the op has its own scale)
    clamp: int = 127  # upper bound of the int8 result before the output flip (ReLU6 = 6 / output scale)
    table: np.ndarray | None = None  # "lut": 256 output bytes indexed by the input byte
    stem_chunk: int | None = (
        None  # stem GEMM chunk: its output block is drained into a shared 64 KB map
    )
    out_span_slots: int = (
        1  # arena slots the job's output occupies (the stem map spans 4)
    )
    out_layout: Layout = field(init=False)
    taps: list[int] = field(init=False)

    def __post_init__(self):
        oc, _, kh, _ = self.weight.shape
        s = self.stride
        pad = kh // 2 if self.pad is None else self.pad
        ow = (self.in_layout.w + 2 * pad - kh) // s + 1
        oh = (self.in_layout.h + 2 * pad - kh) // s + 1
        if self.kind == "pool":
            ow, oh = self.in_layout.w // 2, self.in_layout.h // 2
        if self.kind in ("lut", "copy", "add", "bmul"):
            ow, oh = self.in_layout.w, self.in_layout.h
        if self.kind == "gap":
            ow, oh = 1, 1
        if self.kind == "avgpool":
            ow, oh = (self.in_layout.w - self.factor) // s + 1, (self.in_layout.h - self.factor) // s + 1
        if self.kind == "d2s":
            ow, oh = self.in_layout.w * self.factor, self.in_layout.h * self.factor
        if self.kind == "up":
            ow, oh = self.in_layout.w * self.factor, self.in_layout.h * self.factor
        if self.kind == "maxpool":
            ow, oh = (self.in_layout.w - 1) // s + 1, (self.in_layout.h - 1) // s + 1
        self.out_layout = layout_for(oc, ow, oh)
        if (
            self.kind in ("dw", "lut", "up", "maxpool", "add")
            and self.out_layout.nbc != self.in_layout.nbc
        ):
            raise ValueError(
                f"{self.name}: depthwise/table jobs need the same channel blocking in and out"
            )
        if self.kind in ("lut", "copy", "up", "maxpool", "add", "gap", "bmul", "avgpool", "d2s"):
            self.taps = [0]
        elif kh == 3:
            self.taps = valid_taps(self.in_layout.h, self.in_layout.w, oh, ow, s)
        else:
            self.taps = (
                [4] if s > 1 else [0]
            )  # 1x1: direct mode ignores the tap id; strided uses the centre tap

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
        if any(
            0 <= y * stride + ky - 1 < h and 0 <= x * stride + kx - 1 < w
            for y in range(oh)
            for x in range(ow)
        ):
            taps.append(tap)
    return taps


def _blocks(job: Job, core: int) -> range:
    lo = core * job.out_layout.nbc
    return (
        range(lo, min(lo + job.out_layout.nbc, job.out_layout.nb))
        if lo < job.out_layout.nb
        else range(0)
    )


def plan_chunks(job: Job, payload: int) -> int:
    """Reduction steps (tap x region) per chunk so that every core's tiles + bias fit ``payload``."""
    nbp = job.in_layout.nbc
    step_bytes = job.out_layout.nbc * nbp * 64
    bias = job.out_layout.nbc * TILE * 4
    per_chunk = (payload - bias) // step_bytes
    if per_chunk < 1:
        raise ValueError(
            f"{job.name}: one reduction step ({step_bytes} B + bias) does not fit a {payload} B slot"
        )
    return per_chunk


def n_chunks(job: Job, payload: int) -> int:
    if job.kind in ("pool", "dw", "lut", "copy", "up", "maxpool", "add", "gap", "bmul", "avgpool", "d2s"):
        return 1
    steps = len(job.taps) * job.in_layout.ncp
    return math.ceil(steps / plan_chunks(job, payload))


def pack_job(job: Job, slot_bytes: int) -> np.ndarray:
    """Weight stream of one job: uint8 [column][chunk][row][slot_bytes]."""
    payload = slot_bytes - DESC_BYTES
    lay_in, lay_out = job.in_layout, job.out_layout
    if job.kind in ("dw", "lut", "copy", "up", "maxpool", "add", "gap", "bmul", "avgpool", "d2s"):
        out = np.zeros((COLS, 1, ROWS, slot_bytes), dtype=np.uint8)
        for core in range(CORES):
            col, row = divmod(core, ROWS)
            blocks = _blocks(job, core)
            desc = np.zeros(DESC_BYTES // 4, dtype=np.int32)
            desc[D_W], desc[D_H], desc[D_OW], desc[D_OH] = (
                lay_in.w,
                lay_in.h,
                lay_out.w,
                lay_out.h,
            )
            desc[D_S], desc[D_NB], desc[D_CORE] = job.stride, len(blocks), core
            desc[D_FIRST], desc[D_LAST] = 1, 1
            desc[D_SHIFT], desc[D_RELU] = job.shift, int(job.relu)
            desc[D_IN_FLIP], desc[D_OUT_FLIP], desc[D_CLAMP] = (
                int(job.in_flip),
                int(job.out_flip),
                job.clamp,
            )
            desc[D_REG] = lay_in.region_bytes
            slot = out[col, 0, row]
            if job.kind == "avgpool":
                desc[D_MODE], desc[D_NTAPS], desc[D_S] = 12, job.factor, job.stride
                slot[:DESC_BYTES] = desc.view(np.uint8)
                continue
            if job.kind == "d2s":
                f = job.factor
                desc[D_MODE], desc[D_S] = 13, f
                slot[:DESC_BYTES] = desc.view(np.uint8)
                if blocks:
                    tab = np.zeros((len(blocks), f * f), dtype=np.int32)
                    for ol, g in enumerate(blocks):
                        for tap in range(f * f):
                            cp, local = divmod(tap * lay_out.nb + g, lay_in.nbc)
                            tab[ol, tap] = cp * lay_in.region_bytes + local * lay_in.pixels * TILE
                    slot[DESC_BYTES : DESC_BYTES + tab.nbytes] = tab.view(np.uint8).reshape(-1)
                continue
            if job.kind in ("gap", "bmul"):
                desc[D_MODE] = 10 if job.kind == "gap" else 11
                if job.kind == "bmul":
                    desc[D_TT0] = job.b_layout.region_bytes
                slot[:DESC_BYTES] = desc.view(np.uint8)
                continue
            if job.kind == "add":
                desc[D_MODE], desc[D_EA], desc[D_EB] = 9, job.ea, job.eb
                desc[D_TT0] = job.b_layout.region_bytes
                slot[:DESC_BYTES] = desc.view(np.uint8)
                continue
            if job.kind in ("copy", "up", "maxpool"):
                desc[D_MODE] = {"copy": 6, "up": 7, "maxpool": 8}[job.kind]
                if job.kind == "up":
                    desc[D_S] = job.factor
                if job.kind == "maxpool":
                    desc[D_NTAPS], desc[D_S] = job.factor, job.stride
                desc[D_EA] = job.exp
                slot[:DESC_BYTES] = desc.view(np.uint8)
                if job.kind == "copy" and blocks:
                    tab = np.zeros((len(blocks), 3), dtype=np.int32)
                    for ol, g in enumerate(blocks):
                        which, src_block, exp = job.copy_spec[g]
                        src_layout = job.b_layout if which else job.in_layout
                        cp, local = divmod(src_block, src_layout.nbc)
                        tab[ol] = (
                            which,
                            cp * src_layout.region_bytes
                            + local * src_layout.pixels * TILE,
                            exp,
                        )
                    slot[DESC_BYTES : DESC_BYTES + tab.nbytes] = tab.view(
                        np.uint8
                    ).reshape(-1)
                continue
            if job.kind == "lut":
                desc[D_MODE] = 5
                slot[:DESC_BYTES] = desc.view(np.uint8)
                slot[DESC_BYTES : DESC_BYTES + 256] = job.table.astype(np.uint8)
                continue
            ksz = job.weight.shape[2]
            desc[D_MODE], desc[D_NTAPS], desc[D_KSZ] = 4, ksz * ksz, ksz
            nb = len(blocks)
            tiles = nb * ksz * ksz * 64
            desc[D_BIAS] = (tiles + 3) & ~3
            if (desc[D_BIAS] + nb * TILE * 4) > payload:
                raise ValueError(
                    f"{job.name}: depthwise weights of {nb} blocks do not fit a {slot_bytes} B slot"
                )
            slot[:DESC_BYTES] = desc.view(np.uint8)
            body = np.zeros(desc[D_BIAS] + nb * TILE * 4, dtype=np.uint8)
            vec = np.zeros((nb, ksz * ksz, 64), dtype=np.int8)
            for ol, g in enumerate(blocks):
                for ti in range(ksz * ksz):
                    ky, kx = divmod(ti, ksz)
                    vec[ol, ti] = np.tile(
                        job.weight[g * TILE : (g + 1) * TILE, 0, ky, kx], 8
                    )  # lane = row * 8 + channel
            if nb:
                body[:tiles] = vec.view(np.uint8).reshape(-1)
                body[desc[D_BIAS] :] = (
                    job.bias[blocks[0] * TILE : (blocks[-1] + 1) * TILE]
                    .astype(np.int32)
                    .view(np.uint8)
                )
            slot[DESC_BYTES : DESC_BYTES + body.size] = body
        return out
    if (
        job.kind == "pool"
    ):  # the stem MaxPool is a k=3, stride-2 max pool job over the dense stem map
        out = np.zeros((COLS, 1, ROWS, slot_bytes), dtype=np.uint8)
        for core in range(CORES):
            col, row = divmod(core, ROWS)
            desc = np.zeros(DESC_BYTES // 4, dtype=np.int32)
            desc[D_W], desc[D_H], desc[D_OW], desc[D_OH] = (
                lay_in.w,
                lay_in.h,
                lay_out.w,
                lay_out.h,
            )
            desc[D_MODE], desc[D_NTAPS], desc[D_S] = 8, 3, 2
            desc[D_NB] = 1 if core < lay_out.nb else 0
            desc[D_CORE], desc[D_REG] = core, lay_in.region_bytes
            out[col, 0, row, :DESC_BYTES] = desc.view(np.uint8)
        return out
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
            desc[D_REG] = lay_in.region_bytes
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
            desc[D_CLAMP] = job.clamp
            slot = out[col, chunk, row]
            slot[:DESC_BYTES] = desc.view(np.uint8)
            body = np.zeros(desc[D_BIAS] + nb * TILE * 4, dtype=np.uint8)
            tile_view = np.zeros(
                (nb, ttn, nbp, TILE, TILE), dtype=np.int8
            )  # [ocl][tt][l][k][n]
            for ol, g in enumerate(blocks):
                for i in range(ttn):
                    tt = tt0 + i
                    ti, cp = divmod(tt, ncp)
                    ky, kx = divmod(job.taps[ti], 3) if kh == 3 else (0, 0)
                    for l in range(nbp):
                        icb = cp * nbp + l
                        # B[k][n] = W[oc = g*8+n][ic = icb*8+k]
                        blk = job.weight[
                            g * TILE : (g + 1) * TILE,
                            icb * TILE : (icb + 1) * TILE,
                            ky,
                            kx,
                        ]
                        tile_view[ol, i, l] = blk.T
            if nb:
                body[:tiles] = tile_view.view(np.uint8).reshape(-1)
                body[desc[D_BIAS] :] = (
                    job.bias[blocks[0] * TILE : (blocks[-1] + 1) * TILE]
                    .astype(np.int32)
                    .view(np.uint8)
                )
            slot[DESC_BYTES : DESC_BYTES + body.size] = body
    return out


def _rse(v: np.ndarray, s: int) -> np.ndarray:
    a = np.abs(v)
    return np.sign(v) * np.rint(a / float(1 << s))


def reference(job: Job, act: np.ndarray, resid: np.ndarray | None) -> np.ndarray:
    """dense [pixel][cin] uint8/int8 -> dense [pixel][cout] (uint8 view when out_flip else int8 view)."""
    lay = job.in_layout

    def rescale(u8, exp):
        if exp == 0:
            return u8
        v = u8.astype(np.int64) - 128
        v = _rse(v * (2.0**exp), 0) if exp > 0 else _rse(v, -exp)
        return (np.clip(v, -128, 127) + 128).astype(np.uint8)

    if job.kind == "copy":
        a, b = act, resid
        parts = []
        for which, block, exp in job.copy_spec:
            src = b if which else a
            parts.append(rescale(src[:, block * TILE : (block + 1) * TILE], exp))
        return np.concatenate(parts, axis=1)
    if job.kind == "avgpool":
        fmap = act.reshape(lay.h, lay.w, -1).astype(np.int64) - 128
        k, st = job.factor, job.stride
        oh, ow = job.out_layout.h, job.out_layout.w
        out = np.zeros((oh, ow, fmap.shape[2]))
        for oy in range(oh):
            for ox in range(ow):
                out[oy, ox] = fmap[oy * st : oy * st + k, ox * st : ox * st + k].reshape(k * k, -1).sum(axis=0)
        return (np.clip(_rse(out / (2.0**job.shift), 0), -128, 127) + 128).astype(np.uint8).reshape(oh * ow, -1)
    if job.kind == "d2s":
        f = job.factor
        nb_out = job.out_layout.nb
        fmap = act.reshape(lay.h, lay.w, -1)
        out = np.zeros((lay.h * f, lay.w * f, nb_out * TILE), dtype=np.uint8)
        for tap in range(f * f):
            ky, kx = divmod(tap, f)
            out[ky::f, kx::f] = fmap[:, :, tap * nb_out * TILE : (tap + 1) * nb_out * TILE]
        return out.reshape(-1, nb_out * TILE)
    if job.kind == "gap":
        v = act.astype(np.int64).reshape(lay.pixels, -1) - 128
        return (np.clip(_rse(v.sum(axis=0) / (2.0**job.shift), 0), -128, 127) + 128).astype(np.uint8).reshape(1, -1)
    if job.kind == "bmul":
        prod = (act.astype(np.int64) - 128) * (resid.astype(np.int64).reshape(1, -1) - 128)
        return (np.clip(_rse(prod / (2.0**job.shift), 0), -128, 127) + 128).astype(np.uint8)
    if job.kind == "add":
        c = max(-job.ea, 0) if job.ea < job.eb else max(-job.eb, 0)
        total = ((act.astype(np.int64) - 128) << (job.ea + c)) + (
            (resid.astype(np.int64) - 128) << (job.eb + c)
        )
        return (np.clip(_rse(total, c), -128, 127) + 128).astype(np.uint8)
    if job.kind == "up":
        fmap = act.reshape(lay.h, lay.w, -1)
        out = np.repeat(np.repeat(fmap, job.factor, axis=0), job.factor, axis=1)
        return rescale(out.reshape(-1, out.shape[2]), job.exp)
    if job.kind == "maxpool":
        fmap = act.reshape(lay.h, lay.w, -1)
        k, s, pad = job.factor, job.stride, (job.factor - 1) // 2
        padded = np.pad(fmap, ((pad, pad), (pad, pad), (0, 0)), constant_values=0)
        oh, ow = job.out_layout.h, job.out_layout.w
        out = np.zeros((oh, ow, fmap.shape[2]), dtype=np.uint8)
        for oy in range(oh):
            for ox in range(ow):
                out[oy, ox] = (
                    padded[oy * s : oy * s + k, ox * s : ox * s + k]
                    .reshape(k * k, -1)
                    .max(axis=0)
                )
        return rescale(out.reshape(oh * ow, -1), job.exp)
    if job.kind == "lut":
        return job.table.astype(np.uint8)[act.astype(np.uint8)]
    if job.kind == "dw":
        x = (
            (act.astype(np.int64) - 128)
            if job.in_flip
            else act.view(np.int8).astype(np.int64)
        )
        x = x.reshape(lay.h, lay.w, -1)
        s = job.stride
        k = job.weight.shape[2]
        padk = (k - 1) // 2
        xp = np.pad(x, ((padk, padk), (padk, padk), (0, 0)))
        oh, ow = job.out_layout.h, job.out_layout.w
        acc = np.zeros((oh, ow, x.shape[2]), dtype=np.int64)
        for ky in range(k):
            for kx in range(k):
                sub = xp[
                    ky : ky + (oh - 1) * s + 1 : s, kx : kx + (ow - 1) * s + 1 : s, :
                ]
                acc += sub * job.weight[:, 0, ky, kx].astype(np.int64)
        acc += job.bias.astype(np.int64)
        q = (
            np.clip(_rse(acc, job.shift), -128, 127)
            .astype(np.int64)
            .reshape(oh * ow, -1)
        )
        if job.relu:
            q = np.maximum(q, 0)
        q = np.minimum(q, job.clamp)
        return (
            (q + 128).astype(np.uint8)
            if job.out_flip
            else q.astype(np.int8).view(np.uint8)
        )
    if job.kind == "pool":
        fmap = act.reshape(lay.h, lay.w, -1)
        padded = np.pad(
            fmap, ((1, 1), (1, 1), (0, 0)), constant_values=0
        )  # post-ReLU bytes are >= 128: 0 never wins
        oh, ow = job.out_layout.h, job.out_layout.w
        out = np.zeros((oh, ow, fmap.shape[2]), dtype=np.uint8)
        for oy in range(oh):
            for ox in range(ow):
                out[oy, ox] = (
                    padded[oy * 2 : oy * 2 + 3, ox * 2 : ox * 2 + 3]
                    .reshape(9, -1)
                    .max(axis=0)
                )
        return out.reshape(oh * ow, -1)
    x = act.astype(np.int64)
    if job.in_flip:
        x = x - 128
    else:
        x = act.view(np.int8).astype(np.int64)
    oc, ic, kh, kw = job.weight.shape
    x = x.reshape(lay.h, lay.w, ic)
    s = job.stride
    pad = (kh // 2) if job.pad is None else job.pad
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    oh, ow = job.out_layout.h, job.out_layout.w
    # float64 BLAS is exact for these integer sums (|acc| < 2^53) and far faster than an int64 matmul
    acc = np.zeros((oh, ow, oc), dtype=np.float64)
    xf = xp.astype(np.float64)
    for ky in range(kh):
        for kx in range(kw):
            sub = xf[ky : ky + (oh - 1) * s + 1 : s, kx : kx + (ow - 1) * s + 1 : s, :]
            acc += sub @ job.weight[:, :, ky, kx].astype(np.float64).T
    acc += job.bias.astype(np.float64)
    q = np.clip(_rse(acc, job.shift), -128, 127).astype(np.int64).reshape(oh * ow, oc)
    if job.res_mode:
        r = (
            resid.astype(np.int64) - 128
            if job.res_mode == 2
            else resid.view(np.int8).astype(np.int64)
        )
        common = max(-job.ea, 0) if job.ea < job.eb else max(-job.eb, 0)
        total = (q << (job.ea + common)) + (r << (job.eb + common))
        q = np.clip(_rse(total, common), -128, 127).astype(np.int64)
    if job.relu:
        q = np.maximum(q, 0)
    q = np.minimum(q, job.clamp)
    if job.out_flip:
        return (q + 128).astype(np.uint8)
    return q.astype(np.int8).view(np.uint8)


def assign_slots(
    jobs: list[Job],
    pinned: list[int] | None = None,
    keep: list[int] | None = None,
    mapping: dict[int, int] | None = None,
) -> int:
    """Rewrite the jobs' logical slot ids to a small set of reused arena slots; returns the slot count.

    Slot 0 (the network input) stays 0 unless ``pinned`` lists the logical slots that already hold data before
    job 0 (host-computed maps): they become physical slots 0..n-1 and are never reused. ``keep`` lists logical
    slots whose contents must survive to the end (graph outputs). A slot is live from the job that writes it
    until the last job that reads it (a residual is read one job early: its fill is queued during the previous
    job). ``mapping`` (optional) receives logical -> physical.
    """
    pinned = list(pinned) if pinned is not None else [0]
    last_use: dict[int, int] = {}
    for index, job in enumerate(jobs):
        last_use[job.in_slot] = max(last_use.get(job.in_slot, -1), index)
        if job.res_slot is not None:
            last_use[job.res_slot] = max(last_use.get(job.res_slot, -1), index)
    for logical in keep or ():
        last_use[logical] = len(jobs)
    physical: dict[int, int] = {logical: i for i, logical in enumerate(pinned)}
    free: list[int] = []
    count = len(pinned)
    busy_until: dict[int, int] = {
        i: len(jobs) for i in range(len(pinned))
    }  # pinned slots are never reused
    for index, job in enumerate(jobs):
        for logical in (job.in_slot, job.res_slot):
            if logical is not None and logical not in physical:
                raise ValueError("slot read before it is written")
        # a slot is reusable once its last reader is strictly before this job (writes land after job start)
        for p, until in list(busy_until.items()):
            if until < index and p not in free:
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
    if mapping is not None:
        mapping.update(physical)
    return count


def stem_jobs(stem, in_slots, map_slot, pool_slot):
    """Stem Conv as 1x1 GEMM jobs over the host im2col chunks, then the MaxPool job.

    ``stem`` is ``stem_pool.extract_stem``; chunk ``c`` (64 pixels) is read from arena slot
    ``in_slots[c]`` and drained into the shared stem map at ``map_slot`` (spans 4 slots); the pool job
    turns that map into the first bottleneck's input at ``pool_slot``.
    """
    from stem_pool import CHUNK_PIXELS, geometry

    g = geometry(stem)
    weight = np.zeros((g["out_channels"], g["k_pad"], 1, 1), dtype=np.int8)
    weight[:, : g["k"], 0, 0] = stem["weights"].reshape(g["out_channels"], -1)
    lay_in = layout_for(g["k_pad"], CHUNK_PIXELS, 1)
    jobs = [
        Job(
            f"stem{c}",
            weight,
            stem["bias"],
            in_slots[c],
            map_slot,
            lay_in,
            in_flip=True,
            out_flip=True,
            relu=True,
            shift=int(stem["shift"]),
            stem_chunk=c,
            out_span_slots=4,
        )
        for c in range(g["chunks"])
    ]
    map_layout = Layout(
        g["out_channels"] // TILE, 1, g["ow"], g["oh"], region_bytes=g["pixels"] * TILE
    )
    jobs.append(
        Job(
            "pool",
            np.zeros((g["out_channels"], TILE, 1, 1), dtype=np.int8),
            np.zeros(g["out_channels"], dtype=np.int32),
            map_slot,
            pool_slot,
            map_layout,
            kind="pool",
        )
    )
    return jobs


STEM_JOBS = 5  # four stem GEMM chunks + the pool job
STEM_SLOTS = (
    9  # 4 im2col input slots, the 4-slot stem map, the pooled map (first body input)
)


def assemble_full(stem, body_jobs):
    """[stem x4, pool] + body jobs (whose slots were already reused by ``assign_slots``), all in one arena.

    Body slot ``p`` moves to ``STEM_SLOTS - 1 + p`` so its input slot 0 is the pool job's output.
    """
    shift = STEM_SLOTS - 1
    for job in body_jobs:
        job.in_slot += shift
        job.out_slot += shift
        if job.res_slot is not None:
            job.res_slot += shift
    return stem_jobs(stem, [0, 1, 2, 3], 4, STEM_SLOTS - 1) + list(body_jobs)


def arena_slots(jobs) -> int:
    return max(max(job.out_slot + job.out_span_slots, job.in_slot + 1) for job in jobs)
