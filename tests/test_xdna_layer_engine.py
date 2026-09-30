"""Device-free checks of the layer-sequential XDNA engine's host side (arena layout, weight packing, slots)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

import layer_engine as le  # noqa: E402
import layer_engine_nets as nets  # noqa: E402


def _emulate(job, slot_bytes, act_dense, resid_dense):
    """Numpy model of kernels/layer_engine.cc driven by the packed descriptors of every core."""
    packed = le.pack_job(job, slot_bytes)
    lay_in, lay_out = job.in_layout, job.out_layout
    act = le.to_arena(act_dense, lay_in)
    resid = le.to_arena(resid_dense, lay_out) if resid_dense is not None else None
    out = np.zeros(le.SLOT_BYTES, dtype=np.uint8)
    p_in, oh, ow = lay_in.pixels, lay_out.h, lay_out.w
    for core in range(le.CORES):
        col, row = divmod(core, le.ROWS)
        acc = None
        for chunk in range(packed.shape[1]):
            slot = packed[col, chunk, row]
            d = slot[: le.DESC_BYTES].view(np.int32)
            nb = int(d[le.D_NB])
            if nb == 0:
                continue
            nbp, ncp, ntaps = int(d[le.D_NBP]), int(d[le.D_NCP]), int(d[le.D_NTAPS])
            taps = [int(t) for t in d[le.D_TAP0 : le.D_TAP0 + ntaps]]
            tt0, ttn = int(d[le.D_TT0]), int(d[le.D_TTN])
            bias_at = le.DESC_BYTES + int(d[le.D_BIAS])
            weights = (
                slot[le.DESC_BYTES : bias_at].view(np.int8).reshape(nb, ttn, nbp, 8, 8)
            )
            if d[le.D_FIRST]:
                bias = (
                    slot[bias_at : bias_at + nb * 32].view(np.int32).reshape(nb, 1, 8)
                )
                acc = np.tile(bias, (1, oh * ow, 1)).astype(np.int64)
            for i in range(ttn):
                ti, cp = divmod(tt0 + i, ncp)
                ky, kx = (
                    divmod(taps[ti], 3)
                    if job.weight.shape[2] == 3 or job.stride > 1
                    else (1, 1)
                )
                for blk in range(nbp):
                    raw = act[cp * le.REGION_BYTES + blk * p_in * 8 :][: p_in * 8]
                    # uint8-with-zero-point-128 inputs are re-centred (the kernel xors them with 0x80)
                    block = (
                        raw.astype(np.int64) - 128
                        if job.in_flip
                        else raw.view(np.int8).astype(np.int64)
                    )
                    block = block.reshape(lay_in.h, lay_in.w, 8)
                    for o in range(oh * ow):
                        oy, ox = divmod(o, ow)
                        if job.gather:
                            iy, ix = oy * job.stride + ky - 1, ox * job.stride + kx - 1
                        else:
                            iy, ix = oy, ox
                        if 0 <= iy < lay_in.h and 0 <= ix < lay_in.w:
                            for ol in range(nb):
                                acc[ol, o] += block[iy, ix] @ weights[
                                    ol, i, blk
                                ].astype(np.int64)
            if d[le.D_LAST]:
                sh, ea, eb = int(d[le.D_SHIFT]), int(d[le.D_EA]), int(d[le.D_EB])
                q = np.clip(le._rse(acc, sh), -128, 127).astype(np.int64)
                if d[le.D_RES]:
                    r = (
                        resid[core * le.REGION_BYTES :][: nb * oh * ow * 8]
                        .view(np.uint8)
                        .reshape(nb, oh * ow, 8)
                        .astype(np.int64)
                    )
                    r = (
                        r - 128
                        if d[le.D_RES] == 2
                        else r.astype(np.uint8).view(np.int8).astype(np.int64)
                    )
                    common = max(-ea, 0) if ea < eb else max(-eb, 0)
                    q = np.clip(
                        le._rse((q << (ea + common)) + (r << (eb + common)), common),
                        -128,
                        127,
                    )
                if d[le.D_RELU]:
                    q = np.maximum(q, 0)
                q = (
                    (q + 128).astype(np.uint8)
                    if d[le.D_OUT_FLIP]
                    else q.astype(np.int8).view(np.uint8)
                )
                out[core * le.REGION_BYTES :][: q.size] = q.reshape(-1)
    return le.from_arena(out, lay_out)


@pytest.mark.parametrize(
    "channels,w,h", [(64, 8, 8), (256, 8, 8), (512, 4, 4), (1024, 2, 2), (2048, 1, 1)]
)
def test_arena_roundtrip(channels, w, h):
    layout = le.layout_for(channels, w, h)
    dense = np.random.default_rng(0).integers(0, 256, (w * h, channels), dtype=np.uint8)
    assert np.array_equal(le.from_arena(le.to_arena(dense, layout), layout), dense)


@pytest.mark.parametrize("net", ["l1proj", "l2proj", "l3id", "l4id"])
@pytest.mark.parametrize(
    "slot_bytes", [4096, 2048]
)  # 2048 forces K-split chunks on the wide layers
def test_packed_chunks_reproduce_the_reference(net, slot_bytes):
    jobs, x = nets.build(net)
    maps, _ = nets.run_reference(jobs, x)
    for job in jobs:
        resid = maps[job.res_slot] if job.res_slot is not None else None
        want = maps[job.out_slot]
        try:
            got = _emulate(job, slot_bytes, maps[job.in_slot], resid)
        except ValueError:  # a single reduction step no longer fits this slot
            continue
        assert np.array_equal(got, want), job.name


def test_slot_reuse_never_overwrites_a_live_map():
    jobs, _segments, _x = nets.body()
    writer = {0: -1}
    for index, job in enumerate(jobs):
        writer[job.out_slot] = index
    wanted = [(job.in_slot, job.res_slot) for job in jobs]
    producers = [
        (writer[i] if i in writer else None, writer[r] if r is not None else None)
        for i, r in wanted
    ]
    slots = le.assign_slots(jobs)
    assert slots <= 5
    current = {0: -1}
    for index, job in enumerate(jobs):
        assert current[job.in_slot] == producers[index][0]
        if job.res_slot is not None:
            assert current[job.res_slot] == producers[index][1]
        current[job.out_slot] = index


def test_body_segments_repeat_identical_blocks():
    jobs, segments, _x = nets.body()
    covered = 0
    for first, count, repeat in segments:
        pattern = [
            le.n_chunks(j, le.ENGINE_SLOT_BYTES - le.DESC_BYTES)
            for j in jobs[first : first + count]
        ]
        for r in range(repeat):
            again = [
                le.n_chunks(j, le.ENGINE_SLOT_BYTES - le.DESC_BYTES)
                for j in jobs[first + r * count : first + (r + 1) * count]
            ]
            assert again == pattern
        covered += count * repeat
    assert covered == len(jobs) == 52


def test_stem_and_pool_jobs_match_the_stem_pool_emulation():
    import stem_pool

    stem = nets.synthetic_stem()
    image = np.random.default_rng(1).random((1, 3, 32, 32), dtype=np.float32)
    chunks = stem_pool.im2col_chunks(image, stem)
    per = chunks.size // 4
    jobs = le.stem_jobs(stem, [0, 1, 2, 3], 4, 8)
    conv = []
    for index, job in enumerate(jobs[:4]):
        dense = (
            chunks[index * per : (index + 1) * per]
            .reshape(19, 64, 8)
            .transpose(1, 0, 2)
            .reshape(64, 152)
        )
        got = le.reference(job, dense, None)
        assert np.array_equal(
            _emulate(job, 4096, dense, None), got
        )  # the packed chunks compute the same thing
        conv.append(got)
    pooled = le.reference(jobs[4], np.concatenate(conv), None)
    assert np.array_equal(
        pooled.reshape(8, 8, 64).transpose(2, 0, 1)[None],
        stem_pool.emulate(image, stem),
    )


def test_pool_job_descriptors_only_activate_the_eight_channel_blocks():
    stem = nets.synthetic_stem()
    pool = le.stem_jobs(stem, [0, 1, 2, 3], 4, 8)[4]
    packed = le.pack_job(pool, le.ENGINE_SLOT_BYTES)
    assert packed.shape == (le.COLS, 1, le.ROWS, le.ENGINE_SLOT_BYTES)
    for core in range(le.CORES):
        col, row = divmod(core, le.ROWS)
        desc = packed[col, 0, row, : le.DESC_BYTES].view(np.int32)
        assert (
            desc[le.D_MODE] == 8  # the stem pool is a k=3, stride-2 max-pool job
            and desc[le.D_NTAPS] == 3
            and desc[le.D_S] == 2
            and desc[le.D_KSZ] == 1  # the kernel reads the max-pool padding from here
            and desc[le.D_NB] == (1 if core < 8 else 0)
            and desc[le.D_CORE] == core
        )


def test_full_net_uses_disjoint_stem_slots_and_a_small_arena():
    jobs, segments, _stem = nets.full()
    assert [job.name for job in jobs[:5]] == [
        "stem0",
        "stem1",
        "stem2",
        "stem3",
        "pool",
    ]
    assert (
        jobs[4].out_slot == jobs[5].in_slot == le.STEM_SLOTS - 1
    )  # the pooled map feeds the first block
    assert le.arena_slots(jobs) <= le.STEM_SLOTS + 4
    assert (
        sum(count * repeat for _first, count, repeat in segments)
        == len(jobs) - le.STEM_JOBS
    )


def test_basic_blocks_pack_and_match_the_reference():
    # ResNet-18/34 style blocks: conv3x3 (stride 2 + uint8 input on the projection block) + conv3x3 + residual.
    jobs, segments, x = nets.basic_body(0, counts=(1, 2, 1, 1), hw=8)
    assert [job.name for job in jobs[:5]] == [
        "a",
        "b",
        "sk",
        "a",
        "b",
    ]  # stage 1 identity, then the stage-2 projection block
    maps, _ = nets.run_reference(jobs, x)
    for job in jobs:
        resid = maps[job.res_slot] if job.res_slot is not None else None
        got = _emulate(job, 4096, maps[job.in_slot], resid)
        assert np.array_equal(got, maps[job.out_slot]), job.name


def test_basic_body_segments_cover_every_job():
    jobs, segments, _x = nets.basic_body(0, counts=(3, 4, 6, 3))
    assert sum(count * repeat for _first, count, repeat in segments) == len(jobs)
    assert (
        segments[0][1] == 2 and segments[0][2] == 3
    )  # stage 1: three identity blocks [a, b]
    assert [seg[1] for seg in segments[1:]] == [
        3,
        2,
        3,
        2,
        3,
        2,
    ]  # projection [sk, a, b] + identity [a, b] per stage


def test_pow2_zero_padding_stays_zero_for_uint8_inputs():
    # A 3x3 over a uint8-with-zero-point-128 input pads with the *value* zero, i.e. byte 128.
    rng = np.random.default_rng(3)
    lay = le.layout_for(64, 4, 4)
    weight = rng.integers(-8, 8, (64, 64, 3, 3), dtype=np.int8)
    job = le.Job(
        "t", weight, np.zeros(64, dtype=np.int32), 0, 1, lay, in_flip=True, shift=6
    )
    dense = rng.integers(128, 256, (16, 64), dtype=np.uint8)
    got = _emulate(job, 4096, dense, None)
    assert np.array_equal(got, le.reference(job, dense, None))
