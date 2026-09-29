"""Device-free checks of the blocked-layout weight packing used by the XDNA body kernels."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

from blocked_stage import (  # noqa: E402
    HEADER_BYTES,
    _tile_1x1,
    _tile_3x3,
    pack_blocked_params,
    runtime_header,
    valid_taps,
)


def _tile_matmul(a_tiles, b_tiles, k_tiles):
    """Emulate the kernel's per-output-block accumulation over 8x8 MMUL tiles."""
    acc = np.zeros((a_tiles[0].shape[0], b_tiles[0].shape[1]), dtype=np.int64)
    for a, b in zip(a_tiles, b_tiles):
        acc += a.astype(np.int64) @ b.astype(np.int64)
    return acc


def test_valid_taps_prune_padding_only_taps():
    # A 1x1 map only ever reads the centre tap; a 2x2 map keeps every tap.
    assert valid_taps(1, 1, 1, 1, 1) == [4]
    assert valid_taps(2, 2, 2, 2, 1) == list(range(9))
    # 4x4 -> 2x2 with stride 2 still touches all taps.
    assert valid_taps(4, 4, 2, 2, 2) == list(range(9))
    # 2x2 -> 1x1 with stride 2: only the taps covering the real pixels.
    assert valid_taps(2, 2, 1, 1, 2) == [4, 5, 7, 8]


def test_1x1_tiles_reproduce_the_matmul():
    rng = np.random.default_rng(0)
    oc, ic = 16, 24
    w = rng.integers(-128, 128, (oc, ic, 1, 1), dtype=np.int8)
    x = rng.integers(-128, 128, (8, ic), dtype=np.int8)  # 8 pixels x ic
    tiles = _tile_1x1(w).reshape(oc // 8, ic // 8, 8, 8)  # [ocb][icb][k][n]
    want = x.astype(np.int64) @ w.reshape(oc, ic).T.astype(np.int64)
    for ocb in range(oc // 8):
        a_tiles = [x[:, i * 8 : (i + 1) * 8] for i in range(ic // 8)]
        got = _tile_matmul(a_tiles, list(tiles[ocb]), ic // 8)
        assert np.array_equal(got, want[:, ocb * 8 : (ocb + 1) * 8])


def test_3x3_tiles_are_ordered_tap_major_within_each_output_block():
    rng = np.random.default_rng(1)
    oc, ic = 8, 16
    taps = [4, 5]
    w = rng.integers(-128, 128, (oc, ic, 1, len(taps)), dtype=np.int8)
    tiles = _tile_3x3(w).reshape(
        oc // 8, len(taps), ic // 8, 8, 8
    )  # ocb, tap, icb, k, n
    for t in range(len(taps)):
        for icb in range(ic // 8):
            for k in range(8):
                for n in range(8):
                    assert tiles[0, t, icb, k, n] == w[n, icb * 8 + k, 0, t]


def test_runtime_header_and_slot_layout():
    binding = {
        "shifts": (5, 9, 5),
        "skip_output_shift": None,
        "main_residual_shift": -1,
        "skip_residual_shift": 2,
    }
    assert runtime_header(binding).tolist() == [5, 9, 5, 0, -1, 2]
    assert HEADER_BYTES >= runtime_header(binding).nbytes


def test_pack_blocked_params_headers_every_slot():
    rng = np.random.default_rng(2)
    mid, cin, cout = 16, 8, 16
    raw = {
        "w1": rng.integers(-128, 128, (mid, cin, 1, 1), dtype=np.int8),
        "w2": rng.integers(-128, 128, (mid, mid, 1, 9), dtype=np.int8),
        "w3": rng.integers(-128, 128, (cout, mid, 1, 1), dtype=np.int8),
        "b1": np.arange(mid, dtype=np.int32),
        "b2": np.arange(mid, dtype=np.int32),
        "b3": np.arange(cout, dtype=np.int32),
        "skip_weight": None,
        "skip_bias": None,
    }
    slot = 4096
    binding = {
        "raw_weights": raw,
        "chunk_counts": (1, 1, 1),
        "skip_chunk_count": 0,
        "chunk_slot_bytes": slot,
        "params": np.zeros(4 * slot, dtype=np.uint8),
        "shifts": (3, 4, 5),
        "skip_output_shift": None,
        "main_residual_shift": 0,
        "skip_residual_shift": 1,
    }
    plain = pack_blocked_params(binding)
    headed = pack_blocked_params(binding, header=True)
    stride = slot + HEADER_BYTES
    assert plain.size == 4 * slot and headed.size == 4 * stride
    for chunk in range(4):
        payload = headed[chunk * stride : chunk * stride + slot]
        assert np.array_equal(payload, plain[chunk * slot : (chunk + 1) * slot])
        header = headed[chunk * stride + slot : chunk * stride + slot + 24].view(
            np.int32
        )
        assert header.tolist() == [3, 4, 5, 0, 0, 1]


def test_rt_descriptor_fields_and_slot_layout():
    from blocked_stage import RT_DESC_BYTES, pack_rt_params, rt_descriptor

    rng = np.random.default_rng(3)
    mid, cin, cout = 32, 16, 64
    raw = {
        "w1": rng.integers(-128, 128, (mid, cin, 1, 1), dtype=np.int8),
        "w2": rng.integers(-128, 128, (mid, mid, 1, 4), dtype=np.int8),
        "w3": rng.integers(-128, 128, (cout, mid, 1, 1), dtype=np.int8),
        "b1": np.arange(mid, dtype=np.int32),
        "b2": np.arange(mid, dtype=np.int32),
        "b3": np.arange(cout, dtype=np.int32),
        "skip_weight": rng.integers(-128, 128, (cout, cin, 1, 1), dtype=np.int8),
        "skip_bias": np.arange(cout, dtype=np.int32),
    }
    slot = 8192
    binding = {
        "raw_weights": raw,
        "chunk_counts": (2, 1, 2),
        "skip_chunk_count": 2,
        "input_width": 4,
        "input_height": 4,
        "input_channels": cin,
        "output_width": 2,
        "output_height": 2,
        "output_channels": cout,
        "conv2_stride": (2, 2),
        "conv2_taps": (4, 5, 7, 8),
        "chunk_slot_bytes": slot,
        "params": np.zeros(8 * slot, dtype=np.uint8),
        "shifts": (5, 6, 7),
        "skip_output_shift": 4,
        "main_residual_shift": -1,
        "skip_residual_shift": 0,
    }
    desc = rt_descriptor(binding)
    assert desc.size * 4 == RT_DESC_BYTES
    assert desc[:6].tolist() == [5, 6, 7, 4, -1, 0]  # shifts / residual exponents
    assert desc[6:15].tolist() == [
        4,
        4,
        cin,
        mid,
        cout,
        2,
        2,
        2,
        2,
    ]  # W H C MID OUT OW OH S SS
    assert desc[15:20].tolist() == [2, 1, 2, 2, 4]  # chunk counts, skip chunks, taps
    assert desc[20:24].tolist() == [4, 5, 7, 8]
    # bias offsets are relative to the tile area: align4(rows * K [* taps])
    assert desc[29:33].tolist() == [16 * 16, 16 * 32 * 4, 32 * 32, 32 * 16]
    # blocks per chunk, precomputed so the core never divides
    assert desc[33:37].tolist() == [
        (mid // 8) // 2,
        (mid // 16) // 1,
        (cout // 8) // 2,
        (cout // 8) // 2,
    ]

    packed = pack_rt_params(binding)
    stride = RT_DESC_BYTES + slot
    assert packed.size == 8 * stride  # conv1 x2, skip x2, conv2a, conv2b, conv3 x2
    for chunk in range(8):  # every slot starts with the same descriptor
        assert np.array_equal(
            packed[chunk * stride : chunk * stride + RT_DESC_BYTES], desc.view(np.uint8)
        )
