"""Device-free checks of the on-device stem Conv + MaxPool host side (stem_pool.py)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

import stem_pool  # noqa: E402


def _synthetic_stem(seed=0, out_channels=16):
    rng = np.random.default_rng(seed)
    return {
        "weights": rng.integers(-8, 8, (out_channels, 3, 7, 7), dtype=np.int8),
        "bias": rng.integers(-2000, 2000, out_channels, dtype=np.int32),
        "shift": 9,
        "in_scale": 2.0**-7,
        "in_zero": 128,
        "strides": [2, 2],
        "pads": [3, 3, 3, 3],
        "pool_kernel": [3, 3],
        "pool_strides": [2, 2],
        "pool_pads": [1, 1, 1, 1],
        "input_name": "input",
    }


def _reference(image, stem):
    """Direct loops: quantize, 7x7/2 conv with bias, requantize (half-even shift), relu, 3x3/2 pool."""
    q = (
        np.clip(np.rint(image / stem["in_scale"]) + stem["in_zero"], 0, 255).astype(
            np.int64
        )[0]
        - 128
    )
    pad = np.pad(q, ((0, 0), (3, 3), (3, 3)))
    w = stem["weights"].astype(np.int64)
    oc = w.shape[0]
    conv = np.zeros((oc, 16, 16), dtype=np.int64)
    for oy in range(16):
        for ox in range(16):
            win = pad[:, oy * 2 : oy * 2 + 7, ox * 2 : ox * 2 + 7]
            acc = np.tensordot(w, win, axes=([1, 2, 3], [0, 1, 2])) + stem["bias"]
            mag = np.rint(np.abs(acc) / (1 << stem["shift"]))
            conv[:, oy, ox] = np.clip(np.sign(acc) * mag, -128, 127)
    act = np.maximum(conv, 0) + 128
    out = np.zeros((oc, 8, 8), dtype=np.int64)
    for oy in range(8):
        for ox in range(8):
            ys = range(max(2 * oy - 1, 0), min(2 * oy + 2, 16))
            xs = range(max(2 * ox - 1, 0), min(2 * ox + 2, 16))
            out[:, oy, ox] = np.max([act[:, y, x] for y in ys for x in xs], axis=0)
    return out[None].astype(np.uint8)


def test_emulated_device_pipeline_matches_direct_reference():
    stem = _synthetic_stem()
    image = np.random.default_rng(1).random((1, 3, 32, 32), dtype=np.float32) * 2 - 1
    assert np.array_equal(stem_pool.emulate(image, stem), _reference(image, stem))


def test_im2col_chunks_layout():
    stem = _synthetic_stem()
    image = np.random.default_rng(2).random((1, 3, 32, 32), dtype=np.float32)
    g = stem_pool.geometry(stem)
    assert (g["k"], g["k_pad"], g["pixels"], g["chunks"]) == (147, 152, 256, 4)
    data = stem_pool.im2col_chunks(image, stem)
    assert data.size == 4 * 152 * 64 and data.dtype == np.uint8
    # chunk 0, pixel 0, K element 0 is channel 0, window (ky=0, kx=0) = padding -> zero point
    assert data[0] == 128
    # K padding (elements 147..151) holds the zero point in every pixel
    chunk = data[: 152 * 64].reshape(19, 64, 8)
    assert (chunk[18, :, 3:] == 128).all()  # k = 18*8 + (3..7) = 147..151


def test_stem_slot_and_descriptor():
    stem = _synthetic_stem(out_channels=64)
    slot = stem_pool.pack_stem_params(stem)
    assert slot.size == stem_pool.stem_slot_bytes(stem)
    desc = slot[: stem_pool.RT_DESC_BYTES].view(np.int32)
    assert desc[3] == 9  # requantization shift
    assert desc[8] == 152 and desc[10] == 64  # padded K and output channels
    assert desc[36] == 8  # output blocks
    bias_at = stem_pool.RT_DESC_BYTES + desc[32]
    assert np.array_equal(slot[bias_at : bias_at + 64 * 4].view(np.int32), stem["bias"])
