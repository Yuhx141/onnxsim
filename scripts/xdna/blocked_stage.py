"""Blocked-layout parameter packing for the vectorized linked bottleneck stage.

The `fused_bottleneck_blocked.cc` kernels keep activations as ``[C/8][pixel][8]``
int8 tiles and read weights as pre-tiled 8x8 MMUL ``B[k][n]`` operands, so every
MMUL operand is one contiguous 64-byte load. Parameter chunks keep the existing
order (conv1, skip, conv2a, conv2b, conv3) and slot size so the ObjectFifo
weight schedule is unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TILE = 8


def blocked_supported(binding: dict[str, Any]) -> bool:
    """Whether a block can use the blocked kernels (bound with ``blocked=True``)."""
    channels = int(binding["input_channels"])
    out_channels = int(binding["output_channels"])
    mid = binding["raw_weights"]["w1"].shape[0]
    c1, c2, c3 = binding["chunk_counts"]
    rows = (mid // c1, (mid // 2) // c2, out_channels // c3)
    return (
        bool(binding.get("blocked"))
        and tuple(binding["conv2_stride"]) in ((1, 1), (2, 2))
        and channels % TILE == 0 and out_channels % TILE == 0 and mid % (2 * TILE) == 0
        and all(r % TILE == 0 for r in rows)
    )


def _tile_1x1(weight: np.ndarray) -> np.ndarray:
    """[oc][ic] -> tiles ordered [ocb][icb] each B[k=ic%8][n=oc%8]."""
    oc, ic = weight.shape[:2]
    w = weight.reshape(oc // TILE, TILE, ic // TILE, TILE)  # ocb, n, icb, k
    return np.ascontiguousarray(w.transpose(0, 2, 3, 1))  # ocb, icb, k, n


def valid_taps(height: int, width: int, out_height: int, out_width: int, stride: int) -> list[int]:
    """3x3 taps (ky*3+kx) that read at least one real (non-padding) input pixel."""
    taps = []
    for tap in range(9):
        ky, kx = divmod(tap, 3)
        if any(
            0 <= oy * stride + ky - 1 < height and 0 <= ox * stride + kx - 1 < width
            for oy in range(out_height) for ox in range(out_width)
        ):
            taps.append(tap)
    return taps


def _tile_3x3(weight: np.ndarray) -> np.ndarray:
    """[oc][ic][1][ntaps] (pruned taps) -> tiles ordered [ocb][tap][icb] each B[k][n]."""
    oc, ic = weight.shape[:2]
    w = weight.reshape(oc // TILE, TILE, ic // TILE, TILE, -1)  # ocb,n,icb,k,tap
    return np.ascontiguousarray(w.transpose(0, 4, 2, 3, 1))  # ocb,tap,icb,k,n


HEADER_BYTES = 64


def runtime_header(binding: dict[str, Any]) -> np.ndarray:
    """int32 [shift1, shift2, shift3, skip_shift, main_residual_shift, skip_residual_shift]."""
    shifts = binding["shifts"]
    return np.array(
        [shifts[0], shifts[1], shifts[2], binding["skip_output_shift"] or 0,
         binding["main_residual_shift"], binding["skip_residual_shift"]],
        dtype=np.int32,
    )


def pack_blocked_params(binding: dict[str, Any], *, header: bool = False) -> np.ndarray:
    """Return the packed parameter tensor for one block in blocked layout.

    With ``header`` every weight slot grows by ``HEADER_BYTES`` and ends with the block's
    requantization shifts (see ``FUSED_RT_SHIFTS`` in fused_bottleneck_blocked.cc).
    """
    raw = binding["raw_weights"]
    mid = raw["w1"].shape[0]
    half = mid // 2
    c1, c2, c3 = binding["chunk_counts"]
    skip_chunks = int(binding["skip_chunk_count"])
    chunks: list[tuple[np.ndarray, np.ndarray]] = []

    def add(weight, bias, count, tiler):
        rows = weight.shape[0] // count
        for index in range(count):
            sl = slice(index * rows, (index + 1) * rows)
            chunks.append((tiler(weight[sl]), bias[sl]))

    add(raw["w1"], raw["b1"], c1, _tile_1x1)
    if skip_chunks:
        add(raw["skip_weight"], raw["skip_bias"], skip_chunks, _tile_1x1)
    for worker in range(2):
        sl = slice(worker * half, (worker + 1) * half)
        add(raw["w2"][sl], raw["b2"][sl], c2, _tile_3x3)
    add(raw["w3"], raw["b3"], c3, _tile_1x1)
    packed = []
    for weight, bias in chunks:
        raw_weight = np.ascontiguousarray(weight).view(np.uint8).reshape(-1)
        bias_offset = (raw_weight.size + 3) & ~3
        blob = np.zeros(bias_offset + bias.nbytes, dtype=np.uint8)
        blob[: raw_weight.size] = raw_weight
        blob[bias_offset:] = np.ascontiguousarray(bias).view(np.uint8)
        packed.append(blob)
    slot = int(binding["chunk_slot_bytes"])
    if max(blob.size for blob in packed) > slot or len(packed) * slot != binding["params"].size:
        raise ValueError("blocked parameter chunks do not match the compiled weight slot layout")
    stride = slot + (HEADER_BYTES if header else 0)
    params = np.zeros(len(packed) * stride, dtype=np.uint8)
    hdr = runtime_header(binding).view(np.uint8) if header else None
    for index, blob in enumerate(packed):
        params[index * stride : index * stride + blob.size] = blob
        if header:
            params[index * stride + slot : index * stride + slot + hdr.size] = hdr
    return params
