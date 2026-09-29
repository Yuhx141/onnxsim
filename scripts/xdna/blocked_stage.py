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
    """Whether a block can use the blocked kernels (stride-1, 8-aligned dims)."""
    width = int(binding["input_width"])
    channels = int(binding["input_channels"])
    out_channels = int(binding["output_channels"])
    raw = binding["raw_weights"]
    mid = raw["w1"].shape[0]
    return (
        tuple(binding["conv2_stride"]) == (1, 1)
        and width % TILE == 0 and int(binding["output_width"]) == width
        and channels % TILE == 0 and out_channels % TILE == 0 and mid % (2 * TILE) == 0
        and tuple(binding["chunk_counts"]) == (1, 1, 1)
        and int(binding["skip_chunk_count"]) in (0, 1)
    )


def _tile_1x1(weight: np.ndarray) -> np.ndarray:
    """[oc][ic] -> tiles ordered [ocb][icb] each B[k=ic%8][n=oc%8]."""
    oc, ic = weight.shape[:2]
    w = weight.reshape(oc // TILE, TILE, ic // TILE, TILE)  # ocb, n, icb, k
    return np.ascontiguousarray(w.transpose(0, 2, 3, 1))  # ocb, icb, k, n


def _tile_3x3(weight: np.ndarray) -> np.ndarray:
    """[oc][ic][3][3] -> tiles ordered [ocb][tap][icb] each B[k][n]."""
    oc, ic = weight.shape[:2]
    w = weight.reshape(oc // TILE, TILE, ic // TILE, TILE, 3, 3)  # ocb,n,icb,k,ky,kx
    return np.ascontiguousarray(w.transpose(0, 4, 5, 2, 3, 1))  # ocb,ky,kx,icb,k,n


def pack_blocked_params(binding: dict[str, Any]) -> np.ndarray:
    """Return the packed parameter tensor for one block in blocked layout."""
    raw = binding["raw_weights"]
    mid = raw["w1"].shape[0]
    half = mid // 2
    chunks: list[tuple[np.ndarray, np.ndarray]] = [(_tile_1x1(raw["w1"]), raw["b1"])]
    if raw["skip_weight"] is not None:
        chunks.append((_tile_1x1(raw["skip_weight"]), raw["skip_bias"]))
    for worker in range(2):
        sl = slice(worker * half, (worker + 1) * half)
        chunks.append((_tile_3x3(raw["w2"][sl]), raw["b2"][sl]))
    chunks.append((_tile_1x1(raw["w3"]), raw["b3"]))
    packed = []
    for weight, bias in chunks:
        raw_weight = np.ascontiguousarray(weight).view(np.uint8).reshape(-1)
        bias_offset = (raw_weight.size + 3) & ~3
        blob = np.zeros(bias_offset + bias.nbytes, dtype=np.uint8)
        blob[: raw_weight.size] = raw_weight
        blob[bias_offset:] = np.ascontiguousarray(bias).view(np.uint8)
        packed.append(blob)
    slot = int(binding["chunk_slot_bytes"])
    if max(blob.size for blob in packed) > slot:
        raise ValueError("blocked parameter chunk exceeds the compiled weight slot")
    params = np.zeros(len(packed) * slot, dtype=np.uint8)
    for index, blob in enumerate(packed):
        params[index * slot : index * slot + blob.size] = blob
    return params
