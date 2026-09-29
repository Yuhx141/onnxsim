"""Host side of the on-device stem Conv + MaxPool of a quantized (QDQ) ResNet.

The stem is a 7x7 stride-2 Conv followed by ReLU and a 3x3 stride-2 MaxPool. On the NPU it runs as
a 1x1 GEMM over a host-built im2col image (K = C*7*7 padded to a multiple of 8) in ``CHUNKS`` chunks
of pixels, then a pooling kernel over the assembled map (see BLK_STEM / BLK_POOL in
kernels/fused_bottleneck_rt.cc). Everything is integer: the requantization is a power-of-two shift.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from .blocked_stage import RT_DESC_BYTES, _align4, _tile_1x1
except ImportError:  # run as a script
    from blocked_stage import RT_DESC_BYTES, _align4, _tile_1x1

CHUNK_PIXELS = 64


def _values(model) -> dict[str, np.ndarray]:
    from onnx import (
        numpy_helper,  # lazy: the numpy-only helpers (im2col, packing) need no onnx
    )

    values = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    values[node.output[0]] = numpy_helper.to_array(attr.t)
    return values


def extract_stem(model) -> dict[str, Any]:
    """Read the first 7x7 Conv's quantization parameters and weights from a QDQ graph."""
    values = _values(model)
    producers = {out: node for node in model.graph.node for out in node.output}
    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    conv = next(
        n for n in model.graph.node
        if n.op_type == "Conv" and list(next(a for a in n.attribute if a.name == "kernel_shape").ints) == [7, 7]
    )
    attrs = {a.name: list(a.ints) for a in conv.attribute if a.ints}
    act_dq = producers[conv.input[0]]
    act_q = producers[act_dq.input[0]]
    weight_dq, bias_dq = producers[conv.input[1]], producers[conv.input[2]]
    relu = consumers[conv.output[0]][0]
    out_q = next(c for c in consumers[relu.output[0]] if c.op_type == "QuantizeLinear")
    pool = next(c for c in consumers[out_q.output[0]] + [x for q in consumers[out_q.output[0]] for x in consumers.get(q.output[0], [])] if c.op_type == "MaxPool")
    in_scale = float(values[act_dq.input[1]])
    in_zero = int(values[act_dq.input[2]])
    weights = np.asarray(values[weight_dq.input[0]], dtype=np.int8)
    w_scale = float(values[weight_dq.input[1]])
    bias_raw = np.asarray(values[bias_dq.input[0]], dtype=np.int32).reshape(-1)
    b_scale = float(values[bias_dq.input[1]])
    out_scale = float(values[out_q.input[1]])
    out_zero = int(values[out_q.input[2]])
    product = in_scale * w_scale
    shift_exact = math.log2(out_scale / product)
    if abs(shift_exact - round(shift_exact)) > 1e-9 or in_zero != 128 or out_zero != 128:
        raise ValueError("stem requantization must be a power of two with zero point 128")
    bias = bias_raw.astype(np.float64) * b_scale / product
    if not np.allclose(bias, np.rint(bias), atol=1e-6):
        raise ValueError("stem bias is not an exact integer accumulator")
    pool_attrs = {a.name: list(a.ints) for a in pool.attribute if a.ints}
    return {
        "weights": weights, "bias": np.rint(bias).astype(np.int32), "shift": int(round(shift_exact)),
        "in_scale": in_scale, "in_zero": in_zero, "strides": attrs["strides"], "pads": attrs["pads"],
        "pool_kernel": pool_attrs["kernel_shape"], "pool_strides": pool_attrs["strides"], "pool_pads": pool_attrs["pads"],
        "input_name": act_q.input[0],
    }


def geometry(stem: dict[str, Any], image_shape=(1, 3, 32, 32)) -> dict[str, int]:
    _, channels, height, width = image_shape
    kh = kw = stem["weights"].shape[2]
    sh, sw = stem["strides"]
    pt, pl = stem["pads"][0], stem["pads"][1]
    oh = (height + 2 * pt - kh) // sh + 1
    ow = (width + 2 * pl - kw) // sw + 1
    k = channels * kh * kw
    return {"k": k, "k_pad": (k + 7) // 8 * 8, "oh": oh, "ow": ow, "pixels": oh * ow,
            "chunks": oh * ow // CHUNK_PIXELS, "out_channels": stem["weights"].shape[0]}


def im2col_chunks(image: np.ndarray, stem: dict[str, Any]) -> np.ndarray:
    """float NCHW image -> the device input: quantize, pad, im2col, blocked ``[kb][64 px][8]`` chunks."""
    g = geometry(stem, image.shape)
    q = np.clip(np.rint(image / stem["in_scale"]) + stem["in_zero"], 0, 255).astype(np.uint8)[0]
    pt, pl = stem["pads"][0], stem["pads"][1]
    padded = np.pad(q, ((0, 0), (pt, stem["pads"][2]), (pl, stem["pads"][3])), constant_values=stem["in_zero"])
    sh, sw = stem["strides"]
    kh = kw = stem["weights"].shape[2]
    # All 7x7 windows at once: [C][OH][OW][kh][kw] -> [OH*OW][C*kh*kw] (K order c, ky, kx).
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(1, 2))[:, ::sh, ::sw]
    windows = windows[:, : g["oh"], : g["ow"]]
    cols = np.full((g["pixels"], g["k_pad"]), stem["in_zero"], dtype=np.uint8)
    cols[:, : g["k"]] = windows.transpose(1, 2, 0, 3, 4).reshape(g["pixels"], g["k"])
    chunks = []
    for c in range(g["chunks"]):
        block = cols[c * CHUNK_PIXELS : (c + 1) * CHUNK_PIXELS]                  # [64][k_pad]
        chunks.append(block.reshape(CHUNK_PIXELS, g["k_pad"] // 8, 8).transpose(1, 0, 2).reshape(-1))
    return np.concatenate(chunks)


def stem_descriptor(stem: dict[str, Any], image_shape=(1, 3, 32, 32)) -> np.ndarray:
    """int32[48] descriptor for fused_stem_chunk (field layout: blocked_stage.rt_descriptor)."""
    g = geometry(stem, image_shape)
    desc = np.zeros(RT_DESC_BYTES // 4, dtype=np.int32)
    desc[3] = stem["shift"]                                           # D_SKIPSHIFT
    desc[6:15] = [CHUNK_PIXELS, 1, g["k_pad"], 8, g["out_channels"], CHUNK_PIXELS, 1, 1, 1]
    desc[32] = _align4(g["out_channels"] * g["k_pad"])                # D_SKBIAS
    desc[36] = g["out_channels"] // 8                                 # D_NBS
    return desc


def stem_slot_bytes(stem: dict[str, Any], image_shape=(1, 3, 32, 32)) -> int:
    g = geometry(stem, image_shape)
    return RT_DESC_BYTES + _align4(g["out_channels"] * g["k_pad"]) + g["out_channels"] * 4


def pack_stem_params(stem: dict[str, Any], image_shape=(1, 3, 32, 32)) -> np.ndarray:
    g = geometry(stem, image_shape)
    weights = np.zeros((g["out_channels"], g["k_pad"]), dtype=np.int8)
    weights[:, : g["k"]] = stem["weights"].reshape(g["out_channels"], -1)
    tiles = np.ascontiguousarray(_tile_1x1(weights.reshape(g["out_channels"], g["k_pad"], 1, 1))).view(np.uint8).reshape(-1)
    out = np.zeros(stem_slot_bytes(stem, image_shape), dtype=np.uint8)
    out[:RT_DESC_BYTES] = stem_descriptor(stem, image_shape).view(np.uint8)
    out[RT_DESC_BYTES : RT_DESC_BYTES + tiles.size] = tiles
    bias_offset = RT_DESC_BYTES + _align4(tiles.size)
    out[bias_offset : bias_offset + g["out_channels"] * 4] = stem["bias"].view(np.uint8)
    return out


def emulate(image: np.ndarray, stem: dict[str, Any]) -> np.ndarray:
    """Numpy model of the device pipeline: returns the pooled uint8 NCHW map (zero point 128)."""
    g = geometry(stem, image.shape)
    data = im2col_chunks(image, stem)
    per = g["k_pad"] // 8 * CHUNK_PIXELS * 8
    weights = np.zeros((g["out_channels"], g["k_pad"]), dtype=np.int64)
    weights[:, : g["k"]] = stem["weights"].reshape(g["out_channels"], -1)
    conv = np.zeros((g["pixels"], g["out_channels"]), dtype=np.uint8)
    for c in range(g["chunks"]):
        blk = data[c * per : (c + 1) * per].reshape(g["k_pad"] // 8, CHUNK_PIXELS, 8).transpose(1, 0, 2).reshape(CHUNK_PIXELS, g["k_pad"])
        acc = (blk.astype(np.int64) - 128) @ weights.T + stem["bias"][None, :]
        mag = np.rint(np.abs(acc) / (1 << stem["shift"]))
        q = np.clip(np.sign(acc) * mag, -128, 127).astype(np.int64)
        conv[c * CHUNK_PIXELS : (c + 1) * CHUNK_PIXELS] = (np.maximum(q, 0) + 128).astype(np.uint8)
    fmap = conv.reshape(g["oh"], g["ow"], -1).transpose(2, 0, 1)  # [C][H][W]
    padded = np.pad(fmap, ((0, 0), (1, 1), (1, 1)), constant_values=0)
    out = np.zeros((fmap.shape[0], g["oh"] // 2, g["ow"] // 2), dtype=np.uint8)
    for oy in range(out.shape[1]):
        for ox in range(out.shape[2]):
            out[:, oy, ox] = padded[:, oy * 2 : oy * 2 + 3, ox * 2 : ox * 2 + 3].reshape(fmap.shape[0], -1).max(axis=1)
    return out[None]


def stem_nodes(model, value_name: str) -> set[int]:
    """Indices of every non-Constant node needed to produce ``value_name`` (its transitive producers).

    For the first bottleneck's input this is the whole stem: input Q, stem weight/bias DQ, Conv,
    ReLU, Q/DQ, MaxPool, Q (and the DQ after it): the nodes the device stem + pool replace.
    """
    producers = {out: index for index, node in enumerate(model.graph.node) for out in node.output}
    seen: set[int] = set()
    stack = [value_name]
    while stack:
        index = producers.get(stack.pop())
        if index is None or index in seen:
            continue
        seen.add(index)
        stack.extend(model.graph.node[index].input)
    return {i for i in seen if model.graph.node[i].op_type != "Constant"}
