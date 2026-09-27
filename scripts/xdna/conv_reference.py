"""NumPy correctness oracle for the XDNA Conv→GEMM lowering."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

try:
    from .conv_lowering import ConvGemmPlan
except ImportError:  # direct script-directory imports used by tests/tooling
    from conv_lowering import ConvGemmPlan


def im2col_nchw(x: np.ndarray, plan: ConvGemmPlan) -> np.ndarray:
    """Pack NCHW input into grouped row-major GEMM panels."""
    if x.ndim != 4 or tuple(x.shape) != plan.input_shape:
        raise ValueError(f"input shape {x.shape} does not match {plan.input_shape}")
    batch, channels, height, width = x.shape
    _, _, kh, kw = plan.weight_shape
    sh, sw = plan.stride
    dh, dw = plan.dilation
    pt, pl, pb, pr = plan.pads
    padded = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant")
    out_h, out_w = plan.output_shape[2:]
    rows = batch * out_h * out_w
    channels_per_group = channels // plan.groups
    k = channels_per_group * kh * kw
    effective_h = dh * (kh - 1) + 1
    effective_w = dw * (kw - 1) + 1
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (effective_h, effective_w), axis=(2, 3)
    )
    windows = windows[:, :, ::sh, ::sw, ::dh, ::dw]
    panels = np.empty((plan.groups, rows, k), dtype=x.dtype)
    for group in range(plan.groups):
        channel_start = group * channels_per_group
        channel_end = channel_start + channels_per_group
        group_windows = windows[:, channel_start:channel_end]
        panels[group] = group_windows.transpose(0, 2, 3, 1, 4, 5).reshape(rows, k)
    return panels


def execute_conv_reference(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    plan: ConvGemmPlan,
) -> np.ndarray:
    """Execute the planned grouped im2col GEMM using NumPy int32 arithmetic."""
    if tuple(weight.shape) != plan.weight_shape:
        raise ValueError(f"weight shape {weight.shape} does not match {plan.weight_shape}")
    if bias is not None and tuple(bias.shape) != (plan.weight_shape[0],):
        raise ValueError("bias shape does not match Conv output channels")
    panels = im2col_nchw(np.asarray(x), plan)
    batch, out_channels, out_h, out_w = plan.output_shape
    out = np.empty(plan.output_shape, dtype=np.int32)
    channels_per_group = out_channels // plan.groups
    for group in range(plan.groups):
        w = weight[group * channels_per_group : (group + 1) * channels_per_group]
        matrix = w.reshape(channels_per_group, -1).T
        values = panels[group].astype(np.int32) @ matrix.astype(np.int32)
        if bias is not None:
            values += bias[group * channels_per_group : (group + 1) * channels_per_group].astype(np.int32)
        if plan.fused_relu:
            values = np.maximum(values, 0)
        out[:, group * channels_per_group : (group + 1) * channels_per_group] = values.reshape(
            batch, out_h, out_w, channels_per_group
        ).transpose(0, 3, 1, 2)
    return out
