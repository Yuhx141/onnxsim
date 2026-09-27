#!/usr/bin/env python3
"""Compare Vitis full-graph timing with XDNA codegen/kernel reports.

Kernel-only XDNA timings are reported beside Vitis for context, but are never
presented as a graph-level speedup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vitis_json", type=Path)
    parser.add_argument("xdna_json", type=Path)
    args = parser.parse_args()
    vitis = json.loads(args.vitis_json.read_text())
    xdna = json.loads(args.xdna_json.read_text())
    kernels = xdna.get("results", [])
    measurements = [
        measurement
        for kernel in kernels
        for measurement in (kernel.get("results") or [kernel])
    ]
    kernel_end_to_end_ms = [
        float(item["end_to_end_avg_us"]) / 1000.0
        for item in measurements
        if item.get("end_to_end_avg_us") is not None
    ]
    kernel_npu_ms = [
        float(item["npu_avg_us"]) / 1000.0
        for item in measurements
        if item.get("npu_avg_us") is not None
    ]
    xdna_full_graph = (
        xdna.get("execution") in {
            "full_graph_xdna_conv_host_ops",
            "full_graph_hybrid_conv_host_ops",
        }
        and xdna.get("avg_ms") is not None
    )
    comparison_valid = xdna_full_graph and vitis.get("execution") == "real_npu"
    result = {
        "vitis": {
            "execution": vitis.get("execution", "unspecified"),
            "full_graph_ms": vitis.get("avg_ms"),
            "full_graph_fps": vitis.get("fps"),
            "providers": vitis.get("providers", []),
        },
        "xdna": {
            "execution": xdna.get("execution", "kernel_level"),
            "graph_dispatches": xdna.get("graph_dispatches"),
            "full_graph_ms": xdna.get("avg_ms") if xdna_full_graph else None,
            "full_graph_fps": xdna.get("fps") if xdna_full_graph else None,
            "execution_counts": xdna.get("execution_counts"),
            "native_conv_required": xdna.get("native_conv_required"),
            "kernel_results": len(kernels),
            "kernel_end_to_end_ms_sum": sum(kernel_end_to_end_ms) if kernel_end_to_end_ms else None,
            "kernel_npu_ms_sum": sum(kernel_npu_ms) if kernel_npu_ms else None,
            "unsupported_ops": xdna.get("unsupported_ops", []),
        },
        "comparison_valid": comparison_valid,
        "comparison_note": "Both reports contain full-graph timing; the XDNA run may include CPU Conv fallback."
        if comparison_valid
        else "A graph-level comparison requires full-graph XDNA timing and a successful VitisAIExecutionProvider run. XDNA host-side ops are included in its timing.",
    }
    if comparison_valid and vitis.get("avg_ms"):
        result["xdna_vs_vitis_latency_ratio"] = float(xdna["avg_ms"]) / float(vitis["avg_ms"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
