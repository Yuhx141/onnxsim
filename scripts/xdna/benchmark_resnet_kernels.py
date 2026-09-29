#!/usr/bin/env python3
"""Benchmark the whole-array-compatible Conv GEMMs selected from a ResNet."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import onnx

from resnet_codegen import build_codegen_plan
from resnet_emitter import emit_kernel_specs


def _run_kernel(example: Path, spec: Any, device: str, warmup: int, iters: int) -> dict[str, Any]:
    m, k, n = spec.compiled_shape
    tm, tk, tn = spec.tile
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_gemm.py")),
        "--example", str(example), "--device", device,
        "--m", str(m), "--k", str(k), "--n", str(n),
        "--tile-m", str(tm), "--tile-k", str(tk), "--tile-n", str(tn),
        "--dtype-in", "i8", "--dtype-out", "i32",
        "--cols", str(spec.columns), "--warmup", str(warmup), "--iters", str(iters),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    output = result.stdout.strip()
    if result.returncode:
        raise RuntimeError(f"benchmark failed for {spec.key}:\n{result.stdout}\n{result.stderr}")
    start = output.find("{")
    if start < 0:
        raise RuntimeError(f"benchmark returned no JSON for {spec.key}:\n{output}")
    report = json.loads(output[start:])
    report["kernel_key"] = spec.key
    report["logical_shape"] = list(spec.gemm_shape)
    report["compiled_shape"] = list(spec.compiled_shape)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("example", type=Path)
    parser.add_argument("--device", default="npu2", choices=("npu", "npu2"))
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    plan = build_codegen_plan(onnx.load(args.model), columns=args.columns, strict=True)
    specs = emit_kernel_specs(plan, columns=args.columns)
    results = []
    for spec in specs:
        if spec.buildable_with_whole_array:
            results.append(_run_kernel(args.example, spec, args.device, args.warmup, args.iters))
    report = {
        "backend": "amd_xdna_iron_xrt",
        "execution": "kernel_level",
        "graph_dispatches": plan.estimated_dispatches,
        "kernel_specs": len(specs),
        "whole_array_benchmarks": len(results),
        "native_conv_required": sum(not spec.buildable_with_whole_array for spec in specs),
        "unsupported_ops": list(plan.unsupported_ops),
        "results": results,
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
