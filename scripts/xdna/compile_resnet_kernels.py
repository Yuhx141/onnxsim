#!/usr/bin/env python3
"""Compile buildable ResNet Conv GEMM specs with the installed IRON example.

This prepares kernel artifacts only. The generated kernels do not by
themselves execute the ONNX graph: runtime im2col, QDQ, bias/requantization,
residual, and fallback handling are still required.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import onnx

try:
    from .resnet_codegen import build_codegen_plan
    from .resnet_emitter import emit_kernel_specs, render_build_manifest
except ImportError:  # executed directly as a script
    from resnet_codegen import build_codegen_plan
    from resnet_emitter import emit_kernel_specs, render_build_manifest


def _compile(example: Path, spec: Any, device: str, output_dir: Path) -> dict[str, str]:
    stem = spec.key
    xclbin = output_dir / f"{stem}.xclbin"
    insts = output_dir / f"{stem}.insts.bin"
    m, k, n = spec.compiled_shape
    tm, tk, tn = spec.tile
    command = [
        sys.executable, str(example), "--dev", device,
        "-M", str(m), "-K", str(k), "-N", str(n),
        "-m", str(tm), "-k", str(tk), "-n", str(tn),
        "--n-aie-cols", str(spec.columns),
        "--dtype_in", "i8", "--dtype_out", "i32",
        "--xclbin-path", str(xclbin), "--insts-path", str(insts),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"IRON compile failed for {spec.key}:\n{completed.stdout}\n{completed.stderr}"
        )
    missing = [str(path) for path in (xclbin, insts) if not path.is_file()]
    if missing:
        raise RuntimeError(f"compiler reported success but did not create: {', '.join(missing)}")
    return {"xclbin": str(xclbin), "insts": str(insts)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("example", type=Path, help="MLIR-AIE whole_array.py")
    parser.add_argument("output", type=Path, help="JSON build and compiled-artifact manifest")
    parser.add_argument("--artifact-dir", type=Path, default=Path("xdna-resnet-artifacts"))
    parser.add_argument("--device", choices=("npu", "npu2"), default="npu2")
    parser.add_argument("--columns", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument(
        "--compile-all",
        action="store_true",
        help="also compile specs whose padding makes them inefficient for whole-array GEMM",
    )
    parser.add_argument(
        "--optimize-small-m",
        action="store_true",
        help="use the AIE2P 16-row int8 tile and minimum viable columns for small feature maps",
    )
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if not args.example.is_file():
        parser.error(f"IRON example does not exist: {args.example}")

    model = onnx.load(args.model)
    plan = build_codegen_plan(
        model,
        columns=args.columns,
        strict=True,
        optimize_small_m=args.optimize_small_m,
    )
    specs = emit_kernel_specs(plan, columns=args.columns)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    compiled: dict[str, dict[str, str]] = {}
    for spec in specs:
        if spec.buildable_with_whole_array or args.compile_all:
            compiled[spec.key] = _compile(args.example, spec, args.device, args.artifact_dir)

    report = dict(render_build_manifest(plan, model=model, columns=args.columns, source=str(args.example)))
    for item in report["kernels"]:
        artifact = compiled.get(item["key"])
        item["compiled_artifact"] = artifact
        item["compile_status"] = (
            "compiled" if artifact and item["buildable_with_whole_array"]
            else "compiled_with_padding" if artifact
            else "native_conv_required"
        )
    report["execution"] = "kernels_compiled_graph_runtime_pending"
    report["compile_all_specs"] = args.compile_all
    report["optimize_small_m"] = args.optimize_small_m
    report["compile_device"] = args.device
    report["compiled_kernel_count"] = len(compiled)
    report["graph_runtime_blockers"] = [
        "NCHW im2col packing and logical-to-padded GEMM staging",
        "QDQ scale/zero-point conversion and Conv bias/requantization",
        "non-Conv dispatches, residual paths, and unsupported operators",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"compiled {len(compiled)}/{len(specs)} kernel specs; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
