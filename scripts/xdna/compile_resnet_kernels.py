#!/usr/bin/env python3
"""Compile ResNet Conv GEMM and selected native operator kernels.

This prepares kernel artifacts only. The generated kernels do not by
themselves execute the ONNX graph: runtime im2col, QDQ, bias/requantization,
residual, and fallback handling are still required.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _relu_tile_width(elements: int) -> int:
    """Pick the largest supported transform tile dividing this tensor."""
    for tile in (128, 64, 32, 16, 8, 4, 2, 1):
        if elements % tile == 0:
            return tile
    return 1


def _compile_relu(elements: int, device: str, output_dir: Path) -> dict[str, str]:
    key = f"relu_int8_e{elements}_t{_relu_tile_width(elements)}"
    xclbin = output_dir / f"{key}.xclbin"
    insts = output_dir / f"{key}.insts.bin"
    design = Path(__file__).with_name("relu_design.py")
    command = [
        sys.executable, str(design), "--dev", device,
        "--elements", str(elements), "--tile-width", str(_relu_tile_width(elements)),
        "--xclbin-path", str(xclbin), "--insts-path", str(insts),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"IRON ReLU compile failed for {key}:\n{completed.stdout}\n{completed.stderr}")
    missing = [str(path) for path in (xclbin, insts) if not path.is_file()]
    if missing:
        raise RuntimeError(f"compiler reported success but did not create: {', '.join(missing)}")
    return {"key": key, "xclbin": str(xclbin), "insts": str(insts)}


def _compile_quantized_add_relu(
    elements: int, quantization: dict[str, Any], device: str, output_dir: Path
) -> dict[str, str]:
    tile = _relu_tile_width(elements)
    scales = quantization["input_scales"]
    zeros = quantization["input_zero_points"]
    scale_out = float(quantization["output_scale"])
    zero_out = int(quantization["output_zero_point"])
    multipliers = quantization["multipliers_q30"]
    key = f"qadd_relu_u8_e{elements}_t{tile}_m{multipliers[0]}_{multipliers[1]}_z{zeros[0]}_{zeros[1]}_{zero_out}"
    xclbin = output_dir / f"{key}.xclbin"
    insts = output_dir / f"{key}.insts.bin"
    design = Path(__file__).with_name("quantized_add_relu_design.py")
    command = [
        sys.executable, str(design), "--dev", device,
        "--elements", str(elements), "--tile-width", str(tile),
        "--scale-a", str(scales[0]), "--scale-b", str(scales[1]),
        "--scale-out", str(scale_out), "--zero-a", str(zeros[0]),
        "--zero-b", str(zeros[1]), "--zero-out", str(zero_out),
        "--xclbin-path", str(xclbin), "--insts-path", str(insts),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"IRON quantized Add+ReLU compile failed for {key}:\n{completed.stdout}\n{completed.stderr}"
        )
    missing = [str(path) for path in (xclbin, insts) if not path.is_file()]
    if missing:
        raise RuntimeError(f"compiler reported success but did not create: {', '.join(missing)}")
    return {"key": key, "xclbin": str(xclbin), "insts": str(insts)}


def _compile_mul_scalar(elements: int, scalar: float, device: str, output_dir: Path) -> dict[str, str]:
    tile = _relu_tile_width(elements)
    scalar_key = float(scalar).hex().replace("+", "").replace("-", "m").replace(".", "p")
    key = f"mul_scalar_f32_e{elements}_t{tile}_s{scalar_key}"
    xclbin = output_dir / f"{key}.xclbin"
    insts = output_dir / f"{key}.insts.bin"
    design = Path(__file__).with_name("mul_scalar_design.py")
    command = [
        sys.executable, str(design), "--dev", device,
        "--elements", str(elements), "--tile-width", str(tile), "--scalar", str(scalar),
        "--xclbin-path", str(xclbin), "--insts-path", str(insts),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"IRON scalar Mul compile failed for {key}:\n{completed.stdout}\n{completed.stderr}")
    missing = [str(path) for path in (xclbin, insts) if not path.is_file()]
    if missing:
        raise RuntimeError(f"compiler reported success but did not create: {', '.join(missing)}")
    return {"key": key, "xclbin": str(xclbin), "insts": str(insts)}


def _compile_global_avgpool(
    channels: int, spatial: int, tile_channels: int, device: str, output_dir: Path
) -> dict[str, str]:
    key = f"global_avgpool_nchw_f32_c{channels}_s{spatial}_tc{tile_channels}"
    xclbin = output_dir / f"{key}.xclbin"
    insts = output_dir / f"{key}.insts.bin"
    design = Path(__file__).with_name("global_avgpool_design.py")
    command = [
        sys.executable, str(design), "--dev", device,
        "--channels", str(channels), "--spatial", str(spatial),
        "--tile-channels", str(tile_channels),
        "--xclbin-path", str(xclbin), "--insts-path", str(insts),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"IRON GlobalAveragePool compile failed for {key}:\n{completed.stdout}\n{completed.stderr}")
    missing = [str(path) for path in (xclbin, insts) if not path.is_file()]
    if missing:
        raise RuntimeError(f"compiler reported success but did not create: {', '.join(missing)}")
    return {"key": key, "xclbin": str(xclbin), "insts": str(insts)}


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

    compiled_relu: dict[int, dict[str, str]] = {}
    compiled_qadd: dict[tuple[Any, ...], dict[str, str]] = {}
    compiled_mul: dict[tuple[int, float], dict[str, str]] = {}
    compiled_gap: dict[tuple[int, int, int], dict[str, str]] = {}
    report = dict(render_build_manifest(plan, model=model, columns=args.columns, source=str(args.example)))
    fused_relu_indices = {
        int(index)
        for item in report["operation_kernels"]
        if item["status"] == "compilable_quantized_add_relu" and item.get("quantization")
        for index in item["quantization"]["fused_node_indices"][1:2]
    }
    for item in report["operation_kernels"]:
        if item["status"] != "compilable_global_avgpool_f32":
            continue
        params = item["parameters"]
        cache_key = (int(params["channels"]), int(params["spatial"]), int(params["tile_channels"]))
        artifact = compiled_gap.get(cache_key)
        if artifact is None:
            artifact = _compile_global_avgpool(*cache_key, args.device, args.artifact_dir)
            compiled_gap[cache_key] = artifact
        item["compiled_artifact"] = artifact
        item["compile_status"] = "compiled"
    for item in report["operation_kernels"]:
        if item["op_type"] == "Relu" and int(item["node_index"]) in fused_relu_indices:
            item["status"] = "fused_into_quantized_add_relu"
            item["compile_status"] = "covered_by_quantized_add_relu"
            continue
        if item["op_type"] != "Relu" or item["status"] != "compilable_iron_kernel":
            continue
        output_shape = next((shape for shape in item["output_shapes"] if shape), None)
        if not output_shape:
            item["compile_status"] = "shape_required"
            continue
        elements = math.prod(output_shape)
        artifact = compiled_relu.get(elements)
        if artifact is None:
            artifact = _compile_relu(elements, args.device, args.artifact_dir)
            compiled_relu[elements] = artifact
        item["compiled_artifact"] = artifact
        item["compile_status"] = "compiled"
    for item in report["operation_kernels"]:
        if item["status"] != "compilable_mul_scalar_f32":
            continue
        output_shape = next((shape for shape in item["output_shapes"] if shape), None)
        if not output_shape:
            item["compile_status"] = "shape_required"
            continue
        elements = math.prod(output_shape)
        scalar = float(item["parameters"]["scalar"])
        cache_key = (elements, scalar)
        artifact = compiled_mul.get(cache_key)
        if artifact is None:
            artifact = _compile_mul_scalar(elements, scalar, args.device, args.artifact_dir)
            compiled_mul[cache_key] = artifact
        item["compiled_artifact"] = artifact
        item["compile_status"] = "compiled"
    for item in report["operation_kernels"]:
        if item["status"] != "compilable_quantized_add_relu":
            continue
        output_shape = next((shape for shape in item["output_shapes"] if shape), None)
        if not output_shape:
            item["compile_status"] = "shape_required"
            continue
        elements = math.prod(output_shape)
        quantization = item["quantization"]
        cache_key = (
            elements, tuple(quantization["multipliers_q30"]),
            tuple(quantization["input_zero_points"]), quantization["output_scale"],
            quantization["output_zero_point"],
        )
        artifact = compiled_qadd.get(cache_key)
        if artifact is None:
            artifact = _compile_quantized_add_relu(
                elements, quantization, args.device, args.artifact_dir
            )
            compiled_qadd[cache_key] = artifact
        item["compiled_artifact"] = artifact
        item["compile_status"] = "compiled"

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
    report["compiled_kernel_count"] = len(compiled) + len(compiled_relu) + len(compiled_qadd) + len(compiled_mul) + len(compiled_gap)
    report["compiled_operator_kernel_count"] = len(compiled_relu) + len(compiled_qadd) + len(compiled_mul) + len(compiled_gap)
    report["graph_runtime_blockers"] = [
        "NCHW im2col packing and logical-to-padded GEMM staging",
        "QDQ scale/zero-point conversion and Conv bias/requantization",
        "runtime dispatch wiring for compiled standalone operators",
        "MaxPool, generic QDQ conversion, and tensor-layout kernels",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"compiled {len(compiled)}/{len(specs)} kernel specs; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
