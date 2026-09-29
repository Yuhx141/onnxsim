#!/usr/bin/env python3
"""Emit an offline XDNA build manifest for a QDQ ResNet ONNX graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx

try:
    from .resnet_codegen import build_codegen_plan
    from .resnet_emitter import write_build_manifest
except ImportError:  # executed as a standalone script
    from resnet_codegen import build_codegen_plan
    from resnet_emitter import write_build_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--source", default="mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py")
    args = parser.parse_args()
    model = onnx.load(args.model)
    plan = build_codegen_plan(model, columns=args.columns, strict=True)
    write_build_manifest(plan, args.output, model=model, columns=args.columns, source=args.source)
    print(f"wrote {args.output} ({plan.estimated_dispatches} dispatches, {len(plan.conv_dispatches)} Conv dispatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
