#!/usr/bin/env python3
"""Run and collect a real-device IRON whole-array GEMM sweep.

The IRON example is supplied explicitly because it belongs to the installed
MLIR-AIE toolchain, not this repository.  This wrapper makes the benchmark
result reproducible without copying the compiler's design into onnxsim.

Example:
  python3 scripts/xdna/benchmark_gemm.py \
    --example /tmp/mlir-aie-xdna/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py \
    --cols 1 2 4 8 --json xdna-gemm.json
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run_one(args, columns):
    command = [
        sys.executable,
        str(args.example),
        "--dev",
        args.device,
        "-M",
        str(args.m),
        "-K",
        str(args.k),
        "-N",
        str(args.n),
        "--dtype_in",
        args.dtype_in,
        "--dtype_out",
        args.dtype_out,
        "--n-aie-cols",
        str(columns),
        "-m",
        str(args.tile_m),
        "-k",
        str(args.tile_k),
        "-n",
        str(args.tile_n),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    output = result.stdout + result.stderr
    npu = re.search(r"NPU time\s+\(avg/min/max us\):\s+([\d.]+)", output)
    end_to_end = re.search(r"End-to-end\s+\(avg/min/max us\):\s+([\d.]+)", output)
    gflops = re.search(r"NPU GFLOPS\s+:\s+([\d.]+)", output)
    if result.returncode or not (npu and end_to_end and gflops):
        raise RuntimeError(f"GEMM benchmark failed for {columns} columns:\n{output}")
    return {
        "columns": columns,
        "npu_avg_us": float(npu.group(1)),
        "end_to_end_avg_us": float(end_to_end.group(1)),
        "npu_gflops": float(gflops.group(1)),
        "verified": "PASS!" in output,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--device", default="npu2", choices=("npu", "npu2"))
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=32)
    parser.add_argument("--dtype-in", default="i16")
    parser.add_argument("--dtype-out", default="i32")
    parser.add_argument("--cols", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--peak-tops", type=float, default=50.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if not args.example.exists():
        parser.error(f"example does not exist: {args.example}")
    try:
        results = [run_one(args, columns) for columns in args.cols]
    except RuntimeError as exc:
        parser.exit(2, f"{exc}\n")
    report = {
        "backend": "amd_xdna_iron_xrt",
        "execution": "real_npu",
        "shape": [args.m, args.k, args.n],
        "tile": [args.tile_m, args.tile_k, args.tile_n],
        "dtype": f"{args.dtype_in}->{args.dtype_out}",
        "warmup": args.warmup,
        "iters": args.iters,
        "peak_tops_reference": args.peak_tops,
        "results": results,
        "profile": {
            f"{args.dtype_in.lower()}:{args.m}x{args.k}x{args.n}:c{result['columns']}": [
                args.tile_m, args.tile_k, args.tile_n
            ]
            for result in results
            if result["verified"]
        },
    }
    for result in report["results"]:
        result["peak_utilization_percent"] = (
            100.0 * result["npu_gflops"] / (args.peak_tops * 1000.0)
        )
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
