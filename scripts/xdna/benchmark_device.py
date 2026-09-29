#!/usr/bin/env python3
"""Run a real IRON/XRT benchmark on an AMD XDNA device.

This is the first hardware-backed benchmark for the XDNA path.  It uses a
single vector-add kernel to validate the full pipeline: host allocation, DMA,
AIE execution, synchronization, and readback.  It intentionally does not
claim to benchmark an ONNX model yet.

The MLIR-AIE Python wheel, Peano, XRT, and a CPython-compatible ``pyxrt`` are
required.  On this host the compatible interpreter is Python 3.12.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def _percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def run(device_name: str, problem_size: int, tile_width: int, warmup: int, runs: int):
    # Imports are delayed so the planner benchmark remains dependency-light.
    import numpy as np
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out
    from aie.iron.algorithms import transform
    from aie.iron.device import from_name

    @iron.jit
    def vector_add(
        inp: In,
        out: Out,
        *,
        problem_size: CompileTime[int] = 1024,
        aie_tile_width: CompileTime[int] = 32,
    ):
        tensor_type = np.ndarray[(problem_size,), np.dtype[np.int32]]
        return transform(lambda x: x + 1, tensor_type, tile_size=aie_tile_width)

    iron.set_current_device(from_name(device_name, n_cols=None))
    inp = iron.arange(1, problem_size + 1, dtype=np.int32, device="npu")
    out = iron.zeros_like(inp)
    kwargs = {"problem_size": problem_size, "aie_tile_width": tile_width}

    start = time.perf_counter_ns()
    vector_add(inp, out, **kwargs)
    np_out = out.numpy()  # Synchronizes the device and includes readback.
    first_call_ns = time.perf_counter_ns() - start
    if not np.array_equal(np_out, inp.numpy() + 1):
        raise RuntimeError("XDNA vector-add result did not match the CPU reference")

    for _ in range(warmup):
        vector_add(inp, out, **kwargs)
        out.numpy()

    samples_ns = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        vector_add(inp, out, **kwargs)
        out.numpy()
        samples_ns.append(time.perf_counter_ns() - start)

    # One int32 input read and one int32 output write.  This is a transfer
    # estimate, not a measured DRAM bandwidth value.
    bytes_moved = problem_size * 4 * 2
    median_ns = statistics.median(samples_ns)
    return {
        "backend": "amd_xdna_iron_xrt",
        "device": device_name,
        "execution": "real_npu",
        "operation": "vector_add_int32",
        "problem_size": problem_size,
        "tile_width": tile_width,
        "warmup": warmup,
        "runs": runs,
        "first_call_ms": first_call_ns / 1_000_000,
        "warm_ms": {
            "median": median_ns / 1_000_000,
            "p95": _percentile(samples_ns, 0.95) / 1_000_000,
        },
        "estimated_bytes_per_run": bytes_moved,
        "estimated_effective_gib_per_s": bytes_moved / median_ns * 1e9 / (1024**3),
        "verified": True,
        "cache_mode": os.environ.get("NPU_CACHE_HOME", "default"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu2", choices=("npu", "npu2"))
    parser.add_argument("--problem-size", type=int, default=1024)
    parser.add_argument("--tile-width", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if min(args.problem_size, args.tile_width, args.runs) < 1 or args.warmup < 0:
        parser.error("sizes and runs must be positive; warmup cannot be negative")
    try:
        report = run(args.device, args.problem_size, args.tile_width, args.warmup, args.runs)
    except (ImportError, RuntimeError) as exc:
        parser.exit(2, f"XDNA device benchmark unavailable: {exc}\n")
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
