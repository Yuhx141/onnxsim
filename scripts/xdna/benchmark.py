#!/usr/bin/env python3
"""Benchmark the dependency-light XDNA planning path.

This benchmark intentionally measures planning only unless a future XRT
launcher is connected.  It is useful on every machine and never labels a
CPU/planner result as NPU throughput.

Examples:
    python scripts/xdna/benchmark.py
    python scripts/xdna/benchmark.py --runs 1000 --json results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from xdna_backend import load_kernel_manifest, optimize_partitions, partition_model  # noqa: E402


@dataclass(frozen=True)
class Workload:
    name: str
    ops: Sequence[str]
    macs: int
    bytes_moved: int
    kernel: str


WORKLOADS = (
    Workload("elementwise", ("Add", "Mul", "Relu"), 0, 3 * 4096 * 2, "elementwise_f16"),
    Workload("mlp_block", ("MatMul", "Gemm", "Gelu", "MatMul"), 2 * 4096 * 4096, 3 * 4096 * 4096 * 2, "gemm_f16"),
    Workload("conv_block", ("Conv", "Relu", "Conv", "Add"), 2 * 56 * 56 * 64 * 64 * 9, 4 * 56 * 56 * 64 * 2, "conv_f16"),
    Workload("attention_like", ("MatMul", "Softmax", "MatMul", "UnsupportedAttentionOp"), 2 * 2048 * 2048 * 128, 4 * 2048 * 2048 * 2, "attention_f16"),
)


def _model(ops: Sequence[str]):
    return SimpleNamespace(
        graph=SimpleNamespace(
            node=[SimpleNamespace(op_type=op, name=f"{op}_{i}") for i, op in enumerate(ops)]
        )
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def benchmark_workload(workload: Workload, runs: int, available_kernels=None) -> Dict[str, object]:
    model = _model(workload.ops)
    # Warm up object allocation and imports before collecting timings.
    partition_model(model)
    samples_ns: List[int] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        partitions = partition_model(model)
        dispatches = optimize_partitions(partitions)
        samples_ns.append(time.perf_counter_ns() - start)
    xdna_nodes = sum(len(part.nodes) for part in partitions if part.device == "XDNA")
    total_nodes = sum(len(part.nodes) for part in partitions)
    return {
        "name": workload.name,
        "nodes": total_nodes,
        "xdna_nodes": xdna_nodes,
        "coverage_percent": 100.0 * xdna_nodes / total_nodes if total_nodes else 0.0,
        "estimated_macs": workload.macs,
        "estimated_bytes": workload.bytes_moved,
        "arithmetic_intensity_macs_per_byte": workload.macs / workload.bytes_moved if workload.bytes_moved else 0.0,
        "kernel": workload.kernel,
        "kernel_artifact": workload.kernel in (available_kernels or ()),
        "naive_dispatches": total_nodes,
        "optimized_dispatches": len(dispatches),
        "dispatch_reduction_percent": (
            100.0 * (total_nodes - len(dispatches)) / total_nodes if total_nodes else 0.0
        ),
        "planner_us": {
            "median": statistics.median(samples_ns) / 1_000.0,
            "p95": _percentile(samples_ns, 0.95) / 1_000.0,
        },
        "execution": "not_available",
    }


def run(runs: int, manifest=None) -> Dict[str, object]:
    available_kernels = ()
    if manifest is not None:
        available_kernels = tuple(manifest["kernels"])
    return {
        "backend": "amd_xdna_phase1_planner",
        "npu_execution": False,
        "runs": runs,
        "artifact_manifest": bool(manifest),
        "workloads": [benchmark_workload(workload, runs, available_kernels) for workload in WORKLOADS],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200, help="timed planner iterations per workload")
    parser.add_argument("--manifest", type=Path, help="offline XDNA kernel manifest to check for coverage")
    parser.add_argument("--json", type=Path, help="also write the report to this path")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be positive")
    try:
        manifest = load_kernel_manifest(args.manifest) if args.manifest else None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"invalid XDNA manifest: {exc}")
    report = run(args.runs, manifest)
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
