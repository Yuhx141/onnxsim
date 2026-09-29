#!/usr/bin/env python3
"""Measure an FP32 Axera binary template on the AXCL device.

The reported device time comes from ``axclrtEngineExecute`` inside the
persistent LXD guest runner. Host round-trip time includes tensor staging and
the runner protocol. This is intended for a compiled ``.axmodel`` fixture,
not a Pulsar compiler profile.

    python profile_fp32_binary.py [--model fixtures/fp32_binary/add_16x1000.axmodel.gz]
        --warmup 20 --runs 500
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import time

import numpy as np

import axcl_session

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "fixtures", "fp32_binary", "add_16x1000.axmodel.gz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--op", choices=("Add", "Mul", "Div"), default="Add")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", help="also write the JSON result here")
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be nonnegative and --runs must be positive")

    with gzip.open(args.model, "rb") as stream:
        blob = stream.read()
    exec_times_us: list[int] = []
    roundtrip_ms: list[float] = []
    rng = np.random.default_rng(args.seed)
    with axcl_session.AXSession(subdir="fp32_binary_profile") as session:
        model = session.load(blob)
        try:
            if len(model.inputs) != 2 or len(model.outputs) != 1:
                raise ValueError("expected a two-input, one-output binary model")
            if any(spec.dtype != np.float32 for spec in (*model.inputs, *model.outputs)):
                raise ValueError("expected a float32 model")
            inputs = [
                rng.normal(size=spec.shape).astype(np.float32) for spec in model.inputs
            ]
            if args.op == "Div":
                inputs[1] = np.abs(inputs[1]) + np.float32(0.5)
            for _ in range(args.warmup):
                outputs = session.run(model, inputs)
            if args.op == "Add":
                expected = np.add(inputs[0], inputs[1])
            elif args.op == "Mul":
                expected = np.multiply(inputs[0], inputs[1])
            else:
                expected = np.divide(inputs[0], inputs[1])
            max_error = float(np.max(np.abs(outputs[0] - expected)))

            for _ in range(args.runs):
                before_us = session.exec_us
                start = time.perf_counter()
                outputs = session.run(model, inputs)
                roundtrip_ms.append((time.perf_counter() - start) * 1000)
                exec_times_us.append(session.exec_us - before_us)

            result = {
                "model": os.path.relpath(args.model, HERE),
                "op": args.op,
                "device": "AX8850 via axcl-vm",
                "shape": list(outputs[0].shape),
                "dtype": str(outputs[0].dtype),
                "warmup_runs": args.warmup,
                "measured_runs": args.runs,
                "max_abs_error_vs_fp32": max_error,
                "device_exec_us": {
                    "mean": statistics.fmean(exec_times_us),
                    "median": statistics.median(exec_times_us),
                    "p95": float(np.percentile(exec_times_us, 95)),
                    "total": sum(exec_times_us),
                },
                "host_roundtrip_ms": {
                    "mean": statistics.fmean(roundtrip_ms),
                    "median": statistics.median(roundtrip_ms),
                    "p95": float(np.percentile(roundtrip_ms, 95)),
                },
            }
            rendered = json.dumps(result, indent=2)
            print(rendered)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as stream:
                    stream.write(rendered + "\n")
        finally:
            session.unload(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
