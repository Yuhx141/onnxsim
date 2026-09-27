#!/usr/bin/env python3
"""Measure the full quicktest ResNet through AMD's Vitis AI EP."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="enable ONNX Runtime profiling and write a compact event summary",
    )
    args = parser.parse_args()
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if args.profile_json:
        opts.enable_profiling = True
        opts.profile_file_prefix = str(args.profile_json.with_suffix(""))
    session = ort.InferenceSession(str(args.model), sess_options=opts, providers=["VitisAIExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_data = np.random.default_rng(0).random((1, 3, 32, 32), dtype=np.float32)
    feeds = {input_name: input_data}
    for _ in range(args.warmup):
        session.run(None, feeds)
    start = time.perf_counter_ns()
    for _ in range(args.iters):
        session.run(None, feeds)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    profile_trace = None
    if args.profile_json:
        profile_trace = Path(session.end_profiling())
        events = json.loads(profile_trace.read_text(encoding="utf-8"))
        grouped = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
        for event in events:
            if "dur" not in event:
                continue
            args_data = event.get("args", {})
            provider = args_data.get("provider", "runtime")
            key = f"{provider}:{event.get('cat', 'event')}:{event.get('name', 'unknown')}"
            grouped[key]["count"] += 1
            grouped[key]["duration_us"] += float(event["dur"])
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(
            json.dumps(
                {
                    "trace_file": str(profile_trace),
                    "events": sorted(
                        ({"event": key, **value} for key, value in grouped.items()),
                        key=lambda item: item["duration_us"],
                        reverse=True,
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    result = {
        "backend": "vitis_ai_ep",
        "execution": "real_npu" if "VitisAIExecutionProvider" in session.get_providers() else "provider_fallback",
        "requested_provider": "VitisAIExecutionProvider",
        "model": str(args.model),
        "warmup": args.warmup,
        "iters": args.iters,
        "avg_ms": elapsed_ms / args.iters,
        "fps": 1000.0 * args.iters / elapsed_ms,
        "providers": session.get_providers(),
    }
    if profile_trace is not None:
        result["profile_trace"] = str(profile_trace)
    if result["execution"] != "real_npu":
        result["execution_note"] = "VitisAIExecutionProvider did not load; these timings are not NPU measurements."
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
