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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="enable ONNX Runtime profiling and write a compact event summary",
    )
    parser.add_argument("--capture-npz", type=Path, help="save selected ONNX intermediate outputs")
    parser.add_argument("--capture-output-name", action="append", default=[], help="ONNX value name to expose and capture")
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 0:
        parser.error("--iters must be positive and --warmup must be nonnegative")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(
        str(args.model), sess_options=opts,
        providers=["VitisAIExecutionProvider"], provider_options=[{}],
    )
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    shape = tuple(int(dim) if isinstance(dim, int) and dim > 0 else 1 for dim in input_info.shape)
    input_data = np.random.default_rng(args.seed).random(shape, dtype=np.float32)
    feeds = {input_name: input_data}
    for _ in range(args.warmup):
        session.run(None, feeds)
    start = time.perf_counter_ns()
    for _ in range(args.iters):
        session.run(None, feeds)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    provider_output = session.run(None, feeds)[0]
    cpu_session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    cpu_output = cpu_session.run(None, feeds)[0]
    delta = np.abs(provider_output.astype(np.float64) - cpu_output.astype(np.float64))
    profile_trace = None
    vitis_node_count = None
    if args.profile_json:
        profile_opts = ort.SessionOptions()
        profile_opts.log_severity_level = 3
        profile_opts.enable_profiling = True
        profile_opts.profile_file_prefix = str(args.profile_json.with_suffix(""))
        profile_session = ort.InferenceSession(
            str(args.model), sess_options=profile_opts,
            providers=["VitisAIExecutionProvider"], provider_options=[{}],
        )
        profile_session.run(None, feeds)
        profile_trace = Path(profile_session.end_profiling())
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
        vitis_node_count = sum(
            value["count"] for key, value in grouped.items()
            if key.startswith("VitisAIExecutionProvider:Node:")
        )
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
        "input_seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "avg_ms": elapsed_ms / args.iters,
        "fps": 1000.0 * args.iters / elapsed_ms,
        "providers": session.get_providers(),
        "input_shape": list(shape),
        "cpu_reference": {
            "max_abs_error": float(delta.max(initial=0.0)),
            "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
            "argmax_match": int(np.argmax(provider_output)) == int(np.argmax(cpu_output)),
        },
    }
    if vitis_node_count is not None:
        result["vitis_npu_node_count"] = vitis_node_count
    if profile_trace is not None:
        result["profile_trace"] = str(profile_trace)
    if args.capture_output_name:
        import onnx

        tapped = onnx.load(args.model)
        inferred = onnx.shape_inference.infer_shapes(tapped)
        value_info = {
            item.name: item
            for item in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
        }
        existing = {item.name for item in tapped.graph.output}
        for name in args.capture_output_name:
            if name not in value_info:
                raise ValueError(f"cannot find inferred type/shape for capture value {name!r}")
            if name not in existing:
                tapped.graph.output.append(value_info[name])
                existing.add(name)
        capture_model = args.model.with_name(args.model.stem + "-capture.onnx")
        onnx.save(tapped, capture_model)
        capture_session = ort.InferenceSession(
            str(capture_model), providers=["VitisAIExecutionProvider"], provider_options=[{}],
        )
        captured_values = capture_session.run(list(args.capture_output_name), feeds)
        capture_arrays = {
            name: np.asarray(value) for name, value in zip(args.capture_output_name, captured_values)
        }
        if args.capture_npz:
            args.capture_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.capture_npz, **capture_arrays)
            result["capture_npz"] = str(args.capture_npz)
        result["capture_outputs"] = {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in capture_arrays.items()
        }
        result["capture_providers"] = capture_session.get_providers()
    if result["execution"] != "real_npu":
        result["execution_note"] = "VitisAIExecutionProvider did not load; these timings are not NPU measurements."
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
