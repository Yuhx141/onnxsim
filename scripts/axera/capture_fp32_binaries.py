#!/usr/bin/env python3
"""Capture and device-profile FP32 binary templates for step fallbacks.

The build list is derived from nodes left on the host by ``step_runner``'s
current default plan. Each unique (op, input shapes, output shape) gets one
Pulsar2 FP32 layer-config build; each capture is checked for FP32 IO and run
against NumPy before its latency is recorded.

    AXCL_LXD_VM=axcl-vm python capture_fp32_binaries.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Mapping

import numpy as np
import onnx
from onnx import TensorProto, helper

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "fixtures", "fp32_binary")
INDEX = os.path.join(FIXTURE_DIR, "index.json")
sys.path.insert(0, HERE)

import axcl_session  # noqa: E402
import step_runner  # noqa: E402


def _shape_map(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(model)
    values = (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    shapes = {}
    for value in values:
        dims = value.type.tensor_type.shape.dim
        if all(d.dim_value > 0 for d in dims):
            shapes[value.name] = tuple(int(d.dim_value) for d in dims)
    for value in inferred.graph.initializer:
        shapes[value.name] = tuple(int(d) for d in value.dims)
    return shapes


def required_signatures() -> list[dict]:
    model = step_runner.load_step()
    records = step_runner.load_records()
    calib = step_runner.axb.load_calibration(step_runner.STEP_CALIB)
    overrides = step_runner.load_step_precision_overrides(model, records, calib)
    _, host = step_runner.build_plan(model, records, calib, precision_overrides=overrides)
    shapes = _shape_map(model)
    records_by_name = {record["name"]: record for record in records}
    signatures: dict[str, dict] = {}
    for node in model.graph.node:
        if node.name not in host or node.op_type not in ("Add", "Sub", "Mul", "Div"):
            continue
        record = records_by_name.get(node.name)
        if not record or len(record.get("inputs", ())) != 2 or len(node.output) != 1:
            continue
        names = list(record["inputs"])
        input_shapes = [shapes.get(name) for name in names]
        output_shape = shapes.get(node.output[0])
        if any(shape is None for shape in input_shapes) or output_shape is None:
            raise ValueError(f"dynamic or unknown FP32 capture shape at {node.name}")
        item = {
            "op": node.op_type,
            "input_shapes": [list(shape) for shape in input_shapes],
            "output_shape": list(output_shape),
            "source_nodes": [],
        }
        key = signature_key(item)
        signatures.setdefault(key, item)["source_nodes"].append(node.name)
    return sorted(signatures.values(), key=signature_key)


def signature_key(item: Mapping) -> str:
    return json.dumps(
        [item["op"], item["input_shapes"], item["output_shape"]],
        separators=(",", ":"),
    )


def filename(item: Mapping) -> str:
    digest = hashlib.sha256(signature_key(item).encode()).hexdigest()[:12]
    dims = "x".join(str(x) for x in item["output_shape"])
    return f"{item['op'].lower()}_{dims}_{digest}.axmodel.gz"


def _write_model(item: Mapping, work: str) -> None:
    os.makedirs(os.path.join(work, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(work, "config"), exist_ok=True)
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in zip(("x", "z"), item["input_shapes"])
    ]
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, item["output_shape"])
    graph = helper.make_graph(
        [helper.make_node(item["op"], ["x", "z"], ["y"], name="binary")],
        "fp32_binary_capture",
        inputs,
        [output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, os.path.join(work, "binary.onnx"))

    input_configs = []
    for name, shape in zip(("x", "z"), item["input_shapes"]):
        sample = os.path.join(work, f"{name}.npy")
        # FP32 overrides do not depend on calibration statistics. A one-sample
        # zero tensor avoids allocating a second copy of the largest state
        # tensors while still giving Pulsar2 a valid Numpy calibration archive.
        np.save(sample, np.zeros(shape, dtype=np.float32))
        tar_path = os.path.join(work, "dataset", f"{name}.tar")
        with tarfile.open(tar_path, "w") as archive:
            archive.add(sample, arcname="0.npy")
        os.unlink(sample)
        input_configs.append(
            {
                "tensor_name": name,
                "calibration_dataset": f"./dataset/{name}.tar",
                "calibration_format": "Numpy",
                "calibration_size": 1,
            }
        )
    config = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": input_configs,
            "calibration_method": "MinMax",
            "precision_analysis": False,
            "layer_configs": [{"op_types": [item["op"]], "data_type": "FP32"}],
        },
        "compiler": {"check": 0},
    }
    with open(os.path.join(work, "config", "step.json"), "w", encoding="utf-8") as stream:
        json.dump(config, stream)


def _inputs(item: Mapping, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = [rng.normal(size=shape).astype(np.float32) for shape in item["input_shapes"]]
    if item["op"] == "Div":
        xs[1] = np.abs(xs[1]) + np.float32(0.5)
    return xs


def _expected(op: str, xs: list[np.ndarray]) -> np.ndarray:
    return {
        "Add": np.add,
        "Sub": np.subtract,
        "Mul": np.multiply,
        "Div": np.divide,
    }[op](xs[0], xs[1])


def capture_and_profile(item: dict, destination: str, runs: int) -> dict:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    work = tempfile.mkdtemp(prefix="axera-fp32-capture-")
    try:
        _write_model(item, work)
        name = f"fp32-capture-{os.getpid()}-{time.time_ns()}"
        command = [
            "docker", "run", "--rm", "--name", name,
            "-v", f"{work}:/data", "pulsar2:7.0-lite", "pulsar2", "build",
            "--target_hardware", "AX650", "--input", "binary.onnx",
            "--output_dir", "out", "--config", "config/step.json",
        ]
        try:
            built = subprocess.run(command, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
            raise RuntimeError("Pulsar2 build timed out after 900 seconds") from None
        if built.returncode:
            raise RuntimeError((built.stdout + built.stderr)[-3000:])
        model_path = os.path.join(work, "out", "compiled.axmodel")
        with open(model_path, "rb") as stream:
            blob = stream.read()
        with axcl_session.AXSession(subdir="fp32_binary_capture") as session:
            model = session.load(blob)
            try:
                if len(model.inputs) != 2 or len(model.outputs) != 1:
                    raise ValueError("capture IO is not binary")
                if any(spec.dtype != np.float32 for spec in (*model.inputs, *model.outputs)):
                    raise ValueError("Pulsar2 did not preserve FP32 IO")
                if [list(s.shape) for s in model.inputs] != item["input_shapes"]:
                    raise ValueError(f"captured input shapes changed: {model.inputs}")
                if list(model.outputs[0].shape) != item["output_shape"]:
                    raise ValueError(f"captured output shape changed: {model.outputs}")
                xs = _inputs(item, 1701)
                (actual,) = session.run(model, xs)
                expected = _expected(item["op"], xs)
                error = float(np.max(np.abs(actual - expected)))
                if not np.array_equal(actual, expected):
                    raise ValueError(f"device output differs from FP32 {item['op']}: max error {error}")

                elements = int(np.prod(item["output_shape"], dtype=np.int64))
                profile_runs = min(runs, max(5, 4_000_000 // max(elements, 1)))
                warmups = min(3, profile_runs)
                for _ in range(warmups):
                    session.run(model, xs)
                device_us, host_ms = [], []
                for _ in range(profile_runs):
                    before = session.exec_us
                    start = time.perf_counter()
                    session.run(model, xs)
                    host_ms.append((time.perf_counter() - start) * 1000)
                    device_us.append(session.exec_us - before)
            finally:
                session.unload(model)
        with gzip.open(destination, "wb", compresslevel=6) as stream:
            stream.write(blob)
        import statistics

        result = {
            **item,
            "file": os.path.basename(destination),
            "validated": True,
            "max_abs_error": error,
            "warmup_runs": warmups,
            "profile_runs": profile_runs,
            "device_exec_us": {
                "mean": statistics.fmean(device_us),
                "median": statistics.median(device_us),
                "p95": float(np.percentile(device_us, 95)),
            },
            "host_roundtrip_ms": {
                "mean": statistics.fmean(host_ms),
                "median": statistics.median(host_ms),
                "p95": float(np.percentile(host_ms, 95)),
            },
        }
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30, help="device profile repetitions per template")
    parser.add_argument("--limit", type=int, default=0, help="capture only the first N signatures")
    parser.add_argument(
        "--refresh", action="store_true",
        help="recapture and reprofile every signature already in the index",
    )
    parser.add_argument("--output", default=INDEX)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be positive")
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    items = required_signatures()
    if args.limit:
        items = items[: args.limit]
    requested = len(items)
    try:
        with open(args.output, encoding="utf-8") as stream:
            index = json.load(stream)
    except FileNotFoundError:
        index = {"schema_version": 1, "templates": []}
    entries = {signature_key(entry): entry for entry in index.get("templates", [])}
    if args.refresh:
        for entry in entries.values():
            items_by_key = {signature_key(item): item for item in items}
            items_by_key.setdefault(signature_key(entry), dict(entry))
            items = list(items_by_key.values())
        items.sort(key=signature_key)
        if args.limit:
            items = items[: args.limit]
        requested = len(items)
    failures = 0
    for position, item in enumerate(items, 1):
        key = signature_key(item)
        destination = os.path.join(FIXTURE_DIR, filename(item))
        print(f"[{position}/{len(items)}] {item['op']} {item['input_shapes']} -> {item['output_shape']}", flush=True)
        try:
            entry = capture_and_profile(item, destination, args.runs)
        except Exception as exc:
            failures += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        entries[key] = entry
        index = {"schema_version": 1, "templates": sorted(entries.values(), key=signature_key)}
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(index, stream, indent=2)
            stream.write("\n")
        print(f"  device mean={entry['device_exec_us']['mean']:.1f} us, exact FP32", flush=True)
    done = sum(bool(entries.get(signature_key(item), {}).get("validated")) for item in items)
    print(
        f"validated this run: {done}/{requested}; failures: {failures}; "
        f"validated templates in index: {sum(bool(x.get('validated')) for x in entries.values())}"
    )
    return 0 if failures == 0 and done == requested else 1


if __name__ == "__main__":
    raise SystemExit(main())
