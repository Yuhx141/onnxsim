#!/usr/bin/env python3
"""Run a QDQ CNN graph on the XDNA layer engine and report timing (and, with --check, exactness vs ONNX Runtime).

    layer_engine_design.py --dev npu2 --net onnx:MODEL.onnx --slot 4096 --l2 2 --xclbin-path X --insts-path I
    run_graph_engine.py MODEL.onnx X I [--iters 50] [--check] [--json report.json] [--dump-boundaries out.npz]

Host jobs (maps too large for a core's region) run first with the numpy reference, then one engine launch; the
engine tensors left for the host tail are the "boundaries". ``--check`` needs onnxruntime and compares every
boundary byte-for-byte with ORT running the same QDQ model.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import onnx

import layer_engine as le
from layer_engine_graph import compile_graph
from layer_engine_host import HostRunner, run_levels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("xclbin", type=Path)
    parser.add_argument("insts", type=Path)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="compare the boundary tensors with ONNX Runtime")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--dump-boundaries", type=Path)
    parser.add_argument("--dump-outputs", type=Path, help="save the model outputs computed through the host tail (npz)")
    args = parser.parse_args()

    import aie.iron as iron
    from aie.utils import NPUKernel

    model = onnx.load(str(args.model))
    plan = compile_graph(model)
    model = plan.model or model  # the onnxsim-folded model the jobs came from
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    x = np.random.default_rng(args.seed).random(shape, dtype=np.float32)
    q = np.clip(np.rint(x / plan.input_scale) + 128, 0, 255).astype(np.uint8)[0]
    channels, height, width = q.shape
    image = np.full((height * width, plan.input_layout.nb * 8), 128, dtype=np.uint8)
    image[:, :channels] = q.transpose(1, 2, 0).reshape(height * width, channels)
    slots = max(le.arena_slots(plan.jobs), 1 + max([j.out_slot for j in plan.host_jobs] + [0]))
    packs = [le.pack_job(job, le.ENGINE_SLOT_BYTES) for job in plan.jobs]
    params = np.concatenate([np.concatenate([p[col].reshape(-1) for p in packs]) for col in range(le.COLS)])
    kernel = NPUKernel(str(args.xclbin), str(args.insts))
    params_t = iron.tensor(params, dtype=np.uint8, device="npu")
    arena = iron.tensor(np.zeros(slots * le.SLOT_BYTES, dtype=np.int8), dtype=np.int8, device="npu")
    if any(j.in_slot == 0 or j.res_slot == 0 for j in plan.jobs):
        with arena.overwrite() as host:
            host.view(np.uint8)[: le.SLOT_BYTES] = le.to_arena(image, plan.input_layout)

    read_ms: list[float] = []
    host_runner = HostRunner(plan)

    def run_once() -> tuple[dict[str, np.ndarray], float, float]:
        started = time.perf_counter()
        dense = {0: image}
        for job in plan.host_jobs:
            resid = dense.get(job.res_slot) if job.res_slot is not None else None
            dense[job.out_slot] = le.reference(job, dense[job.in_slot], resid)
            with arena.overwrite() as host:
                host.view(np.uint8)[job.out_slot * le.SLOT_BYTES : (job.out_slot + 1) * le.SLOT_BYTES] = le.to_arena(
                    dense[job.out_slot], job.out_layout
                )
        prefix_ms = (time.perf_counter() - started) * 1000.0
        engine_ms = 0.0

        def launch() -> dict[str, np.ndarray]:
            nonlocal engine_ms
            started_launch = time.perf_counter()
            kernel(arena, params_t, arena)
            engine_ms += (time.perf_counter() - started_launch) * 1000.0
            started_read = time.perf_counter()
            data = arena.numpy().view(np.uint8)
            read_ms.append((time.perf_counter() - started_read) * 1000.0)
            return {
                name: le.from_arena(data[t.slot * le.SLOT_BYTES : (t.slot + 1) * le.SLOT_BYTES], t.layout)
                for name, t in plan.boundaries.items()
            }

        def write_slot(slot: int, data: np.ndarray) -> None:
            with arena.overwrite() as host:
                host.view(np.uint8)[slot * le.SLOT_BYTES : (slot + 1) * le.SLOT_BYTES] = data

        found, floats = run_levels(plan, launch, write_slot, host_runner)
        run_once.floats = floats
        return found, prefix_ms, engine_ms

    for _ in range(args.warmup):
        run_once()
    totals, prefix, engine = [], [], []
    for _ in range(args.iters):
        started = time.perf_counter()
        boundaries, p_ms, e_ms = run_once()
        totals.append((time.perf_counter() - started) * 1000.0)
        prefix.append(p_ms)
        engine.append(e_ms)
    report = {
        "model": str(args.model), "engine_jobs": len(plan.jobs), "host_jobs": len(plan.host_jobs),
        "boundaries": {n: list(v.shape) for n, v in boundaries.items()},
        "launches": plan.levels, "min_ms": min(totals), "median_ms": float(np.median(totals)),
        "host_prefix_ms": float(np.median(prefix)), "engine_call_ms": float(np.median(engine)), "arena_readback_ms": float(np.median(read_ms)),
    }
    if args.dump_boundaries:
        np.savez(args.dump_boundaries, **{k.replace("/", "|"): v for k, v in boundaries.items()})
    if args.dump_outputs:
        floats = getattr(run_once, "floats", {})
        np.savez(args.dump_outputs, **{o.name.replace("/", "|"): floats[o.name] for o in model.graph.output if o.name in floats})
    if args.check:
        import onnxruntime as ort

        probe = onnx.ModelProto()
        probe.CopyFrom(model)
        for name in boundaries:
            probe.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.UINT8, None))
        session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = session.run(list(boundaries), {model.graph.input[0].name: x})
        differing = 0
        for (name, got), want in zip(boundaries.items(), expected):
            want = want[0].transpose(1, 2, 0).reshape(-1, want.shape[1])
            differing += int((got != want).sum())
        report["bytes_differing_from_ort"] = differing
        outputs = [o.name for o in model.graph.output]
        floats = getattr(run_once, "floats", {})
        if all(o in floats for o in outputs):
            reference = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(outputs, {model.graph.input[0].name: x})
            report["max_abs_output_error_vs_ort"] = max(float(np.abs(floats[o] - r).max()) for o, r in zip(outputs, reference))
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
