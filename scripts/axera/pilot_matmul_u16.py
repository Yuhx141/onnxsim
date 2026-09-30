#!/usr/bin/env python3
"""Pilot: does Pulsar2 build a live-operand MatMul at U16, and is it more exact?

Builds ``y = MatMul(x, w)`` (both operands graph inputs, as in the step's
MatMul chains) once at the default U8 and once with a ``quant.layer_configs``
U16/S16 entry, then compares the quantization bit widths and the device output
against float32.  Usage: pilot_matmul_u16.py WORKDIR [--shape M K N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import parser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pulsar2_docker as pd  # noqa: E402


def build_case(work, tag, m, k, n, precision, xs, ws):
    root = os.path.join(work, tag)
    os.makedirs(os.path.join(root, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(root, "config"), exist_ok=True)
    model = parser.parse_model(
        f"<ir_version: 8, opset_import: [\"\": 13]> g (float[1,{m},{k}] x, "
        f"float[1,{k},{n}] w) => (float[1,{m},{n}] y) {{ y = MatMul(x, w) }}"
    )
    onnx.save(model, os.path.join(root, "m.onnx"))
    inputs = []
    for name, samples in (("x", xs), ("w", ws)):
        pd.make_numpy_calibration_tar(
            os.path.join(root, f"dataset/{name}.tar"), samples
        )
        inputs.append(
            {
                "tensor_name": name,
                "calibration_dataset": f"./dataset/{name}.tar",
                "calibration_format": "Numpy",
                "calibration_size": len(samples),
            }
        )
    quant = {
        "input_configs": inputs,
        "calibration_method": "MinMax",
        "precision_analysis": False,
    }
    if precision != "U8":
        quant["layer_configs"] = [{"op_types": ["MatMul"], "data_type": precision}]
    cfg = {"model_type": "ONNX", "npu_mode": "NPU1", "quant": quant,
           "compiler": {"check": 0}}
    with open(os.path.join(root, "config/c.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    res = pd.build(root, "m.onnx", "out", config_path="config/c.json", image=os.environ.get("PULSAR2_IMAGE", pd.DEFAULT_IMAGE))
    return root, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--shape", nargs=3, type=int, default=[64, 128, 64])
    ap.add_argument("--precisions", default="U8,U16,S16")
    ap.add_argument("--tag", default="", help="suffix for the build directory")
    ap.add_argument("--xscale", type=float, default=1.0)
    ap.add_argument("--wscale", type=float, default=1.0)
    ap.add_argument("--xshift", type=float, default=0.0)
    a = ap.parse_args()
    m, k, n = a.shape
    rng = np.random.RandomState(0)
    xs = [(rng.randn(1, m, k) * a.xscale + a.xshift).astype(np.float32) for _ in range(8)]
    ws = [(rng.randn(1, k, n) * 0.1 * a.wscale).astype(np.float32) for _ in range(8)]
    for prec in a.precisions.split(","):
        root, res = build_case(a.work, prec + a.tag, m, k, n, prec, xs, ws)
        print(prec, "success", res.success, (res.error or res.stdout_tail or "")[-1500:] if not res.success else "")
        if not res.success:
            continue
        q = os.path.join(root, "out/quant/quant_axmodel.json")
        if os.path.exists(q):
            j = json.load(open(q))
            print(prec, "quant keys", list(j)[:6])
            print(prec, "bits:", json.dumps(j.get("mix_precision_configs", j.get("bits", "?")))[:300])
        dev = pd.run_on_device_with_inputs(
            res.axmodel_path,
            {"x": xs[0].tobytes(), "w": ws[0].tobytes()},
        )
        if getattr(dev, "outputs", None):
            out = np.frombuffer(dev.outputs[0], np.float32).reshape(1, m, n)
            ref = xs[0] @ ws[0]
            rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
            print(prec, "device rel err vs float:", float(rel))
        else:
            print(prec, "device:", dev)


if __name__ == "__main__":
    main()
