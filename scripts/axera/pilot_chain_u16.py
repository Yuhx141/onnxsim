#!/usr/bin/env python3
"""Real-data accuracy of a step MatMul chain at U8 vs U16.

Extracts the chain a manifest template serves (in the legalized step), takes its
real input tensors from a float run of the step's reference batch, builds it
with Pulsar2 at each layer precision on that same data, and compares the
device output with the float chain.  Usage: pilot_chain_u16.py WORKDIR TEMPLATE
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import utils

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pulsar2_docker as pd  # noqa: E402
import step_calibration as sc  # noqa: E402
import step_runner as sr  # noqa: E402


def chain(model, names):
    nodes = {n.name: n for n in model.graph.node}
    inits = {i.name for i in model.graph.initializer}
    lower = {k.lower(): v for k, v in nodes.items()}

    def find(n):
        return nodes.get(n) or lower[n.removeprefix("distill__").lower()]

    members = [find(n) for n in names if n in nodes or n.removeprefix("distill__").lower() in lower]
    produced = {o for n in members for o in n.output}
    ins = []
    for n in members:
        for t in n.input:
            if t and t not in produced and t not in inits and t not in ins:
                ins.append(t)
    out = members[-1].output[0]
    return ins, out


def build(work, tag, sub, data, precision, image):
    root = os.path.join(work, tag)
    os.makedirs(os.path.join(root, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(root, "config"), exist_ok=True)
    onnx.save(sub, os.path.join(root, "t.onnx"))
    inputs = []
    for name, arr in data.items():
        pd.make_numpy_calibration_tar(
            os.path.join(root, f"dataset/{name}.tar"), [arr] * 4
        )
        inputs.append(
            {
                "tensor_name": name,
                "calibration_dataset": f"./dataset/{name}.tar",
                "calibration_format": "Numpy",
                "calibration_size": 4,
            }
        )
    quant = {"input_configs": inputs, "calibration_method": "MinMax",
             "precision_analysis": False}
    if precision != "U8":
        quant["layer_configs"] = [{"op_types": ["MatMul"], "data_type": precision}]
    with open(os.path.join(root, "config/c.json"), "w") as f:
        json.dump({"model_type": "ONNX", "npu_mode": "NPU1", "quant": quant,
                   "compiler": {"check": 0}}, f)
    return pd.build(root, "t.onnx", "out", config_path="config/c.json",
                    image=image, timeout=1800)


def main():
    work, template = sys.argv[1], sys.argv[2]
    manifest = json.load(open(os.path.join(HERE, "fixtures/matmul_step_templates/manifest.json")))
    names = [manifest["templates"][template]["step_node"]]
    model = sr.load_step()
    ins, out = chain(model, names)
    print("chain inputs", ins, "output", out, flush=True)
    feeds = sr.load_reference()["feeds"]
    outs, _ = sr.StepRunner(model, []).run(feeds, "float", keep=ins + [out])
    data = {t: np.asarray(outs[t], np.float32) for t in ins}
    ref = np.asarray(outs[out], np.float32)
    sub = onnx.shape_inference.infer_shapes(
        utils.Extractor(model).extract_model(ins, [out]))
    del sub.opset_import[:]
    sub.opset_import.extend([onnx.helper.make_opsetid("", 13)])
    sub.ir_version = 8
    print({t: (a.shape, float(a.min()), float(a.max())) for t, a in data.items()}, flush=True)
    image = os.environ.get("PULSAR2_IMAGE", "pulsar2:7.0-lite")
    for prec in sys.argv[3].split(",") if len(sys.argv) > 3 else ["U8", "U16"]:
        res = build(work, f"{template}_{prec}", sub, data, prec, image)
        if not res.success:
            print(prec, "build failed:", (res.error or "")[-800:]); continue
        dev = pd.run_on_device_with_inputs(
            res.axmodel_path, {t: a.tobytes() for t, a in data.items()}, repeat=20, warmup=3)
        if not dev.outputs:
            print(prec, "device:", dev.error); continue
        y = np.frombuffer(dev.outputs[0], np.float32).reshape(ref.shape)
        rel = np.linalg.norm(y - ref) / np.linalg.norm(ref)
        cos = float((y * ref).sum() / (np.linalg.norm(y) * np.linalg.norm(ref)))
        print(prec, f"rel err {rel:.3e} cos {cos:.6f} avg_ms {dev.avg_ms}", flush=True)


if __name__ == "__main__":
    main()
