"""Build a step segment's subgraph with Pulsar2 at 16-bit MatMul precision.

The step's 8-bit MatMul/Conv/Gemm chains are 2-7% off float on real data
(``docs/axera-step-runner.md``); the same chain built with
``quant.layer_configs`` U16 is about 240x closer, at about 2.5x the device
time. The chain is built on real tensors of the reference batch, so the
result is one axmodel per segment rather than a recalibrated template.
Segment inputs and outputs stay float32, like every other template.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import onnx
from onnx import utils

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pulsar2_docker as pd  # noqa: E402

OPS_16BIT = ("MatMul", "Conv", "Gemm")
PASSIVE_OPS = frozenset({"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity", "Gather", "Slice", "Flatten"})
DEFAULT_IMAGE = "pulsar2:7.0-lite"


def chain_model(
    model: onnx.ModelProto, inputs: Sequence[str], outputs: Sequence[str]
) -> tuple[onnx.ModelProto, Callable[[np.ndarray], np.ndarray] | None]:
    """The subgraph between ``inputs`` and ``outputs``, legalized the way the
    step is compiled (live-weight Conv/Gemm as MatMul) and in the opset the
    Pulsar2 frontend takes.

    A Transpose/Reshape that ends the chain makes Pulsar2 quantize the whole
    output path to 8 bits even under a 16-bit layer config (the bias Add and
    the output stay uint8). Those trailing ops are cut off and returned as a
    host function of the axmodel's output."""
    import step_calibration

    if len(outputs) != 1:
        raise ValueError("16-bit chains have one output")
    sub = utils.Extractor(model).extract_model(list(inputs), list(outputs))
    sub = onnx.shape_inference.infer_shapes(sub)
    sub = step_calibration.legalized(sub)
    del sub.opset_import[:]
    sub.opset_import.extend([onnx.helper.make_opsetid("", 13)])
    sub.ir_version = 8
    g = sub.graph
    final_shape = tuple(
        d.dim_value for d in g.output[0].type.tensor_type.shape.dim
    )
    steps: list[tuple[str, object]] = []
    by_out = {o: n for n in g.node for o in n.output}
    tail = g.output[0].name
    while tail in by_out and by_out[tail].op_type in ("Transpose", "Reshape"):
        node = by_out[tail]
        if node.op_type == "Transpose":
            perm = [a.ints for a in node.attribute if a.name == "perm"]
            steps.append(("T", tuple(perm[0]) if perm else None))
        else:
            steps.append(("R", None))
        g.node.remove(node)
        tail = node.input[0]
    if not steps:
        return sub, None
    sub = onnx.shape_inference.infer_shapes(sub)
    g = sub.graph
    vi = next(v for v in g.value_info if v.name == tail)
    del g.output[:]
    g.output.append(vi)
    order = list(reversed(steps))

    def post(y: np.ndarray) -> np.ndarray:
        for kind, perm in order:
            if kind == "T":
                y = np.transpose(y, perm)
        return np.ascontiguousarray(y).reshape(final_shape)

    return sub, post


def build_chain(
    work: str,
    tag: str,
    sub: onnx.ModelProto,
    data: Mapping[str, np.ndarray],
    precision: str = "U16",
    image: str = DEFAULT_IMAGE,
    timeout: int = 1800,
):
    """``pulsar2 build`` of ``sub`` calibrated on ``data`` (four identical
    samples, so MinMax sees exactly the real range)."""
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
    quant: dict = {
        "input_configs": inputs,
        "calibration_method": "MinMax",
        "precision_analysis": False,
    }
    if precision != "U8":
        # every op of the chain: a Concat/Add/Relu left at 8 bits after the
        # 16-bit MatMuls puts the 8-bit error back (legalized Convs)
        ops = sorted({n.op_type for n in sub.graph.node} - PASSIVE_OPS)
        quant["layer_configs"] = [{"op_types": ops, "data_type": precision}]
        # an op type entry is not applied to a bias Add that follows the
        # MatMul (its output stayed 8-bit); select those by layer name
        adds = [n.name for n in sub.graph.node if n.op_type == "Add"]
        if adds:
            quant["layer_configs"].append(
                {"layer_names": adds, "data_type": precision}
            )
    with open(os.path.join(root, "config/c.json"), "w") as f:
        json.dump(
            {
                "model_type": "ONNX",
                "npu_mode": "NPU1",
                "quant": quant,
                "compiler": {"check": 0},
            },
            f,
        )
    return pd.build(
        root, "t.onnx", "out", config_path="config/c.json", image=image, timeout=timeout
    )


def cached_chain_axmodel(
    cache_dir: str,
    work: str,
    name: str,
    sub: onnx.ModelProto,
    data: Mapping[str, np.ndarray],
    precision: str = "U16",
    image: str = DEFAULT_IMAGE,
) -> bytes:
    """The compiled axmodel bytes, from ``cache_dir`` when the same graph,
    calibration data and precision were built before."""
    h = hashlib.sha256(sub.SerializeToString())
    h.update(precision.encode())
    for k in sorted(data):
        h.update(k.encode())
        h.update(np.ascontiguousarray(data[k], np.float32).tobytes())
    path = os.path.join(cache_dir, f"{name}.{h.hexdigest()[:16]}.axmodel")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    res = build_chain(work, name, sub, data, precision, image)
    if not res.success:
        raise RuntimeError(f"{name}: pulsar2 build failed: {(res.error or '')[-500:]}")
    with open(res.axmodel_path, "rb") as f:
        blob = f.read()
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)
    return blob
