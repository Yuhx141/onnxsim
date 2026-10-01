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
import shutil
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
    model: onnx.ModelProto,
    inputs: Sequence[str],
    outputs: Sequence[str],
    split: int = 1,
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
    if split > 1:
        # a chain too large to compile at the step's batch: every input whose
        # leading dimension is the batch (the first input's, or the largest)
        # shrinks by ``split``; the runner calls the model ``split`` times
        batch = split_batch(sub)
        for vi in sub.graph.input:
            dims = vi.type.tensor_type.shape.dim
            if dims and dims[0].dim_value == batch:
                dims[0].dim_value = batch // split
        del sub.graph.value_info[:]
        del sub.graph.output[:]
        sub.graph.output.extend(
            onnx.helper.make_tensor_value_info(o, onnx.TensorProto.FLOAT, None)
            for o in outputs
        )
    sub = onnx.shape_inference.infer_shapes(sub)
    sub = step_calibration.legalized(sub)
    del sub.opset_import[:]
    sub.opset_import.extend([onnx.helper.make_opsetid("", 13)])
    sub.ir_version = 8
    g = sub.graph
    final_shape = tuple(
        d.dim_value for d in g.output[0].type.tensor_type.shape.dim
    )
    if split > 1:
        # the runner concatenates the chunks' outputs along axis 0 first
        final_shape = (final_shape[0] * split, *final_shape[1:])
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

    # the same ops as device models: the chain's output shape, the stripped
    # steps in order, and the final shape (see ``transpose_models``)
    post.pre_shape = tuple(d.dim_value for d in vi.type.tensor_type.shape.dim)  # type: ignore[attr-defined]
    post.steps = order  # type: ignore[attr-defined]
    post.final_shape = final_shape  # type: ignore[attr-defined]
    return sub, post


STEP_BATCH = 16
"""The training step's batch size (a fixed-batch compiled graph)."""


def split_batch(sub: onnx.ModelProto) -> int:
    """The batch a chain is split along: the step's batch, taken from the
    inputs whose leading dimension equals it."""
    return STEP_BATCH


def split_flags(sub_inputs: Sequence[onnx.ValueInfoProto], batch: int) -> list[bool]:
    return [
        bool(vi.type.tensor_type.shape.dim)
        and vi.type.tensor_type.shape.dim[0].dim_value == batch
        for vi in sub_inputs
    ]


def build_chain(
    work: str,
    tag: str,
    sub: onnx.ModelProto,
    data: Mapping[str, np.ndarray | list[np.ndarray]],
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
        samples = list(arr) if isinstance(arr, list) else [arr] * 4
        pd.make_numpy_calibration_tar(os.path.join(root, f"dataset/{name}.tar"), samples)
        inputs.append(
            {
                "tensor_name": name,
                "calibration_dataset": f"./dataset/{name}.tar",
                "calibration_format": "Numpy",
                "calibration_size": len(samples),
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
        types = {n.op_type for n in sub.graph.node}
        ops = sorted(types - PASSIVE_OPS) or sorted(types)
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


def chain_cache_path(
    cache_dir: str,
    name: str,
    sub: onnx.ModelProto,
    data: Mapping[str, np.ndarray | list[np.ndarray]],
    precision: str,
) -> str:
    """Where ``cached_chain_axmodel`` keeps the build of this graph,
    calibration data and precision."""
    h = hashlib.sha256(sub.SerializeToString())
    h.update(precision.encode())
    for k in sorted(data):
        h.update(k.encode())
        for a in data[k] if isinstance(data[k], list) else [data[k]]:
            h.update(np.ascontiguousarray(a, np.float32).tobytes())
    return os.path.join(cache_dir, f"{name}.{h.hexdigest()[:16]}.axmodel")


def cached_chain_axmodel(
    cache_dir: str,
    work: str,
    name: str,
    sub: onnx.ModelProto,
    data: Mapping[str, np.ndarray | list[np.ndarray]],
    precision: str = "U16",
    image: str = DEFAULT_IMAGE,
    need_quant: bool = False,
) -> bytes:
    """The compiled axmodel bytes, from ``cache_dir`` when the same graph,
    calibration data and precision were built before. Every build also leaves
    its ``quant_axmodel.json`` beside the axmodel (``<axmodel>.quant.json``);
    with ``need_quant`` a cached axmodel without one is built again."""
    path = chain_cache_path(cache_dir, name, sub, data, precision)
    if os.path.exists(path) and (not need_quant or os.path.exists(path + ".quant.json")):
        with open(path, "rb") as f:
            return f.read()
    failed = path + ".failed"
    if os.path.exists(failed) and not os.environ.get("U16_RETRY_FAILED"):
        with open(failed) as f:
            raise RuntimeError(f"{name}: earlier build failed ({f.read().strip()})")
    res = build_chain(
        work,
        name,
        sub,
        data,
        precision,
        image,
        timeout=int(os.environ.get("U16_BUILD_TIMEOUT", "1800")),
    )
    if not res.success:
        os.makedirs(cache_dir, exist_ok=True)
        with open(failed, "w") as f:
            f.write((res.error or "")[-200:].replace("\n", " "))
        raise RuntimeError(f"{name}: pulsar2 build failed: {(res.error or '')[-500:]}")
    with open(res.axmodel_path, "rb") as f:
        blob = f.read()
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)
    quant = os.path.join(os.path.dirname(res.axmodel_path), "quant", "quant_axmodel.json")
    if os.path.exists(quant):
        shutil.copyfile(quant, path + ".quant.json")
    return blob


def node_model(
    node: onnx.NodeProto, shapes: Mapping[str, Sequence[int]]
) -> onnx.ModelProto:
    """A one-node model whose every input (constants included) is a float32
    graph input, so that one build serves every node with the same operator,
    attributes and input shapes. A rank-0 input becomes ``[1]`` (Pulsar2's
    calibrator rejects rank-0 tensors; the runner reshapes the value)."""
    from onnx import TensorProto, helper

    def shape_of(t: str) -> list[int]:
        return list(shapes[t]) or [1]

    n = onnx.NodeProto()
    n.CopyFrom(node)
    n.name = node.op_type.lower()
    ins = [
        helper.make_tensor_value_info(t, TensorProto.FLOAT, shape_of(t))
        for t in node.input
    ]
    outs = [
        helper.make_tensor_value_info(t, TensorProto.FLOAT, shape_of(t))
        for t in node.output
    ]
    m = helper.make_model(
        helper.make_graph([n], "one", ins, outs),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    m.ir_version = 8
    return m


def signature_data(node: onnx.NodeProto, shapes: Mapping[str, Sequence[int]]):
    """Calibration tensors of a node's shapes. An FP32 layer does not quantize,
    so the values only have to be valid for the operator (ones)."""
    return {
        t: np.ones(list(shapes[t]) or [1], np.float32) for t in node.input if t
    }


def chain_ranges(
    sub: onnx.ModelProto, samples: Mapping[str, list[np.ndarray]]
) -> dict[str, tuple[float, float]]:
    """``{tensor: (min, max)}`` of every float tensor of ``sub`` over the
    calibration ``samples`` (a list per graph input)."""
    import step_calibration

    return step_calibration.collect_ranges(sub, samples)


def predict_scales16(
    quant_json: str | Mapping,
    ranges: Mapping[str, tuple[float, float]],
    margin: float = 1.0,
) -> dict[str, tuple[float, float]]:
    """``{tensor: (scale, zero point)}`` Pulsar2 would assign a 16-bit chain
    whose tensors span ``ranges`` (widened by ``margin``), without building it.

    The rule, checked against native U16 builds of a bare MatMul and a forward
    Conv chain at two calibrations: a tensor that is (or shares a quantization
    with) a live MatMul operand is symmetric int16, ``scale = max|x| / 32767.5``;
    every other tensor is unsigned 16-bit over its range widened to include 0,
    ``scale = f32((hi - lo) / 65535)``, ``zero point = round(-lo / scale)``. Tensors
    that Pulsar2 marks OVERLAPPED share their dominator's quantization, taken
    over the union of the group's ranges. ``quant_json`` is a template build's
    ``quant_axmodel.json`` (path or parsed), which says which tensors are
    symmetric and which are grouped."""
    import json

    if not isinstance(quant_json, Mapping):
        with open(quant_json) as f:
            quant_json = json.load(f)
    info: dict[str, Mapping] = {}
    for per_op in quant_json["tensor_configs"].values():
        for t, v in per_op.items():
            if t not in info or v["state"] != "OVERLAPPED":
                info[t] = v
    group: dict[int, list[float]] = {}
    for t, v in info.items():
        if t in ranges:
            lo, hi = ranges[t]
            lo, hi = lo * margin, hi * margin
            g = group.setdefault(v["dominator"], [lo, hi])
            g[0], g[1] = min(g[0], lo), max(g[1], hi)
    out: dict[str, tuple[float, float]] = {}
    for t, v in info.items():
        if v["dominator"] not in group or v["bit_width"] != 16:
            continue
        lo, hi = group[v["dominator"]]
        if v["quant_min"] < 0:
            out[t] = (max(abs(lo), abs(hi)) / 32767.5, 0.0)
        else:
            lo0, hi0 = min(lo, 0.0), max(hi, 0.0)
            s = float(np.float32((hi0 - lo0) / 65535))
            out[t] = (s, float(round(-lo0 / s)))
    return out


def transpose_models(
    pre_shape: Sequence[int], steps: Sequence[tuple[str, object]]
) -> list[onnx.ModelProto]:
    """One-op Transpose models that apply the stripped transposes of a chain to
    its output on the device (a Reshape moves no data, so it is only a change of
    shape). A Transpose-only model is bit-exact on the NPU (measured at U8, U16
    and FP32), so it adds no error."""
    from onnx import TensorProto, helper

    models = []
    shape = list(pre_shape)
    for kind, perm in steps:
        if kind != "T":
            continue
        perm = list(perm) if perm else list(reversed(range(len(shape))))
        out = [shape[i] for i in perm]
        node = helper.make_node("Transpose", ["x"], ["y"], name="tr", perm=perm)
        m = helper.make_model(
            helper.make_graph(
                [node],
                "t",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, out)],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
        m.ir_version = 8
        models.append(m)
        shape = out
    return models
