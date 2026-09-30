#!/usr/bin/env python3
"""Build a power-of-two QDQ (XINT8-style) torchvision ResNet ONNX model with random weights.

Used to compare the XDNA code generator with the Vitis AI EP on other ResNet depths: the graph has the
same QDQ pattern as the Ryzen AI quicktest model (uint8 zero point 128 activations, int8 per-tensor
power-of-two weights, exact int32 biases, Q/DQ after every Relu / MaxPool / residual-free Conv), so both
runtimes accept it and the requantization is exact shifts.

    python quantize_pow2_resnet.py resnet101 out.onnx [--size 32] [--seed 0]
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from onnx import TensorProto, helper, numpy_helper


def _pow2_up(value: float) -> float:
    return 2.0 ** math.ceil(math.log2(max(value, 1e-12)))


def export_fp32(name: str, size: int, seed: int, path: Path) -> None:
    torch.manual_seed(seed)
    model = getattr(torchvision.models, name)(weights=None).eval()
    with torch.no_grad():  # non-trivial BN statistics so the folded convs are not identity-scaled
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.normal_(0, 0.1)
                module.running_var.uniform_(0.5, 1.5)
                module.weight.uniform_(0.8, 1.2)
                module.bias.normal_(0, 0.05)
    torch.onnx.export(model, torch.rand(1, 3, size, size), str(path), opset_version=13, dynamo=False, input_names=["input"], output_names=["output"])


def quantize(fp32_path: Path, out_path: Path, size: int, seed: int, samples: int = 8) -> None:
    model = onnx.load(str(fp32_path))
    graph = model.graph
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    # Calibration: absmax of every activation tensor over random inputs.
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    exposed = sorted({o for n in graph.node for o in n.output if n.op_type in ("Conv", "Relu", "Clip", "MaxPool", "AveragePool", "Concat", "Add", "GlobalAveragePool", "Flatten", "Gemm")})
    for name in exposed:
        probe.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    absmax: dict[str, float] = {"input": 1.0}
    for _ in range(samples):
        outputs = session.run(exposed, {"input": rng.random((1, 3, size, size), dtype=np.float32)})
        for name, value in zip(exposed, outputs):
            absmax[name] = max(absmax.get(name, 0.0), float(np.abs(value).max()))

    new_nodes, new_init = [], []
    scale_of: dict[str, float] = {}
    dq_of: dict[str, str] = {}

    def const(name, array):
        new_init.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def qdq(tensor: str, scale: float, signed_span: bool) -> str:
        """Q + DQ after ``tensor`` (uint8, zero point 128); returns the DQ output name."""
        scale_of[tensor] = scale
        s = const(f"{tensor}_scale", np.float32(scale))
        z = const(f"{tensor}_zero_point", np.uint8(128))
        new_nodes.append(helper.make_node("QuantizeLinear", [tensor, s, z], [f"{tensor}_q"], name=f"{tensor}_Q"))
        new_nodes.append(helper.make_node("DequantizeLinear", [f"{tensor}_q", s, z], [f"{tensor}_dq"], name=f"{tensor}_DQ"))
        dq_of[tensor] = f"{tensor}_dq"
        return f"{tensor}_dq"

    # The activation scale of a tensor: 127 uint8 levels on each side of the zero point.
    def act_scale(name: str) -> float:
        return _pow2_up(absmax[name] / 127.0)

    graph_input = graph.input[0].name
    qdq(graph_input, 2.0**-7, True)
    for node in graph.node:
        if node.op_type == "Conv":
            x = node.input[0]
            w = init[node.input[1]]
            b = init[node.input[2]] if len(node.input) > 2 else np.zeros(w.shape[0], dtype=np.float32)
            w_scale = _pow2_up(float(np.abs(w).max()) / 127.0)
            wq = np.clip(np.rint(w / w_scale), -128, 127).astype(np.int8)
            bias_scale = max(_pow2_up(float(np.abs(b).max()) / 127.0), scale_of[x] * w_scale)  # int8 bias, power-of-two scale
            bq = np.clip(np.rint(b / bias_scale), -128, 127).astype(np.int8)
            wn = const(f"{node.name}_w_q", wq)
            ws = const(f"{node.name}_w_scale", np.float32(w_scale))
            wz = const(f"{node.name}_w_zp", np.int8(0))
            bn = const(f"{node.name}_b_q", bq)
            bs = const(f"{node.name}_b_scale", np.float32(bias_scale))
            bz = const(f"{node.name}_b_zp", np.int8(0))
            new_nodes.append(helper.make_node("DequantizeLinear", [wn, ws, wz], [f"{node.name}_w_dq"], name=f"{node.name}_wDQ"))
            new_nodes.append(helper.make_node("DequantizeLinear", [bn, bs, bz], [f"{node.name}_b_dq"], name=f"{node.name}_bDQ"))
            conv = helper.make_node("Conv", [dq_of[x], f"{node.name}_w_dq", f"{node.name}_b_dq"], list(node.output), name=node.name)
            conv.attribute.extend(node.attribute)
            new_nodes.append(conv)
            out = node.output[0]
            following = consumers.get(out, [])
            if not (len(following) == 1 and following[0].op_type in ("Relu", "Clip")):
                qdq(out, act_scale(out), True)  # conv3 / downsample: Q/DQ straight after the Conv
        elif node.op_type in ("Relu", "Clip"):
            new_nodes.append(node)  # Clip = ReLU6: min/max stay float constants
            qdq(node.output[0], act_scale(node.output[0]), False)
        elif node.op_type == "Concat":
            cat = helper.make_node("Concat", [dq_of[i] for i in node.input], list(node.output), name=node.name)
            cat.attribute.extend(node.attribute)
            new_nodes.append(cat)
            qdq(node.output[0], act_scale(node.output[0]), True)
        elif node.op_type == "AveragePool":
            pooled = helper.make_node("AveragePool", [dq_of[node.input[0]]], list(node.output), name=node.name)
            pooled.attribute.extend(node.attribute)
            new_nodes.append(pooled)
            qdq(node.output[0], act_scale(node.output[0]), False)
        elif node.op_type == "MaxPool":
            pooled = helper.make_node("MaxPool", [dq_of[node.input[0]]], list(node.output), name=node.name)
            pooled.attribute.extend(node.attribute)
            new_nodes.append(pooled)
            qdq(node.output[0], scale_of[node.input[0]], False)  # same scale/zero point as its input
        elif node.op_type == "Add":
            new_nodes.append(helper.make_node("Add", [dq_of[i] for i in node.input], list(node.output), name=node.name))
            following = consumers.get(node.output[0], [])
            if not (len(following) == 1 and following[0].op_type in ("Relu", "Clip")):
                qdq(node.output[0], act_scale(node.output[0]), True)  # linear residual (MobileNet-style)
        elif node.op_type == "Constant":
            new_nodes.append(node)
        elif node.op_type == "Identity":
            dq_of[node.output[0]] = dq_of[node.input[0]]
            scale_of[node.output[0]] = scale_of[node.input[0]]
        elif node.op_type in ("GlobalAveragePool", "Flatten"):
            copy = helper.make_node(node.op_type, [dq_of[node.input[0]]], list(node.output), name=node.name)
            copy.attribute.extend(node.attribute)
            new_nodes.append(copy)
            qdq(node.output[0], act_scale(node.output[0]), False)
        elif node.op_type == "Gemm":
            x = node.input[0]
            trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
            w = init[node.input[1]]
            w = w if trans_b else w.T  # [out][in]
            b = init[node.input[2]]
            w_scale = _pow2_up(float(np.abs(w).max()) / 127.0)
            wq = np.clip(np.rint(w / w_scale), -128, 127).astype(np.int8)
            bias_scale = max(_pow2_up(float(np.abs(b).max()) / 127.0), scale_of[x] * w_scale)  # int8 bias, power-of-two scale
            bq = np.clip(np.rint(b / bias_scale), -128, 127).astype(np.int8)
            wn = const(f"{node.name}_w_q", wq)
            ws = const(f"{node.name}_w_scale", np.float32(w_scale))
            wz = const(f"{node.name}_w_zp", np.int8(0))
            bn = const(f"{node.name}_b_q", bq)
            bs = const(f"{node.name}_b_scale", np.float32(bias_scale))
            bz = const(f"{node.name}_b_zp", np.int8(0))
            new_nodes.append(helper.make_node("DequantizeLinear", [wn, ws, wz], [f"{node.name}_w_dq"], name=f"{node.name}_wDQ"))
            new_nodes.append(helper.make_node("DequantizeLinear", [bn, bs, bz], [f"{node.name}_b_dq"], name=f"{node.name}_bDQ"))
            pre = f"{node.output[0]}_pre"
            new_nodes.append(helper.make_node("Gemm", [dq_of[x], f"{node.name}_w_dq", f"{node.name}_b_dq"], [pre], name=node.name, transB=1))
            # final Q/DQ: the graph output keeps its original name
            scale = _pow2_up(absmax[node.output[0]] / 127.0)
            s_name = const(f"{pre}_scale", np.float32(scale))
            z_name = const(f"{pre}_zero_point", np.uint8(128))
            new_nodes.append(helper.make_node("QuantizeLinear", [pre, s_name, z_name], [f"{pre}_q"], name=f"{pre}_Q"))
            new_nodes.append(helper.make_node("DequantizeLinear", [f"{pre}_q", s_name, z_name], list(node.output), name=f"{pre}_DQ"))
        else:
            raise ValueError(f"unsupported op {node.op_type}")
    out = onnx.ModelProto()
    out.CopyFrom(model)
    del out.graph.node[:]
    del out.graph.initializer[:]
    out.graph.node.extend(new_nodes)
    consumed = {i for n in graph.node if n.op_type in ("Conv", "Gemm") for i in n.input[1:]}
    out.graph.initializer.extend(new_init + [i for i in graph.initializer if i.name not in consumed])
    out = onnx.shape_inference.infer_shapes(out)  # the XDNA planner needs static shapes on every edge
    onnx.checker.check_model(out)
    onnx.save(out, str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="torchvision.models name, e.g. resnet101")
    parser.add_argument("out", type=Path)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "fp32.onnx"
        export_fp32(args.model, args.size, args.seed, fp32)
        quantize(fp32, args.out, args.size, args.seed)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
