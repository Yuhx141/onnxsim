"""Basic-block (ResNet-18/34) binding for the layer engine.

A torchvision ``BasicBlock`` is conv3x3(+ReLU) -> conv3x3, then Add with the block input (identity) or a
1x1 stride-2 ``downsample`` Conv, then ReLU. It uses only job kinds the engine already has, so the only
new host work is reading the QDQ parameters. The returned binding has the fields the runner and
``layer_engine_net`` need (a subset of ``bind_fused_bottleneck``'s).
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import numpy as np
from benchmark_fused_bottleneck import _power_of_two_exponent, _power_of_two_shift, _quantizer_after, _single_q_params
from conv_lowering import plan_all_convs
from onnx import numpy_helper
from qdq_runtime import qdq_edge_map


def basic_block_prefixes(model: Any) -> list[str]:
    """Prefixes of every ``/layerN/layerN.i`` block that has conv1+conv2 but no conv3."""
    convs = {n.name for n in model.graph.node if n.op_type == "Conv"}
    prefixes = []
    for name in sorted(convs):
        m = re.match(r"(/layer\d/layer\d\.\d+)/conv1/Conv$", name)
        if m and f"{m.group(1)}/conv3/Conv" not in convs:
            prefixes.append(m.group(1))
    return prefixes


def bind_basic_block(model: Any, prefix: str) -> dict[str, Any]:
    nodes = list(model.graph.node)
    edges = dict(qdq_edge_map(model))
    init = {str(i.name): numpy_helper.to_array(i) for i in model.graph.initializer}
    plans = {p.node_name: p for p in plan_all_convs(model)}
    index_of = {n.name: i for i, n in enumerate(nodes)}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for name in node.input:
            if name:
                consumers.setdefault(str(name), []).append(index)
    conv_a, conv_b = nodes[index_of[f"{prefix}/conv1/Conv"]], nodes[index_of[f"{prefix}/conv2/Conv"]]
    ds = nodes[index_of[f"{prefix}/downsample/downsample.0/Conv"]] if f"{prefix}/downsample/downsample.0/Conv" in index_of else None
    add = nodes[index_of[f"{prefix}/Add"]]
    plan_a, plan_b = plans[conv_a.name], plans[conv_b.name]
    if tuple(plan_a.weight_shape[2:]) != (3, 3) or tuple(plan_b.weight_shape[2:]) != (3, 3) or tuple(plan_b.stride) != (1, 1):
        raise ValueError(f"{prefix}: basic blocks need 3x3 conv1 (any stride) and 3x3 stride-1 conv2")
    stride = int(plan_a.stride[0])
    input_shape, output_shape = tuple(plan_a.input_shape), tuple(plan_b.output_shape)
    input_edge = edges[str(conv_a.input[0])]
    input_scale, input_zero = _single_q_params(input_edge, "block input")
    if input_zero != 128:
        raise ValueError("the layer engine requires uint8 activations with zero point 128")

    def conv_params(node, activation_edge):
        act_scale, _ = _single_q_params(activation_edge, f"{node.name} activation")
        w_edge, b_edge = edges[str(node.input[1])], edges[str(node.input[2])]
        w_scale, w_zero = _single_q_params(w_edge, "weight")
        b_scale, b_zero = _single_q_params(b_edge, "bias")
        if w_zero != 0 or b_zero != 0 or w_edge.params.dtype != "i8" or b_edge.params.dtype != "i8":
            raise ValueError(f"{node.name}: weight and bias must be signed int8 with zero point 0")
        weight = np.asarray(init[w_edge.input_name], dtype=np.int8)
        product = act_scale * w_scale
        bias = np.asarray(init[b_edge.input_name], dtype=np.int32).reshape(-1).astype(np.float64) * b_scale / product
        if not np.allclose(bias, np.rint(bias), atol=1e-6):
            raise ValueError(f"{node.name}: bias is not an exact integer accumulator")
        return weight, np.rint(bias).astype(np.int32), product

    q_a = _quantizer_after(str(conv_a.output[0]), nodes, consumers, edges)
    q_b = _quantizer_after(str(conv_b.output[0]), nodes, consumers, edges)
    final_q = _quantizer_after(str(add.output[0]), nodes, consumers, edges)
    w1, b1, prod1 = conv_params(conv_a, input_edge)
    a_scale, _ = _single_q_params(q_a, "conv1 output")
    w2, b2, prod2 = conv_params(conv_b, edges[str(conv_b.input[0])])
    b_scale, _ = _single_q_params(q_b, "conv2 output")
    final_scale, final_zero = _single_q_params(final_q, "block output")
    skip_w = skip_b = skip_shift = None
    if ds is not None:
        q_ds = _quantizer_after(str(ds.output[0]), nodes, consumers, edges)
        skip_w, skip_b, prod_ds = conv_params(ds, input_edge)
        ds_scale, _ = _single_q_params(q_ds, "downsample output")
        skip_shift = _power_of_two_shift(ds_scale / prod_ds, f"{ds.name} requantization")
        skip_res = _power_of_two_exponent(ds_scale / final_scale, "downsample branch scale ratio")
    else:
        skip_res = _power_of_two_exponent(input_scale / final_scale, "identity branch scale ratio")
    final_dq = next(e for e in edges.values() if e.op_type == "DequantizeLinear" and e.input_name == final_q.output_name)
    covered = tuple(i for i, n in enumerate(nodes) if prefix in str(n.name) and index_of[conv_a.name] <= i <= index_of[add.name] + 3)
    return {
        "block": SimpleNamespace(prefix=prefix), "kind": "basic",
        "raw_weights": {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "skip_weight": skip_w, "skip_bias": skip_b},
        "shifts": (_power_of_two_shift(a_scale / prod1, f"{conv_a.name} requantization"),
                   _power_of_two_shift(b_scale / prod2, f"{conv_b.name} requantization")),
        "skip_output_shift": skip_shift,
        "main_residual_shift": _power_of_two_exponent(b_scale / final_scale, "residual branch scale ratio"),
        "skip_residual_shift": skip_res,
        "conv2_stride": (stride, stride),
        "input_shape": input_shape, "output_shape": output_shape,
        "input_raw_name": input_edge.input_name, "input_zero_point": input_zero,
        "output_raw_name": final_q.output_name, "output_dequant_name": final_dq.output_name,
        "output_scale": final_scale, "output_zero_point": final_zero,
        "covered_nodes": covered,
    }
