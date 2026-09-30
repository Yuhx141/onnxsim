"""Compile a straight-line QDQ CNN graph into layer-engine jobs.

Handles the operators the engine has kernels or table lowerings for:

* ``Conv`` 1x1 / 3x3 (stride 1 or 2), and depthwise 3x3 (``group == channels``), followed by ``Relu`` /
  ``Clip`` (ReLU6 becomes an int8 clamp) and a ``QuantizeLinear``;
* ``Add`` of a Conv result and an earlier activation (fused into the Conv job as its residual: the same
  epilogue as a ResNet bottleneck's last conv);
* any pointwise unary operator between a ``DequantizeLinear`` and a ``QuantizeLinear`` (HardSwish,
  Sigmoid, Tanh, GELU, ...): a 256-entry table built through tinygrad (``tinygrad_lower.unary_table``).

Every activation lives in the arena as uint8 with zero point 128 (jobs re-centre on load and re-bias on store).
The graph stops at the first operator it does not handle (normally ``GlobalAveragePool``): that tensor is the
engine output and the classifier tail stays on the host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from onnx import numpy_helper

from layer_engine import Job, Layout, assign_slots, layout_for

SUPPORTED_UNARY_FALLBACK = frozenset({"HardSwish", "HardSigmoid", "Sigmoid", "Tanh", "Gelu", "Erf", "Softplus", "Mish"})


@dataclass
class Tensor:
    slot: int
    layout: Layout
    scale: float
    zero: int


def _shift(ratio: float, label: str) -> int:
    shift = round(math.log2(ratio))
    if shift < 0 or not math.isclose(ratio, 2.0**shift, rel_tol=1e-6):
        raise ValueError(f"{label}: scale ratio {ratio} is not a non-negative power of two")
    return shift


def _exp2(ratio: float, label: str) -> int:
    exponent = round(math.log2(ratio))
    if not math.isclose(ratio, 2.0**exponent, rel_tol=1e-6):
        raise ValueError(f"{label}: ratio {ratio} is not a power of two")
    return exponent


def compile_graph(model: Any, in_channels: int | None = None, reuse_slots: bool = False):
    """Returns ``(jobs, input_tensor_name, output_tensor_name, output_layout)``."""
    from tinygrad_lower import unary_table

    graph = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            init[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    producers = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list[Any]] = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    scale_of = lambda node: (float(init[node.input[1]]), int(init[node.input[2]]))  # noqa: E731

    first_q = next(n for n in graph.node if n.op_type == "QuantizeLinear")
    shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    _, channels, height, width = shape
    s0, z0 = scale_of(first_q)
    tensors: dict[str, Tensor] = {first_q.output[0]: Tensor(0, layout_for(channels, width, height), s0, z0)}
    jobs: list[Job] = []
    slot = 1
    out_name = first_q.output[0]
    handled: set[str] = set()

    def dq_source(name: str) -> Tensor:
        node = producers[name]
        if node.op_type != "DequantizeLinear":
            raise ValueError(f"{name}: expected a DequantizeLinear activation")
        return tensors[node.input[0]]

    def constant(name: str) -> np.ndarray:
        return init[producers[name].input[0]] if name in producers and producers[name].op_type == "DequantizeLinear" else init[name]

    def qparams(dq_name: str):
        node = producers[dq_name]
        return float(init[node.input[1]]), int(init[node.input[2]]), np.asarray(init[node.input[0]])

    for node in graph.node:
        if node.name in handled:
            continue
        if node.op_type == "Conv":
            src = dq_source(node.input[0])
            w_scale, _, weight = qparams(node.input[1])
            b_scale, _, bias_q = qparams(node.input[2])
            group = next((a.i for a in node.attribute if a.name == "group"), 1)
            strides = next((list(a.ints) for a in node.attribute if a.name == "strides"), [1, 1])
            oc, icg, kh, kw = weight.shape
            product = src.scale * w_scale
            bias = np.rint(bias_q.astype(np.float64) * b_scale / product)
            if not np.allclose(bias, bias_q.astype(np.float64) * b_scale / product, atol=1e-6):
                raise ValueError(f"{node.name}: bias is not an exact integer accumulator")
            # the conv output: [Relu|Clip] then QuantizeLinear (or a Q that also feeds an Add)
            relu, clamp, tail = False, 127, consumers[node.output[0]]
            act = None
            if len(tail) == 1 and tail[0].op_type in ("Relu", "Clip"):
                act = tail[0]
                relu = True
                tail = consumers[act.output[0]]
                if act.op_type == "Clip":
                    hi = float(init[act.input[2]]) if len(act.input) > 2 and act.input[2] in init else None
                    lo = float(init[act.input[1]]) if len(act.input) > 1 and act.input[1] in init else None
                    if lo != 0.0:
                        raise ValueError(f"{act.name}: only Clip(0, hi) (ReLU6) is supported")
                    clamp_after = hi
                else:
                    clamp_after = None
            else:
                clamp_after = None
            (qnode,) = tail
            if qnode.op_type != "QuantizeLinear":
                raise ValueError(f"{node.name}: expected QuantizeLinear after the conv (got {qnode.op_type})")
            out_scale, out_zero = scale_of(qnode)
            if out_zero != 128 or src.zero != 128:
                raise ValueError("the engine keeps activations as uint8 with zero point 128")
            if clamp_after is not None:
                clamp = min(127, int(round(clamp_after / out_scale)))
            job_out_scale, out_tensor = out_scale, qnode.output[0]
            res_slot, res_mode, ea, eb = None, 0, 0, 0
            # Add fusion: the Q output is only consumed by a DQ feeding a residual Add
            dqs = consumers.get(qnode.output[0], [])
            if len(dqs) == 1:
                users = consumers.get(dqs[0].output[0], [])
                adds = [a for a in users if a.op_type == "Add"] if len(users) == 1 else []  # a tensor that also feeds a conv is the skip, not the branch
                if adds:
                    add = adds[0]
                    other = next(i for i in add.input if i != dqs[0].output[0])
                    skip = dq_source(other)
                    (final_q,) = consumers[add.output[0]]
                    final_scale, final_zero = scale_of(final_q)
                    if final_zero != 128 or skip.zero != 128 or relu:
                        raise ValueError("fused Add needs uint8/128 operands and a linear (no ReLU) conv")
                    ea = _exp2(out_scale / final_scale, "residual branch scale ratio")
                    eb = _exp2(skip.scale / final_scale, "skip branch scale ratio")
                    res_slot, res_mode = skip.slot, 2
                    job_out_scale, out_tensor = final_scale, final_q.output[0]
                    handled.update({dqs[0].name, add.name, final_q.name})
            shift = _shift(job_out_scale / product if res_mode == 0 else out_scale / product, node.name)
            layout_in = src.layout
            if group == oc == icg * group and group > 1:  # depthwise
                if (kh, kw) != (3, 3):
                    raise ValueError(f"{node.name}: only 3x3 depthwise is supported")
                job = Job(node.name, weight, bias.astype(np.int32), src.slot, slot, layout_in, stride=strides[0], shift=shift, relu=relu,
                          in_flip=True, out_flip=True, clamp=clamp, kind="dw", res_slot=res_slot, res_mode=res_mode, ea=ea, eb=eb)
            elif group == 1:
                job = Job(node.name, weight, bias.astype(np.int32), src.slot, slot, layout_in, stride=strides[0], shift=shift, relu=relu,
                          in_flip=True, out_flip=True, clamp=clamp, res_slot=res_slot, res_mode=res_mode, ea=ea, eb=eb)
            else:
                raise ValueError(f"{node.name}: grouped convolution (group={group}) has no kernel")
            if res_mode and job.kind == "dw":
                raise ValueError("a residual Add on a depthwise conv is not supported")
            slot += 1
            jobs.append(job)
            tensors[out_tensor] = Tensor(job.out_slot, job.out_layout, job_out_scale, 128)
            out_name = out_tensor
            handled.update({node.name, qnode.name} | ({act.name} if act else set()))
        elif node.op_type in ("QuantizeLinear", "DequantizeLinear", "Constant"):
            continue
        elif node.op_type in ("Relu", "Clip", "Add"):
            raise ValueError(f"{node.name}: standalone {node.op_type} outside a Conv pattern is not supported")
        elif node.op_type in ("GlobalAveragePool", "Flatten", "Gemm", "AveragePool", "MaxPool"):
            break  # the host tail starts here
        else:
            # pointwise unary between DQ and Q: table job, defined by tinygrad's lowering of the op itself
            src = dq_source(node.input[0])
            (qnode,) = consumers[node.output[0]]
            if qnode.op_type != "QuantizeLinear":
                raise ValueError(f"{node.name}: {node.op_type} is not followed by QuantizeLinear")
            out_scale, out_zero = scale_of(qnode)
            attrs = {}
            for a in node.attribute:
                attrs[a.name] = a.f if a.type == 1 else (a.i if a.type == 2 else list(a.ints))
            table = unary_table(node.op_type, src.scale, 128, False, out_scale, out_zero, False, attrs)
            job = Job(node.name, np.zeros((src.layout.nb * 8, 1, 1, 1), dtype=np.int8), np.zeros(src.layout.nb * 8, dtype=np.int32),
                      src.slot, slot, src.layout, kind="lut", table=table)
            slot += 1
            jobs.append(job)
            tensors[qnode.output[0]] = Tensor(job.out_slot, job.out_layout, out_scale, out_zero)
            out_name = qnode.output[0]
            handled.update({node.name, qnode.name})
    if reuse_slots:
        assign_slots(jobs)
    return jobs, first_q.output[0], out_name, jobs[-1].out_layout
