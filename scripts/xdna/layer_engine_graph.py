"""Compile a QDQ CNN graph (ResNet, MobileNet-style, YOLO backbone/neck, ...) into layer-engine jobs.

Engine operators:

* ``Conv`` 1x1 / 3x3 (stride 1 or 2) and depthwise 3x3, with ``Relu`` / ``Clip`` (ReLU6 = int8 clamp) and the
  following ``QuantizeLinear``; an ``Add`` of a conv result and an earlier tensor becomes the conv's residual
  epilogue;
* any chain of pointwise float nodes between a ``DequantizeLinear`` and a ``QuantizeLinear`` (HardSwish,
  Sigmoid -> Mul = SiLU, GELU, ...): one 256-entry table job, built by running the chain through tinygrad;
* ``Split`` / ``Concat`` (channel axis, with per-source re-scaling), ``MaxPool``, nearest ``Resize``.

Every activation lives in the arena as uint8 with zero point 128 and a power-of-two scale. A Conv whose
output map is too large for one core's 512-byte region (the first, high-resolution layers) runs on the *host*
with the same numpy reference the tests use, and its result is written into the arena before launch. Nodes the
engine cannot run (Reshape, Softmax, the detection-head decode, ...) are the host tail: the engine tensors
they consume are the *boundaries* the caller reads back.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from onnx import numpy_helper

from layer_engine import REGION_BYTES, Job, Layout, assign_slots, layout_for

POINTWISE = frozenset(
    {
        "Sigmoid",
        "Mul",
        "Add",
        "Sub",
        "Div",
        "Neg",
        "Relu",
        "Clip",
        "HardSwish",
        "HardSigmoid",
        "Tanh",
        "Erf",
        "Exp",
        "Softplus",
        "Mish",
        "Gelu",
        "LeakyRelu",
        "Abs",
        "Sqrt",
        "Reciprocal",
        "Identity",
    }
)


@dataclass
class Tensor:
    slot: int
    layout: Layout
    scale: float
    zero: int
    host: bool = False


@dataclass
class Compiled:
    host_jobs: list[Job]
    jobs: list[Job]
    input_name: str
    input_layout: Layout
    input_scale: float
    input_channels: int
    boundaries: dict[str, Tensor] = field(
        default_factory=dict
    )  # Q-output name -> tensor (after slot assignment)
    pinned: list[int] = field(default_factory=list)


def _shift(ratio: float, label: str) -> int:
    shift = round(math.log2(ratio))
    if shift < 0 or not math.isclose(ratio, 2.0**shift, rel_tol=1e-6):
        raise ValueError(
            f"{label}: scale ratio {ratio} is not a non-negative power of two"
        )
    return shift


def _exp2(ratio: float, label: str) -> int:
    exponent = round(math.log2(ratio))
    if not math.isclose(ratio, 2.0**exponent, rel_tol=1e-6):
        raise ValueError(f"{label}: ratio {ratio} is not a power of two")
    return exponent


def compile_graph(model: Any, reuse_slots: bool = False) -> Compiled:
    from tinygrad_lower import subgraph_table

    graph = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    alias: dict[str, str] = {}
    for node in graph.node:
        if node.op_type == "Constant":
            init[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
        elif node.op_type == "Identity":
            src = node.input[0]
            if src in init:
                init[node.output[0]] = init[src]
            alias[node.output[0]] = alias.get(src, src)
    res = lambda name: alias.get(name, name)  # noqa: E731
    nodes = [n for n in graph.node if n.op_type not in ("Constant", "Identity")]
    producers = {res(o): n for n in nodes for o in n.output}
    consumers: dict[str, list[Any]] = {}
    for n in nodes:
        for i in n.input:
            consumers.setdefault(res(i), []).append(n)

    def qp(node):  # (scale, zero) of a Q/DQ node
        return float(init[res(node.input[1])]), int(init[res(node.input[2])])

    first_q = next(
        n
        for n in nodes
        if n.op_type == "QuantizeLinear" and res(n.input[0]) == graph.input[0].name
    )
    shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    _, channels, height, width = shape
    padded_channels = -(-channels // 8) * 8
    s0, z0 = qp(first_q)
    in_layout = layout_for(padded_channels, width, height)
    tensors: dict[str, Tensor] = {
        first_q.output[0]: Tensor(0, in_layout, s0, z0, host=True)
    }
    jobs: list[Job] = []
    host_jobs: list[Job] = []
    counter = [1]
    handled: set[str] = {first_q.name}

    def new_slot() -> int:
        counter[0] += 1
        return counter[0] - 1

    def dq_source(name: str) -> Tensor:
        node = producers[res(name)]
        if node.op_type != "DequantizeLinear":
            raise ValueError(f"{name}: expected a DequantizeLinear activation")
        return tensors[res(node.input[0])]

    def add_job(job: Job, host: bool = False) -> Job:
        if job.out_layout.nbc * job.out_layout.pixels * 8 > REGION_BYTES:
            host = True
        if host:
            slots_on_host = {j.out_slot for j in host_jobs} | {0}
            if (
                not {job.in_slot}
                | ({job.res_slot} if job.res_slot is not None else set())
                <= slots_on_host
            ):
                raise ValueError(
                    f"{job.name}: output map ({job.out_layout.pixels} px) is too large for the engine but its input is not a host tensor"
                )
        if host:
            # too large for a core's region: computed on the host, one dense block per region in the arena
            ol = job.out_layout
            job.out_layout = Layout(ol.nb, 1, ol.w, ol.h, region_bytes=ol.pixels * 8)
            host_jobs.append(job)
        else:
            jobs.append(job)
        return job

    def register(q_out: str, job: Job, scale: float, zero: int = 128) -> None:
        tensors[res(q_out)] = Tensor(
            job.out_slot, job.out_layout, scale, zero, host=job in host_jobs
        )

    def attrs_of(node):
        return {
            a.name: (
                a.f
                if a.type == 1
                else a.i
                if a.type == 2
                else a.s
                if a.type == 3
                else list(a.ints)
            )
            for a in node.attribute
        }

    def engine_inputs(node, indices=None) -> bool:
        """All activation inputs come from a DequantizeLinear over an engine tensor (else the node is host tail)."""
        names = [
            node.input[i]
            for i in (indices if indices is not None else range(len(node.input)))
            if node.input[i] and res(node.input[i]) not in init
        ]
        return bool(names) and all(
            res(n) in producers
            and producers[res(n)].op_type == "DequantizeLinear"
            and res(producers[res(n)].input[0]) in tensors
            for n in names
        )

    for node in nodes:
        if node.name in handled:
            continue
        op = node.op_type
        if op in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if not engine_inputs(
            node, [0] if op in ("Split", "Resize", "MaxPool", "Conv") else None
        ):
            continue  # float input: host tail
        if op in ("Split", "Concat", "MaxPool", "Resize", "Add"):
            handled.add(
                node.name
            )  # consumed by the engine (its Q nodes are added below)
        if op == "Conv":
            src = dq_source(node.input[0])
            w_scale, _ = qp(producers[res(node.input[1])])
            weight = init[res(producers[res(node.input[1])].input[0])]
            b_scale, _ = qp(producers[res(node.input[2])])
            bias_q = init[res(producers[res(node.input[2])].input[0])]
            a = attrs_of(node)
            group, strides = a.get("group", 1), a.get("strides", [1, 1])
            oc, icg, kh, kw = weight.shape
            if (
                icg * group < src.layout.nb * 8
            ):  # padded input channels (e.g. RGB -> 8): extra weights are zero
                pad = np.zeros(
                    (oc, src.layout.nb * 8 - icg, kh, kw), dtype=weight.dtype
                )
                weight, icg = np.concatenate([weight, pad], axis=1), src.layout.nb * 8
            product = src.scale * w_scale
            bias = bias_q.astype(np.float64) * b_scale / product
            if not np.all(np.isfinite(bias)) or np.abs(bias).max() >= 2**31:
                raise ValueError(
                    f"{node.name}: bad quantization scales (input {src.scale}, weight {w_scale}, bias {b_scale})"
                )
            if not np.allclose(bias, np.rint(bias), atol=1e-6):
                raise ValueError(
                    f"{node.name}: bias is not an exact integer accumulator"
                )
            relu, clamp, tail, act = False, 127, consumers[res(node.output[0])], None
            if len(tail) == 1 and tail[0].op_type in ("Relu", "Clip"):
                act, relu = tail[0], True
                tail = consumers[res(act.output[0])]
                if act.op_type == "Clip":
                    lo = (
                        float(init[res(act.input[1])])
                        if len(act.input) > 1 and act.input[1]
                        else None
                    )
                    hi = (
                        float(init[res(act.input[2])])
                        if len(act.input) > 2 and act.input[2]
                        else None
                    )
                    if lo != 0.0:
                        raise ValueError(
                            f"{act.name}: only Clip(0, hi) (ReLU6) is supported"
                        )
                    clamp_hi = hi
                else:
                    clamp_hi = None
            else:
                clamp_hi = None
            (qnode,) = tail
            out_scale, out_zero = qp(qnode)
            if out_zero != 128 or src.zero != 128:
                raise ValueError(
                    "the engine keeps activations as uint8 with zero point 128"
                )
            if clamp_hi is not None:
                clamp = min(127, int(round(clamp_hi / out_scale)))
            job_scale, out_name = out_scale, qnode.output[0]
            res_slot, res_mode, ea, eb = None, 0, 0, 0
            dqs = consumers.get(res(qnode.output[0]), [])
            if len(dqs) == 1 and dqs[0].op_type == "DequantizeLinear":
                users = consumers.get(res(dqs[0].output[0]), [])
                adds = (
                    [u for u in users if u.op_type == "Add"] if len(users) == 1 else []
                )
                if adds:
                    add = adds[0]
                    other = next(
                        i for i in add.input if res(i) != res(dqs[0].output[0])
                    )
                    skip = dq_source(other)
                    (final_q,) = consumers[res(add.output[0])]
                    final_scale, final_zero = qp(final_q)
                    if final_zero != 128 or skip.zero != 128 or relu:
                        raise ValueError(
                            "fused Add needs uint8/128 operands and a linear (no ReLU) conv"
                        )
                    ea = _exp2(out_scale / final_scale, "residual branch scale ratio")
                    eb = _exp2(skip.scale / final_scale, "skip branch scale ratio")
                    res_slot, res_mode = skip.slot, 2
                    job_scale, out_name = final_scale, final_q.output[0]
                    handled.update({dqs[0].name, add.name, final_q.name})
            shift = _shift(out_scale / product, node.name)
            common = dict(
                shift=shift,
                relu=relu,
                in_flip=True,
                out_flip=True,
                clamp=clamp,
                res_slot=res_slot,
                res_mode=res_mode,
                ea=ea,
                eb=eb,
            )
            slot = new_slot()
            if group > 1 and group == oc == weight.shape[0] and icg == 1:
                if (kh, kw) != (3, 3):
                    raise ValueError(f"{node.name}: only 3x3 depthwise is supported")
                job = Job(
                    node.name,
                    weight,
                    np.rint(bias).astype(np.int32),
                    src.slot,
                    slot,
                    src.layout,
                    stride=strides[0],
                    kind="dw",
                    **common,
                )
            elif group == 1:
                pads = a.get("pads", [0, 0, 0, 0])
                job = Job(
                    node.name,
                    weight,
                    np.rint(bias).astype(np.int32),
                    src.slot,
                    slot,
                    src.layout,
                    stride=strides[0],
                    pad=pads[0],
                    **common,
                )
                if kh not in (1, 3) or pads[0] != kh // 2:
                    add_job(
                        job, host=True
                    )  # e.g. YOLOv5's 6x6 stride-2 stem: no engine kernel, runs on the host
                    register(out_name, job, job_scale)
                    handled.update(
                        {node.name, qnode.name} | ({act.name} if act else set())
                    )
                    continue
            else:
                raise ValueError(
                    f"{node.name}: grouped convolution (group={group}) has no kernel"
                )
            add_job(job)
            register(out_name, job, job_scale)
            handled.update({node.name, qnode.name} | ({act.name} if act else set()))
        elif op == "Split":
            src = dq_source(node.input[0])
            sizes = (
                list(init[res(node.input[1])])
                if len(node.input) > 1
                else attrs_of(node)["split"]
            )
            offset = 0
            for out, size in zip(node.output, sizes):
                (qnode,) = consumers[res(out)]
                out_scale, out_zero = qp(qnode)
                if size % 8 or offset % 8:
                    raise ValueError(
                        f"{node.name}: split sizes must be multiples of 8 channels"
                    )
                e = _exp2(src.scale / out_scale, "split rescale")
                spec = [(0, offset // 8 + g, e) for g in range(size // 8)]
                job = Job(
                    f"{node.name}:{out}",
                    np.zeros((size, 1, 1, 1), dtype=np.int8),
                    np.zeros(size, dtype=np.int32),
                    src.slot,
                    new_slot(),
                    src.layout,
                    kind="copy",
                    copy_spec=spec,
                )
                add_job(job)
                register(qnode.output[0], job, out_scale, out_zero)
                handled.add(qnode.name)
                offset += size
        elif op == "Concat":
            srcs = [dq_source(i) for i in node.input]
            (qnode,) = consumers[res(node.output[0])]
            out_scale, out_zero = qp(qnode)
            current = srcs[0]
            cur_scale = current.scale
            for nxt in srcs[1:]:
                spec = [
                    (0, g, _exp2(cur_scale / out_scale, "concat rescale"))
                    for g in range(current.layout.nb)
                ]
                spec += [
                    (1, g, _exp2(nxt.scale / out_scale, "concat rescale"))
                    for g in range(nxt.layout.nb)
                ]
                channels = (current.layout.nb + nxt.layout.nb) * 8
                job = Job(
                    f"{node.name}:{len(spec)}",
                    np.zeros((channels, 1, 1, 1), dtype=np.int8),
                    np.zeros(channels, dtype=np.int32),
                    current.slot,
                    new_slot(),
                    current.layout,
                    kind="copy",
                    copy_spec=spec,
                    res_slot=nxt.slot,
                    b_layout=nxt.layout,
                )
                add_job(job)
                current, cur_scale = (
                    Tensor(job.out_slot, job.out_layout, out_scale, out_zero),
                    out_scale,
                )
            tensors[res(qnode.output[0])] = current
            handled.add(qnode.name)
        elif op == "MaxPool":
            src = dq_source(node.input[0])
            (qnode,) = consumers[res(node.output[0])]
            out_scale, out_zero = qp(qnode)
            a = attrs_of(node)
            k, s = a["kernel_shape"][0], a.get("strides", [1, 1])[0]
            if (
                a["kernel_shape"][0] != a["kernel_shape"][1]
                or k % 2 == 0
                or list(a.get("pads", [0] * 4)) != [(k - 1) // 2] * 4
            ):
                raise ValueError(
                    f"{node.name}: only square odd 'same'-padded max pools are supported"
                )
            channels = src.layout.nb * 8
            job = Job(
                node.name,
                np.zeros((channels, 1, 1, 1), dtype=np.int8),
                np.zeros(channels, dtype=np.int32),
                src.slot,
                new_slot(),
                src.layout,
                stride=s,
                kind="maxpool",
                factor=k,
                exp=_exp2(src.scale / out_scale, "pool rescale"),
            )
            add_job(job)
            register(qnode.output[0], job, out_scale, out_zero)
            handled.add(qnode.name)
        elif op == "Resize":
            src = dq_source(node.input[0])
            (qnode,) = consumers[res(node.output[0])]
            out_scale, out_zero = qp(qnode)
            scales = (
                init[res(node.input[2])]
                if len(node.input) > 2 and node.input[2]
                else None
            )
            if (
                scales is None
                or scales[2] != scales[3]
                or scales[2] != int(scales[2])
                or attrs_of(node).get("mode", b"nearest") not in (b"nearest", "nearest")
            ):
                raise ValueError(
                    f"{node.name}: only nearest resize by an integer factor is supported"
                )
            channels = src.layout.nb * 8
            job = Job(
                node.name,
                np.zeros((channels, 1, 1, 1), dtype=np.int8),
                np.zeros(channels, dtype=np.int32),
                src.slot,
                new_slot(),
                src.layout,
                kind="up",
                factor=int(scales[2]),
                exp=_exp2(src.scale / out_scale, "resize rescale"),
            )
            add_job(job)
            register(qnode.output[0], job, out_scale, out_zero)
            handled.add(qnode.name)
        elif op == "Add" and all(
            res(i) in producers and producers[res(i)].op_type == "DequantizeLinear"
            for i in node.input
        ):
            a_t, b_t = dq_source(node.input[0]), dq_source(node.input[1])
            (qnode,) = consumers[res(node.output[0])]
            out_scale, out_zero = qp(qnode)
            channels = a_t.layout.nb * 8
            job = Job(
                node.name,
                np.zeros((channels, 1, 1, 1), dtype=np.int8),
                np.zeros(channels, dtype=np.int32),
                a_t.slot,
                new_slot(),
                a_t.layout,
                kind="add",
                res_slot=b_t.slot,
                b_layout=b_t.layout,
                ea=_exp2(a_t.scale / out_scale, "add branch A ratio"),
                eb=_exp2(b_t.scale / out_scale, "add branch B ratio"),
            )
            add_job(job)
            register(qnode.output[0], job, out_scale, out_zero)
            handled.add(qnode.name)
        elif (
            op in POINTWISE
            and node.input
            and res(node.input[0]) in producers
            and producers[res(node.input[0])].op_type == "DequantizeLinear"
        ):
            x_dq = res(node.input[0])
            src = dq_source(x_dq)
            chain: list[Any] = []
            q_end: list[Any] = []

            def visit(tensor: str) -> None:
                for c in consumers.get(res(tensor), []):
                    if c.op_type == "QuantizeLinear":
                        if c not in q_end:
                            q_end.append(c)
                    elif c.op_type in POINTWISE and c not in chain:
                        chain.append(c)
                        for o in c.output:
                            visit(o)

            visit(x_dq)
            if len(q_end) != 1:
                raise ValueError(
                    f"{node.name}: a pointwise chain must end in exactly one QuantizeLinear"
                )
            chain = [n for n in nodes if n in chain]
            defined = {x_dq} | {res(o) for n in chain for o in n.output}
            if any(
                res(i) not in defined and res(i) not in init
                for n in chain
                for i in n.input
                if i
            ):
                raise ValueError(
                    f"{node.name}: the chain has an activation input other than its DequantizeLinear"
                )
            qnode = q_end[0]
            out_scale, out_zero = qp(qnode)
            y_name = res(qnode.input[0])
            table = subgraph_table(
                chain,
                x_dq,
                y_name,
                {k: v for k, v in init.items()},
                src.scale,
                src.zero,
                False,
                out_scale,
                out_zero,
                False,
            )
            channels = src.layout.nb * 8
            job = Job(
                node.name,
                np.zeros((channels, 1, 1, 1), dtype=np.int8),
                np.zeros(channels, dtype=np.int32),
                src.slot,
                new_slot(),
                src.layout,
                kind="lut",
                table=table,
            )
            add_job(job)
            register(qnode.output[0], job, out_scale, out_zero)
            handled.update({n.name for n in chain} | {qnode.name})
        # anything else belongs to the host tail
    handled_outputs = {res(o) for n in nodes if n.name in handled for o in n.output}
    boundaries: dict[str, Tensor] = {}
    for n in nodes:
        if n.name in handled or n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        for i in n.input:
            p = producers.get(res(i))
            if (
                p is not None
                and p.op_type == "DequantizeLinear"
                and res(p.input[0]) in tensors
            ):
                boundaries[res(p.input[0])] = tensors[res(p.input[0])]
    keep = [t.slot for t in boundaries.values()]
    if os.environ.get(
        "ENGINE_KEEP_ALL"
    ):  # debugging: never reuse a slot and expose every job output as a boundary
        keep = [j.out_slot for j in jobs]
        boundaries = {j.name: Tensor(j.out_slot, j.out_layout, 1.0, 128) for j in jobs}
    pinned = sorted({j.out_slot for j in host_jobs} | {0})
    mapping: dict[int, int] = {}
    count = assign_slots(jobs, pinned=pinned, keep=keep, mapping=mapping)
    for name, t in boundaries.items():
        t.slot = mapping.get(t.slot, t.slot)
    for j in host_jobs:  # host job slots follow the pinned mapping
        j.out_slot = mapping[j.out_slot]
        j.in_slot = mapping.get(j.in_slot, j.in_slot)
        if j.res_slot is not None:
            j.res_slot = mapping.get(j.res_slot, j.res_slot)
    del handled_outputs, count
    return Compiled(
        host_jobs, jobs, first_q.output[0], in_layout, s0, channels, boundaries, pinned
    )
