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

import layer_engine as le
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
    level: int = 0  # host round trips this tensor's value depends on


@dataclass
class Entry:
    """A float host tensor re-entering the engine through a QuantizeLinear (written into a pinned arena slot)."""

    q_name: str
    float_name: str
    slot: int
    layout: Layout
    scale: float
    zero: int
    level: int


@dataclass
class Compiled:
    host_jobs: list[Job]
    jobs: list[Job]
    input_name: str
    input_layout: Layout | None  # None: the network input is a float host tensor
    input_scale: float
    input_channels: int
    boundaries: dict[str, Tensor] = field(
        default_factory=dict
    )  # Q-output name -> tensor (after slot assignment)
    pinned: list[int] = field(default_factory=list)
    model: Any = None  # the (onnxsim-folded) model the jobs were compiled from
    entries: list = field(default_factory=list)  # host -> engine re-entries (see Entry)
    levels: int = 1  # engine launches needed: one per host round trip + 1
    dequant: dict = field(default_factory=dict)  # boundary Q name -> (float DQ output name, scale)
    host_nodes: list = field(default_factory=list)  # (node, level) of every host-tail operator, in graph order


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


def prepare_model(model: Any) -> Any:
    """Constant-fold the graph with onnxsim before codegen (Shape/Gather-derived Slice bounds, Reshape shapes, ...).

    onnxsim is optional: without it (or if it fails) the model is compiled as given. Set ``ONNXSIM_PATH`` to a
    directory holding a built ``onnxsim`` package if it is not installed.
    """
    import os
    import sys

    root = os.environ.get("ONNXSIM_PATH")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        import onnxsim

        simplified, ok = onnxsim.simplify(model)
    except Exception:
        return model
    return simplified if ok else model


def compile_graph(model: Any, reuse_slots: bool = False, simplify: bool = True) -> Compiled:
    from tinygrad_lower import subgraph_table

    if simplify:
        model = prepare_model(model)
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
        (
            n
            for n in nodes
            if n.op_type == "QuantizeLinear" and res(n.input[0]) == graph.input[0].name
        ),
        None,
    )
    shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    _, channels, height, width = shape
    padded_channels = -(-channels // 8) * 8
    if first_q is not None:
        s0, z0 = qp(first_q)
        in_layout = layout_for(padded_channels, width, height)
        tensors: dict[str, Tensor] = {first_q.output[0]: Tensor(0, in_layout, s0, z0, host=True)}
    else:  # the input is a float host tensor (a transformer's residual stream): everything starts on the host
        s0, in_layout, tensors = 1.0, None, {}
    jobs: list[Job] = []
    host_jobs: list[Job] = []
    counter = [1]
    handled: set[str] = {first_q.name} if first_q is not None else set()
    cur = [0]  # level of the node being compiled (max over the tensors it reads)
    hlevel: dict[str, int] = {} if first_q is not None else {graph.input[0].name: 0}  # float host tensor -> level
    entries: list[Entry] = []
    host_nodes: list = []
    from onnx import shape_inference

    shape_of = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in shape_inference.infer_shapes(model).graph.value_info}

    def new_slot() -> int:
        counter[0] += 1
        return counter[0] - 1

    def enter(node) -> None:
        """A QuantizeLinear over a float host tensor: the tensor re-enters the engine through a pinned arena slot."""
        fin = res(node.input[0])
        if fin not in hlevel or res(node.output[0]) in tensors:
            return
        dims = shape_of.get(fin)
        if not dims or len(dims) != 4 or dims[0] != 1 or dims[1] % 8:
            return  # not an engine-shaped map (e.g. the DFL decode): its consumers stay on the host
        layout = layout_for(dims[1], dims[3], dims[2])
        if layout.nbc * layout.pixels * 8 > REGION_BYTES:
            return
        scale, zero = qp(node)
        slot = new_slot()
        entries.append(Entry(res(node.output[0]), fin, slot, layout, scale, zero, hlevel[fin] + 1))
        tensors[res(node.output[0])] = Tensor(slot, layout, scale, zero, host=True, level=hlevel[fin] + 1)

    def dq_source(name: str) -> Tensor:
        node = producers[res(name)]
        if node.op_type != "DequantizeLinear":
            raise ValueError(f"{name}: expected a DequantizeLinear activation")
        if res(node.input[0]) not in tensors:
            enter(producers[res(node.input[0])])
        t = tensors[res(node.input[0])]
        cur[0] = max(cur[0], t.level)
        return t

    def add_job(job: Job, host: bool = False) -> Job:
        if job.out_layout.nbc * job.out_layout.pixels * 8 > REGION_BYTES:
            host = True
        if any(l is not None and l.nb * l.w * l.h * 8 > le.SLOT_BYTES for l in (job.in_layout, job.b_layout)):
            host = True  # an input map larger than an arena slot cannot be read by the engine: its consumers stay on the host until maps shrink
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

    def split_dw(job: Job, name: str, weight, bias) -> Job:
        """Add a depthwise job; one whose per-core weights overflow a slot becomes channel-block parts (slice, dw, concat)."""
        k = weight.shape[2]
        max_nbc = (le.ENGINE_SLOT_BYTES - le.DESC_BYTES) // (k * k * 64 + le.TILE * 4)
        nb = job.in_layout.nb
        parts = -(-(-(-nb // le.CORES)) // max_nbc)
        if parts <= 1:
            return add_job(job)
        per = -(-nb // parts)
        outs = []
        for lo in range(0, nb, per):
            hi = min(lo + per, nb)
            spec = [(0, g, 0) for g in range(lo, hi)]
            sl = add_job(Job(f"{name}:slice{lo}", np.zeros(((hi - lo) * 8, 1, 1, 1), dtype=np.int8), np.zeros((hi - lo) * 8, dtype=np.int32),
                             job.in_slot, new_slot(), job.in_layout, kind="copy", copy_spec=spec))
            part = Job(f"{name}:{lo}", weight[lo * 8 : hi * 8], bias[lo * 8 : hi * 8], sl.out_slot, new_slot(), sl.out_layout,
                       stride=job.stride, kind="dw", shift=job.shift, relu=job.relu, in_flip=True, out_flip=True, clamp=job.clamp)
            outs.append(add_job(part))
        cur = outs[0]
        for nxt in outs[1:]:
            spec = [(0, g, 0) for g in range(cur.out_layout.nb)] + [(1, g, 0) for g in range(nxt.out_layout.nb)]
            channels = (cur.out_layout.nb + nxt.out_layout.nb) * 8
            cur = add_job(Job(f"{name}:cat{len(spec)}", np.zeros((channels, 1, 1, 1), dtype=np.int8), np.zeros(channels, dtype=np.int32),
                              cur.out_slot, new_slot(), cur.out_layout, kind="copy", copy_spec=spec, res_slot=nxt.out_slot, b_layout=nxt.out_layout))
        return cur

    def register(q_out: str, job: Job, scale: float, zero: int = 128) -> None:
        reads = {job.in_slot, job.res_slot}
        level = max([cur[0]] + [t.level for t in tensors.values() if t.slot in reads])  # depends on whatever its inputs do
        tensors[res(q_out)] = Tensor(
            job.out_slot, job.out_layout, scale, zero, host=job in host_jobs, level=level
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

    def mark_host(node) -> None:
        lvl = max([hlevel.get(res(i), 0) for i in node.input if i and res(i) not in init] + [0])
        for o in node.output:
            hlevel[res(o)] = lvl
        host_nodes.append((node, lvl))

    def back(name: str):
        """Follow Reshape/Transpose producers back to a DequantizeLinear: (its output, chain nodes, transposed, dims)."""
        chain, transposed, dims = [], False, None
        while True:
            p = producers.get(res(name))
            if p is None:
                return None
            if p.op_type == "DequantizeLinear":
                return p.output[0], chain, transposed, dims
            if p.op_type == "Reshape" and res(p.input[1]) in init:
                dims = dims or [int(v) for v in init[res(p.input[1])]]
            elif p.op_type != "Transpose":
                return None
            transposed = transposed or p.op_type == "Transpose"
            chain.append(p)
            name = p.input[0]

    # attention matmuls between two quantized tensors: MatMul(Transpose(Reshape(K)), Reshape(Q)) -> Reshape -> Q  is
    # scores = K^T Q per head, MatMul(Reshape(V), Reshape(P)) -> Reshape -> Q  is context = V P (engine "amm" jobs)
    pending_amm: dict[str, dict] = {}
    for node in nodes:
        if node.op_type != "MatMul":
            continue
        outs = consumers.get(res(node.output[0]), [])
        if len(outs) != 1 or outs[0].op_type != "Reshape":
            continue
        tails = consumers.get(res(outs[0].output[0]), [])
        a, b = back(node.input[0]), back(node.input[1])
        if len(tails) != 1 or tails[0].op_type != "QuantizeLinear" or a is None or b is None or not b[1] or b[2] or not a[1]:
            continue
        pending_amm[node.name] = dict(qk=a[2], a_dq=b[0], b_dq=a[0], heads=(b[3] or [1])[0], reshape=outs[0], q=tails[0])
        handled.update({n.name for n in a[1] + b[1]} | {outs[0].name})

    for node in nodes:
        cur[0] = 0
        if node.name in handled:
            continue
        op = node.op_type
        if op == "MatMul" and node.name in pending_amm:
            info = pending_amm[node.name]
            a_t, b_t = dq_source(info["a_dq"]), dq_source(info["b_dq"])
            out_scale, out_zero = qp(info["q"])
            k = -_exp2(a_t.scale * b_t.scale / out_scale, "attention matmul rescale")
            heads = info["heads"]
            if k < 0 or a_t.layout.pixels % 8 or a_t.layout.pixels != b_t.layout.pixels:
                raise ValueError(f"{node.name}: attention matmul needs a multiple of 8 tokens and a coarser output scale")
            oc = heads * b_t.layout.pixels if info["qk"] else b_t.layout.nb * 8
            job = Job(node.name, np.zeros((oc, 1, 1, 1), dtype=np.int8), np.zeros(oc, dtype=np.int32), a_t.slot, new_slot(), a_t.layout,
                      kind="amm", res_slot=b_t.slot, b_layout=b_t.layout, shift=k, heads=heads, amm_t=info["qk"])
            add_job(job)
            register(info["q"].output[0], job, out_scale, out_zero)
            handled.add(info["q"].name)
            continue
        if op == "DequantizeLinear":
            if res(node.input[0]) in tensors:
                hlevel[res(node.output[0])] = tensors[res(node.input[0])].level
            else:
                mark_host(node)  # Q/DQ inside a float host region (e.g. the DFL decode)
            continue
        if op == "QuantizeLinear":
            enter(node)
            if res(node.output[0]) not in tensors:
                mark_host(node)
            continue
        if op in ("Conv", "ConvTranspose") and res(node.input[1]) in init:
            mark_host(node)  # float weights (not quantized): stays in the float host tail
            continue
        if not engine_inputs(
            node, [0] if op in ("Split", "Resize", "MaxPool", "Conv", "Slice", "AveragePool", "ConvTranspose") else None
        ):
            mark_host(node)
            continue  # float input: host tail
        if op in ("Split", "Concat", "MaxPool", "Resize", "Add", "Slice", "AveragePool", "ConvTranspose"):
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
            if group > 1 and not (group == oc and icg == 1):
                # grouped (not depthwise): a dense conv with a block-diagonal weight (extra MACs, exact result)
                dense = np.zeros((oc, icg * group, kh, kw), dtype=weight.dtype)
                per = oc // group
                for gi in range(group):
                    dense[gi * per : (gi + 1) * per, gi * icg : (gi + 1) * icg] = weight[gi * per : (gi + 1) * per]
                weight, icg, group = dense, icg * group, 1
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
            if group == 1 and len(dqs) == 1 and dqs[0].op_type == "DequantizeLinear":  # the dw kernel has no fused residual
                users = consumers.get(res(dqs[0].output[0]), [])
                adds = (
                    [u for u in users if u.op_type == "Add"] if len(users) == 1 else []
                )
                if adds:
                    add = adds[0]
                    other = next(
                        i for i in add.input if res(i) != res(dqs[0].output[0])
                    )
                    skip_src = producers.get(res(other))
                    if skip_src is None or skip_src.op_type != "DequantizeLinear":
                        adds = []  # the other operand is a float host tensor: the Add stays a host node
                    elif (
                        skip_src is not None
                        and skip_src.op_type == "DequantizeLinear"
                        and res(skip_src.input[0]) not in tensors
                        and res(skip_src.input[0]) in producers
                    ):
                        adds = []  # the skip branch (e.g. a ResNet downsample conv) compiles later and fuses the Add itself
                if adds:
                    skip = dq_source(other)
                    after = consumers[res(add.output[0])]
                    post_relu = len(after) == 1 and after[0].op_type == "Relu"  # residual block: Add -> ReLU -> Q
                    (final_q,) = consumers[res(after[0].output[0])] if post_relu else after
                    final_scale, final_zero = qp(final_q)
                    if final_zero != 128 or skip.zero != 128 or relu:
                        raise ValueError(
                            "fused Add needs uint8/128 operands and a linear (no ReLU) conv"
                        )
                    ea = _exp2(out_scale / final_scale, "residual branch scale ratio")
                    eb = _exp2(skip.scale / final_scale, "skip branch scale ratio")
                    res_slot, res_mode = skip.slot, 2
                    job_scale, out_name = final_scale, final_q.output[0]
                    handled.update({dqs[0].name, add.name, final_q.name} | ({after[0].name} if post_relu else set()))
                    relu = post_relu
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
                if kh != kw or kh not in (3, 5, 7):
                    raise ValueError(f"{node.name}: only 3x3, 5x5 and 7x7 depthwise are supported")
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
            if job.kind == "dw":
                job = split_dw(job, node.name, weight, bias)
            else:
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
        elif op == "Slice":
            src = dq_source(node.input[0])
            starts, ends = init[res(node.input[1])], init[res(node.input[2])]
            axes = init[res(node.input[3])] if len(node.input) > 3 and node.input[3] else np.arange(len(starts))
            steps = init[res(node.input[4])] if len(node.input) > 4 and node.input[4] else np.ones(len(starts), dtype=np.int64)
            (qnode,) = consumers[res(node.output[0])]
            out_scale, out_zero = qp(qnode)
            channels = src.layout.nb * 8
            begin, end = int(starts[0]), min(int(ends[0]), channels)
            if list(axes) != [1] or list(steps) != [1] or begin % 8 or end % 8:
                raise ValueError(f"{node.name}: only a channel Slice on multiples of 8 channels is supported")
            e = _exp2(src.scale / out_scale, "slice rescale")
            spec = [(0, begin // 8 + g, e) for g in range((end - begin) // 8)]
            job = Job(node.name, np.zeros((end - begin, 1, 1, 1), dtype=np.int8), np.zeros(end - begin, dtype=np.int32), src.slot,
                      new_slot(), src.layout, kind="copy", copy_spec=spec)
            add_job(job)
            register(qnode.output[0], job, out_scale, out_zero)
            handled.add(qnode.name)
        elif op == "AveragePool":
            src = dq_source(node.input[0])
            (qnode,) = consumers[res(node.output[0])]
            out_scale, out_zero = qp(qnode)
            a = attrs_of(node)
            k, st = a["kernel_shape"][0], a.get("strides", [1, 1])[0]
            if a["kernel_shape"][0] != a["kernel_shape"][1] or any(a.get("pads", [0] * 4)) or (k * k) & (k * k - 1):
                raise ValueError(f"{node.name}: only unpadded square average pools with a power-of-two window area are supported")
            shift = int(math.log2(k * k)) + _exp2(out_scale / src.scale, "average pool rescale")
            if shift < 0:
                raise ValueError(f"{node.name}: average pool output scale is finer than its input")
            channels = src.layout.nb * 8
            job = Job(node.name, np.zeros((channels, 1, 1, 1), dtype=np.int8), np.zeros(channels, dtype=np.int32), src.slot, new_slot(),
                      src.layout, stride=st, kind="avgpool", factor=k, shift=shift)
            add_job(job)
            register(qnode.output[0], job, out_scale, out_zero)
            handled.add(qnode.name)
        elif op == "ConvTranspose":
            src = dq_source(node.input[0])
            w_scale, _ = qp(producers[res(node.input[1])])
            weight = init[res(producers[res(node.input[1])].input[0])]  # [ic][oc][kh][kw]
            b_scale, _ = qp(producers[res(node.input[2])])
            bias_q = init[res(producers[res(node.input[2])].input[0])]
            a = attrs_of(node)
            ic, oc, kh, kw = weight.shape
            f = a.get("strides", [1, 1])[0]
            if (kh, kw) != (f, f) or f < 2 or oc % 8 or a.get("group", 1) != 1 or any(a.get("pads", [0] * 4)):
                raise ValueError(f"{node.name}: only ConvTranspose with kernel == stride and 8-multiple output channels is supported")
            (qnode,) = consumers[res(node.output[0])]
            out_scale, out_zero = qp(qnode)
            product = src.scale * w_scale
            bias = np.rint(bias_q.astype(np.float64) * b_scale / product).astype(np.int32)
            # kernel == stride: every input pixel writes a disjoint f x f patch = 1x1 conv to f*f*oc channels, then depth-to-space
            wt = np.zeros((f * f * oc, ic, 1, 1), dtype=weight.dtype)
            for tap in range(f * f):
                ky, kx = divmod(tap, f)
                wt[tap * oc : (tap + 1) * oc, :, 0, 0] = weight[:, :, ky, kx].T
            conv = Job(node.name + ":pw", wt, np.tile(bias, f * f), src.slot, new_slot(), src.layout, shift=_shift(out_scale / product, node.name),
                       relu=False, in_flip=True, out_flip=True)
            add_job(conv)
            d2s = Job(node.name + ":d2s", np.zeros((oc, 1, 1, 1), dtype=np.int8), np.zeros(oc, dtype=np.int32), conv.out_slot, new_slot(),
                      conv.out_layout, kind="d2s", factor=f)
            add_job(d2s)
            register(qnode.output[0], d2s, out_scale, out_zero)
            handled.add(qnode.name)
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
                    Tensor(job.out_slot, job.out_layout, out_scale, out_zero, level=max(t.level for t in srcs)),
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
            pads = list(a.get("pads", [0] * 4))
            if (
                a["kernel_shape"][0] != a["kernel_shape"][1]
                or a.get("strides", [s, s])[0] != a.get("strides", [s, s])[1]
                or len(set(pads[:2] + pads[2:])) > 1
                or any(d != 1 for d in a.get("dilations", [1, 1]))
                or pads[0] >= k
            ):
                raise ValueError(f"{node.name}: only square, symmetrically padded max pools are supported")
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
                pad=pads[0],
                ceil_pool=bool(a.get("ceil_mode", 0)),
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
        elif op == "GlobalAveragePool":
            src = dq_source(node.input[0])
            (qnode,) = consumers[res(node.output[0])]
            out_scale, out_zero = qp(qnode)
            pixels = src.layout.pixels
            if pixels & (pixels - 1):
                raise ValueError(f"{node.name}: the engine averages a power-of-two pixel count (got {pixels})")
            shift = int(math.log2(pixels)) + _exp2(out_scale / src.scale, "average pool rescale")
            if shift < 0:
                raise ValueError(f"{node.name}: average pool output scale is finer than its input")
            channels = src.layout.nb * 8
            job = Job(node.name, np.zeros((channels, 1, 1, 1), dtype=np.int8), np.zeros(channels, dtype=np.int32), src.slot, new_slot(),
                      src.layout, kind="gap", shift=shift)
            add_job(job)
            register(qnode.output[0], job, out_scale, out_zero)
            handled.update({node.name, qnode.name})
        elif (
            op == "Mul"
            and all(res(i) in producers and producers[res(i)].op_type == "DequantizeLinear" for i in node.input)
        ):
            a_t, b_t = dq_source(node.input[0]), dq_source(node.input[1])
            if a_t.layout.pixels == 1 and b_t.layout.pixels > 1:
                a_t, b_t = b_t, a_t
            if b_t.layout.pixels != 1 and (b_t.layout.pixels != a_t.layout.pixels or b_t.layout.nb != a_t.layout.nb):
                raise ValueError(f"{node.name}: only a broadcast multiply by a 1x1 map (squeeze-excite) or a same-shape product is supported")
            (qnode,) = consumers[res(node.output[0])]
            out_scale, out_zero = qp(qnode)
            k = -_exp2(a_t.scale * b_t.scale / out_scale, "broadcast multiply rescale")
            if k < 0:
                raise ValueError(f"{node.name}: multiply output scale is finer than the product scale")
            channels = a_t.layout.nb * 8
            job = Job(node.name, np.zeros((channels, 1, 1, 1), dtype=np.int8), np.zeros(channels, dtype=np.int32), a_t.slot, new_slot(),
                      a_t.layout, kind="bmul", res_slot=b_t.slot, b_layout=b_t.layout, shift=k)
            add_job(job)
            register(qnode.output[0], job, out_scale, out_zero)
            handled.update({node.name, qnode.name})
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
        if node.name not in handled:
            mark_host(node)  # anything else belongs to the host tail
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
    for o in graph.output:  # a network output that is itself an engine tensor (DequantizeLinear of an engine result)
        p = producers.get(res(o.name))
        if p is not None and p.op_type == "DequantizeLinear" and res(p.input[0]) in tensors:
            boundaries[res(p.input[0])] = tensors[res(p.input[0])]
    keep = [t.slot for t in boundaries.values()]
    if os.environ.get(
        "ENGINE_KEEP_ALL"
    ):  # debugging: never reuse a slot and expose every job output as a boundary
        keep = [j.out_slot for j in jobs]
        boundaries = {j.name: Tensor(j.out_slot, j.out_layout, 1.0, 128) for j in jobs}
    if os.environ.get("ENGINE_KEEP_EVERY"):  # debugging: keep and expose every Nth job output but still reuse the other slots
        spec = os.environ["ENGINE_KEEP_EVERY"]
        wanted = (lambda i: i in {int(v) for v in spec[1:].split(",")}) if spec.startswith("@") else (lambda i: i % int(spec) == 0)
        extra = {j.name: Tensor(j.out_slot, j.out_layout, 1.0, 128) for i, j in enumerate(jobs) if wanted(i)}
        keep = keep + [t.slot for t in extra.values()]
        boundaries = {**boundaries, **extra}
    if os.environ.get("ENGINE_MAX_JOBS"):  # debugging: truncate the job list (no host tail)
        del jobs[int(os.environ["ENGINE_MAX_JOBS"]) :]
        keep, boundaries = [jobs[-1].out_slot], {}
    pinned = sorted({j.out_slot for j in host_jobs} | {0} | {e.slot for e in entries})
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
    for e in entries:
        e.slot = mapping[e.slot]
    dequant = {}
    for n in nodes:
        if n.op_type == "DequantizeLinear" and res(n.input[0]) in boundaries:
            dequant.setdefault(res(n.input[0]), (n.output[0], float(init[res(n.input[1])])))
    levels = max([e.level for e in entries] + [0]) + 1
    return Compiled(
        host_jobs, jobs, first_q.output[0] if first_q is not None else graph.input[0].name, in_layout, s0, channels, boundaries, pinned, model, entries, levels, dequant, host_nodes
    )
