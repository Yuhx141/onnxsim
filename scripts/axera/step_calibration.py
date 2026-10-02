"""Offline prediction of Pulsar2's per-tensor activation quantization for a
graph, so template coverage can be judged at a real calibration instead of
"covered if the zero points happen to match".

Pulsar2 7.0-lite (``calibration_method: MinMax``) quantizes activations per
tensor. Checked against the ``quant/quant_axmodel.json`` of our own template
builds (``validate`` below; ``docs/axera-step-real-calibration.md``):

* **Asymmetric uint8** (``quant_min`` 0) by default: the calibration range is
  widened to include 0, ``scale = f32((hi - lo) / 255)`` and
  ``zp = round(-lo / scale)``.
* **Symmetric int8** (``quant_min`` -128, zero point 0) for a MatMul/Gemm/
  live-weight Conv input: ``scale = max(|lo|, |hi|) / 127.5``. The producer's
  own output is int8 only when every consumer is such an input; otherwise it
  stays uint8 and the MatMul requantizes (``consumer_int8_scale``).
* **Passive ops** (Reshape, Transpose, Squeeze, Relu, MaxPool, ...) do not get
  their own parameters: their output is ``OVERLAPPED`` with their input, so
  input and output share one scale and zero point over the union of both
  ranges. A Relu that is its producer's only consumer fuses into it instead:
  both take the Relu output's range.

All of it matches every tensor of 435 of our builds
(``docs/axera-step-real-calibration.md``).

Ranges come from running the float graph over the calibration set with
onnxruntime (``collect_ranges``); onnxruntime is only needed for that step.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import onnx
import onnx.shape_inference

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Output shares the input's quantization (OVERLAPPED in quant_axmodel.json).
PASSIVE_OPS = frozenset(
    {
        "Reshape",
        "Transpose",
        "Squeeze",
        "Unsqueeze",
        "Flatten",
        "Relu",
        "MaxPool",
        "Slice",
        "Pad",
        "Identity",
    }
)
# Data inputs quantized symmetric int8 (AxQuantizedMatMul inputs are -128..127).
SYMMETRIC_CONSUMERS = frozenset({"MatMul", "Gemm", "Conv"})
# Outputs that are not quantized activations (bool / int results).
_UNQUANTIZED_OUTPUT_OPS = frozenset({"Greater", "Less", "Equal", "Shape"})


def qparams(lo: float, hi: float, symmetric: bool = False) -> tuple[float, int]:
    """``(scale, zero_point)`` MinMax gives the range ``[lo, hi]``."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if symmetric:
        s = float(np.float32(max(-lo, hi) / 127.5))
        return (s or 1.0), 0
    s = float(np.float32((hi - lo) / 255.0))
    if s == 0:
        return 1.0, 0
    return s, int(np.clip(np.round(-lo / s), 0, 255))


def _load_npy_tar(path: str) -> list[np.ndarray]:
    out = []
    with tarfile.open(path) as t:
        for m in sorted(t.getmembers(), key=lambda m: m.name):
            if m.isfile():
                out.append(np.load(io.BytesIO(t.extractfile(m).read())))
    return out


def load_pulsar2_dataset(config_path: str) -> dict[str, list[np.ndarray]]:
    """A Pulsar2 build config's ``quant.input_configs`` Numpy tars, by input
    name (paths resolved against the config's parent directory, as
    ``pulsar2 build`` does from its working directory)."""
    with open(config_path) as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    feeds = {}
    for c in cfg["quant"]["input_configs"]:
        if c.get("calibration_format", "Numpy") != "Numpy":
            raise ValueError(f"{c['tensor_name']}: only Numpy datasets are read")
        feeds[c["tensor_name"]] = _load_npy_tar(
            os.path.join(base, c["calibration_dataset"])
        )
    return feeds


def collect_ranges(
    model: onnx.ModelProto, feeds: Mapping[str, list[np.ndarray]]
) -> dict[str, tuple[float, float]]:
    """``{tensor: (min, max)}`` over every float tensor, across the samples.

    Runs the graph one node at a time (one small onnxruntime session per
    distinct node signature) in its own topological order and drops each
    tensor after its last use: the ResNet18 step's live set never exceeds
    1 GiB that way, while a whole-graph onnxruntime session of it passes 16 GiB.
    """
    import onnxruntime as ort
    from onnx import helper, numpy_helper, shape_inference

    m = shape_inference.infer_shapes(model)
    g = m.graph
    elem = {
        v.name: v.type.tensor_type.elem_type
        for v in list(g.value_info) + list(g.output) + list(g.input)
    }
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    last: dict[str, int] = {}
    for k, n in enumerate(g.node):
        for t in n.input:
            last[t] = k
    keep = {o.name for o in g.output}
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    sessions: dict[bytes, Any] = {}

    def session(n: onnx.NodeProto, args: list[np.ndarray]):
        ins = [
            helper.make_tensor_value_info(
                f"i{j}", helper.np_dtype_to_tensor_dtype(a.dtype), a.shape
            )
            for j, a in enumerate(args)
        ]
        node = onnx.NodeProto()
        node.CopyFrom(n)
        live = [t for t in n.input if t]
        del node.input[:]
        node.input.extend(f"i{live.index(t)}" if t else "" for t in n.input)
        del node.output[:]
        node.output.extend(f"o{j}" if t else "" for j, t in enumerate(n.output))
        node.name = "n"
        outs = [
            helper.make_tensor_value_info(f"o{j}", elem.get(t, 0), None)
            for j, t in enumerate(n.output)
            if t
        ]
        sig = (
            node.SerializeToString()
            + repr([(a.dtype.str, a.shape) for a in args]).encode()
        )
        if sig not in sessions:
            one = helper.make_model(
                helper.make_graph([node], "one", ins, outs),
                opset_imports=m.opset_import,
            )
            one.ir_version = m.ir_version
            sessions[sig] = ort.InferenceSession(
                one.SerializeToString(), so, providers=["CPUExecutionProvider"]
            )
        return sessions[sig]

    count = min(len(v) for v in feeds.values())
    ranges: dict[str, tuple[float, float]] = {}

    def add(name: str, v: np.ndarray) -> None:
        if v.dtype.kind != "f" or v.size == 0:
            return
        lo, hi = float(v.min()), float(v.max())
        old = ranges.get(name)
        ranges[name] = (min(old[0], lo), max(old[1], hi)) if old else (lo, hi)

    for i in range(count):
        env: dict[str, np.ndarray] = {}
        for k, v in feeds.items():
            env[k] = np.asarray(v[i])
            add(k, env[k])
        for k, n in enumerate(g.node):
            args = [env[t] if t in env else inits[t] for t in n.input if t]
            res = session(n, args).run(None, {f"i{j}": a for j, a in enumerate(args)})
            for t, v in zip([t for t in n.output if t], res):
                env[t] = v
                add(t, v)
            for t in set(n.input):
                if last.get(t) == k and t not in keep:
                    env.pop(t, None)
            for t in n.output:
                if t and t not in last and t not in keep:
                    env.pop(t, None)
    return ranges


class _Groups:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def _symmetric_inputs(node: onnx.NodeProto, consts: set[str]) -> list[str]:
    """Data inputs ``node`` requantizes to symmetric int8: both operands of a
    MatMul/Gemm/Conv whose second operand is live (a constant-weight Conv
    keeps a uint8 activation input)."""
    if node.op_type not in SYMMETRIC_CONSUMERS or len(node.input) < 2:
        return []
    if node.input[1] in consts:
        return []
    return [i for i in node.input[:2] if i and i not in consts]


def assign(
    model: onnx.ModelProto, ranges: Mapping[str, tuple[float, float]]
) -> dict[str, dict]:
    """``{tensor: {"scale", "zero_point", "signed"}}`` for every quantized
    activation of ``model`` (constants and non-float tensors are skipped).

    This is the tensor's own (producer-side) quantization. A MatMul/Gemm/live
    Conv input is requantized to symmetric int8 by that consumer; the tensor
    itself is symmetric only when every consumer of its passive chain is such
    an input (otherwise its producer stays uint8, e.g. a Mul feeding both a
    MatMul and a ReduceSum). A Relu that is its producer's only consumer fuses
    into it: both take the Relu output's range (zero point 0)."""
    g = model.graph
    consts = {i.name for i in g.initializer} | {
        n.output[0] for n in g.node if n.op_type == "Constant"
    }
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    graph_outputs = {o.name for o in g.output}
    groups = _Groups()
    passive_edges: set[tuple[int, str]] = set()
    fused_range: dict[str, str] = {}  # member tensor -> Relu output giving the range
    int8_memo: dict[str, bool] = {}

    def int8_only(t: str) -> bool:
        """Every use of ``t`` ends, through passive ops, in a MatMul-like
        symmetric input (and ``t`` is not a graph output)."""
        if t not in int8_memo:
            int8_memo[t] = False  # cycle guard
            uses = [t not in graph_outputs] + [
                t in _symmetric_inputs(c, consts)
                or (
                    c.op_type in PASSIVE_OPS
                    and c.input[0] == t
                    and all(int8_only(o) for o in c.output)
                )
                for c in consumers.get(t, [])
            ]
            int8_memo[t] = bool(consumers.get(t)) and all(uses)
        return int8_memo[t]

    for n in g.node:
        if n.op_type not in PASSIVE_OPS or not n.input or n.input[0] in consts:
            continue
        x, y = n.input[0], n.output[0]
        if x not in ranges or y not in ranges:
            continue
        if int8_only(y) and not int8_only(x):
            # A passive op whose output only feeds MatMuls while its input
            # has other uses (a residual Add, the optimizer update, a graph
            # output): Pulsar2 requantizes there. The output gets its own
            # symmetric int8 parameters over its own range; it does not
            # overlap its uint8 input (template builds with a side output,
            # docs/axera-matmul-step-templates.md).
            continue
        groups.union(y, x)
        passive_edges.add((id(n), x))
        p = producer.get(x)
        if (
            n.op_type == "Relu"
            and p is not None
            and p.op_type not in PASSIVE_OPS
            and len(consumers.get(x, [])) == 1
            and x not in graph_outputs
        ):
            fused_range[x] = y
    members: dict[str, list[str]] = {}
    for t in ranges:
        if t not in consts:
            members.setdefault(groups.find(t), []).append(t)
    symmetric: set[str] = set()
    for root, ts in members.items():
        uses = []
        for t in ts:
            if t in graph_outputs:
                uses.append(False)
            for c in consumers.get(t, []):
                if (id(c), t) in passive_edges:
                    continue
                uses.append(t in _symmetric_inputs(c, consts))
        if uses and all(uses):
            symmetric.add(root)
    span: dict[str, tuple[float, float]] = {}
    for root, ts in members.items():
        fused = [fused_range[t] for t in ts if t in fused_range]
        src = fused or ts
        span[root] = (min(ranges[t][0] for t in src), max(ranges[t][1] for t in src))
    unquantized = {
        o for n in g.node if n.op_type in _UNQUANTIZED_OUTPUT_OPS for o in n.output
    }
    out = {}
    for t in ranges:
        if t in consts or t in unquantized:
            continue
        r = groups.find(t)
        sym = r in symmetric
        s, z = qparams(*span[r], symmetric=sym)
        out[t] = {"scale": s, "zero_point": z, "signed": sym}
        # the symmetric int8 requantization a MatMul-like consumer applies
        if not sym and any(
            t in _symmetric_inputs(c, consts) for c in consumers.get(t, [])
        ):
            out[t]["consumer_int8_scale"] = qparams(*span[r], symmetric=True)[0]
    return out


def legalized(model: onnx.ModelProto) -> onnx.ModelProto:
    """``model`` in the form Pulsar2 compiles a training step in: live-weight
    Convs as per-tap MatMuls, live-operand Gemms as MatMul
    (``legalize.py``). Tensor names are the ones the MatMul step templates'
    manifests use."""
    import legalize

    m = onnx.ModelProto()
    m.CopyFrom(model)
    m = onnx.shape_inference.infer_shapes(m)
    legalize.act_weight_conv_to_matmul(m)
    legalize.gemm_to_matmul(m)
    return m


def calibrate(onnx_path: str, config_path: str, legalize: bool = True) -> dict:
    """The JSON ``coverage_report(..., calibration=)`` reads: every tensor's
    predicted quantization, over the legalized graph by default."""
    return calibrate_model(onnx.load(onnx_path), config_path, legalize, os.path.basename(onnx_path))


def calibrate_model(
    model: onnx.ModelProto,
    config_path: str,
    legalize: bool = True,
    name: str = "step.onnx",
) -> dict:
    if legalize:
        model = legalized(model)
    ranges = collect_ranges(model, load_pulsar2_dataset(config_path))
    return {
        "model": name,
        "rule": "pulsar2-7.0-lite MinMax (step_calibration.py)",
        "tensors": assign(model, ranges),
        "ranges": {k: list(v) for k, v in ranges.items()},
    }


def compare_to_pulsar2(
    model: onnx.ModelProto, ranges: Mapping[str, tuple[float, float]], quant: Mapping
) -> tuple[int, list[dict]]:
    """``(tensors compared, misses)``: ``assign(model, ranges)`` against a
    build's ``quant_axmodel.json``. A tensor's own quantization is the one its
    producer declares (for a graph input, any consumer's); an OVERLAPPED
    config follows its dominator, which carries the values."""
    pred = assign(model, ranges)
    producer = {o: n.name for n in model.graph.node for o in n.output}
    real: dict[str, list[tuple]] = {}
    for op, t in quant["tensor_configs"].items():
        for name, c in t.items():
            v = quant["values"].get(str(c.get("dominator", c["hash"]))) or {}
            if (
                name not in pred
                or not v.get("scale")
                or c["state"] in ("SOI", "BAKED", "FP32")
            ):
                continue
            if name in producer and producer[name] != op:
                continue
            q = (v["scale"][0], int(v["zero_point"][0]), c["quant_min"] < 0)
            real.setdefault(name, []).append(q)
    misses = []
    for name, qs in sorted(real.items()):
        p = pred[name]
        if not any(
            p["signed"] == q[2]
            and abs(p["scale"] - q[0]) <= 1e-4 * q[0]
            and abs(p["zero_point"] - q[1]) <= 1
            for q in qs
        ):
            misses.append(
                {
                    "tensor": name,
                    "predicted": p,
                    "pulsar2": [
                        dict(zip(("scale", "zero_point", "signed"), q)) for q in qs
                    ],
                }
            )
    return len(real), misses


def validate(build_dirs: Iterable[str]) -> dict:
    """``compare_to_pulsar2`` over builds laid out as ``<dir>/t.onnx``,
    ``<dir>/config/*.json``, ``<dir>/out/quant/quant_axmodel.json``."""
    tot = 0
    misses = []
    for d in build_dirs:
        cfgs = sorted(os.listdir(os.path.join(d, "config")))
        model = onnx.load(os.path.join(d, "t.onnx"))
        ranges = collect_ranges(
            model, load_pulsar2_dataset(os.path.join(d, "config", cfgs[0]))
        )
        with open(os.path.join(d, "out", "quant", "quant_axmodel.json")) as f:
            n, m = compare_to_pulsar2(model, ranges, json.load(f))
        tot += n
        misses += [{"build": os.path.basename(d.rstrip("/")), **x} for x in m]
    return {"tensors": tot, "match": tot - len(misses), "misses": misses}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="predict a graph's tensor qparams")
    c.add_argument("onnx")
    c.add_argument("config", help="pulsar2 build config (quant.input_configs)")
    c.add_argument("-o", "--output", required=True)
    c.add_argument("--no-legalize", action="store_true")
    v = sub.add_parser("validate", help="check the rule against real builds")
    v.add_argument("build_dirs", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "calibrate":
        with open(args.output, "w") as f:
            json.dump(
                calibrate(args.onnx, args.config, not args.no_legalize),
                f,
                sort_keys=True,
            )
        return 0
    res = validate(args.build_dirs)
    print(f"{res['match']}/{res['tensors']} tensors match")
    for m in res["misses"][:50]:
        print(json.dumps(m, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
