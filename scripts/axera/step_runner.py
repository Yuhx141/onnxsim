"""Run a whole training step with its covered nodes on the AX650 NPU.

``coverage_report`` says which nodes of a step graph our emitters can produce
at the step's calibration; this module actually executes the step that way:

* Every node ``plan_at_calibration`` covers becomes an **NPU segment**: its
  template retargeted to the predicted calibration (``step_calibration.py``)
  and run on the card. A covered live-operand MatMul runs its whole chain
  template (``matmul_record_emit``: the Gather/Mul/Reshape/Transpose feeding
  it) as one segment; Greater/Less -> Cast runs as its pair template.
* Everything else runs on the host, one node at a time in onnxruntime (a
  whole-graph session of the step passes 16 GiB; ``collect_ranges`` does the
  same).
* Every segment's input and output is float32: the templates quantize and
  dequantize inside, so segments pass plain float tensors.

Each NPU segment is also evaluated on the host as a **simulated** segment on
the same inputs: inputs fake-quantized at the segment's predicted input
parameters, the float ops, the output fake-quantized at its output
parameters. The device-vs-simulation difference, in output LSBs, is the
per-segment check that the emitted model computes what the template claims.

    python step_runner.py --mode npu --out report.json            # validation pass
    python step_runner.py --mode npu --validated report.json \
        --exclude '^(Softmax|Log|Neg)_' --host-optimizer --out final.json

``docs/axera-step-runner.md`` has the measured numbers.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import math
import os
import pickle
import re
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import elementwise_scale_emit as ew  # noqa: E402
import matmul_record_emit as mre  # noqa: E402
import misc_op_record_emit as misc  # noqa: E402
import reshape_record_emit as rre  # noqa: E402
import step_recalibrate  # noqa: E402
import tinygrad_ax_backend as axb  # noqa: E402

STEP_ONNX = "/home/takecheeze/npu-scratch/t6-r18fold/step.onnx"
STEP_REF = "/home/takecheeze/npu-scratch/t6-r18fold/step1_ref.pkl"
STEP_OPS = os.path.join(
    _HERE, "fixtures", "tinygrad_ax_backend", "resnet18_step_ops.json.gz"
)
STEP_CALIB = os.path.join(
    _HERE, "fixtures", "step_calibration", "resnet18_step_calibration.json.gz"
)
STEP_PRECISION_OVERRIDES = os.path.join(
    _HERE, "fixtures", "step_calibration", "resnet18_s16_overrides.json"
)
FP32_BINARY_FIXTURES = os.path.join(_HERE, "fixtures", "fp32_binary")
SAFE_MASKED_DIV_FIXTURE = os.path.join(
    _HERE, "fixtures", "safe_masked_div", "max_expand_div_1024x9x3136.axmodel.gz"
)
FP32_BINARY_SPEED_PROFILES = os.path.join(
    FP32_BINARY_FIXTURES, "native_mul_speed_profiles.json"
)
STEP_BINARY_FIXTURES = os.path.join(_HERE, "fixtures", "step_binary_templates")


# --------------------------------------------------------------------------
# quantization helpers


def qparams_of(calib: Mapping, name: str) -> tuple[float, int, bool]:
    q = calib["tensors"][name]
    return float(q["scale"]), int(q["zero_point"]), bool(q.get("signed"))


def load_step_precision_overrides(
    model: onnx.ModelProto,
    records: Sequence[Mapping],
    calib: Mapping,
    path: str = STEP_PRECISION_OVERRIDES,
) -> dict[str, dict]:
    """Load exact overrides and expand validated shared-scale ``lr * tensor`` templates."""
    with open(path) as stream:
        overrides = json.load(stream)
    if not isinstance(overrides, dict):
        raise ValueError("precision override JSON must map node names to calibration")
    index_path = os.path.join(_HERE, "fixtures", "binary_op_precision", "index.json")
    with open(index_path) as stream:
        index = json.load(stream)
    lr_templates = {
        tuple(entry["shape"]): entry
        for entry in index
        if entry.get("source", "").startswith(
            (
                "Pulsar2 7.0-lite S16 ResNet18 lr-times-tensor broadcast shared calibration",
                "Pulsar2 7.0-lite S16 ResNet18 lr-times-vector broadcast shared calibration",
            )
        )
    }
    shapes = {
        value.name: tuple(int(d.dim_value) for d in value.type.tensor_type.shape.dim)
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    records_by_name = {record["name"]: record for record in records}
    for node in model.graph.node:
        if node.op_type != "Mul" or not node.input or node.input[0] != "lr":
            continue
        record = records_by_name.get(node.name)
        if record is None or len(record.get("inputs", ())) != 2:
            continue
        shape = shapes.get(node.output[0], ())
        entry = lr_templates.get(shape)
        if entry is None and len(shape) == 1:
            entry = lr_templates.get((1, *shape))
        if entry is None:
            continue
        scales = dict(zip(("x", "z", "y"), entry["scales"]))
        zero_points = dict(zip(("x", "y", "z"), entry["zero_points"]))
        limit = 32767
        tensors = ("lr", node.input[1], node.output[0])
        roles = ("x", "z", "y")
        if any(
            tensor not in calib.get("ranges", {})
            or calib["ranges"][tensor][0] < -scales[role] * limit - scales[role]
            or calib["ranges"][tensor][1] > scales[role] * limit + scales[role]
            for tensor, role in zip(tensors, roles)
        ):
            continue
        overrides[node.name] = {
            "layer_precision": "S16",
            "scales": scales,
            "zero_points": zero_points,
        }
        if tuple(entry["shape"]) != shape:
            overrides[node.name]["template_shape"] = list(entry["shape"])
    return overrides


def fake_quant(
    x: np.ndarray, scale: float, zp: int, signed: bool, bits: int = 8
) -> np.ndarray:
    if bits not in (8, 16):
        raise ValueError(f"unsupported fake-quant precision: {bits}")
    lo, hi = (
        (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    )
    s = np.float32(scale)
    q = np.clip(np.rint(x.astype(np.float32) / s) + zp, lo, hi)
    return ((q - zp) * s).astype(np.float32)


# --------------------------------------------------------------------------
# host execution: one onnxruntime session per distinct node signature


class HostOps:
    def __init__(self, model: onnx.ModelProto):
        import onnxruntime as ort

        self._ort = ort
        self.model = model
        self.elem = {
            v.name: v.type.tensor_type.elem_type
            for v in list(model.graph.value_info)
            + list(model.graph.output)
            + list(model.graph.input)
        }
        self.inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        self.so = ort.SessionOptions()
        self.so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.so.log_severity_level = 3
        self.so.intra_op_num_threads = 8
        self.sessions: dict[bytes, Any] = {}

    def _session(self, n: onnx.NodeProto, args: Sequence[np.ndarray]):
        node = onnx.NodeProto()
        node.CopyFrom(n)
        live = [t for t in n.input if t]
        del node.input[:]
        node.input.extend(f"i{live.index(t)}" if t else "" for t in n.input)
        del node.output[:]
        node.output.extend(f"o{j}" if t else "" for j, t in enumerate(n.output))
        node.name = "n"
        sig = (
            node.SerializeToString()
            + repr([(a.dtype.str, a.shape) for a in args]).encode()
        )
        if sig not in self.sessions:
            ins = [
                helper.make_tensor_value_info(
                    f"i{j}", helper.np_dtype_to_tensor_dtype(a.dtype), a.shape
                )
                for j, a in enumerate(args)
            ]
            outs = [
                helper.make_tensor_value_info(f"o{j}", self.elem.get(t, 0), None)
                for j, t in enumerate(n.output)
                if t
            ]
            one = helper.make_model(
                helper.make_graph([node], "one", ins, outs),
                opset_imports=self.model.opset_import,
            )
            one.ir_version = self.model.ir_version
            self.sessions[sig] = self._ort.InferenceSession(
                one.SerializeToString(), self.so, providers=["CPUExecutionProvider"]
            )
        return self.sessions[sig]

    def run(self, n: onnx.NodeProto, env: Mapping[str, np.ndarray]) -> list:
        args = [env[t] if t in env else self.inits[t] for t in n.input if t]
        return self._session(n, args).run(
            None, {f"i{j}": a for j, a in enumerate(args)}
        )


# --------------------------------------------------------------------------
# the plan: which nodes form which NPU segment


@dataclasses.dataclass
class Segment:
    name: str
    kind: str  # emitter family
    nodes: list[str]  # step node names the template computes
    inputs: list[str]  # step tensors, in the template's input order
    outputs: list[str]  # step tensors, in the template's output order
    detail: str
    emit: Callable[[], onnx.ModelProto] = dataclasses.field(repr=False)
    in_q: list[tuple[float, int, bool] | None] = dataclasses.field(default_factory=list)
    out_q: list[tuple[float, int, bool] | None] = dataclasses.field(default_factory=list)
    unsafe: str = ""  # why the template's semantics differ from the node's
    # A template built at batch N/k runs k times on batch slices; ``split``
    # marks the inputs that carry the batch axis (step_template batch_split).
    batch_split: int = 1
    split: list[bool] = dataclasses.field(default_factory=list)
    input_shapes: list[tuple[int, ...]] = dataclasses.field(default_factory=list)
    output_shape: tuple[int, ...] = ()
    constant_inputs: list[str] = dataclasses.field(default_factory=list)
    # Shape decompositions may run a replicated lane template and retain only
    # the prefix corresponding to the graph output.
    output_take: int | None = None
    # Calibration-exact scalar folds are guarded at runtime; an input outside
    # the singleton calibration point is evaluated by the normal host path.
    constant_value: np.ndarray | None = dataclasses.field(default=None, repr=False)
    constant_guard: tuple[str, np.ndarray] | None = dataclasses.field(
        default=None, repr=False
    )
    nan_guard: bool = False
    input_transforms: dict[str, Callable[[np.ndarray], np.ndarray]] = dataclasses.field(
        default_factory=dict, repr=False
    )
    output_transform: Callable[[np.ndarray], np.ndarray] | None = dataclasses.field(
        default=None, repr=False
    )
    profiled_faster: bool = False
    prefer_fp32: bool = False
    quantize_device_io: bool = False


_RETARGET_KEY = re.compile(r"retarget of (\S+) \(")
_FUSED_KEY = re.compile(r"retarget of the fused chain (\S+) \(")
_CLASS = re.compile(r"\((x\d+,y\d+(?:,z\d+)?)\)")


def _scale_dict(calib, names: Mapping[str, str]) -> tuple[dict, dict]:
    sc, zp = {}, {}
    for role, t in names.items():
        s, z, _ = qparams_of(calib, t)
        sc[role], zp[role] = s, z
    return sc, zp


def _dim0(model: onnx.ModelProto, name: str) -> int | None:
    for vi in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if vi.name == name and vi.type.tensor_type.shape.dim:
            return vi.type.tensor_type.shape.dim[0].dim_value
    return None


def _segment_for(
    rec: Mapping,
    detail: str,
    calib: Mapping,
    model: onnx.ModelProto,
    consumers: Mapping[str, list[onnx.NodeProto]],
    inits: Mapping[str, np.ndarray],
) -> Segment | None:
    op, name = rec["op"], rec["name"]
    ins, outs = list(rec["inputs"]), list(rec["outputs"])
    attrs = rec.get("attrs", {})

    def q(ts):
        return [qparams_of(calib, t) for t in ts]

    # Exact algebraic reduction for optimizer masks that are literally all
    # ones.  The live and output quantization records must match, otherwise
    # removing the binary would change the graph's quantization boundary.
    if (
        op == "Mul"
        and rec.get("attrs", {}).get("form") == "const"
        and rec.get("attrs", {}).get("constant_input") is not None
    ):
        const_index = int(rec["attrs"]["constant_input"])
        const_name = ins[const_index]
        live_name = ins[1 - const_index]
        value = inits.get(const_name)
        if (
            value is not None
            and np.all(np.asarray(value) == 1)
            and qparams_of(calib, live_name) == qparams_of(calib, outs[0])
        ):
            return Segment(
                name,
                "algebraic_identity",
                [name],
                [live_name],
                outs,
                "Mul by an all-ones initializer reduced to Identity",
                lambda: None,
                q([live_name]),
                q(outs),
            )

    if detail.startswith("matmul_record_emit.recalibrate"):
        entry = mre.step_template(name)
        tmpl = mre.load_model(entry["axmodel"])
        inv = {v: k for k, v in entry["names"].items()}
        # Template-only tensors (docs/axera-matmul-step-templates.md): a
        # ``__pre`` input feeds a Relu whose output is the step tensor (Relu
        # is idempotent, so the step's Relu output goes in); ``__side``
        # outputs only keep an input uint8 and are not step outputs.
        aliases = entry.get("aliases", {})
        for a, src in aliases.items():
            if a.endswith("__pre") and src in inv:
                inv[a] = inv[src]
        t_in = [inv[i.name] for i in tmpl.graph.input]
        t_out = [inv[o.name] for o in tmpl.graph.output if o.name not in aliases]
        # A ``__pre`` input stands for its step tensor (the template's Relu
        # is template-only), so that tensor is an input, not computed here.
        t_inputs = {i.name for i in tmpl.graph.input}
        t_inputs |= {src for a, src in aliases.items() if a in t_inputs}
        internal = [s for s, t in entry["names"].items() if t not in t_inputs]
        producers = {o: n.name for n in model.graph.node for o in n.output}
        nodes = sorted({producers[t] for t in internal if t in producers})

        def emit_mm():
            old = mre.load_scales(entry["quant"])
            real = {}
            for step_name, tname in entry["names"].items():
                qq = calib["tensors"][step_name]
                if "consumer_int8_scale" in qq and tname in old and old[tname][1] == 0:
                    real[step_name] = (qq["consumer_int8_scale"], 0.0)
                else:
                    real[step_name] = (qq["scale"], float(qq["zero_point"]))
                if "consumer_int8_scale" in qq and tname + mre.I8 in old:
                    real[step_name + mre.I8] = (qq["consumer_int8_scale"], 0.0)
            new = mre.step_node_scales(entry, old, real)
            out, _ = mre.recalibrate(tmpl, old, new)
            return out

        in_q = []
        old = mre.load_scales(entry["quant"])
        for t in t_in:
            qq = calib["tensors"][t]
            tn = entry["names"].get(t, "")
            if "consumer_int8_scale" in qq and tn in old and old[tn][1] == 0:
                in_q.append((float(qq["consumer_int8_scale"]), 0, True))
            else:
                in_q.append(qparams_of(calib, t))
        k = entry.get("batch_split", 1)
        split = [
            k > 1 and _dim0(model, t) == _dim0(tmpl, i.name) * k
            for t, i in zip(t_in, tmpl.graph.input)
        ]
        return Segment(
            name, "matmul_chain", nodes, t_in, t_out, detail, emit_mm, in_q, q(t_out),
            batch_split=k, split=split,
        )  # fmt: skip

    if op in ("Greater", "Less"):
        key = rec["attrs"]["misc_key"]
        cast = [c for c in consumers.get(outs[0], []) if c.op_type == "Cast"]
        if len(cast) != 1:
            return None
        cast_out = cast[0].output[0]
        subtract = [c for c in consumers.get(cast_out, []) if c.op_type == "Sub"]
        if len(subtract) == 1:
            sub = subtract[0]
            if (
                len(sub.input) == 2
                and sub.input[1] == cast_out
                and sub.input[0] in inits
                and np.all(np.asarray(inits[sub.input[0]]) == 1)
            ):
                shape = rec.get("shapes", [[]])[0]
                inverse_key = misc.template_key(
                    "GreaterOrEqualCast" if op == "Greater" else "LessOrEqualCast",
                    shape,
                )
                misc.load_template(inverse_key)
                return Segment(
                    name,
                    "compare_complement",
                    [name, cast[0].name, sub.name],
                    [ins[1], ins[0]],
                    [sub.output[0]],
                    f"{inverse_key}: reversed inclusive comparison replaces 1-Cast({op})",
                    lambda: misc.emit_model(inverse_key),
                    nan_guard=True,
                )
        tmpl, _ = misc.load_template(key)
        live = [t for t in ins if t not in inits][: len(tmpl.graph.input)]
        # not quantized: the device compares the float input directly
        return Segment(
            name, "compare_cast", [name, cast[0].name], live, [cast[0].output[0]],
            detail, lambda: misc.emit_model(key), [], [],
        )  # fmt: skip
    if op == "Cast":
        return None  # served by its Greater/Less pair segment

    fused = _FUSED_KEY.search(detail)
    if fused and op == "Reshape":
        # bias flatten: one program with its producing ReduceSum (#1908)
        key = fused.group(1)
        src = rec["attrs"]["fused_input"]
        producer = [
            c
            for c in consumers.get(src, [])
            if c.op_type == "ReduceSum" and ins[0] in c.output
        ]
        if len(producer) != 1:
            return None
        sc, zp = _scale_dict(calib, {"x": src, "y": outs[0]})

        def emit_fused():
            return misc.emit_model(key, sc, zp)

        return Segment(
            name, "reducesum_flatten", [producer[0].name, name], [src], outs, detail,
            emit_fused, q([src]), q(outs),
        )  # fmt: skip

    m = _RETARGET_KEY.search(detail)
    if m and detail.startswith("misc_op_record_emit"):
        key = m.group(1)
        sc, zp = _scale_dict(calib, {"x": ins[0], "y": outs[0]})

        def emit_misc():
            return misc.emit_model(key, sc, zp)

        return Segment(
            name, "misc", [name], [ins[0]], outs, detail, emit_misc, q(ins[:1]), q(outs)
        )

    if op == "Relu" and detail.startswith("Relu record retarget"):
        s, z, _ = qparams_of(calib, ins[0])
        shape = rec["shapes"][0]

        def emit_relu():
            tm, _ = ew.load_template("Relu", shape, {"x": 128, "y": 128})
            mc = ew.retarget_relu_records(
                bytes(ew._mcode_initializer(tm).raw_data), s, z
            )
            return step_recalibrate.with_mcode(tm, mc)

        return Segment(
            name, "relu", [name], ins, outs, detail, emit_relu, q(ins), q(outs)
        )

    if detail.startswith("ElementwiseScaleEdit"):
        cls = _CLASS.search(detail).group(1)
        template_rec = rec
        output_take = None
        if "shape-expanded" in detail:
            template_rec = dict(rec)
            template_rec["shapes"] = [[1, 128]]
            output_take = 1
        key = axb.key_for_record(template_rec, cls)
        constant_inputs = []
        if op in ew.OPS:
            sc, _ = _scale_dict(calib, {"x": ins[0], "y": outs[0]})
            live = ins[:1]
        else:
            const_index = rec.get("attrs", {}).get("constant_input")
            if (
                rec.get("attrs", {}).get("form") == "const"
                and const_index is not None
                and ins[const_index] in inits
            ):
                const_name = ins[const_index]
                live_index = 1 - int(const_index)
                z = int(re.search(r",z(\d+)\)?", detail).group(1))
                value = np.asarray(inits[const_name], dtype=np.float32)
                bound = max(float(np.max(np.abs(value))), np.finfo(np.float32).tiny)
                sc, _ = _scale_dict(calib, {"x": ins[live_index], "y": outs[0]})
                sc["z"] = bound / (255.0 if z == 0 else 127.0)
                live = [ins[live_index]]
                constant_inputs = [const_name]
            else:
                sc, _ = _scale_dict(calib, {"x": ins[0], "z": ins[1], "y": outs[0]})
                live = ins[:2]

        # ``op_values`` is the inexpensive part of the binary emitter's
        # validation. Reject scale-collision cases during planning instead of
        # advertising a segment that only fails when the VM loads it: a
        # collision is a distinct Pulsar2 program family, not a retargetable
        # instance of the selected template.
        template_only = False
        if op in axb.bse.OPS:
            try:
                values = axb.bse.op_values(op, sc)
                axb.bse._check_distinct(values, "target")
            except ValueError:
                # A binary 0/1 mask can have the same input and output
                # scale. That is a compiler collision for scale retargeting,
                # but it is safe to use the native template unchanged when
                # the flat frame was built at exactly that calibration.
                exact_template = False
                if attrs.get("tile_blocks") and op == "Mul":
                    entry = axb.TemplateCache().lookup(key)
                    exact_template = all(
                        abs(float(sc[name]) - float(entry.meta["scales"][name])) < 1e-7
                        for name in ("x", "z", "y")
                    )
                flat_mask = (
                    attrs.get("flat_blocks")
                    and op == "Mul"
                    and cls == "x0,y0,z0"
                    and float(sc["x"]) == float(sc["y"])
                    and abs(float(sc["z"]) - 1.0 / 255.0) < 1e-7
                )
                if not (flat_mask or exact_template):
                    return None
                template_only = True

        def emit_ew():
            edit = (
                axb.BinaryTemplateOnly()
                if template_only
                else axb.ElementwiseScaleEdit(sc)
            )
            return axb.EditSet([edit]).build(key)

        input_transforms = {}
        output_transform = None
        batch_split = 1
        split = []
        tile_match = re.search(r"tiled-(\d+)x\1", detail)
        flat_match = re.search(r"flat-(\d+)x(\d+)", detail)
        if tile_match:
            tile_side = int(tile_match.group(1))
            original_shape = tuple(
                int(d.dim_value)
                for d in next(
                    v
                    for v in (*model.graph.value_info, *model.graph.output)
                    if v.name == outs[0]
                ).type.tensor_type.shape.dim
            )
            if attrs.get("tile_layout") == "nchw":
                b, channels, height, width = original_shape
                tiles_y = height // tile_side
                tiles_x = width // tile_side
                batch_split = tiles_y * tiles_x

                def tile_input(value, shape=original_shape):
                    expanded = np.broadcast_to(np.asarray(value), shape)
                    x = expanded.reshape(b, channels, height, width)
                    return (
                        x.reshape(b, channels, tiles_y, tile_side, tiles_x, tile_side)
                        .transpose(0, 2, 4, 1, 3, 5)
                        .reshape(b * tiles_y * tiles_x, channels, tile_side, tile_side)
                    )

                def untile_output(value):
                    x = np.asarray(value).reshape(
                        b, tiles_y, tiles_x, channels, tile_side, tile_side
                    )
                    return x.transpose(0, 3, 1, 4, 2, 5).reshape(original_shape)
            else:
                b, _, channels, pixels = original_shape
                side = math.isqrt(pixels)
                tiles = side // tile_side
                batch_split = tiles * tiles

                def tile_input(value, shape=original_shape):
                    expanded = np.broadcast_to(np.asarray(value), shape)
                    x = expanded[:, 0].reshape(b, channels, side, side)
                    return (
                        x.reshape(b, channels, tiles, tile_side, tiles, tile_side)
                        .transpose(0, 2, 4, 1, 3, 5)
                        .reshape(b * tiles * tiles, channels, tile_side, tile_side)
                    )

                def untile_output(value):
                    x = np.asarray(value).reshape(
                        b, tiles, tiles, channels, tile_side, tile_side
                    )
                    return x.transpose(0, 3, 1, 4, 2, 5).reshape(b, 1, channels, pixels)

            for tensor in live:
                input_transforms[tensor] = tile_input
            for tensor in constant_inputs:
                input_transforms[tensor] = tile_input
            output_transform = untile_output
            output_shape = ()
            split = [True] * (len(live) + len(constant_inputs))
        elif flat_match:
            rows, cols = (int(x) for x in flat_match.groups())
            original_shape = tuple(
                int(d.dim_value)
                for d in next(
                    v
                    for v in (*model.graph.value_info, *model.graph.output)
                    if v.name == outs[0]
                ).type.tensor_type.shape.dim
            )
            elements = int(np.prod(original_shape))
            chunk = rows * cols
            blocks = (elements + chunk - 1) // chunk
            batch_split = blocks

            def pack_input(value, shape=original_shape):
                flat = np.broadcast_to(np.asarray(value), shape).reshape(-1)
                padded = np.zeros(blocks * chunk, dtype=np.float32)
                padded[: flat.size] = flat
                return padded.reshape(blocks * rows, cols)

            def unpack_output(value):
                return np.asarray(value).reshape(-1)[:elements].reshape(original_shape)

            for tensor in live:
                input_transforms[tensor] = pack_input
            for tensor in constant_inputs:
                input_transforms[tensor] = pack_input
            output_transform = unpack_output
            output_shape = ()
            split = [True] * (len(live) + len(constant_inputs))

        input_shapes = [
            tuple(int(d) for d in s)
            for s in rec.get("attrs", {}).get("input_shapes", [])
        ]
        output_shape = tuple(
            int(d) for d in rec.get("attrs", {}).get("output_shape", ())
        )
        return Segment(
            name,
            "elementwise",
            [name],
            live,
            outs,
            detail,
            emit_ew,
            q(live),
            q(outs),
            input_shapes=input_shapes,
            output_shape=output_shape,
            constant_inputs=constant_inputs,
            output_take=output_take,
            batch_split=batch_split,
            split=split,
            input_transforms=input_transforms,
            output_transform=output_transform,
        )

    if op in ("Reshape", "Squeeze") and detail.startswith("reshape_record_emit"):
        s, z, _ = qparams_of(calib, ins[0])
        in_shape = rec["shapes"][0]
        out_shape = rec["attrs"].get("out", [])

        def emit_reshape():
            return rre.emit_step_reshape(in_shape, out_shape, s, z)

        seg = Segment(
            name,
            "reshape",
            [name],
            ins[:1],
            outs,
            detail,
            emit_reshape,
            q(ins[:1]),
            q(outs),
        )
        # a signed input (nonzero zero point) is emitted from the Reshape ->
        # Identity template; the Reshape -> Relu ones (#1891) would clip it
        return seg

    if detail == "GatherIndexEdit":
        key = axb.key_for_record(rec)
        idx = [int(i) for i in np.asarray(inits[ins[1]]).ravel()]
        s, z, _ = qparams_of(calib, ins[0])

        def emit_gather():
            # GatherIndexEdit keeps the template's own calibration; a Gather is
            # passive (its output shares the input's quantization), so move the
            # template's one (1/s, s, zero point) to the step's like a Reshape.
            gm = axb.EditSet([axb.GatherIndexEdit(idx)]).build(key)
            mc = rre.retarget_scale(
                bytes(axb._mcode_initializer(gm).raw_data),
                s,
                z,
                zp_regs=rre.GATHER_ZP_REGS,
            )
            return step_recalibrate.with_mcode(gm, mc)

        seg = Segment(
            name,
            "gather",
            [name],
            ins[:1],
            outs,
            detail,
            emit_gather,
            q(ins[:1]),
            q(outs),
        )
        if z == 0:
            seg.unsafe = "a zero point of 0 is a different Gather program (no template)"
        return seg

    if detail == "TemplateOnly" and op == "Transpose":
        key = axb.key_for_record(rec)

        def emit_transpose():
            return axb.EditSet([axb.TemplateOnly()]).build(key)

        return Segment(
            name, "transpose", [name], ins[:1], outs, detail, emit_transpose, [], []
        )
    return None


def _fp32_binary_segment_for(rec: Mapping, model: onnx.ModelProto) -> Segment | None:
    """Use a captured, unquantized binary template for an exact shape tuple."""
    op = rec.get("op")
    if op not in ("Add", "Sub", "Mul", "Div") or len(rec.get("inputs", ())) != 2:
        return None
    if len(rec.get("outputs", ())) != 1:
        return None
    by_name = {
        v.name: tuple(int(d.dim_value) for d in v.type.tensor_type.shape.dim)
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    by_name.update(
        {t.name: tuple(int(d) for d in t.dims) for t in model.graph.initializer}
    )
    inputs, output = list(rec["inputs"]), rec["outputs"][0]
    if any(t not in by_name for t in (*inputs, output)):
        return None
    input_shapes = [list(by_name[t]) for t in inputs]
    output_shape = list(by_name[output])
    path = None
    template_inputs = inputs
    profiled_faster = False
    prefer_fp32 = False
    index_path = os.path.join(FP32_BINARY_FIXTURES, "index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as stream:
            entries = json.load(stream).get("templates", [])
        entry = next(
            (
                item
                for item in entries
                if item.get("validated") is True
                and item.get("op") == op
                and item.get("input_shapes") == input_shapes
                and item.get("output_shape") == output_shape
            ),
            None,
        )
        if entry is not None:
            path = os.path.join(FP32_BINARY_FIXTURES, entry["file"])
            prefer_fp32 = rec["name"] in entry.get("prefer_fp32_nodes", ())
        # Mul is commutative. For the measured native-fast broadcast cases,
        # the source graph presents a [1] constant before the full tensor,
        # while the captured FP32 model uses full-tensor then [1]. Preserve
        # the compiled model's input order at the segment boundary.
        if path is None and op == "Mul" and input_shapes == [[1], output_shape]:
            try:
                with open(FP32_BINARY_SPEED_PROFILES, encoding="utf-8") as stream:
                    speed_profiles = json.load(stream)
            except FileNotFoundError:
                speed_profiles = {}
            profile = next(
                (
                    item
                    for item in speed_profiles.get("profiles", [])
                    if item.get("op") == op
                    and item.get("source_input_shapes") == input_shapes
                    and item.get("output_shape") == output_shape
                    and float(item.get("median_speedup", 0.0))
                    >= float(speed_profiles.get("minimum_speedup", 1.10))
                ),
                None,
            )
            if profile is not None:
                entry = next(
                    (
                        item
                        for item in entries
                        if item.get("validated") is True
                        and item.get("op") == op
                        and item.get("input_shapes")
                        == profile.get("template_input_shapes")
                        and item.get("output_shape") == output_shape
                    ),
                    None,
                )
                if entry is not None:
                    path = os.path.join(FP32_BINARY_FIXTURES, entry["file"])
                    template_inputs = [inputs[1], inputs[0]]
                    profiled_faster = True
                    prefer_fp32 = rec["name"] in entry.get("prefer_fp32_nodes", ())
    # Preserve the initial one-off Add capture as a compatible legacy entry.
    if (
        path is None
        and op == "Add"
        and input_shapes == [[16, 1000], [16, 1000]]
        and output_shape == [16, 1000]
    ):
        path = os.path.join(FP32_BINARY_FIXTURES, "add_16x1000.axmodel.gz")
    if path is None or not os.path.isfile(path):
        return None
    node = next((n for n in model.graph.node if n.name == rec["name"]), None)
    if node is None or node.op_type != op or list(node.input) != inputs:
        return None
    # Restrict this entry point to ordinary float32 tensors. In particular,
    # don't route integer/bool ONNX binaries through an FP32 template.
    elem_types = {
        v.name: v.type.tensor_type.elem_type
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    elem_types.update({t.name: t.data_type for t in model.graph.initializer})
    if any(elem_types.get(t) != onnx.TensorProto.FLOAT for t in (*inputs, output)):
        return None

    def emit_fp32():
        with gzip.open(path, "rb") as f:
            return onnx.load_model_from_string(f.read())

    constants = {item.name for item in model.graph.initializer}
    return Segment(
        rec["name"],
        "fp32_binary",
        [rec["name"]],
        template_inputs,
        [output],
        f"Pulsar2 FP32 {op} exact template for {input_shapes} -> {output_shape}"
        + (" (AX8850-profiled faster broadcast Mul)" if profiled_faster else ""),
        emit_fp32,
        [],
        [],
        constant_inputs=[t for t in template_inputs if t in constants],
        profiled_faster=profiled_faster,
        prefer_fp32=prefer_fp32,
    )


def _safe_masked_div_segment_for(
    rec: Mapping, model: onnx.ModelProto
) -> Segment | None:
    """Use the device-validated zero-safe normalization for the crop mask.

    This exact step node divides a nonnegative mask by its ReduceSum count.
    Floating-point execution guarantees a count in [1, 9], while quantized
    comparison inputs can produce an empty mask on device. The native model
    expands the count, clamps it to one, then divides, making an empty mask
    map to zero without relying on implicit broadcast in AxDiv.
    """
    if rec.get("name") != "Div_453" or rec.get("op") != "Div":
        return None
    inputs, outputs = list(rec.get("inputs", ())), list(rec.get("outputs", ()))
    if len(inputs) != 2 or len(outputs) != 1 or not os.path.isfile(SAFE_MASKED_DIV_FIXTURE):
        return None
    producer = next(
        (node for node in model.graph.node if inputs[1] in node.output), None
    )
    if (
        producer is None
        or producer.op_type != "ReduceSum"
        or not producer.input
        or producer.input[0] != inputs[0]
    ):
        return None
    shapes = {
        value.name: tuple(int(d.dim_value) for d in value.type.tensor_type.shape.dim)
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    if (
        shapes.get(inputs[0]) != (1024, 9, 3136)
        or shapes.get(inputs[1]) != (1024, 1, 3136)
        or shapes.get(outputs[0]) != (1024, 9, 3136)
    ):
        return None

    def emit_safe_div():
        with gzip.open(SAFE_MASKED_DIV_FIXTURE, "rb") as stream:
            return onnx.load_model_from_string(stream.read())

    return Segment(
        rec["name"],
        "safe_masked_div",
        [rec["name"]],
        inputs,
        outputs,
        "FP32 Expand -> Max(count, 1) -> Div; zero mask count maps to zero",
        emit_safe_div,
        [None, None],
        [None],
        input_shapes=[shapes[inputs[0]], shapes[inputs[1]]],
    )


def _exact_mask_mul_segment_for(
    rec: Mapping, calib: Mapping, model: onnx.ModelProto
) -> Segment | None:
    """Exact-calibration native Mul for a common one-hot mask frame.

    The frame is calibrated so x and y share a scale (making the scale
    retarget registers ambiguous); only this exact measured scale/zp tuple is
    accepted. Other masks still use the regular retargetable templates.
    """
    if rec.get("op") != "Mul" or rec.get("attrs", {}).get("form") != "same_shape":
        return None
    inputs, outputs = list(rec.get("inputs", ())), list(rec.get("outputs", ()))
    if len(inputs) != 2 or len(outputs) != 1:
        return None
    shape = (16, 1000)
    by_name = {
        v.name: tuple(int(d.dim_value) for d in v.type.tensor_type.shape.dim)
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    if any(by_name.get(t) != shape for t in (*inputs, outputs[0])):
        return None
    expected = {
        inputs[0]: (0.00012254902685526758, 255),
        inputs[1]: (0.003921568859368563, 0),
        outputs[0]: (0.00012254902685526758, 255),
    }
    for tensor, (scale, zp) in expected.items():
        if tensor not in calib.get("tensors", {}):
            return None
        actual = calib["tensors"][tensor]
        if (
            bool(actual.get("signed"))
            or int(actual["zero_point"]) != zp
            or abs(float(actual["scale"]) - scale) > 1e-10
        ):
            return None
    path = os.path.join(STEP_BINARY_FIXTURES, "mul_mask_16x1000.axmodel.gz")
    if not os.path.isfile(path):
        return None

    def emit_mask_mul():
        with gzip.open(path, "rb") as f:
            return onnx.load_model_from_string(f.read())

    return Segment(
        rec["name"],
        "mul_mask_exact",
        [rec["name"]],
        inputs,
        outputs,
        "native Mul mask template at exact x255/y255/z0 calibration, shape 16x1000",
        emit_mask_mul,
        [qparams_of(calib, t) for t in inputs],
        [qparams_of(calib, outputs[0])],
    )


def _exact_loss_sub_segment_for(
    rec: Mapping, calib: Mapping, model: onnx.ModelProto
) -> Segment | None:
    """Exact-calibration native Sub for the loss-tail broadcast at 16x1000."""
    if rec.get("op") != "Sub" or len(rec.get("inputs", ())) != 2:
        return None
    inputs, outputs = list(rec["inputs"]), list(rec.get("outputs", ()))
    if len(outputs) != 1:
        return None
    shapes = {
        v.name: tuple(int(d.dim_value) for d in v.type.tensor_type.shape.dim)
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    output_shape = (16, 1000)
    if (
        shapes.get(inputs[0]) != output_shape
        or shapes.get(inputs[1]) != (16, 1)
        or shapes.get(outputs[0]) != output_shape
    ):
        return None
    expected = (
        (6.96580696105957, 255),
        (0.00012254902685526758, 255),
        (6.96580696105957, 255),
    )
    for tensor, (scale, zp) in zip((*inputs, outputs[0]), expected):
        actual = calib.get("tensors", {}).get(tensor)
        if (
            actual is None
            or bool(actual.get("signed"))
            or int(actual["zero_point"]) != zp
            or abs(float(actual["scale"]) - scale) > 1e-10
        ):
            return None
    path = os.path.join(STEP_BINARY_FIXTURES, "sub_loss_16x1000.axmodel.gz")
    if not os.path.isfile(path):
        return None

    def emit_loss_sub():
        with gzip.open(path, "rb") as f:
            return onnx.load_model_from_string(f.read())

    return Segment(
        rec["name"],
        "sub_loss_exact",
        [rec["name"]],
        inputs,
        outputs,
        "native Sub loss-tail template at exact calibration, broadcast to 16x1000",
        emit_loss_sub,
        [qparams_of(calib, t) for t in inputs],
        [qparams_of(calib, outputs[0])],
        input_shapes=[shapes[t] for t in inputs],
        output_shape=output_shape,
    )


def _exact_div2_segment_for(
    rec: Mapping,
    calib: Mapping,
    model: onnx.ModelProto,
    inits: Mapping[str, np.ndarray],
) -> Segment | None:
    """Native Pulsar2 lowering for exact ``x / 2`` loss-tail templates."""
    if rec.get("op") != "Div" or rec.get("attrs", {}).get("form") != "const":
        return None
    inputs, outputs = list(rec.get("inputs", ())), list(rec.get("outputs", ()))
    if len(inputs) != 2 or len(outputs) != 1:
        return None
    const_index = rec.get("attrs", {}).get("constant_input")
    if const_index is None or int(const_index) != 1:
        return None
    const_name, live_name, output = inputs[1], inputs[0], outputs[0]
    if const_name not in inits or not np.all(np.asarray(inits[const_name]) == 2.0):
        return None
    shapes = {
        v.name: tuple(int(d.dim_value) for d in v.type.tensor_type.shape.dim)
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    shape = (16, 1000)
    if shapes.get(live_name) != shape or shapes.get(output) != shape:
        return None
    qx = calib.get("tensors", {}).get(live_name)
    qy = calib.get("tensors", {}).get(output)
    if qx is None or qy is None or qx.get("signed") or qy.get("signed"):
        return None
    zp = int(qx["zero_point"])
    if int(qy["zero_point"]) != zp:
        return None
    scales = {
        98: (0.05171579495072365, 0.025857897475361824),
        101: (0.0531538687646389, 0.02657693438231945),
        211: (2.16483895201236e-05, 1.08241947600618e-05),
    }
    expected = scales.get(zp)
    if expected is None:
        return None
    sx, sy = expected
    if abs(float(qx["scale"]) - sx) > 1e-10 or abs(float(qy["scale"]) - sy) > 1e-10:
        return None
    path = os.path.join(STEP_BINARY_FIXTURES, f"div2_16x1000_z{zp}.axmodel.gz")
    if not os.path.isfile(path):
        return None

    def emit_div2():
        with gzip.open(path, "rb") as f:
            return onnx.load_model_from_string(f.read())

    return Segment(
        rec["name"],
        "div2_exact",
        [rec["name"]],
        [live_name],
        [output],
        f"native Pulsar2 x/2 template at exact zp{zp} calibration, shape 16x1000",
        emit_div2,
        [qparams_of(calib, live_name)],
        [qparams_of(calib, output)],
    )


def _singleton_scalar_div_fold(
    rec: Mapping, calib: Mapping, inits: Mapping[str, np.ndarray]
) -> Segment | None:
    """Fold scalar constant/live Div only at a calibrated singleton input."""
    if rec.get("op") != "Div" or rec.get("attrs", {}).get("constant_input") != 0:
        return None
    inputs, outputs = list(rec.get("inputs", ())), list(rec.get("outputs", ()))
    if len(inputs) != 2 or len(outputs) != 1 or inputs[0] not in inits:
        return None
    ranges = calib.get("ranges", {})
    bounds = ranges.get(inputs[1])
    if not bounds or len(bounds) != 2 or float(bounds[0]) != float(bounds[1]):
        return None
    numerator = np.asarray(inits[inputs[0]], dtype=np.float32)
    if numerator.size != 1 or float(bounds[0]) == 0.0:
        return None
    value = np.asarray(numerator / np.float32(bounds[0]), dtype=np.float32)
    return Segment(
        rec["name"],
        "algebraic_constant",
        [rec["name"]],
        [inputs[1]],
        outputs,
        f"constant Div folded at singleton calibrated input {float(bounds[0])}",
        lambda: None,
        constant_value=value,
        constant_guard=(inputs[1], np.asarray(bounds[0], dtype=np.float32)),
    )


def _precision_binary_segment_for(
    rec: Mapping,
    override: Mapping | None,
    model: onnx.ModelProto,
) -> Segment | None:
    """Build a binary segment only for an explicitly selected exact template.

    Overrides use the same role-keyed scale/zero-point contract as compile_onnx.
    This path is deliberately opt-in: high-precision fixtures are not
    interchangeable with the step graph's U8 activation calibration.
    """
    if override is None:
        return None
    op = rec.get("op")
    inputs, outputs = list(rec.get("inputs", ())), list(rec.get("outputs", ()))
    if op not in axb.bse.OPS or len(inputs) != 2 or len(outputs) != 1:
        raise ValueError(f"{rec.get('name')}: precision override requires a binary op")
    values = {
        v.name: tuple(int(d.dim_value) for d in v.type.tensor_type.shape.dim)
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    input_shapes = [values.get(t, ()) for t in inputs]
    output_shape = values.get(outputs[0], ())
    shape = output_shape
    input_transforms = {}
    if not shape or any(not item for item in input_shapes):
        raise ValueError(
            f"{rec.get('name')}: precision override requires known tensor shapes"
        )
    packed_vector = (
        op == "Mul"
        and len(output_shape) == 1
        and tuple(override.get("template_shape", ())) == (1, *output_shape)
    )
    if packed_vector:
        shape = (1, *output_shape)
    if input_shapes != [shape, shape]:
        if np.broadcast_shapes(*input_shapes) != output_shape or (
            not packed_vector and np.broadcast_shapes(*input_shapes) != shape
        ):
            raise ValueError(
                f"{rec.get('name')}: precision override inputs must broadcast to "
                "the output shape"
            )
        for tensor, actual_shape in zip(inputs, input_shapes):
            if actual_shape != shape:
                if packed_vector and int(np.prod(actual_shape)) == 1:
                    # Keep a scalar-shaped port scalar. The compiled `[1,N]`
                    # model performs the broadcast inside its Mul node.
                    continue
                input_transforms[tensor] = lambda value, target=shape: np.broadcast_to(
                    value, target
                ).copy()
    precision = override.get("layer_precision")
    scales, zero_points = override.get("scales"), override.get("zero_points")
    roles = {"x", "z", "y"}
    if (
        precision not in ("U16", "S16")
        or not isinstance(scales, Mapping)
        or not isinstance(zero_points, Mapping)
    ):
        raise ValueError(
            f"{rec.get('name')}: precision override needs U16/S16 scales and zero_points"
        )
    if not roles <= set(scales) or not roles <= set(zero_points):
        raise ValueError(
            f"{rec.get('name')}: precision override needs x, z, and y calibration"
        )
    zps = {role: int(zero_points[role]) for role in roles}
    if any(float(zero_points[role]) != zps[role] for role in roles):
        raise ValueError(
            f"{rec.get('name')}: precision override zero points must be integral"
        )
    key = axb.TemplateKey(
        op,
        (shape,),
        calibration_class=",".join(f"{r}{zps[r]}" for r in ("x", "y", "z")),
        layer_precision=precision,
        calibration_scales=tuple(float(scales[r]) for r in ("x", "z", "y")),
    )
    cache = axb.TemplateCache()
    entry = cache.lookup(key)
    template = cache.load(key)
    signed = precision == "S16"

    def quant(role: str) -> tuple[float, int, bool, int]:
        return float(scales[role]), zps[role], signed, 16

    segment_input_shapes = [shape, shape]
    return Segment(
        rec["name"],
        "binary_precision",
        [rec["name"]],
        inputs,
        outputs,
        f"exact {precision} binary template ({entry.meta['file']})",
        lambda: template,
        [quant("x"), quant("z")],
        [quant("y")],
        input_shapes=segment_input_shapes,
        output_shape=() if packed_vector else output_shape,
        input_transforms=input_transforms,
    )


def build_plan(
    model: onnx.ModelProto,
    records: Sequence[Mapping],
    calib: Mapping,
    kinds: set[str] | None = None,
    include_unsafe: bool = False,
    *,
    precision_overrides: Mapping[str, Mapping] | None = None,
    fp32_only: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[Segment], dict[str, str]]:
    """NPU segments (only of ``kinds`` if given) and a per-node reason for
    every node left on the host.

    Nodes in ``fp32_only`` (the Adam update: ``--fp32-optimizer``) never get a
    quantized template. They run as an unquantized FP32 binary template when
    one is captured for their exact shapes, otherwise on the host."""
    cache = axb.TemplateCache()
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    value_shapes = {
        v.name: tuple(int(d.dim_value) for d in v.type.tensor_type.shape.dim)
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    planned_records = []
    for original in records:
        rec = dict(original)
        attrs = dict(rec.get("attrs", {}))
        if attrs.get("form") == "const":
            const = next((t for t in rec["inputs"][1:] if t in inits), None)
            const_index = next(
                (i for i, t in enumerate(rec["inputs"]) if t in inits), None
            )
            if const is None and const_index is not None:
                const = rec["inputs"][const_index]
            if const is not None:
                value = np.asarray(inits[const], dtype=np.float32)
                attrs["constant_zero_point"] = 0 if float(value.min()) >= 0 else 128
                attrs["constant_input"] = const_index
                # Constant broadcasts are compiled as the full output shape.
                # The compact records historically kept the live input shape,
                # which hid validated full-shape binary templates from the
                # planner (for example the 16x512x1x1 * 1x1x7x7 path).
                out_shape = value_shapes.get(rec["outputs"][0], ())
                if out_shape and tuple(rec["shapes"][0]) != out_shape:
                    attrs["output_shape"] = list(out_shape)
                live_name = rec["inputs"][1 - int(const_index)]
                live_shape = value_shapes.get(live_name, ())
                tile_side = None
                if (
                    rec["op"] == "Mul"
                    and len(live_shape) == 4
                    and live_shape[1] == 1
                    and live_shape[3] > 0
                    and int(math.isqrt(live_shape[3])) ** 2 == live_shape[3]
                ):
                    side = math.isqrt(live_shape[3])
                    # Prefer the largest validated spatial tile already in
                    # the fixture corpus. A tiled mask keeps the same
                    # contiguous channel/spatial layout, so a template can
                    # serve any larger side divisible by its tile.
                    for candidate in (56, 28, 14, 7):
                        if side % candidate:
                            continue
                        trial = dict(rec)
                        trial_attrs = dict(attrs)
                        trial_attrs["template_shape"] = [
                            live_shape[0],
                            live_shape[2],
                            candidate,
                            candidate,
                        ]
                        trial["attrs"] = trial_attrs
                        trial["shapes"] = [list(trial_attrs["template_shape"])]
                        try:
                            cache.lookup(axb.key_for_record(trial, "x0,y0,z0"))
                        except ValueError:
                            continue
                        tile_side = candidate
                        break
                if tile_side is not None:
                    attrs["template_shape"] = [
                        live_shape[0],
                        live_shape[2],
                        tile_side,
                        tile_side,
                    ]
                    attrs["tile_blocks"] = (side // tile_side) ** 2
                    attrs["tile_side"] = tile_side
                    rec["shapes"] = [list(attrs["template_shape"])]
                elif (
                    rec["op"] == "Mul"
                    and np.all((value == 0) | (value == 1))
                    and out_shape
                    and int(np.prod(out_shape)) > 1024 * 512
                ):
                    # Binary masks are contiguous regardless of their ONNX
                    # rank. Pack them into the validated 1000x512 frame and
                    # run one native invocation per frame; padding is dropped
                    # by the inverse transform at the segment boundary.
                    flat_template = [1024, 512]
                    trial = dict(rec)
                    trial_attrs = dict(attrs)
                    trial_attrs["template_shape"] = flat_template
                    trial["attrs"] = trial_attrs
                    trial["shapes"] = [flat_template]
                    try:
                        cache.lookup(axb.key_for_record(trial, "x0,y0,z0"))
                    except ValueError:
                        pass
                    else:
                        attrs["template_shape"] = flat_template
                        attrs["flat_blocks"] = (
                            int(np.prod(out_shape)) + 1024 * 512 - 1
                        ) // (1024 * 512)
                        attrs["tile_blocks"] = attrs["flat_blocks"]
                        rec["shapes"] = [flat_template]
        if (
            rec["op"] == "Mul"
            and attrs.get("form") in ("same_shape", "broadcast")
            and not attrs.get("flat_blocks")
        ):
            out_shape = value_shapes.get(rec["outputs"][0], ())
            if out_shape and int(np.prod(out_shape)) > 1024 * 512:
                # Binary classes are ordered x, y, z while graph inputs are
                # ordered x, z. Keep the output in the middle here.
                qnames = (rec["inputs"][0], rec["outputs"][0], rec["inputs"][1])
                zps = [
                    int(calib["tensors"][t]["zero_point"])
                    for t in qnames
                    if t in calib.get("tensors", {})
                ]
                if len(zps) == 3 and zps == [0, 0, 0]:
                    scales = [float(calib["tensors"][t]["scale"]) for t in qnames]
                    if not all(abs(s - 1.0 / 255.0) < 1e-7 for s in scales):
                        rec["attrs"] = attrs
                        planned_records.append(rec)
                        continue
                    flat_template = [1024, 512]
                    trial = dict(rec)
                    trial_attrs = dict(attrs)
                    trial_attrs["template_shape"] = flat_template
                    trial["attrs"] = trial_attrs
                    trial["shapes"] = [flat_template]
                    try:
                        cache.lookup(axb.key_for_record(trial, "x0,y0,z0"))
                    except ValueError:
                        pass
                    else:
                        attrs["template_shape"] = flat_template
                        attrs["flat_blocks"] = (
                            int(np.prod(out_shape)) + 1024 * 512 - 1
                        ) // (1024 * 512)
                        attrs["tile_blocks"] = attrs["flat_blocks"]
                        rec["shapes"] = [flat_template]
        if (
            rec["op"] == "Mul"
            and attrs.get("form") == "same_shape"
            and len(value_shapes.get(rec["outputs"][0], ())) == 4
        ):
            out_shape = value_shapes[rec["outputs"][0]]
            height, width = out_shape[2:]
            if height == width and height % 56 == 0:
                qnames = (rec["inputs"][0], rec["outputs"][0], rec["inputs"][1])
                if all(t in calib.get("tensors", {}) for t in qnames):
                    cls = "x{},y{},z{}".format(
                        *(int(calib["tensors"][t]["zero_point"]) for t in qnames)
                    )
                    trial = dict(rec)
                    trial_attrs = dict(attrs)
                    trial_attrs["template_shape"] = [out_shape[0], out_shape[1], 56, 56]
                    trial["attrs"] = trial_attrs
                    trial["shapes"] = [list(trial_attrs["template_shape"])]
                    try:
                        cache.lookup(axb.key_for_record(trial, cls))
                    except ValueError:
                        pass
                    else:
                        attrs["template_shape"] = [out_shape[0], out_shape[1], 56, 56]
                        attrs["tile_blocks"] = (height // 56) * (width // 56)
                        attrs["tile_side"] = 56
                        attrs["tile_layout"] = "nchw"
                        rec["shapes"] = [list(attrs["template_shape"])]
        rec["attrs"] = attrs
        planned_records.append(rec)
    unknown_overrides = set(precision_overrides or {}) - {
        rec["name"] for rec in planned_records
    }
    if unknown_overrides:
        raise ValueError(
            f"precision overrides name unknown step nodes: {sorted(unknown_overrides)}"
        )
    # Live MatMul/Conv validation scans the compiled MCode.  The same scan is
    # required by the segment emitter below, so defer it to
    # ``drop_unemittable`` instead of doing it once during planning and again
    # while materializing the models.
    plans = [
        axb.plan_at_calibration(r, calib, cache, validate_live=False)
        for r in planned_records
    ]
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for n in model.graph.node:
        for t in n.input:
            consumers.setdefault(t, []).append(n)
    segs: list[Segment] = []
    host: dict[str, str] = {}
    taken: set[str] = set()
    candidates = []
    value_shapes = {
        v.name: tuple(int(d.dim_value) for d in v.type.tensor_type.shape.dim)
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    for rec, (status, detail) in zip(planned_records, plans):
        if rec["name"] in fp32_only:
            fp32_seg = _fp32_binary_segment_for(rec, model)
            if fp32_seg is None:
                host[rec["name"]] = "fp32-only optimizer node: no FP32 template"
            elif kinds and fp32_seg.kind not in kinds:
                host[rec["name"]] = f"covered, kind {fp32_seg.kind} not selected"
            else:
                candidates.append(fp32_seg)
            continue
        folded = _singleton_scalar_div_fold(rec, calib, inits)
        if folded is not None:
            if not kinds or folded.kind in kinds:
                candidates.append(folded)
            else:
                host[rec["name"]] = f"covered, kind {folded.kind} not selected"
            continue
        safe_div = _safe_masked_div_segment_for(rec, model)
        if safe_div is not None:
            if not kinds or safe_div.kind in kinds:
                candidates.append(safe_div)
            else:
                host[rec["name"]] = f"covered, kind {safe_div.kind} not selected"
            continue
        precision_seg = _precision_binary_segment_for(
            rec,
            (precision_overrides or {}).get(rec["name"]),
            model,
        )
        if precision_seg is not None:
            profiled_fp32 = _fp32_binary_segment_for(rec, model)
            if (
                profiled_fp32 is not None
                and profiled_fp32.profiled_faster
                and (not kinds or profiled_fp32.kind in kinds)
            ):
                candidates.append(profiled_fp32)
                continue
            if not kinds or precision_seg.kind in kinds:
                candidates.append(precision_seg)
                continue
            host[rec["name"]] = f"covered, kind {precision_seg.kind} not selected"
            continue
        if (
            status in ("refused", "covered")
            and rec["op"] in axb.bse.OPS
            and rec.get("attrs", {}).get("form") == "broadcast"
            and detail.startswith("ElementwiseScaleEdit")
        ):
            input_shapes = [value_shapes.get(t, ()) for t in rec["inputs"][:2]]
            output_shape = value_shapes.get(rec["outputs"][0], ())
            if (
                output_shape
                and all(input_shapes)
                and all(t in calib.get("tensors", {}) for t in rec["inputs"][:2])
            ):
                # A full-shape binary template is semantically identical when
                # a live operand is broadcast at the segment boundary. Keep
                # constants and uncalibrated operands conservative.
                expanded = dict(rec)
                expanded["shapes"] = [list(output_shape)]
                expanded["attrs"] = dict(rec.get("attrs", {}))
                expanded["attrs"].update(
                    {
                        "form": "same_shape",
                        "output_shape": list(output_shape),
                        "input_shapes": [list(s) for s in input_shapes],
                    }
                )
                status, detail = axb.plan_at_calibration(
                    expanded, calib, cache, validate_live=False
                )
                rec = expanded
        exact_seg = _exact_mask_mul_segment_for(rec, calib, model)
        if exact_seg is None:
            exact_seg = _exact_loss_sub_segment_for(rec, calib, model)
        if exact_seg is None:
            exact_seg = _exact_div2_segment_for(rec, calib, model, inits)
        if exact_seg is not None and (not kinds or exact_seg.kind in kinds):
            candidates.append(exact_seg)
            continue
        fp32_seg = _fp32_binary_segment_for(rec, model)
        if fp32_seg is not None and fp32_seg.prefer_fp32:
            if not kinds or fp32_seg.kind in kinds:
                qtable = calib.get("tensors", {})
                if any(tensor in qtable for tensor in (*fp32_seg.inputs, *fp32_seg.outputs)):
                    fp32_seg.in_q = [
                        qparams_of(calib, tensor) if tensor in qtable else None
                        for tensor in fp32_seg.inputs
                    ]
                    fp32_seg.out_q = [
                        qparams_of(calib, tensor) if tensor in qtable else None
                        for tensor in fp32_seg.outputs
                    ]
                    fp32_seg.quantize_device_io = True
                candidates.append(fp32_seg)
                continue
            host[rec["name"]] = "covered, kind fp32_binary not selected"
            continue
        legacy_fp32_add = rec.get("op") == "Add" and [
            value_shapes.get(t) for t in rec.get("inputs", ())
        ] == [(16, 1000), (16, 1000)]
        use_fp32 = fp32_seg is not None and (not kinds or fp32_seg.kind in kinds)
        if use_fp32 and legacy_fp32_add:
            # Keep the original exact Add probe as the one explicit override
            # of a regular calibrated segment.
            candidates.append(fp32_seg)
            continue
        if status != "covered":
            if use_fp32:
                candidates.append(fp32_seg)
                continue
            host[rec["name"]] = f"{status}: {detail}"
            continue
        seg = _segment_for(rec, detail, calib, model, consumers, inits)
        if seg is None or (kinds and seg.kind not in kinds):
            if use_fp32:
                candidates.append(fp32_seg)
                continue
            host[rec["name"]] = (
                f"covered ({detail}) but no runner segment"
                if seg is None
                else f"covered, kind {seg.kind} not selected"
            )
            continue
        if seg.unsafe and not include_unsafe:
            if use_fp32:
                candidates.append(fp32_seg)
                continue
            host[rec["name"]] = f"covered, but unsafe: {seg.unsafe}"
            continue
        if (
            seg.kind == "binary_precision"
            and fp32_seg is not None
            and fp32_seg.profiled_faster
            and (not kinds or fp32_seg.kind in kinds)
        ):
            candidates.append(fp32_seg)
            continue
        candidates.append(seg)
    # multi-node segments (chains, fused pairs) claim their nodes first
    # A node inside two chains (the fc Squeeze feeds both the forward Gemm
    # chain and the fc dW chain) is recomputed by each; only a clash on a
    # node whose output a segment exports keeps the smaller segment out.
    exported: dict[str, set[str]] = {}
    for seg in candidates:
        exported[seg.name] = {
            n.name for n in model.graph.node if set(n.output) & set(seg.outputs)
        }
    claimed: dict[str, str] = {}
    for seg in sorted(candidates, key=lambda s: -len(s.nodes)):
        clash = [
            n
            for n in taken & set(seg.nodes)
            if n in exported[seg.name] or n in exported[claimed[n]]
        ]
        if clash:
            for n in seg.nodes:
                host.setdefault(n, f"covered, but inside another segment ({seg.name})")
            continue
        segs.append(seg)
        for n in seg.nodes:
            claimed.setdefault(n, seg.name)
        taken.update(seg.nodes)
    order = {r["name"]: k for k, r in enumerate(records)}
    segs.sort(key=lambda s: max(order.get(n, 0) for n in s.nodes))
    for s in segs:
        for n in s.nodes:
            host.pop(n, None)
    return segs, host


def drop_unemittable(
    segs: Sequence[Segment],
    host: dict[str, str],
    emit_cache_dir: str | None = None,
) -> tuple[list[Segment], dict[str, bytes]]:
    """Emit every segment up front; one whose emitter refuses goes back to
    the host with the emitter's reason.  When ``emit_cache_dir`` is set,
    emitted models are reused across training-graph preparations.  The key
    includes calibration inputs/outputs and a format version, so a different
    calibration or emitter format cannot reuse an old model."""
    keep, blobs = [], {}
    if emit_cache_dir:
        os.makedirs(emit_cache_dir, exist_ok=True)

    def emit_one(seg: Segment) -> tuple[Segment, bytes | None, Exception | None]:
        try:
            if seg.kind in ("algebraic_identity", "algebraic_constant"):
                return seg, None, None
            cache_path = None
            if emit_cache_dir:
                payload = {
                    "version": 1,
                    "name": seg.name,
                    "kind": seg.kind,
                    "nodes": seg.nodes,
                    "inputs": seg.inputs,
                    "outputs": seg.outputs,
                    "detail": seg.detail,
                    "in_q": seg.in_q,
                    "out_q": seg.out_q,
                    "input_shapes": seg.input_shapes,
                    "output_shape": seg.output_shape,
                }
                digest = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, default=list).encode()
                ).hexdigest()
                cache_path = os.path.join(emit_cache_dir, f"{digest}.axmodel")
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        blob = f.read()
                    onnx.load_model_from_string(blob)
                else:
                    blob = seg.emit().SerializeToString()
                    tmp = f"{cache_path}.tmp-{os.getpid()}"
                    with open(tmp, "wb") as f:
                        f.write(blob)
                    os.replace(tmp, cache_path)
            else:
                blob = seg.emit().SerializeToString()
            return seg, blob, None
        except Exception as exc:
            return seg, None, exc

    # Emission is CPU-bound serialization/retargeting work and each segment
    # owns its output buffer.  Keep the result collection in input order so
    # plans and reports remain deterministic, while avoiding a long serial
    # preparation tail for the training graph.  A small fixed pool avoids
    # overwhelming the host when a graph has many elementwise segments.
    workers = min(8, max(1, len(segs)))
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="axera-emit"
    ) as pool:
        results = list(pool.map(emit_one, segs))
    for seg, blob, error in results:
        if error is not None:
            for n in seg.nodes:
                host[n] = f"covered, but the emitter refused: {error}"
            continue
        if seg.kind in ("algebraic_identity", "algebraic_constant"):
            # Algebraic segments need no AX program and must not be demoted to
            # HostOps/ORT unless a guarded fold sees a different runtime value.
            keep.append(seg)
            continue
        assert blob is not None
        blobs[seg.name] = blob
        keep.append(seg)
    return keep, blobs


# --------------------------------------------------------------------------
# execution


@dataclasses.dataclass
class SegStat:
    segment: str
    kind: str
    nodes: int
    max_lsb: float = 0.0
    frac_gt1: float = 0.0
    max_abs: float = 0.0
    device_s: float = 0.0
    emit_s: float = 0.0
    sim_s: float = 0.0
    float_rel: float = 0.0  # device vs the float ops on the same inputs
    sim_float_rel: float = 0.0  # simulation vs float: the quantization alone
    float_mismatch: float = 0.0  # unquantized outputs: fraction != float
    error: str = ""
    runtime_fallback: bool = False
    fallback_reason: str = ""


class StepRunner:
    """Execute ``model`` node by node, NPU segments on the device (``mode``
    ``"npu"``), simulated on the host (``"sim"``), or everything in float
    (``"float"``)."""

    def __init__(
        self,
        model: onnx.ModelProto,
        segments: Sequence[Segment],
        session=None,
        emit_dir: str | None = None,
        health_every: int = 0,
        fallback_on_failure: bool = False,
    ):
        self.model = model
        self.host = HostOps(model)
        self.session = session
        self.emit_dir = emit_dir
        self.health_every = health_every
        self.fallback_on_failure = fallback_on_failure
        self.device_runs = 0
        self.stalled: list[str] = []
        self.nodes = list(model.graph.node)
        self.index = {n.name: k for k, n in enumerate(self.nodes)}
        self.by_name = {n.name: n for n in self.nodes}
        self.segments = list(segments)
        self.seg_of: dict[str, Segment] = {}
        for s in self.segments:
            for n in s.nodes:
                self.seg_of[n] = s
        self.fire_at = {
            s.name: max(self.index[n] for n in s.nodes) for s in self.segments
        }
        self._host_dup = self._host_duplicates()
        last: dict[str, int] = {}
        for k, n in enumerate(self.nodes):
            for t in n.input:
                last[t] = k
        for s in self.segments:
            for t in s.inputs:
                last[t] = max(last.get(t, 0), self.fire_at[s.name])
        self.last = last
        self._emitted: dict[str, bytes] = {}

    def _host_duplicates(self) -> set[str]:
        """Segment-internal nodes the host must compute as well: their output
        is consumed outside the segment (or before the segment fires) but is
        not a segment output."""
        producers = {o: n for n in self.nodes for o in n.output}
        dup: set[str] = set()
        for s in self.segments:
            members = set(s.nodes)
            fire = self.fire_at[s.name]

            def need(t: str) -> None:
                p = producers.get(t)
                if p is None or p.name not in members or p.name in dup:
                    return
                dup.add(p.name)
                for i in p.input:
                    need(i)

            for n in s.nodes:
                for t in self.by_name[n].output:
                    outside = [
                        c for c in self.nodes if t in c.input and c.name not in members
                    ]
                    early = any(self.index[c.name] < fire for c in outside)
                    if outside and (t not in s.outputs or early):
                        need(t)
            for o in self.model.graph.output:
                if (
                    o.name in {t for n in s.nodes for t in self.by_name[n].output}
                    and o.name not in s.outputs
                ):
                    need(o.name)
        return dup

    def emitted(self, seg: Segment) -> bytes:
        if seg.name not in self._emitted:
            self._emitted[seg.name] = seg.emit().SerializeToString()
            if self.emit_dir:
                with open(
                    os.path.join(self.emit_dir, f"{seg.name}.axmodel"), "wb"
                ) as f:
                    f.write(self._emitted[seg.name])
        return self._emitted[seg.name]

    def _float(self, seg: Segment, env: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        if seg.kind == "safe_masked_div":
            numerator = np.asarray(env[seg.inputs[0]], np.float32)
            count = np.asarray(env[seg.inputs[1]], np.float32)
            return [np.divide(numerator, np.maximum(count, np.float32(1.0)))]
        local = dict(env)
        for tensor in seg.constant_inputs:
            local[tensor] = np.asarray(self.host.inits[tensor], dtype=np.float32)
        for n in sorted(seg.nodes, key=self.index.get):
            node = self.by_name[n]
            for t, v in zip([t for t in node.output if t], self.host.run(node, local)):
                local[t] = v
        return [local[t] for t in seg.outputs]

    def _sim(self, seg: Segment, env: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        if seg.kind == "safe_masked_div":
            numerator = np.asarray(env[seg.inputs[0]], np.float32)
            count = np.asarray(env[seg.inputs[1]], np.float32)
            return [np.divide(numerator, np.maximum(count, np.float32(1.0)))]
        local = dict(env)
        for tensor in seg.constant_inputs:
            local[tensor] = np.asarray(self.host.inits[tensor], dtype=np.float32)
        values = []
        for j, (t, qq) in enumerate(zip(seg.inputs, seg.in_q)):
            value = fake_quant(local[t], *qq) if qq else np.asarray(local[t])
            values.append(value)
        if seg.output_shape:
            target = np.broadcast_shapes(*(value.shape for value in values))
            for t, value in zip(seg.inputs, values):
                local[t] = np.broadcast_to(value, target)
        else:
            for t, value in zip(seg.inputs, values):
                local[t] = value
        for n in sorted(seg.nodes, key=self.index.get):
            node = self.by_name[n]
            for t, v in zip([t for t in node.output if t], self.host.run(node, local)):
                local[t] = v
        outs = [local[t] for t in seg.outputs]
        return [
            fake_quant(o, *qq) if qq and o.dtype.kind == "f" else o
            for o, qq in zip(outs, seg.out_q or [None] * len(outs))
        ]

    def _device(self, seg: Segment, env: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        m = self.session.load(self.emitted(seg))
        try:
            ins = []
            for t in seg.inputs:
                if t in env:
                    value = env[t]
                elif t in seg.constant_inputs:
                    value = self.host.inits[t]
                else:
                    raise KeyError(t)
                ins.append(
                    seg.input_transforms.get(t, lambda x: x)(
                        np.asarray(value, dtype=np.float32)
                    )
                )
            if seg.quantize_device_io:
                ins = [
                    fake_quant(value, *q) if q else value
                    for value, q in zip(ins, seg.in_q)
                ]
            model_inputs = getattr(m, "inputs", None)
            if model_inputs is not None and len(model_inputs) > len(ins):
                initializer = {
                    item.name: numpy_helper.to_array(item)
                    for item in self.model.graph.initializer
                }
                for node_name in seg.nodes:
                    for tensor in self.by_name[node_name].input:
                        if tensor not in seg.inputs and tensor in initializer:
                            value = np.asarray(initializer[tensor], dtype=np.float32)
                            ins.append(
                                seg.input_transforms.get(tensor, lambda x: x)(value)
                            )
            if model_inputs is not None and len(ins) != len(model_inputs):
                raise ValueError(
                    f"segment {seg.name} emitted {len(m.inputs)} inputs, "
                    f"but runner prepared {len(ins)}"
                )
            if model_inputs is not None:
                matched = []
                split_flags = seg.split or [False] * len(ins)
                for value, spec, split in zip(ins, model_inputs, split_flags):
                    # A measured template may run a smaller batch repeatedly.
                    # Keep the expanded value intact until the split below;
                    # broadcasting it to the per-run shape would reject a
                    # valid (template_batch * batch_split) input.
                    expanded_batch = (
                        seg.batch_split > 1
                        and split
                        and value.ndim > 0
                        and value.shape[0] == spec.shape[0] * seg.batch_split
                    )
                    same_size = value.size == math.prod(spec.shape)
                    matched.append(
                        value
                        if expanded_batch or value.shape == spec.shape
                        else value.reshape(spec.shape)
                        if same_size
                        else np.broadcast_to(value, spec.shape).copy()
                    )
                ins = matched
            if seg.output_shape:
                target = np.broadcast_shapes(*(x.shape for x in ins))
                ins = [np.broadcast_to(x, target) for x in ins]
            if seg.batch_split > 1:
                parts = [
                    self.session.run(
                        m,
                        [
                            np.array_split(x, seg.batch_split)[j] if s else x
                            for x, s in zip(ins, seg.split)
                        ],
                    )
                    for j in range(seg.batch_split)
                ]
                ys = [np.concatenate(p) for p in zip(*parts)]
            else:
                ys = self.session.run(m, ins)
        finally:
            self.session.unload(m)
        want = {o.name: o for o in self.model.graph.value_info}
        out = []
        for j, (t, y) in enumerate(zip(seg.outputs, ys)):
            if seg.output_transform is not None:
                y = seg.output_transform(y)
            vi = want.get(t)
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim] if vi else None
            y = y.astype(np.float32)
            if seg.quantize_device_io and j < len(seg.out_q):
                if seg.out_q[j]:
                    y = fake_quant(y, *seg.out_q[j])
            if seg.output_take is not None:
                y = y.reshape(-1)[: seg.output_take]
            out.append(
                y.reshape(shape) if shape and int(np.prod(shape)) == y.size else y
            )
        return out

    def run(
        self,
        feeds: Mapping[str, np.ndarray],
        mode: str = "npu",
        check: bool = True,
        keep: Sequence[str] = (),
        progress: bool = False,
        stats_out: list | None = None,
    ) -> tuple[dict[str, np.ndarray], list[SegStat]]:
        env: dict[str, np.ndarray] = dict(feeds)
        keep_set = set(keep) | {o.name for o in self.model.graph.output}
        stats: list[SegStat] = stats_out if stats_out is not None else []
        t0 = time.time()
        for k, node in enumerate(self.nodes):
            seg = self.seg_of.get(node.name)
            if seg is None or mode == "float" or node.name in self._host_dup:
                for t, v in zip(
                    [t for t in node.output if t], self.host.run(node, env)
                ):
                    env[t] = v
            if seg is not None and mode != "float" and self.fire_at[seg.name] == k:
                if seg.kind == "algebraic_identity":
                    env[seg.outputs[0]] = env[seg.inputs[0]]
                    stats.append(SegStat(seg.name, seg.kind, len(seg.nodes)))
                    continue
                if seg.kind == "algebraic_constant":
                    assert (
                        seg.constant_value is not None
                        and seg.constant_guard is not None
                    )
                    guard_name, expected = seg.constant_guard
                    actual = np.asarray(env[guard_name])
                    if actual.size == 1 and np.array_equal(
                        actual.reshape(()), expected.reshape(())
                    ):
                        env[seg.outputs[0]] = seg.constant_value.copy()
                    else:
                        for t, v in zip(
                            [t for t in node.output if t], self.host.run(node, env)
                        ):
                            env[t] = v
                    stats.append(SegStat(seg.name, seg.kind, len(seg.nodes)))
                    continue
                if (
                    mode == "npu"
                    and seg.nan_guard
                    and any(np.isnan(np.asarray(env[t])).any() for t in seg.inputs)
                ):
                    for tensor, value in zip(seg.outputs, self._float(seg, env)):
                        env[tensor] = value
                    stats.append(SegStat(seg.name, "host_nan_guard", len(seg.nodes)))
                    continue
                st = SegStat(seg.name, seg.kind, len(seg.nodes))
                t1 = time.time()
                sim = self._sim(seg, env) if (mode == "sim" or check) else None
                st.sim_s = time.time() - t1
                if mode == "npu":
                    t1 = time.time()
                    try:
                        self.emitted(seg)
                    except Exception as exc:
                        st.error = f"emit: {type(exc).__name__}: {exc}"
                    st.emit_s = time.time() - t1
                    t1 = time.time()
                    if not st.error:
                        try:
                            dev = self._device(seg, env)
                            self.device_runs += 1
                            if (
                                self.health_every
                                and self.device_runs % self.health_every == 0
                            ):
                                import axcl_session

                                axcl_session.health_check(self.session)
                        except Exception as exc:  # recorded, then fall back to sim
                            st.error = f"{type(exc).__name__}: {exc}"
                            if isinstance(exc, _device_errors()):
                                # the card is suspect: void everything from here on
                                self.stalled = [
                                    s.segment for s in stats[-self.health_every :]
                                ] + [seg.name]
                                stats.append(st)
                                raise
                    if st.error:
                        dev = sim if sim is not None else self._sim(seg, env)
                    st.device_s = time.time() - t1
                    if sim is not None and not st.error:
                        _compare(st, seg, dev, sim)
                        flt = self._float(seg, env)
                        st.float_rel = max(_rel(d, f) for d, f in zip(dev, flt))
                        st.sim_float_rel = max(_rel(s_, f) for s_, f in zip(sim, flt))
                        if not seg.out_q:
                            st.float_mismatch = max(
                                float(
                                    (
                                        np.abs(
                                            np.asarray(d, np.float32).ravel()
                                            - np.asarray(f, np.float32).ravel()
                                        )
                                        > 1e-6
                                    ).mean()
                                )
                                for d, f in zip(dev, flt)
                            )
                    res = dev
                    if (
                        mode == "npu"
                        and self.fallback_on_failure
                        and not segment_passed(dataclasses.asdict(st))
                    ):
                        st.runtime_fallback = True
                        st.fallback_reason = st.error or "device output failed validation"
                        res = self._float(seg, env)
                else:
                    res = sim
                for t, v in zip(seg.outputs, res):
                    env[t] = v
                stats.append(st)
            for t in set(node.input):
                if self.last.get(t) == k and t not in keep_set and t not in feeds:
                    env.pop(t, None)
            if progress and k % 100 == 0:
                print(
                    f"  node {k}/{len(self.nodes)} {time.time() - t0:.1f}s", flush=True
                )
        return {t: env[t] for t in keep_set if t in env}, stats


def _rel(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return math.inf
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _device_errors() -> tuple:
    import axcl_session

    return (axcl_session.DeviceStall,)


def _compare(st: SegStat, seg: Segment, dev, sim) -> None:
    for k, (d, s) in enumerate(zip(dev, sim)):
        d = np.asarray(d, np.float32).ravel()
        s = np.asarray(s, np.float32).ravel()
        if d.size != s.size:
            st.error = f"output {k}: {d.size} elements, expected {s.size}"
            return
        if not np.isfinite(d).all() or not np.isfinite(s).all():
            st.error = f"output {k}: non-finite device or simulation values"
            return
        diff = np.abs(d - s)
        st.max_abs = max(st.max_abs, float(diff.max(initial=0)))
        if k < len(seg.out_q) and seg.out_q[k]:
            lsb = diff / np.float32(seg.out_q[k][0])
            st.max_lsb = max(st.max_lsb, float(lsb.max(initial=0)))
            st.frac_gt1 = max(
                st.frac_gt1, float((lsb > 1.01).mean()) if lsb.size else 0.0
            )
        else:  # unquantized output (compare/cast, transpose, gather): exact values
            st.frac_gt1 = max(
                st.frac_gt1, float((diff > 1e-6).mean()) if diff.size else 0.0
            )


# --------------------------------------------------------------------------
# step feeds / references


def _float_gradients(model, feeds, grad_names) -> dict[str, np.ndarray]:
    """The float gradients, from a host-only run (cached next to the step)."""
    cache = STEP_REF + ".grads.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        if set(z.files) == set(grad_names):
            return {k: z[k] for k in z.files}
    outs, _ = StepRunner(model, []).run(feeds, "float", keep=list(grad_names.values()))
    grads = {w: np.asarray(outs[t]) for w, t in grad_names.items()}
    np.savez(cache, **grads)
    return grads


def load_step(path: str = STEP_ONNX) -> onnx.ModelProto:
    return shape_inference.infer_shapes(onnx.load(path))


def rewrite_softmax_ratio_gradients(model: onnx.ModelProto) -> int:
    """Remove unstable ``p * (a/p - sum(a/p*p))`` gradient chains.

    For finite nonzero probabilities this is exactly ``a - p*sum(a)``. The
    latter is also the continuous limit for zero-probability entries when
    their corresponding ``a`` is zero, avoiding the ``0/0`` and ``0*inf``
    generated by a quantized Softmax. Only rewrite the complete, exclusively
    consumed pattern with a keepdims reduction; ambiguous graphs are left
    untouched.
    """
    nodes = list(model.graph.node)
    shapes = {
        value.name: tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
        if value.type.tensor_type.HasField("shape")
    }
    graph_outputs = {value.name for value in model.graph.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)
    remove: set[str] = set()
    replacements: dict[str, onnx.NodeProto] = {}
    for div in nodes:
        if div.op_type != "Div" or len(div.input) != 2 or len(div.output) != 1:
            continue
        numerator, probability = div.input
        ratio = div.output[0]
        if not shapes.get(numerator) or shapes.get(numerator) != shapes.get(probability):
            continue
        ratio_users = consumers.get(ratio, [])
        mul_users = [
            n for n in ratio_users
            if n.op_type == "Mul" and len(n.input) == 2 and probability in n.input
            and ratio in n.input
        ]
        sub_users = [
            n for n in ratio_users
            if n.op_type == "Sub" and len(n.input) == 2 and n.input[0] == ratio
        ]
        if len(ratio_users) != 2 or len(mul_users) != 1 or len(sub_users) != 1:
            continue
        product, subtract = mul_users[0], sub_users[0]
        if product.input[0] != probability and product.input[1] != probability:
            continue
        if len(consumers.get(product.output[0], [])) != 1:
            continue
        reduce = consumers[product.output[0]][0]
        if reduce.op_type != "ReduceSum" or len(reduce.input) < 1:
            continue
        if reduce.input[0] != product.output[0] or len(reduce.output) != 1:
            continue
        keepdims = next(
            (helper.get_attribute_value(a) for a in reduce.attribute if a.name == "keepdims"),
            1,
        )
        if keepdims != 1 or len(consumers.get(reduce.output[0], [])) != 1:
            continue
        if consumers[reduce.output[0]][0] is not subtract:
            continue
        if len(consumers.get(subtract.output[0], [])) != 1:
            continue
        final = consumers[subtract.output[0]][0]
        if final.op_type != "Mul" or len(final.input) != 2 or probability not in final.input:
            continue
        intermediates = {
            ratio, product.output[0], reduce.output[0], subtract.output[0]
        }
        if intermediates & graph_outputs:
            continue

        reduce.input[0] = numerator
        product.input[:] = [probability, reduce.output[0]]
        subtract.input[:] = [numerator, product.output[0]]
        replacements[final.output[0]] = helper.make_node(
            "Identity", [subtract.output[0]], list(final.output), name=final.name
        )
        remove.add(ratio)

    if remove:
        kept = []
        for node in model.graph.node:
            if any(output in remove for output in node.output):
                continue
            replacement = next(
                (replacements[output] for output in node.output if output in replacements),
                None,
            )
            kept.append(replacement if replacement is not None else node)
        available = {value.name for value in model.graph.input}
        available.update(value.name for value in model.graph.initializer)
        ordered = []
        pending = list(kept)
        while pending:
            ready = [n for n in pending if all(t in available for t in n.input)]
            if not ready:
                raise ValueError("stable softmax-gradient rewrite made graph unsortable")
            for node in ready:
                pending.remove(node)
                ordered.append(node)
                available.update(node.output)
        del model.graph.node[:]
        model.graph.node.extend(ordered)
    return len(remove)


STEP_CALIB_CONFIG = "/home/takecheeze/npu-scratch/t6-r18fold/wd/config/step.json"


def apply_stable_softmax_grad(
    model: onnx.ModelProto, records: list[dict], cache: str
) -> tuple[list[dict], dict]:
    """Rewrite the softmax-ratio gradient chains in ``model`` (in place) and
    return records and a calibration that match the rewritten graph.

    The rewrite changes intermediate tensors, so the checked-in calibration no
    longer describes them; the ranges are re-collected on the rewritten graph
    over the same calibration set and cached in ``cache``."""
    import step_calibration

    if not rewrite_softmax_ratio_gradients(model):
        raise ValueError("no softmax-ratio gradient chain found to rewrite")
    by_name = {n.name: n for n in model.graph.node}
    kept = []
    for rec in records:
        node = by_name.get(rec["name"])
        if node is None or node.op_type != rec["op"]:
            continue  # removed Div, or Mul turned into Identity: host
        rec = dict(rec)
        rec["inputs"] = list(node.input)
        kept.append(rec)
    if os.path.exists(cache):
        return kept, axb.load_calibration(cache)
    calib = step_calibration.calibrate_model(model, STEP_CALIB_CONFIG)
    with open(cache, "w") as f:
        json.dump(calib, f, sort_keys=True)
    return kept, calib


def load_records(path: str = STEP_OPS) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def load_reference(path: str = STEP_REF) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def gradient_tensors(model: onnx.ModelProto, state_map: Mapping) -> dict[str, str]:
    """``{weight: gradient tensor}``: each weight's Adam first-moment update
    is ``m' = beta1 * m + (1 - beta1) * g``; ``g`` is the non-constant input of
    the second Mul, i.e. the backward pass's output for that weight."""
    prod = {o: n for n in model.graph.node for o in n.output}
    consts = {i.name for i in model.graph.initializer}
    out = {}
    for state, new in state_map.items():
        if not state.endswith("__m"):
            continue
        add = prod[new]
        for t in add.input:
            mul = prod.get(t)
            if mul is None or mul.op_type != "Mul" or state in mul.input:
                continue
            live = [i for i in mul.input if i not in consts]
            if len(live) == 1:
                out[state[:-3]] = live[0]
    return out


def optimizer_nodes(model: onnx.ModelProto, state_map: Mapping) -> set[str]:
    """Nodes of the optimizer update: ancestors of the new state (weights,
    moments) that are not ancestors of any gradient, i.e. the Adam math."""
    prod = {o: n for n in model.graph.node for o in n.output}

    def ancestors(tensors) -> set[str]:
        seen: set[str] = set()
        stack = list(tensors)
        while stack:
            n = prod.get(stack.pop())
            if n is None or n.name in seen:
                continue
            seen.add(n.name)
            stack.extend(n.input)
        return seen

    grads = gradient_tensors(model, state_map)
    return ancestors(state_map.values()) - ancestors(grads.values())


def gradients_from_moments(outputs: Mapping, state_map: Mapping) -> dict:
    """At step 1 (m0 = 0), Adam's new first moment is 0.1 * grad: each weight's
    gradient is 10 * its ``__m`` output."""
    grads = {}
    for w, out in state_map.items():
        if w.endswith("__m"):
            grads[w[:-3]] = 10.0 * np.asarray(outputs[out], np.float32)
    return grads


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def segment_passed(st: Mapping) -> bool:
    """A device run agrees with the simulation: no error, at most 2 LSB off
    (rounding at a half-LSB tie on both sides), and 1 LSB or less on all but
    0.1% of the elements."""
    return (
        not st["error"]
        and math.isfinite(float(st["max_lsb"]))
        and math.isfinite(float(st["frac_gt1"]))
        and st["max_lsb"] <= 2.01
        and st["frac_gt1"] <= 0.001
    )


def validated_failures(report: Mapping, recheck: Sequence[str] = ()) -> set[str]:
    """Failures from a report, minus explicitly selected segments to recheck."""
    records = report.get("validation_stats", report.get("segment_stats", ()))
    by_name = {item["segment"]: item for item in records}
    failed = {name for name, stat in by_name.items() if not segment_passed(stat)}
    recheck = set(recheck)
    unknown = recheck - set(by_name)
    if unknown:
        raise ValueError(f"cannot recheck segments absent from validation report: {sorted(unknown)}")
    not_failed = recheck - failed
    if not_failed:
        raise ValueError(f"recheck only accepts previously failing segments: {sorted(not_failed)}")
    return failed - recheck


def merge_validation_stats(previous: Mapping, current: Sequence[SegStat]) -> list[dict]:
    """Retain unselected results while refreshing statuses for executed segments."""
    records = previous.get("validation_stats", previous.get("segment_stats", ()))
    merged = {item["segment"]: dict(item) for item in records}
    merged.update({item.segment: dataclasses.asdict(item) for item in current})
    return [merged[name] for name in sorted(merged)]


def _reason_counts(host: Mapping[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in host.values():
        k = re.sub(r"\d+", "#", v)[:90]
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def summarize(stats: Sequence[SegStat]) -> dict:
    by: dict[str, dict] = {}
    for s in stats:
        d = by.setdefault(
            s.kind,
            {
                "segments": 0,
                "nodes": 0,
                "errors": 0,
                "max_lsb": 0.0,
                "worst": "",
                "bad": 0,
                "device_s": 0.0,
            },
        )
        d["segments"] += 1
        d["nodes"] += s.nodes
        d["device_s"] += s.device_s
        if s.error:
            d["errors"] += 1
        if s.max_lsb > d["max_lsb"]:
            d["max_lsb"], d["worst"] = s.max_lsb, s.segment
        if s.frac_gt1 > 0.001:
            d["bad"] += 1
    return by


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--kinds", help="comma-separated segment kinds to put on the NPU")
    p.add_argument(
        "--precision-overrides",
        default=STEP_PRECISION_OVERRIDES,
        help="JSON file mapping node names to exact U16/S16 binary calibration "
        f"(default: {os.path.basename(STEP_PRECISION_OVERRIDES)})",
    )
    p.add_argument("--mode", default="npu", choices=["npu", "sim", "float"])
    p.add_argument(
        "--stable-softmax-grad",
        metavar="CALIB_CACHE",
        help="apply rewrite_softmax_ratio_gradients and use a calibration "
        "regenerated for the rewritten graph (cached in this JSON file)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--emit-dir")
    p.add_argument(
        "--emit-cache-dir",
        default=os.path.join(tempfile.gettempdir(), "axera-step-emission-cache-v1"),
        help="persistent cache for emitted segment models (set empty to disable)",
    )
    p.add_argument(
        "--limit", type=int, default=0, help="only the first N segments on the NPU"
    )
    p.add_argument("--only", help="comma-separated segment names to put on the NPU")
    p.add_argument(
        "--exclude", help="regex: segments whose name matches stay on the host"
    )
    p.add_argument(
        "--host-optimizer",
        action="store_true",
        help="keep the Adam update (every node after the gradients) on the host in float",
    )
    p.add_argument(
        "--fp32-optimizer",
        action="store_true",
        help="run the Adam update unquantized: FP32 binary templates where "
        "captured, host float elsewhere (Sqrt, +eps, scalar Mul). The uint8 "
        "Sqrt/Add(eps) segments round sqrt(v)+eps to 0 (the simulation "
        "divides by 0 -> NaN) and cannot hold w - lr*update either",
    )
    p.add_argument(
        "--validated",
        help="a previous npu report: segments whose device output was more than "
        "2 LSB from the simulation there (or failed) stay on the host",
    )
    p.add_argument(
        "--recheck",
        default="",
        help="comma-separated previously failing segments to retry on-device; "
        "their new results are merged into the validation history",
    )
    p.add_argument(
        "--fallback-on-failure",
        action="store_true",
        help="for NPU execution, substitute the host float op when a segment fails its live check",
    )
    p.add_argument(
        "--include-unsafe",
        action="store_true",
        help="also run segments whose template semantics differ (Reshape->Relu on signed data)",
    )
    p.add_argument(
        "--health-every",
        type=int,
        default=1,
        help="native health check every N device runs",
    )
    args = p.parse_args(argv)
    if args.recheck and not args.validated:
        p.error("--recheck requires --validated")

    model = load_step()
    records = load_records()
    calib = axb.load_calibration(STEP_CALIB)
    if args.stable_softmax_grad:
        records, calib = apply_stable_softmax_grad(
            model, records, args.stable_softmax_grad
        )
    kinds = set(args.kinds.split(",")) if args.kinds else None
    precision_overrides = None
    if args.precision_overrides:
        if os.path.abspath(args.precision_overrides) == os.path.abspath(
            STEP_PRECISION_OVERRIDES
        ):
            precision_overrides = load_step_precision_overrides(
                model, records, calib, args.precision_overrides
            )
        else:
            with open(args.precision_overrides) as stream:
                precision_overrides = json.load(stream)
            if not isinstance(precision_overrides, dict):
                raise ValueError(
                    "precision override JSON must map node names to calibration"
                )
    fp32_only: set[str] = set()
    if args.fp32_optimizer:
        fp32_only = optimizer_nodes(model, load_reference()["state_map"])
    if args.stable_softmax_grad and precision_overrides:
        live = {rec["name"] for rec in records}
        precision_overrides = {
            k: v for k, v in precision_overrides.items() if k in live
        }
    segs, host = build_plan(
        model,
        records,
        calib,
        kinds,
        args.include_unsafe,
        precision_overrides=precision_overrides,
        fp32_only=fp32_only,
    )
    if args.only:
        only = set(args.only.split(","))
        segs = [s for s in segs if s.name in only]
    if args.exclude:
        for sg in [sg for sg in segs if re.search(args.exclude, sg.name)]:
            for n in sg.nodes:
                host[n] = "excluded on the command line"
        segs = [sg for sg in segs if not re.search(args.exclude, sg.name)]
    if args.host_optimizer:
        opt = optimizer_nodes(model, load_reference()["state_map"])
        for sg in [sg for sg in segs if set(sg.nodes) & opt]:
            for n in sg.nodes:
                host[n] = "optimizer update kept in float on the host"
        segs = [sg for sg in segs if not set(sg.nodes) & opt]
    previous_validation = None
    if args.validated:
        with open(args.validated) as f:
            previous_validation = json.load(f)
        failed = validated_failures(
            previous_validation,
            [name for name in args.recheck.split(",") if name],
        )
        for sg in [sg for sg in segs if sg.name in failed]:
            for n in sg.nodes:
                host[n] = (
                    f"device output != simulation in {os.path.basename(args.validated)}"
                )
        segs = [sg for sg in segs if sg.name not in failed]
    if args.limit:
        segs = segs[: args.limit]
    segs, blobs = drop_unemittable(segs, host, args.emit_cache_dir or None)
    ref = load_reference()
    feeds = ref["feeds"]
    grad_names = gradient_tensors(model, ref["state_map"])
    float_grads = _float_gradients(model, feeds, grad_names)
    npu_nodes = sum(len(s.nodes) for s in segs)
    print(
        f"{len(segs)} NPU segments covering {npu_nodes}/{len(model.graph.node)} nodes",
        flush=True,
    )
    if args.emit_dir:
        os.makedirs(args.emit_dir, exist_ok=True)

    npu_nodes = sum(len(s.nodes) for s in segs)
    report: dict[str, Any] = {
        "segments": len(segs),
        "npu_nodes": npu_nodes,
        "nodes": len(model.graph.node),
        "host_reasons": _reason_counts(host),
    }
    t0 = time.time()
    if args.mode == "npu":
        import axcl_session

        with axcl_session.AXSession() as sess:
            report["health_before_lsb"] = axcl_session.health_check(sess)
            runner = StepRunner(
                model,
                segs,
                sess,
                args.emit_dir,
                args.health_every,
                fallback_on_failure=args.fallback_on_failure,
            )
            runner._emitted.update(blobs)
            try:
                outs, stats = runner.run(
                    feeds,
                    "npu",
                    keep=list(grad_names.values()),
                    progress=True,
                    stats_out=(stats := []),
                )
                report["health_after_lsb"] = axcl_session.health_check(sess)
            except axcl_session.DeviceStall as exc:
                report["stall"] = {"error": str(exc), "suspects": runner.stalled}
                report["segment_stats"] = [dataclasses.asdict(s) for s in stats]
                with open(args.out, "w") as f:
                    json.dump(report, f, indent=1)
                print(json.dumps(report["stall"], indent=1))
                return 2
            report["device_exec_ms"] = sess.exec_us / 1000
            report["device_runs"] = sess.runs
    else:
        runner = StepRunner(model, segs, None, args.emit_dir)
        outs, stats = runner.run(
            feeds, args.mode, check=False, keep=list(grad_names.values()), progress=True
        )
    report["wall_s"] = time.time() - t0
    report["per_kind"] = summarize(stats)
    report["segment_stats"] = [dataclasses.asdict(s) for s in stats]
    report["runtime_fallback_segments"] = sum(s.runtime_fallback for s in stats)
    report["runtime_fallback_nodes"] = sum(
        s.nodes for s in stats if s.runtime_fallback
    )
    if previous_validation is not None:
        report["validation_stats"] = merge_validation_stats(previous_validation, stats)

    ref_out = ref["ref"]
    loss_name = "distill__add_27"
    if loss_name in outs and loss_name in ref_out:
        report["loss"] = {
            "run": float(np.ravel(outs[loss_name])[0]),
            "float": float(np.ravel(ref_out[loss_name])[0]),
        }
    # backward-pass gradients (the tensors the Adam update consumes)
    report["grads"] = {
        w: {
            "rel_err": rel_err(outs[t], float_grads[w]),
            "cos": cos(outs[t], float_grads[w]),
        }
        for w, t in grad_names.items()
    }
    # the step's actual result: each weight's update w' - w
    report["updates"] = {}
    for w, out in ref["state_map"].items():
        if w.endswith("__m") or w.endswith("__v") or out not in outs:
            continue
        d_run = np.asarray(outs[out], np.float64) - feeds[w]
        d_ref = np.asarray(ref_out[out], np.float64) - feeds[w]
        report["updates"][w] = {
            "rel_err": rel_err(d_run, d_ref),
            "cos": cos(d_run, d_ref),
        }
    for what in ("grads", "updates"):
        vals = report[what].values()
        report[f"{what}_summary"] = {
            "cos_median": float(np.nanmedian([v["cos"] for v in vals])),
            "cos_min": float(np.nanmin([v["cos"] for v in vals])),
            "rel_err_median": float(np.nanmedian([v["rel_err"] for v in vals])),
            "nan": int(sum(not np.isfinite(v["cos"]) for v in vals)),
        }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("segment_stats", "validation_stats", "grads", "updates")
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
