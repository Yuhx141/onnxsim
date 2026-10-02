"""Host side of a multi-launch engine run: float operators between engine launches.

A graph like YOLO11's C2PSA has float operators (Reshape, MatMul, Softmax, ...) in the middle of the network.
The compiler leaves them as ``Compiled.host_nodes`` (each with the number of host round trips it depends on) and
the tensors that come back into the engine as ``Compiled.entries``. ``run_levels`` drives one full engine
launch per level (the same xclbin and arena every time; earlier levels recompute identical values), evaluates the
host nodes of that level with onnx's reference evaluator and quantizes the re-entering tensors into their pinned
arena slots. The last launch is the one whose boundaries are final.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import layer_engine as le


def _level_model(model: Any, nodes: list, init: dict[str, np.ndarray]):
    produced = {o for n in nodes for o in n.output}
    external = []
    for n in nodes:
        for i in n.input:
            if i and i not in produced and i not in init and i not in external:
                external.append(i)
    outputs = [o for n in nodes for o in n.output]
    graph = helper.make_graph(
        list(nodes), "host_level",
        [helper.make_tensor_value_info(i, TensorProto.FLOAT, None) for i in external],
        [helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in outputs],
        initializer=[numpy_helper.from_array(np.asarray(init[i]), i) for n in nodes for i in n.input if i in init],
    )
    opset = max([o.version for o in model.opset_import if o.domain in ("", "ai.onnx")][0], 19)  # the reference evaluator has no DequantizeLinear-13
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)]), external, outputs


def _fast_ops():
    """onnx's reference ConvTranspose scatters through a python col2im (16 ms for a 64-channel 2x upsample)."""
    from onnx.reference.op_run import OpRun

    class ConvTranspose(OpRun):
        op_domain = ""

        def _run(self, x, w, b=None, auto_pad=None, dilations=None, group=None, kernel_shape=None, output_padding=None, output_shape=None, pads=None, strides=None):
            if (group or 1) != 1 or any(d != 1 for d in (dilations or [1, 1])) or output_shape or x.ndim != 4:
                from onnx.reference.ops.op_conv_transpose import ConvTranspose as Reference

                return Reference._run(self, x, w, b, auto_pad=auto_pad, dilations=dilations, group=group, kernel_shape=kernel_shape, output_padding=output_padding, output_shape=output_shape, pads=pads, strides=strides)
            n, ic, ih, iw = x.shape
            _, oc, kh, kw = w.shape
            sy, sx = strides or (1, 1)
            top, left, bottom, right = pads or (0, 0, 0, 0)
            opy, opx = output_padding or (0, 0)
            full_h, full_w = (ih - 1) * sy + kh + opy, (iw - 1) * sx + kw + opx
            full = np.zeros((n, oc, full_h, full_w), dtype=x.dtype)
            cols = np.einsum("nihw,iokl->nokl hw".replace(" ", ""), x, w, optimize=True)  # [n][oc][kh][kw][ih][iw]
            for ky in range(kh):
                for kx in range(kw):
                    full[:, :, ky : ky + (ih - 1) * sy + 1 : sy, kx : kx + (iw - 1) * sx + 1 : sx] += cols[:, :, ky, kx]
            out = full[:, :, top : full_h - bottom, left : full_w - right]
            if b is not None:
                out = out + b.reshape(1, -1, 1, 1)
            return (out.astype(x.dtype),)

    return [ConvTranspose]


def _evaluator(model):
    """onnxruntime (one thread: these are tiny tensors) when installed, else onnx's reference evaluator."""
    import os

    if not os.environ.get("ENGINE_HOST_REFERENCE"):
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads, opts.log_severity_level = 1, 3
            session = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
            names = [o.name for o in session.get_outputs()]

            class Ort:
                def run(self, _, feeds):
                    return session.run(names, feeds)

            return Ort()
        except Exception:  # noqa: BLE001 - fall back to the reference evaluator
            pass
    from onnx.reference import ReferenceEvaluator

    return ReferenceEvaluator(model, new_ops=_fast_ops())


class HostRunner:
    def __init__(self, plan) -> None:
        from onnx.reference import ReferenceEvaluator

        init = {i.name: numpy_helper.to_array(i) for i in plan.model.graph.initializer}
        for n in plan.model.graph.node:
            if n.op_type == "Constant":
                init[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
        self.plan = plan
        self.levels = {}
        host = [n for n, _ in plan.host_nodes if n.op_type != "Constant"]
        constant = set()  # nodes fed only by initializers (weight DequantizeLinear...) run once, not per launch
        for n in host:
            if n.input and all((not i) or i in init for i in n.input):
                constant.add(id(n))
                model, _, outputs = _level_model(plan.model, [n], init)
                for name, value in zip(outputs, ReferenceEvaluator(model, new_ops=_fast_ops()).run(None, {})):
                    init[name] = value
        for level in range(plan.levels):
            nodes = [n for n, lvl in plan.host_nodes if lvl == level and n.op_type != "Constant" and id(n) not in constant]
            if not nodes:
                continue
            model, external, outputs = _level_model(plan.model, nodes, init)
            self.levels[level] = (_evaluator(model), external, outputs)

    def boundary_floats(self, boundaries: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Dequantize engine boundary tensors ([P][C] uint8) to NCHW float32 under their DQ output names."""
        floats = {}
        for name, data in boundaries.items():
            dq_name, scale = self.plan.dequant.get(name, (None, None))
            if dq_name is None:
                continue
            t = self.plan.boundaries[name]
            image = ((data.astype(np.float32) - t.zero) * np.float32(scale)).reshape(t.layout.h, t.layout.w, -1)
            floats[dq_name] = image.transpose(2, 0, 1)[None].copy()
        return floats

    def run_level(self, level: int, floats: dict[str, np.ndarray]) -> None:
        if level not in self.levels:
            return
        evaluator, external, outputs = self.levels[level]
        values = evaluator.run(None, {i: floats[i] for i in external})
        floats.update(dict(zip(outputs, values)))

    def entry_slots(self, level: int, floats: dict[str, np.ndarray]) -> list[tuple[int, np.ndarray]]:
        """(arena slot, slot bytes) for every tensor re-entering the engine at ``level``."""
        out = []
        for e in self.plan.entries:
            if e.level != level:
                continue
            y = floats[e.float_name][0]  # [C][H][W]
            q = np.clip(np.rint(y / e.scale) + e.zero, 0, 255).astype(np.uint8)
            dense = q.transpose(1, 2, 0).reshape(-1, q.shape[0])
            out.append((e.slot, le.to_arena(dense, e.layout)))
        return out


def run_levels(plan, launch: Callable[[int], dict[str, np.ndarray]], write_slot: Callable[[int, np.ndarray], None], runner: HostRunner | None = None, inputs: dict[str, np.ndarray] | None = None):
    """Run ``plan.levels`` launches; returns (last boundaries, floats including every host tensor)."""
    runner = runner or HostRunner(plan)  # building it folds constants: callers that run repeatedly pass one in
    floats: dict[str, np.ndarray] = dict(inputs or {})
    boundaries = {}
    for level in range(plan.levels):
        fresh = launch(level)  # only the boundaries this launch completes (level == its level) need decoding
        boundaries.update(fresh)
        floats.update(runner.boundary_floats(fresh))
        runner.run_level(level, floats)
        for slot, data in runner.entry_slots(level + 1, floats):
            write_slot(slot, data)
    return boundaries, floats
