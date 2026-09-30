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
                for name, value in zip(outputs, ReferenceEvaluator(model).run(None, {})):
                    init[name] = value
        for level in range(plan.levels):
            nodes = [n for n, lvl in plan.host_nodes if lvl == level and n.op_type != "Constant" and id(n) not in constant]
            if not nodes:
                continue
            model, external, outputs = _level_model(plan.model, nodes, init)
            self.levels[level] = (ReferenceEvaluator(model), external, outputs)

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


def run_levels(plan, launch: Callable[[], dict[str, np.ndarray]], write_slot: Callable[[int, np.ndarray], None], runner: HostRunner | None = None):
    """Run ``plan.levels`` launches; returns (last boundaries, floats including every host tensor)."""
    runner = runner or HostRunner(plan)  # building it folds constants: callers that run repeatedly pass one in
    floats: dict[str, np.ndarray] = {}
    boundaries = {}
    for level in range(plan.levels):
        boundaries = launch()
        floats.update(runner.boundary_floats(boundaries))
        runner.run_level(level, floats)
        for slot, data in runner.entry_slots(level + 1, floats):
            write_slot(slot, data)
    return boundaries, floats
