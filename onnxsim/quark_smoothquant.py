"""SmoothQuant the way Quark's ONNX flow applies it
(``quark.onnx.algorithm.sq.smooth_quant``).

It differs from :func:`onnxsim.smoothquant.apply_smoothquant` (the general
purpose pass) in what it matches and how it scales, and
``QConfig(algo_config=[SmoothQuantConfig(alpha=...)])`` in
:mod:`onnxsim.quark_compat` follows Quark:

- candidates are ``MatMul`` nodes whose weight (input 1) is a 2-D constant;
  ``Gemm`` is *not* smoothed, activations of any rank are (their statistics are
  taken over the last axis);
- ``act_scale[j] = max |X[..., j]|`` over all calibration batches, and
  ``scale[j] = act_scale[j]**alpha / (max_k |W[j, k]| + 1e-9)**(1 - alpha)``;
- the weight rows are multiplied by ``scale`` and ``X * (1 / (scale + 1e-9))``
  is fed to the MatMul through a new ``Mul`` -- one per MatMul, even when
  several MatMuls share an activation (each uses its own weight's range).

There is no epsilon floor on the activation range, so an all-zero activation
channel gets ``scale = 0`` (weight row zeroed, activation multiplied by 1e9),
exactly like Quark.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import onnx
from onnx import numpy_helper


def activation_channel_absmax(
    model: onnx.ModelProto,
    tensors: Sequence[str],
    batches: Sequence[Dict[str, np.ndarray]],
    providers: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Per-last-axis-channel ``max |x|`` of each tensor over all batches."""
    import onnxruntime as ort

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    existing = {o.name for o in probe.graph.output}
    for t in tensors:
        if t not in existing:
            probe.graph.output.append(onnx.ValueInfoProto(name=t))
    sess = ort.InferenceSession(
        probe.SerializeToString(),
        providers=list(providers or ["CPUExecutionProvider"]),
    )
    out: Dict[str, np.ndarray] = {}
    for batch in batches:
        vals = sess.run(list(tensors), batch)
        for name, v in zip(tensors, vals):
            a = np.abs(v.reshape(-1, v.shape[-1])).max(axis=0)
            out[name] = a if name not in out else np.where(out[name] > a, out[name], a)
    return out


def smooth_quant(
    model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    alpha: float = 0.5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Return a smoothed copy of ``model`` (see the module docstring)."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}

    cands: List[int] = []
    for idx, n in enumerate(g.node):
        if n.op_type != "MatMul" or len(n.input) != 2 or n.input[1] not in inits:
            continue
        w = inits[n.input[1]]
        if w.data_type == onnx.TensorProto.FLOAT and len(w.dims) == 2:
            cands.append(idx)
    if not cands or not calibration_data:
        return m

    acts: List[str] = list(dict.fromkeys(g.node[i].input[0] for i in cands))
    act_scale = activation_channel_absmax(m, acts, calibration_data, providers)

    taken = {x for n in g.node for x in list(n.input) + list(n.output)} | set(inits)
    new_nodes: Dict[int, onnx.NodeProto] = {}
    for idx in cands:
        n = g.node[idx]
        w_t = inits[n.input[1]]
        w = numpy_helper.to_array(w_t)
        a = act_scale[n.input[0]]
        if a.shape[0] != w.shape[0]:
            continue
        w_scale = np.max(np.abs(w), axis=1)
        scale = np.power(a, alpha) / np.power(w_scale + 1e-9, 1 - alpha)
        factor = (1.0 / (scale + 1e-9)).astype(np.float32)

        base = f"{n.input[0]}_{n.name}"
        names = []
        for suffix in ("_smooth_scale", "_smooth_output", "_smooth_mul"):
            cand, i = base + suffix, 0
            while cand in taken:
                i += 1
                cand = f"{base}{suffix}_{i}"
            taken.add(cand)
            names.append(cand)
        s_name, o_name, mul_name = names
        g.initializer.append(numpy_helper.from_array(factor, s_name))
        new_nodes[idx] = onnx.helper.make_node(
            "Mul", [n.input[0], s_name], [o_name], name=mul_name
        )
        w_t.CopyFrom(
            numpy_helper.from_array(
                (scale.reshape(-1, 1) * w).astype(w.dtype), w_t.name
            )
        )
        n.input[0] = o_name

    # splice each Mul in directly before its MatMul (keeps topological order)
    nodes = list(g.node)
    del g.node[:]
    for idx, n in enumerate(nodes):
        mul = new_nodes.get(idx)
        if mul is not None:
            g.node.append(mul)
        g.node.append(n)
    return m


def apply_smooth_quant_config(
    model: onnx.ModelProto,
    params: Dict[str, Any],
    extra_options: Dict[str, Any],
    calibration_data: Sequence[Dict[str, np.ndarray]],
) -> onnx.ModelProto:
    """A ``SmoothQuantConfig``'s run; ``extra_options["SmoothAlpha"]`` wins
    over ``params["alpha"]`` as in Quark."""
    alpha = extra_options.get("SmoothAlpha", params.get("alpha", 0.5))
    return smooth_quant(model, calibration_data, alpha=alpha)


__all__ = ["apply_smooth_quant_config", "smooth_quant"]
