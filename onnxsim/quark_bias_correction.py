"""Bias correction the way Quark's ONNX flow does it
(``quark.onnx.algorithm.bc.bias_correction``), for the Q/DQ models
:mod:`onnxsim.quark_compat` emits.

:func:`onnxsim.bias_correction.correct_bias` (the general tool) adds a constant
after every layer using the end-to-end error of the whole model. Quark's
version is local and rewrites the *quantized bias constant* instead:

1. for each quantized ``Conv`` / ``Gemm``, the quantized model is run on the
   calibration data and the node's input (the dequantized activation) and its
   output after any following ``Relu`` / ``Clip`` / ``QuantizeLinear`` /
   ``DequantizeLinear`` chain are recorded;
2. the *float* layer (plus the float ``Relu`` / ``Clip`` that follows it) is run
   on those same quantized inputs, so the difference ``float - quantized``
   isolates this layer's own weight, bias and output rounding error;
3. its mean per output channel (4-D ``NCHW`` outputs over batch and space,
   2-D ``Gemm`` outputs over the batch; other ranks are skipped) is added to
   the dequantized bias, damped when large: with ``m = max|mean|`` and
   ``b = max|bias| / 256``, a correction with ``m > b`` and ``m > 0.1`` is
   scaled by ``b / m``;
4. the bias is re-quantized with its *existing* scale (``round(bias / scale)``
   as int32). Layers without a bias are left alone, as there is nothing to
   rewrite.

With power-of-two calibration (``XINT8``) Quark re-derives the bias scale
through its power-of-two quantizer without updating the scale tensor; that is
not reproduced -- the existing scale and zero point are kept (rounded to the
bias dtype's range), see :data:`approximate_methods`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

_CHAIN_Q = ("Relu", "Clip", "QuantizeLinear", "DequantizeLinear")
_CHAIN_F = ("Relu", "Clip")
approximate_methods = ("minmse_pof2", "nonoverflow")


def _consumers(graph: onnx.GraphProto, node: onnx.NodeProto) -> List[onnx.NodeProto]:
    outs = {o for o in node.output if o}
    return [n for n in graph.node if outs & set(n.input)]


def _chain_end(graph: onnx.GraphProto, node: onnx.NodeProto, ops: Sequence[str]):
    """The node after ``node`` along the first-consumer ``ops`` chain, and the
    chain's nodes."""
    chain = [node]
    nxt = _consumers(graph, node)
    while nxt and nxt[0].op_type in ops:
        chain.append(nxt[0])
        nxt = _consumers(graph, nxt[0])
    return chain


def _float_submodel(
    float_model: onnx.ModelProto, chain: Sequence[onnx.NodeProto]
) -> onnx.ModelProto:
    g = float_model.graph
    inits = {t.name: t for t in g.initializer}
    sub = onnx.ModelProto()
    sub.ir_version = float_model.ir_version
    sub.opset_import.extend(float_model.opset_import)
    sub.graph.name = "bc_sub"
    start = chain[0].input[0]
    sub.graph.input.append(
        onnx.helper.make_tensor_value_info(start, onnx.TensorProto.FLOAT, None)
    )
    sub.graph.output.append(onnx.ValueInfoProto(name=chain[-1].output[0]))
    seen = set()
    for n in chain:
        sub.graph.node.append(n)
        for x in n.input:
            if x in inits and x not in seen:
                seen.add(x)
                sub.graph.initializer.append(inits[x])
    return sub


def _session(model: onnx.ModelProto, providers: Optional[Sequence[str]]):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=so,
        providers=list(providers or ["CPUExecutionProvider"]),
    )


def _float_nodes(float_model: onnx.ModelProto) -> List[onnx.NodeProto]:
    return [n for n in float_model.graph.node if n.op_type in ("Conv", "Gemm")]


def correct_bias_quark(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    providers: Optional[Sequence[str]] = None,
    activation_symmetric: bool = False,
) -> onnx.ModelProto:
    """Return ``quant_model`` with Quark's bias correction applied (see the
    module docstring). Float and quantized ``Conv`` / ``Gemm`` nodes are paired
    by name, or by position when unnamed."""
    qm = onnx.ModelProto()
    qm.CopyFrom(quant_model)
    qg = qm.graph
    if not calibration_data:
        return qm

    q_nodes = [n for n in qg.node if n.op_type in ("Conv", "Gemm")]
    f_nodes = _float_nodes(float_model)
    f_by_name = {n.name: n for n in f_nodes if n.name}
    producer = {o: n for n in qg.node for o in n.output}
    inits = {t.name: t for t in qg.initializer}

    plan: List[Tuple[onnx.NodeProto, onnx.NodeProto, str, str, str]] = []
    for i, qn in enumerate(q_nodes):
        fn = f_by_name.get(qn.name) if qn.name else None
        if fn is None and not qn.name and len(f_nodes) == len(q_nodes):
            fn = f_nodes[i]
        if fn is None or len(qn.input) != 3 or not qn.input[2]:
            continue
        end = _chain_end(qg, qn, _CHAIN_Q)[-1]
        plan.append((qn, fn, qn.input[0], end.output[0], qn.input[2]))
    if not plan:
        return qm

    probe = onnx.ModelProto()
    probe.CopyFrom(qm)
    have = {o.name for o in probe.graph.output}
    for _, _, tin, tout, _ in plan:
        for t in (tin, tout):
            if t not in have:
                have.add(t)
                probe.graph.output.append(onnx.ValueInfoProto(name=t))
    sess = _session(probe, providers)
    names = [o.name for o in sess.get_outputs()]
    seen: Dict[str, List[np.ndarray]] = {}
    for batch in calibration_data:
        for k, v in zip(names, sess.run(None, batch)):
            seen.setdefault(k, []).append(v)

    for qn, fn, tin, tout, bias_t in plan:
        dq = producer.get(bias_t)
        if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) < 3:
            continue
        chain = _chain_end(float_model.graph, fn, _CHAIN_F)
        fsess = _session(_float_submodel(float_model, chain), providers)
        f_in = fsess.get_inputs()[0].name
        f_out = [fsess.run(None, {f_in: x})[0] for x in seen[tin]]
        q_out = seen[tout]
        try:
            fo, qo = np.array(f_out), np.array(q_out)
        except ValueError:
            continue
        if qo.ndim == 5:
            diff = np.mean(fo - qo, axis=(0, 1, 3, 4))
        elif qo.ndim == 3:
            diff = np.mean(fo - qo, axis=(0, 1))
        else:
            continue
        b_name, s_name, z_name = dq.input[0], dq.input[1], dq.input[2]
        if not all(x in inits for x in (b_name, s_name, z_name)):
            continue
        b_init, s_init, z_init = inits[b_name], inits[s_name], inits[z_name]
        scale = numpy_helper.to_array(s_init)
        zp = numpy_helper.to_array(z_init)
        bias_q = numpy_helper.to_array(b_init)
        bias_f = ((bias_q.astype(np.float32) - zp.astype(np.float32)) * scale).astype(
            np.float32
        )
        max_diff = np.max(np.abs(diff))
        max_bias = np.max(np.abs(bias_f))
        damp = 1
        plus = max_bias / 256
        if max_diff > plus and max_diff > 0.1:
            damp = damp * plus / max_diff
        new_f = bias_f + diff * damp
        q = (np.asarray(new_f) / scale).round()
        if bias_q.dtype == np.int32:
            q = q.astype(np.int32)
        else:  # power-of-two (XINT8) int8 bias: keep the scale, clip to range
            info = np.iinfo(bias_q.dtype)
            q = np.clip(q + zp, info.min, info.max).astype(bias_q.dtype)
        b_init.CopyFrom(numpy_helper.from_array(q.reshape(b_init.dims), b_name))
    return qm


__all__: Any = ["correct_bias_quark"]
