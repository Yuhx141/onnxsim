"""Turn a QDQ-quantized ONNX model (as the Hexagon/QNN demo models are) into its float twin.

    python qdq_to_float.py in.onnx out.onnx

Why: the WebGPU EP has no quantized kernels, so timing a QDQ model there measures CPU fallbacks. The
twin keeps the architecture and the (de)quantized weight values but runs in fp32:

- DequantizeLinear of a constant (weights, biases) becomes a float initializer.
- QuantizeLinear -> DequantizeLinear on an activation is removed (the activation stays float).
- A QuantizeLinear that produces a graph output becomes an Identity (the output turns float).
- A uint8 graph input that feeds DequantizeLinear becomes a float32 input holding the same 0..255
  values, and the DequantizeLinear becomes Sub(zero_point) / Mul(scale) on the GPU.

Anything else is left alone, so a model without Q/DQ nodes comes out unchanged. It is a timing
vehicle, not an accuracy-preserving conversion (activation quantization noise is gone).
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _const(name, inits, consts):
    if name in inits:
        return numpy_helper.to_array(inits[name])
    if name in consts:
        return consts[name]
    return None


def _bcast(v, x_ndim, axis):
    v = np.asarray(v)
    if v.ndim == 0 or v.size == 1:
        return v.reshape(())
    shape = [1] * x_ndim
    shape[axis % x_ndim] = v.size
    return v.reshape(shape)


def convert(model):
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    consts = {}
    for n in g.node:
        if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    producer = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    graph_inputs = {i.name: i for i in g.input}

    PASS = {"Transpose", "Reshape", "Identity", "Squeeze", "Unsqueeze", "Flatten"}

    def root_input(t):
        """graph input that t is a pure re-layout of, or None"""
        while t not in graph_inputs:
            p = producer.get(t)
            if p is None or p.op_type not in PASS:
                return None
            t = p.input[0]
        return t

    drop = set()  # ids of nodes to remove
    replace = {}  # id(node) -> nodes that stand in its place (keeps topological order)
    rename = {}  # tensor name -> replacement tensor name
    new_inits = []

    out_types = {o.name: o for o in g.output}
    for n in g.node:
        if n.op_type == "QuantizeLinear" and n.output[0] in out_types:
            # a quantized graph output (e.g. value maps handed to a HVX kernel): keep it float
            drop.add(id(n))
            replace[id(n)] = [helper.make_node("Identity", [n.input[0]], [n.output[0]])]
            out_types[n.output[0]].type.tensor_type.elem_type = TensorProto.FLOAT
            continue
        if n.op_type == "DequantizeLinear":
            x, scale = n.input[0], n.input[1]
            zp = n.input[2] if len(n.input) > 2 and n.input[2] else None
            axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
            xv = _const(x, inits, consts)
            sv = _const(scale, inits, consts)
            zv = _const(zp, inits, consts) if zp else np.zeros((), np.int32)
            if xv is not None and sv is not None and zv is not None:
                out = (xv.astype(np.float64) - _bcast(zv, xv.ndim, axis)) * _bcast(
                    sv, xv.ndim, axis
                )
                new_inits.append(
                    numpy_helper.from_array(out.astype(np.float32), n.output[0])
                )
                drop.add(id(n))
            elif (
                root_input(x)
                and sv is not None
                and zv is not None
                and np.asarray(sv).size == 1
            ):
                # uint8 image input (possibly re-laid-out first): float input, then Sub/Mul on the GPU
                drop.add(id(n))
                zf, sf = n.output[0] + "_zp", n.output[0] + "_scale"
                new_inits.append(
                    numpy_helper.from_array(np.asarray(zv, np.float32).reshape(()), zf)
                )
                new_inits.append(
                    numpy_helper.from_array(np.asarray(sv, np.float32).reshape(()), sf)
                )
                sub = n.output[0] + "_sub"
                replace[id(n)] = [
                    helper.make_node("Sub", [x, zf], [sub]),
                    helper.make_node("Mul", [sub, sf], [n.output[0]]),
                ]
                graph_inputs[
                    root_input(x)
                ].type.tensor_type.elem_type = TensorProto.FLOAT
                del g.value_info[:]  # tensor types between the input and here changed
            else:
                q = producer.get(x)
                if (
                    q is not None
                    and q.op_type == "QuantizeLinear"
                    and len(consumers.get(q.output[0], [])) == 1
                ):
                    rename[n.output[0]] = q.input[
                        0
                    ]  # Q -> DQ pair: pass the float through
                    drop.add(id(n))
                    drop.add(id(q))
        # QuantizeLinear whose output is not consumed by a lone DequantizeLinear stays (reported below)

    kept = []
    for n in g.node:
        if id(n) in replace:
            kept.extend(replace[id(n)])
        elif id(n) not in drop:
            kept.append(n)

    # resolve rename chains and apply
    def res(t):
        while t in rename:
            t = rename[t]
        return t

    for n in kept:
        for i, t in enumerate(n.input):
            n.input[i] = res(t)
    for o in g.output:
        if o.name in rename:  # keep the public output name
            kept.append(helper.make_node("Identity", [res(o.name)], [o.name]))
    del g.node[:]
    g.node.extend(kept)
    drop_inits = {n.output[0] for n in []}
    g.initializer.extend(new_inits)
    # drop now-unused integer initializers
    used = {i for n in g.node for i in n.input}
    keep_inits = [
        i
        for i in g.initializer
        if i.name in used or any(o.name == i.name for o in g.output)
    ]
    del g.initializer[:]
    g.initializer.extend(keep_inits)
    del drop_inits
    left = [
        n.op_type for n in g.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")
    ]
    return model, left


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    m, left = convert(m)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"{src} -> {dst}: {len(m.graph.node)} nodes, leftover Q/DQ: {len(left)}")
