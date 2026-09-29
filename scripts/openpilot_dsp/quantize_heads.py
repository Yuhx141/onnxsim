"""Write the heads' mixed int8 / int16 weights (§8) into a QDQ driving model as real integer initializers.

`head_bits.py build` rounds the head weights onto the int8 / int16 grid but stores them as float (fake quantization,
for measuring accuracy). This writes the same grid as integers behind DequantizeLinear (per output channel, zero
point 0), so a runner can stream int8 / int16 weights, e.g. tinygrad's ONNX_QDQ_INT_GEMM on the v65 DSP:

  python quantize_heads.py <fp32 model> <QDQ backbone model> out.onnx --w8 results/heads_driving_w8_groups.txt

Groups not in --w8 get int16. The rounding is head_bits.apply's: scale = max|w| / qmax per output channel.
"""

import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper


# head_bits.head_weights / group and quantize.backbone_nodes, repeated here: importing those modules pulls in the
# calibration stack (onnxsim.full_qdq, onnxruntime), which writing the weights doesn't need
def backbone_nodes(model):
    nodes = list(model.graph.node)
    prod = {o: i for i, n in enumerate(nodes) for o in n.output}
    seen, stack = set(), [i for i, n in enumerate(nodes) if n.op_type == "Conv"]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        stack += [prod[x] for x in nodes[i].input if x in prod]
    return [nodes[i] for i in sorted(seen)]


def head_weights(fp32):
    bb = {n.name for n in backbone_nodes(fp32)}
    init = {i.name for i in fp32.graph.initializer}
    out = {}
    for n in fp32.graph.node:
        if n.op_type in ("Gemm", "MatMul") and n.name not in bb and n.input[1] in init:
            tb = any(a.name == "transB" and a.i for a in n.attribute)
            out[n.input[1]] = (
                0 if (n.op_type == "Gemm" and tb) else 1
            )  # output-channel axis
    return out


def group(name):
    return ".".join(name.split(".")[:3])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("fp32")
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument(
        "--w8", required=True, help="file with the comma-separated int8 groups"
    )
    a = ap.parse_args()
    fp32, m = onnx.load(a.fp32), onnx.load(a.base)
    w8 = {g for g in open(a.w8).read().replace("\n", ",").split(",") if g}
    fw = {i.name: numpy_helper.to_array(i) for i in fp32.graph.initializer}
    inits = {i.name: k for k, i in enumerate(m.graph.initializer)}
    users = {}
    for n in m.graph.node:
        for k, x in enumerate(n.input):
            users.setdefault(x, []).append((n, k))
    new_nodes, drop, counts = [], set(), {8: 0, 16: 0}
    for name, ax in head_weights(fp32).items():
        w = fw[name].astype(np.float32)
        if w.ndim != 2 or name not in inits:
            continue
        bits = 8 if group(name) in w8 else 16
        qmax = 127 if bits == 8 else 32767
        s = np.abs(w).max(axis=1 - ax) / qmax
        s[s == 0] = 1
        q = np.clip(np.round(w / np.expand_dims(s, 1 - ax)), -qmax, qmax).astype(
            np.int8 if bits == 8 else np.int16
        )
        qn, sn, zn, dn = name + "/q", name + "/scale", name + "/zp", name + "/dq"
        m.graph.initializer.extend(
            [
                numpy_helper.from_array(q, qn),
                numpy_helper.from_array(s.astype(np.float32), sn),
                numpy_helper.from_array(np.zeros_like(s, dtype=q.dtype), zn),
            ]
        )
        new_nodes.append(
            helper.make_node("DequantizeLinear", [qn, sn, zn], [dn], name=dn, axis=ax)
        )
        for n, k in users.get(name, []):
            n.input[k] = dn
        drop.add(name)
        counts[bits] += w.size
    keep = [i for i in m.graph.initializer if i.name not in drop]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    nodes = new_nodes + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.save(m, a.out)
    print(
        f"{len(drop)} head weights: {counts[8] / 1e6:.1f} M int8, {counts[16] / 1e6:.1f} M int16 -> {a.out}"
    )


if __name__ == "__main__":
    main()
