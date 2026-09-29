"""Count multiply-accumulate FLOPs of Conv / ConvTranspose / MatMul / Gemm nodes (2 * MACs).

    python model_flops.py model.onnx [input_name:d0,d1,...]

Elementwise / normalization / softmax work is not counted, so "utilization" computed from this is a
lower bound on how busy the GPU's arithmetic really is.
"""

import sys

import numpy as np
import onnx
from onnx import shape_inference


def flops(path, overrides=()):
    m = onnx.load(path)
    for spec in overrides:
        name, dims = spec.split(":")
        for i in m.graph.input:
            if i.name == name:
                del i.type.tensor_type.shape.dim[:]
                for d in dims.split(","):
                    i.type.tensor_type.shape.dim.add().dim_value = int(d)
    m = shape_inference.infer_shapes(m)
    shapes = {}
    for v in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input):
        if v.type.HasField("tensor_type") and v.type.tensor_type.HasField("shape"):
            d = [x.dim_value for x in v.type.tensor_type.shape.dim]
            if all(d):
                shapes[v.name] = d
    for i in m.graph.initializer:
        shapes[i.name] = list(i.dims)
    total = 0
    by = {}
    for n in m.graph.node:
        f = 0
        try:
            if n.op_type == "Conv":
                out, w = shapes[n.output[0]], shapes[n.input[1]]
                f = (
                    2 * int(np.prod(out)) * int(np.prod(w[1:]))
                )  # out elems * (Cin/g * kh * kw)
            elif n.op_type == "ConvTranspose":
                x, w = shapes[n.input[0]], shapes[n.input[1]]
                f = 2 * int(np.prod(x)) * int(np.prod(w[1:]))
            elif n.op_type == "MatMul":
                a, b = shapes[n.input[0]], shapes[n.input[1]]
                out = shapes[n.output[0]]
                f = 2 * int(np.prod(out)) * a[-1]
            elif n.op_type == "Gemm":
                a, b = shapes[n.input[0]], shapes[n.input[1]]
                ta = next((x.i for x in n.attribute if x.name == "transA"), 0)
                tb = next((x.i for x in n.attribute if x.name == "transB"), 0)
                M, K = (a[1], a[0]) if ta else (a[0], a[1])
                N = b[0] if tb else b[1]
                f = 2 * M * K * N
        except KeyError:
            f = 0
        if f:
            total += f
            by[n.op_type] = by.get(n.op_type, 0) + f
    return total, by


if __name__ == "__main__":
    t, by = flops(sys.argv[1], sys.argv[2:])
    print(
        f"{sys.argv[1]}: {t / 1e9:.3f} GFLOP  "
        + "  ".join(f"{k}={v / 1e9:.3f}" for k, v in by.items())
    )
