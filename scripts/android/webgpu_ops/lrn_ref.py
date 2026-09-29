import numpy as np, onnx, glob, os, sys

x = np.fromfile("x.bin", dtype="f").reshape(1, 3, 32, 32)


def lrn(v, size, alpha, beta, bias):
    C = v.shape[1]
    sq = v.astype("d") ** 2
    out = np.empty_like(sq)
    for c in range(C):
        lo = max(0, c - (size - 1) // 2)
        hi = min(C - 1, c + size // 2)  # hi = c + ceil((size-1)/2)
        out[:, c] = v[:, c] / (bias + alpha / size * sq[:, lo : hi + 1].sum(1)) ** beta
    return out


bad = 0
for f in sorted(glob.glob("m/[vw]_LRN*.onnx")):
    n = os.path.basename(f)[:-5]
    m = onnx.load(f)
    node = [nd for nd in m.graph.node if nd.op_type == "LRN"][0]
    at = {a.name: (a.i if a.name == "size" else a.f) for a in node.attribute}
    size = at["size"]
    alpha = at.get("alpha", 1e-4)
    beta = at.get("beta", 0.75)
    bias = at.get("bias", 1.0)
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    v = x.copy()
    if "K" in inits:
        v = v * inits["K"]
    if "S" in inits:
        v = v.reshape(inits["S"])
    ref = lrn(v.astype("f"), size, alpha, beta, bias)
    p = (
        np.fromfile(f"out/{n}.bin", dtype="f")
        if os.path.exists(f"out/{n}.bin")
        else None
    )
    if p is None:
        print(n, "NO OUTPUT")
        bad += 1
        continue
    e = np.abs(p.reshape(ref.shape) - ref).max()
    eff = np.abs(ref - v).max()
    ok = e < 1e-4
    bad += not ok
    print(
        f"{n:16s} size={size} maxerr={e:.2e} (LRN changes values by up to {eff:.3f}) {'OK' if ok else 'BAD'}"
    )
print("bad:", bad)
