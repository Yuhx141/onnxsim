"""Targeted tests for the ops the WebGPU EP had no kernel for (Sign, Round, Softsign, Selu, IsNaN,
LogSoftmax, SpaceToDepth, LRN). Run after gen_ops.py; writes into ./m. NaN/Inf inputs are spliced in
with Where, so only bits move. ORT's CPU LRN needs an odd size, so the even-size LRN models have no
host reference: check them with lrn_ref.py."""

import onnx, numpy as np
from onnx import parser, numpy_helper, helper

rng = np.random.default_rng(5)


def mk(name, body, inits=None, opset=20):
    m = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": {opset}]>\ng (float[1,3,32,32] X) => (float[?] Y) {{\n{body}\n}}'
    )
    for k, v in (inits or {}).items():
        m.graph.initializer.append(numpy_helper.from_array(np.asarray(v), k))
    del m.graph.output[:]
    m.graph.output.append(helper.make_tensor_value_info("Y", 1, None))
    onnx.save(m, f"m/{name}.onnx")


f = lambda *s: (rng.standard_normal(s)).astype("f")
# IsNaN with real NaN/Inf/-Inf/denormal bit patterns spliced in via Where (bit moves only)
C = np.zeros((1, 3, 32, 32), "f")
M = np.zeros((1, 3, 32, 32), bool)
idx = rng.choice(3072, 600, replace=False)
vals = np.array(
    [np.nan, np.inf, -np.inf, 1e-42, -0.0, np.float32(np.nan) * -1, 3.4e38], dtype="f"
)
for i, j in enumerate(idx):
    C.flat[j] = vals[i % len(vals)]
    M.flat[j] = True
mk("v_IsNaN", "A=Where(M,C,X)\nB=IsNaN(A)\nY=Cast<to=1>(B)", {"C": C, "M": M})
mk(
    "v_IsNaN9", "A=Where(M,C,X)\nB=IsNaN(A)\nY=Cast<to=1>(B)", {"C": C, "M": M}
) if False else None
# Round: exact halves (ties to even), negatives, large
mk(
    "v_Round_half",
    "A=Mul(X,K)\nB=Add(A,H)\nY=Round(B)",
    {"K": np.float32(10), "H": np.float32(0.5)},
)
mk("v_Round_int", "A=Mul(X,K)\nY=Round(A)", {"K": np.float32(1000)})
mk("v_Round_big", "A=Mul(X,K)\nY=Round(A)", {"K": np.float32(1e7)})
mk("v_Sign", "A=Mul(X,K)\nY=Sign(A)", {"K": np.float32(0)})  # all zeros
mk("v_Sign2", "Y=Sign(X)")
mk("v_Softsign_big", "A=Mul(X,K)\nY=Softsign(A)", {"K": np.float32(1e4)})
for nm, al, ga in [
    ("def", None, None),
    ("a1g2", "1.0", "2.0"),
    ("a05g3", "0.5", "3.0"),
]:
    at = "" if al is None else f"<alpha={al},gamma={ga}>"
    mk("v_Selu_" + nm, f"A=Mul(X,K)\nY=Selu{at}(A)", {"K": np.float32(20)})
# LRN: sizes, params, dims
for s in [1, 2, 3, 4, 5, 7]:
    mk(f"v_LRN_s{s}", f"Y=LRN<size={s}>(X)")
mk(
    "v_LRN_par",
    "A=Mul(X,K)\nY=LRN<size=3,alpha=0.05,beta=0.5,bias=2.0>(A)",
    {"K": np.float32(4)},
)
mk(
    "v_LRN_C1",
    "R=Reshape(X,S)\nY=LRN<size=3>(R)",
    {"S": np.array([1, 1, 96, 32], dtype="i8")},
)
mk(
    "v_LRN_C32",
    "R=Reshape(X,S)\nY=LRN<size=5,alpha=0.0002>(R)",
    {"S": np.array([1, 32, 3, 32], dtype="i8")},
)
mk(
    "v_LRN_3d",
    "R=Reshape(X,S)\nY=LRN<size=3>(R)",
    {"S": np.array([3, 32, 32], dtype="i8")},
) if False else None
# SpaceToDepth block sizes / non-square / batch
for b in [2, 4, 8]:
    mk(f"v_S2D_b{b}", f"Y=SpaceToDepth<blocksize={b}>(X)")
mk(
    "v_S2D_rect",
    "R=Reshape(X,S)\nY=SpaceToDepth<blocksize=2>(R)",
    {"S": np.array([1, 6, 16, 32], dtype="i8")},
)
mk(
    "v_S2D_batch",
    "R=Reshape(X,S)\nY=SpaceToDepth<blocksize=2>(R)",
    {"S": np.array([2, 3, 16, 32], dtype="i8")},
)
mk(
    "v_S2D_D2S_roundtrip",
    'A=SpaceToDepth<blocksize=2>(X)\nB=DepthToSpace<blocksize=2,mode="CRD">(A)\nY=Sub(B,X)',
)
# LogSoftmax: long rows, odd sizes, axes, small values
for ax in [0, 1, 2, 3]:
    mk(
        f"v_LogSoftmax_a{ax}",
        f"A=Mul(X,K)\nY=LogSoftmax<axis={ax}>(A)",
        {"K": np.float32(20)},
    )
mk(
    "v_LogSoftmax_long",
    "R=Reshape(X,S)\nY=LogSoftmax<axis=-1>(R)",
    {"S": np.array([1, 3072], dtype="i8")},
)
mk(
    "v_LogSoftmax_odd",
    "R=Reshape(X,S)\nY=LogSoftmax<axis=-1>(R)",
    {"S": np.array([3, 1024], dtype="i8")},
)
mk("v_LogSoftmax_big", "A=Mul(X,K)\nY=LogSoftmax<axis=3>(A)", {"K": np.float32(300)})
mk("v_LogSoftmax_op11", "Y=LogSoftmax<axis=1>(X)", opset=11)
mk("v_Softmax_op11", "Y=Softmax<axis=1>(X)", opset=11)
mk("v_Selu_op6", "Y=Selu(X)", opset=6) if False else None

# strong-alpha LRN at every window size, with 3 and 12 channels (the default alpha=1e-4 barely
# changes the values, so it would not catch a wrong window)
for s in [1, 2, 3, 4, 5, 6, 7]:
    mk(
        f"w_LRN_s{s}_C3",
        "A=Mul(X,K)\nY=LRN<size=%d,alpha=0.5,beta=0.75,bias=1.0>(A)" % s,
        {"K": np.float32(3)},
    )
    mk(
        f"w_LRN_s{s}_C12",
        "A=Mul(X,K)\nR=Reshape(A,S)\nY=LRN<size=%d,alpha=0.5,beta=0.75,bias=1.0>(R)"
        % s,
        {"K": np.float32(3), "S": np.array([1, 12, 16, 16], dtype="i8")},
    )
