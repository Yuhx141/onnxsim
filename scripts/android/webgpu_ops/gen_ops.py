import onnx, numpy as np, itertools, os
from onnx import parser, numpy_helper, helper

rng = np.random.default_rng(3)
f = lambda *s: (rng.standard_normal(s) * 0.3).astype("f")
I = lambda *v: np.array(v, dtype="i8")
os.makedirs("m", exist_ok=True)
N = 0


def mk(name, body, inits=None):
    global N
    m = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 20]>\ng (float[1,3,32,32] X) => (float[?] Y) {\n'
        + body
        + "\n}"
    )
    for k, v in (inits or {}).items():
        m.graph.initializer.append(numpy_helper.from_array(np.asarray(v), k))
    del m.graph.output[:]
    m.graph.output.append(helper.make_tensor_value_info("Y", 1, None))
    onnx.save(m, f"m/{name}.onnx")
    N += 1


# unary
for op in [
    "Abs",
    "Neg",
    "Relu",
    "Sigmoid",
    "Tanh",
    "Exp",
    "Erf",
    "Sin",
    "Cos",
    "Floor",
    "Ceil",
    "Sign",
    "Softplus",
    "Softsign",
    "Gelu",
    "Atan",
    "Sinh",
    "Cosh",
    "Asinh",
    "Elu",
    "Selu",
    "HardSwish",
    "Mish",
    "Round",
]:
    mk("u_" + op, f"Y = {op}(X)")
mk("u_Sqrt", "A=Abs(X)\nY=Sqrt(A)")
mk("u_Log", "A=Abs(X)\nB=Add(A,C)\nY=Log(B)", {"C": np.float32(1)})
mk("u_Recip", "A=Abs(X)\nB=Add(A,C)\nY=Reciprocal(B)", {"C": np.float32(1)})
mk("u_LeakyRelu", "Y=LeakyRelu<alpha=0.1>(X)")
mk("u_HardSigmoid", "Y=HardSigmoid<alpha=0.3,beta=0.4>(X)")
mk("u_Clip", "Y=Clip(X,lo,hi)", {"lo": np.float32(-0.2), "hi": np.float32(0.3)})
mk("u_CastInt", "A=Mul(X,C)\nB=Cast<to=6>(A)\nY=Cast<to=1>(B)", {"C": np.float32(10)})
mk("u_CastF16", "A=Cast<to=10>(X)\nY=Cast<to=1>(A)")
mk("u_IsNaN", "A=IsNaN(X)\nY=Cast<to=1>(A)")
# binary w/ const & broadcasts
for op in ["Add", "Sub", "Mul", "Div", "Max", "Min"]:
    mk(f"b_{op}_scalar", f"Y={op}(X,C)", {"C": np.float32(0.7)})
    mk(f"b_{op}_chan", f"Y={op}(X,C)", {"C": f(1, 3, 1, 1) + 1.5})
    mk(f"b_{op}_row", f"Y={op}(X,C)", {"C": f(32) + 1.5})
    mk(f"b_{op}_full", f"Y={op}(X,C)", {"C": f(1, 3, 32, 32) + 1.5})
    mk(f"b_{op}_xx", f"A=Abs(X)\nB=Add(A,K)\nY={op}(X,B)", {"K": np.float32(1)})
mk("b_Pow", "A=Abs(X)\nY=Pow(A,C)", {"C": np.float32(1.7)})
mk("b_Where", "A=Greater(X,Z)\nB=Neg(X)\nY=Where(A,X,B)", {"Z": np.float32(0)})
for op in ["Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"]:
    mk(f"b_{op}", f"A={op}(X,C)\nY=Cast<to=1>(A)", {"C": np.float32(0.1)})
mk(
    "b_And",
    "A=Greater(X,Z)\nB=Less(X,W)\nC=And(A,B)\nY=Cast<to=1>(C)",
    {"Z": np.float32(-0.2), "W": np.float32(0.3)},
)
mk("b_Not", "A=Greater(X,Z)\nB=Not(A)\nY=Cast<to=1>(B)", {"Z": np.float32(0)})
# reductions
for op in [
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceL1",
    "ReduceL2",
    "ReduceSumSquare",
    "ReduceLogSumExp",
]:
    for tag, ax, kd in [
        ("c1", [1], 1),
        ("hw", [2, 3], 1),
        ("hw0", [2, 3], 0),
        ("w", [-1], 1),
        ("all", None, 1),
        ("nc", [0, 1], 0),
    ]:
        if ax is None:
            mk(f"r_{op}_{tag}", f"Y={op}<keepdims={kd}>(X)")
        else:
            mk(f"r_{op}_{tag}", f"Y={op}<keepdims={kd}>(X,A)", {"A": I(*ax)})
for op in ["ArgMax", "ArgMin"]:
    for ax in [1, 2, 3]:
        mk(f"r_{op}_{ax}", f"A={op}<axis={ax},keepdims=1>(X)\nY=Cast<to=1>(A)")
for op in ["Softmax", "LogSoftmax"]:
    for ax in [1, 2, 3, -1]:
        mk(f"s_{op}_{ax}", f"Y={op}<axis={ax}>(X)")
mk("s_CumSum", "Y=CumSum(X,A)", {"A": np.int32(3)})
mk("s_CumSum1", "Y=CumSum(X,A)", {"A": np.int32(1)})
# shape ops
mk("h_Reshape", "Y=Reshape(X,S)", {"S": I(1, 3, 1024)})
mk("h_Reshape2", "Y=Reshape(X,S)", {"S": I(3, -1)})
mk("h_Flatten", "Y=Flatten<axis=1>(X)")
mk("h_Squeeze", "U=Unsqueeze(X,A)\nY=Squeeze(U,A)", {"A": I(0)})
mk("h_Concat1", "Y=Concat<axis=1>(X,X)")
mk("h_Concat3", "Y=Concat<axis=3>(X,X)")
mk("h_Concat2", "Y=Concat<axis=2>(X,X,X)")
mk("h_Split1", "A,B,C=Split<axis=1,num_outputs=3>(X)\nY=Sub(A,C)")
mk(
    "h_Split3", "A,B=Split<axis=3>(X,S)\nY=Concat<axis=1>(A,B)", {"S": I(10, 22)}
) if False else None
mk("h_Split3", "A,B=Split<axis=3,num_outputs=2>(X)\nY=Sub(A,B)")
for nm, st, en, ax, sp in [
    ("a", [0], [16], [3], [1]),
    ("b", [8], [24], [2], [1]),
    ("c", [0], [32], [3], [2]),
    ("d", [1], [3], [1], [1]),
    ("e", [31], [-33], [3], [-1]),
    ("f", [2, 3], [30, 29], [2, 3], [1, 1]),
]:
    mk(
        "h_Slice_" + nm,
        "Y=Slice(X,st,en,ax,sp)",
        {"st": I(*st), "en": I(*en), "ax": I(*ax), "sp": I(*sp)},
    )
for mode in ["constant", "reflect", "edge"]:
    mk("h_Pad_" + mode, f'Y=Pad<mode="{mode}">(X,P)', {"P": I(0, 0, 1, 2, 0, 0, 3, 1)})
mk("h_Tile", "Y=Tile(X,R)", {"R": I(1, 2, 1, 2)})
mk("h_Expand", "Y=Expand(X,S)", {"S": I(2, 3, 32, 32)})
mk("h_Gather1", "Y=Gather<axis=1>(X,G)", {"G": I(2, 0, 2)})
mk("h_Gather3", "Y=Gather<axis=3>(X,G)", {"G": I(31, 0, 5, 5, 7)})
mk(
    "h_GatherE",
    "Y=GatherElements<axis=3>(X,G)",
    {"G": rng.integers(0, 32, (1, 3, 32, 32)).astype("i8")},
)
mk(
    "h_Resize_nn",
    'Y=Resize<mode="nearest">(X,,Sc)',
    {"Sc": np.array([1, 1, 2, 2], dtype="f")},
)
mk(
    "h_Resize_lin",
    'Y=Resize<mode="linear">(X,,Sc)',
    {"Sc": np.array([1, 1, 2, 2], dtype="f")},
)
mk(
    "h_Resize_down",
    'Y=Resize<mode="linear">(X,,Sc)',
    {"Sc": np.array([1, 1, 0.5, 0.5], dtype="f")},
)
mk("h_D2S", "Y=SpaceToDepth<blocksize=2>(X)")
mk("h_S2D", "A=SpaceToDepth<blocksize=2>(X)\nY=DepthToSpace<blocksize=2>(A)")
for p in itertools.permutations(range(4)):
    mk("t_" + "".join(map(str, p)), f"Y=Transpose<perm=[{','.join(map(str, p))}]>(X)")
mk("t_r3", "R=Reshape(X,S)\nY=Transpose<perm=[2,0,1]>(R)", {"S": I(3, 32, 32)})
mk("t_r3b", "R=Reshape(X,S)\nY=Transpose<perm=[1,2,0]>(R)", {"S": I(3, 32, 32)})
mk("t_r2", "R=Reshape(X,S)\nY=Transpose(R)", {"S": I(96, 32)})
# pooling
mk("p_Max2", "Y=MaxPool<kernel_shape=[2,2],strides=[2,2]>(X)")
mk("p_Max3s2", "Y=MaxPool<kernel_shape=[3,3],strides=[2,2],pads=[1,1,1,1]>(X)")
mk("p_Avg2", "Y=AveragePool<kernel_shape=[2,2],strides=[2,2]>(X)")
mk("p_Avg3", "Y=AveragePool<kernel_shape=[3,3],strides=[1,1],pads=[1,1,1,1]>(X)")
mk("p_GMax", "Y=GlobalMaxPool(X)")
mk("p_GAvg", "Y=GlobalAveragePool(X)")
# norms
mk(
    "n_BN",
    "Y=BatchNormalization<epsilon=1e-5>(X,sc,b,mu,var)",
    {"sc": f(3) + 1, "b": f(3), "mu": f(3), "var": np.abs(f(3)) + 0.5},
)
mk("n_IN", "Y=InstanceNormalization<epsilon=1e-5>(X,sc,b)", {"sc": f(3) + 1, "b": f(3)})
mk(
    "n_LNw",
    "Y=LayerNormalization<axis=-1,epsilon=1e-5>(X,sc,b)",
    {"sc": f(32) + 1, "b": f(32)},
)
mk(
    "n_LNc",
    "Y=LayerNormalization<axis=1,epsilon=1e-5>(X,sc,b)",
    {"sc": f(3, 32, 32) + 1, "b": f(3, 32, 32)},
)
mk("n_LRN", "Y=LRN<size=3>(X)")
# matmul/gemm
mk("m_MM4x2", "Y=MatMul(X,W)", {"W": f(32, 16)})
mk("m_MM2x2", "R=Reshape(X,S)\nY=MatMul(R,W)", {"S": I(96, 32), "W": f(32, 40)})
mk("m_MMbatch", "Y=MatMul(X,W)", {"W": f(1, 3, 32, 20)})
mk("m_MMrev", "Y=MatMul(W,X)", {"W": f(1, 3, 16, 32)})
mk("m_MMvec", "R=Reshape(X,S)\nY=MatMul(R,W)", {"S": I(1, 3072), "W": f(3072, 10)})
mk("m_MMbig", "R=Reshape(X,S)\nY=MatMul(R,W)", {"S": I(3, 1024), "W": f(1024, 257)})
mk(
    "m_Gemm",
    "R=Reshape(X,S)\nY=Gemm(R,W,B)",
    {"S": I(3, 1024), "W": f(1024, 17), "B": f(17)},
)
mk(
    "m_GemmT",
    "R=Reshape(X,S)\nY=Gemm<transB=1,alpha=0.5,beta=2.0>(R,W,B)",
    {"S": I(3, 1024), "W": f(17, 1024), "B": f(17)},
)
mk(
    "m_GemmA",
    "R=Reshape(X,S)\nY=Gemm<transA=1>(R,W)",
    {"S": I(1024, 3), "W": f(1024, 7)},
) if False else None


# conv
def conv(nm, cin, cout, k, **kw):
    attrs = ",".join(f"{a}={v}" for a, v in kw.items())
    at = f"<{attrs}>" if attrs else ""
    g = kw.get("group", 1)
    mk("c_" + nm, f"Y=Conv{at}(X,W)", {"W": f(cout, cin // g, k, k)})
    mk(
        "c_" + nm + "_b",
        f"Y=Conv{at}(X,W,B)",
        {"W": f(cout, cin // g, k, k), "B": f(cout)},
    )


conv("k3", 3, 16, 3, pads="[1,1,1,1]")
conv("k1", 3, 16, 1)
conv("k5", 3, 8, 5, pads="[2,2,2,2]")
conv("s2", 3, 8, 3, strides="[2,2]", pads="[1,1,1,1]")
conv("dw", 3, 3, 3, group=3, pads="[1,1,1,1]")
conv("dil", 3, 8, 3, dilations="[2,2]", pads="[2,2,2,2]")
conv("c1", 3, 1, 3, pads="[1,1,1,1]")
conv("c4", 3, 4, 3, pads="[1,1,1,1]")
mk(
    "c_relu",
    "C=Conv<pads=[1,1,1,1]>(X,W,B)\nY=Relu(C)",
    {"W": f(16, 3, 3, 3), "B": f(16)},
)
mk("c_T", "Y=ConvTranspose<strides=[2,2]>(X,W)", {"W": f(3, 8, 2, 2)})
mk("c_T3", "Y=ConvTranspose<strides=[2,2],pads=[1,1,1,1]>(X,W)", {"W": f(3, 8, 3, 3)})
# nchw->1-D-ish
mk("x_TopK", "V,I=TopK<axis=3>(X,K)\nY=V", {"K": I(4)}) if False else None
print(N, "models")
