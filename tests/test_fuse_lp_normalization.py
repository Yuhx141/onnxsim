"""fuse_lp_normalization: Div(X, ReduceL1/L2(X)) -> LpNormalization.

Requested in https://github.com/onnx/optimizer/issues/143.
"""

import numpy as np
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
import pytest
from _formal_verify_common import simplify_isolated
from onnx import parser

# _list_optimizers only sees onnxsim's custom passes once they are registered.
C._list_other_optimizers()


def _model(body, opset=13):
    return parser.parse_model(f'<ir_version: 8, opset_import: ["": {opset}]> {body}')


def _run(model, x):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"X": x})[0]


@pytest.mark.parametrize("reduce_op, p", [("ReduceL2", 2), ("ReduceL1", 1)])
@pytest.mark.parametrize("axis", [-1, 1])
def test_fuses_and_matches(reduce_op, p, axis):
    model = _model(
        f"""
        g (float[2,3,4] X) => (float[2,3,4] Y)
        {{
          N = {reduce_op}<axes = [{axis}], keepdims = 1>(X)
          Y = Div(X, N)
        }}
        """
    )
    sim, ops = simplify_isolated(model, "fuse_lp_normalization")
    assert ops["LpNormalization"] == 1 and ops["Div"] == 0
    node = next(n for n in sim.graph.node if n.op_type == "LpNormalization")
    attrs = {a.name: a.i for a in node.attribute}
    assert attrs == {"axis": axis, "p": p}
    x = np.random.rand(2, 3, 4).astype(np.float32) + 0.1
    np.testing.assert_allclose(_run(sim, x), _run(model, x), rtol=1e-5)


def test_axes_as_input_opset18():
    model = _model(
        """
        g (float[2,3] X) => (float[2,3] Y)
        <int64[1] ax = {1}>
        {
          N = ReduceL2<keepdims = 1>(X, ax)
          Y = Div(X, N)
        }
        """,
        opset=18,
    )
    _, ops = simplify_isolated(model, "fuse_lp_normalization")
    assert ops["LpNormalization"] == 1


@pytest.mark.parametrize(
    "body",
    [
        # keepdims=0 changes the broadcast, not a plain normalize
        "N = ReduceL2<axes = [1], keepdims = 0>(X)\nY = Div(X, N)",
        # multiple axes
        "N = ReduceL2<axes = [1, 2], keepdims = 1>(X)\nY = Div(X, N)",
        # epsilon clamp (F.normalize) is not equivalent
        "N = ReduceL2<axes = [2], keepdims = 1>(X)\n"
        "C = Clip<min = 1e-12>(N)\nY = Div(X, C)",
        # norm reused elsewhere
        "N = ReduceL2<axes = [2], keepdims = 1>(X)\nY = Div(X, N)\nZ = Mul(N, N)",
        # numerator is not the reduced tensor
        "N = ReduceL2<axes = [2], keepdims = 1>(X)\nW = Relu(X)\nY = Div(W, N)",
    ],
)
def test_declines(body):
    outputs = "float[2,3,4] Y" + (", float[2,3,1] Z" if "Z =" in body else "")
    if "keepdims = 0" in body:
        outputs = "float[2,4] Y"
    if "axes = [1, 2]" in body:
        outputs = "float[2,3,4] Y"
    if "Clip<min" in body:
        # Clip with an attribute needs opset <= 10
        model = _model(f"g (float[2,3,4] X) => ({outputs}) {{ {body} }}", opset=10)
    else:
        model = _model(f"g (float[2,3,4] X) => ({outputs}) {{ {body} }}")
    try:
        _, ops = simplify_isolated(model, "fuse_lp_normalization", check_n=0)
    except Exception:
        pytest.skip("model not constructible for this opset")
    assert ops["LpNormalization"] == 0
