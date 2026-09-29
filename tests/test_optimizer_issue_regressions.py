"""Regressions for onnx/optimizer issues #342-#350.

Each test isolates one optimizer pass (``simplify_isolated``) and checks that it
either declines an unsound rewrite or produces a valid, equivalent model.
"""

import collections

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from _formal_verify_common import simplify_isolated
from onnx import parser


def _model(body, opset=13, ir_version=8):
    return parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["": {opset}, "custom": 1]> {body}'
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


# --- #345: Slice after Shape(start=...) -------------------------------------


@pytest.mark.parametrize("end, expected", [(1, [3]), (2, [3, 4])])
def test_slice_after_shape_honours_shape_start(end, expected):
    model = _model(
        f"""
        g (float[2,3,4] X) => (int64[?] Y)
        <int64[1] starts = {{0}}, int64[1] ends = {{{end}}}>
        {{
          s = Shape<start = 1>(X)
          Y = Slice(s, starts, ends)
        }}
        """,
        opset=15,
    )
    sim, _ = simplify_isolated(model, "eliminate_slice_after_shape")
    x = np.zeros((2, 3, 4), np.float32)
    assert list(_run(sim, {"X": x})[0]) == expected
    assert list(_run(model, {"X": x})[0]) == expected


# --- #343: CSE across operator domains --------------------------------------


def test_cse_keeps_nodes_of_different_domains():
    model = _model(
        """
        g (float[3] X) => (float[3] A, float[3] B)
        {
          A = Relu(X)
          B = custom.Relu(X)
        }
        """
    )
    model.functions.extend(
        [
            parser.parse_function(
                """
                <domain: "custom", opset_import: ["": 13]>
                Relu (x) => (y) { y = Neg(x) }
                """
            )
        ]
    )
    sim, _ = simplify_isolated(model, "eliminate_common_subexpression", check_n=0)
    domains = collections.Counter(n.domain for n in sim.graph.node)
    assert domains["custom"] == 1 and domains[""] == 1


# --- #344 / #342: covered by onnxsim's own overrides; keep as guards --------


def test_consecutive_reshape_declines_ambiguous_zero():
    model = _model(
        """
        g (float[2,3,4] X) => (float[6,4] Y)
        <int64[2] s1 = {6, 4}, int64[2] s2 = {0, 4}>
        {
          r = Reshape(X, s1)
          Y = Reshape(r, s2)
        }
        """
    )
    sim, ops = simplify_isolated(model, "fuse_consecutive_reshapes")
    x = np.random.rand(2, 3, 4).astype(np.float32)
    np.testing.assert_array_equal(_run(sim, {"X": x})[0], _run(model, {"X": x})[0])


# --- #346 / #347 / #348: Pad into pools -------------------------------------


def test_avgpool_existing_pads_without_count_include_pad_not_fused():
    model = _model(
        """
        g (float[1,1,4,4] X) => (float[1,1,4,4] Y)
        <int64[8] pads = {0,0,1,1,0,0,1,1}>
        {
          p = Pad(X, pads)
          Y = AveragePool<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(p)
        }
        """
    )
    sim, ops = simplify_isolated(model, "fuse_pad_into_pool")
    assert ops["Pad"] == 1


def test_avgpool_existing_pads_with_count_include_pad_fused():
    model = _model(
        """
        g (float[1,1,4,4] X) => (float[1,1,6,6] Y)
        <int64[8] pads = {0,0,1,1,0,0,1,1}>
        {
          p = Pad(X, pads)
          Y = AveragePool<kernel_shape = [3, 3], pads = [1, 1, 1, 1],
                          count_include_pad = 1>(p)
        }
        """
    )
    sim, ops = simplify_isolated(model, "fuse_pad_into_pool")
    assert ops["Pad"] == 0


def test_maxpool_used_indices_not_fused():
    model = _model(
        """
        g (float[1,1,4,4] X) => (float[1,1,4,4] Y, int64[1,1,4,4] I)
        <int64[8] pads = {0,0,1,1,0,0,1,1},
         float ninf = {-1.0e38}>
        {
          p = Pad(X, pads, ninf)
          Y, I = MaxPool<kernel_shape = [3, 3]>(p)
        }
        """
    )
    sim, ops = simplify_isolated(model, "fuse_pad_into_pool")
    assert ops["Pad"] == 1


# --- #349: legacy Pad value into Conv ---------------------------------------


def test_pad10_nonzero_value_not_fused_into_conv():
    model = _model(
        """
        g (float[1,1,4,4] X) => (float[1,1,4,4] Y)
        <float[1,1,3,3] W = {1,1,1,1,1,1,1,1,1}>
        {
          p = Pad<pads = [0,0,1,1,0,0,1,1], value = 5.0>(X)
          Y = Conv<kernel_shape = [3, 3]>(p, W)
        }
        """,
        opset=10,
    )
    sim, ops = simplify_isolated(model, "fuse_pad_into_conv")
    assert ops["Pad"] == 1


def test_pad10_zero_value_fused_into_conv():
    model = _model(
        """
        g (float[1,1,4,4] X) => (float[1,1,4,4] Y)
        <float[1,1,3,3] W = {1,1,1,1,1,1,1,1,1}>
        {
          p = Pad<pads = [0,0,1,1,0,0,1,1], value = 0.0>(X)
          Y = Conv<kernel_shape = [3, 3]>(p, W)
        }
        """,
        opset=10,
    )
    sim, ops = simplify_isolated(model, "fuse_pad_into_conv")
    assert ops["Pad"] == 0


# --- #350: duplicate initializer names --------------------------------------


def test_extract_constant_gives_unique_initializer_names():
    model = _model(
        """
        g (float[1] X) => (float[1] Y, float[1] Z)
        {
          c1 = Constant<value = float[1] {1.0}>()
          c2 = Constant<value = float[1] {2.0}>()
          Y = Add(X, c1)
          Z = Add(X, c2)
        }
        """
    )
    for node in model.graph.node:
        if node.op_type == "Constant":
            node.attribute[0].t.name = "dup"
    sim, _ = simplify_isolated(model, "extract_constant_to_initializer", check_n=0)
    names = [i.name for i in sim.graph.initializer]
    assert len(names) == len(set(names))
    onnx.checker.check_model(sim)
