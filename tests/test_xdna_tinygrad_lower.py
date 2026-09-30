"""tinygrad-lowered operators for the XDNA layer engine: lookup tables and depthwise convolution semantics."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

import layer_engine as le  # noqa: E402
import layer_engine_nets as nets  # noqa: E402
import tinygrad_lower as tl  # noqa: E402

try:
    tl._tinygrad()
except tl.TinygradUnavailable as exc:  # tinygrad is an optional dependency
    pytest.skip(f"tinygrad unavailable: {exc}", allow_module_level=True)


def _numpy_table(fn, scale_in, scale_out):
    x = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float64) * scale_in
    return np.clip(np.rint(fn(x) / scale_out), -128, 127).astype(np.int8).view(np.uint8)


@pytest.mark.parametrize(
    "op,fn",
    [
        ("Sigmoid", lambda x: 1 / (1 + np.exp(-x))),
        ("HardSwish", lambda x: x * np.clip(x / 6 + 0.5, 0, 1)),
        ("Tanh", np.tanh),
    ],
)
def test_tinygrad_tables_match_the_float_definition(op, fn):
    table = tl.unary_table(op, 1 / 16, 0, True, 1 / 64, 0, True)
    want = _numpy_table(fn, 1 / 16, 1 / 64)
    diff = np.abs(table.view(np.int8).astype(int) - want.view(np.int8).astype(int))
    assert (
        diff.max() <= 1 and (diff > 0).sum() <= 4
    )  # float32 vs float64 only flips exact rounding ties


def test_pointwise_unary_detection_by_execution():
    for op in ("HardSwish", "Sigmoid", "Erf", "Mish", "LeakyRelu"):
        assert tl.is_pointwise_unary(op), op
    assert not tl.is_pointwise_unary("Softmax")  # depends on the whole row


def test_lut_job_reference_and_packing_use_the_table():
    table = nets.silu_table()
    lay = le.layout_for(32, 4, 4)
    job = le.Job(
        "t",
        np.zeros((32, 1, 1, 1), dtype=np.int8),
        np.zeros(32, dtype=np.int32),
        0,
        1,
        lay,
        kind="lut",
        table=table,
    )
    dense = np.random.default_rng(0).integers(0, 256, (16, 32), dtype=np.uint8)
    assert np.array_equal(le.reference(job, dense, None), table[dense])
    packed = le.pack_job(job, le.ENGINE_SLOT_BYTES)
    assert np.array_equal(packed[0, 0, 0, le.DESC_BYTES : le.DESC_BYTES + 256], table)


def test_depthwise_reference_matches_tinygrad_grouped_conv():
    from tinygrad import Tensor

    rng = np.random.default_rng(1)
    channels, size = 16, 6
    weight = rng.integers(-8, 8, (channels, 1, 3, 3), dtype=np.int8)
    bias = rng.integers(-300, 300, channels, dtype=np.int32)
    lay = le.layout_for(channels, size, size)
    job = le.Job("dw", weight, bias, 0, 1, lay, shift=5, clamp=90, kind="dw")
    x = rng.integers(0, 128, (size * size, channels), dtype=np.uint8)
    got = le.reference(job, x, None).view(np.int8)
    image = (
        x.view(np.int8)
        .reshape(1, size, size, channels)
        .transpose(0, 3, 1, 2)
        .astype(np.float32)
    )
    acc = Tensor(image).conv2d(
        Tensor(weight.astype(np.float32)), padding=1, groups=channels
    ).numpy() + bias.reshape(1, -1, 1, 1)
    q = np.clip(
        le._rse(acc.astype(np.int64).transpose(0, 2, 3, 1).reshape(-1, channels), 5),
        -128,
        127,
    )
    want = np.minimum(np.maximum(q, 0), 90).astype(np.int8)
    assert np.array_equal(got, want)


def _qdq_graph():
    """dw3x3(ReLU6) -> HardSwish -> 1x1 (linear) + residual Add: the MobileNet-style pattern, built by hand."""
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(2)
    nodes, inits = [], []

    def const(name, array):
        inits.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def qdq(src, name, scale):
        s, z = const(f"{name}_s", np.float32(scale)), const(f"{name}_z", np.uint8(128))
        nodes.append(
            helper.make_node(
                "QuantizeLinear", [src, s, z], [f"{name}_q"], name=f"{name}_Q"
            )
        )
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [f"{name}_q", s, z],
                [f"{name}_dq"],
                name=f"{name}_DQ",
            )
        )
        return f"{name}_dq"

    def conv(src, name, weight, in_scale, group=1):
        w_scale = 2.0**-4
        wq = const(f"{name}_wq", weight)
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [
                    wq,
                    const(f"{name}_ws", np.float32(w_scale)),
                    const(f"{name}_wz", np.int8(0)),
                ],
                [f"{name}_w"],
                name=f"{name}_wDQ",
            )
        )
        bq = const(f"{name}_bq", np.zeros(weight.shape[0], dtype=np.int8))
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [
                    bq,
                    const(f"{name}_bs", np.float32(in_scale * w_scale)),
                    const(f"{name}_bz", np.int8(0)),
                ],
                [f"{name}_b"],
                name=f"{name}_bDQ",
            )
        )
        nodes.append(
            helper.make_node(
                "Conv",
                [src, f"{name}_w", f"{name}_b"],
                [f"{name}_out"],
                name=name,
                kernel_shape=[weight.shape[2]] * 2,
                pads=[weight.shape[2] // 2] * 4,
                group=group,
            )
        )
        return f"{name}_out"

    x_dq = qdq("input", "x", 2.0**-7)
    dw = conv(
        x_dq, "dw", rng.integers(-8, 8, (16, 1, 3, 3), dtype=np.int8), 2.0**-7, group=16
    )
    nodes.append(
        helper.make_node(
            "Clip",
            [dw, const("lo", np.float32(0)), const("hi", np.float32(6))],
            ["clip"],
            name="clip",
        )
    )
    a_dq = qdq("clip", "a", 2.0**-4)
    nodes.append(helper.make_node("HardSwish", [a_dq], ["hs"], name="hs"))
    h_dq = qdq("hs", "h", 2.0**-4)
    pw = conv(h_dq, "pw", rng.integers(-8, 8, (16, 16, 1, 1), dtype=np.int8), 2.0**-4)
    p_dq = qdq(pw, "p", 2.0**-3)
    nodes.append(helper.make_node("Add", [x_dq, p_dq], ["sum"], name="sum"))
    qdq("sum", "y", 2.0**-3)
    graph = helper.make_graph(
        nodes,
        "mb",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 4, 4])],
        [helper.make_tensor_value_info("y_dq", TensorProto.FLOAT, None)],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])


def test_graph_compiler_lowers_depthwise_relu6_hardswish_and_a_fused_residual():
    from layer_engine_graph import compile_graph

    compiled = compile_graph(_qdq_graph())
    jobs, in_name = compiled.jobs, compiled.input_name
    out_name = next(iter(compiled.boundaries), "y_q")
    assert [job.kind for job in jobs] == ["dw", "lut", "conv"]
    dw, lut, pw = jobs
    assert dw.clamp == 96 and dw.relu  # ReLU6 at scale 2^-4 -> 6 / 0.0625
    assert (
        lut.table is not None and lut.table.dtype == np.uint8 and lut.table.size == 256
    )
    assert (
        pw.res_slot == 0 and pw.res_mode == 2 and not pw.relu
    )  # the Add became the conv's residual epilogue
    assert in_name == "x_q" and out_name == "y_q"
    # the whole chain evaluates end to end with the numpy reference
    x = np.random.default_rng(0).integers(100, 156, (16, 16), dtype=np.uint8)
    maps = {0: x}
    for job in jobs:
        res = maps[job.res_slot] if job.res_slot is not None else None
        maps[job.out_slot] = le.reference(job, maps[job.in_slot], res)
    assert maps[jobs[-1].out_slot].shape == (16, 16)


def test_movement_and_add_job_references():
    rng = np.random.default_rng(4)
    lay = le.layout_for(32, 4, 4)
    a = rng.integers(0, 256, (16, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 32), dtype=np.uint8)
    zero = (np.zeros((32, 1, 1, 1), dtype=np.int8), np.zeros(32, dtype=np.int32))
    # concat with a re-scaled second source
    cat = le.Job(
        "cat",
        np.zeros((64, 1, 1, 1), dtype=np.int8),
        np.zeros(64, dtype=np.int32),
        0,
        2,
        lay,
        kind="copy",
        res_slot=1,
        b_layout=lay,
        copy_spec=[(0, g, 0) for g in range(4)] + [(1, g, 1) for g in range(4)],
    )
    out = le.reference(cat, a, b)
    assert np.array_equal(out[:, :32], a)
    doubled = np.clip((b.astype(np.int64) - 128) * 2, -128, 127) + 128
    assert np.array_equal(out[:, 32:], doubled.astype(np.uint8))
    # split = a channel range copy
    split = le.Job(
        "split",
        *zero,
        0,
        2,
        lay,
        kind="copy",
        copy_spec=[(0, 2 + g, 0) for g in range(2)],
    )
    assert np.array_equal(le.reference(split, a, None), a[:, 16:32])
    # add of two activations with ratios 1:1 -> plain saturating sum
    add = le.Job(
        "add", *zero, 0, 2, lay, kind="add", res_slot=1, b_layout=lay, ea=0, eb=0
    )
    want = (
        np.clip((a.astype(np.int64) - 128) + (b.astype(np.int64) - 128), -128, 127)
        + 128
    )
    assert np.array_equal(le.reference(add, a, b), want.astype(np.uint8))
    # nearest upsample and a 3x3 same-padded max pool
    up = le.Job("up", *zero, 0, 2, lay, kind="up", factor=2)
    fmap = le.reference(up, a, None).reshape(8, 8, 32)
    assert np.array_equal(fmap[::2, ::2], a.reshape(4, 4, 32)) and np.array_equal(
        fmap[1::2, 1::2], a.reshape(4, 4, 32)
    )
    pool = le.Job("pool", *zero, 0, 2, lay, kind="maxpool", factor=3, stride=1)
    pooled = le.reference(pool, a, None).reshape(4, 4, 32)
    assert (
        pooled[1, 1].tolist()
        == a.reshape(4, 4, 32)[0:3, 0:3].reshape(9, 32).max(axis=0).tolist()
    )


def test_subgraph_table_runs_a_sigmoid_mul_chain_through_tinygrad():
    from onnx import helper

    nodes = [
        helper.make_node("Sigmoid", ["x"], ["s"]),
        helper.make_node("Mul", ["x", "s"], ["y"]),
    ]
    table = tl.subgraph_table(
        nodes, "x", "y", {}, 1 / 16, 128, False, 1 / 16, 128, False
    )
    x = (np.arange(256) - 128) / 16.0
    want = np.clip(np.rint(x / (1 + np.exp(-x)) / (1 / 16)) + 128, 0, 255)
    assert np.abs(table.astype(int) - want).max() <= 1  # SiLU


def test_gap_bmul_and_depthwise_5x5_references():
    rng = np.random.default_rng(6)
    lay = le.layout_for(16, 4, 4)
    x = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    zero = (np.zeros((16, 1, 1, 1), dtype=np.int8), np.zeros(16, dtype=np.int32))
    gap = le.Job(
        "gap", *zero, 0, 1, lay, kind="gap", shift=4
    )  # 16 pixels -> mean = sum / 2^4
    pooled = le.reference(gap, x, None)
    mean = (x.astype(np.int64) - 128).sum(axis=0) / 16.0
    assert np.array_equal(
        pooled[0].astype(int) - 128, np.clip(np.rint(mean), -128, 127)
    )
    gate = rng.integers(100, 200, (1, 16), dtype=np.uint8)
    mul = le.Job(
        "mul",
        *zero,
        0,
        2,
        lay,
        kind="bmul",
        res_slot=1,
        b_layout=le.layout_for(16, 1, 1),
        shift=6,
    )
    got = le.reference(mul, x, gate).astype(int) - 128
    want = np.clip(
        np.rint(((x.astype(np.int64) - 128) * (gate.astype(np.int64) - 128)) / 64.0),
        -128,
        127,
    )
    assert np.array_equal(got, want)
    dw5 = le.Job(
        "dw5",
        rng.integers(-4, 4, (16, 1, 5, 5), dtype=np.int8),
        np.zeros(16, dtype=np.int32),
        0,
        3,
        lay,
        shift=6,
        kind="dw",
    )
    out = le.reference(dw5, x, None)
    assert out.shape == (16, 16)
    packed = le.pack_job(dw5, le.ENGINE_SLOT_BYTES)
    assert packed[
        0, 0, 0, le.DESC_BYTES : le.DESC_BYTES + 25 * 64
    ].any()  # all 25 tap vectors are packed
