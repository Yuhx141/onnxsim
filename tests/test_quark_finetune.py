"""Quark-free tests of :mod:`onnxsim.quark_finetune`, the numpy port of AMD
Quark's ``FastFinetune`` (AdaRound / AdaQuant). The comparison with the real
package lives in ``tests/test_quark_finetune_parity.py``.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc
from onnxsim import quark_finetune as qf
from onnxsim.full_qdq import quantize_full_qdq

rng_global = np.random.default_rng(1234)


def _w(name, *shape, scale=0.3, rng=rng_global):
    return numpy_helper.from_array(
        (rng.standard_normal(shape) * scale).astype(np.float32), name
    )


def _model(body, inits, opset=17, io="float[N,3,8,8] x) => (float[N,4] y"):
    model = parser.parse_model(
        f'<ir_version: 10, opset_import: ["": {opset}]> g ({io}) {{ {body} }}'
    )
    model.graph.initializer.extend(inits)
    return model


def _session(model):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), so)


def _mse(float_model, quantized, x):
    ref = _session(float_model).run(None, {"x": x})[0]
    got = _session(quantized).run(None, {"x": x})[0]
    return float(np.mean((ref - got) ** 2))


@pytest.fixture(scope="module")
def cnn():
    rng = np.random.default_rng(0)
    model = _model(
        """
        c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        r = Relu(c1)
        c2 = Conv<pads=[1,1,1,1], strides=[2,2]>(r, w2, b2)
        r2 = Relu(c2)
        f = Flatten(r2)
        y = Gemm<transB=1>(f, w3, b3)
        """,
        [
            _w("w1", 6, 3, 3, 3, rng=rng),
            _w("b1", 6, scale=0.1, rng=rng),
            _w("w2", 8, 6, 3, 3, rng=rng),
            _w("b2", 8, scale=0.1, rng=rng),
            _w("w3", 4, 128, scale=0.2, rng=rng),
            _w("b3", 4, scale=0.1, rng=rng),
        ],
    )
    data = [
        {"x": rng.standard_normal((4, 3, 8, 8)).astype(np.float32)} for _ in range(4)
    ]
    return model, data


@pytest.fixture(scope="module")
def cnn_q(cnn):
    model, data = cnn
    return quantize_full_qdq(model, calibration_data=data, per_channel=True)


def _codes(model):
    return {
        t.name: numpy_helper.to_array(t)
        for t in model.graph.initializer
        if t.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.INT32)
    }


# -- the numpy ops: forward == ONNX Runtime, gradient == finite differences ----------


def _op_cases():
    r = np.random.default_rng(7)

    def conv(attrs, wshape, xshape, op="Conv", opset=17):
        node = f"<{attrs}>" if attrs else ""
        body = f"y = {op}{node}(x, w)"
        io = f"float[{','.join('N' if i == 0 else str(d) for i, d in enumerate(xshape))}] x) => (float y"
        return _model(body, [_w("w", *wshape, scale=0.5, rng=r)], opset, io), xshape

    cases = {
        "conv-asym-dil": conv(
            "pads=[1,0,2,1], strides=[2,1], dilations=[1,2]", (4, 3, 3, 2), (2, 3, 7, 6)
        ),
        "conv-grouped": conv("group=2, pads=[1,1,1,1]", (4, 2, 3, 3), (2, 4, 5, 5)),
        "conv-1d": conv("pads=[1,2], strides=[2]", (4, 3, 3), (2, 3, 9)),
        "convT": conv(
            "strides=[2,2], pads=[1,0,1,1], dilations=[1,2]",
            (3, 4, 3, 2),
            (2, 3, 4, 5),
            "ConvTranspose",
        ),
        "convT-1d": conv(
            "strides=[2], pads=[1,0]", (3, 4, 3), (2, 3, 5), "ConvTranspose"
        ),
        "conv-3d": conv(
            "pads=[1,0,2,1,0,1], strides=[1,2,1], dilations=[1,1,2]",
            (4, 3, 2, 3, 2),
            (2, 3, 5, 6, 7),
        ),
        "conv-3d-grouped": conv(
            "group=2, pads=[1,1,1,1,1,1]", (4, 2, 3, 3, 3), (2, 4, 4, 5, 5)
        ),
        "convT-grouped": conv(
            "group=2, strides=[2,2], pads=[1,0,1,1]",
            (4, 3, 3, 2),
            (2, 4, 4, 5),
            "ConvTranspose",
        ),
        "convT-3d": conv(
            "strides=[2,1,2], pads=[1,0,1,0,1,1]",
            (3, 2, 3, 2, 2),
            (2, 3, 3, 4, 3),
            "ConvTranspose",
        ),
        "convT-1d-grouped": conv(
            "group=3, strides=[2], dilations=[2]", (6, 2, 3), (2, 6, 5), "ConvTranspose"
        ),
    }
    return cases


@pytest.mark.parametrize("name", list(_op_cases()))
def test_conv_ops_match_onnx_runtime_and_their_gradient_is_exact(name):
    model, xshape = _op_cases()[name]
    node = model.graph.node[0]
    w = numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    x = np.random.default_rng(3).standard_normal(xshape)
    op = (
        qf._ConvTransposeOp(node, w.shape)
        if node.op_type == "ConvTranspose"
        else qf._ConvOp(node, w.shape)
    )
    y, ctx = op.forward(x, w)
    ref = _session(model).run(None, {"x": x.astype(np.float32)})[0]
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-5)
    _check_gradient(op, x, w)


def _check_gradient(op, x, w, act=None):
    blk = qf._Block(
        "t", "X", op, w, None, None, None, None, 1.0, 1.0, None, "", "", "", act, None
    )  # type: ignore[arg-type]
    y0, cache = qf._block_forward(blk, x, w, None)
    y_ref = np.random.default_rng(5).standard_normal(y0.shape)
    _, dw, _ = qf._recon_grad(blk, cache, y0, y_ref)

    def loss(wv):
        y, c = qf._block_forward(blk, x, wv, None)
        return qf._recon_grad(blk, c, y, y_ref)[0]

    num = np.zeros_like(w)
    for idx in np.ndindex(*w.shape):
        e = np.zeros_like(w)
        e[idx] = 1e-6
        num[idx] = (loss(w + e) - loss(w - e)) / 2e-6
    np.testing.assert_allclose(dw, num, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "act",
    [
        qf._Relu(),
        qf._LeakyRelu(0.1),
        qf._Clip(-0.3, 0.4),
        qf._Sigmoid(),
        qf._Tanh(),
        qf._Gelu(),
        qf._Softmax(-1),
        None,
    ],
)
def test_matmul_gradient_through_every_supported_activation(act):
    r = np.random.default_rng(9)
    # [B, T, K] activations: Quark's loss reduces over dim 1 (= T) here
    _check_gradient(
        qf._MatMulOp(False),
        r.standard_normal((3, 5, 6)),
        r.standard_normal((6, 4)),
        act,
    )
    _check_gradient(
        qf._MatMulOp(True), r.standard_normal((5, 6)), r.standard_normal((4, 6)), act
    )


def test_norm_ops_match_onnx_runtime_and_their_gradient_is_exact():
    r = np.random.default_rng(11)
    ln = _model(
        "y = LayerNormalization<axis=-1>(x, g)",
        [
            numpy_helper.from_array(
                1 + r.standard_normal(6).astype(np.float32) * 0.3, "g"
            )
        ],
        opset=17,
        io="float[N,4,6] x) => (float y",
    )
    x = r.standard_normal((3, 4, 6))
    g = numpy_helper.to_array(ln.graph.initializer[0]).astype(np.float64)
    op = qf._LayerNormOp(1e-5)
    np.testing.assert_allclose(
        op.forward(x, g)[0],
        _session(ln).run(None, {"x": x.astype(np.float32)})[0],
        rtol=1e-4,
        atol=1e-5,
    )
    _check_gradient(op, x, g)
    inorm = _model(
        "y = InstanceNormalization(x, g, b)",
        [
            numpy_helper.from_array(
                1 + r.standard_normal(3).astype(np.float32) * 0.3, "g"
            ),
            numpy_helper.from_array(np.zeros(3, np.float32), "b"),
        ],
        io="float[N,3,4,4] x) => (float y",
    )
    x = r.standard_normal((2, 3, 4, 4))
    g = numpy_helper.to_array(inorm.graph.initializer[0]).astype(np.float64)
    op = qf._InstanceNormOp(1e-5)
    np.testing.assert_allclose(
        op.forward(x, g)[0],
        _session(inorm).run(None, {"x": x.astype(np.float32)})[0],
        rtol=1e-4,
        atol=1e-5,
    )
    _check_gradient(op, x, g)


def test_quark_loss_and_beta_schedule():
    # (||q - f||_F over dim 1)^2, averaged over every other dim
    blk = qf._Block(
        "t", "Gemm", qf._MatMulOp(False), np.eye(2), None, None, None, None, 1.0, 1.0,
        None, "", "", "", None, None,
    )  # type: ignore[arg-type]  # fmt: skip
    y = np.array([[1.0, 2.0], [3.0, 5.0]])
    ref = np.zeros((2, 2))
    loss, _, _ = qf._recon_grad(blk, (y, y, y, None), y, ref)
    assert loss == pytest.approx((1 + 4 + 9 + 25) / 2)
    # cosine decay from beta_range[0] (end of warm start) to beta_range[1]
    assert qf._beta(100, 20, (20, 2), 0.2) == pytest.approx(20.0)
    assert qf._beta(100, 60, (20, 2), 0.2) == pytest.approx(11.0)
    assert qf._beta(100, 99, (20, 2), 0.2) == pytest.approx(2.0, abs=1e-2)


# -- block semantics, on a model quantized by onnxsim --------------------------------


def _fq(x, scale, zp, lo, hi):
    return (np.clip(np.round(x / scale) + zp, lo, hi) - zp) * scale


def test_block_loss_is_input_qdq_weight_bias_and_optionally_output_qdq():
    """One Gemm, all samples in one mini-batch, one iteration: the traced loss
    is the numpy value of ``Q/DQ(x) @ w_soft + DQ(b)`` (then ``Q/DQ``) against
    the float layer output."""
    rng = np.random.default_rng(2)
    float_model = _model(
        "y = Gemm<transB=1>(x, w, b)",
        [_w("w", 4, 16, rng=rng), _w("b", 4, scale=0.1, rng=rng)],
        io="float[N,16] x) => (float[N,4] y",
    )
    data = [{"x": rng.standard_normal((8, 16)).astype(np.float32)} for _ in range(2)]
    q = quantize_full_qdq(float_model, calibration_data=data, per_channel=True)
    inits = {t.name: numpy_helper.to_array(t) for t in q.graph.initializer}
    nodes = {n.output[0]: n for n in q.graph.node}
    gemm = next(n for n in q.graph.node if n.op_type == "Gemm")

    def dq(name):
        n = nodes[name]
        codes, s, z = (inits[i].astype(np.float64) for i in n.input)
        return (codes - z) * s

    xin = nodes[gemm.input[0]]
    s_in, z_in = float(inits[xin.input[1]]), float(inits[xin.input[2]])
    x = np.concatenate([d["x"] for d in data]).astype(np.float64)
    wf = numpy_helper.to_array(float_model.graph.initializer[0]).astype(np.float64)
    bf = numpy_helper.to_array(float_model.graph.initializer[1]).astype(np.float64)
    xq = _fq(x, s_in, z_in, 0, 255)
    # AdaRound starts from the soft rounding h(alpha) == w/s - floor(w/s), whose
    # weight is the float weight itself
    w_hat = wf
    b_hat = dq(gemm.input[2])
    y_float = x @ wf.T + bf
    y_out = xq @ w_hat.T + b_hat
    expected = {False: np.mean(np.sum((y_out - y_float) ** 2, axis=1))}
    q_out = next(n for n in q.graph.node if n.input and n.input[0] == gemm.output[0])
    s_o, z_o = float(inits[q_out.input[1]]), float(inits[q_out.input[2]])
    lo, hi = (0, 255) if inits[q_out.input[2]].dtype == np.uint8 else (-128, 127)
    expected[True] = np.mean(
        np.sum((_fq(y_out, s_o, z_o, lo, hi) - y_float) ** 2, axis=1)
    )
    for output_qdq in (False, True):
        trace = []
        qf.finetune(
            float_model,
            q,
            data,
            qf.FinetuneOptions(
                num_iterations=1, batch_size=16, output_qdq=output_qdq, guard=False
            ),
            trace=trace,
        )
        assert trace[0][0][1] == pytest.approx(expected[output_qdq], rel=1e-4)
    assert expected[True] != pytest.approx(expected[False], rel=1e-3)


def test_relu_folded_into_the_output_quantizer_is_trained_as_conv_plus_relu(cnn, cnn_q):
    # quantize_full_qdq drops the Relu and lets the output Q's range clamp at
    # 0; Quark's graphs keep the Relu node, so the block must target the float
    # *Relu* output, or output_qdq would chase negative float values
    model, _ = cnn
    assert [n.op_type for n in cnn_q.graph.node if n.op_type == "Relu"] == []
    blocks = qf._find_blocks(model, cnn_q, qf.FinetuneOptions())
    assert [type(b.act).__name__ for b in blocks] == ["_Relu", "_Relu", "NoneType"]
    assert [b.f_end for b in blocks] == ["r", "r2", "y"]


def test_blocks_found_for_every_target_op_type():
    r = np.random.default_rng(4)
    model = _model(
        """
        c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        n1 = InstanceNormalization(c1, g1, be1)
        a = LeakyRelu<alpha=0.1>(n1)
        ct = ConvTranspose<strides=[2,2], kernel_shape=[2,2]>(a, wt, bt)
        y = Sigmoid(ct)
        """,
        [
            _w("w1", 4, 3, 3, 3, rng=r),
            _w("b1", 4, scale=0.1, rng=r),
            numpy_helper.from_array(
                1 + r.standard_normal(4).astype(np.float32) * 0.2, "g1"
            ),
            _w("be1", 4, scale=0.1, rng=r),
            _w("wt", 4, 2, 2, 2, rng=r),
            _w("bt", 2, scale=0.1, rng=r),
        ],
        io="float[N,3,6,6] x) => (float[N,2,12,12] y",
    )
    data = [{"x": r.standard_normal((4, 3, 6, 6)).astype(np.float32)} for _ in range(3)]
    q = quantize_full_qdq(
        model, calibration_data=data, per_channel=False, int8_constants=True
    )
    out, reports = qf.finetune(
        model, q, data, qf.FinetuneOptions(num_iterations=60, batch_size=4)
    )
    assert [rep.op for rep in reports] == [
        "Conv",
        "InstanceNormalization",
        "ConvTranspose",
    ]
    assert all(rep.error_after <= rep.error_before for rep in reports)
    # target_ops restricts the blocks, in the order of the graph
    _, only = qf.finetune(
        model,
        q,
        data,
        qf.FinetuneOptions(num_iterations=5, target_ops=("ConvTranspose",)),
    )
    assert [rep.op for rep in only] == ["ConvTranspose"]

    ln = _model(
        """
        ln = LayerNormalization<axis=-1>(x, g1, be1)
        h = MatMul(ln, wm1)
        y = MatMul(h, wm2)
        """,
        [
            numpy_helper.from_array(
                1 + r.standard_normal(8).astype(np.float32) * 0.2, "g1"
            ),
            _w("be1", 8, scale=0.1, rng=r),
            _w("wm1", 8, 12, rng=r),
            _w("wm2", 12, 6, rng=r),
        ],
        io="float[N,5,8] x) => (float[N,5,6] y",
    )
    data = [{"x": r.standard_normal((4, 5, 8)).astype(np.float32)} for _ in range(3)]
    q = quantize_full_qdq(
        ln, calibration_data=data, per_channel=True, int8_constants=True
    )
    _, reports = qf.finetune(
        ln, q, data, qf.FinetuneOptions(num_iterations=60, batch_size=4)
    )
    assert [rep.op for rep in reports] == ["LayerNormalization", "MatMul", "MatMul"]


# -- training-loop options -------------------------------------------------------------


def _run(cnn, cnn_q, **kw):
    model, data = cnn
    trace = []
    opts = dict(num_iterations=40, batch_size=2)
    opts.update(kw)
    out, reports = qf.finetune(
        model, cnn_q, data, qf.FinetuneOptions(**opts), trace=trace
    )
    return out, reports, trace


def test_same_seed_same_result_and_batch_size_changes_it(cnn, cnn_q):
    a = _codes(_run(cnn, cnn_q)[0])
    b = _codes(_run(cnn, cnn_q)[0])
    c = _codes(_run(cnn, cnn_q, batch_size=7)[0])
    d = _codes(_run(cnn, cnn_q, seed=3)[0])
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert any(not np.array_equal(a[k], c[k]) for k in a)
    assert any(not np.array_equal(a[k], d[k]) for k in a)


def test_mini_batches_come_from_perm_fn_and_an_invalid_batch_size_means_one(cnn, cnn_q):
    model, data = cnn
    calls = []

    def perm(n):
        calls.append(n)
        return np.arange(n)[::-1]

    qf.finetune(
        model,
        cnn_q,
        data,
        qf.FinetuneOptions(num_iterations=5, batch_size=3),
        perm_fn=perm,
    )
    # one call per iteration and layer, over every sample of every batch
    assert calls == [16] * (5 * 3)
    ref = _run(cnn, cnn_q, num_iterations=10, batch_size=1)[2]
    big = _run(cnn, cnn_q, num_iterations=10, batch_size=10_000)[2]  # > samples: 1
    assert ref == big


def test_early_stop_is_quarks_rule(cnn, cnn_q):
    def quark_rule(losses, num_iter, warm, window, adaround):
        """Quark's loop, verbatim, over a loss sequence (``None``: no stop)."""
        n = window if window > 1 else num_iter / 10
        best, mean = float("inf"), 0.0
        for it, loss in enumerate(losses):
            if it >= num_iter * warm:
                if it % n == n - 1:
                    mean /= n
                    if mean < best:
                        best = mean
                    else:
                        return it
                    mean = 0.0
                else:
                    mean += loss
        return None

    stops = 0
    for window in (1, 4):
        kw = dict(
            algorithm="adaquant",
            learning_rate=0.05,
            num_iterations=80,
            num_batches=window,
        )
        full = _run(cnn, cnn_q, early_stop=False, **kw)[2]
        early = _run(cnn, cnn_q, early_stop=True, **kw)[2]
        # the first layer's loss sequence is identical up to the break
        n0 = len(early[0])
        assert [t[1] for t in full[0][:n0]] == [t[1] for t in early[0]]
        for layer_full, layer_early in zip(full, early):
            # the rule, applied to what each run saw, stops at its last iteration
            # (later layers train on the outputs of earlier, differently
            # stopped, layers, so only their own trace is comparable)
            seen = [t[1] for t in layer_early]
            expect = quark_rule(seen, 80, 0.2, window, False)
            assert len(seen) == (80 if expect is None else expect + 1)
            # and without early stopping it never fires early on its own trace
            assert len(layer_full) == 80
            stops += expect is not None
    assert stops  # the rule really fired somewhere


def test_adaround_early_stop_compares_the_rounding_loss(cnn, cnn_q):
    full = _run(cnn, cnn_q, num_iterations=60, early_stop=False)[2]
    early = _run(cnn, cnn_q, num_iterations=60, early_stop=True, num_batches=5)[2]
    for lf, le in zip(full, early):
        n, best, mean, stop = 5, float("inf"), 0.0, None
        for it, (_, _, rnd) in enumerate(lf):
            if it >= 60 * 0.2:
                if it % n == n - 1:
                    mean /= n
                    if mean < best:
                        best = mean
                    else:
                        stop = it
                        break
                    mean = 0.0
                else:
                    mean += rnd
        assert len(le) == (60 if stop is None else stop + 1)


def test_lr_adjust_swaps_the_learning_rate_of_layers_with_large_error(cnn, cnn_q):
    plain = _codes(_run(cnn, cnn_q, learning_rate=0.0)[0])
    # every layer's error is above 0 -> lr 0 for all of them: nothing moves
    adjusted = _codes(_run(cnn, cnn_q, lr_adjust=(0.0, 0.0))[0])
    never = _codes(_run(cnn, cnn_q, lr_adjust=(1e9, 0.0))[0])
    default = _codes(_run(cnn, cnn_q)[0])
    assert all(np.array_equal(plain[k], adjusted[k]) for k in plain)
    assert any(not np.array_equal(never[k], adjusted[k]) for k in plain)
    assert all(np.array_equal(never[k], default[k]) for k in plain)


def test_sequential_capture_follows_updated_layers_parallel_does_not(cnn, cnn_q):
    seq = _run(cnn, cnn_q, num_iterations=1, batch_size=16, guard=False)[2]
    par = _run(cnn, cnn_q, num_iterations=1, batch_size=16, parallel=True, guard=False)[
        2
    ]
    # the first layer sees the same inputs in both modes ...
    assert seq[0] == par[0]
    # ... later layers do not: their input comes from the already-updated model
    # (num_iterations=1 left layer 1's codes alone, so train longer for that)
    seq = _run(cnn, cnn_q, num_iterations=60, batch_size=16)[2]
    par = _run(cnn, cnn_q, num_iterations=60, batch_size=16, parallel=True)[2]
    assert seq[0] == par[0]
    assert seq[1][0] != par[1][0]


def test_drop_ratio_mixes_quantized_and_float_inputs(cnn, cnn_q):
    full_q = _run(cnn, cnn_q, drop_ratio=1.0, seed=1)[2]
    assert full_q == _run(cnn, cnn_q, drop_ratio=1.0, seed=1)[2]
    mixed_a = _run(cnn, cnn_q, drop_ratio=0.5, seed=1)[2]
    mixed_b = _run(cnn, cnn_q, drop_ratio=0.5, seed=1)[2]
    only_f = _run(cnn, cnn_q, drop_ratio=0.0, seed=1)[2]
    assert mixed_a == mixed_b
    # layer 0 reads the model input, which both models share: nothing to mix
    assert mixed_a[0] == full_q[0] == only_f[0]
    assert mixed_a[1] != full_q[1] and only_f[1] != full_q[1]
    assert mixed_a[1] != only_f[1]


def test_rand_fn_replaces_the_mixing_draw(cnn, cnn_q):
    model, data = cnn
    shapes = []

    def rand(shape):
        shapes.append(shape)
        return np.zeros(shape)  # < drop_ratio everywhere: all quantized

    mixed, _ = qf.finetune(
        model, cnn_q, data,
        qf.FinetuneOptions(num_iterations=20, batch_size=2, drop_ratio=0.5),
        rand_fn=rand,
    )  # fmt: skip
    ones, _ = qf.finetune(
        model,
        cnn_q,
        data,
        qf.FinetuneOptions(num_iterations=20, batch_size=2, drop_ratio=1.0),
    )
    assert shapes and shapes[0][0] == 2
    a, b = _codes(mixed), _codes(ones)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_guard_rejects_a_layer_that_got_worse(cnn, cnn_q):
    # AdaQuant at an absurd learning rate wrecks the weights
    bad, reports, _ = _run(
        cnn, cnn_q, algorithm="adaquant", learning_rate=5.0, num_iterations=30
    )
    assert not any(rep.accepted for rep in reports)
    before, after = _codes(cnn_q), _codes(bad)
    assert all(np.array_equal(before[k], after[k]) for k in before)
    wrecked, rep2, _ = _run(
        cnn,
        cnn_q,
        algorithm="adaquant",
        learning_rate=5.0,
        num_iterations=30,
        guard=False,
    )
    assert all(rep.accepted for rep in rep2)
    assert any(not np.array_equal(before[k], _codes(wrecked)[k]) for k in before)


def test_adaquant_update_bias_only_moves_biases_when_asked(cnn, cnn_q):
    kw = dict(algorithm="adaquant", learning_rate=2e-3, num_iterations=40, guard=False)
    base = _codes(cnn_q)
    off = _codes(_run(cnn, cnn_q, update_bias=False, **kw)[0])
    on = _codes(_run(cnn, cnn_q, update_bias=True, **kw)[0])
    biases = [k for k in base if base[k].dtype == np.int32]
    assert biases and all(np.array_equal(base[k], off[k]) for k in biases)
    assert any(not np.array_equal(base[k], on[k]) for k in biases)
    weights = [k for k in base if base[k].dtype == np.int8]
    assert any(not np.array_equal(base[k], off[k]) for k in weights)


def test_adaround_changes_only_weight_codes(cnn, cnn_q):
    out = _run(cnn, cnn_q)[0]
    a, b = _codes(cnn_q), _codes(out)
    assert any(not np.array_equal(a[k], b[k]) for k in a if a[k].dtype == np.int8)
    assert all(np.array_equal(a[k], b[k]) for k in a if a[k].dtype == np.int32)
    out.graph.initializer.sort(key=lambda t: t.name)
    cnn_q_sorted = onnx.ModelProto()
    cnn_q_sorted.CopyFrom(cnn_q)
    assert [n.name for n in out.graph.node] == [n.name for n in cnn_q.graph.node]
    onnx.checker.check_model(out)


def test_selective_update_never_increases_the_output_distance(cnn, cnn_q):
    model, data = cnn
    out, reports, _ = _run(
        cnn,
        cnn_q,
        selective_update=True,
        num_iterations=20,
        algorithm="adaquant",
        learning_rate=0.02,
    )
    f = [_session(model).run(None, b) for b in data]

    def l2(m):
        s = _session(m)
        return np.mean(
            [np.linalg.norm(s.run(None, b)[0] - r[0]) for b, r in zip(data, f)]
        )

    assert l2(out) <= l2(cnn_q) + 1e-12
    assert any(not r.accepted for r in reports) or l2(out) < l2(cnn_q)


def test_select_max_mem_layer_trains_only_the_largest_block(cnn, cnn_q):
    _, reports, _ = _run(cnn, cnn_q, select_max_mem_layer=True)
    assert len(reports) == 1
    # w3 (4 x 128) dominates the parameters, but the conv outputs are bigger
    assert reports[0].op in ("Conv", "Gemm")
    both = _run(cnn, cnn_q)[1]
    assert len(both) == 3


def test_default_learning_rates_follow_the_algorithm():
    assert qf.FinetuneOptions().lr() == 0.1
    assert qf.FinetuneOptions(algorithm="adaquant").lr() == 1e-5
    assert qf.FinetuneOptions(algorithm="adaquant", learning_rate=3e-4).lr() == 3e-4


def test_unknown_algorithm_and_empty_data_raise(cnn, cnn_q):
    model, data = cnn
    with pytest.raises(ValueError, match="unknown algorithm"):
        qf.finetune(model, cnn_q, data, qf.FinetuneOptions(algorithm="gptq"))
    with pytest.raises(ValueError, match="calibration_data"):
        qf.finetune(model, cnn_q, [])


def test_per_tensor_weights_work_too(cnn):
    model, data = cnn
    q = quantize_full_qdq(model, calibration_data=data, per_channel=False)
    out, reports = qf.finetune(
        model, q, data, qf.FinetuneOptions(num_iterations=40, batch_size=4)
    )
    assert len(reports) == 3
    x = np.random.default_rng(8).standard_normal((32, 3, 8, 8)).astype(np.float32)
    assert _mse(model, out, x) < 1.3 * _mse(model, q, x)


# -- quark_compat wiring ---------------------------------------------------------------


class _Reader:
    def __init__(self, data):
        self.it = iter(data)

    def get_next(self):
        return next(self.it, None)


def _spy(monkeypatch):
    seen = {}

    def fake(float_model, quantized, calibration, opt=None, **kw):
        seen["opt"], seen["n"] = opt, len(calibration)
        return quantized, []

    monkeypatch.setattr(qf, "finetune", fake)
    return seen


def _quantize(cfg, cnn):
    model, data = cnn
    return qc.ModelQuantizer(cfg).quantize_model(
        model, calibration_data_reader=_Reader(data)
    )


def test_presets_carry_quarks_fastfinetune_dict():
    for preset, algo, lr in (
        ("A8W8_ADAROUND", "adaround", 0.1),
        ("A8W8_ADAQUANT", "adaquant", 1e-5),
        ("INT8_CNN_ACCURATE", "adaround", 0.1),
    ):
        (cfg,) = qc.QConfig.get_default_config(preset).algo_config
        assert cfg.name == algo
        assert cfg.params["batch_size"] == 2 and cfg.params["early_stop"] is True
        assert cfg.params["data_size"] == 1000 and cfg.params["learning_rate"] == lr
        assert cfg.params["fixed_seed"] == 1705472343
        assert cfg.params["num_iterations"] == 1000
    # no UpdateBias key in the preset dict: Quark's training default (on) applies
    assert (
        qc.QConfig.get_default_config("A8W8_ADAQUANT")
        .algo_config[0]
        .params["update_bias"]
    )


def test_algo_config_params_reach_the_engine_with_quarks_defaults(monkeypatch, cnn):
    seen = _spy(monkeypatch)
    cfg = qc.QConfig.get_default_config("A8W8")
    cfg.algo_config = [qc.AdaRoundConfig()]
    with pytest.warns(UserWarning, match="FastFinetune"):
        _quantize(cfg, cnn)
    o = seen["opt"]
    # AdaRoundConfig's own defaults
    assert (o.algorithm, o.num_iterations, o.batch_size, o.num_batches) == (
        "adaround",
        1000,
        1,
        1,
    )
    assert (o.drop_ratio, o.early_stop, o.output_qdq, o.lr()) == (
        1.0,
        False,
        False,
        0.1,
    )
    assert o.seed == 1705472343 and not o.update_bias
    cfg.algo_config = [
        qc.AdaQuantConfig(update_bias=True, output_qdq=True, lr_adjust=(1.0, 2.0))
    ]
    with pytest.warns(UserWarning, match="FastFinetune"):
        _quantize(cfg, cnn)
    o = seen["opt"]
    assert (o.algorithm, o.num_iterations, o.lr(), o.update_bias, o.output_qdq) == (
        "adaquant", 3000, 1e-5, True, True
    )  # fmt: skip
    assert o.lr_adjust == (1.0, 2.0)
    # update_bias means nothing to AdaRound, as in Quark
    cfg.algo_config = [qc.AdaRoundConfig(update_bias=True)]
    with pytest.warns(UserWarning):
        _quantize(cfg, cnn)
    assert not seen["opt"].update_bias


def test_every_forwarded_adaround_param_reaches_the_engine(monkeypatch, cnn):
    seen = _spy(monkeypatch)
    cfg = qc.QConfig.get_default_config("A8W8")
    cfg.algo_config = [
        qc.AdaRoundConfig(
            num_iterations=11, learning_rate=0.5, batch_size=3, num_batches=4,
            early_stop=True, drop_ratio=0.25, selective_update=True, output_qdq=True,
            mem_opt_level=0, select_max_mem_layer=True,
            target_op_type=["Conv", "MatMul"], fixed_seed=5, data_size=2,
        )
    ]  # fmt: skip
    with pytest.warns(UserWarning):
        _quantize(cfg, cnn)
    o = seen["opt"]
    assert (o.num_iterations, o.learning_rate, o.batch_size, o.num_batches) == (
        11,
        0.5,
        3,
        4,
    )
    assert (o.early_stop, o.drop_ratio, o.selective_update, o.output_qdq) == (
        True,
        0.25,
        True,
        True,
    )
    assert (o.mem_opt_level, o.select_max_mem_layer) == (0, True)
    assert tuple(o.target_ops) == ("Conv", "MatMul") and o.seed == 5
    assert seen["n"] == 2  # data_size caps the calibration batches


def test_config_fields_quark_never_forwards_are_ignored_but_extra_options_work(
    monkeypatch, cnn
):
    # Quark 0.13's AdaRoundConfig._get_config drops these seven (checked
    # against the real package in test_quark_finetune_parity.py)
    seen = _spy(monkeypatch)
    cfg = qc.QConfig.get_default_config("A8W8")
    cfg.algo_config = [
        qc.AdaRoundConfig(
            reg_param=0.5, beta_range=(10, 1), warm_start=0.3, parallel=True,
            output_index=0, ref_model_path="x.onnx", dynamic_batch=True,
        )
    ]  # fmt: skip
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning, match="does not forward"):
        _quantize_with(q, cnn)
    o = seen["opt"]
    assert (o.reg_param, o.beta_range, o.warm_start, o.parallel, o.output_index) == (
        0.01, (20.0, 2.0), 0.2, False, None
    )  # fmt: skip
    cfg.extra_options["FastFinetune"] = {
        "RegParam": 0.5, "BetaRange": (10, 1), "WarmStart": 0.3, "Parallel": True, "OutputIndex": 0,
    }  # fmt: skip
    with pytest.warns(UserWarning):
        _quantize_with(q, cnn)
    o = seen["opt"]
    assert (o.reg_param, o.beta_range, o.warm_start, o.parallel, o.output_index) == (
        0.5, (10, 1), 0.3, True, 0
    )  # fmt: skip


def test_extra_options_fastfinetune_wins_over_the_algo_config(monkeypatch, cnn):
    seen = _spy(monkeypatch)
    cfg = qc.QConfig.get_default_config("A8W8")
    cfg.algo_config = [qc.AdaRoundConfig(num_iterations=11, batch_size=3)]
    cfg.extra_options["FastFinetune"] = {
        "NumIterations": 7,
        "DropRatio": 0.5,
        "OutputQDQ": True,
    }
    with pytest.warns(UserWarning):
        _quantize(cfg, cnn)
    o = seen["opt"]
    assert (o.num_iterations, o.batch_size, o.drop_ratio, o.output_qdq) == (
        7,
        3,
        0.5,
        True,
    )


def test_quantization_preference_accuracy_applies_quarks_overrides(monkeypatch, cnn):
    seen = _spy(monkeypatch)
    cfg = qc.QConfig.get_default_config("A8W8_ADAQUANT")
    cfg.extra_options["QuantizationPreference"] = "accuracy"
    with pytest.warns(UserWarning):
        _quantize(cfg, cnn)
    o = seen["opt"]
    assert (o.early_stop, o.update_bias, o.output_qdq) == (False, True, True)


def test_weights_stay_per_tensor_for_adaround_and_adaquant(cnn):
    cfg = qc.QConfig.get_default_config("A8W8")
    cfg.algo_config = [qc.AdaRoundConfig(num_iterations=20, batch_size=4)]
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning) as w:
        out = _quantize_with(q, cnn)
    assert not any("per channel" in str(x.message) for x in w)
    onnx.checker.check_model(out)
    scales = [
        numpy_helper.to_array(t)
        for t in out.graph.initializer
        if t.name.endswith("scale")
    ]
    assert any(s.ndim == 0 or s.size == 1 for s in scales)


def _quantize_with(quantizer, cnn):
    model, data = cnn
    return quantizer.quantize_model(model, calibration_data_reader=_Reader(data))


def test_adaquant_end_to_end_and_legacy_engine_switch(cnn):
    cfg = qc.QConfig.get_default_config("A8W8")
    cfg.algo_config = [qc.AdaQuantConfig(num_iterations=20, batch_size=4)]
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning):
        _quantize_with(q, cnn)
    assert q.last_weight_rounding["adaquant"]
    cfg2 = qc.QConfig.get_default_config("A8W8")
    cfg2.algo_config = [qc.AdaQuantConfig(num_iterations=5, legacy_engine=True)]
    q2 = qc.ModelQuantizer(cfg2)
    with pytest.warns(UserWarning, match="per channel"):
        _quantize_with(q2, cnn)
    assert "adaquant" not in q2.last_weight_rounding  # the onnxsim AdaQuant ran


# -- coverage of the layers Quark skips / trains (probed against amd-quark 0.13) ------------
# The comparison with the real package is tests/test_quark_finetune_coverage_parity.py.


def _qdq(model, data, **kw):
    return quantize_full_qdq(
        model, calibration_data=data, per_channel=False, int8_constants=True, **kw
    )


def _fuse_activation(q, op_type):
    """Drop the Q/DQ pair between a compute op and the following ``op_type``
    node, the shape of Quark's own graphs for an activation it keeps."""
    prod = {o: n for n in q.graph.node for o in n.output}
    for act in [n for n in q.graph.node if n.op_type == op_type]:
        dq = prod[act.input[0]]
        qn = prod[dq.input[0]]
        act.input[0] = qn.input[0]
        q.graph.node.remove(qn)
        q.graph.node.remove(dq)
    return q


def _tiny(body, inits, io, shape, n=3, opset=17, seed=0):
    r = np.random.default_rng(seed)
    model = _model(body, inits, opset, io)
    data = [{"x": r.standard_normal(shape).astype(np.float32)} for _ in range(n)]
    return model, data, _qdq(model, data)


def _blocks(model, q, **kw):
    errors: list = []
    blocks = qf._find_blocks(model, q, qf.FinetuneOptions(**kw), errors)
    return blocks, errors


def test_quark_pad_list_is_right_for_2d_and_scrambled_for_3d():
    # 1-D / 2-D: (before, after) per axis, last axis first
    assert qf._quark_pad_list([1, 2]) == [1, 2]
    assert qf._quark_pad_list([1, 2, 3, 4]) == [2, 4, 1, 3]  # [pt,pl,pb,pr]
    # 3-D: "swap H and W" is wrong for three axes, and a list that starts with
    # four zeros collapses to its last pair
    assert qf._quark_pad_list([1, 0, 0, 0, 1, 1]) == [1, 0, 0, 1, 0, 1]
    assert qf._quark_pad_list([0, 1, 0, 0, 2, 0]) == [1, 2]
    assert qf._quark_is_symmetric([1, 2, 3, 1, 2, 3])
    assert not qf._quark_is_symmetric([1, 2, 3, 1, 2, 4])
    node = onnx.helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 0, 0, 0, 1, 1])
    shape = (3, 2, 3, 3, 3)
    assert qf._ConvOp(node, shape).widths == [(1, 0), (0, 1), (0, 1)]  # the real pads
    quirk = qf._ConvOp(node, shape, quark_pads=True)
    assert quirk.widths == [(0, 1), (0, 1), (1, 0)]  # what torch is handed
    assert quirk.pad_layer == [1, 0, 0, 1, 0, 1]
    assert quirk.pad_layer_shape((2, 2, 4, 4, 4)) == [2, 2, 5, 5, 5]
    sym = qf._ConvOp(
        onnx.helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 2, 3, 1, 2, 3]), shape
    )
    assert sym.pad_layer is None and sym.pad_layer_shape((2, 2, 4, 4, 4)) is None


@pytest.mark.parametrize(
    "act",
    [qf._PRelu(), qf._Identity(), qf._Clip(0.0, 6.0, strict=True), qf._Clip(0.0, 1.0)],
)
def test_gradient_through_the_activations_quark_falls_back_to(act):
    r = np.random.default_rng(13)
    _check_gradient(
        qf._MatMulOp(False),
        2 * r.standard_normal((4, 6)),
        r.standard_normal((6, 3)),
        act,
    )


def _act_node(expr, inits=(), opset=13):
    model = _model(
        f"c = Identity(x)\n y = {expr}", inits, opset, "float[N,4] x) => (float y"
    )
    return model.graph.node[-1], {t.name: t for t in model.graph.initializer}


def _const(name, v):
    return numpy_helper.from_array(np.array(v, np.float32), name)


def test_make_act_follows_quarks_convert_act():
    z = np.linspace(-9, 9, 37)
    # PReLU: the module Quark builds has slope 0.25, not the node's
    node, inits = _act_node("PRelu(c, sl)", [_const("sl", [0.1])])
    act = qf._make_act(node, inits)
    assert isinstance(act, qf._PRelu)
    np.testing.assert_allclose(act.forward(z), np.where(z > 0, z, 0.25 * z))
    # Clip with both bounds as initializers is torch.clamp ...
    node, inits = _act_node("Clip(c, lo, hi)", [_const("lo", -1.0), _const("hi", 2.0)])
    act = qf._make_act(node, inits)
    assert (act.lo, act.hi, act.strict) == (-1.0, 2.0, False)
    # ... and every other form that has inputs is the ReLU6 table entry
    for expr, ini in (
        ("Clip(c, lo)", [_const("lo", -1.0)]),  # a min only
        ("Clip(c, , hi)", [_const("hi", 2.0)]),  # no min
        ("Clip(c, lo, hi)", [_const("lo", -1.0)]),  # hi is not an initializer
    ):
        node, inits = _act_node(expr, ini)
        act = qf._make_act(node, inits)
        assert (act.lo, act.hi, act.strict) == (0.0, 6.0, True), expr
    # Clip with attributes (opset < 11): both bounds -> clamp, else the identity
    both = onnx.helper.make_node("Clip", ["c"], ["y"], min=-1.0, max=2.0)
    one = onnx.helper.make_node("Clip", ["c"], ["y"], max=2.0)
    assert isinstance(qf._make_act(both, {}), qf._Clip)
    assert isinstance(qf._make_act(one, {}), qf._Identity)
    bare = onnx.helper.make_node("Clip", ["c"], ["y"])
    assert isinstance(qf._make_act(bare, {}), qf._Identity)
    # a bound with more than one element: Quark's ``.item()`` fails
    node, inits = _act_node(
        "Clip(c, lo, hi)", [_const("lo", [0.0, 1.0]), _const("hi", [2.0, 3.0])]
    )
    assert qf._make_act(node, inits) is None


def test_layers_whose_torch_conversion_raises_are_skipped_and_reported():
    r = np.random.default_rng(1)
    model, data, q = _tiny(
        """
        c1 = Conv<auto_pad="SAME_UPPER", kernel_shape=[3,3]>(x, w1, b1)
        r = Relu(c1)
        y = Conv<pads=[1,1,1,1]>(r, w2, b2)
        """,
        [_w("w1", 4, 3, 3, 3, rng=r), _w("b1", 4, scale=0.1, rng=r),
         _w("w2", 4, 4, 3, 3, rng=r), _w("b2", 4, scale=0.1, rng=r)],
        "float[N,3,6,6] x) => (float[N,4,6,6] y",
        (4, 3, 6, 6),
    )  # fmt: skip
    blocks, errors = _blocks(model, q)
    assert [b.op_type for b in blocks] == ["Conv"] and len(errors) == 1
    assert "auto_pad=SAME_UPPER" in errors[0]
    out, reports = qf.finetune(model, q, data, qf.FinetuneOptions(num_iterations=20))
    assert len(reports) == 1
    # SelectMaxMemLayer converts every layer first and Quark aborts there
    with pytest.raises(NotImplementedError, match="auto_pad=SAME_UPPER"):
        qf.finetune(
            model,
            q,
            data,
            qf.FinetuneOptions(num_iterations=5, select_max_mem_layer=True),
        )
    # VALID is not NOTSET either; NOTSET itself is fine
    for pad, n in (("VALID", 0), ("NOTSET", 1)):
        m2, _, q2 = _tiny(
            f'y = Conv<auto_pad="{pad}", kernel_shape=[3,3]>(x, w1, b1)',
            [_w("w1", 4, 3, 3, 3, rng=r), _w("b1", 4, scale=0.1, rng=r)],
            "float[N,3,6,6] x) => (float y",
            (4, 3, 6, 6),
        )
        assert len(_blocks(m2, q2)[0]) == n


@pytest.mark.parametrize(
    "attrs, in_errors",
    [
        ("output_padding=[0,0]", True),  # even an all-zero attribute raises
        ("output_padding=[1,1]", True),
        ("output_shape=[9,9]", True),
        ("pads=[0,1,1,0]", False),  # a pad layer: the first forward fails
    ],
)
def test_convtranspose_variants_quark_cannot_train_are_skipped(attrs, in_errors):
    r = np.random.default_rng(2)
    model, data, q = _tiny(
        f"y = ConvTranspose<strides=[2,2], kernel_shape=[3,3], {attrs}>(x, w1, b1)",
        [_w("w1", 3, 2, 3, 3, rng=r), _w("b1", 2, scale=0.1, rng=r)],
        "float[N,3,4,4] x) => (float y",
        (4, 3, 4, 4),
    )
    blocks, errors = _blocks(model, q)
    assert blocks == [] and bool(errors) == in_errors


def test_grouped_and_3d_convolutions_are_blocks_that_train():
    r = np.random.default_rng(3)
    cases = {
        "ConvTranspose-grouped": (
            "y = ConvTranspose<group=2, strides=[2,2], pads=[1,1,1,1]>(x, w1, b1)",
            [_w("w1", 4, 3, 3, 3, rng=r), _w("b1", 6, scale=0.1, rng=r)],
            "float[N,4,4,4] x) => (float y",
            (4, 4, 4, 4),
        ),
        "Conv3d": (
            "y = Conv<pads=[1,1,1,1,1,1], strides=[1,2,1]>(x, w1, b1)",
            [_w("w1", 3, 2, 3, 3, 3, rng=r), _w("b1", 3, scale=0.1, rng=r)],
            "float[N,2,4,6,5] x) => (float y",
            (4, 2, 4, 6, 5),
        ),
        "Conv3d-grouped": (
            "y = Conv<group=2, pads=[1,0,1,1,0,1]>(x, w1, b1)",
            [_w("w1", 4, 2, 3, 3, 3, rng=r), _w("b1", 4, scale=0.1, rng=r)],
            "float[N,4,5,5,5] x) => (float y",
            (4, 4, 5, 5, 5),
        ),
    }
    for name, (body, inits, io, shape) in cases.items():
        model, data, q = _tiny(body, inits, io, shape)
        out, reports = qf.finetune(
            model, q, data, qf.FinetuneOptions(num_iterations=60, batch_size=4)
        )
        assert len(reports) == 1, name
        assert reports[0].error_after <= reports[0].error_before, name
        assert reports[0].changed_fraction > 0, name


def test_prelu_block_is_trained_with_torchs_fixed_slope():
    r = np.random.default_rng(4)
    model, data, q = _tiny(
        "c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)\n y = PRelu(c1, sl)",
        [_w("w1", 4, 3, 3, 3, rng=r), _w("b1", 4, scale=0.1, rng=r),
         _const("sl", [[[0.05]], [[0.1]], [[0.2]], [[0.4]]])],
        "float[N,3,6,6] x) => (float y",
        (4, 3, 6, 6),
    )  # fmt: skip
    # a PRelu behind its own Q/DQ is not part of the block; fused it is
    assert _blocks(model, q)[0][0].act is None
    q = _fuse_activation(q, "PRelu")
    (blk,), errors = _blocks(model, q)
    assert isinstance(blk.act, qf._PRelu) and errors == []
    assert blk.f_end == "y"  # the target is the float PRelu output (real slopes)
    _, reports = qf.finetune(
        model, q, data, qf.FinetuneOptions(num_iterations=30, batch_size=4)
    )
    assert len(reports) == 1


def test_gemm_trans_a_trains_only_when_quarks_shapes_line_up():
    r = np.random.default_rng(5)
    io = "float[3,3] x) => (float y"
    inits = [_w("w1", 3, 5, rng=r), _w("b1", 5, scale=0.1, rng=r)]
    one = _model("y = Gemm<transA=1>(x, w1, b1)", inits, 17, io)
    data = [{"x": r.standard_normal((3, 3)).astype(np.float32)}]
    q = _qdq(one, data)
    # one calibration batch, K == M == batch size: the "samples" are rows of A
    trace = []
    _, reports = qf.finetune(
        one, q, data, qf.FinetuneOptions(num_iterations=20, batch_size=3), trace=trace
    )
    assert len(reports) == 1 and len(trace[0]) == 20
    # any other mini-batch / more batches: the matmul does not fit, so the layer
    # is skipped (and leaves no trace behind)
    for kw, batches in (({"batch_size": 2}, 1), ({"batch_size": 3}, 2)):
        d = [
            {"x": r.standard_normal((3, 3)).astype(np.float32)} for _ in range(batches)
        ]
        qd = _qdq(one, d)
        trace = []
        out, reports = qf.finetune(
            one, qd, d, qf.FinetuneOptions(num_iterations=5, **kw), trace=trace
        )
        assert reports == [] and trace == []
        assert all(np.array_equal(v, _codes(qd)[k]) for k, v in _codes(out).items())
    # the op itself is x^T @ W
    op = qf._MatMulOp(False, trans_a=True)
    x, w = r.standard_normal((4, 3)), r.standard_normal((4, 5))
    np.testing.assert_allclose(op.forward(x, w)[0], x.T @ w)
    _check_gradient(op, x, w)


def test_bias_less_gemm_owns_torchs_random_linear_bias():
    r = np.random.default_rng(6)
    model, data, q = _tiny(
        "y = Gemm<transB=1>(x, w1)",
        [_w("w1", 5, 8, rng=r)],
        "float[N,8] x) => (float y",
        (6, 8),
    )
    (blk,), _ = _blocks(model, q)
    assert blk.phantom_bias == pytest.approx(1 / np.sqrt(8))  # U(-1/sqrt(K), 1/sqrt(K))
    assert blk.b_plain.shape == (5,)
    seen = []

    def hook(i, name):
        seen.append(name)
        return {"phantom_bias": np.full(5, 0.5, np.float32)}

    opts = qf.FinetuneOptions(num_iterations=20)
    _, rep1 = qf.finetune(model, q, data, opts, block_hook=hook)
    assert seen and rep1[0].error_before > 0.1  # the bias is part of the output
    # without a hook numpy draws it, inside the bound, reproducibly
    a = qf.finetune(model, q, data, opts)[1]
    b = qf.finetune(model, q, data, opts)[1]
    assert a[0].error_before == b[0].error_before != rep1[0].error_before
    # a Gemm that has a bias keeps it and draws nothing
    model2, _, q2 = _tiny(
        "y = Gemm<transB=1>(x, w1, b1)",
        [_w("w1", 5, 8, rng=r), _w("b1", 5, scale=0.1, rng=r)],
        "float[N,8] x) => (float y",
        (6, 8),
    )
    assert _blocks(model2, q2)[0][0].phantom_bias is None


def test_gemm_bias_runs_along_axis_1_when_that_axis_has_its_length():
    op = qf._MatMulOp(False)
    b = np.arange(4.0)
    np.testing.assert_array_equal(op.add_bias(np.zeros((2, 4, 4)), b)[0, :, 0], b)
    np.testing.assert_array_equal(op.add_bias(np.zeros((2, 3, 4)), b)[0, 0], b)
    np.testing.assert_array_equal(op.add_bias(np.zeros((5, 4)), b)[0], b)
    dy = np.arange(32.0).reshape(2, 4, 4)
    np.testing.assert_array_equal(op.bias_grad(dy), dy.sum(axis=(0, 2)))
    np.testing.assert_array_equal(op.bias_grad(dy[:, :3]), dy[:, :3].sum(axis=(0, 1)))


def test_activation_fake_quantization_is_float32_like_torchs():
    f32 = np.float32
    for scale, zp, lo, hi in ((3.1e-5, 0.0, 0.0, 65535.0), (0.017, 128.0, 0.0, 255.0)):
        a = qf._ActQ(float(f32(scale)), zp, lo, hi)
        x = np.random.default_rng(0).standard_normal(5000) * (hi - lo) * scale / 8
        ref = (
            np.clip(np.round(x.astype(f32) / f32(scale)) + f32(zp), lo, hi) - f32(zp)
        ) * f32(scale)
        np.testing.assert_array_equal(a.fq(x), ref.astype(np.float64))
        y, mask = a.fq_mask(x)
        np.testing.assert_array_equal(y, ref.astype(np.float64))
        assert mask.dtype == bool


# -- MemOptLevel 2 (Quark's DataLoader loop), NumWorkers, DynamicBatch -------------------


@pytest.fixture(scope="module")
def mlp_ll():
    r = np.random.default_rng(7)
    model = _model(
        """
        h = Gemm(x, w1, b1)
        t = Relu(h)
        y = Gemm<transB=1>(t, w2, b2)
        """,
        [_w("w1", 6, 7, rng=r), _w("b1", 7, scale=0.1, rng=r),
         _w("w2", 4, 7, rng=r), _w("b2", 4, scale=0.1, rng=r)],
        io="float[N,6] x) => (float[N,4] y",
    )  # fmt: skip
    data = [{"x": r.standard_normal((1, 6)).astype(np.float32)} for _ in range(10)]
    return model, data, _qdq(model, data)


def _loader_run(mlp, **kw):
    model, data, q = mlp
    calls, trace = [], []

    def perm(n):
        calls.append(n)
        return np.random.default_rng(len(calls)).permutation(n)

    opts = dict(num_iterations=20, batch_size=3, mem_opt_level=2)
    opts.update(kw)
    out, reports = qf.finetune(
        model, q, data, qf.FinetuneOptions(**opts), perm_fn=perm, trace=trace
    )
    return out, reports, calls, trace


def test_mem_opt_level_2_is_an_epoch_loop_that_drops_each_epochs_last_batch(mlp_ll):
    # 10 samples, batch 3: 3 mini-batches per epoch of which only 2 are used, but
    # the number of epochs is ceil(20 / 3) = 7 -- so 14 iterations, not 20 (a
    # quirk of Quark's loop); one shuffle per epoch, not per iteration
    _, reports, calls, trace = _loader_run(mlp_ll)
    assert len(reports) == 2 and [len(t) for t in trace] == [14, 14]
    assert calls == [10] * 14
    # batch size 1: 10 mini-batches, 9 used per epoch, ceil(20 / 10) = 2 epochs
    _, _, calls, trace = _loader_run(mlp_ll, batch_size=1)
    assert [len(t) for t in trace] == [18, 18] and calls == [10] * 4
    # a mini-batch as large as the data is one step per epoch, none of it is used
    _, _, calls, trace = _loader_run(mlp_ll, batch_size=10, num_iterations=7)
    assert [len(t) for t in trace] == [0, 0] and calls == [10] * 14
    # an oversized one means 1
    _, _, _, trace = _loader_run(mlp_ll, batch_size=99)
    assert [len(t) for t in trace] == [18, 18]


def test_mem_opt_level_2_early_stop_has_patience_two_and_per_epoch_means(mlp_ll):
    _, _, calls, trace = _loader_run(
        mlp_ll, num_iterations=400, batch_size=3, early_stop=True, warm_start=0.0
    )
    steps = [len(t) for t in trace]
    # epochs of 2 steps; it stops after the second epoch without improvement
    assert all(s % 2 == 0 and 4 <= s < 400 for s in steps)
    assert len(calls) == sum(steps) // 2


def test_mem_opt_level_2_ignores_lr_adjust_and_parallel_and_needs_workers(mlp_ll):
    base = _loader_run(mlp_ll)[0]
    same = _loader_run(mlp_ll, lr_adjust=(0.0, 5.0), parallel=True)[0]
    assert all(np.array_equal(v, _codes(base)[k]) for k, v in _codes(same).items())
    lvl1 = _loader_run(mlp_ll, mem_opt_level=1, lr_adjust=(0.0, 5.0))[0]
    assert any(not np.array_equal(v, _codes(base)[k]) for k, v in _codes(lvl1).items())
    # DataLoader(persistent_workers, prefetch_factor) fails with NumWorkers=0
    # when the mini-batch is larger than one: every layer is skipped
    out, reports, _, trace = _loader_run(mlp_ll, num_workers=0)
    assert reports == [] and trace == []
    assert all(np.array_equal(v, _codes(mlp_ll[2])[k]) for k, v in _codes(out).items())
    _, reports, _, _ = _loader_run(mlp_ll, num_workers=0, batch_size=1)
    assert len(reports) == 2


def test_mem_opt_level_2_samples_are_whole_calibration_batches():
    r = np.random.default_rng(8)
    arrs = [r.standard_normal((1, 3, 4)) for _ in range(5)]
    assert qf._loader_samples(arrs).shape == (5, 3, 4)  # squeeze(0) of a size-1 axis
    arrs = [r.standard_normal((2, 3, 4)) for _ in range(5)]
    assert qf._loader_samples(arrs).shape == (5, 2, 3, 4)  # no squeeze
    with pytest.raises(qf._SkipLayer):
        qf._loader_samples([np.zeros((1, 3)), np.zeros((2, 3))])


def test_a_conv_cannot_train_on_stacked_batches_at_mem_opt_level_2(cnn, cnn_q):
    model, data = cnn  # batches of 4: torch's conv input becomes 5-D
    _, reports = qf.finetune(
        model, cnn_q, data, qf.FinetuneOptions(num_iterations=5, mem_opt_level=2)
    )
    # the two convolutions fail (as in Quark); the Gemm trains on [bs, 4, 128]
    assert [rep.op for rep in reports] == ["Gemm"]


def test_dynamic_batch_only_works_for_single_sample_batches(mlp_ll, cnn, cnn_q):
    model, data, q = mlp_ll
    opts = dict(num_iterations=10, batch_size=2)
    on = qf.finetune(model, q, data, qf.FinetuneOptions(dynamic_batch=True, **opts))
    off = qf.finetune(model, q, data, qf.FinetuneOptions(**opts))
    assert len(on[1]) == 2 and all(
        np.array_equal(v, _codes(off[0])[k]) for k, v in _codes(on[0]).items()
    )
    # batches of more than one sample: ONNX Runtime rejects the input in every
    # layer, nothing is trained; SelectMaxMemLayer has no guard and raises
    cmodel, cdata = cnn
    out, reports = qf.finetune(
        cmodel, cnn_q, cdata, qf.FinetuneOptions(num_iterations=5, dynamic_batch=True)
    )
    assert reports == []
    assert all(np.array_equal(v, _codes(cnn_q)[k]) for k, v in _codes(out).items())
    with pytest.raises(RuntimeError, match="DynamicBatch"):
        qf.finetune(
            cmodel,
            cnn_q,
            cdata,
            qf.FinetuneOptions(dynamic_batch=True, select_max_mem_layer=True),
        )


def test_dynamic_batch_and_num_workers_reach_the_engine_through_extra_options(
    monkeypatch, cnn
):
    seen = _spy(monkeypatch)
    cfg = qc.QConfig.get_default_config("A8W8")
    cfg.algo_config = [qc.AdaRoundConfig(num_workers=3, dynamic_batch=True)]
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning):
        _quantize_with(q, cnn)
    # the config forwards num_workers (not dynamic_batch), as in Quark
    assert (seen["opt"].num_workers, seen["opt"].dynamic_batch) == (3, False)
    cfg.extra_options["FastFinetune"] = {"DynamicBatch": True, "NumWorkers": 0}
    with pytest.warns(UserWarning):
        _quantize_with(q, cnn)
    assert (seen["opt"].num_workers, seen["opt"].dynamic_batch) == (0, True)


# -- int16 / uint8 / asymmetric weights ------------------------------------------------------


@pytest.mark.parametrize("preset", ["INT16_CNN_ACCURATE", "INT8_CNN_ACCURATE"])
def test_adaround_runs_on_int16_weights_like_on_int8(preset, cnn):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.algo_config[0].params["num_iterations"] = 40
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning):
        out = _quantize_with(q, cnn)
    reports = q.last_weight_rounding["adaround"]
    assert len(reports) == 3 and all(rep.accepted for rep in reports)
    dtype = onnx.TensorProto.INT16 if "16" in preset else onnx.TensorProto.INT8
    weights = [t for t in out.graph.initializer if len(t.dims) >= 2]
    assert weights and all(t.data_type == dtype for t in weights)


def test_finetune_options_accept_asymmetric_uint8_weights(cnn):
    for algo in (qc.AdaRoundConfig, qc.AdaQuantConfig):
        cfg = qc.QConfig.get_default_config("U8U8_AAWA")
        cfg.algo_config = [algo(num_iterations=30, batch_size=4)]
        q = qc.ModelQuantizer(cfg)
        with pytest.warns(UserWarning):
            out = _quantize_with(q, cnn)
        assert not any("instead of uint8" in m for m in q.last_approximations)
        w = [t for t in out.graph.initializer if len(t.dims) >= 2]
        assert w and all(t.data_type == onnx.TensorProto.UINT8 for t in w)
        assert q.last_weight_rounding[algo().name]


def test_gptq_still_needs_int8_weights():
    cfg = qc.QConfig.get_default_config("INT16_CNN_DEFAULT")
    cfg.algo_config = [qc.GPTQConfig()]
    model = _model("y = Gemm(x, w)", [_w("w", 8, 4)], io="float[N,8] x) => (float y")
    with pytest.raises(NotImplementedError, match="gptq"):
        qc.ModelQuantizer(cfg).quantize_model(
            model,
            calibration_data_reader=_Reader([{"x": np.ones((2, 8), np.float32)}]),
        )


def test_matmul_on_a_4d_activation_is_a_block_like_in_quark():
    r = np.random.default_rng(10)
    model, data, q = _tiny(
        "h = MatMul(x, w1)\n t = Relu(h)\n y = MatMul(t, w2)",
        [_w("w1", 8, 6, rng=r), _w("w2", 6, 4, rng=r)],
        "float[N,2,5,8] x) => (float y",
        (4, 2, 5, 8),
    )
    _, reports = qf.finetune(
        model, q, data, qf.FinetuneOptions(num_iterations=30, batch_size=4)
    )
    assert [rep.op for rep in reports] == ["MatMul", "MatMul"]
    assert all(rep.error_after <= rep.error_before for rep in reports)
