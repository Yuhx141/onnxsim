"""Parity of onnxsim's Quark-compat *algorithms* (CLE, stem equalization,
SmoothQuant, BiasCorrection, Quarot R1, AutoMixprecision) against the real AMD
Quark ONNX package. Skipped unless ``quark.onnx`` is importable.

Quark is the ground truth. The float -> float passes (CLE, SmoothQuant, Quarot)
are compared on the transformed weights, which are bit-identical; BiasCorrection
on the integer biases it writes into a Quark-quantized model; AutoMixprecision
on the activation precision each tensor ends up with. See
:mod:`onnxsim.quark_compat` for what is and is not replicated.
"""

import contextlib
import copy
import io
import json
import warnings

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

warnings.filterwarnings("ignore")

with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
):
    try:
        import quark.onnx as quark_onnx
        from quark.onnx.algorithm import (
            bias_correction as quark_bias_correction,
        )
        from quark.onnx.algorithm import (
            cle_transforms,
            smooth_transforms,
            stem_equalize_transforms,
        )
        from quark.onnx.algorithm.cle.equalization import replace_all_clip6_to_relu
    except Exception as e:  # pragma: no cover - environment dependent
        quark_onnx = None
        _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    quark_onnx is None, reason="AMD Quark (amd-quark) is not installed"
)

from onnxsim import quark_compat as qc  # noqa: E402
from onnxsim.quark_bias_correction import correct_bias_quark  # noqa: E402
from onnxsim.quark_equalization import (  # noqa: E402
    apply_cle_config,
    equalize,
    replace_clip6_with_relu,
    stem_equalize,
)
from onnxsim.quark_smoothquant import smooth_quant  # noqa: E402

OPS = ["Conv", "Gemm", "MatMul"]


# -- helpers ---------------------------------------------------------------------


def _w(rng, *shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _with(text, inits):
    m = parser.parse_model(text)
    m.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in inits.items())
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return m


def _inits(model):
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def _assert_same_weights(a, b):
    ia, ib = _inits(a), _inits(b)
    assert set(ia) == set(ib)
    for k in ia:
        np.testing.assert_array_equal(ia[k], ib[k], err_msg=k)


def _changed(a, b):
    ia, ib = _inits(a), _inits(b)
    return any(not np.array_equal(ia[k], ib[k]) for k in ia)


def _conv_chain():
    rng = np.random.default_rng(0)
    return _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[1,3,8,8] x) => (float[1,16,8,8] y) {
            c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0)
            r0 = Relu(c0)
            c1 = Conv<group=8, pads=[1,1,1,1]>(r0, w1, b1)
            r1 = Relu(c1)
            c2 = Conv<group=1>(r1, w2, b2)
            r2 = Relu(c2)
            y = Conv<group=1>(r2, w3)
        }""",
        dict(
            w0=_w(rng, 8, 3, 3, 3, scale=2.0),
            b0=_w(rng, 8),
            w1=_w(rng, 8, 1, 3, 3),
            b1=_w(rng, 8),
            w2=_w(rng, 16, 8, 1, 1, scale=0.3),
            b2=_w(rng, 16),
            w3=_w(rng, 16, 16, 1, 1, scale=0.2),
        ),
    )


def _gemm_chain():
    rng = np.random.default_rng(1)
    return _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,8] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            h2 = Gemm<transB=1>(h1, w2, b2)
            h3 = LeakyRelu<alpha=0.1>(h2)
            y = Gemm(h3, w3)
        }""",
        dict(
            w1=_w(rng, 16, 32, scale=3),
            b1=_w(rng, 32),
            w2=_w(rng, 24, 32),
            b2=_w(rng, 24),
            w3=_w(rng, 24, 8, scale=0.2),
        ),
    )


CHAINS = {"conv": _conv_chain, "gemm": _gemm_chain}


# -- CLE ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CHAINS))
@pytest.mark.parametrize(
    "steps, threshold, append_bias, use_threshold",
    [
        (1, 0.5, True, True),  # Quark's defaults
        (1, 0.5, False, True),
        (1, 0.5, True, False),
        (1, 5.0, True, True),  # most channels below the threshold
        (3, 0.5, True, True),
        (-1, 0.5, True, True),  # adaptive: until the weights settle
        (-1, 5.0, False, False),
    ],
)
def test_cle_options_are_bit_identical_to_quark(
    name, steps, threshold, append_bias, use_threshold
):
    model = CHAINS[name]()
    theirs = cle_transforms(
        copy.deepcopy(model),
        OPS,
        [],
        [],
        steps,
        "max",
        threshold,
        append_bias,
        use_threshold,
        1.9e-7,
    )
    ours = equalize(
        model,
        steps=steps,
        weight_threshold=threshold,
        scale_append_bias=append_bias,
        scale_use_threshold=use_threshold,
    )
    _assert_same_weights(theirs, ours)
    assert name != "conv" or _changed(model, theirs) or threshold > 1


def test_cle_total_layer_diff_threshold_matches_quark():
    model = _conv_chain()
    for diff in (1.0, 1e-3, 1.9e-7):
        theirs = cle_transforms(
            copy.deepcopy(model), OPS, [], [], -1, "max", 0.5, True, True, diff
        )
        ours = equalize(model, steps=-1, total_layer_diff_threshold=diff)
        _assert_same_weights(theirs, ours)


def test_cle_excluded_nodes_match_quark():
    model = _conv_chain()
    for exclude in ([], ["n2_Conv"], ["n0_Conv", "n4_Conv"]):
        theirs = cle_transforms(copy.deepcopy(model), OPS, [], exclude, 3)
        ours = equalize(model, steps=3, nodes_to_exclude=exclude)
        _assert_same_weights(theirs, ours)


def test_cle_without_group_attribute_is_skipped_like_quark():
    rng = np.random.default_rng(2)
    model = _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[1,3,8,8] x) => (float[1,4,8,8] y) {
            c0 = Conv(x, w0, b0)
            r0 = Relu(c0)
            y = Conv(r0, w1)
        }""",
        dict(w0=_w(rng, 8, 3, 3, 3, scale=2), b0=_w(rng, 8), w1=_w(rng, 4, 8, 1, 1)),
    )
    theirs = cle_transforms(copy.deepcopy(model), OPS, [], [], 1)
    assert not _changed(model, theirs)
    _assert_same_weights(theirs, equalize(model))


def _clip_model():
    rng = np.random.default_rng(3)
    return _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[1,3,8,8] x) => (float[1,4,8,8] y) {
            c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0)
            r0 = Clip(c0, lo, hi)
            y = Conv<group=1>(r0, w1)
        }""",
        dict(
            w0=_w(rng, 8, 3, 3, 3, scale=2),
            b0=_w(rng, 8),
            w1=_w(rng, 4, 8, 1, 1),
            lo=np.float32(0),
            hi=np.float32(6),
        ),
    )


@pytest.mark.parametrize("replace", [False, True])
def test_cle_replace_clip6_relu_matches_quark(replace):
    model = _clip_model()
    src = (
        replace_all_clip6_to_relu(copy.deepcopy(model), OPS, [], [])
        if replace
        else copy.deepcopy(model)
    )
    theirs = cle_transforms(src, OPS, [], [], 1)
    ours = equalize(model, replace_clip6=replace)
    _assert_same_weights(theirs, ours)
    # Clip(0, 6) only lets CLE through once it is a Relu
    assert _changed(model, ours) == replace
    ref = replace_clip6_with_relu(copy.deepcopy(model))
    assert "Clip" not in {n.op_type for n in ref.graph.node}


def _stem_model(channels=8):
    rng = np.random.default_rng(4)
    ramp = np.linspace(0.05, 2, channels).astype(np.float32)
    return _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[1,3,8,8] x) => (float[1,4,8,8] y) {
            c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0)
            r0 = Relu(c0)
            bn = BatchNormalization(r0, g, be, mu, var)
            y = Conv<group=1>(bn, w1)
        }""",
        dict(
            w0=_w(rng, channels, 3, 3, 3) * ramp[:, None, None, None],
            b0=_w(rng, channels),
            g=(np.abs(_w(rng, channels)) + 0.5).astype(np.float32),
            be=_w(rng, channels),
            mu=_w(rng, channels),
            var=(np.abs(_w(rng, channels)) + 0.5).astype(np.float32),
            w1=_w(rng, 4, channels, 1, 1),
        ),
    )


def test_stem_equalization_is_bit_identical_to_quark():
    model = _stem_model()
    theirs = stem_equalize_transforms(copy.deepcopy(model), OPS, [], [])
    ours = stem_equalize(model, OPS)
    assert _changed(model, theirs)
    _assert_same_weights(theirs, ours)


def test_stem_equalization_stops_at_unsupported_ops_like_quark():
    # a Sigmoid between the stem and the BatchNorm is not positively homogeneous
    model = _stem_model()
    for n in model.graph.node:
        if n.op_type == "Relu":
            n.op_type = "Sigmoid"
    theirs = stem_equalize_transforms(copy.deepcopy(model), OPS, [], [])
    assert not _changed(model, theirs)
    _assert_same_weights(theirs, stem_equalize(model, OPS))


def test_apply_cle_config_matches_a_quark_run_end_to_end(tmp_path):
    """Quark's implicit CLE (stem equalization, then CLE) on an FP16 model,
    whose weights stay float initializers."""
    rng = np.random.default_rng(11)
    model = _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[1,3,8,8] x) => (float[1,4,8,8] y) {
            c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0)
            r0 = Relu(c0)
            y = Conv<group=1>(r0, w1)
        }""",
        dict(w0=_w(rng, 8, 3, 3, 3, scale=2), b0=_w(rng, 8), w1=_w(rng, 4, 8, 1, 1)),
    )
    q = quark_quantize(model, "FP16", (1, 3, 8, 8), tmp_path, "cle", cle=True)
    ours = apply_cle_config(model, {}, {}, [])
    theirs = _inits(q)
    assert _changed(model, ours)
    for name, arr in _inits(ours).items():
        np.testing.assert_allclose(
            arr, theirs[name].astype(np.float32), rtol=1e-3, atol=1e-5, err_msg=name
        )


# -- SmoothQuant ---------------------------------------------------------------------


def _calibration(shape, outlier_channel, n=4, seed=0):
    rng = np.random.default_rng(seed)
    last = shape[-1]
    boost = 1 + 4 * (np.arange(last) == outlier_channel)
    return [
        {"x": (rng.standard_normal(shape) * boost).astype(np.float32)} for _ in range(n)
    ]


def _sq_transformer():
    rng = np.random.default_rng(5)
    d = 16
    gamma = (1 + 3 * (np.arange(d) == 3)).astype(np.float32)
    return _with(
        f"""<ir_version: 9, opset_import: ["": 17]>
        g (float[2,5,{d}] x) => (float[2,5,{d}] y) {{
            ln = LayerNormalization<axis=-1>(x, gam, bet)
            q = MatMul(ln, wq)
            k = MatMul(ln, wk)
            a = Add(q, k)
            y = MatMul(a, wo)
        }}""",
        dict(
            gam=gamma,
            bet=np.zeros(d, np.float32),
            wq=_w(rng, d, d),
            wk=_w(rng, d, d),
            wo=_w(rng, d, d),
        ),
    )


def _sq_mlp():
    rng = np.random.default_rng(6)
    return _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,16] y) {
            h = MatMul(x, w1)
            r = Relu(h)
            y = MatMul(r, w2)
        }""",
        dict(w1=_w(rng, 16, 16), w2=_w(rng, 16, 16)),
    )


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
@pytest.mark.parametrize(
    "build, shape", [(_sq_transformer, (2, 5, 16)), (_sq_mlp, (3, 16))]
)
def test_smooth_quant_is_bit_identical_to_quark(build, shape, alpha):
    """3-D activations, an activation shared by two MatMuls (each gets its own
    ``Mul`` and its own weight-range scale) and a plain 2-D MLP."""
    model = build()
    data = _calibration(shape, outlier_channel=2)
    theirs = smooth_transforms(copy.deepcopy(model), data, alpha=alpha)
    ours = smooth_quant(model, data, alpha=alpha)
    ti, oi = _inits(theirs), _inits(ours)
    assert set(ti) == set(oi)
    for k in ti:
        np.testing.assert_array_equal(ti[k], oi[k], err_msg=k)
    # same nodes; Quark appends its Muls at the end of the node list, onnxsim
    # keeps the graph topologically sorted
    assert sorted(n.op_type for n in theirs.graph.node) == sorted(
        n.op_type for n in ours.graph.node
    )
    x = data[0]["x"]
    import onnxruntime as ort  # noqa: PLC0415

    def run(m):
        return ort.InferenceSession(m.SerializeToString()).run(None, {"x": x})[0]

    np.testing.assert_array_equal(run(theirs), run(ours))


def test_smooth_quant_leaves_gemm_alone_like_quark():
    rng = np.random.default_rng(7)
    model = _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,16] y) { y = Gemm(x, w, b) }""",
        dict(w=_w(rng, 16, 16), b=_w(rng, 16)),
    )
    data = _calibration((3, 16), 2)
    theirs = smooth_transforms(copy.deepcopy(model), data, alpha=0.5)
    ours = smooth_quant(model, data, alpha=0.5)
    assert [n.op_type for n in theirs.graph.node] == ["Gemm"]
    _assert_same_weights(theirs, ours)
    _assert_same_weights(model, ours)


def quark_quantize(model, preset, shape, tmp_path, tag="m", cle=False):
    """Quark's preset quantization; ``cle`` turns on its (otherwise implicit)
    stem equalization + CLE."""
    from quark.onnx import ModelQuantizer, QConfig

    src, dst = str(tmp_path / f"{tag}.onnx"), str(tmp_path / f"{tag}_q.onnx")
    onnx.save(model, src)
    reader, _ = _reader(shape)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        cfg = QConfig.get_default_config(preset)
        cfg.global_quant_config.include_cle = cle
        ModelQuantizer(cfg).quantize_model(src, dst, reader())
    return onnx.load(dst)


# -- BiasCorrection --------------------------------------------------------------------


def _reader(shape, n=6):
    from onnxruntime.quantization import CalibrationDataReader

    rng = np.random.default_rng(3)
    data = [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]

    class R(CalibrationDataReader):
        def __init__(self):
            self.i = 0

        def get_next(self):
            if self.i >= len(data):
                return None
            self.i += 1
            return data[self.i - 1]

        def reset_iter(self):
            self.i = 0

    return R, data


def _bc_conv():
    rng = np.random.default_rng(8)
    return _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[1,3,8,8] x) => (float[1,4,8,8] y) {
            c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0)
            r0 = Relu(c0)
            c1 = Conv<group=1>(r0, w1, b1)
            r1 = Relu(c1)
            y = Conv<group=1>(r1, w2, b2)
        }""",
        dict(
            w0=_w(rng, 8, 3, 3, 3),
            b0=_w(rng, 8, scale=2),
            w1=_w(rng, 8, 8, 1, 1),
            b1=_w(rng, 8),
            w2=_w(rng, 4, 8, 1, 1),
            b2=_w(rng, 4, scale=0.1),
        ),
    ), (1, 3, 8, 8)


def _bc_mlp():
    rng = np.random.default_rng(9)
    return _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,8] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            y = Gemm(h1, w2, b2)
        }""",
        dict(w1=_w(rng, 16, 32), b1=_w(rng, 32), w2=_w(rng, 32, 8), b2=_w(rng, 8)),
    ), (3, 16)


def _int32_biases(model):
    inits = _inits(model)
    return {k: v for k, v in inits.items() if v.dtype == np.int32 and k.startswith("b")}


@pytest.mark.parametrize("build", [_bc_conv, _bc_mlp])
def test_bias_correction_writes_the_same_integer_biases_as_quark(build, tmp_path):
    """Both run on the *same* Quark-quantized model, so only the algorithm is
    compared: per-layer float - quantized mean, damping, re-quantization."""
    from onnxruntime.quantization import CalibrationMethod, QuantType

    model, shape = build()
    # S8S8_AAWS folds the Relu into the output quantizer (Quark's BC cannot
    # handle the A8W8 layout, where it fails on a missing value_info)
    quant = quark_quantize(model, "S8S8_AAWS", shape, tmp_path, "bc")
    reader, data = _reader(shape)
    theirs = quark_bias_correction(
        onnx.shape_inference.infer_shapes(copy.deepcopy(model)),
        copy.deepcopy(quant),
        False,
        reader(),
        QuantType.QInt8,
        CalibrationMethod.MinMax,
        {},
    )
    ours = correct_bias_quark(model, quant, data)
    before = _int32_biases(quant)
    t, o = _int32_biases(theirs), _int32_biases(ours)
    assert before and set(t) == set(o) == set(before)
    assert any(not np.array_equal(t[k], before[k]) for k in t), "Quark changed nothing"
    for k in t:
        np.testing.assert_array_equal(t[k], o[k], err_msg=k)


# -- Quarot R1 ------------------------------------------------------------------------


def _rotation_llm():
    rng = np.random.default_rng(10)
    d = 16
    return _with(
        f"""<ir_version: 9, opset_import: ["": 17]>
        g (int64[3] ids) => (float[3,{d}] y) {{
            e = Gather(embed_weight, ids)
            n = Mul(e, norm_weight)
            q = Gemm<alpha=1.0, beta=1.0, transB=1>(n, q_weight, q_bias)
            o = Gemm<alpha=1.0, beta=1.0, transB=1>(q, o_weight, o_bias)
            n2 = Mul(o, norm2_weight)
            y = MatMul(n2, head_weight)
        }}""",
        {
            "embed_weight": _w(rng, 32, d),
            "norm_weight": 1 + _w(rng, d, scale=0.3),
            "q_weight": _w(rng, d, d),
            "q_bias": _w(rng, d),
            "o_weight": _w(rng, d, d),
            "o_bias": _w(rng, d),
            "norm2_weight": 1 + _w(rng, d, scale=0.3),
            "head_weight": _w(rng, d, d),
        },
    )


_ROTATION_CONFIG = {
    "R1_pairs": [
        {"prev_nodes": ["n0_Gather"], "next_nodes": ["n2_Gemm"], "norm_node": "n1_Mul"},
        {"prev_nodes": ["n3_Gemm"], "next_nodes": ["n5_MatMul"], "norm_node": "n4_Mul"},
    ]
}


def test_quarot_r1_weights_are_bit_identical_to_quark(tmp_path):
    from quark.onnx.algorithm.quarot.quarot import rotation_transforms

    from onnxsim.quark_quarot import make_rotation, rotate_model

    model = _rotation_llm()
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_ROTATION_CONFIG))
    r1 = make_rotation(16, False)
    theirs = rotation_transforms(copy.deepcopy(model), {"R1": r1}, str(path))
    ours = rotate_model(model, _ROTATION_CONFIG, r1=r1)
    assert _changed(model, theirs)
    for k, v in _inits(ours).items():
        np.testing.assert_allclose(
            v.astype(np.float64), _inits(theirs)[k], rtol=0, atol=1e-7, err_msg=k
        )


@pytest.mark.parametrize("size", [2, 16, 64, 128])
def test_quarot_power_of_two_hadamard_equals_quarks(size):
    torch = pytest.importorskip("torch")
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.find_spec("quark")
    assert spec is not None and spec.submodule_search_locations
    path = (
        f"{list(spec.submodule_search_locations)[0]}"
        "/torch/algorithm/rotation/hadamard.py"
    )
    mod_spec = importlib.util.spec_from_file_location("quark_hadamard", path)
    assert mod_spec is not None and mod_spec.loader is not None
    had = importlib.util.module_from_spec(mod_spec)
    try:
        mod_spec.loader.exec_module(had)
    except Exception as e:  # pragma: no cover - needs quark's torch extras
        pytest.skip(f"cannot load quark's hadamard module: {e}")
    h1, _, _ = had.get_hadamard_matrices(size)
    theirs = (h1.to(torch.float64) / torch.tensor(float(size)).sqrt()).numpy()

    from onnxsim.quark_quarot import make_rotation

    np.testing.assert_allclose(make_rotation(size, False), theirs, atol=1e-12)


# -- AutoMixprecision --------------------------------------------------------------------

_D = 16


def _amp_model():
    rng = np.random.default_rng(0)
    return _with(
        f"""<ir_version: 9, opset_import: ["": 17]>
        g (float[3,{_D}] x) => (float[3,{_D}] y) {{
            h1 = Gemm(x, w1, b1)
            t1 = Tanh(h1)
            h2 = Gemm(t1, w2, b2)
            t2 = Tanh(h2)
            y = Gemm(t2, w3, b3)
        }}""",
        dict(
            w1=_w(rng, _D, _D, scale=1.5),
            b1=_w(rng, _D),
            w2=_w(rng, _D, _D, scale=0.3),
            b2=_w(rng, _D),
            w3=_w(rng, _D, _D),
            b3=_w(rng, _D),
        ),
    )


_AMP_TENSORS = ["x", "h1", "t1", "h2", "t2", "y"]


def _activation_dtypes(model):
    """Original tensor -> activation dtype, from each Q node's zero point."""
    inits = {t.name: t for t in model.graph.initializer}
    out = {}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and len(n.input) > 2 and n.input[2] in inits:
            t = n.input[0].removesuffix("/f").removesuffix("_QuantizeLinear_Input")
            dt = onnx.TensorProto.DataType.Name(inits[n.input[2]].data_type)
            if dt != "INT32":
                out[t] = dt
    return {t: out.get(t) for t in _AMP_TENSORS}


class _Amp:
    """The same AutoMixprecision request, run through Quark and onnxsim."""

    def __init__(self, tmp_path):
        self.model = _amp_model()
        self.dir = tmp_path
        onnx.save(self.model, str(tmp_path / "a.onnx"))
        self.data = [
            {"x": np.random.default_rng(i).standard_normal((3, _D)).astype(np.float32)}
            for i in range(6)
        ]

    def _reader(self):
        from onnxruntime.quantization import CalibrationDataReader

        it = iter(self.data)

        class R(CalibrationDataReader):
            def get_next(self):
                return next(it, None)

        return R()

    @staticmethod
    def _specs(kind):
        from quark.onnx.quantization.config import spec as s

        return {"u8": s.UInt8Spec, "u16": s.UInt16Spec, "s16": s.Int16Spec}[kind]

    def quark(self, base, target, **kw):
        from quark.onnx import AutoMixprecisionConfig, ModelQuantizer, QConfig
        from quark.onnx import QLayerConfig as QL
        from quark.onnx.quantization.config.spec import Int8Spec

        def conv(t):
            if isinstance(t, dict):
                return {conv(k): v for k, v in t.items()}
            if isinstance(t, list):
                return [conv(c) for c in t]
            return QL(activation=self._specs(t)(), weight=Int8Spec())

        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            cfg = QConfig(
                global_config=QL(activation=self._specs(base)(), weight=Int8Spec()),
                algo_config=[AutoMixprecisionConfig(conv(target), **kw)],
            )
            ModelQuantizer(cfg).quantize_model(
                str(self.dir / "a.onnx"), str(self.dir / "q.onnx"), self._reader()
            )
        return onnx.load(str(self.dir / "q.onnx"))

    def mine(self, base, target, **kw):
        specs = {"u8": qc.UInt8Spec, "u16": qc.UInt16Spec, "s16": qc.Int16Spec}

        def conv(t):
            if isinstance(t, dict):
                return {conv(k): v for k, v in t.items()}
            if isinstance(t, list):
                return [conv(c) for c in t]
            return qc.QLayerConfig(activation=specs[t](), weight=qc.Int8Spec())

        cfg = qc.QConfig(
            global_config=qc.QLayerConfig(
                activation=specs[base](), weight=qc.Int8Spec()
            ),
            algo_config=[
                qc.AutoMixprecisionConfig(target_layer_config=conv(target), **kw)
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return qc.ModelQuantizer(cfg).quantize_model(
                self.model, calibration_data_reader=self.data
            )

    def both(self, base, target, **kw):
        return (
            _activation_dtypes(self.quark(base, target, **kw)),
            _activation_dtypes(self.mine(base, target, **kw)),
        )


@pytest.fixture()
def amp(tmp_path):
    return _Amp(tmp_path)


_AMP_CASES = {
    "single": ("u8", "u16", {}),
    "include": ("u8", "u16", {"include_layers": ["n2_Gemm"]}),
    "exclude": ("u8", "u16", {"exclude_layers": ["n2_Gemm"]}),
    "list-with-noop": ("u8", ["u8", "u16"], {}),
    "list-reversed": ("u8", ["u16", "u8"], {}),
    # the baseline already misses the threshold: nothing to do
    "speed-threshold": ("u8", "u16", {"metric_threshold": 0.2}),
    "no-threshold-only-analysis": ("u8", "u16", {"metric_threshold": None}),
    "quality-threshold": (
        "u8",
        "u16",
        {"metric_threshold": 0.05, "metric_optimize_object": "quality"},
    ),
    "workers": ("u8", "u16", {"worker_num": 2}),
    "no-input-qdq-shared": ("u8", "u16", {"no_input_qdq_shared": True}),
    "shared-param-unshare": ("u8", "u16", {"shared_param_mode": "unshare"}),
}


@pytest.mark.parametrize("case", sorted(_AMP_CASES))
def test_auto_mixprecision_promotes_the_layers_quark_promotes(amp, case):
    base, target, kw = _AMP_CASES[case]
    theirs, ours = amp.both(base, target, **kw)
    assert ours == theirs


def test_auto_mixprecision_dict_forms_match_quark(amp):
    # {config: [names]}: named layers use their config, the rest the one
    # mapped to [] (or, with none, the first entry)
    for target in (
        {"s16": ["n2_Gemm"], "u16": []},
        {"s16": ["n2_Gemm"], "u16": ["n0_Gemm"]},
    ):
        theirs, ours = amp.both("u8", target)
        assert ours == theirs, target
    assert theirs["t1"] == theirs["h2"] == "INT16"


def test_auto_mixprecision_actually_mixes_in_these_cases(amp):
    """Guard against the table above passing vacuously."""
    _, ours = amp.both("u8", "u16", include_layers=["n2_Gemm"])
    assert {ours["t1"], ours["h2"]} == {"UINT16"} and ours["x"] == "UINT8"


def test_auto_mixprecision_subgraph_json_matches_quark(amp):
    path = amp.dir / "sg.json"
    path.write_text(
        json.dumps(
            {
                "quantized": False,
                "num_subgraphs": 2,
                "subgraphs": [
                    {
                        "name": "front",
                        "start_nodes": ["n0_Gemm"],
                        "end_nodes": ["n1_Tanh"],
                    },
                    {
                        "name": "back",
                        "start_nodes": ["n2_Gemm"],
                        "end_nodes": ["n3_Tanh"],
                    },
                ],
            }
        )
    )
    theirs_cache, ours_cache = amp.dir / "tc.json", amp.dir / "oc.json"
    theirs, ours = amp.both("u8", "u16", subgraph_json=str(path))
    assert ours == theirs
    amp.quark(
        "u8", "u16", subgraph_json=str(path), sensitivity_cache_file=str(theirs_cache)
    )
    amp.mine(
        "u8", "u16", subgraph_json=str(path), sensitivity_cache_file=str(ours_cache)
    )

    def groups(p):
        return sorted(
            (r["name"], tuple(r["candidate_nodes"]))
            for r in json.loads(p.read_text())["results"]
        )

    assert groups(ours_cache) == groups(theirs_cache)
    assert dict(groups(ours_cache))["__ungrouped__"] == ("n4_Gemm",)


def test_auto_mixprecision_sensitivity_cache_matches_quarks_schema_and_pins(amp):
    theirs_cache, ours_cache = amp.dir / "tc.json", amp.dir / "oc.json"
    amp.quark("u8", "u16", sensitivity_cache_file=str(theirs_cache))
    amp.mine("u8", "u16", sensitivity_cache_file=str(ours_cache))
    theirs, ours = (json.loads(p.read_text()) for p in (theirs_cache, ours_cache))
    assert set(theirs) == set(ours) == {"version", "cache_key", "results"}
    assert set(theirs["results"][0]) == set(ours["results"][0])
    assert [r["name"] for r in ours["results"]] == [
        r["name"] for r in theirs["results"]
    ] or {r["name"] for r in ours["results"]} == {r["name"] for r in theirs["results"]}
    # pin n0 at its original precision by disabling it in the cache
    for p, doc in ((theirs_cache, theirs), (ours_cache, ours)):
        for r in doc["results"]:
            r["enabled"] = r["name"] != "n0_Gemm"
        p.write_text(json.dumps(doc))
    pinned_theirs = _activation_dtypes(
        amp.quark("u8", "u16", sensitivity_cache_file=str(theirs_cache))
    )
    pinned_ours = _activation_dtypes(
        amp.mine("u8", "u16", sensitivity_cache_file=str(ours_cache))
    )
    assert pinned_ours == pinned_theirs
    assert pinned_ours["x"] == "UINT8" and pinned_ours["t1"] == "UINT16"
