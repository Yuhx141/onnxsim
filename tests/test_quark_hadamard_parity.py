"""Quarot R1 rotation matrices for non-power-of-two hidden sizes.

AMD Quark tabulates Hadamard matrices of order 12, 20, 28, 36, 40, 52, 60, 108,
140, 156 and 172 and builds the R1 matrix of a hidden size ``K * 2**j`` as
``kron(H_K, H_{2**j}) / sqrt(n)``; every other size raises. The tests marked
``quark`` compare onnxsim's matrices and rotated weights against the real Quark
(skipped when it is not installed); the rest need no Quark.
"""

import contextlib
import copy
import io
import json
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc
from onnxsim import quark_hadamard as qh
from onnxsim import quark_quarot as qr

warnings.filterwarnings("ignore")

with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
):
    try:
        import quark.onnx  # noqa: F401

        _HAVE_QUARK = True
    except Exception:  # pragma: no cover - environment dependent
        _HAVE_QUARK = False

quark = pytest.mark.skipif(
    not _HAVE_QUARK, reason="AMD Quark (amd-quark) is not installed"
)


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Quark writes scratch files into the current directory."""
    monkeypatch.chdir(tmp_path)


# Every size Quark supports below ~400 (and a few LLM-sized ones), with the
# tabulated order it must pick.
SUPPORTED = {
    1: 1,
    2: 1,
    16: 1,
    12: 12,
    24: 12,
    48: 12,
    96: 12,
    192: 12,
    768: 12,
    144: 36,
    20: 20,
    28: 28,
    56: 28,
    36: 36,
    72: 36,
    40: 40,
    80: 40,
    160: 40,
    52: 52,
    104: 52,
    60: 60,
    120: 60,
    108: 108,
    216: 108,
    140: 140,
    280: 140,
    156: 156,
    312: 156,
    172: 172,
    344: 172,
    5120: 40,
    11008: 172,
}
# No Hadamard matrix of these sizes in Quark's library.
UNSUPPORTED = [3, 6, 14, 18, 30, 44, 88, 100, 176, 1000, 4097]


# -- Quark-free ------------------------------------------------------------------


@pytest.mark.parametrize("k", qh.KNOWN_SIZES)
def test_tabulated_matrices_are_hadamard(k):
    h = qh.hadamard_factor(k)[0]
    assert h.shape == (k, k)
    assert set(np.unique(h)) == {-1.0, 1.0}
    np.testing.assert_array_equal(h @ h.T, k * np.eye(k))
    assert not h.flags.writeable  # shared and cached


def test_known_sizes_are_the_documented_library():
    assert qh.KNOWN_SIZES == (172, 156, 140, 108, 60, 52, 40, 36, 28, 20, 12)


@pytest.mark.parametrize("n, k", sorted(SUPPORTED.items()))
def test_size_selection_and_orthogonality(n, k):
    assert qh.hadamard_factor(n)[1] == k
    assert qh.supports(n)
    if n <= 400:
        r = qh.hadamard_rotation(n)
        assert r.shape == (n, n)
        np.testing.assert_allclose(r @ r.T, np.eye(n), atol=1e-12)
        np.testing.assert_array_equal(qr.make_rotation(n), r)


def test_power_of_two_is_sylvester():
    for n in (1, 2, 4, 64):
        h = qh.hadamard_matrix(n)
        assert np.array_equal(h, qh.sylvester_hadamard(n))
    np.testing.assert_array_equal(
        qh.sylvester_hadamard(4),
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
    )
    with pytest.raises(ValueError, match="power of two"):
        qh.sylvester_hadamard(12)


@pytest.mark.parametrize("n", UNSUPPORTED)
def test_unsupported_sizes_mirror_quarks_error(n):
    assert not qh.supports(n)
    msg = f"Could not find an Hadamard matrix for the size n={n}."
    with pytest.raises(ValueError, match=msg.replace(".", r"\.")):
        qr.make_rotation(n)
    with pytest.raises(ValueError, match="Could not find"):
        qr.make_rotation(n, True, seed=1)
    # opt-in fallback keeps the old behaviour: a Haar-random orthogonal matrix
    r = qr.make_rotation(n, orthogonal_fallback=True) if n <= 1000 else None
    if r is not None:
        np.testing.assert_allclose(r.T @ r, np.eye(n), atol=1e-10)


@pytest.mark.parametrize("n", [0, -12])
def test_non_positive_sizes_raise(n):
    with pytest.raises(ValueError):
        qr.make_rotation(n)


@pytest.mark.parametrize("n", [12, 20, 24, 28, 60, 172])
def test_random_hadamard_flips_rows_and_is_seeded(n):
    base = qh.hadamard_rotation(n)
    a = qr.make_rotation(n, True, seed=1)
    np.testing.assert_array_equal(a, qr.make_rotation(n, True, seed=1))
    assert not np.array_equal(a, qr.make_rotation(n, True, seed=2))
    # same matrix up to a +-1 per row
    ratio = a / base
    np.testing.assert_allclose(np.abs(ratio), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(ratio.mean(axis=1)), 1.0, atol=1e-12)
    np.testing.assert_allclose(a @ a.T, np.eye(n), atol=1e-6)  # float32 divisor
    signs = np.where(np.arange(n) % 3 == 0, -1.0, 1.0)
    given = qr.make_rotation(n, True, signs=signs)
    np.testing.assert_allclose(given, signs[:, None] * base, atol=1e-15)
    with pytest.raises(ValueError, match="signs"):
        qh.hadamard_rotation(n, np.full(n, 2.0))


# -- models --------------------------------------------------------------------


def _w(rng, *shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _with(text, inits):
    m = parser.parse_model(text)
    m.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in inits.items())
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return m


def _gemm_llm(d, seed=10):
    """embed -> norm -> Gemm(transB) -> Gemm(transB) -> norm -> MatMul head."""
    rng = np.random.default_rng(seed)
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


_GEMM_CONFIG = {
    "R1_pairs": [
        {"prev_nodes": ["n0_Gather"], "next_nodes": ["n2_Gemm"], "norm_node": "n1_Mul"},
        {"prev_nodes": ["n3_Gemm"], "next_nodes": ["n5_MatMul"], "norm_node": "n4_Mul"},
    ]
}


def _matmul_llm(d, ff=None, seed=11):
    """embed -> norm -> MatMul up -> Relu -> MatMul down -> norm -> MatMul head
    (no biases, MatMul weights stored [in, out], a wider hidden layer)."""
    ff = ff or 2 * d
    rng = np.random.default_rng(seed)
    return _with(
        f"""<ir_version: 9, opset_import: ["": 17]>
        g (int64[2] ids) => (float[2,{d}] y) {{
            e = Gather(embed_weight, ids)
            n = Mul(e, norm_weight)
            u = MatMul(n, up_weight)
            a = Relu(u)
            o = MatMul(a, down_weight)
            n2 = Mul(o, norm2_weight)
            y = MatMul(n2, head_weight)
        }}""",
        {
            "embed_weight": _w(rng, 16, d),
            "norm_weight": 1 + _w(rng, d, scale=0.3),
            "up_weight": _w(rng, d, ff),
            "down_weight": _w(rng, ff, d),
            "norm2_weight": 1 + _w(rng, d, scale=0.3),
            "head_weight": _w(rng, d, d),
        },
    )


_MATMUL_CONFIG = {
    "R1_pairs": [
        {
            "prev_nodes": ["n0_Gather"],
            "next_nodes": ["n2_MatMul"],
            "norm_node": "n1_Mul",
        },
        {
            "prev_nodes": ["n4_MatMul"],
            "next_nodes": ["n6_MatMul"],
            "norm_node": "n5_Mul",
        },
    ]
}

MODELS = {
    "gemm": (_gemm_llm, _GEMM_CONFIG, np.array([3, 0, 31], dtype=np.int64)),
    "matmul": (_matmul_llm, _MATMUL_CONFIG, np.array([5, 15], dtype=np.int64)),
}


def _inits(model):
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def _assert_same_function(a, b, ids):
    ref = _run(b, ids)
    # float32 outputs of magnitude up to ~1e2 for the widest tables
    np.testing.assert_allclose(
        _run(a, ids), ref, rtol=1e-4, atol=1e-5 * max(1.0, float(np.abs(ref).max()))
    )


def _run(model, ids):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"ids": ids})[0]


@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("d", [12, 20, 24, 28, 36, 40, 52, 60])
@pytest.mark.parametrize("random_had", [False, True])
def test_rotation_preserves_the_function(name, d, random_had):
    build, cfg, ids = MODELS[name]
    model = build(d)
    out = qr.rotate_model(model, cfg, r_matrix_dim=d, use_random_had=random_had, seed=5)
    onnx.checker.check_model(out)
    before, after = _inits(model), _inits(out)
    assert any(not np.array_equal(before[k], after[k]) for k in before)
    _assert_same_function(out, model, ids)


@pytest.mark.parametrize("d", [108, 140, 156, 172])
def test_rotation_preserves_the_function_large_tables(d):
    build, cfg, ids = MODELS["matmul"]
    model = build(d, ff=d)
    out = qr.rotate_model(model, cfg, r_matrix_dim=d)
    _assert_same_function(out, model, ids)


def test_unsupported_size_through_rotate_model_raises():
    build, cfg, _ = MODELS["matmul"]
    with pytest.raises(ValueError, match="Could not find an Hadamard"):
        qr.rotate_model(build(30), cfg, r_matrix_dim=30)


def _quarot_compat(tmp_path, model, cfg, **params):
    path = tmp_path / "rot.json"
    path.write_text(json.dumps(cfg))
    qcfg = qc.QConfig.get_default_config("U8S8_AAWS")
    algo = qc.QuarotConfig(r_config_path=str(path), **params)
    return qc.ModelQuantizer(qcfg)._quarot(model, algo)


@pytest.mark.parametrize("d", [12, 40, 60])
def test_compat_quarot_uses_the_tabulated_matrix(tmp_path, d):
    build, cfg, _ = MODELS["gemm"]
    model = build(d)
    out = _quarot_compat(tmp_path, model, cfg, r_matrix_dim=d)
    ref = qr.rotate_model(model, cfg, r1=qh.hadamard_rotation(d))
    for k, v in _inits(ref).items():
        np.testing.assert_array_equal(_inits(out)[k], v, err_msg=k)


def test_compat_quarot_unsupported_size_raises_like_quark(tmp_path):
    build, cfg, _ = MODELS["gemm"]
    with pytest.raises(
        AssertionError,
        match=r"The dim of the target R1 matrix is not support due to "
        r"Could not find an Hadamard matrix for the size n=30\.",
    ):
        _quarot_compat(tmp_path, build(30), cfg, r_matrix_dim=30)


# -- parity with the real Quark ----------------------------------------------------


@pytest.fixture(scope="module")
def quark_rotation():
    """Quark's ``get_rotation_matrix`` (it pulls in ``quark.torch``, whose first
    import can fail on optional extras such as ``transformers`` and succeeds on
    the retry once the package is half-initialized, as in Quark's own flow)."""
    pytest.importorskip("torch")
    last = None
    for _ in range(2):
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                from quark.torch.algorithm.rotation import hadamard as had
                from quark.torch.algorithm.rotation.rotation_utils import (
                    get_rotation_matrix,
                )
            return get_rotation_matrix, had
        except Exception as e:  # pragma: no cover - environment dependent
            last = e
    pytest.skip(f"cannot import quark's rotation utilities: {last}")


@quark
@pytest.mark.parametrize("n", sorted(SUPPORTED))
def test_rotation_matrix_is_bit_identical_to_quarks(quark_rotation, n):
    get_rotation_matrix, _ = quark_rotation
    theirs = get_rotation_matrix(num_channels=n, random=False, device="cpu").numpy()
    assert theirs.dtype == np.float64
    np.testing.assert_array_equal(qr.make_rotation(n), theirs)
    np.testing.assert_array_equal(qh.hadamard_rotation(n), theirs)


@quark
def test_tabulated_set_and_selection_equal_quarks(quark_rotation):
    _, had = quark_rotation
    assert tuple(had.KNOWN_HADAMARD_MATRICES) == qh.KNOWN_SIZES
    for k in qh.KNOWN_SIZES:
        theirs = had.KNOWN_HADAMARD_MATRICES[k]().numpy().astype(np.float64)
        np.testing.assert_array_equal(qh.hadamard_factor(k)[0], theirs)
    for n in SUPPORTED:
        _, k = had._get_hadamard_K(n)
        assert qh.hadamard_factor(n)[1] == k, n


@quark
@pytest.mark.parametrize("n", UNSUPPORTED)
def test_unsupported_sizes_are_unsupported_in_quark_too(quark_rotation, n):
    get_rotation_matrix, _ = quark_rotation
    with pytest.raises(ValueError) as theirs:
        get_rotation_matrix(num_channels=n, random=False, device="cpu")
    with pytest.raises(ValueError) as ours:
        qr.make_rotation(n)
    assert str(ours.value) == str(theirs.value)


@quark
@pytest.mark.parametrize("n", [2, 12, 20, 24, 28, 36, 40, 60, 108, 172, 344])
@pytest.mark.parametrize("seed", [0, 7])
def test_random_hadamard_equals_quarks_for_the_same_signs(quark_rotation, n, seed):
    torch = pytest.importorskip("torch")
    _, had = quark_rotation
    torch.manual_seed(seed)
    theirs = had.random_hadamard_matrix(n).numpy()
    torch.manual_seed(seed)  # the signs Quark drew: its first RNG call
    signs = (torch.randint(low=0, high=2, size=(n,)).to(torch.float64) * 2 - 1).numpy()
    np.testing.assert_array_equal(qr.make_rotation(n, True, signs=signs), theirs)


@quark
@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("d", [12, 20, 24, 28, 36, 40, 52, 60, 108, 172])
def test_rotated_weights_are_bit_identical_to_quarks(quark_rotation, tmp_path, name, d):
    from quark.onnx.algorithm.quarot.quarot import rotation_transforms

    get_rotation_matrix, _ = quark_rotation
    build, cfg, ids = MODELS[name]
    model = build(d)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(cfg))
    r1 = get_rotation_matrix(num_channels=d, random=False, device="cpu").numpy()
    theirs = rotation_transforms(copy.deepcopy(model), {"R1": r1}, str(path))
    # ours builds the matrix itself, from r_matrix_dim
    ours = qr.rotate_model(model, cfg, r_matrix_dim=d)
    a, b = _inits(ours), _inits(theirs)
    assert set(a) == set(b)
    assert any(not np.array_equal(_inits(model)[k], b[k]) for k in b)
    for k in a:
        assert a[k].dtype == b[k].dtype, k
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    _assert_same_function(ours, model, ids)


@quark
@pytest.mark.parametrize("d", [12, 40, 60])
def test_quarks_apply_quarot_and_onnxsims_compat_agree(quark_rotation, tmp_path, d):
    from quark.onnx.algorithm.interface import apply_QuaRot

    build, cfg, ids = MODELS["gemm"]
    model = build(d)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(cfg))
    with contextlib.redirect_stdout(io.StringIO()):
        theirs = apply_QuaRot(
            copy.deepcopy(model),
            extra_options={"RMatrixDim": d, "RConfigPath": str(path)},
        )
    ours = _quarot_compat(tmp_path, model, cfg, r_matrix_dim=d)
    for k, v in _inits(theirs).items():
        np.testing.assert_array_equal(_inits(ours)[k], v, err_msg=k)


@quark
def test_quarks_apply_quarot_rejects_the_same_sizes(quark_rotation, tmp_path):
    from quark.onnx.algorithm.interface import apply_QuaRot

    build, cfg, _ = MODELS["gemm"]
    model = build(30)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(AssertionError) as theirs:
        apply_QuaRot(
            copy.deepcopy(model),
            extra_options={"RMatrixDim": 30, "RConfigPath": str(path)},
        )
    with pytest.raises(AssertionError) as ours:
        _quarot_compat(tmp_path, model, cfg, r_matrix_dim=30)
    assert str(ours.value) == str(theirs.value)
