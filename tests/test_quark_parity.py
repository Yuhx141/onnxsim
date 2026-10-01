"""Parity of onnxsim's Quark-compat layer against the real AMD Quark ONNX
package. Skipped unless ``quark.onnx`` is importable (the
``quark-parity`` workflow installs it).

Quark is the ground truth: each test runs Quark and onnxsim on the same graph
and compares what they emit (op placement, attributes, axes) and, where the
ONNX Runtime custom-op library Quark JIT-builds is available, what the emitted
graphs compute. Differences that are known and deliberate are listed in the
``KNOWN_*`` constants so a *new* difference fails the test.

Run with ``-s`` (or see ``QUARK_PARITY_REPORT``) for the quality report.
"""

import contextlib
import io
import json
import os
import warnings

import numpy as np
import onnx
import pytest
from onnx import parser

warnings.filterwarnings("ignore")

with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
):
    try:
        import quark.onnx as quark_onnx
        from quark.onnx.quantization.config import DefaultConfigMapping
    except Exception as e:  # pragma: no cover - environment dependent
        quark_onnx = None
        _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    quark_onnx is None, reason="AMD Quark (amd-quark) is not installed"
)

from onnxsim import quark_auto_mixprecision as amp  # noqa: E402
from onnxsim import quark_compat as qc  # noqa: E402
from onnxsim import quark_tools  # noqa: E402
from onnxsim.quark_fakequant_graph import apply_fake_quant_format  # noqa: E402

# Quark presets onnxsim does not implement (NPU CNN/transformer quantizers,
# MatMulNBits, dynamic/VINT8, mixed block formats, ...).
KNOWN_MISSING = {
    "BF16_ADAQUANT",
    "BF16_BFP16",
    "BF16_MIXED_BFP16",
    "BF16_MIXED_BFP16_ADAQUANT",
    "BF16_MIXED_MXINT8",
    "BF16_MIXED_MXINT8_ADAQUANT",
    "BF16_MXINT8",
    "FP16_ADAQUANT",
    "INT16_CNN_ACCURATE",
    "INT16_CNN_DEFAULT",
    "INT16_TRANSFORMER_ACCURATE",
    "INT16_TRANSFORMER_DEFAULT",
    "INT8_CNN_ACCURATE",
    "INT8_CNN_DEFAULT",
    "INT8_TRANSFORMER_ACCURATE",
    "INT8_TRANSFORMER_DEFAULT",
    "MATMUL_NBITS",
    "MX9_INT8",
    "S16S16_MIXED_S8S8",
    "VINT8",
}
# onnxsim-only presets (Quark has no ADAROUND/ADAQUANT variant for U8U8_AAWA).
KNOWN_EXTRA = {"U8U8_AAWA_ADAQUANT", "U8U8_AAWA_ADAROUND"}
# Ops whose single-op graph Quark rewrites before quantizing (ReduceMean ->
# GlobalAveragePool); onnxsim quantizes the graph as given.
KNOWN_GRAPH_DIFF = {"ReduceMean"}

_BLOCK = ["BFP16", "MX4", "MX9", "MXINT8", "MXFP8E4M3", "MXFP4E2M1"]
_HALF = ["FP16", "BF16"]


# -- helpers ---------------------------------------------------------------------


def _reader(shape, n=4, seed=3, name="x"):
    from onnxruntime.quantization import CalibrationDataReader

    rng = np.random.default_rng(seed)
    data = [{name: rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter(data)

        def get_next(self):
            return next(self.it, None)

    return R


def quark_quantize(model, preset, shape, tmp_path, tag="m", cle=False):
    """Run Quark. Every Quark preset enables cross-layer equalization
    (``include_cle``) implicitly, which rewrites weights; onnxsim only runs CLE
    when the config lists a ``CLEConfig``. It is therefore off here unless a
    test asks for it."""
    from quark.onnx import ModelQuantizer, QConfig

    src, dst = str(tmp_path / f"{tag}.onnx"), str(tmp_path / f"{tag}_q.onnx")
    onnx.save(model, src)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        cfg = QConfig.get_default_config(preset)
        cfg.global_quant_config.include_cle = cle
        ModelQuantizer(cfg).quantize_model(src, dst, _reader(shape)())
    return onnx.load(dst)


def mine_quantize(model, preset, shape):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(qc.QConfig.get_default_config(preset)).quantize_model(
            model, calibration_data_reader=_reader(shape)()
        )


def _attrs(n):
    out = {}
    for a in n.attribute:
        v = onnx.helper.get_attribute_value(a)
        out[a.name] = v.decode() if isinstance(v, bytes) else v
    return out


def _cop_map(model):
    """tensor -> (op_type, attrs) for every com.amd.quark node."""
    out = {}
    for n in model.graph.node:
        if n.domain == "com.amd.quark":
            src = n.input[0].removesuffix("_QuantizeLinear_Input")
            out[src] = (n.op_type, tuple(sorted(_attrs(n).items())))
    return out


def _ext_map(model):
    """tensor -> op type, for every (Extended)QuantizeLinear."""
    return {
        n.input[0].removesuffix("_QuantizeLinear_Input"): n.op_type
        for n in model.graph.node
        if n.op_type in ("QuantizeLinear", "ExtendedQuantizeLinear")
    }


def _op_counts(model):
    counts = {}
    for n in model.graph.node:
        counts[n.op_type] = counts.get(n.op_type, 0) + 1
    return counts


def _ops_lib():
    path = os.environ.get("QUARK_ONNX_OPS_LIB")
    if path:
        return path
    try:
        from quark.onnx.operators.custom_ops import get_library_path

        return get_library_path()
    except Exception:  # pragma: no cover - environment dependent
        return None


def _run(model, x):
    import onnxruntime as ort

    so = ort.SessionOptions()
    lib = _ops_lib()
    if lib:
        so.register_custom_ops_library(lib)
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x})[0]


# -- models ---------------------------------------------------------------------


def _w(rng, *shape):
    return (rng.standard_normal(shape) * 0.5).astype(np.float32)


def _mlp():
    # Gemm, not MatMul+Add: Quark's preprocessing fuses the latter into a Gemm
    # (with renamed tensors) before quantizing, which onnxsim does not do.
    rng = np.random.default_rng(0)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,8] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            y = Gemm(h1, w2, b2)
        }
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, 16, 32), "w1"),
            onnx.numpy_helper.from_array(_w(rng, 32), "b1"),
            onnx.numpy_helper.from_array(_w(rng, 32, 8), "w2"),
            onnx.numpy_helper.from_array(_w(rng, 8), "b2"),
        ]
    )
    return m, (3, 16)


def _conv():
    rng = np.random.default_rng(1)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[1,3,8,8] x) => (float[1,8,4,4] y) {
            c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
            r0 = Relu(c0)
            y = MaxPool<kernel_shape=[2,2], strides=[2,2]>(r0)
        }
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, 8, 3, 3, 3), "w1"),
            onnx.numpy_helper.from_array(_w(rng, 8), "b1"),
        ]
    )
    return m, (1, 3, 8, 8)


def _gemm_transb():
    rng = np.random.default_rng(2)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,8] y) {
            y = Gemm<transB=1>(x, w, b)
        }
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, 8, 16), "w"),
            onnx.numpy_helper.from_array(_w(rng, 8), "b"),
        ]
    )
    return m, (3, 16)


MODELS = {"mlp": _mlp, "conv": _conv, "gemm_transb": _gemm_transb}


# -- presets ----------------------------------------------------------------------


def test_preset_coverage_matches_documented_gaps():
    quark_names, mine = set(DefaultConfigMapping), set(qc._PRESETS)
    assert quark_names - mine == KNOWN_MISSING, "Quark gained/lost presets"
    assert mine - quark_names == KNOWN_EXTRA


_DTYPE = {
    "QInt8": "int8",
    "QUInt8": "uint8",
    "QInt16": "int16",
    "QUInt16": "uint16",
    "QFloat16": "float16",
    "QBFloat16": "bfloat16",
}


def _shared_presets():
    return sorted(set(DefaultConfigMapping) & set(qc._PRESETS)) if quark_onnx else []


@pytest.mark.parametrize("preset", _shared_presets())
def test_preset_dtypes_match(preset):
    from quark.onnx import QConfig

    with contextlib.redirect_stdout(io.StringIO()):
        q = QConfig.get_default_config(preset).global_quant_config
    mine = qc._PRESETS[preset]().global_config
    for quark_t, spec in (
        (q.activation_type, mine.activation),
        (q.weight_type, mine.weight),
    ):
        if quark_t.name in _DTYPE:
            assert spec.dtype == _DTYPE[quark_t.name]
        else:  # QBFP / QMX: a block format on both sides
            assert spec.dtype.startswith(("bfp", "mx"))


# -- block formats and half precision: graph parity --------------------------------


@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize("preset", _BLOCK)
def test_block_format_graph_matches_quark(preset, model_name, tmp_path):
    model, shape = MODELS[model_name]()
    q = quark_quantize(model, preset, shape, tmp_path)
    m = mine_quantize(model, preset, shape)
    assert _cop_map(m) == _cop_map(q)
    assert _op_counts(m) == _op_counts(q)


@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize("preset", _HALF)
def test_half_precision_graph_matches_quark(preset, model_name, tmp_path):
    model, shape = MODELS[model_name]()
    q = quark_quantize(model, preset, shape, tmp_path)
    m = mine_quantize(model, preset, shape)
    assert _ext_map(m) == _ext_map(q)
    assert _op_counts(m) == _op_counts(q)


def _single_op_cases():
    X = [1, 4, 6, 6]
    rng = np.random.default_rng(0)
    f = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    return {
        **{
            op: dict(shapes=[X])
            for op in (
                "Relu",
                "Sigmoid",
                "Tanh",
                "LeakyRelu",
                "Erf",
                "Abs",
                "Neg",
                "Exp",
                "Sqrt",
                "Flatten",
                "GlobalAveragePool",
            )
        },
        "Softmax": dict(shapes=[X], attrs=dict(axis=1)),
        "Add": dict(shapes=[X, X]),
        "Mul": dict(shapes=[X, X]),
        "AddConst": dict(op="Add", shapes=[X], consts={"c": f(1, 4, 1, 1)}),
        "Concat": dict(shapes=[X, X], attrs=dict(axis=1)),
        "MaxPool": dict(shapes=[X], attrs=dict(kernel_shape=[2, 2])),
        "AveragePool": dict(shapes=[X], attrs=dict(kernel_shape=[2, 2])),
        "ReduceMean": dict(shapes=[X], attrs=dict(axes=[2, 3])),
        "Reshape": dict(shapes=[X], consts={"s": np.array([1, 144], np.int64)}),
        "Transpose": dict(shapes=[X], attrs=dict(perm=[0, 2, 3, 1])),
        "Resize": dict(
            shapes=[X],
            consts={
                "roi": np.array([], np.float32),
                "sc": np.array([1, 1, 2, 2], np.float32),
            },
        ),
        "BatchNormalization": dict(
            shapes=[X],
            consts={"s": f(4), "b": f(4), "m": f(4), "v": np.abs(f(4)) + 1},
        ),
        "InstanceNormalization": dict(shapes=[X], consts={"s": f(4), "b": f(4)}),
        "ConvTranspose": dict(
            shapes=[X], consts={"w": f(4, 4, 2, 2)}, attrs=dict(strides=[2, 2])
        ),
        "Gemm": dict(shapes=[[3, 8]], consts={"w": f(8, 5)}),
    }


def _single_op_model(case):
    from onnx import TensorProto as T
    from onnx import helper as H

    op = case.get("op")
    shapes = case["shapes"]
    names = ["x"] + [f"in{i}" for i in range(1, len(shapes))]
    inputs = [H.make_tensor_value_info(n, T.FLOAT, s) for n, s in zip(names, shapes)]
    inits = []
    for nm, arr in (case.get("consts") or {}).items():
        inits.append(onnx.numpy_helper.from_array(arr, nm))
        names.append(nm)
    node = H.make_node(op, names, ["y"], **(case.get("attrs") or {}))
    g = H.make_graph(
        [node],
        "g",
        inputs,
        [H.make_tensor_value_info("y", T.FLOAT, None)],
        initializer=inits,
    )
    return H.make_model(g, opset_imports=[H.make_opsetid("", 17)], ir_version=9)


def _single_op_params():
    return sorted(_single_op_cases()) if quark_onnx else []


@pytest.mark.parametrize(
    "dtype, preset", [("bfp16", "BFP16"), ("float16", "FP16"), ("bfloat16", "BF16")]
)
def test_single_op_placement_matches_quark(dtype, preset, tmp_path):
    """Quark's op coverage, op by op (``KNOWN_GRAPH_DIFF`` aside)."""
    diffs = []
    for name, case in _single_op_cases().items():
        case = dict(case, op=case.get("op", name))
        model = _single_op_model(case)
        try:
            q = quark_quantize(model, preset, case["shapes"][0], tmp_path, name)
        except Exception:  # Quark itself rejects this graph
            continue
        m = apply_fake_quant_format(model, dtype)
        got, want = _op_counts(m), _op_counts(q)
        if got != want and name not in KNOWN_GRAPH_DIFF:
            diffs.append((name, got, want))
    assert not diffs, json.dumps(diffs, indent=1)


# -- numerics --------------------------------------------------------------------


@pytest.mark.skipif(_ops_lib() is None, reason="Quark's custom-op library is not built")
@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize("preset", _BLOCK + _HALF)
def test_outputs_match_quark(preset, model_name, tmp_path):
    model, shape = MODELS[model_name]()
    q = quark_quantize(model, preset, shape, tmp_path)
    m = mine_quantize(model, preset, shape)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32) * 2
    np.testing.assert_allclose(_run(m, x), _run(q, x), rtol=1e-5, atol=1e-5)


# -- tools ------------------------------------------------------------------------


def test_remove_qdq_matches_quarks_convert_quant_to_float(tmp_path):
    from quark.onnx.tools.convert_quant_to_float import (
        convert_quant_to_float as quark_to_float,
    )

    model, shape = _mlp()
    q = quark_quantize(model, "A8W8", shape, tmp_path)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        theirs = quark_to_float(q)
    # Quark converts the int32 bias initializers to float64 (which ORT rejects
    # next to float32 Gemm operands); cast them back before comparing.
    for t in theirs.graph.initializer:
        if t.data_type == onnx.TensorProto.DOUBLE:
            t.CopyFrom(
                onnx.numpy_helper.from_array(
                    onnx.numpy_helper.to_array(t).astype(np.float32), t.name
                )
            )
    ours = quark_tools.convert_quant_to_float(q)
    x = np.random.default_rng(5).standard_normal(shape).astype(np.float32)
    np.testing.assert_allclose(_run(ours, x), _run(theirs, x), rtol=1e-5, atol=1e-5)
    assert not any(
        n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in ours.graph.node
    )


# -- auto mixed precision metrics -------------------------------------------------


def test_amp_metrics_match_quark():
    from quark.onnx.algorithm.mprecision import metric_funcs as qm

    rng = np.random.default_rng(0)
    f = [[rng.standard_normal((4, 6)).astype(np.float32)] for _ in range(5)]
    q = [[a[0] + 0.1 * rng.standard_normal((4, 6)).astype(np.float32)] for a in f]
    for name, theirs in (
        ("l2", qm.l2_metric),
        ("cosine", qm.cosine_metric),
        ("sqnr", qm.sqnr_metric),
        ("psnr", qm.psnr_metric),
        ("kl", qm.kl_divergence_metric),
    ):
        np.testing.assert_allclose(
            amp.resolve_metric(name)(f, q), theirs(f, q), rtol=1e-5, err_msg=name
        )


# -- cross-layer equalization ------------------------------------------------------


def _mlp3():
    rng = np.random.default_rng(6)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,8] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            h2 = Gemm(h1, w2, b2)
            h3 = LeakyRelu<alpha=0.1>(h2)
            y = Gemm(h3, w3, b3)
        }
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, 16, 32) * 3, "w1"),
            onnx.numpy_helper.from_array(_w(rng, 32), "b1"),
            onnx.numpy_helper.from_array(_w(rng, 32, 24), "w2"),
            onnx.numpy_helper.from_array(_w(rng, 24), "b2"),
            onnx.numpy_helper.from_array(_w(rng, 24, 8) * 0.2, "w3"),
            onnx.numpy_helper.from_array(_w(rng, 8), "b3"),
        ]
    )
    return m, (3, 16)


@pytest.mark.parametrize("build", [_mlp, _mlp3])
def test_cle_matches_quarks_equalization(build, tmp_path):
    """FP16 leaves the weights as initializers, so Quark's implicit CLE shows
    up as changed ``w*`` / ``b*``."""
    from onnxsim.quark_cle import equalize_linear_layers

    model, shape = build()
    q = quark_quantize(model, "FP16", shape, tmp_path, "cle", cle=True)
    ours = equalize_linear_layers(model)
    theirs = {i.name: onnx.numpy_helper.to_array(i) for i in q.graph.initializer}
    mine = {i.name: onnx.numpy_helper.to_array(i) for i in ours.graph.initializer}
    orig = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    assert not np.allclose(theirs["w1"], orig["w1"]), "Quark did not equalize"
    for name in orig:
        np.testing.assert_allclose(
            mine[name], theirs[name], rtol=1e-3, atol=1e-5, err_msg=name
        )
    x = np.random.default_rng(2).standard_normal(shape).astype(np.float32)
    np.testing.assert_allclose(_run(ours, x), _run(model, x), rtol=1e-4, atol=1e-4)


# -- quarot ------------------------------------------------------------------------


def _torch_style_llm():
    rng = np.random.default_rng(4)
    d = 16
    m = parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,{d}] x) => (float[3,{d}] y) {{
            h = Gemm<alpha=1.0, beta=1.0, transB=1>(x, w_in, b_in)
            y = Gemm<alpha=1.0, beta=1.0, transB=1>(h, w_out, b_out)
        }}
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, d, d), "w_in"),
            onnx.numpy_helper.from_array(_w(rng, d), "b_in"),
            onnx.numpy_helper.from_array(_w(rng, d, d), "w_out"),
            onnx.numpy_helper.from_array(_w(rng, d), "b_out"),
        ]
    )
    return m, (3, d)


def test_quarot_preserves_function_like_quark():
    from onnxsim.quark_quarot import rotate_model

    model, shape = _torch_style_llm()
    for n, name in zip(model.graph.node, ("g_in", "g_out")):
        n.name = name
    cfg = {"R1_pairs": [{"prev_nodes": ["g_in"], "next_nodes": ["g_out"]}]}
    out = rotate_model(model, cfg, r_matrix_dim=16)
    x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    # the rotation changes the hidden basis only; the first Gemm's output is
    # rotated, the final output is unchanged only when the last layer is
    # un-rotated by its reader -- compare the whole function.
    np.testing.assert_allclose(_run(out, x), _run(model, x), rtol=1e-4, atol=1e-4)


# -- quality report (non-asserting) ------------------------------------------------


def test_quality_report_against_quark(tmp_path):
    """Records, per preset, each side's output error versus float. No
    assertion on who is better -- onnxsim's calibration differs from Quark's
    (MinMSE power-of-two scales) -- only that both produce a usable model."""
    rows = []
    for mname, build in MODELS.items():
        model, shape = build()
        x = np.random.default_rng(9).standard_normal(shape).astype(np.float32)
        ref = _run(model, x)
        for preset in (
            "A8W8",
            "XINT8",
            "U8S8_AAWS",
            "A16W8",
            "FP16",
            "BF16",
            "BFP16",
            "MX9",
        ):
            try:
                q = _run(
                    quark_quantize(model, preset, shape, tmp_path, f"{mname}_{preset}"),
                    x,
                )
                m = _run(mine_quantize(model, preset, shape), x)
            except Exception as e:  # pragma: no cover - op library missing etc.
                rows.append(dict(model=mname, preset=preset, error=str(e)[:80]))
                continue

            def rel(a):
                return float(np.linalg.norm(a - ref) / (np.linalg.norm(ref) + 1e-12))

            rows.append(
                dict(
                    model=mname,
                    preset=preset,
                    quark_rel_err=rel(q),
                    onnxsim_rel_err=rel(m),
                )
            )
            assert np.isfinite(rel(m)) and rel(m) < 1.0
    path = os.environ.get("QUARK_PARITY_REPORT")
    text = "\n".join(
        f"{r['model']:12s} {r['preset']:10s} "
        + (
            r["error"]
            if "error" in r
            else f"quark {r['quark_rel_err']:.4f}  onnxsim {r['onnxsim_rel_err']:.4f}"
        )
        for r in rows
    )
    print("\n" + text)
    if path:
        with open(path, "w") as fh:
            fh.write(
                "| model | preset | Quark rel. err | onnxsim rel. err |\n|---|---|---|---|\n"
            )
            for r in rows:
                if "error" not in r:
                    fh.write(
                        f"| {r['model']} | {r['preset']} | {r['quark_rel_err']:.4f} | {r['onnxsim_rel_err']:.4f} |\n"
                    )


# -- integer presets: quantization parameters ------------------------------------


def _int_params(model):
    """Sorted ``(scale, zero_point, dtype)`` of every activation QuantizeLinear
    and the scales of the weight DequantizeLinear nodes (per-tensor)."""
    inits = {i.name: i for i in model.graph.initializer}

    def arr(name):
        return onnx.numpy_helper.to_array(inits[name])

    acts, weights = [], []
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[1] in inits:
            zp = arr(n.input[2])
            acts.append((float(arr(n.input[1])), int(zp), str(zp.dtype)))
        elif (
            n.op_type == "DequantizeLinear"
            and n.input[0] in inits
            and arr(n.input[0]).dtype == np.int8
            and arr(n.input[0]).ndim >= 2
        ):
            weights.append(float(np.max(arr(n.input[1]))))
    return sorted(acts), sorted(weights)


@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize(
    "preset",
    ["A8W8", "S8S8_AAWS", "U8S8_AAWS", "A16W8", "S16S8_ASWS", "U16S8_AAWS", "XINT8"],
)
def test_integer_preset_quantization_parameters_match_quark(
    preset, model_name, tmp_path
):
    """Same scale / zero point / dtype on every activation Q node and the same
    per-tensor weight scales. (``U8S8_AAWS``-style presets calibrate with
    percentiles; ours uses the same percentile, so the parameters agree up to
    histogram binning.)"""
    model, shape = MODELS[model_name]()
    q_acts, q_w = _int_params(quark_quantize(model, preset, shape, tmp_path))
    m_acts, m_w = _int_params(mine_quantize(model, preset, shape))
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    # histogram binning moves a percentile's zero point by a few codes of 16
    np.testing.assert_allclose(
        [a[1] for a in m_acts],
        [a[1] for a in q_acts],
        atol=40 if "16" in preset else 1,
    )
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=2e-3
    )
    np.testing.assert_allclose(m_w, q_w, rtol=2e-3)


# -- per-layer / per-type overrides ------------------------------------------------


def _named_mlp():
    model, shape = _mlp()
    for n, name in zip(model.graph.node, ("g1", "relu", "g2")):
        n.name = name
    return model, shape


def _layer_config(api, case):
    """``QConfig`` for the override ``case``, built from ``api`` (either
    ``quark.onnx`` or ``onnxsim.quark_compat`` -- the class names match)."""

    def glob():
        return api.QLayerConfig(activation=api.Int8Spec(), weight=api.Int8Spec())

    def int16():
        return api.QLayerConfig(
            input_tensors=api.Int16Spec(),
            weight=api.Int8Spec(),
            output_tensors=api.Int16Spec(),
        )

    kwargs = {
        "specific_name": dict(specific_layer_config={int16(): ["g2"]}),
        "specific_regex": dict(specific_layer_config={int16(): ["^g.*"]}),
        "type_gemm": dict(layer_type_config={int16(): ["Gemm"]}),
        "exclude_node": dict(exclude=["g1"]),
        "specific_over_type": dict(
            layer_type_config={int16(): ["Gemm"]},
            specific_layer_config={
                api.QLayerConfig(
                    input_tensors=api.Int8Spec(),
                    weight=api.Int8Spec(),
                    output_tensors=api.Int8Spec(),
                ): ["g1"]
            },
        ),
    }[case]
    return api.QConfig(global_config=glob(), **kwargs)


_LAYER_CASES = [
    "specific_name",
    "specific_regex",
    "type_gemm",
    "exclude_node",
    "specific_over_type",
]


@pytest.mark.parametrize("case", _LAYER_CASES)
def test_layer_overrides_match_quark(case, tmp_path):
    from quark.onnx import ModelQuantizer

    model, shape = _named_mlp()
    src, dst = str(tmp_path / "m.onnx"), str(tmp_path / "m_q.onnx")
    onnx.save(model, src)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        ModelQuantizer(_layer_config(quark_onnx, case)).quantize_model(
            src, dst, _reader(shape)()
        )
    q_acts, q_w = _int_params(onnx.load(dst))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mine = qc.ModelQuantizer(_layer_config(qc, case)).quantize_model(
            model, calibration_data_reader=_reader(shape)()
        )
    m_acts, m_w = _int_params(mine)
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=2e-3
    )
    np.testing.assert_allclose(m_w, q_w, rtol=2e-3)


# -- dynamic quantization ------------------------------------------------------------


@pytest.mark.parametrize("build", [_mlp, _conv, _gemm_transb])
def test_dynamic_quantization_matches_quark(build, tmp_path):
    model, shape = build()
    q = quark_quantize(model, "UINT8_DYNAMIC_QUANT", shape, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mine = qc.ModelQuantizer(
            qc.QConfig.get_default_config("UINT8_DYNAMIC_QUANT")
        ).quantize_model(model)
    assert [n.op_type for n in mine.graph.node] == [n.op_type for n in q.graph.node]
    x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    np.testing.assert_allclose(_run(mine, x), _run(q, x), rtol=1e-4, atol=1e-5)


# =============================================================================
# quark_tools_extra: the rest of quark.onnx.tools and the model_utils helpers.
# Each test runs Quark's own tool and onnxsim's on the same graph. Deliberate
# differences are documented next to the test that sees them.
# =============================================================================


def _quiet(fn, *args, **kwargs):
    # (logging is disabled too: some Quark helpers call logger.info with
    # extra positional args, which blows up under pytest's log capture)
    import logging

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        logging.disable(logging.CRITICAL)
        try:
            return fn(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)


def _copy_model(model):
    out = onnx.ModelProto()
    out.CopyFrom(model)
    return out


def _tx_ort(model, feed):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 4
    lib = _ops_lib()
    if lib:
        so.register_custom_ops_library(lib)
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feed)


def _tx_inits(model):
    from onnx import numpy_helper

    return {
        t.name: (t.data_type, numpy_helper.to_array(t)) for t in model.graph.initializer
    }


def _tx_nodes(model):
    return [(n.op_type, n.domain) for n in model.graph.node]


def _tx_sorted_nodes(model):
    return sorted(_tx_nodes(model))


def _tx_same_inits(a_model, b_model):
    a, b = _tx_inits(a_model), _tx_inits(b_model)
    assert set(a) == set(b)
    for k in a:
        assert a[k][0] == b[k][0], k
        np.testing.assert_array_equal(a[k][1], b[k][1], err_msg=k)


def _tx_conv_qdq(bias_dtype="int8", bias_scale=0.0007, act_zp_type="int8"):
    """x -> Q/DQ -> Conv(w DQ, bias DQ) -> Q/DQ -> y. Scalars come from the
    parser (``float_data``); the integer weights from numpy (``raw_data``),
    the form Quark's A8W8 converter reads. The bias / weight DQs are listed
    before the activation Q/DQ because Quark's converter assumes that graph
    order (it pairs Conv inputs with producers by node position)."""
    from onnx import numpy_helper

    rng = np.random.default_rng(0)
    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x) => (float[1,4,6,6] y)
        <float xs = {{0.02}}, {act_zp_type} xz = {{0}}, float ws = {{0.01}},
         int8 wz = {{0}}, float bs = {{{bias_scale}}}, {bias_dtype} bz = {{0}},
         float ys = {{0.05}}, {act_zp_type} yz = {{0}}>
        {{
            bd = DequantizeLinear(bq, bs, bz)
            wd = DequantizeLinear(wq, ws, wz)
            xq = QuantizeLinear(x, xs, xz)
            xd = DequantizeLinear(xq, xs, xz)
            c = Conv(xd, wd, bd)
            cq = QuantizeLinear(c, ys, yz)
            y = DequantizeLinear(cq, ys, yz)
        }}
        """
    )
    bq = rng.integers(-100, 100, (4,)).astype(
        np.int8 if bias_dtype == "int8" else np.int32
    )
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                rng.integers(-100, 100, (4, 3, 3, 3)).astype(np.int8), "wq"
            ),
            numpy_helper.from_array(bq, "bq"),
        ]
    )
    return model


def _tx_x(shape=(1, 3, 8, 8), seed=1):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_tools_a8w8_npu_to_cpu_matches_quark():
    from quark.onnx.tools.convert_a8w8_npu_to_a8w8_cpu import (
        convert_a8w8_npu_to_a8w8_cpu as q_fn,
    )

    model = _tx_conv_qdq()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_a8w8_npu_to_a8w8_cpu(model)
    _tx_same_inits(ours, theirs)
    assert _tx_inits(ours)["bq"][1].dtype == np.int32
    x = _tx_x()
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(theirs, {"x": x})[0]
    )


def test_tools_bias_int32_to_int16_matches_quark():
    from quark.onnx.tools.convert_bias_int32_to_int16 import (
        convert_bias_int32_to_int16 as q_fn,
    )

    model = _tx_conv_qdq(bias_dtype="int32")
    theirs, t_flag = _quiet(q_fn, _copy_model(model))
    ours, o_flag = quark_tools.convert_bias_int32_to_int16(model)
    assert o_flag is True and bool(t_flag) is True
    _tx_same_inits(ours, theirs)
    assert _tx_inits(ours)["bq"][1].dtype == np.int16
    assert _tx_inits(ours)["bz"][1].dtype == np.int16
    # nothing to convert -> flag False on both
    plain = _tx_conv_qdq()
    assert not _quiet(q_fn, _copy_model(plain))[1]
    assert quark_tools.convert_bias_int32_to_int16(plain)[1] is False


def test_tools_customqdq_to_qdq_matches_quark():
    from quark.onnx.tools.convert_customqdq_to_qdq import (
        convert_customqdq_to_qdq as q_fn,
    )

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21, "com.amd.quark": 1]>
        g (float[4] x) => (float[4] y, float[4] z)
        <float s = {0.1}, uint16 z16 = {32768}, int8 z8 = {0}, bfloat16 zb = {0}>
        {
            a = com.amd.quark.ExtendedQuantizeLinear(x, s, z16)
            y = com.amd.quark.ExtendedDequantizeLinear(a, s, z16)
            b = com.amd.quark.ExtendedQuantizeLinear(x, s, z8)
            c = com.amd.quark.ExtendedDequantizeLinear(b, s, z8)
            d = com.amd.quark.ExtendedQuantizeLinear(c, s, zb)
            z = com.amd.quark.ExtendedDequantizeLinear(d, s, zb)
        }
        """
    )
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_customqdq_to_qdq(model)
    assert _tx_nodes(ours) == _tx_nodes(theirs)
    assert [n.op_type for n in ours.graph.node] == [
        "QuantizeLinear",
        "DequantizeLinear",
        "QuantizeLinear",
        "DequantizeLinear",
        "ExtendedQuantizeLinear",
        "ExtendedDequantizeLinear",
    ]
    # deliberate: we also register the com.microsoft opset so the model loads
    assert "com.microsoft" in {o.domain for o in ours.opset_import}


@pytest.mark.parametrize("reverse", [False, True])
def test_tools_convert_custom_ops_matches_quark(reverse):
    from quark.onnx.tools import convert_custom_ops as q_mod

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21, "com.amd.quark": 1]>
        g (float[4] x) => (float[4] y)
        <float s = {0.1}, int8 z = {0}>
        {
            a = com.amd.quark.ExtendedQuantizeLinear(x, s, z)
            b = com.amd.quark.ExtendedDequantizeLinear(a, s, z)
            y = Relu(b)
        }
        """
    )
    if reverse:
        model = _quiet(
            q_mod.convert_custom_ops,
            _copy_model(model),
            q_mod.OLD_DOMAIN,
            q_mod.NAME_MAPPING,
        )
        domain = q_mod.NEW_DOMAIN
        mapping = {v: k for k, v in q_mod.NAME_MAPPING.items()}
        ours_map = {v: k for k, v in quark_tools.CUSTOM_OP_NAME_MAPPING.items()}
    else:
        domain, mapping = q_mod.OLD_DOMAIN, q_mod.NAME_MAPPING
        ours_map = quark_tools.CUSTOM_OP_NAME_MAPPING
    assert ours_map == mapping
    theirs = _quiet(q_mod.convert_custom_ops, _copy_model(model), domain, mapping)
    ours = quark_tools.convert_custom_ops(model, domain, ours_map)
    assert _tx_nodes(ours) == _tx_nodes(theirs)
    assert {(o.domain, o.version) for o in ours.opset_import} == {
        (o.domain, o.version) for o in theirs.opset_import
    }


def test_tools_fp16_to_bf16_matches_quarks_bf16_format():
    from quark.onnx.quantization.quant_utils import convert_to_bf16

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float16[2,4] x) => (float16[2,4] y)
        {
            t = Cast<to = 1>(x)
            u = Add(x, w)
            y = Relu(u)
        }
        """
    )
    # float16 / bfloat16 literals are not parseable: attach programmatically
    model.graph.initializer.append(
        onnx.helper.make_tensor(
            "w", onnx.TensorProto.FLOAT16, [4], [0.1, -2.5, 3.14159, 1000.0]
        )
    )
    theirs = _quiet(convert_to_bf16, _copy_model(model), onnx.TensorProto.BFLOAT16, 10)
    ours = quark_tools.convert_fp16_to_bf16(model)

    # deliberate: Quark appends the boundary casts at the *end* of the node
    # list (not topologically sorted) and adds one input Cast per consuming
    # node, so an input read twice gets two identical Casts writing the same
    # `<in>_cast` (an invalid graph). We put one Cast first. Same nodes and
    # wiring once the duplicates are folded.
    def sig(m):
        return sorted(
            {
                (
                    n.op_type,
                    tuple(n.input),
                    tuple(n.output),
                    tuple((t.name, t.i) for t in n.attribute),
                )
                for n in m.graph.node
            }
        )

    assert sig(ours) == sig(theirs)
    onnx.checker.check_model(ours)
    wa = {t.name: t for t in ours.graph.initializer}
    wb = {t.name: t for t in theirs.graph.initializer}
    assert set(wa) == set(wb)
    for k in wa:
        assert wa[k].data_type == wb[k].data_type == onnx.TensorProto.BFLOAT16
        assert wa[k].raw_data == wb[k].raw_data
    assert [o.type.tensor_type.elem_type for o in ours.graph.output] == [
        o.type.tensor_type.elem_type for o in theirs.graph.output
    ]


def test_tools_nchw_to_nhwc_matches_quark():
    from quark.onnx.utils.model_utils import convert_nchw_to_nhwc as q_fn

    plain = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x) => (float[1,3,8,8] y)
        { y = Relu(x) }
        """
    )
    quant = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x) => (float[1,3,8,8] y)
        <float s = {0.1}, int8 z = {0}>
        {
            r = Relu(x)
            q = QuantizeLinear(r, s, z)
            y = DequantizeLinear(q, s, z)
        }
        """
    )
    flat = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,5] x) => (float[1,5] y)
        { y = Relu(x) }
        """
    )

    def sig(m):
        return (
            sorted(
                (n.op_type, n.name, list(n.input), list(n.output)) for n in m.graph.node
            ),
            [o.name for o in m.graph.output],
            [
                [d.dim_value for d in v.type.tensor_type.shape.dim]
                for v in list(m.graph.input) + list(m.graph.output)
            ],
        )

    for model in (plain, quant, flat):
        theirs = _quiet(q_fn, _copy_model(model))
        ours = quark_tools.convert_nchw_to_nhwc(model)
        assert sig(ours) == sig(theirs)
        if model is not flat:
            x = _tx_x((1, 8, 8, 3))
            out_o = _tx_ort(ours, {"x": x})[0]
            np.testing.assert_array_equal(out_o, _tx_ort(theirs, {"x": x})[0])
            assert out_o.shape == (1, 8, 8, 3)


def test_tools_qdq_to_qop_matches_quark():
    from quark.onnx.tools.convert_qdq_to_qop import convert_qdq_to_qop as q_fn

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 13]>
        g (float[2,4] x, float[2,4] u) => (float[2,4] y)
        <float s = {0.1}, uint8 z = {128}, float sw = {0.05}, uint8 zw = {120},
         uint8[4,4] wq = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}>
        {
            xq = QuantizeLinear(x, s, z)
            xd = DequantizeLinear(xq, s, z)
            uq = QuantizeLinear(u, s, z)
            ud = DequantizeLinear(uq, s, z)
            wd = DequantizeLinear(wq, sw, zw)
            m = MatMul(xd, wd)
            mq = QuantizeLinear(m, s, z)
            md = DequantizeLinear(mq, s, z)
            a = Add(md, ud)
            aq = QuantizeLinear(a, s, z)
            ad = DequantizeLinear(aq, s, z)
            p = Mul(ad, xd)
            pq = QuantizeLinear(p, s, z)
            pd = DequantizeLinear(pq, s, z)
            g1 = Sigmoid(pd)
            gq = QuantizeLinear(g1, s, z)
            y = DequantizeLinear(gq, s, z)
        }
        """
    )
    # Quark's CLI names every node (and un-shares DQs) before converting
    from quark.onnx.utils.model_utils import copy_shared_nodes

    model = _quiet(copy_shared_nodes, model)
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_qdq_to_qop(model)

    def sig(m):
        return sorted(
            (n.op_type, n.domain, list(n.input), list(n.output)) for n in m.graph.node
        )

    assert sig(ours) == sig(theirs)
    assert {"QLinearMatMul", "QLinearAdd", "QLinearMul", "QLinearSigmoid"} <= {
        n.op_type for n in ours.graph.node
    }
    feed = {"x": _tx_x((2, 4)), "u": _tx_x((2, 4), 2)}
    np.testing.assert_array_equal(_tx_ort(ours, feed)[0], _tx_ort(theirs, feed)[0])
    # fused integer kernels agree with the QDQ graph to a few quantization steps
    np.testing.assert_allclose(
        _tx_ort(ours, feed)[0], _tx_ort(model, feed)[0], atol=0.3
    )


def test_tools_resize_fs_to_pof2s_matches_quark():
    from quark.onnx.tools.convert_resize_fs_to_pof2s import (
        convert_resize_fs_to_pof2s as q_fn,
    )

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 13]>
        g (float[1,1,4,4] x) => (float[1,1,8,8] y)
        <float s1 = {0.037}, int8 z1 = {3}, float s2 = {0.0123}, int8 z2 = {-5},
         float[4] scales = {1.0, 1.0, 2.0, 2.0}>
        {
            q1 = QuantizeLinear(x, s1, z1)
            d1 = DequantizeLinear(q1, s1, z1)
            r = Resize<mode = "nearest">(d1, , scales)
            q2 = QuantizeLinear(r, s2, z2)
            y = DequantizeLinear(q2, s2, z2)
        }
        """
    )
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_resize_fs_to_pof2s(model)
    _tx_same_inits(ours, theirs)
    a = _tx_inits(ours)
    assert a["z1"][1] == 0 and np.log2(float(a["s1"][1])) % 1 == 0
    x = _tx_x((1, 1, 4, 4))
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(theirs, {"x": x})[0]
    )


def _tx_u16_model():
    return parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,4] x) => (float[2,3] y)
        <float s = {0.0001}, uint16 z = {32768}, float sw = {0.0001}, uint16 zw = {32768},
         uint16[4,3] wq = {7068, 33000, 32768, 33025, 19918, 32768, 32768, 55879,
                            31997, 35338, 50798, 27628},
         float sb = {0.00001}, int32 zb = {0}, int32[3] bq = {5, -7, 100}>
        {
            xq = QuantizeLinear(x, s, z)
            xd = DequantizeLinear(xq, s, z)
            wd = DequantizeLinear(wq, sw, zw)
            m = MatMul(xd, wd)
            bd = DequantizeLinear(bq, sb, zb)
            a = Add(m, bd)
            aq = QuantizeLinear(a, s, z)
            y = DequantizeLinear(aq, s, z)
        }
        """
    )


def test_tools_u16s8_to_s16s8_matches_quark():
    from quark.onnx.tools.convert_u16s8_to_s16s8 import convert_u16s8_to_s16s8 as q_fn

    model = _tx_u16_model()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_u16s8_to_s16s8(model)
    # the activation zero point becomes int16 0; the weight DQ is untouched
    assert [n.input[2] for n in ours.graph.node if len(n.input) > 2] == [
        n.input[2] for n in theirs.graph.node if len(n.input) > 2
    ]
    _tx_same_inits(ours, theirs)
    x = _tx_x((2, 4))
    np.testing.assert_allclose(
        _tx_ort(ours, {"x": x})[0], _tx_ort(model, {"x": x})[0], atol=1e-6
    )
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(theirs, {"x": x})[0]
    )


def test_tools_u16u8_to_u8u8_matches_quark():
    from quark.onnx.tools.convert_u16u8_to_u8u8 import convert_u16u8_to_u8u8 as q_fn

    model = _tx_u16_model()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_u16u8_to_u8u8(model)
    a, b = _tx_inits(ours), _tx_inits(theirs)
    assert set(a) == set(b)
    for k in a:
        assert a[k][0] == b[k][0], k
        if k == "wq":
            # deliberate differences on the re-quantized uint16 constant:
            # (1) we round to nearest, Quark truncates toward zero -> one
            # code apart where both are right; (2) Quark computes `q - zp` in
            # uint16, which wraps for q < zp and saturates those weights to
            # 255 -- we dequantize them correctly.
            src = _tx_inits(model)["wq"][1].astype(np.int64)
            ok = src >= 32768
            assert np.abs(a[k][1].astype(int) - b[k][1].astype(int))[ok].max() <= 1
            exact = np.clip(
                np.rint((src - 32768) * 0.0001 / (0.0001 * 65535 / 255) + 128), 0, 255
            )
            assert np.abs(a[k][1].astype(int) - exact).max() <= 1
        else:
            np.testing.assert_array_equal(a[k][1], b[k][1], err_msg=k)
    x = _tx_x((2, 4))
    ref = _tx_ort(model, {"x": x})[0]
    # 8-bit activations: coarser by 257x, so allow a few steps of 0.0257
    np.testing.assert_allclose(_tx_ort(ours, {"x": x})[0], ref, atol=0.5)


def test_tools_fix_shapes_matches_quark():
    from quark.onnx.tools import fix_shapes as q_mod

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[N,3] x) => (float[N,2] y)
        <float[3,2] w = {1,2,3,4,5,6}>
        {
            a = Relu(x)
            y = MatMul(a, w)
        }
        """
    )
    spec = "x:[4,3];y:[4,2]"
    t = _quiet(q_mod.fix_input_and_output_shapes, _copy_model(model), spec)
    o = quark_tools.fix_input_and_output_shapes(model, spec)

    def dims(m):
        return [
            [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in list(m.graph.input) + list(m.graph.output)
        ]

    assert dims(o) == dims(t) == [[4, 3], [4, 2]]
    assert quark_tools.parse_input_and_output_shapes(
        spec
    ) == q_mod.parse_input_and_output_shapes(spec)
    # intermediate tensors: Quark runs the model; the result must agree
    inferred = onnx.shape_inference.infer_shapes(o)
    shapes = _quiet(q_mod.infer_all_tensors_shape, inferred)
    t_full = _quiet(q_mod.save_all_tensors_shape, inferred, shapes)
    o_full = quark_tools.fix_shapes(model, spec)

    def vi(m):
        return {
            v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in m.graph.value_info
        }

    assert vi(o_full)["a"] == vi(t_full)["a"] == [4, 3]


def test_tools_a16w8_a8w8_nodes_match_quark(tmp_path):
    from onnx import numpy_helper
    from quark.onnx.tools.print_a16w8_a8w8_nodes import a16w8_a8w8_nodes as q_fn

    m8 = _tx_conv_qdq()
    m16 = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x) => (float[1,4,6,6] y)
        <float xs = {0.02}, int16 xz = {0}, float ws = {0.01}, int8 wz = {0}>
        {
            xq = QuantizeLinear(x, xs, xz)
            xd = DequantizeLinear(xq, xs, xz)
            wd = DequantizeLinear(wq, ws, wz)
            y = Conv(xd, wd)
        }
        """
    )
    m16.graph.initializer.append(
        numpy_helper.from_array(np.ones((4, 3, 3, 3), np.int8), "wq")
    )
    m8.graph.node[4].name = "conv8"
    m16.graph.node[3].name = "conv16"
    none = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2] x) => (float[2] y)
        { y = Relu(x) }
        """
    )
    for model, want in (
        (m8, (["conv8"], [])),
        (m16, ([], ["conv16"])),
        (none, ([], [])),
    ):
        path = str(tmp_path / "m.onnx")
        onnx.save(model, path)
        assert quark_tools.a16w8_a8w8_nodes(model) == want
        assert tuple(_quiet(q_fn, path)) == want


def _tx_bf16_model():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21, "com.amd.quark": 1]>
        g (float[2,4] x) => (float[2,4] y)
        <float s = {0.5}, float one = {1.0}, bfloat16 zb = {0}, int8 z8 = {0}>
        {
            a = com.amd.quark.ExtendedQuantizeLinear(x, s, zb)
            b = com.amd.quark.ExtendedDequantizeLinear(a, s, zb)
            c = com.amd.quark.ExtendedQuantizeLinear(b, one, zb)
            d = com.amd.quark.ExtendedDequantizeLinear(c, one, zb)
            e = com.amd.quark.ExtendedQuantizeLinear(d, s, z8)
            y = com.amd.quark.ExtendedDequantizeLinear(e, s, z8)
        }
        """
    )
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}"
    return model


def test_tools_replace_bfloat16_qdq_cast_matches_quark():
    from quark.onnx.tools.replace_bfloat16_qdq_cast import (
        replace_bfloat16_qdq_cast as q_fn,
    )

    model = _tx_bf16_model()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.replace_bfloat16_qdq_cast(model)

    def sig(m):
        return [
            (n.op_type, n.domain, list(n.input), list(n.output)) for n in m.graph.node
        ]

    assert sorted(sig(ours)) == sorted(sig(theirs))
    assert sorted(n.op_type for n in ours.graph.node) == sorted(
        ["Mul", "Cast", "Cast", "Mul", "Cast", "Cast", "ExtendedQuantizeLinear"]
        + ["ExtendedDequantizeLinear"]
    )
    _tx_same_inits(ours, theirs)
    assert {k for k in _tx_inits(ours) if k.endswith("_scale")} == {
        "n0_scale",
        "n1_scale",
    }


def test_tools_insert_clip_bfloat16_qdq_matches_quark():
    from quark.onnx.tools.insert_clip_bfloat16_qdq import (
        insert_clip_bfloat16_qdq as q_fn,
    )

    model = _tx_bf16_model()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.insert_clip_bfloat16_qdq(model)
    assert _tx_sorted_nodes(ours) == _tx_sorted_nodes(theirs)

    def clip_inits(m):
        return {k: v for k, v in _tx_inits(m).items() if "clip" in k}

    a, b = clip_inits(ours), clip_inits(theirs)
    assert set(a) == set(b) and len(a) == 4
    for k in a:
        assert a[k][0] == b[k][0]
        np.testing.assert_array_equal(a[k][1], b[k][1], err_msg=k)

    def fed_by_clip(m):
        prod = {o: n for n in m.graph.node for o in n.output}
        return sorted(
            n.output[0]
            for n in m.graph.node
            if n.op_type == "ExtendedQuantizeLinear"
            and n.input[0] in prod
            and prod[n.input[0]].op_type == "Clip"
        )

    assert fed_by_clip(ours) == fed_by_clip(theirs) == ["a", "c"]


def _tx_cast_model():
    # Quark reconnects the consumers of the second cast only when the first
    # node's output name is a *substring* of that cast's output name (it tests
    # `a in b` on strings), hence the a / a_c1 / a_c2 naming. onnxsim rewires
    # unconditionally.
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,4] x) => (float[2,4] y)
        <float[4] w = {0.1234567, -2.7182818, 3.14159265, 1000.123}>
        {
            a = Relu(x)
            a_c1 = Cast<to = 16>(a)
            a_c2 = Cast<to = 1>(a_c1)
            wb = Cast<to = 16>(w)
            wf = Cast<to = 1>(wb)
            m = Add(a_c2, wf)
            n = Mul(m, a_c2)
            o1 = Cast<to = 16>(n)
            y = Cast<to = 1>(o1)
        }
        """
    )
    return model


def test_tools_remove_bf16_cast_matches_quark():
    from quark.onnx.tools.remove_bf16_cast import remove_bf16_cast as q_fn

    base = _tx_cast_model()
    theirs = _quiet(q_fn, _copy_model(base))
    ours = quark_tools.remove_bf16_cast(base)
    assert [n.op_type for n in ours.graph.node] == [
        n.op_type for n in theirs.graph.node
    ]
    assert [n.op_type for n in ours.graph.node] == ["Relu", "Add", "Mul"]
    a, b = _tx_inits(ours), _tx_inits(theirs)
    assert set(a) == set(b) == {"w_bf16"}
    np.testing.assert_array_equal(a["w_bf16"][1], b["w_bf16"][1])
    x = _tx_x((2, 4))
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(theirs, {"x": x})[0]
    )


def _tx_between_model():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x, float[1,4,6,6] u) => (float[1,4,6,6] y)
        <float s = {0.1}, int8 z = {0}>
        {
            c = Conv(x, wf)
            cq = QuantizeLinear(c, s, z)
            cd = DequantizeLinear(cq, s, z)
            r = Relu(cd)
            rq = QuantizeLinear(r, s, z)
            rd = DequantizeLinear(rq, s, z)
            mu = Mul(rd, u)
            mq = QuantizeLinear(mu, s, z)
            md = DequantizeLinear(mq, s, z)
            y = Add(md, u)
        }
        """
    )
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(np.full((4, 3, 3, 3), 0.5, np.float32), "wf")
    )
    return model


@pytest.mark.parametrize(
    "between",
    [[("Conv", "Relu")], [("Relu", "Mul"), ("Mul", "Add")], [("Mul", "Add")]],
)
def test_tools_remove_qdq_between_ops_matches_quark(between):
    from quark.onnx.tools.remove_qdq_between_ops import remove_qdq_between_ops as q_fn

    model = _tx_between_model()
    theirs = _quiet(q_fn, _copy_model(model), between)
    ours = quark_tools.remove_qdq_between_ops(model, between)
    assert sorted(n.op_type for n in ours.graph.node) == sorted(
        n.op_type for n in theirs.graph.node
    )
    assert _tx_inits(ours).keys() == _tx_inits(theirs).keys()
    assert len(ours.graph.node) == len(model.graph.node) - 2 * len(between)
    feed = {"x": _tx_x(), "u": _tx_x((1, 4, 6, 6), 3)}
    np.testing.assert_array_equal(_tx_ort(ours, feed)[0], _tx_ort(theirs, feed)[0])


def test_tools_remove_qdq_mul_add_matches_quark():
    from quark.onnx.tools.remove_qdq_mul_add import remove_qdq_mul_add as q_fn

    model = _tx_between_model()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.remove_qdq_mul_add(model)
    assert sorted(n.op_type for n in ours.graph.node) == sorted(
        n.op_type for n in theirs.graph.node
    )
    feed = {"x": _tx_x(), "u": _tx_x((1, 4, 6, 6), 3)}
    np.testing.assert_array_equal(_tx_ort(ours, feed)[0], _tx_ort(theirs, feed)[0])


def test_tools_onnxtxt_roundtrip_matches_quark():
    from google.protobuf import text_format

    model = _tx_conv_qdq()
    text = quark_tools.convert_onnx_to_onnxtxt(model)
    assert text == text_format.MessageToString(model)  # what Quark's CLI writes
    back = quark_tools.convert_onnxtxt_to_onnx(text)
    assert back == model
    parsed = onnx.ModelProto()
    text_format.Parse(text.encode(), parsed)  # Quark's CLI reads bytes
    assert parsed == back


def _tx_shared_models():
    shared_dq = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,4] x) => (float[2,4] y)
        <float s = {0.1}, int8 z = {0},
         int8[4,4] wq = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}>
        {
            w = DequantizeLinear(wq, s, z)
            a = MatMul(x, w)
            y = MatMul(a, w)
        }
        """
    )
    shared_init = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,4] x) => (float[2,4] y)
        <float[4,4] w = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}>
        {
            a = MatMul(x, w)
            y = MatMul(a, w)
        }
        """
    )
    return shared_dq, shared_init


@pytest.mark.parametrize("which", [0, 1])
def test_tools_copy_shared_nodes_matches_quark(which):
    from quark.onnx.utils.model_utils import check_shared_initializers
    from quark.onnx.utils.model_utils import copy_shared_nodes as q_fn

    model = _tx_shared_models()[which]
    # a DQ-shared model has no shared *initializer* (wq is read once)
    expect = bool(which)
    assert check_shared_initializers(model) is quark_tools.check_shared_initializers(
        model
    )
    assert quark_tools.check_shared_initializers(model) is expect
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.copy_shared_nodes(model)

    def sig(m):
        return (
            sorted(n.op_type for n in m.graph.node),
            sorted(t.name for t in m.graph.initializer),
            sorted(n.name for n in m.graph.node),
        )

    assert sig(ours) == sig(theirs)
    assert not quark_tools.check_shared_initializers(ours)
    x = _tx_x((2, 4))
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(model, {"x": x})[0]
    )
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(theirs, {"x": x})[0]
    )


def test_tools_clean_initializer_in_input_matches_quark():
    from quark.onnx.utils.model_utils import clean_initializer_in_input as q_fn

    model = parser.parse_model(
        """
        <ir_version: 3, opset_import: ["": 9]>
        g (float[2] x, float[2] w) => (float[2] y)
        <float[2] w = {1.0, 2.0}>
        { y = Add(x, w) }
        """
    )
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.clean_initializer_in_input(model)
    assert [i.name for i in ours.graph.input] == [i.name for i in theirs.graph.input]
    assert ours.ir_version == theirs.ir_version == 4
    assert model.ir_version == 3  # ours does not mutate the argument


def test_tools_save_with_external_data_matches_quark(tmp_path):
    from quark.onnx.utils.model_utils import (
        save_onnx_model_with_external_data as q_fn,
    )

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,64] x) => (float[2,64] y)
        { y = MatMul(x, w) }
        """
    )
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32), "w"
        )
    )
    for tag, fn in (
        ("q", q_fn),
        ("o", quark_tools.save_onnx_model_with_external_data),
    ):
        path = str(tmp_path / f"{tag}.onnx")
        _quiet(fn, _copy_model(model), path, True)
        assert (tmp_path / f"{tag}.onnx.data").exists()
        loaded = onnx.load(path)
        assert _tx_inits(loaded).keys() == _tx_inits(model).keys()
        for k, v in _tx_inits(loaded).items():
            np.testing.assert_array_equal(v[1], _tx_inits(model)[k][1])


# == calibration / scale parity (power-of-two MinMSE, int8 biases, methods) =======


def _heavy(rng, *shape):
    """Student-t weights: heavy tails make clipping beat ``ceil(log2)`` scales."""
    return (rng.standard_t(2.5, shape) * 0.3).astype(np.float32)


def _cal_inits(pairs):
    return [onnx.numpy_helper.from_array(a, n) for n, a in pairs]


def _cal_mlp(seed):
    rng = np.random.default_rng(seed)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[4,24] x) => (float[4,10] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            h2 = Gemm(h1, w2, b2)
            h3 = Relu(h2)
            y = Gemm(h3, w3, b3)
        }
        """
    )
    m.graph.initializer.extend(
        _cal_inits(
            [
                ("w1", _heavy(rng, 24, 48)),
                ("b1", _heavy(rng, 48)),
                ("w2", _heavy(rng, 48, 32)),
                ("b2", _heavy(rng, 32)),
                ("w3", _heavy(rng, 32, 10)),
                ("b3", _heavy(rng, 10)),
            ]
        )
    )
    return m, (4, 24)


def _cal_conv(seed):
    rng = np.random.default_rng(seed)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[2,3,12,12] x) => (float[2,6,6,6] y) {
            c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
            r0 = Relu(c0)
            c1 = Conv<pads=[1,1,1,1]>(r0, w2, b2)
            r1 = Relu(c1)
            y = MaxPool<kernel_shape=[2,2], strides=[2,2]>(r1)
        }
        """
    )
    m.graph.initializer.extend(
        _cal_inits(
            [
                ("w1", _heavy(rng, 8, 3, 3, 3)),
                ("b1", _heavy(rng, 8)),
                ("w2", _heavy(rng, 6, 8, 3, 3)),
                ("b2", _heavy(rng, 6)),
            ]
        )
    )
    return m, (2, 3, 12, 12)


def _cal_attn(seed):
    """LayerNorm / Softmax / residual Add around four projections."""
    rng = np.random.default_rng(seed)
    d = 16
    m = parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["": 17]>
        g (float[1,6,{d}] x) => (float[1,6,{d}] y) {{
            n = LayerNormalization<axis=-1, epsilon=1e-5>(x, ln_s, ln_b)
            q = MatMul(n, wq)
            k = MatMul(n, wk)
            kt = Transpose<perm=[0,2,1]>(k)
            s = MatMul(q, kt)
            sc = Mul(s, c)
            p = Softmax<axis=-1>(sc)
            v = MatMul(n, wv)
            a = MatMul(p, v)
            o = MatMul(a, wo)
            y = Add(x, o)
        }}
        """
    )
    m.graph.initializer.extend(
        _cal_inits(
            [
                ("ln_s", (1 + 0.2 * rng.standard_normal(d)).astype(np.float32)),
                ("ln_b", (0.1 * rng.standard_normal(d)).astype(np.float32)),
                ("wq", _heavy(rng, d, d)),
                ("wk", _heavy(rng, d, d)),
                ("wv", _heavy(rng, d, d)),
                ("wo", _heavy(rng, d, d)),
                ("c", np.array(0.25, np.float32)),
            ]
        )
    )
    return m, (1, 6, d)


CAL_MODELS = {"mlp": _cal_mlp, "conv": _cal_conv, "attn": _cal_attn}


def _norm_name(name):
    for cut in ("_QuantizeLinear_Input", "_quantized", "/f", "/dq"):
        name = name.removesuffix(cut)
    return name.split("/qdq")[0]


def _qparam_map(model):
    """``({tensor: (scale, zero_point, dtype)}, {initializer: (dequantized,
    int8-code abs-sum)})``: every activation Q node, and every DQ that reads
    an initializer."""
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    acts, consts = {}, {}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[1] in inits:
            zp = inits[n.input[2]]
            acts[_norm_name(n.input[0])] = (
                float(inits[n.input[1]]),
                int(zp),
                str(zp.dtype),
            )
        elif n.op_type == "DequantizeLinear" and n.input[0] in inits:
            q = inits[n.input[0]]
            deq = (q.astype(np.float64) - inits[n.input[2]]) * inits[n.input[1]]
            consts[_norm_name(n.input[0])] = (
                deq,
                q.dtype.name,
                float(np.max(inits[n.input[1]])),
            )
    return acts, consts


def _check_cal_parity(model, shape, preset, tmp_path, rtol, zp_atol=0, constants=True):
    q_acts, q_consts = _qparam_map(quark_quantize(model, preset, shape, tmp_path))
    m_acts, m_consts = _qparam_map(mine_quantize(model, preset, shape))
    assert q_acts and set(q_acts) == set(m_acts)
    for name, (scale, zp, dt) in q_acts.items():
        ms, mz, mdt = m_acts[name]
        assert mdt == dt, name
        np.testing.assert_allclose(ms, scale, rtol=rtol, err_msg=name)
        assert abs(mz - zp) <= zp_atol, (name, mz, zp)
    if not constants:
        return
    assert set(q_consts) == set(m_consts)
    for name, (want, dt, scale) in q_consts.items():
        got, mdt, _ = m_consts[name]
        assert mdt == dt, name
        # int8 weights / biases dequantize identically; an int32 bias carries
        # the activation scale's (percentile-binning) difference: a code or two
        atol = 2 * scale if dt == "int32" else 1e-4
        np.testing.assert_allclose(
            got, want, rtol=max(rtol, 1e-6), atol=atol, err_msg=name
        )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("model_name", sorted(CAL_MODELS))
def test_xint8_minmse_pof2_matches_quark(model_name, seed, tmp_path):
    """XINT8: power-of-two scales picked by MinMSE -- activations (histogram
    search), weights, and the int8 biases / constants -- are identical to
    Quark's: same scale, zero point and dtype per tensor, same int8 codes."""
    model, shape = CAL_MODELS[model_name](seed)
    _check_cal_parity(model, shape, "XINT8", tmp_path, rtol=0)


def _drain(reader):
    out = []
    while (b := reader.get_next()) is not None:
        out.append(b)
    return out


def test_xint8_cases_where_ceil_log2_differs_from_quark(tmp_path):
    """The previous ``ceil(log2(scale))`` / int32-bias rule lands on different
    scales than Quark on these heavy-tailed models, so the parity above is not
    vacuous."""
    from onnxsim.full_qdq import quantize_full_qdq

    differing = 0
    for name, build in sorted(CAL_MODELS.items()):
        for seed in range(4):
            model, shape = build(seed)
            q_acts, q_consts = _qparam_map(
                quark_quantize(model, "XINT8", shape, tmp_path)
            )
            old = quantize_full_qdq(
                model,
                _drain(_reader(shape)()),
                activation_dtype="uint8",
                method="minmax",
                symmetric_activations=True,
                power_of_two=True,
                per_channel=False,
            )
            o_acts, o_consts = _qparam_map(old)
            differing += any(
                o_acts[k][0] != v[0] for k, v in q_acts.items() if k in o_acts
            )
    assert differing >= 4


@pytest.mark.parametrize("model_name", sorted(CAL_MODELS))
def test_xint8_int32_bias_option_keeps_int32(model_name):
    model, shape = CAL_MODELS[model_name](0)
    cfg = qc.QConfig.get_default_config("XINT8")
    cfg.extra_options["Int32Bias"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_reader(shape)()
        )
    dtypes = {onnx.numpy_helper.to_array(i).dtype for i in out.graph.initializer}
    has_bias = any(n.op_type in ("Gemm", "Conv") for n in model.graph.node)
    assert (np.dtype(np.int32) in dtypes) == has_bias


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("model_name", sorted(CAL_MODELS))
@pytest.mark.parametrize(
    "preset, rtol, zp_atol",
    [
        # MinMax: the same range -> the same scale
        ("A8W8", 1e-6, 0),
        ("A16W8", 1e-6, 0),
        # Percentile: histogram binning moves a scale by a few 1e-4
        ("U8S8_AAWS", 2e-3, 1),
        ("S8S8_AAWS", 2e-3, 1),
        ("U8U8_AAWA", 2e-3, 1),
        ("S16S8_ASWS", 2e-3, 40),
        ("U16S8_AAWS", 2e-3, 40),
    ],
)
def test_calibration_methods_match_quark_per_preset(
    preset, rtol, zp_atol, model_name, seed, tmp_path
):
    """Per-tensor scale / zero point / dtype and dequantized weights, biases
    and constants of each preset's calibration method (MinMax, Percentile
    99.999 / 99.9999, symmetric or not), including Softmax's fixed (0, 1)
    output range and the int8 weight treatment of non-weight constants."""
    model, shape = CAL_MODELS[model_name](seed)
    # U8U8_AAWA's uint8 asymmetric weights are a documented approximation
    # (int8 symmetric here): only its activations are compared
    _check_cal_parity(
        model, shape, preset, tmp_path, rtol, zp_atol, constants=preset != "U8U8_AAWA"
    )
