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
    "UINT8_DYNAMIC_QUANT",
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
