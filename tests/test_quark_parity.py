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
    "INT16_TRANSFORMER_ACCURATE",
    "INT16_TRANSFORMER_DEFAULT",
    "INT8_TRANSFORMER_ACCURATE",
    "INT8_TRANSFORMER_DEFAULT",
    "MATMUL_NBITS",
    "UINT8_DYNAMIC_QUANT",
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


# -- preset combinations: mixed formats and mixed precision -------------------------
#
# BF16_BFP16 / BF16_MXINT8 (bfloat16 activations over block-format constants),
# MX9_INT8 (block-format activations over int8 constants) and the
# BF16_MIXED_BFP16 / BF16_MIXED_MXINT8 AutoMixprecision presets. Quark's mixed
# presets pick candidate layers by *node name*, so these models name every node.


def _named(model):
    for i, n in enumerate(model.graph.node):
        n.name = f"{n.op_type}_{i}"
    return model


def _transformer():
    """Attention + MLP block. No ``MatMul`` -> ``Add(const)`` (Quark's
    pre-processing would fuse that into a ``Gemm``)."""
    rng = np.random.default_rng(11)
    d = 16
    m = parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["": 20]>
        g (float[1,4,{d}] x) => (float[1,4,{d}] y) {{
            q = MatMul(x, wq)
            k = MatMul(x, wk)
            v = MatMul(x, wv)
            kt = Transpose<perm=[0,2,1]>(k)
            s0 = MatMul(q, kt)
            s1 = Mul(s0, scale)
            p = Softmax<axis=-1>(s1)
            c = MatMul(p, v)
            o = MatMul(c, wo)
            r = Add(x, o)
            n = LayerNormalization<axis=-1, epsilon=1e-5>(r, g1, b1)
            h = MatMul(n, w1)
            ge = Gelu(h)
            f = MatMul(ge, w2)
            y = Add(n, f)
        }}
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, d, d), "wq"),
            onnx.numpy_helper.from_array(_w(rng, d, d), "wk"),
            onnx.numpy_helper.from_array(_w(rng, d, d), "wv"),
            onnx.numpy_helper.from_array(_w(rng, d, d), "wo"),
            onnx.numpy_helper.from_array(np.array(0.25, np.float32), "scale"),
            onnx.numpy_helper.from_array(np.ones(d, np.float32), "g1"),
            onnx.numpy_helper.from_array(np.zeros(d, np.float32), "b1"),
            onnx.numpy_helper.from_array(_w(rng, d, 32), "w1"),
            onnx.numpy_helper.from_array(_w(rng, 32, d), "w2"),
        ]
    )
    return m, (1, 4, d)


def _branchy():
    """A residual ``Add`` reading a tensor that also feeds a ``Gemm``, a
    Gemm -> Gemm chain without activation in between, and a ``MatMul`` last."""
    rng = np.random.default_rng(12)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,16] y) {
            a = Gemm(x, w1, b1)
            b = Gemm(a, w2, b2)
            r = Add(x, b)
            y = MatMul(r, w3)
        }
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, 16, 16), "w1"),
            onnx.numpy_helper.from_array(_w(rng, 16), "b1"),
            onnx.numpy_helper.from_array(_w(rng, 16, 16), "w2"),
            onnx.numpy_helper.from_array(_w(rng, 16), "b2"),
            onnx.numpy_helper.from_array(_w(rng, 16, 16), "w3"),
        ]
    )
    return m, (3, 16)


MIXED_MODELS = {**MODELS, "branchy": _branchy, "transformer": _transformer}
_MIXED_FORMATS = ["BF16_BFP16", "BF16_MXINT8", "MX9_INT8"]
_MIXED_PRECISION = ["BF16_MIXED_BFP16", "BF16_MIXED_MXINT8"]


def _mixed_pair(preset, model_name, tmp_path):
    model, shape = MIXED_MODELS[model_name]()
    _named(model)
    return (
        quark_quantize(model, preset, shape, tmp_path, f"{model_name}_{preset}"),
        mine_quantize(model, preset, shape),
        shape,
    )


@pytest.mark.parametrize("model_name", sorted(MIXED_MODELS))
@pytest.mark.parametrize("preset", _MIXED_FORMATS + _MIXED_PRECISION)
def test_mixed_preset_graph_matches_quark(preset, model_name, tmp_path):
    """Same custom-op / (Extended)Q/DQ placement, attributes, block axes and
    even the names of the dual nodes Quark inserts at precision boundaries."""
    q, m, _ = _mixed_pair(preset, model_name, tmp_path)
    assert _cop_map(m) == _cop_map(q)
    assert _op_counts(m) == _op_counts(q)


@pytest.mark.skipif(_ops_lib() is None, reason="Quark's custom-op library is not built")
@pytest.mark.parametrize("model_name", sorted(MIXED_MODELS))
@pytest.mark.parametrize("preset", _MIXED_FORMATS + _MIXED_PRECISION)
def test_mixed_preset_outputs_match_quark(preset, model_name, tmp_path):
    q, m, shape = _mixed_pair(preset, model_name, tmp_path)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32) * 2
    np.testing.assert_allclose(_run(m, x), _run(q, x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model_name", ["mlp", "conv", "transformer"])
def test_mx9_int8_constants_are_bit_identical(model_name, tmp_path):
    """The int8 codes, scales and zero points of the constants (weights *and*
    biases: symmetric per tensor, ``max|w| / 127``) equal Quark's."""
    q, m, _ = _mixed_pair("MX9_INT8", model_name, tmp_path)

    def consts(model):
        return {
            i.name: onnx.numpy_helper.to_array(i)
            for i in model.graph.initializer
            if i.name.endswith(("_quantized", "_scale", "_zero_point"))
        }

    theirs, ours = consts(q), consts(m)
    assert theirs and set(ours) == set(theirs)
    for name in theirs:
        assert ours[name].dtype == theirs[name].dtype, name
        np.testing.assert_array_equal(ours[name], theirs[name], err_msg=name)


@pytest.mark.parametrize("preset", _MIXED_FORMATS + _MIXED_PRECISION)
def test_mixed_preset_single_op_placement_matches_quark(preset, tmp_path):
    """Quark's op coverage, op by op, for the combined presets: the op counts
    (nodes, custom ops, Q/DQ) agree on every single-op graph Quark accepts."""
    diffs = []
    for name, case in _single_op_cases().items():
        case = dict(case, op=case.get("op", name))
        model = _named(_single_op_model(case))
        try:
            q = quark_quantize(model, preset, case["shapes"][0], tmp_path, name)
        except Exception:  # Quark itself rejects this graph
            continue
        m = mine_quantize(model, preset, case["shapes"][0])
        got, want = _op_counts(m), _op_counts(q)
        if got != want and name not in KNOWN_GRAPH_DIFF:
            diffs.append((name, got, want))
    assert not diffs, json.dumps(diffs, indent=1)


def test_mixed_precision_keeps_biases_float_and_matches_quark(tmp_path):
    """Quark's default ``metric_threshold=0`` promotes every Conv / Gemm /
    MatMul; the biases stay float constants (``QuantizeBias=False``)."""
    model, shape = _mlp()
    _named(model)
    q = quark_quantize(model, "BF16_MIXED_BFP16", shape, tmp_path, "promote")
    m = mine_quantize(model, "BF16_MIXED_BFP16", shape)
    for out in (q, m):
        reads = {i for n in out.graph.node if n.op_type == "Gemm" for i in n.input}
        assert {"b1", "b2"} <= reads  # read straight from the float initializers
    assert _cop_map(m) == _cop_map(q)


@pytest.mark.parametrize("preset", ["FP16_ADAQUANT", "BF16_ADAQUANT"])
def test_half_adaquant_graph_matches_quark(preset, tmp_path):
    """Quark's FP16/BF16 AdaQuant presets emit the plain FP16/BF16 graph (the
    algorithm only retunes initializer values; it also fails outright on a
    graph whose last op is a pooling, so only the MLP is probed). onnxsim has
    no AdaQuant for float formats: the preset exists, and runs only when told
    to ignore the algorithm."""
    model, shape = _mlp()
    q = quark_quantize(model, preset, shape, tmp_path, preset)
    cfg = qc.QConfig.get_default_config(preset)
    with pytest.raises(NotImplementedError):
        qc.ModelQuantizer(cfg).quantize_model(model, calibration_data_reader=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=None, ignore_unsupported_algos=True
        )
    assert _ext_map(m) == _ext_map(q)
    assert _op_counts(m) == _op_counts(q)


@pytest.mark.parametrize(
    "preset", ["BF16_MIXED_BFP16_ADAQUANT", "BF16_MIXED_MXINT8_ADAQUANT"]
)
def test_mixed_adaquant_graph_matches_quark(preset, tmp_path):
    model, shape = _mlp()
    _named(model)
    q = quark_quantize(model, preset, shape, tmp_path, preset)
    cfg = qc.QConfig.get_default_config(preset)
    with pytest.raises(NotImplementedError):
        qc.ModelQuantizer(cfg).quantize_model(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = qc.ModelQuantizer(cfg).quantize_model(model, ignore_unsupported_algos=True)
    assert _cop_map(m) == _cop_map(q)
    assert _op_counts(m) == _op_counts(q)


@pytest.mark.parametrize("model_name", ["branchy", "transformer"])
@pytest.mark.parametrize("preset", _BLOCK + _HALF)
def test_block_and_half_presets_match_quark_on_transformer_graphs(
    preset, model_name, tmp_path
):
    """The single-format presets on the attention block (``LayerNormalization``
    is quantized end to end when its input already is, scale / bias included)
    and the residual graph."""
    q, m, _ = _mixed_pair(preset, model_name, tmp_path)
    assert _cop_map(m) == _cop_map(q)
    assert _ext_map(m) == _ext_map(q)
    assert _op_counts(m) == _op_counts(q)


# -- integer presets: the CNN presets and the mixed int16 / int8 preset ---------------


def _qdq_params(model):
    """``(activations, weights, int8_biases)``: every activation
    ``QuantizeLinear``'s ``(scale, zero_point, dtype)``, each weight's
    ``(max scale, dtype)`` (int8 / int16 codes of rank >= 2) and the int8
    bias codes by name."""
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    acts, weights, biases = [], [], {}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[1] in inits:
            zp = inits[n.input[2]]
            acts.append((float(inits[n.input[1]]), int(zp), str(zp.dtype)))
        elif n.op_type == "DequantizeLinear" and n.input[0] in inits:
            q = inits[n.input[0]]
            if q.dtype in (np.int8, np.int16) and q.ndim >= 2:
                weights.append((float(np.max(inits[n.input[1]])), str(q.dtype)))
            elif q.dtype == np.int8:
                key = n.input[0].replace("_quantized", "").split("/")[0]
                biases[key] = (q, float(inits[n.input[1]]))
    return sorted(acts), sorted(weights), biases


def _quantize_int_pair(preset, model_name, tmp_path):
    model, shape = MIXED_MODELS[model_name]()
    _named(model)
    q = quark_quantize(model, preset, shape, tmp_path, f"{model_name}_{preset}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # INT16_CNN_ACCURATE's AdaRound cannot run on int16 weights
        m = qc.ModelQuantizer(qc.QConfig.get_default_config(preset)).quantize_model(
            model,
            calibration_data_reader=_reader(shape)(),
            ignore_unsupported_algos=True,
        )
    return model, q, m, shape


@pytest.mark.parametrize("model_name", ["mlp", "conv", "gemm_transb", "branchy"])
@pytest.mark.parametrize("preset", ["INT8_CNN_DEFAULT", "INT16_CNN_DEFAULT"])
def test_cnn_default_presets_match_quark_exactly(preset, model_name, tmp_path):
    """Plain min/max calibration, asymmetric uint8 / uint16 activations,
    per-tensor int8 / int16 weights: the same quantization parameters, and the
    same outputs, as Quark."""
    _, q, m, shape = _quantize_int_pair(preset, model_name, tmp_path)
    q_acts, q_w, _ = _qdq_params(q)
    m_acts, m_w, _ = _qdq_params(m)
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    np.testing.assert_array_equal([a[1] for a in m_acts], [a[1] for a in q_acts])
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=1e-5
    )
    assert [w[1] for w in m_w] == [w[1] for w in q_w]
    np.testing.assert_allclose([w[0] for w in m_w], [w[0] for w in q_w], rtol=1e-6)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    np.testing.assert_allclose(_run(m, x), _run(q, x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("model_name", ["mlp", "conv", "gemm_transb"])
@pytest.mark.parametrize("preset", ["INT8_CNN_ACCURATE", "INT16_CNN_ACCURATE"])
def test_cnn_accurate_presets_match_quark_parameters(preset, model_name, tmp_path):
    """Percentile 99.9999 calibration: activation parameters agree up to
    histogram binning, weight scales exactly. AdaRound only changes weight
    codes (and runs for int8 weights only: with int16 weights the preset
    raises unless ``ignore_unsupported_algos``)."""
    model, q, m, shape = _quantize_int_pair(preset, model_name, tmp_path)
    q_acts, q_w, _ = _qdq_params(q)
    m_acts, m_w, _ = _qdq_params(m)
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    np.testing.assert_allclose(
        [a[1] for a in m_acts], [a[1] for a in q_acts], atol=40 if "16" in preset else 2
    )
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=2e-3
    )
    np.testing.assert_allclose([w[0] for w in m_w], [w[0] for w in q_w], rtol=1e-6)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    ref = _run(model, x)

    def rel(a):
        return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))

    assert rel(_run(m, x)) < max(3 * rel(_run(q, x)), 0.05)


def test_int16_cnn_accurate_needs_ignore_flag_for_adaround():
    model, shape = _mlp()
    cfg = qc.QConfig.get_default_config("INT16_CNN_ACCURATE")
    with pytest.raises(NotImplementedError, match="adaround"):
        qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_reader(shape)()
        )


@pytest.mark.parametrize("model_name", ["mlp", "conv", "gemm_transb", "branchy"])
def test_s16s16_mixed_s8s8_matches_quark(model_name, tmp_path):
    """int16 everywhere, every Conv / Gemm / MatMul promoted to int8 inputs,
    weights and (int8, per-tensor) biases; the promoted layers' outputs stay
    int16 and no convert pairs appear."""
    model, q, m, shape = _quantize_int_pair("S16S16_MIXED_S8S8", model_name, tmp_path)
    # (Quark keeps a Relu behind its producer; onnxsim folds it into the Q)
    assert {k: v for k, v in _op_counts(m).items() if k != "Relu"} == {
        k: v for k, v in _op_counts(q).items() if k != "Relu"
    }
    q_acts, q_w, q_b = _qdq_params(q)
    m_acts, m_w, m_b = _qdq_params(m)
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    np.testing.assert_allclose([a[1] for a in m_acts], [a[1] for a in q_acts], atol=40)
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=2e-3
    )
    assert m_w == q_w
    assert set(m_b) == set(q_b) and q_b
    for k in q_b:  # biases: scale max|b|/127, codes equal (up to a rounding tie)
        np.testing.assert_allclose(m_b[k][1], q_b[k][1], rtol=1e-6)
        np.testing.assert_allclose(m_b[k][0], q_b[k][0], atol=1)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    ref = _run(model, x)

    def rel(a):
        return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))

    assert rel(_run(m, x)) < max(2 * rel(_run(q, x)), 0.02)
    assert float(np.linalg.norm(_run(m, x) - _run(q, x)) / np.linalg.norm(ref)) < 0.02


@pytest.mark.parametrize("model_name", sorted(MIXED_MODELS))
def test_vint8_matches_quark(model_name, tmp_path):
    """Signed power-of-two int8 everywhere: Quark's ``VINT8`` quantizes every
    activation (no Relu folding, one dedicated Q/DQ pair per consumer) and
    stores weights *and biases* as per-tensor int8. Same graph structure and
    weight / bias parameters; activation scales are powers of two that may sit
    one octave off (Quark searches the MSE-best power of two, onnxsim rounds
    up)."""
    model, q, m, shape = _quantize_int_pair("VINT8", model_name, tmp_path)
    assert _op_counts(m) == _op_counts(q)
    q_acts, q_w, q_b = _qdq_params(q)
    m_acts, m_w, m_b = _qdq_params(m)
    assert [(a[1], a[2]) for a in m_acts] == [(0, "int8")] * len(q_acts)
    assert len(m_acts) == len(q_acts)
    log2 = lambda acts: np.log2([a[0] for a in acts])  # noqa: E731
    assert np.all(log2(m_acts) == np.round(log2(m_acts)))
    assert np.all(np.abs(log2(m_acts) - log2(q_acts)) <= 1)
    assert m_w == q_w
    assert set(m_b) == set(q_b)
    if model_name != "transformer":  # (there the constants are activation-path ones)
        for k in q_b:
            assert m_b[k][1] == q_b[k][1], k
            np.testing.assert_array_equal(m_b[k][0], q_b[k][0], err_msg=k)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    ref = _run(model, x)
    err = float(np.linalg.norm(_run(m, x) - _run(q, x)) / np.linalg.norm(ref))
    assert err < 0.1
