"""AutoMixprecision parity with the real AMD Quark (``amd-quark`` 0.13).

Every case runs the same request through Quark and through
``onnxsim.quark_compat`` and compares the results: the same layers moved, the
same quantization parameters on every Q/DQ slot (activation scales / zero
points, weight codes, bias codes) and bit-identical outputs of the two models
under ``ORT_DISABLE_ALL``. Covered here: weight / bias precision mixing, the
scoring conventions that decide the candidate ranking (graph optimizations,
``data_size`` counting one batch more, the in-place weight / bias handling),
``no_input_qdq_shared`` with shared inputs, the int16 -> int8 promotion
(``S16S16_MIXED_S8S8``) with ``subgraph_json`` / ``sensitivity_cache_file``, and
the same options on the block-format presets (``BF16_MIXED_BFP16`` /
``BF16_MIXED_MXINT8``). Quark-free counterparts: ``tests/test_quark_amp_mixing.py``.
"""

import contextlib
import copy
import io
import json
import os
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

warnings.filterwarnings("ignore")

with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
):
    try:
        import quark.onnx as quark_onnx
        from quark.onnx.quantization.config import spec as quark_spec
    except Exception as e:  # pragma: no cover - environment dependent
        quark_onnx = None
        _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    quark_onnx is None, reason="AMD Quark (amd-quark) is not installed"
)

from onnxsim import quark_compat as qc  # noqa: E402

D = 16


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    # Quark drops report files (quantized_info.csv, ...) into the working directory
    monkeypatch.chdir(tmp_path)


# -- models ------------------------------------------------------------------------------


def _w(rng, *shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _model(body, inits, inputs, outputs):
    m = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": 17]> g ({inputs}) => ({outputs}) {{ {body} }}'
    )
    m.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in inits.items())
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return m


def _mlp():
    rng = np.random.default_rng(0)
    return _model(
        """h1 = Gemm(x, w1, b1)  t1 = Tanh(h1)  h2 = Gemm(t1, w2, b2)
           t2 = Tanh(h2)  y = Gemm(t2, w3, b3)""",
        dict(
            w1=_w(rng, D, D, scale=1.5),
            b1=_w(rng, D),
            w2=_w(rng, D, D, scale=0.3),
            b2=_w(rng, D),
            w3=_w(rng, D, D),
            b3=_w(rng, D),
        ),
        f"float[3,{D}] x",
        f"float[3,{D}] y",
    )


def _convnet():
    rng = np.random.default_rng(0)
    return _model(
        """c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)  r1 = Relu(c1)
           c2 = Conv<pads=[1,1,1,1]>(r1, w2, b2)  r2 = Relu(c2)
           c3 = Conv<pads=[1,1,1,1]>(r2, w3)
           a = Add(c3, r1)  p = GlobalAveragePool(a)  f = Flatten(p)
           y = Gemm(f, w4, b4)""",
        dict(
            w1=_w(rng, 8, 3, 3, 3, scale=0.3),
            b1=_w(rng, 8, scale=0.3),
            w2=_w(rng, 8, 8, 3, 3, scale=0.2),
            b2=_w(rng, 8, scale=0.3),
            w3=_w(rng, 8, 8, 3, 3, scale=0.1),
            w4=_w(rng, 8, 10, scale=0.3),
            b4=_w(rng, 10, scale=0.3),
        ),
        "float[2,3,8,8] x",
        "float[2,10] y",
    )


def _fanout():
    # x feeds two layers (shared input), and so does t2 (a shared mid tensor)
    rng = np.random.default_rng(1)
    return _model(
        """a = Gemm(x, w1, b1)  b = Gemm(x, w2, b2)  s = Add(a, b)  t = Tanh(s)
           c = Gemm(t, w3, b3)  d = Gemm(t, w4, b4)  e = Add(c, d)  y = Gemm(e, w5, b5)""",
        dict(
            w1=_w(rng, D, D),
            b1=_w(rng, D),
            w2=_w(rng, D, D, scale=1.2),
            b2=_w(rng, D),
            w3=_w(rng, D, D),
            b3=_w(rng, D),
            w4=_w(rng, D, D),
            b4=_w(rng, D),
            w5=_w(rng, D, D),
            b5=_w(rng, D),
        ),
        f"float[3,{D}] x",
        f"float[3,{D}] y",
    )


MODELS = {"mlp": _mlp, "conv": _convnet, "fanout": _fanout}


def _data(model, n=6):
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    return [
        {"x": np.random.default_rng(i).standard_normal(shape).astype(np.float32)}
        for i in range(n)
    ]


def _first_layer(name):
    return {"mlp": "n0_Gemm", "conv": "n0_Conv", "fanout": "n0_Gemm"}[name]


# -- running Quark / onnxsim ----------------------------------------------------------


class L(dict):
    """A target ``QLayerConfig`` spelled field by field: ``L(input_tensors="Int8Spec")``."""


def _target(t, lib, kind):
    if isinstance(t, L):
        return lib.QLayerConfig(**{k: getattr(kind, v)() for k, v in t.items()})
    if isinstance(t, dict):
        return {_target(k, lib, kind): v for k, v in t.items()}
    if isinstance(t, list):
        return [_target(c, lib, kind) for c in t]
    act, wt = t
    return lib.QLayerConfig(activation=getattr(kind, act)(), weight=getattr(kind, wt)())


def _quark_amp(model, data, base, target, qkw=None, **amp):
    from onnxruntime.quantization import CalibrationDataReader
    from quark.onnx import AutoMixprecisionConfig, ModelQuantizer, QConfig

    it = iter(data)

    class Reader(CalibrationDataReader):
        def get_next(self):
            return next(it, None)

    onnx.save(model, "float.onnx")
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        cfg = QConfig(
            global_config=_target(base, quark_onnx, quark_spec),
            algo_config=[
                AutoMixprecisionConfig(_target(target, quark_onnx, quark_spec), **amp)
            ],
            **(qkw or {}),
        )
        ModelQuantizer(cfg).quantize_model("float.onnx", "quark.onnx", Reader())
    return onnx.load("quark.onnx")


def _mine_amp(model, data, base, target, qkw=None, **amp):
    cfg = qc.QConfig(
        global_config=_target(base, qc, qc),
        algo_config=[
            qc.AutoMixprecisionConfig(
                target_layer_config=_target(target, qc, qc), **amp
            )
        ],
        **(qkw or {}),
    )
    return qc.ModelQuantizer(cfg).quantize_model(model, calibration_data_reader=data)


# Quark's preset ``extra_options["AutoMixprecision"]`` key -> onnxsim parameter
_PRESET_KEYS = {
    "SubgraphJson": "subgraph_json",
    "SensitivityCacheFile": "sensitivity_cache_file",
    "IncludeLayers": "include_layers",
    "ExcludeLayers": "exclude_layers",
    "MetricThreshold": "metric_threshold",
    "MetricOptimizeObject": "metric_optimize_object",
    "NoInputQDQShared": "no_input_qdq_shared",
    "DualQuantNodes": "dual_quant_nodes",
    "DataSize": "data_size",
}


def _quark_preset(model, data, preset, amp=None):
    from onnxruntime.quantization import CalibrationDataReader
    from quark.onnx import ModelQuantizer, QConfig

    it = iter(data)

    class Reader(CalibrationDataReader):
        def get_next(self):
            return next(it, None)

    onnx.save(model, "float.onnx")
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        cfg = copy.deepcopy(QConfig.get_default_config(preset))
        cfg.global_quant_config.include_cle = False  # (every Quark preset runs CLE)
        opts = cfg.global_quant_config.extra_options
        opts["AutoMixprecision"] = {**opts["AutoMixprecision"], **(amp or {})}
        ModelQuantizer(cfg).quantize_model("float.onnx", "quark.onnx", Reader())
    return onnx.load("quark.onnx")


def _mine_preset(model, data, preset, amp=None):
    cfg = qc.QConfig.get_default_config(preset)
    for k, v in (amp or {}).items():
        cfg.algo_config[0].params[_PRESET_KEYS[k]] = v
    return qc.ModelQuantizer(cfg).quantize_model(model, calibration_data_reader=data)


# -- comparing models -----------------------------------------------------------------


def _ops_lib():
    path = os.environ.get("QUARK_ONNX_OPS_LIB")
    if path:
        return path
    try:
        from quark.onnx.operators.custom_ops import get_library_path

        return get_library_path()
    except Exception:  # pragma: no cover - environment dependent
        return None


def _outputs(model, data):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    lib = _ops_lib()
    if lib:
        so.register_custom_ops_library(lib)
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return [sess.run(None, feed)[0] for feed in data]


def _slots(model):
    """``{(node, slot): (zero point dtype, scales, zero points, codes)}`` of every
    constant or tensor read through a DequantizeLinear, and of every output a
    QuantizeLinear reads. (A scale is compared as its set of values: Quark
    stores an unpromoted per-tensor bias scale as a one-element vector.)"""
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    prod = {o: n for n in model.graph.node for o in n.output}
    cons = {}
    for n in model.graph.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)

    def qparams(node):
        zp = inits[node.input[2]] if len(node.input) > 2 else np.int8(0)
        return (
            str(np.asarray(zp).dtype),
            np.unique(inits[node.input[1]]),
            np.unique(zp),
        )

    out = {}
    for n in model.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        for i, x in enumerate(n.input):
            dq = prod.get(x)
            if dq is not None and dq.op_type == "DequantizeLinear":
                codes = inits.get(dq.input[0])
                out[(n.name, f"in{i}")] = (*qparams(dq), codes)
        for o in n.output:
            for c in cons.get(o, []):
                if c.op_type == "QuantizeLinear":
                    out[(n.name, "out")] = (*qparams(c), None)
    return out


def _assert_same(quark, mine, data):
    a, b = _slots(quark), _slots(mine)
    assert set(a) == set(b)
    for key in sorted(a):
        for what, x, y in zip(
            ("zero point dtype", "scale", "zero point", "codes"), a[key], b[key]
        ):
            if x is None or isinstance(x, str):
                assert x == y, (key, what)
            else:
                np.testing.assert_array_equal(x, y, err_msg=f"{key} {what}")
    for x, y in zip(_outputs(quark, data), _outputs(mine, data)):
        np.testing.assert_array_equal(x, y)


def _results(path):
    return json.loads(path.read_text())["results"]


# -- weight / bias precision mixing -----------------------------------------------------

U8, U16, S16 = (
    ("UInt8Spec", "Int8Spec"),
    ("UInt16Spec", "Int8Spec"),
    ("Int16Spec", "Int8Spec"),
)
_MIX = {
    "act-u16": ("mlp", U8, U16, {}),
    "act-u16-first": ("mlp", U8, U16, {"include_layers": ["n0_Gemm"]}),
    "weights-int16": ("mlp", U8, ("UInt16Spec", "Int16Spec"), {}),
    "weights-int16-second-only": (
        "mlp",
        U8,
        ("UInt16Spec", "Int16Spec"),
        {"include_layers": ["n2_Gemm"]},
    ),
    "weights-only": (
        "mlp",
        U8,
        ("UInt8Spec", "Int16Spec"),
        {"include_layers": ["n2_Gemm"]},
    ),
    "uint8-weights": ("mlp", U8, ("UInt16Spec", "UInt8Spec"), {}),
    "signed-activation": (
        "mlp",
        U8,
        ("Int16Spec", "Int16Spec"),
        {"include_layers": ["n2_Gemm"]},
    ),
    "same-precision": ("mlp", U8, U8, {}),
    "s16-to-s8": ("mlp", ("Int16Spec", "Int16Spec"), ("Int8Spec", "Int8Spec"), {}),
    "conv-int16-weights-first": (
        "conv",
        U8,
        ("UInt16Spec", "Int16Spec"),
        {"include_layers": ["n0_Conv"]},
    ),
    "conv-signed-act-second": (
        "conv",
        U8,
        ("Int16Spec", "Int8Spec"),
        {"include_layers": ["n2_Conv"]},
    ),
    "conv-s16-s8": (
        "conv",
        ("Int16Spec", "Int16Spec"),
        ("Int8Spec", "Int8Spec"),
        {"include_layers": ["n0_Conv"]},
    ),
    "fanout-int16-weights": ("fanout", U8, ("UInt16Spec", "Int16Spec"), {}),
}


@pytest.mark.parametrize("case", sorted(_MIX))
def test_weight_bias_and_activation_mixing_matches_quark_exactly(case):
    name, base, target, kw = _MIX[case]
    model = MODELS[name]()
    data = _data(model)
    quark = _quark_amp(model, data, base, target, data_size=5, **kw)
    mine = _mine_amp(model, data, base, target, data_size=5, **kw)
    _assert_same(quark, mine, data)


def test_mixing_actually_changes_weights_in_these_cases():
    """Guard against the table above passing vacuously."""
    model = _mlp()
    data = _data(model)
    mine = _mine_amp(
        model, data, U8, ("UInt16Spec", "Int16Spec"), include_layers=["n2_Gemm"]
    )
    s = _slots(mine)
    assert s[("n2_Gemm", "in1")][3].dtype == np.int16
    assert s[("n0_Gemm", "in1")][3].dtype == np.int8


# -- the candidate ranking ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["mlp", "conv", "fanout"])
@pytest.mark.parametrize("data_size", [None, 3])
@pytest.mark.parametrize(
    "target", [U16, ("UInt16Spec", "Int16Spec")], ids=["w8", "w16"]
)
def test_sensitivity_scores_and_ranking_match_quark(name, data_size, target, tmp_path):
    model = MODELS[name]()
    data = _data(model)
    kw = {} if data_size is None else {"data_size": data_size}
    theirs, ours = tmp_path / "q.json", tmp_path / "m.json"
    quark = _quark_amp(
        model, data, U8, target, sensitivity_cache_file=str(theirs), **kw
    )
    mine = _mine_amp(model, data, U8, target, sensitivity_cache_file=str(ours), **kw)
    rq, rm = _results(theirs), _results(ours)
    assert [r["name"] for r in rq] == [r["name"] for r in rm]
    np.testing.assert_allclose(
        [r["score"] for r in rm], [r["score"] for r in rq], rtol=1e-6
    )
    _assert_same(quark, mine, data)


def test_default_data_size_scores_a_single_batch_like_quark(tmp_path):
    # the documented "0 = all" is not what Quark does: data_size=0 scores one batch
    model = _mlp()
    data = _data(model)
    cache = {}
    for key, kw in (("default", {}), ("all", {"data_size": len(data)})):
        path = tmp_path / f"{key}.json"
        _mine_amp(model, data, U8, U16, sensitivity_cache_file=str(path), **kw)
        cache[key] = [r["score"] for r in _results(path)]
    assert cache["default"] != cache["all"]
    path = tmp_path / "quark.json"
    _quark_amp(model, data, U8, U16, sensitivity_cache_file=str(path))
    np.testing.assert_allclose(
        [r["score"] for r in _results(path)], cache["default"], rtol=1e-6
    )


# -- no_input_qdq_shared with shared inputs -------------------------------------------------


@pytest.mark.parametrize(
    "target", [U16, ("UInt16Spec", "Int16Spec")], ids=["w8", "w16"]
)
@pytest.mark.parametrize(
    "extra",
    [{}, {"include_layers": ["n0_Gemm", "n4_Gemm", "n7_Gemm"]}],
    ids=["all", "subset"],
)
def test_no_input_qdq_shared_with_shared_inputs_matches_quark(target, extra):
    model = _fanout()
    data = _data(model)
    kw = dict(data_size=5, **extra)
    quark = _quark_amp(model, data, U8, target, no_input_qdq_shared=True, **kw)
    mine = _mine_amp(model, data, U8, target, no_input_qdq_shared=True, **kw)
    _assert_same(quark, mine, data)
    # x feeds n0 and n1, t feeds n4 and n5: those four stay at the base precision
    s = _slots(mine)
    for shared in ("n0_Gemm", "n1_Gemm", "n4_Gemm", "n5_Gemm"):
        assert s[(shared, "in0")][0] == "uint8", shared
    assert s[("n7_Gemm", "in0")][0] == "uint16"


# -- the int16 -> int8 promotion --------------------------------------------------------------

S16_BASE = ("Int16Spec", "Int16Spec")
S16_TARGET = L(input_tensors="Int8Spec", weight="Int8Spec", bias="Int8Spec")


def _write_subgraphs(
    path, first="n0_Gemm", second="n2_Gemm", ends=("n1_Tanh", "n3_Tanh")
):
    path.write_text(
        json.dumps(
            {
                "quantized": False,
                "num_subgraphs": 2,
                "subgraphs": [
                    {"name": "front", "start_nodes": [first], "end_nodes": [ends[0]]},
                    {"name": "back", "start_nodes": [second], "end_nodes": [ends[1]]},
                ],
            }
        )
    )
    return str(path)


@pytest.mark.parametrize(
    "target",
    [S16_TARGET, L(activation="Int8Spec", weight="Int8Spec", bias="Int8Spec")],
    ids=["input_tensors", "activation"],
)
def test_int16_to_int8_with_subgraph_json_and_cache_matches_quark(target, tmp_path):
    model = _mlp()
    data = _data(model)
    sg = _write_subgraphs(tmp_path / "sg.json")
    theirs, ours = tmp_path / "q.json", tmp_path / "m.json"
    kw = dict(qkw=dict(Int32Bias=False), subgraph_json=sg, data_size=5)
    quark = _quark_amp(
        model, data, S16_BASE, target, sensitivity_cache_file=str(theirs), **kw
    )
    mine = _mine_amp(
        model, data, S16_BASE, target, sensitivity_cache_file=str(ours), **kw
    )
    rq, rm = _results(theirs), _results(ours)
    assert [(r["name"], r["candidate_nodes"]) for r in rq] == [
        (r["name"], r["candidate_nodes"]) for r in rm
    ]
    np.testing.assert_allclose(
        [r["score"] for r in rm], [r["score"] for r in rq], rtol=1e-6
    )
    _assert_same(quark, mine, data)
    # pin "back": its layer keeps the int16 weights, bias and activations
    for path, rows in ((theirs, rq), (ours, rm)):
        for r in rows:
            r["enabled"] = r["name"] != "back"
        path.write_text(json.dumps({**json.loads(path.read_text()), "results": rows}))
    quark = _quark_amp(
        model, data, S16_BASE, target, sensitivity_cache_file=str(theirs), **kw
    )
    mine = _mine_amp(
        model, data, S16_BASE, target, sensitivity_cache_file=str(ours), **kw
    )
    _assert_same(quark, mine, data)
    s = _slots(mine)
    assert s[("n2_Gemm", "in1")][3].dtype == np.int16
    assert s[("n2_Gemm", "in2")][3].dtype == np.int16
    assert s[("n0_Gemm", "in1")][3].dtype == np.int8


@pytest.mark.parametrize(
    "amp",
    [{}, {"IncludeLayers": ["n2_Gemm"]}, {"IncludeLayers": ["n0_Gemm", "n4_Gemm"]}],
    ids=["all", "second", "outer"],
)
def test_s16s16_mixed_s8s8_preset_is_bit_identical_to_quark(amp):
    model = _mlp()
    data = _data(model)
    quark = _quark_preset(model, data, "S16S16_MIXED_S8S8", amp)
    mine = _mine_preset(model, data, "S16S16_MIXED_S8S8", amp)
    _assert_same(quark, mine, data)


# -- block formats ------------------------------------------------------------------------------

_needs_lib = pytest.mark.skipif(
    not _ops_lib() if quark_onnx is not None else True,
    reason="Quark's ONNX custom-op library is not available",
)
_BLOCK = ["BF16_MIXED_BFP16", "BF16_MIXED_MXINT8"]


def _names(model):
    return sorted(n.name for n in model.graph.node)


def _same_block(quark, mine, data):
    assert _names(quark) == _names(mine)
    for x, y in zip(_outputs(quark, data), _outputs(mine, data)):
        np.testing.assert_array_equal(x, y)


@_needs_lib
@pytest.mark.parametrize("preset", _BLOCK)
@pytest.mark.parametrize("name", ["mlp", "conv", "fanout"])
def test_block_cache_scores_ranking_and_pins_match_quark(preset, name, tmp_path):
    model = MODELS[name]()
    data = _data(model)
    theirs, ours = tmp_path / "q.json", tmp_path / "m.json"
    quark = _quark_preset(model, data, preset, {"SensitivityCacheFile": str(theirs)})
    mine = _mine_preset(model, data, preset, {"SensitivityCacheFile": str(ours)})
    rq, rm = _results(theirs), _results(ours)
    assert [r["name"] for r in rq] == [r["name"] for r in rm]
    np.testing.assert_allclose(
        [r["score"] for r in rm], [r["score"] for r in rq], rtol=1e-5
    )
    _same_block(quark, mine, data)
    # pin the second-ranked candidate in both caches
    pin = rq[1]["name"]
    for path, rows in ((theirs, rq), (ours, rm)):
        for r in rows:
            r["enabled"] = r["name"] != pin
        path.write_text(json.dumps({**json.loads(path.read_text()), "results": rows}))
    quark = _quark_preset(model, data, preset, {"SensitivityCacheFile": str(theirs)})
    mine = _mine_preset(model, data, preset, {"SensitivityCacheFile": str(ours)})
    _same_block(quark, mine, data)
    plain = _mine_preset(model, data, preset)
    assert _names(mine) != _names(plain)


@_needs_lib
@pytest.mark.parametrize("preset", _BLOCK)
def test_block_subgraph_json_matches_quark(preset, tmp_path):
    model = _mlp()
    data = _data(model)
    sg = _write_subgraphs(tmp_path / "sg.json")
    quark = _quark_preset(model, data, preset, {"SubgraphJson": sg})
    mine = _mine_preset(model, data, preset, {"SubgraphJson": sg})
    _same_block(quark, mine, data)
    # with the threshold disabled every candidate moves anyway
    _same_block(quark, _quark_preset(model, data, preset), data)
    theirs, ours = tmp_path / "q.json", tmp_path / "m.json"
    _quark_preset(
        model, data, preset, {"SubgraphJson": sg, "SensitivityCacheFile": str(theirs)}
    )
    _mine_preset(
        model, data, preset, {"SubgraphJson": sg, "SensitivityCacheFile": str(ours)}
    )
    assert [(r["name"], r["candidate_nodes"]) for r in _results(theirs)] == [
        (r["name"], r["candidate_nodes"]) for r in _results(ours)
    ]
    # a subgraph pinned through the cache keeps its layer bfloat16
    rows = _results(ours)
    for path in (theirs, ours):
        doc = json.loads(path.read_text())
        for r in doc["results"]:
            r["enabled"] = r["name"] != "back"
        path.write_text(json.dumps(doc))
    amp = {"SubgraphJson": sg}
    _same_block(
        _quark_preset(
            model, data, preset, {**amp, "SensitivityCacheFile": str(theirs)}
        ),
        _mine_preset(model, data, preset, {**amp, "SensitivityCacheFile": str(ours)}),
        data,
    )
    assert rows


@_needs_lib
@pytest.mark.parametrize("preset", _BLOCK)
@pytest.mark.parametrize(
    "amp",
    [
        {"MetricThreshold": None},
        {"MetricThreshold": 0.05},
        {"MetricThreshold": 0.15},
        {"MetricThreshold": 0.03, "MetricOptimizeObject": "quality"},
        {"NoInputQDQShared": True},
        {"DualQuantNodes": False},
        {"IncludeLayers": ["n0_Gemm"]},
        {"ExcludeLayers": ["n2_Gemm"]},
    ],
    ids=[
        "analysis-only",
        "speed-0.05",
        "speed-0.15",
        "quality",
        "no-input-qdq-shared",
        "no-dual",
        "include",
        "exclude",
    ],
)
def test_block_amp_options_match_quark(preset, amp):
    model = _mlp()
    data = _data(model)
    _same_block(
        _quark_preset(model, data, preset, amp),
        _mine_preset(model, data, preset, amp),
        data,
    )


@_needs_lib
@pytest.mark.parametrize("preset", _BLOCK)
def test_block_no_input_qdq_shared_with_shared_inputs_matches_quark(preset):
    model = _fanout()
    data = _data(model)
    amp = {"NoInputQDQShared": True}
    quark = _quark_preset(model, data, preset, amp)
    mine = _mine_preset(model, data, preset, amp)
    _same_block(quark, mine, data)
    assert _names(mine) != _names(_mine_preset(model, data, preset))


@_needs_lib
@pytest.mark.parametrize("preset", _BLOCK)
def test_block_partial_promotion_of_a_shared_tensor_matches_quark(preset):
    # the conv net reads r1 twice: promoting one reader leaves the other on its
    # own (dedicated) bfloat16 dequantizer
    model = _convnet()
    data = _data(model)
    amp = {"IncludeLayers": ["n2_Conv"]}
    _same_block(
        _quark_preset(model, data, preset, amp),
        _mine_preset(model, data, preset, amp),
        data,
    )
