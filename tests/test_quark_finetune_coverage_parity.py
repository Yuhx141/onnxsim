"""Parity of the *coverage* of :mod:`onnxsim.quark_finetune` with the real AMD
Quark ONNX package (skipped unless ``quark.onnx`` and ``torch`` are importable):
which layers and options Quark's ``FastFinetune`` trains, which it skips (its
driver logs a failed layer and goes on), where it raises -- and, wherever it
trains, that the integer codes come out the same.

``tests/test_quark_finetune_parity.py`` explains the method: Quark's own torch
random stream (module construction, ``randperm``, ``rand_like``) is replayed
into :func:`onnxsim.quark_finetune.finetune`, after which the codes of an
AdaRound run are identical. Probed with ``amd-quark`` 0.13:

* trained, bit for bit: ``MatMul`` on 4-D activations, grouped ``ConvTranspose``, 3-D ``Conv`` /
  ``ConvTranspose`` (also with Quark's scrambled asymmetric 3-D pads), ``PRelu``
  blocks (slope 0.25 whatever the node says), ``Clip`` fallbacks, bias-less
  ``Gemm`` (a random ``torch.nn.Linear`` bias), ``Gemm`` with ``transA`` when
  its shapes line up, int16 / uint8-asymmetric weights, and ``MemOptLevel=2``
  (a ``DataLoader`` epoch loop);
* skipped by Quark and here: ``auto_pad``, ``ConvTranspose`` with
  ``output_padding`` (even zeros) / ``output_shape`` / asymmetric pads, 3-D
  pads whose scrambled version has the wrong shape, ``Gemm`` ``transA`` with any
  other shapes, every layer with ``NumWorkers=0`` and ``MemOptLevel=2``, every
  layer with ``DynamicBatch`` on batches of more than one sample;
* aborted by Quark and here: ``SelectMaxMemLayer`` next to an unconvertible
  layer or an unusable ``DynamicBatch``.

AdaQuant is chaotic (see the other file) and compared on short runs only; on
16-bit weights the float32 rounding of the trained weights decides single codes
(~0.4 % of them per layer at each rounding boundary), so it is compared
statistically there.
"""

import copy
import re
import warnings

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

warnings.filterwarnings("ignore")

import test_quark_finetune_parity as P  # noqa: E402  (also sets up the Quark imports)

torch = P.torch
pytestmark = pytest.mark.skipif(
    torch is None, reason="AMD Quark (amd-quark) and torch are not installed"
)

from onnxsim import quark_compat as qc  # noqa: E402
from onnxsim import quark_finetune as qf  # noqa: E402


@pytest.fixture(autouse=True)
def _scratch_dir(tmp_path, monkeypatch):
    """Quark writes scratch files (sym_shape_infer_temp.onnx, quantized_info.csv,
    ...) into the current directory."""
    monkeypatch.chdir(tmp_path)


# -- building blocks -----------------------------------------------------------------------


def _w(rng, name, *shape, scale=0.3):
    return numpy_helper.from_array(
        (rng.standard_normal(shape) * scale).astype(np.float32), name
    )


def _const(name, v):
    return numpy_helper.from_array(np.array(v, np.float32), name)


def _model(body, inits, shape, opset=17, io=None, n_batches=4, rng_seed=1):
    """``(model, calibration data)`` from the text form of the graph body."""
    dims = ",".join("N" if i == 0 else str(d) for i, d in enumerate(shape))
    io = io or f"float[{dims}] x) => (float y"
    model = parser.parse_model(
        f'<ir_version: 10, opset_import: ["": {opset}]> g ({io}) {{ {body} }}'
    )
    model.graph.initializer.extend(inits)
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}"  # Quark keys its sub-models by node name
    r = np.random.default_rng(rng_seed)
    data = [
        {"x": r.standard_normal(shape).astype(np.float32)} for _ in range(n_batches)
    ]
    return model, data


def _codes_all(model):
    """Every weight / bias code tensor (int8, uint8, int16, int32)."""
    return {
        t.name: numpy_helper.to_array(t).astype(np.int64)
        for t in model.graph.initializer
        if t.data_type
        in (
            onnx.TensorProto.INT8,
            onnx.TensorProto.UINT8,
            onnx.TensorProto.INT16,
            onnx.TensorProto.INT32,
        )
        and numpy_helper.to_array(t).ndim > 0
    }


def _mismatch(a, b):
    ca, cb = _codes_all(a), _codes_all(b)
    return {k: float(np.mean(ca[k] != cb[k])) for k in ca}


def _changed(a, b):
    return {k: v for k, v in _mismatch(a, b).items() if v > 0}


def _opts(ff):
    o = P._options(ff)
    o.mem_opt_level = ff.get("MemOptLevel", 1)
    o.num_workers = ff.get("NumWorkers", 1)
    o.dynamic_batch = ff.get("DynamicBatch", False)
    return o


def _replay(model, data, q, ff):
    """The ``finetune`` keyword arguments that replay Quark's torch stream:
    ``randperm`` / ``rand_like``, the construction of every layer's module (also
    those Quark fails to train), the ``DataLoader`` seeds of ``MemOptLevel=2``
    and the random bias of a bias-less ``Gemm``."""
    with P._Quiet():
        sg = P.Subgraph(
            copy.deepcopy(model),
            copy.deepcopy(q),
            False,
            P._Reader(data),
            {"FastFinetune": ff},
        )
        P.setup_seed(ff["FixedSeed"])
    names = list(sg.subgraph_qmodel.keys())
    done = [0]
    state = {"first": True, "persistent": False}
    bs = ff.get("BatchSize", 1)
    persistent = (bs if 1 <= bs <= len(data) else 1) > 1

    def loader_perm(n):
        # DataLoader: the iterator draws a base seed (a persistent-worker one only
        # in its first epoch) and the shuffling sampler one seed per epoch
        if state["first"] or not state["persistent"]:
            torch.empty((), dtype=torch.int64).random_()
        state["first"] = False
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        g = torch.Generator()
        g.manual_seed(seed)
        return torch.randperm(n, generator=g).numpy()

    def hook(_layer, name):
        state["first"], state["persistent"] = True, persistent
        j = names.index(name)
        module = None
        while done[0] <= j:
            i = done[0]
            done[0] += 1
            with P._Quiet():
                bias = sg.f_bias_list[i]
                try:
                    module = P.convert_onnx_to_torch(
                        sg.subgraph_qmodel_list[i],
                        np.array(sg.f_weight_list[i]),
                        None if bias is None else np.array(bias).reshape(-1),
                    )
                except Exception:  # Quark cannot convert it (and draws nothing)
                    module = None
        if module is not None and sg.f_bias_list[j] is None:
            lin = getattr(module._module, "bias", None)
            if lin is not None:
                return {"phantom_bias": lin.detach().numpy().astype(np.float64)}
        return None

    return dict(
        perm_fn=(
            loader_perm
            if ff.get("MemOptLevel", 1) == 2
            else (lambda n: torch.randperm(n).numpy())
        ),
        rand_fn=lambda shape: torch.rand(shape).numpy(),
        block_hook=hook,
    )


def _both(model, data, ff, q=None, preset="A8W8", replay=True):
    """Quark's fine-tuned model and ours (same start, same random stream)."""
    if q is None:
        q = P._quark_quantize(model, data, preset)[0]
    with P._Quiet() as buf:
        # (DynamicBatch rewrites the batch axis of the models it is handed)
        quark_out = P.fast_finetune(
            copy.deepcopy(model),
            copy.deepcopy(q),
            False,
            P._Reader(data),
            {"FastFinetune": ff},
        )
    kwargs = _replay(model, data, q, ff) if replay else {}
    trace = []
    mine, reports = qf.finetune(model, q, data, _opts(ff), trace=trace, **kwargs)
    return quark_out, mine, q, reports, trace, buf.getvalue()


def _check_equal(model, data, ff, q=None, preset="A8W8", tol=0.0, trains=True):
    """Same codes as Quark's, which (``trains``) changed something -- or, for a
    layer Quark skips, left the model as it was."""
    quark_out, mine, q, reports, _, log = _both(model, data, ff, q, preset)
    moved = _changed(q, quark_out)
    assert bool(moved) == trains, (moved, log[-600:])
    diff = _mismatch(quark_out, mine)
    assert max(diff.values()) <= tol, diff
    if not trains:
        assert reports == [] and not _changed(q, mine)
    return quark_out, mine, q, reports


# -- layers Quark skips / trains: AdaRound, exact given the same random stream ----------------

_R = np.random.default_rng(0)


def _gemm_nobias(transb=0, extra=""):
    shape = (5, 6) if transb else (6, 5)
    return _model(
        f"y = Gemm<transB={transb}{extra}>(x, w1)", [_w(_R, "w1", *shape)], (8, 6)
    )


def _matmul_4d():
    """MatMul on a 4-D activation (Quark's ``torch.matmul`` has no rank limit)."""
    return _model(
        "h = MatMul(x, w1)\n t = Relu(h)\n y = MatMul(t, w2)",
        [_w(_R, "w1", 8, 6), _w(_R, "w2", 6, 4)],
        (4, 2, 5, 8),
    )


def _prelu(per_channel=False, tail=False):
    slope = (
        np.array([0.1, 0.2, 0.3, 0.4], np.float32).reshape(4, 1, 1)
        if per_channel
        else np.array([0.1], np.float32)
    )
    body = "c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)\n p = PRelu(c1, sl)\n"
    inits = [
        _w(_R, "w1", 4, 3, 3, 3),
        _w(_R, "b1", 4, scale=0.1),
        numpy_helper.from_array(slope, "sl"),
    ]
    if tail:
        body += "y = Conv<pads=[1,1,1,1]>(p, w2, b2)"
        inits += [_w(_R, "w2", 4, 4, 3, 3), _w(_R, "b2", 4, scale=0.1)]
    else:
        body += "y = Identity(p)"
    return _model(body, inits, (4, 3, 6, 6))


def _convT(attrs, wshape, cin=4, hw=5, groups=1):
    return _model(
        f"c1 = ConvTranspose<{attrs}>(x, w1, b1)\n y = Relu(c1)",
        [_w(_R, "w1", *wshape), _w(_R, "b1", wshape[1] * groups, scale=0.1)],
        (4, cin, hw, hw),
    )


def _conv3d(pads, wshape=(3, 2, 3, 3, 3), shape=(4, 2, 5, 6, 7), attrs=""):
    return _model(
        f"c1 = Conv<pads={pads}{attrs}>(x, w1, b1)\n y = Relu(c1)",
        [_w(_R, "w1", *wshape), _w(_R, "b1", wshape[0], scale=0.1)],
        shape,
    )


def _two_convs(first_attrs):
    return _model(
        f"c1 = Conv<{first_attrs}>(x, w1, b1)\n r = Relu(c1)\n"
        "y = Conv<pads=[1,1,1,1]>(r, w2, b2)",
        [
            _w(_R, "w1", 4, 3, 3, 3),
            _w(_R, "b1", 4, scale=0.1),
            _w(_R, "w2", 4, 4, 3, 3),
            _w(_R, "b2", 4, scale=0.1),
        ],
        (4, 3, 8, 8),
    )


def _cases():
    # (id, builder, FastFinetune overrides, tolerance, trains)
    c = [
        ("matmul-4d", lambda: _matmul_4d(), {}, 0.0, True),
        (
            "matmul-4d-outqdq",
            lambda: _matmul_4d(),
            dict(OutputQDQ=True, BatchSize=3),
            0.0,
            True,
        ),
        ("gemm-nobias", lambda: _gemm_nobias(), {}, 0.0, True),
        (
            "gemm-nobias-transB-alpha-beta",
            lambda: _gemm_nobias(1, ", alpha=0.5, beta=2.0"),
            {},
            0.0,
            True,
        ),
        (
            "gemm-nobias-outqdq-drop",
            lambda: _gemm_nobias(),
            dict(OutputQDQ=True, BatchSize=4, DropRatio=0.5),
            0.0,
            True,
        ),
        ("prelu-scalar", lambda: _prelu(), {}, 0.0, True),
        ("prelu-per-channel-then-conv", lambda: _prelu(True, True), {}, 0.0, True),
        (
            "prelu-outqdq",
            lambda: _prelu(True, True),
            dict(OutputQDQ=True, BatchSize=3),
            0.0,
            True,
        ),
        (
            "convT-group2",
            lambda: _convT("group=2, strides=[2,2]", (4, 3, 2, 2), groups=2),
            {},
            0.0,
            True,
        ),
        (
            "convT-group4-dil-pads",
            lambda: _convT(
                "group=4, strides=[2,1], dilations=[1,2], pads=[1,1,1,1]",
                (4, 2, 3, 3),
                groups=4,
            ),
            {},
            0.0,
            True,
        ),
        (
            "convT-group2-outqdq-drop",
            lambda: _convT(
                "group=2, strides=[2,2], pads=[1,1,1,1]", (4, 3, 3, 3), groups=2
            ),
            dict(OutputQDQ=True, DropRatio=0.5, BatchSize=4),
            0.0,
            True,
        ),
        (
            "convT-output_padding",
            lambda: _convT(
                "strides=[2,2], output_padding=[1,1], kernel_shape=[3,3]", (4, 2, 3, 3)
            ),
            {},
            0.0,
            False,
        ),
        (
            "convT-output_padding-zeros",
            lambda: _convT(
                "strides=[2,2], output_padding=[0,0], kernel_shape=[3,3]", (4, 2, 3, 3)
            ),
            {},
            0.0,
            False,
        ),
        (
            "convT-output_shape",
            lambda: _convT(
                "strides=[2,2], output_shape=[10,10], kernel_shape=[3,3]", (4, 2, 3, 3)
            ),
            {},
            0.0,
            False,
        ),
        (
            "convT-asym-pads",
            lambda: _convT(
                "strides=[2,2], pads=[0,1,1,0], kernel_shape=[3,3]", (4, 2, 3, 3)
            ),
            {},
            0.0,
            False,
        ),
        (
            "auto_pad-same-then-conv",
            lambda: _two_convs('auto_pad="SAME_UPPER", kernel_shape=[3,3]'),
            {},
            0.0,
            True,
        ),
        (
            "auto_pad-valid-last",
            lambda: _model(
                'c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)\n r = Relu(c1)\n y = Conv<auto_pad="VALID">(r, w2, b2)',
                [
                    _w(_R, "w1", 4, 3, 3, 3),
                    _w(_R, "b1", 4, scale=0.1),
                    _w(_R, "w2", 4, 4, 3, 3),
                    _w(_R, "b2", 4, scale=0.1),
                ],
                (4, 3, 8, 8),
            ),
            {},
            0.0,
            True,
        ),
        (
            "auto_pad-notset",
            lambda: _two_convs('auto_pad="NOTSET", pads=[1,1,1,1]'),
            {},
            0.0,
            True,
        ),
    ]
    # 3-D convolutions: symmetric pads, strides / dilations / groups
    for pads in ("[1,1,1,1,1,1]", "[0,0,0,0,0,0]", "[1,2,0,1,2,0]", "[1,1,1,2,2,2]"):
        c.append((f"conv3d-pads{pads}", lambda p=pads: _conv3d(p), {}, 0.0, True))
    c += [
        (
            "conv3d-outqdq-drop",
            lambda: _conv3d("[1,1,1,1,1,1]"),
            dict(OutputQDQ=True, DropRatio=0.5, BatchSize=4),
            0.0,
            True,
        ),
        (
            "conv3d-group-stride-dil",
            lambda: _conv3d(
                "[1,1,1,1,1,1]",
                (4, 2, 3, 3, 3),
                (4, 4, 7, 6, 7),
                ", group=2, strides=[2,1,1], dilations=[1,2,1]",
            ),
            {},
            0.0,
            True,
        ),
        (
            "convT3d",
            lambda: _model(
                "c1 = ConvTranspose<pads=[1,1,1,1,1,1], strides=[2,1,1]>(x, w1, b1)\n y = Relu(c1)",
                [_w(_R, "w1", 2, 3, 3, 3, 3), _w(_R, "b1", 3, scale=0.1)],
                (4, 2, 3, 3, 3),
            ),
            {},
            0.0,
            True,
        ),
        (
            "convT1d-group2",
            lambda: _model(
                "c1 = ConvTranspose<group=2, strides=[2]>(x, w1, b1)\n y = Relu(c1)",
                [_w(_R, "w1", 4, 3, 3), _w(_R, "b1", 6, scale=0.1)],
                (4, 4, 6),
            ),
            {},
            0.0,
            True,
        ),
        # Quark's asymmetric 3-D pads are scrambled across the axes; where the
        # scrambled pads still give the right shape it trains with them
        ("conv3d-scrambled-pads", lambda: _conv3d("[1,0,0,0,1,1]"), {}, 0.0, True),
        # ... and where they do not, its first forward fails and the layer is skipped
        ("conv3d-asym-wrong-shape", lambda: _conv3d("[1,0,2,2,1,0]"), {}, 0.0, False),
        (
            "conv3d-asym-collapsed-list",
            lambda: _conv3d("[0,1,0,0,2,0]"),
            {},
            0.0,
            False,
        ),
        (
            "conv3d-asym-cubic",
            lambda: _conv3d("[1,0,0,0,1,0]", shape=(4, 2, 5, 5, 5)),
            {},
            0.0,
            False,
        ),
    ]
    return c


_CASES = _cases()


@pytest.mark.parametrize(
    "build, extra, tol, trains",
    [pytest.param(b, e, t, tr, id=i) for i, b, e, t, tr in _CASES],
)
def test_adaround_codes_equal_quarks_where_it_trains_and_nothing_moves_where_it_skips(
    build, extra, tol, trains
):
    model, data = build()
    _check_equal(
        model,
        data,
        P._ff("adaround", NumIterations=40, **extra),
        tol=tol,
        trains=trains,
    )


_ADAQUANT = [
    ("convT-group2", lambda: _convT("group=2, strides=[2,2], pads=[1,1,1,1]", (4, 3, 3, 3), groups=2)),
    ("conv3d", lambda: _conv3d("[1,1,1,1,1,1]")),
    ("prelu-gemm-nobias", lambda: _model("c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)\n p = PRelu(c1, sl)\n f = Flatten(p)\n y = Gemm<transB=1>(f, w3)", [_w(_R, "w1", 4, 3, 3, 3), _w(_R, "b1", 4, scale=0.1), _w(_R, "w3", 5, 144, scale=0.2), numpy_helper.from_array(np.array([0.1, 0.2, 0.3, 0.4], np.float32).reshape(4, 1, 1), "sl")], (4, 3, 6, 6))),
]  # fmt: skip


@pytest.mark.parametrize("build", [pytest.param(b, id=i) for i, b in _ADAQUANT])
@pytest.mark.parametrize("output_qdq", [False, True])
def test_adaquant_codes_equal_quarks_on_the_new_layers_for_short_runs(
    build, output_qdq
):
    model, data = build()
    ff = P._ff(
        "adaquant", LearningRate=1e-3, NumIterations=10, BatchSize=4, UpdateBias=True,
        OutputQDQ=output_qdq,
    )  # fmt: skip
    _check_equal(model, data, ff)


# -- Gemm with transA: trained only when Quark's matmul fits --------------------------------


def _trans_a(transb, bias, k=3, alpha=1.5, n_batches=1):
    rng = np.random.default_rng(3)
    wshape = (k, 5) if transb == 0 else (5, k)
    inits = [_w(rng, "w1", *wshape)] + ([_w(rng, "b1", 5, scale=0.1)] if bias else [])
    model, _ = _model(
        f"y = Gemm<transA=1, transB={transb}, alpha={alpha}>(x, w1{', b1' if bias else ''})",
        inits,
        (k, 3),
        io=f"float[{k},3] x) => (float y",
    )
    data = [
        {"x": np.random.default_rng(5 + i).standard_normal((k, 3)).astype(np.float32)}
        for i in range(n_batches)
    ]
    return model, data


@pytest.mark.parametrize("transb, bias", [(0, True), (1, True), (0, False)])
def test_gemm_trans_a_trains_when_one_batch_and_k_m_and_batch_size_agree(transb, bias):
    model, data = _trans_a(transb, bias)
    ff = P._ff("adaround", NumIterations=40, BatchSize=3)
    _check_equal(model, data, ff)
    ff = P._ff(
        "adaquant", NumIterations=10, BatchSize=3, LearningRate=1e-3, UpdateBias=bias
    )
    _check_equal(model, data, ff)


@pytest.mark.parametrize(
    "bs, n_batches", [(2, 1), (3, 2), (1, 1)], ids=["bs2", "two-batches", "bs1"]
)
def test_gemm_trans_a_is_skipped_by_both_when_the_matmul_does_not_fit(bs, n_batches):
    model, data = _trans_a(0, True, n_batches=n_batches)
    _check_equal(
        model,
        data,
        P._ff("adaround", NumIterations=20, BatchSize=bs),
        trains=False,
    )


# -- Clip / activation fallbacks (Quark's own graphs, edited) ------------------------------


def _clip_pair():
    model, data = _model(
        "c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)\n y = Clip(c1, mn, mx)",
        [
            _w(_R, "w1", 4, 3, 3, 3, scale=1.0),
            _w(_R, "b1", 4, scale=0.1),
            _const("mn", -0.5),
            _const("mx", 1.0),
        ],
        (4, 3, 6, 6),
        opset=13,
    )
    q = P._quark_quantize(model, data)[0]
    # Quark keeps a Q/DQ between the Conv and the Clip; drop it, as its own graphs
    # do for the activations it fuses, so that the Clip is part of the block
    nodes = list(q.graph.node)
    conv = next(n for n in nodes if n.op_type == "Conv")
    clip = next(n for n in nodes if n.op_type == "Clip")
    qn = next(
        n
        for n in nodes
        if n.op_type == "QuantizeLinear" and n.input[0] == conv.output[0]
    )
    dq = next(
        n
        for n in nodes
        if n.op_type == "DequantizeLinear" and n.input[0] == qn.output[0]
    )
    clip.input[0] = conv.output[0]
    q.graph.node.remove(qn)
    q.graph.node.remove(dq)
    return model, data, q


def _edit_clip(q, how):
    q = copy.deepcopy(q)
    clip = next(n for n in q.graph.node if n.op_type == "Clip")
    if how == "min-only":
        del clip.input[2]
    elif how == "empty-min":
        clip.input[1] = ""
    elif how == "bare":
        del clip.input[2], clip.input[1]
    return q


@pytest.mark.parametrize("how", ["bounds", "min-only", "empty-min", "bare"])
@pytest.mark.parametrize("output_qdq", [False, True])
def test_clip_with_bounds_that_are_not_both_initializers_is_trained_as_quark_does(
    how, output_qdq
):
    """torch.clamp for two initializer bounds, ReLU6 for any other input form,
    the identity for a bare Clip (its quantized-graph form is edited here; the
    float model keeps its real bounds as the target)."""
    model, data, q = _clip_pair()
    q = _edit_clip(q, how)
    ff = P._ff("adaround", NumIterations=40, OutputQDQ=output_qdq)
    # the identity with an output Q/DQ lands one code off in 1 of 108 on this model
    tol = 0.03 if (how == "bare" and output_qdq) else 0.0
    _check_equal(model, data, ff, q=q, tol=tol)


# -- bias-less Gemm: the random Linear bias --------------------------------------------------


def test_bias_less_gemm_adds_torchs_random_bias_and_it_changes_the_result():
    model, data = _gemm_nobias()
    ff = P._ff("adaround", NumIterations=40)
    quark_out, mine, q, reports, _, _ = _both(model, data, ff)
    assert max(_mismatch(quark_out, mine).values()) == 0.0
    # with a bias numpy draws instead of torch (no replay of the module) the
    # codes differ: the bias is part of what Quark trains against
    _, other, *_ = _both(model, data, ff, q=q, replay=False)
    assert _changed(mine, other)


# -- int16 / uint8 / asymmetric weights ------------------------------------------------------


def _no_relu_cnn():
    r = np.random.default_rng(0)
    return _model(
        "c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)\n"
        "c2 = Conv<pads=[1,1,1,1], strides=[2,2]>(c1, w2, b2)\n"
        "f = Flatten(c2)\n y = Gemm<transB=1>(f, w3, b3)",
        [
            _w(r, "w1", 6, 3, 3, 3),
            _w(r, "b1", 6, scale=0.1),
            _w(r, "w2", 8, 6, 3, 3),
            _w(r, "b2", 8, scale=0.1),
            _w(r, "w3", 4, 128, scale=0.2),
            _w(r, "b3", 4, scale=0.1),
        ],
        (4, 3, 8, 8),
    )


@pytest.mark.parametrize("preset", ["INT16_CNN_DEFAULT", "A16W8", "U8U8_AAWA"])
@pytest.mark.parametrize(
    "extra, tol_u8",
    [({}, 0.0), (dict(OutputQDQ=True, BatchSize=3), 0.06)],
    ids=["plain", "outqdq"],
)
def test_adaround_codes_equal_quarks_on_16_bit_and_asymmetric_uint8_weights(
    preset, extra, tol_u8
):
    model, data = _no_relu_cnn()
    ff = P._ff("adaround", NumIterations=40, **extra)
    quark_out, mine, q, *_ = _both(model, data, ff, preset=preset)
    assert _changed(q, quark_out)  # Quark really trained something
    kinds = {t.name: t.data_type for t in quark_out.graph.initializer}
    wnames = [k for k, v in _codes_all(quark_out).items() if k.startswith("w")]
    expect = {
        "INT16_CNN_DEFAULT": onnx.TensorProto.INT16,
        "A16W8": onnx.TensorProto.INT8,
        "U8U8_AAWA": onnx.TensorProto.UINT8,
    }[preset]
    assert all(kinds[k] == expect for k in wnames)
    tol = tol_u8 if preset == "U8U8_AAWA" else 0.0
    assert max(_mismatch(quark_out, mine).values()) <= tol


def test_adaquant_on_16_bit_weights_tracks_quarks_in_the_first_layer_and_statistically():
    """Float32 vs float64 decides single 16-bit codes at every rounding
    boundary, and later layers see the earlier ones' tiny differences, so only
    the first layer is compared code by code (a code or two apart at most) and
    the rest through the quantized model's output error."""
    model, data = _no_relu_cnn()
    ff = P._ff(
        "adaquant", LearningRate=1e-5, NumIterations=10, BatchSize=4, UpdateBias=True
    )
    quark_out, mine, q, *_ = _both(model, data, ff, preset="INT16_CNN_DEFAULT")
    cq, co, cm = _codes_all(q), _codes_all(quark_out), _codes_all(mine)
    assert np.mean(co["w1_quantized"] != cq["w1_quantized"]) > 0.1  # it trained
    assert np.mean(co["w1_quantized"] != cm["w1_quantized"]) < 0.05
    assert np.abs(co["w1_quantized"] - cm["w1_quantized"]).max() <= 1
    x = np.random.default_rng(9).standard_normal((64, 3, 8, 8)).astype(np.float32)
    e_q, e_m = P._e2e(model, quark_out, x), P._e2e(model, mine, x)
    assert e_m == pytest.approx(e_q, rel=0.05)


def test_int16_weights_run_end_to_end_like_quarks_accurate_preset():
    model, data = P._build("A")
    cfg = qc.QConfig.get_default_config("INT16_CNN_ACCURATE")
    cfg.algo_config[0].params.update(num_iterations=60, batch_size=2, early_stop=False)
    quantizer = qc.ModelQuantizer(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mine = quantizer.quantize_model(model, calibration_data_reader=P._Reader(data))
    assert quantizer.last_weight_rounding["adaround"]
    quark = P._quark_quantize(
        model,
        data,
        "INT16_CNN_ACCURATE",
        NumIterations=60,
        BatchSize=2,
        EarlyStop=False,
    )[0]
    x = np.random.default_rng(9).standard_normal((64, 3, 8, 8)).astype(np.float32)
    assert P._e2e(model, mine, x) == pytest.approx(P._e2e(model, quark, x), rel=0.02)


# -- MemOptLevel=2 (Quark's DataLoader loop) ------------------------------------------------


def _single_sample_cnn():
    model, data = P._build("A")
    data1 = [{"x": d["x"][i : i + 1]} for d in data for i in range(4)]
    return model, data1


_LEVEL2 = [
    ("default", {}),
    ("bs1", dict(BatchSize=1)),
    ("bs4-50it", dict(BatchSize=4, NumIterations=50)),
    ("bs8", dict(BatchSize=8, NumIterations=30)),
    ("drop-bs3", dict(DropRatio=0.5, BatchSize=3)),
    ("outqdq", dict(OutputQDQ=True)),
    ("early-stop", dict(EarlyStop=True, NumIterations=120, BatchSize=4)),
    ("no-steps-bs16", dict(BatchSize=16)),
    ("lr-adjust-is-ignored", dict(LRAdjust=(0.0, 0.5), BatchSize=2)),
    ("parallel-is-ignored", dict(Parallel=True, BatchSize=2)),
    ("workers2", dict(NumWorkers=2, NumIterations=20, BatchSize=4)),
]


@pytest.mark.parametrize("extra", [pytest.param(e, id=i) for i, e in _LEVEL2])
def test_mem_opt_level_2_adaround_codes_equal_quarks(extra):
    model, data = _single_sample_cnn()
    ff = P._ff("adaround", MemOptLevel=2, **extra)
    quark_out, mine, q, reports, trace, _ = _both(model, data, ff)
    assert _mismatch(quark_out, mine) == {k: 0.0 for k in _codes_all(q)}
    if extra.get("BatchSize") != 16:
        assert _changed(q, quark_out)
    else:  # one step per epoch: the loop never trains
        assert not _changed(q, quark_out) and [len(t) for t in trace] == [0, 0, 0]


def test_mem_opt_level_2_adaquant_codes_equal_quarks():
    model, data = _single_sample_cnn()
    ff = P._ff(
        "adaquant", MemOptLevel=2, LearningRate=1e-3, NumIterations=20, BatchSize=4,
        UpdateBias=True,
    )  # fmt: skip
    quark_out, mine, q, *_ = _both(model, data, ff)
    assert _changed(q, quark_out) and max(_mismatch(quark_out, mine).values()) == 0.0


def test_mem_opt_level_2_iteration_counts_and_early_stop_match_quarks_log():
    model, data = _single_sample_cnn()
    for extra in (
        dict(BatchSize=4, NumIterations=50),
        dict(BatchSize=3, NumIterations=40),
    ):
        ff = P._ff("adaround", MemOptLevel=2, LogPeriod=1, **extra)
        _, _, _, _, trace, log = _both(model, data, ff)
        logged = [int(i) for i in re.findall(r"adaround iterations=(\d+)", log)]
        assert logged  # every iteration logged (LogPeriod=1)
        assert sum(len(t) for t in trace) == len(logged)
    ff = P._ff(
        "adaround", MemOptLevel=2, EarlyStop=True, NumIterations=120, BatchSize=4
    )
    _, _, _, _, trace, log = _both(model, data, ff)
    stops = re.findall(r"adaround Iterations=(\d+), mean loss", log)
    assert len(stops) == 3 and [len(t) for t in trace] == [int(s) for s in stops]


def test_mem_opt_level_2_without_workers_fails_in_every_layer_unless_batch_size_one():
    model, data = _single_sample_cnn()
    _check_equal(
        model,
        data,
        P._ff("adaround", MemOptLevel=2, NumWorkers=0, BatchSize=4, NumIterations=20),
        trains=False,
    )
    _check_equal(
        model,
        data,
        P._ff("adaround", MemOptLevel=2, NumWorkers=0, BatchSize=1, NumIterations=20),
    )


def _gemm_chain(n_out):
    r = np.random.default_rng(0)
    return _model(
        "h = Gemm(x, w1, b1)\n t = Relu(h)\n y = Gemm<transB=1>(t, w2, b2)",
        [
            _w(r, "w1", 6, 7),
            _w(r, "b1", 7, scale=0.1),
            _w(r, "w2", n_out, 7),
            _w(r, "b2", n_out, scale=0.1),
        ],
        (4, 6),
        n_batches=8,
    )


@pytest.mark.parametrize("n_out", [5, 4], ids=["n-5", "n-equals-batch-bias-on-axis-1"])
@pytest.mark.parametrize(
    "extra", [dict(), dict(OutputQDQ=True, DropRatio=0.5)], ids=["plain", "outqdq-drop"]
)
def test_mem_opt_level_2_trains_whole_batches_through_gemm_and_matmul(n_out, extra):
    """Calibration batches of 4 samples stack to [bs, 4, K] in the DataLoader:
    Gemm / MatMul / LayerNorm train on them (Conv fails there, see below), with
    Quark's bias broadcast along axis 1 when it has the bias' length."""
    model, data = _gemm_chain(n_out)
    ff = P._ff("adaround", MemOptLevel=2, NumIterations=20, BatchSize=2, **extra)
    _check_equal(model, data, ff)


@pytest.mark.parametrize("kind", ["C", "F"])
def test_mem_opt_level_2_trains_matmul_and_layernorm_on_stacked_batches(kind):
    model, data = P._build(kind)
    ff = P._ff("adaround", MemOptLevel=2, NumIterations=20, BatchSize=2)
    _check_equal(model, data, ff)


# -- options with no effect on the numbers, DynamicBatch ----------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        dict(LogPeriod=3),
        dict(PinMemory=True),
        dict(MemOptLevel=0),
        dict(NumWorkers=3),
        dict(UseGDS=True),
    ],
    ids=lambda e: next(iter(e)),
)
def test_options_probed_to_be_inert_leave_quarks_codes_unchanged(extra):
    if torch.cuda.is_available():  # pragma: no cover - GPU boxes only
        pytest.skip("device-dependent options")
    model, data = P._build("A")
    q = P._quark_quantize(model, data)[0]
    ff = P._ff("adaround", NumIterations=40)
    with P._Quiet():
        ref = P.fast_finetune(model, q, False, P._Reader(data), {"FastFinetune": ff})
        out = P.fast_finetune(
            model, q, False, P._Reader(data), {"FastFinetune": dict(ff, **extra)}
        )
    assert _changed(q, ref) and not _changed(ref, out)


def test_tmp_dir_and_devices_leave_quarks_codes_unchanged(tmp_path):
    if torch.cuda.is_available():  # pragma: no cover - GPU boxes only
        pytest.skip("device-dependent options")
    model, data = P._build("A")
    q = P._quark_quantize(model, data)[0]
    ff = P._ff("adaround", NumIterations=40)
    extra = dict(TmpDir=str(tmp_path), OptimDevice="cuda", InferDevice="cuda")
    with P._Quiet():
        ref = P.fast_finetune(model, q, False, P._Reader(data), {"FastFinetune": ff})
        out = P.fast_finetune(
            model, q, False, P._Reader(data), {"FastFinetune": dict(ff, **extra)}
        )
    assert not _changed(ref, out)


def _single_sample_gemm_cnn():
    r = np.random.default_rng(0)
    return _model(
        "c1 = Conv<pads=[1,1,1,1]>(x, w1, b1)\n r = Relu(c1)\n f = Flatten(r)\n"
        "y = Gemm<transB=1>(f, w3, b3)",
        [_w(r, "w1", 4, 3, 3, 3), _w(r, "b1", 4, scale=0.1), _w(r, "w3", 4, 256, scale=0.2), _w(r, "b3", 4, scale=0.1)],
        (1, 3, 8, 8),
        io="float[1,3,8,8] x) => (float[1,4] y",  # (Quark needs a shaped output)
        n_batches=12,
    )  # fmt: skip


def test_dynamic_batch_is_a_no_op_for_single_sample_batches():
    model, data = _single_sample_gemm_cnn()
    q = P._quark_quantize(model, data)[0]
    plain = P._ff("adaround", NumIterations=60, BatchSize=3)
    quark_off, mine_off, _, _, _, _ = _both(model, data, plain, q=q)
    on = dict(plain, DynamicBatch=True)
    quark_on, mine_on, *_ = _both(model, data, on, q=q)
    assert _changed(q, quark_on)
    assert not _changed(quark_off, quark_on) and not _changed(quark_on, mine_on)
    assert not _changed(mine_off, mine_on)


def test_dynamic_batch_trains_nothing_when_batches_hold_more_than_one_sample():
    model, data = P._build("A")
    on = P._ff("adaround", NumIterations=40, DynamicBatch=True)
    _check_equal(model, data, on, trains=False)


def test_select_max_mem_layer_aborts_in_both_next_to_a_dynamic_batch_mismatch():
    model, data = P._build("A")
    q = P._quark_quantize(model, data)[0]
    ff = P._ff("adaround", NumIterations=20, DynamicBatch=True, SelectMaxMemLayer=True)
    with pytest.raises(Exception, match="invalid dimensions"):
        with P._Quiet():
            P.fast_finetune(model, q, False, P._Reader(data), {"FastFinetune": ff})
    with pytest.raises(RuntimeError, match="DynamicBatch"):
        qf.finetune(model, q, data, _opts(ff))


def test_select_max_mem_layer_aborts_in_both_next_to_an_unconvertible_layer():
    model, data = _two_convs('auto_pad="SAME_UPPER", kernel_shape=[3,3]')
    q = P._quark_quantize(model, data)[0]
    ff = P._ff("adaround", NumIterations=20, SelectMaxMemLayer=True)
    with pytest.raises(NotImplementedError, match="auto_pad=SAME_UPPER"):
        with P._Quiet():
            P.fast_finetune(model, q, False, P._Reader(data), {"FastFinetune": ff})
    with pytest.raises(NotImplementedError, match="auto_pad=SAME_UPPER"):
        qf.finetune(model, q, data, _opts(ff))


def test_layer_norm_over_another_axis_is_skipped_or_aborts_like_quark():
    r = np.random.default_rng(0)
    model, data = _model(
        "y = LayerNormalization<axis=1>(x, g, b)",
        [_w(r, "g", 5, 8), _w(r, "b", 5, 8, scale=0.1)],
        (4, 5, 8),
    )
    ff = P._ff("adaround", NumIterations=20)
    _check_equal(model, data, ff, trains=False)
    q = P._quark_quantize(model, data)[0]
    sel = dict(ff, SelectMaxMemLayer=True)
    with pytest.raises(NotImplementedError, match="axis is not -1"):
        with P._Quiet():
            P.fast_finetune(model, q, False, P._Reader(data), {"FastFinetune": sel})
    with pytest.raises(NotImplementedError, match="axis is not -1"):
        qf.finetune(model, q, data, _opts(sel))


# -- the statistical agreement still holds for the new layers -------------------------------------


def test_end_to_end_error_of_a_grouped_convtranspose_tracks_quarks():
    model, data = _model(
        "c1 = ConvTranspose<group=2, strides=[2,2], pads=[1,1,1,1]>(x, w1, b1)\n y = Relu(c1)",
        [_w(_R, "w1", 4, 3, 3, 3), _w(_R, "b1", 6, scale=0.1)],
        (4, 4, 5, 5),
        n_batches=8,
    )
    x_test = (
        np.random.default_rng(99).standard_normal((128, 4, 5, 5)).astype(np.float32)
    )
    ff = P._ff("adaround", NumIterations=300, BatchSize=2)
    quark, mine, q, *_ = _both(model, data, ff, replay=False)
    e_q, e_m, e_plain = (P._e2e(model, m, x_test) for m in (quark, mine, q))
    assert e_m == pytest.approx(e_q, rel=0.15)
    assert 0.85 * e_plain < e_m < 1.15 * e_plain
