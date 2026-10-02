"""Quark-free tests of the AutoMixprecision pieces that mirror Quark 0.13's
``MixingStrategy``: weight / bias precision mixing, the scoring conventions
(graph optimizations off, ``data_size`` counting one batch more), the block
formats' candidate scoring on the ONNX reference path, and the options both
paths share (``subgraph_json``, ``sensitivity_cache_file``, ...). Parity with
the real Quark is in ``tests/test_quark_amp_parity.py``."""

import json
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_auto_mixprecision as amp
from onnxsim import quark_block_formats as qbf
from onnxsim import quark_compat as qc
from onnxsim.quark_fakequant_eval import run_fake_quantized
from onnxsim.quark_preset_graphs import dedicate_dq_nodes

D = 16


def _w(rng, *shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _model(body, inits, inputs=f"float[3,{D}] x", outputs=f"float[3,{D}] y"):
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
    )


def _data(n=6):
    return [
        {"x": np.random.default_rng(i).standard_normal((3, D)).astype(np.float32)}
        for i in range(n)
    ]


def _dq_const(model, node_name, slot):
    """``(codes, scale, zero_point, DequantizeLinear)`` feeding input ``slot`` of
    ``node_name`` when it is a quantized constant."""
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    prod = {o: n for n in model.graph.node for o in n.output}
    node = next(n for n in model.graph.node if n.name == node_name)
    dq = prod[node.input[slot]]
    assert dq.op_type == "DequantizeLinear"
    return inits[dq.input[0]], inits[dq.input[1]], inits[dq.input[2]], dq


def _dq_scale(model, tensor):
    """Scale of the DequantizeLinear that produces ``tensor``."""
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    dq = next(n for n in model.graph.node if tensor in n.output)
    return inits[dq.input[1]]


def _baseline(**kw):
    return amp.auto_mixprecision(
        _mlp(), _data(), base_dtype="uint8", metric_threshold=None, **kw
    ).model


# -- weights and biases move with the layer -------------------------------------------


def test_weight_moves_to_int16_from_its_quantized_values():
    spec = amp.TargetSpec(
        inputs=("uint16", None),
        outputs=("uint16", None),
        weight=("int16", True),
    )
    res = amp.auto_mixprecision(
        _mlp(),
        _data(),
        base_dtype="uint8",
        targets=[spec],
        include_layers=["n2_Gemm"],
    )
    base = _baseline(targets=[spec])
    codes8, scale8, zp8, _ = _dq_const(base, "n2_Gemm", 1)
    codes16, scale16, zp16, dq16 = _dq_const(res.model, "n2_Gemm", 1)
    assert codes8.dtype == np.int8 and codes16.dtype == np.int16
    # the int8 resolution survives: dequantized int8 values, one per-tensor
    # scale from their range (symmetric: max |w| / 32767), rounded
    deq = (codes8.astype(np.float32) - zp8.astype(np.float32)) * scale8
    want_scale = np.float32(np.abs(deq).max() / 32767)
    assert scale16.shape == () and np.isclose(scale16, want_scale, rtol=1e-6)
    np.testing.assert_array_equal(codes16, np.round(deq / scale16).astype(np.int16))
    assert dq16.domain == "com.microsoft"  # int16 below opset 21
    assert any(o.domain == "com.microsoft" for o in res.model.opset_import)
    # the other layers keep their int8 weights
    for node in ("n0_Gemm", "n4_Gemm"):
        assert _dq_const(res.model, node, 1)[0].dtype == np.int8


def test_int32_bias_scale_follows_the_moved_input_and_weight_scales():
    res = amp.auto_mixprecision(
        _mlp(),
        _data(),
        base_dtype="uint8",
        targets=[("uint16", None)],
        include_layers=["n2_Gemm"],
    )
    base = _baseline(targets=[("uint16", None)])
    # n2: its input t1 is now uint16 (Quark refreshes the bias scale to
    # input_scale * weight_scale) and the codes are rescaled from the baseline
    # codes -- truncated, not rounded, in float32
    b_codes, b_scale, _, _ = _dq_const(base, "n2_Gemm", 2)
    codes, scale, _, _ = _dq_const(res.model, "n2_Gemm", 2)
    in_scale = _dq_scale(res.model, "t1")
    w_scale = _dq_const(res.model, "n2_Gemm", 1)[1]
    want_scale = (in_scale * w_scale).astype(np.float32)
    np.testing.assert_array_equal(scale, want_scale)
    want = (
        b_codes.astype(np.float32) * b_scale.astype(np.float32) / want_scale
    ).astype(np.int32)
    np.testing.assert_array_equal(codes, want)
    # layers that did not move keep their baseline bias (Quark edits the
    # quantized baseline in place and refreshes only the layers it moves)
    for node in ("n0_Gemm", "n4_Gemm"):
        np.testing.assert_array_equal(
            _dq_const(res.model, node, 2)[0], _dq_const(base, node, 2)[0]
        )
        np.testing.assert_array_equal(
            _dq_const(res.model, node, 2)[1].ravel()[:1],
            _dq_const(base, node, 2)[1].ravel()[:1],
        )


def test_bias_spec_requantizes_the_bias_and_skips_the_scale_refresh():
    spec = amp.TargetSpec(
        inputs=("int8", False), weight=("int8", True), bias=("int8", True)
    )
    res = amp.auto_mixprecision(
        _mlp(),
        _data(),
        base_dtype="int16",
        targets=[spec],
        include_layers=["n2_Gemm"],
        quantize_kwargs=dict(weight_dtype="int16"),
    )
    codes, scale, zp, dq = _dq_const(res.model, "n2_Gemm", 2)
    assert codes.dtype == np.int8 and zp.dtype == np.int8
    assert dq.domain == ""
    # input tensors move, outputs (no ``outputs`` precision) stay int16
    inits = {t.name: t for t in res.model.graph.initializer}
    q_zp = {
        n.input[0].removesuffix("/f"): inits[n.input[2]].data_type
        for n in res.model.graph.node
        if n.op_type == "QuantizeLinear" and len(n.input) > 2
    }
    assert q_zp["t1"] == onnx.TensorProto.INT8
    assert q_zp["h2"] == onnx.TensorProto.INT16


def test_a_target_equal_to_the_base_precision_still_requantizes_weights():
    # Quark does not refuse "nothing to mix": the weights become per-tensor and
    # the bias scales are refreshed
    spec = amp.TargetSpec(
        inputs=("uint8", None), outputs=("uint8", None), weight=("int8", True)
    )
    res = amp.auto_mixprecision(
        _mlp(),
        _data(),
        base_dtype="uint8",
        targets=[spec],
        quantize_kwargs=dict(per_channel=True),
    )
    base = _baseline(targets=[spec], quantize_kwargs=dict(per_channel=True))
    assert _dq_const(base, "n0_Gemm", 1)[1].shape != ()  # per channel
    assert _dq_const(res.model, "n0_Gemm", 1)[1].shape == ()  # per tensor
    assert res.moved


def test_unsupported_constant_precisions_are_refused():
    with pytest.raises(ValueError, match="dtypes"):
        amp.auto_mixprecision(
            _mlp(),
            _data(),
            base_dtype="uint8",
            targets=[amp.TargetSpec(weight=("float16", None))],
        )


# -- the compat layer reads a QLayerConfig like Quark's MixingStrategy --------------


def _amp_config(base_act, base_wt, target, **params):
    return qc.QConfig(
        global_config=qc.QLayerConfig(activation=base_act(), weight=base_wt()),
        algo_config=[qc.AutoMixprecisionConfig(target_layer_config=target, **params)],
    )


def _zp_types(model):
    inits = {t.name: t for t in model.graph.initializer}
    out = {}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and len(n.input) > 2 and n.input[2] in inits:
            dt = onnx.TensorProto.DataType.Name(inits[n.input[2]].data_type)
            if dt != "INT32":
                out[n.input[0].removesuffix("/f")] = dt
    return out


def _quantize(cfg, **kw):
    q = qc.ModelQuantizer(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return q, q.quantize_model(_mlp(), calibration_data_reader=_data(), **kw)


def test_activation_target_moves_inputs_and_outputs_but_input_tensors_only_inputs():
    both = qc.QLayerConfig(activation=qc.Int16Spec(), weight=qc.Int8Spec())
    ins = qc.QLayerConfig(input_tensors=qc.Int16Spec(), weight=qc.Int8Spec())
    kw = dict(include_layers=["n2_Gemm"])
    _, m_both = _quantize(_amp_config(qc.UInt8Spec, qc.Int8Spec, both, **kw))
    _, m_ins = _quantize(_amp_config(qc.UInt8Spec, qc.Int8Spec, ins, **kw))
    assert _zp_types(m_both)["t1"] == _zp_types(m_both)["h2"] == "INT16"
    assert _zp_types(m_ins)["t1"] == "INT16" and _zp_types(m_ins)["h2"] == "UINT8"


def test_target_weight_and_bias_specs_reach_the_model():
    target = qc.QLayerConfig(
        activation=qc.UInt16Spec(),
        weight=qc.Int16Spec(),
        bias=qc.Int16Spec(),
    )
    cfg = _amp_config(qc.UInt8Spec, qc.Int8Spec, target, include_layers=["n2_Gemm"])
    _, out = _quantize(cfg)
    assert _dq_const(out, "n2_Gemm", 1)[0].dtype == np.int16
    assert _dq_const(out, "n2_Gemm", 2)[0].dtype == np.int16
    assert _dq_const(out, "n0_Gemm", 1)[0].dtype == np.int8
    assert _dq_const(out, "n0_Gemm", 2)[0].dtype == np.int32
    onnx.checker.check_model(out)
    ort.InferenceSession(out.SerializeToString(), providers=["CPUExecutionProvider"])


def test_target_symmetry_follows_the_global_specs_like_quark():
    # the global activation spec is asymmetric, so the int16 target is too even
    # though Int16Spec() alone is symmetric (Quark's ActivationSymmetric wins)
    target = qc.QLayerConfig(activation=qc.Int16Spec(), weight=qc.Int8Spec())
    cfg = _amp_config(qc.UInt8Spec, qc.Int8Spec, target, include_layers=["n2_Gemm"])
    _, out = _quantize(cfg)
    inits = {t.name: numpy_helper.to_array(t) for t in out.graph.initializer}
    q = next(
        n
        for n in out.graph.node
        if n.op_type == "QuantizeLinear" and n.input[0].startswith("h2")
    )
    assert inits[q.input[2]] != 0  # an asymmetric int16 zero point


def test_power_of_two_target_weights_are_refused():
    target = qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.XInt8Spec())
    with pytest.raises(NotImplementedError, match="power-of-two"):
        _quantize(_amp_config(qc.UInt8Spec, qc.Int8Spec, target))


def test_float_targets_over_an_integer_base_are_refused():
    target = qc.QLayerConfig(activation=qc.BFP16Spec(), weight=qc.BFP16Spec())
    with pytest.raises(NotImplementedError, match="unsupported"):
        _quantize(_amp_config(qc.UInt8Spec, qc.Int8Spec, target))


# -- scoring conventions ----------------------------------------------------------------


def test_candidates_are_scored_with_graph_optimizations_off(monkeypatch):
    levels = []
    real = ort.InferenceSession

    def spy(model, sess_options=None, *a, **k):
        if sess_options is not None:
            levels.append(sess_options.graph_optimization_level)
        return real(model, sess_options, *a, **k)

    monkeypatch.setattr(ort, "InferenceSession", spy)
    amp.auto_mixprecision(
        _mlp(), _data(), base_dtype="uint8", targets=[("uint16", None)]
    )
    # float model + baseline + three candidates + the mixed trials (the
    # calibration pass runs its own sessions)
    assert levels.count(ort.GraphOptimizationLevel.ORT_DISABLE_ALL) >= 8


@pytest.mark.parametrize("data_size, scored", [(0, 1), (1, 2), (3, 4), (5, 6), (50, 6)])
def test_data_size_counts_one_batch_more_like_quark(data_size, scored):
    # Quark's inference_model stops once len(results) > data_size: the default 0
    # scores only the first batch (the documented "0 = all" does not hold)
    seen = []

    def metric(float_out, quant_out):
        seen.append(len(float_out))
        return float(len(float_out))

    cfg = _amp_config(
        qc.UInt8Spec,
        qc.Int8Spec,
        qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.Int8Spec()),
        metric_distance_fn=metric,
        data_size=data_size,
    )
    _quantize(cfg)
    assert set(seen) == {scored}


def test_missing_subgraph_json_is_ignored_with_a_note():
    cfg = _amp_config(
        qc.UInt8Spec,
        qc.Int8Spec,
        qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.Int8Spec()),
        subgraph_json="/no/such/file.json",
    )
    q, out = _quantize(cfg)
    assert any("does not exist" in a for a in q.last_approximations)
    assert q.last_auto_mixprecision.moved


# -- the int16 -> int8 promotion goes through the same machinery ----------------------


def _s16_config(**params):
    target = qc.QLayerConfig(
        input_tensors=qc.Int8Spec(symmetric=False),
        weight=qc.Int8Spec(),
        bias=qc.Int8Spec(),
    )
    return qc.QConfig(
        global_config=qc.QLayerConfig(
            activation=qc.Int16Spec(symmetric=False), weight=qc.Int16Spec()
        ),
        algo_config=[qc.AutoMixprecisionConfig(target_layer_config=target, **params)],
        Int32Bias=False,
    )


def _write_subgraphs(path):
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
    return str(path)


def test_int16_to_int8_accepts_subgraph_json_and_cache_pins(tmp_path):
    sg, cache = _write_subgraphs(tmp_path / "sg.json"), tmp_path / "cache.json"
    q, full = _quantize(
        _s16_config(subgraph_json=sg, sensitivity_cache_file=str(cache))
    )
    doc = json.loads(cache.read_text())
    assert {r["name"] for r in doc["results"]} == {"front", "back", "__ungrouped__"}
    assert len(q.last_auto_mixprecision.moved) == 3
    for n in ("n0_Gemm", "n2_Gemm", "n4_Gemm"):
        assert _dq_const(full, n, 1)[0].dtype == np.int8
        assert _dq_const(full, n, 2)[0].dtype == np.int8
    # pin "back": its layer keeps int16 weights and an int16 bias, and its
    # input / output stay int16
    for r in doc["results"]:
        r["enabled"] = r["name"] != "back"
    cache.write_text(json.dumps(doc))
    q, pinned = _quantize(
        _s16_config(subgraph_json=sg, sensitivity_cache_file=str(cache))
    )
    assert (
        q.last_auto_mixprecision.moved and "back" not in q.last_auto_mixprecision.moved
    )
    assert _dq_const(pinned, "n2_Gemm", 1)[0].dtype == np.int16
    assert _dq_const(pinned, "n2_Gemm", 2)[0].dtype == np.int16
    assert _dq_const(pinned, "n0_Gemm", 1)[0].dtype == np.int8
    assert _zp_types(pinned)["t1"] == "INT16" and _zp_types(pinned)["x"] == "INT8"


def test_int16_unpromoted_layers_keep_int16_weights_and_biases():
    q, out = _quantize(_s16_config(include_layers=["n2_Gemm"]))
    assert _dq_const(out, "n2_Gemm", 1)[0].dtype == np.int8
    for n in ("n0_Gemm", "n4_Gemm"):
        assert _dq_const(out, n, 1)[0].dtype == np.int16
        codes, scale, _, _ = _dq_const(out, n, 2)
        assert codes.dtype == np.int16  # Int32Bias=False: quantized like a weight


# -- block formats: candidates scored on the reference path ---------------------------


def _bf16_mixed(**params):
    cfg = qc.QConfig.get_default_config("BF16_MIXED_BFP16")
    cfg.algo_config[0].params.update(params)
    return cfg


def _block_run(**params):
    q = qc.ModelQuantizer(_bf16_mixed(**params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = q.quantize_model(_mlp(), calibration_data_reader=_data())
    return q, out


def _block_weights(model):
    """Gemm names whose weight reads through a BFP node."""
    prod = {o: n for n in model.graph.node for o in n.output}
    return {
        n.name
        for n in model.graph.node
        if n.op_type == "Gemm" and prod[n.input[1]].op_type == "BFPQuantizeDequantize"
    }


def test_block_cache_is_written_ranked_and_pins(tmp_path):
    cache = tmp_path / "c.json"
    q, full = _block_run(sensitivity_cache_file=str(cache))
    doc = json.loads(cache.read_text())
    scores = [r["score"] for r in doc["results"]]
    assert scores == sorted(scores) and len(scores) == 3
    assert {r["name"] for r in doc["results"]} == {"n0_Gemm", "n2_Gemm", "n4_Gemm"}
    assert _block_weights(full) == {"n0_Gemm", "n2_Gemm", "n4_Gemm"}
    assert q.last_auto_mixprecision.ranked
    # reuse the file with one candidate pinned: it stays bfloat16
    for r in doc["results"]:
        r["enabled"] = r["name"] != "n2_Gemm"
    cache.write_text(json.dumps(doc))
    _, pinned = _block_run(sensitivity_cache_file=str(cache))
    assert _block_weights(pinned) == {"n0_Gemm", "n4_Gemm"}
    _, excluded = _block_run(exclude_layers=["n2_Gemm"])
    assert pinned.SerializeToString() == excluded.SerializeToString()


def test_block_subgraph_json_groups_the_candidates(tmp_path):
    cache = tmp_path / "c.json"
    sg = _write_subgraphs(tmp_path / "sg.json")
    _, grouped = _block_run(subgraph_json=sg)
    _, plain = _block_run()
    # every candidate moves anyway (threshold 0): same model as without it
    assert grouped.SerializeToString() == plain.SerializeToString()
    _block_run(subgraph_json=sg, sensitivity_cache_file=str(cache))
    names = [
        (r["name"], r["candidate_nodes"])
        for r in json.loads(cache.read_text())["results"]
    ]
    assert sorted(names) == [
        ("__ungrouped__", ["n4_Gemm"]),
        ("back", ["n2_Gemm"]),
        ("front", ["n0_Gemm"]),
    ]
    q, _ = _block_run(subgraph_json="/no/such.json")
    assert any("does not exist" in a for a in q.last_approximations)


def test_block_thresholds_stop_the_mixing_like_the_integer_path():
    q, none = _block_run(metric_threshold=None)
    assert not q.last_auto_mixprecision.moved and not _block_weights(none)
    q, tight = _block_run(metric_threshold=1e-9)  # the baseline is already worse
    assert not q.last_auto_mixprecision.moved and not _block_weights(tight)
    q, loose = _block_run(metric_threshold=1e9)
    assert len(q.last_auto_mixprecision.moved) == 3
    assert _block_weights(loose) == {"n0_Gemm", "n2_Gemm", "n4_Gemm"}
    # "quality": the baseline already meeting the threshold needs no work ...
    q, _ = _block_run(metric_threshold=1e9, metric_optimize_object="quality")
    assert not q.last_auto_mixprecision.moved
    # ... and one that does not moves candidates until the score drops to it
    baseline = q.last_auto_mixprecision.baseline_score
    q, _ = _block_run(
        metric_threshold=baseline * 0.999, metric_optimize_object="quality"
    )
    assert q.last_auto_mixprecision.moved


def test_block_dual_nodes_only_when_asked():
    _, single = _block_run(dual_quant_nodes=False)
    _, dual = _block_run(dual_quant_nodes=True)
    extra = lambda m: {n.name for n in m.graph.node if "_additional_" in n.name}  # noqa: E731
    assert not extra(single) and extra(dual)


def test_block_candidates_are_scored_on_the_fake_quantized_models():
    from onnxsim.quark_preset_graphs import apply_mixed_block_format

    model, data = _mlp(), _data()
    res = amp.auto_mixprecision_blocks(
        model, data, "bfp16", metric_threshold=None, data_size=len(data)
    )
    sess = ort.InferenceSession(model.SerializeToString())
    base = apply_mixed_block_format(
        model, "bfp16", include_layers=["no-such-layer"], dual_nodes=False
    )
    ref = [sess.run(None, d)[0] for d in data]
    got = [o[0] for o in run_fake_quantized(base, data)]
    want = float(np.mean([np.linalg.norm(a - b) for a, b in zip(ref, got)]))
    assert res.baseline_score == pytest.approx(want, rel=1e-6)
    assert [c.score for c in res.ranked] == sorted(c.score for c in res.ranked)
    # moving a candidate changes its score: the block format really is applied
    assert all(c.score != res.baseline_score for c in res.ranked)


def test_reference_evaluator_runs_the_custom_ops_bit_exactly():
    x = np.random.default_rng(1).standard_normal((2, 32)).astype(np.float32)
    g = parser.parse_model(
        """<ir_version: 9, opset_import: ["": 17, "com.amd.quark": 1]>
        g (float[2,32] x) => (float[2,32] y) {
            a = com.amd.quark.BFPQuantizeDequantize<bfp_method="to_bfp", axis=1,
                bit_width=16, block_size=8, rounding_mode=2>(x)
            y = com.amd.quark.MXQuantizeDequantize<element_dtype="int8", axis=1,
                block_size=32, rounding_mode=2>(a)
        }"""
    )
    (y,) = run_fake_quantized(g, [{"x": x}])[0]
    np.testing.assert_array_equal(y, qbf.mx(qbf.bfp16(x, axis=1), "int8", axis=1))


# -- DedicateDQNode ---------------------------------------------------------------------


def test_dedicate_dq_nodes_copies_a_shared_dequantizer_per_reader():
    m = parser.parse_model(
        """<ir_version: 9, opset_import: ["": 17, "com.amd.quark": 1]>
        g (float[2] x) => (float[2] a, float[2] b, float[2] c) {
            q = com.amd.quark.ExtendedQuantizeLinear(x, s, z)
            d = com.amd.quark.ExtendedDequantizeLinear(q, s, z)
            a = Relu(d)
            b = Neg(d)
            c = Identity(d)
        }"""
    )
    m.graph.initializer.append(numpy_helper.from_array(np.float32(1.0), "s"))
    m.graph.initializer.append(numpy_helper.from_array(np.float32(0.0), "z"))
    out = dedicate_dq_nodes(m)
    dqs = [n for n in out.graph.node if n.op_type == "ExtendedDequantizeLinear"]
    assert len(dqs) == 3
    assert [n.output[0] for n in dqs] == ["d", "d_1", "d_2"]
    reads = {
        n.op_type: n.input[0]
        for n in out.graph.node
        if n.op_type in ("Relu", "Neg", "Identity")
    }
    assert reads == {"Relu": "d", "Neg": "d_1", "Identity": "d_2"}
