"""The ResNet18 training-step runner (``scripts/axera/step_runner.py``), the
persistent AXCL session (``axcl_session.py``) and the tinygrad ``AX`` device.

Offline tests need only the committed fixtures. The step graph itself
(``step.onnx``) and the device are local to the Axera box: those tests skip
elsewhere. Device tests run through ``AXCL_LXD_VM`` like every other device
test here, and hold ``/tmp/axcl-device.lock`` while they do.
"""

from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import capture_fp32_binaries as fp32_capture  # noqa: E402
import misc_op_record_emit as misc  # noqa: E402
import step_runner as sr  # noqa: E402

_HAVE_STEP = os.path.exists(sr.STEP_ONNX) and os.path.exists(sr.STEP_REF)
needs_step = pytest.mark.skipif(
    not _HAVE_STEP, reason="step.onnx is local to the Axera box"
)


def _device_ok() -> bool:
    # pulsar2_docker.axcl_available() does the same, but importing it needs the
    # built onnxsim extension
    import subprocess

    vm = os.environ.get("AXCL_LXD_VM")
    if not vm:
        return False
    try:
        return (
            subprocess.run(
                ["lxc", "exec", vm, "--", "test", "-x", "/usr/bin/axcl/axcl_run_model"],
                capture_output=True,
                timeout=30,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_device = pytest.mark.skipif(
    not _device_ok(), reason="needs the AX650 (AXCL_LXD_VM)"
)


def test_fake_quant_rounds_and_clips():
    x = np.array([-1.0, 0.0, 0.004, 0.006, 10.0], np.float32)
    got = sr.fake_quant(x, 0.01, 100, False)
    np.testing.assert_allclose(got, [-1.0, 0.0, 0.0, 0.01, 1.55], atol=1e-6)
    got = sr.fake_quant(x, 0.01, 0, True)
    np.testing.assert_allclose(got, [-1.0, 0.0, 0.0, 0.01, 1.27], atol=1e-6)


def test_softmax_ratio_gradient_rewrite_avoids_zero_probability_nan():
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 11]> '
        "agraph (float[1,4] a, float[1,4] p) => (float[1,4] y) { "
        "q = Div(a, p) "
        "r = Mul(q, p) "
        "s = ReduceSum<axes=[1], keepdims=1>(r) "
        "d = Sub(q, s) "
        "y = Mul(p, d) }"
    )
    assert sr.rewrite_softmax_ratio_gradients(model) == 1
    onnx.checker.check_model(model)
    assert all(node.op_type != "Div" for node in model.graph.node)
    import onnxruntime as ort

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    a = np.array([[0.25, 0.0, 0.5, 0.0]], np.float32)
    p = np.array([[0.5, 0.0, 0.5, 0.0]], np.float32)
    got = session.run(None, {"a": a, "p": p})[0]
    expected = a - p * a.sum(axis=1, keepdims=True)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-7)


def test_softmax_ratio_gradient_rewrite_matches_nonzero_reference():
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 11]> '
        "agraph (float[1,3] a, float[1,3] p) => (float[1,3] y) { "
        "q = Div(a, p) "
        "r = Mul(q, p) "
        "s = ReduceSum<axes=[1], keepdims=1>(r) "
        "d = Sub(q, s) "
        "y = Mul(p, d) }"
    )
    assert sr.rewrite_softmax_ratio_gradients(model) == 1
    import onnxruntime as ort

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    a = np.array([[0.1, 0.3, 0.6]], np.float32)
    p = np.array([[0.2, 0.3, 0.5]], np.float32)
    got = session.run(None, {"a": a, "p": p})[0]
    expected = p * (a / p - np.sum((a / p) * p, axis=1, keepdims=True))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)


@needs_step
def test_step_softmax_ratio_gradient_rewrite_preserves_float_gradients():
    model = sr.load_step()
    reference = sr.load_reference()
    gradients = sr.gradient_tensors(model, reference["state_map"])
    before, _ = sr.StepRunner(model, []).run(
        reference["feeds"], "float", keep=list(gradients.values())
    )
    assert sr.rewrite_softmax_ratio_gradients(model) == 2
    after, _ = sr.StepRunner(model, []).run(
        reference["feeds"], "float", keep=list(gradients.values())
    )
    for tensor in gradients.values():
        original = np.asarray(before[tensor], np.float64)
        rewritten = np.asarray(after[tensor], np.float64)
        relative = np.linalg.norm(original - rewritten) / max(
            np.linalg.norm(original), 1e-30
        )
        assert np.isfinite(rewritten).all()
        assert relative < 2e-6, (tensor, relative)


@needs_step
def test_step_masked_div_selects_safe_native_template():
    model = sr.load_step()
    record = next(r for r in sr.load_records() if r["name"] == "Div_453")
    segment = sr._safe_masked_div_segment_for(record, model)
    assert segment is not None
    assert segment.kind == "safe_masked_div"
    assert segment.output_shape == ()  # preserve the smaller count input for Expand
    template = segment.emit()
    assert [
        tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
        for i in template.graph.input
    ] == [
        (1024, 9, 3136),
        (1024, 1, 3136),
    ]


def test_segment_validation_rejects_nonfinite_device_outputs():
    segment = sr.Segment(
        "nonfinite",
        "elementwise",
        [],
        [],
        [],
        "test",
        lambda: onnx.ModelProto(),
        out_q=[(0.1, 0, False)],
    )
    stat = sr.SegStat("nonfinite", "elementwise", 0)
    nan = np.array([np.nan], dtype=np.float32)
    sr._compare(stat, segment, [nan], [nan])
    assert "non-finite" in stat.error
    assert not sr.segment_passed(vars(stat))
    assert np.isinf(sr._rel(nan, np.ones(1, dtype=np.float32)))


def test_validated_recheck_preserves_unselected_results():
    report = {
        "segment_stats": [
            {"segment": "known-good", "error": "", "max_lsb": 0.0, "frac_gt1": 0.0},
            {"segment": "candidate", "error": "", "max_lsb": 4.0, "frac_gt1": 0.0},
        ]
    }
    assert sr.validated_failures(report, ["candidate"]) == set()
    with pytest.raises(ValueError, match="previously failing"):
        sr.validated_failures(report, ["known-good"])
    current = [sr.SegStat("candidate", "elementwise", 1, max_lsb=1.0)]
    merged = sr.merge_validation_stats(report, current)
    assert [(item["segment"], item["max_lsb"]) for item in merged] == [
        ("candidate", 1.0),
        ("known-good", 0.0),
    ]


def test_runtime_fallback_uses_original_host_op_after_device_mismatch():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "z"], ["y"], name="add")],
        "runtime_fallback",
        [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1])
            for name in ("x", "z")
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 10
    segment = sr.Segment(
        "add",
        "elementwise",
        ["add"],
        ["x", "z"],
        ["y"],
        "test",
        lambda: model,
        in_q=[None, None],
        out_q=[(0.1, 0, False)],
    )
    runner = sr.StepRunner(model, [segment], fallback_on_failure=True)
    runner.emitted = lambda _: b"model"
    runner._device = lambda *_: [np.array([10.0], dtype=np.float32)]
    outputs, stats = runner.run(
        {
            "x": np.array([0.2], dtype=np.float32),
            "z": np.array([0.3], dtype=np.float32),
        },
        "npu",
    )
    assert stats[0].runtime_fallback
    assert stats[0].fallback_reason
    np.testing.assert_allclose(outputs["y"], [0.5])


@pytest.mark.parametrize("bits,signed,zp", [(16, False, 32768), (16, True, 0)])
def test_fake_quant_supports_explicit_16bit_segment_params(bits, signed, zp):
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    got = sr.fake_quant(x, 1.0, zp, signed, bits)
    lo, hi = (-32768, 32767) if signed else (0, 65535)
    expected = (np.clip(np.rint(x) + zp, lo, hi) - zp).astype(np.float32)
    np.testing.assert_array_equal(got, expected)


def test_step_planner_accepts_explicit_exact_16bit_binary_template():
    with open(
        os.path.join(
            HERE,
            "..",
            "scripts",
            "axera",
            "fixtures",
            "binary_op_precision",
            "index.json",
        )
    ) as stream:
        entry = next(
            item
            for item in json.load(stream)
            if item["op"] == "Add" and item["precision"] == "S16"
        )
    shape = entry["shape"]
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["x", "z"], ["y"], name="add16")],
            "explicit_16bit_plan",
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, shape),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    record = {
        "op": "Add",
        "name": "add16",
        "attrs": {"form": "same_shape"},
        "inputs": ["x", "z"],
        "outputs": ["y"],
        "shapes": [shape],
    }
    override = {
        "layer_precision": entry["precision"],
        "scales": dict(
            zip(
                ("x", "z", "y"),
                (entry["scales"][0], entry["scales"][1], entry["scales"][2]),
            )
        ),
        "zero_points": dict(zip(("x", "y", "z"), entry["zero_points"])),
    }
    u8_calib = {
        "tensors": {
            name: {"scale": 0.01, "zero_point": 0, "signed": False}
            for name in ("x", "z", "y")
        }
    }
    segments, host = sr.build_plan(
        model, [record], u8_calib, precision_overrides={"add16": override}
    )
    assert not host
    assert len(segments) == 1 and segments[0].kind == "binary_precision"
    assert segments[0].in_q == [
        (override["scales"]["x"], 0, True, 16),
        (override["scales"]["z"], 0, True, 16),
    ]
    emitted, blobs = sr.drop_unemittable(segments, host)
    assert emitted == segments
    assert blobs["add16"]
    bad_override = {
        **override,
        "scales": {**override["scales"], "y": override["scales"]["y"] * 1.01},
    }
    with pytest.raises(ValueError, match="no exact-calibration S16 template"):
        sr.build_plan(
            model, [record], u8_calib, precision_overrides={"add16": bad_override}
        )


@needs_device
def test_explicit_16bit_segment_runs_through_step_runner_on_axcl_vm():
    import axcl_session

    entry = next(
        item
        for item in json.load(
            open(
                os.path.join(
                    HERE,
                    "..",
                    "scripts",
                    "axera",
                    "fixtures",
                    "binary_op_precision",
                    "index.json",
                )
            )
        )
        if item["op"] == "Add" and item["precision"] == "S16"
    )
    shape = entry["shape"]
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["x", "z"], ["y"], name="add16")],
            "step_precision_vm",
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, shape),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    # ORT's schema registry in the pinned runtime supports IR <= 13.
    model.ir_version = 10
    record = {
        "op": "Add",
        "name": "add16",
        "attrs": {"form": "same_shape"},
        "inputs": ["x", "z"],
        "outputs": ["y"],
        "shapes": [shape],
    }
    override = {
        "layer_precision": entry["precision"],
        "scales": dict(zip(("x", "z", "y"), entry["scales"])),
        "zero_points": dict(zip(("x", "y", "z"), entry["zero_points"])),
    }
    calibration = {
        "tensors": {
            name: {"scale": 0.01, "zero_point": 0, "signed": False}
            for name in ("x", "z", "y")
        }
    }
    segments, host = sr.build_plan(
        model, [record], calibration, precision_overrides={"add16": override}
    )
    assert not host
    rng = np.random.default_rng(97)
    feeds = {
        name: rng.uniform(-0.75, 0.75, shape).astype(np.float32) for name in ("x", "z")
    }
    with axcl_session.AXSession(subdir="step_binary_precision_bridge") as session:
        _, stats = sr.StepRunner(model, segments, session, health_every=0).run(
            feeds, "npu"
        )
    assert len(stats) == 1 and stats[0].kind == "binary_precision"
    assert not stats[0].error, stats[0]
    assert stats[0].max_lsb <= 2.01, stats[0]


def _adam_graph() -> onnx.ModelProto:
    # w' = w - lr * m'; m' = 0.9 m + 0.1 g; g = x * w (a stand-in backward pass)
    return parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 17]>
        g (float[4] x, float[4] w, float[4] w__m) => (float[4] w_new, float[4] m_new)
          <float b1 = {0.9}, float ob1 = {0.1}, float lr = {0.01}> {
            grad = Mul(x, w)
            a = Mul(b1, w__m)
            b = Mul(ob1, grad)
            m_new = Add(a, b)
            step = Mul(lr, m_new)
            w_new = Sub(w, step)
        }
        """
    )


def test_gradient_tensors_follow_the_first_moment_update():
    m = _adam_graph()
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}"
    state_map = {"w": "w_new", "w__m": "m_new"}
    assert sr.gradient_tensors(m, state_map) == {"w": "grad"}
    # the Adam math (not the gradient's own producer) is the optimizer update
    opt = sr.optimizer_nodes(m, state_map)
    by_out = {n.output[0]: n.name for n in m.graph.node}
    assert by_out["grad"] not in opt
    assert {by_out[t] for t in ("a", "b", "m_new", "step", "w_new")} <= opt


def test_misc_emit_keeps_the_mcode_dims_in_step():
    """A zero-point move re-encodes the MCode to a different length. The
    runtime reads the length from the initializer's dims: a stale one fails
    to load (0x80300709) or, inside a long session, wedged the card. This is
    the ReduceSum_62 of the first whole-step run."""
    key = "ReduceSum:16x1x512x4608:axes0:k0"
    tmpl, meta = misc.load_template(key)
    out = misc.emit_model(
        key,
        {"x": 2.1986086721881293e-05, "y": 0.0001273591333301738},
        {"x": 163, "y": 68},
    )
    init = misc.mcode_initializer(out)
    assert len(init.raw_data) != len(misc.mcode_initializer(tmpl).raw_data)
    assert list(init.dims) == [len(init.raw_data)]


@needs_step
def test_plan_covers_the_validated_nodes_and_no_reshape_is_unsafe():
    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    records = sr.load_records()
    segs, host = sr.build_plan(model, records, calib)
    everything, _ = sr.build_plan(model, records, calib, include_unsafe=True)
    assert any(s.kind == "fp32_binary" for s in everything)
    assert sum(s.profiled_faster for s in everything) == 18
    assert any(s.name == "Mul_20" and s.kind == "mul_mask_exact" for s in segs)
    assert any(s.name == "Sub_24" and s.kind == "sub_loss_exact" for s in segs)
    assert {s.name for s in segs if s.kind == "div2_exact"} == {
        "Div_0",
        "Div_1",
        "Div_34",
    }
    assert "Mul_20" not in host
    assert "Sub_24" not in host
    assert not {"Div_0", "Div_1", "Div_34"} & host.keys()
    assert "Sub_32" in host  # different output scale/zp: exact fixture must not match
    # a node inside two chains is recomputed by both: count it once
    covered = len({n for s in everything for n in s.nodes})
    report_covered = sr.axb.coverage_report(records, calibration=calib)["totals"][
        "covered"
    ]
    # A tiled constant-broadcast segment uses a full-shape template that the
    # compact coverage records cannot represent, so subtract those synthetic
    # planner nodes from the record-level total.
    nonemittable = sum("no runner segment" in reason for reason in host.values())
    synthetic = sum(
        "tiled-" in s.detail
        or "flat-" in s.detail
        or ("constant from" in s.detail and bool(s.output_shape))
        for s in everything
    )
    # Sub_24 is a full-shape replacement for a refused broadcast shape.
    synthetic += sum(s.kind == "sub_loss_exact" for s in everything)
    synthetic += sum(s.kind == "div2_exact" for s in everything)
    synthetic += sum(s.kind == "mul_mask_exact" for s in everything)
    # Captured FP32 routes, including speed-selected S16 overrides, are outside
    # the generic backend coverage report. Explicitly retargeted nodes had
    # already counted as generic coverage before switching to their accurate
    # FP32 route, so do not subtract those nodes twice.
    synthetic += sum(len(s.nodes) for s in everything if s.kind == "fp32_binary")
    synthetic -= sum(
        len(s.nodes) for s in everything if s.kind == "fp32_binary" and s.prefer_fp32
    )
    # Singleton scalar divisions fold to guarded host constants, not AX models.
    synthetic += sum(s.kind == "algebraic_constant" for s in everything)
    assert covered + nonemittable - synthetic == report_covered
    unsafe = [s for s in everything if s.unsafe]
    # signed Reshapes take the Reshape -> Identity templates, so none is unsafe
    assert not any(s.kind == "reshape" for s in unsafe)
    assert len({n for s in segs for n in s.nodes}) == covered - len(
        {n for s in unsafe for n in s.nodes}
    )


@needs_step
def test_plan_materializes_live_broadcast_binary_operands():
    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segs, _ = sr.build_plan(model, sr.load_records(), calib)
    broadcast = [s for s in segs if s.output_shape]
    # Two decomposed residual/update broadcasts now use the fixed x128 binary
    # template for their dequantized float boundary as well.
    broadcast = [s for s in broadcast if "constant from" not in s.detail]
    assert len(broadcast) >= 44
    assert all(s.name == "Sub_24" or s.input_shapes[-1] == (1,) for s in broadcast)
    assert all(s.output_shape == s.input_shapes[0] for s in broadcast)


def test_emission_cache_reuses_a_validated_segment(tmp_path):
    calls = 0

    def emit():
        nonlocal calls
        calls += 1
        return onnx.helper.make_model(onnx.helper.make_graph([], "cached", [], []))

    segment = sr.Segment(
        "cached_segment",
        "test",
        ["cached_node"],
        [],
        [],
        "test",
        emit,
    )
    first, _ = sr.drop_unemittable([segment], {}, str(tmp_path))
    second, _ = sr.drop_unemittable([segment], {}, str(tmp_path))
    assert first and second
    assert calls == 1


def test_algebraic_identity_is_kept_without_model_emission():
    segment = sr.Segment(
        "ones_mul",
        "algebraic_identity",
        ["ones_mul"],
        ["x"],
        ["y"],
        "all-ones Mul is an identity",
        lambda: None,
    )
    kept, blobs = sr.drop_unemittable([segment], {})
    assert kept == [segment]
    assert blobs == {}


def _step_add35_s16_override():
    path = os.path.join(
        HERE, "..", "scripts", "axera", "fixtures", "binary_op_precision", "index.json"
    )
    with open(path) as stream:
        entry = next(
            item
            for item in json.load(stream)
            if item.get("source", "").startswith("Pulsar2 7.0-lite S16")
        )
    return {
        "layer_precision": entry["precision"],
        "scales": dict(zip(("x", "z", "y"), entry["scales"])),
        "zero_points": dict(zip(("x", "y", "z"), entry["zero_points"])),
    }


def _step_mul_s16_override(node_name):
    path = os.path.join(
        HERE, "..", "scripts", "axera", "fixtures", "binary_op_precision", "index.json"
    )
    with open(path) as stream:
        entry = next(
            item
            for item in json.load(stream)
            if item.get("file", "").startswith(f"mul_step_{node_name.lower()}_")
        )
    return {
        "layer_precision": entry["precision"],
        "scales": dict(zip(("x", "z", "y"), entry["scales"])),
        "zero_points": dict(zip(("x", "y", "z"), entry["zero_points"])),
    }


@needs_step
def test_resnet_native_templates_reduce_host_fallbacks():
    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    records = sr.load_records()
    overrides = sr.load_step_precision_overrides(model, records, calib)
    segments, host = sr.build_plan(
        model,
        records,
        calib,
        precision_overrides=overrides,
    )
    selected = {s.name: s for s in segments if s.kind == "binary_precision"}
    native_binary = {
        s.name for s in segments if s.kind in ("binary_precision", "fp32_binary")
    }
    # Some exact S16 candidates are deliberately routed through their
    # AX8850-profiled faster FP32 templates; neither route may leave them on
    # the host.
    assert set(overrides) <= native_binary
    assert not set(overrides) & host.keys()
    assert host == {}
    add976 = next(s for s in segments if s.name == "Add_976")
    assert add976.kind == "fp32_binary"
    assert add976.prefer_fp32
    assert add976.quantize_device_io
    assert len(add976.in_q) == len(add976.inputs) == 2
    assert len(add976.out_q) == len(add976.outputs) == 1
    by_name = {s.name: s for s in segments}
    for name in ("Mul_477", "Mul_703", "Mul_715", "Mul_785", "Mul_989"):
        assert by_name[name].kind == "fp32_binary"
        assert by_name[name].quantize_device_io
    assert not {"Sub_446", "Sub_449"} & host.keys()
    assert {"Greater_444", "Less_447"} <= {
        s.name for s in segments if s.kind == "compare_complement"
    }
    folded = {s.name: s for s in segments if s.kind == "algebraic_constant"}
    assert {"Div_18", "Div_26"} <= folded.keys()
    for segment in selected.values():
        for tensor, q in zip(
            (*segment.inputs, *segment.outputs), (*segment.in_q, *segment.out_q)
        ):
            lo, hi = calib["ranges"][tensor]
            limit = q[0] * 32767
            assert lo >= -limit - q[0]
            assert hi <= limit + q[0]


def test_fp32_capture_promotes_scalar_broadcast_input_to_length_one():
    item = {
        "input_shapes": [[], [512]],
        "output_shape": [512],
    }
    assert fp32_capture.template_input_shapes(item) == [[1], [512]]


def test_fp32_capture_merge_preserves_shared_signature_nodes():
    previous = {
        "source_nodes": ["Mul_703"],
        "prefer_fp32_nodes": ["Mul_703"],
    }
    current = {
        "source_nodes": ["Mul_705"],
        "prefer_fp32_nodes": [],
    }
    merged = fp32_capture.merge_entry(previous, current)
    assert merged["source_nodes"] == ["Mul_703", "Mul_705"]
    assert merged["prefer_fp32_nodes"] == ["Mul_703"]


def test_singleton_scalar_div_is_folded_and_guarded():
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Div", ["numerator", "batch"], ["y"], name="div")],
            "singleton_div",
            [onnx.helper.make_tensor_value_info("batch", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
            initializer=[
                numpy_helper.from_array(np.array([-0.5], np.float32), "numerator")
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    segment = sr._singleton_scalar_div_fold(
        {
            "name": "div",
            "op": "Div",
            "inputs": ["numerator", "batch"],
            "outputs": ["y"],
            "attrs": {"constant_input": 0},
        },
        {"ranges": {"batch": [16.0, 16.0]}},
        {"numerator": np.array([-0.5], np.float32)},
    )
    assert segment is not None
    runner = sr.StepRunner(model, [segment])
    result, _ = runner.run({"batch": np.array([16.0], np.float32)}, mode="npu")
    np.testing.assert_array_equal(result["y"], [-0.03125])
    # Inputs outside the calibrated singleton retain exact ONNX/ORT behavior.
    result, _ = runner.run({"batch": np.array([8.0], np.float32)}, mode="npu")
    np.testing.assert_array_equal(result["y"], [-0.0625])


@pytest.mark.parametrize(
    "node_name",
    [
        "Mul_5",
        "Mul_11",
        "Mul_25",
        "Mul_33",
        "Mul_46",
        "Mul_68",
        "Mul_91",
        "Mul_113",
        "Mul_148",
        "Mul_170",
        "Mul_193",
        "Mul_215",
        "Mul_250",
        "Mul_272",
        "Mul_295",
        "Mul_317",
        "Mul_352",
        "Mul_374",
        "Mul_397",
        "Mul_419",
    ],
)
@needs_device
@needs_step
def test_resnet_s16_mul_range_matched_template_on_axcl_vm(node_name):
    import axcl_session

    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segments, _ = sr.build_plan(
        model,
        sr.load_records(),
        calib,
        kinds={"binary_precision"},
        precision_overrides={node_name: _step_mul_s16_override(node_name)},
    )
    segment = next(s for s in segments if s.name == node_name)
    left, right = segment.inputs
    shape = segment.input_shapes[0]
    ranges = calib["ranges"]
    rng = np.random.default_rng(508)
    if node_name == "Mul_5":
        env = {
            left: rng.uniform(*ranges[left], shape).astype(np.float32),
            right: rng.uniform(-6.5, -3.25, shape).astype(np.float32),
        }
    elif node_name == "Mul_11":
        labels = np.zeros(shape, dtype=np.float32)
        labels[np.arange(shape[0]), rng.integers(0, shape[1], shape[0])] = 1.0
        env = {
            left: labels,
            right: rng.uniform(-10.5, -0.98, shape).astype(np.float32),
        }
    elif node_name == "Mul_25":
        # Keep the paired products in the measured output range while
        # spanning the extreme input value pair seen by the real loss path.
        x = rng.uniform(0.001, 0.375055, shape).astype(np.float32)
        product = rng.uniform(-0.0312495, 0.0117205, shape).astype(np.float32)
        z = product / x
        x[0, 0] = np.float32(0.031249450519680977 / 1776.24951171875)
        z[0, 0] = np.float32(-1776.24951171875)
        env = {left: x, right: z}
    elif node_name == "Mul_33":
        x = rng.uniform(0.01, 0.0385606, shape).astype(np.float32)
        product = rng.uniform(-0.0045767, 0.0009436, shape).astype(np.float32)
        z = product / x
        x[0, 0] = np.float32(5.278477692627348e-5)
        z[0, 0] = np.float32(-4.416831016540527)
        env = {left: x, right: z}
    else:
        x = rng.uniform(*ranges[left], shape).astype(np.float32)
        z = rng.integers(0, 2, size=shape).astype(np.float32)
        ylo, yhi = ranges[segment.outputs[0]]
        z[(x < ylo) | (x > yhi)] = 0.0
        x[0, 0, 0, 0] = ylo
        z[0, 0, 0, 0] = 1.0
        x[0, 0, 0, 1] = yhi
        z[0, 0, 0, 1] = 1.0
        x[0, 0, 0, 2] = ranges[left][0]
        z[0, 0, 0, 2] = 0.0
        x[0, 0, 0, 3] = ranges[left][1]
        z[0, 0, 0, 3] = 0.0
        env = {left: x, right: z}
    runner = sr.StepRunner(model, [segment])
    simulated = runner._sim(segment, env)[0]
    with axcl_session.AXSession(
        subdir=f"resnet_{node_name.lower()}_matched_s16"
    ) as session:
        runner.session = session
        actual = runner._device(segment, env)[0]
    max_lsb = float(np.abs(actual - simulated).max() / segment.out_q[0][0])
    assert max_lsb <= 2.01


@pytest.mark.parametrize(
    "node_name",
    [
        "Mul_485",
        "Mul_499",
        "Mul_555",
        "Mul_569",
        "Mul_583",
        "Mul_625",
        "Mul_639",
        "Mul_653",
        "Mul_695",
        "Mul_709",
        "Mul_723",
        "Mul_765",
    ],
)
@needs_device
@needs_step
def test_resnet_lr_broadcast_s16_template_on_axcl_vm(node_name):
    import axcl_session

    model = sr.load_step()
    records = sr.load_records()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    overrides = sr.load_step_precision_overrides(model, records, calib)
    assert node_name in overrides
    segments, _ = sr.build_plan(
        model,
        records,
        calib,
        kinds={"binary_precision"},
        precision_overrides={node_name: overrides[node_name]},
    )
    segment = next(s for s in segments if s.name == node_name)
    assert segment.input_transforms.get("lr") is not None
    rng = np.random.default_rng(985 + int(node_name.split("_")[1]))
    shape = segment.input_shapes[0]
    tensor = segment.inputs[1]
    env = {
        "lr": np.array([1e-4], dtype=np.float32),
        tensor: rng.uniform(*calib["ranges"][tensor], shape).astype(np.float32),
    }
    runner = sr.StepRunner(model, [segment])
    simulated = runner._sim(segment, env)[0]
    with axcl_session.AXSession(subdir=f"resnet_{node_name.lower()}_lr_s16") as session:
        runner.session = session
        actual = runner._device(segment, env)[0]
    max_lsb = float(np.abs(actual - simulated).max() / segment.out_q[0][0])
    assert max_lsb <= 2.01


@pytest.mark.parametrize(
    "node_name", ["Mul_793", "Mul_779", "Mul_863", "Mul_930", "Mul_997"]
)
@needs_device
@needs_step
def test_resnet_lr_vector_s16_template_on_axcl_vm(node_name):
    import axcl_session

    model = sr.load_step()
    records = sr.load_records()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    overrides = sr.load_step_precision_overrides(model, records, calib)
    assert node_name in overrides
    segments, _ = sr.build_plan(
        model,
        records,
        calib,
        kinds={"binary_precision"},
        precision_overrides={node_name: overrides[node_name]},
    )
    segment = next(s for s in segments if s.name == node_name)
    assert segment.output_shape == ()
    assert segment.input_transforms.get("lr") is None
    rng = np.random.default_rng(1985 + int(node_name.split("_")[1]))
    tensor = segment.inputs[1]
    value_shapes = {
        value.name: tuple(int(d.dim_value) for d in value.type.tensor_type.shape.dim)
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    assert segment.input_transforms.get(tensor) is not None
    env = {
        "lr": np.array([1e-4], dtype=np.float32),
        tensor: rng.uniform(*calib["ranges"][tensor], value_shapes[tensor]).astype(
            np.float32
        ),
    }
    runner = sr.StepRunner(model, [segment])
    simulated = runner._sim(segment, env)[0]
    with axcl_session.AXSession(
        subdir=f"resnet_{node_name.lower()}_lr_vector_s16"
    ) as session:
        runner.session = session
        actual = runner._device(segment, env)[0]
    assert actual.ndim == 1
    assert simulated.shape == actual.shape
    max_lsb = float(np.abs(actual - simulated).max() / segment.out_q[0][0])
    assert max_lsb <= 2.01


@needs_device
@needs_step
def test_resnet_sub32_s16_broadcast_template_on_axcl_vm():
    import axcl_session

    model = sr.load_step()
    records = sr.load_records()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    overrides = sr.load_step_precision_overrides(model, records, calib)
    segments, host = sr.build_plan(
        model,
        records,
        calib,
        kinds={"binary_precision"},
        precision_overrides={"Sub_32": overrides["Sub_32"]},
    )
    assert "Sub_32" not in host
    segment = next(s for s in segments if s.name == "Sub_32")
    assert segment.input_transforms.get(segment.inputs[1]) is not None
    rng = np.random.default_rng(2032)
    env = {
        segment.inputs[0]: rng.uniform(
            *calib["ranges"][segment.inputs[0]], (16, 1000)
        ).astype(np.float32),
        segment.inputs[1]: rng.uniform(
            *calib["ranges"][segment.inputs[1]], (16, 1)
        ).astype(np.float32),
    }
    runner = sr.StepRunner(model, [segment])
    simulated = runner._sim(segment, env)[0]
    with axcl_session.AXSession(subdir="resnet_sub32_s16_broadcast") as session:
        runner.session = session
        actual = runner._device(segment, env)[0]
    assert actual.shape == (16, 1000)
    max_lsb = float(np.abs(actual - simulated).max() / segment.out_q[0][0])
    assert max_lsb <= 2.01


@pytest.mark.parametrize("node_name", ["Greater_444", "Less_447"])
@needs_device
@needs_step
def test_resnet_comparison_complement_template_on_axcl_vm(node_name):
    import axcl_session

    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segments, host = sr.build_plan(model, sr.load_records(), calib)
    segment = next(s for s in segments if s.name == node_name)
    assert segment.kind == "compare_complement"
    assert not {"Sub_446", "Sub_449"} & host.keys()
    x_name, z_name = model.graph.node[
        next(i for i, node in enumerate(model.graph.node) if node.name == node_name)
    ].input
    shape = (1024, 9, 3136)
    x = np.zeros(shape, dtype=np.float32)
    z = np.zeros(shape, dtype=np.float32)
    x.reshape(-1)[:6] = [0.0, 1.0, -1.0, 2.0, -2.0, 0.0]
    z.reshape(-1)[:6] = [0.0, 0.0, 0.0, 2.0, -2.0, 1.0]
    env = {x_name: x, z_name: z}
    runner = sr.StepRunner(model, [segment])
    with axcl_session.AXSession(
        subdir=f"resnet_{node_name.lower()}_complement"
    ) as session:
        runner.session = session
        actual = runner._device(segment, env)[0].reshape(-1)[:6]
    expected = (
        x.reshape(-1)[:6] <= z.reshape(-1)[:6]
        if node_name == "Greater_444"
        else x.reshape(-1)[:6] >= z.reshape(-1)[:6]
    )
    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_comparison_complement_nan_guard_uses_original_host_semantics():
    from onnx import numpy_helper

    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Greater", ["x", "z"], ["cmp"], name="greater"),
                onnx.helper.make_node(
                    "Cast", ["cmp"], ["cast"], name="cast", to=onnx.TensorProto.FLOAT
                ),
                onnx.helper.make_node("Sub", ["one", "cast"], ["y"], name="sub"),
            ],
            "nan_guard",
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2]),
                onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [2]),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
            [numpy_helper.from_array(np.array(1.0, np.float32), name="one")],
            value_info=[
                onnx.helper.make_tensor_value_info("cmp", onnx.TensorProto.BOOL, [2]),
                onnx.helper.make_tensor_value_info("cast", onnx.TensorProto.FLOAT, [2]),
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    segment = sr.Segment(
        "greater",
        "compare_complement",
        ["greater", "cast", "sub"],
        ["z", "x"],
        ["y"],
        "test ordered comparison",
        lambda: None,
        nan_guard=True,
    )
    runner = sr.StepRunner(model, [segment])
    outputs, stats = runner.run(
        {
            "x": np.array([np.nan, 2.0], np.float32),
            "z": np.array([0.0, 1.0], np.float32),
        },
        mode="npu",
        check=False,
    )
    np.testing.assert_array_equal(outputs["y"], [1.0, 0.0])
    assert stats[0].kind == "host_nan_guard"


@needs_step
def test_resnet_add35_selects_range_matched_s16_template():
    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segments, _ = sr.build_plan(
        model,
        sr.load_records(),
        calib,
        kinds={"binary_precision"},
        precision_overrides={"Add_35": _step_add35_s16_override()},
    )
    assert len(segments) == 1
    segment = segments[0]
    assert segment.name == "Add_35"
    assert segment.kind == "binary_precision"
    assert segment.in_q == [
        (9.536720995129144e-07, 0, True, 16),
        (6.983623279666062e-08, 0, True, 16),
    ]


@needs_device
@needs_step
def test_resnet_add35_range_matched_s16_template_on_axcl_vm():
    import axcl_session

    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segments, _ = sr.build_plan(
        model,
        sr.load_records(),
        calib,
        kinds={"binary_precision"},
        precision_overrides={"Add_35": _step_add35_s16_override()},
    )
    segment = segments[0]
    ranges = calib["ranges"]
    rng = np.random.default_rng(350)
    env = {
        tensor: rng.uniform(*ranges[tensor], (16, 1000)).astype(np.float32)
        for tensor in ("distill__mul_63", "distill__div_108")
    }
    runner = sr.StepRunner(model, [segment])
    simulated = runner._sim(segment, env)[0]
    with axcl_session.AXSession(subdir="resnet_add35_matched_s16") as session:
        runner.session = session
        actual = runner._device(segment, env)[0]
    max_lsb = float(np.abs(actual - simulated).max() / segment.out_q[0][0])
    assert max_lsb <= 2.01


def test_fp32_binary_fixture_is_exact_shape_and_unquantized():
    op = "Add"
    path = os.path.join(sr.FP32_BINARY_FIXTURES, "add_16x1000.axmodel.gz")
    with gzip.open(path, "rb") as f:
        model = onnx.load_model_from_string(f.read())
    assert [
        tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
        for i in model.graph.input
    ] == [(16, 1000), (16, 1000)]
    assert len(model.graph.output) == 1
    assert [n.op_type for n in model.graph.node] == ["neu mode"]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node(op, ["x", "z"], ["y"], name="n")],
        "g",
        [
            onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, [16, 1000])
            for t in ("x", "z")
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [16, 1000])],
    )
    segment = sr._fp32_binary_segment_for(
        {"op": op, "name": "n", "inputs": ["x", "z"], "outputs": ["y"]},
        onnx.helper.make_model(graph),
    )
    assert segment is not None and not segment.in_q and not segment.out_q


@needs_device
def test_profiled_fast_broadcast_mul_reorders_scalar_operand_on_axcl_vm():
    import axcl_session

    shape = [128, 64, 3, 3]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["scale", "x"], ["y"], name="binary")],
        "profiled_broadcast_mul",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        initializer=[
            onnx.numpy_helper.from_array(np.array([0.73], np.float32), "scale")
        ],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    segment = sr._fp32_binary_segment_for(
        {
            "op": "Mul",
            "name": "binary",
            "inputs": ["scale", "x"],
            "outputs": ["y"],
        },
        model,
    )
    assert segment is not None and segment.profiled_faster
    assert segment.inputs == ["x", "scale"]
    assert segment.constant_inputs == ["scale"]

    rng = np.random.default_rng(12864)
    x = rng.normal(size=shape).astype(np.float32)
    with axcl_session.AXSession(subdir="step_runner_fp32_broadcast_mul_test") as sess:
        runner = sr.StepRunner(model, [segment])
        runner.session = sess
        actual = runner._device(segment, {"x": x})[0]
    np.testing.assert_array_equal(actual, x * np.float32(0.73))


@needs_device
def test_fp32_binary_template_matches_float_on_axcl_vm():
    import axcl_session

    shape = [16, 1000]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "z"], ["y"], name="binary")],
        "fp32",
        [
            onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, shape)
            for t in ("x", "z")
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    seg = sr._fp32_binary_segment_for(
        {"op": "Add", "name": "binary", "inputs": ["x", "z"], "outputs": ["y"]},
        model,
    )
    assert seg is not None
    rng = np.random.default_rng(0)
    x = rng.normal(size=shape).astype(np.float32)
    z = rng.uniform(0.5, 1.5, size=shape).astype(np.float32)
    with axcl_session.AXSession(subdir="step_runner_fp32_binary_test") as sess:
        runner = sr.StepRunner(model, [seg], sess)
        actual = runner._device(seg, {"x": x, "z": z})[0]
    expected = np.add(x, z)
    np.testing.assert_array_equal(actual, expected)


@needs_device
@needs_step
def test_fp32_critical_elementwise_probes_match_step_simulation_on_axcl_vm():
    """The FP32 probes expose accurate output even where the faster route is
    still unresolved; do not make them preferred planner routes by default."""
    import axcl_session

    model = sr.load_step()
    records = {record["name"]: record for record in sr.load_records()}
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    qtable = calib["tensors"]
    segments = []
    for name in ("Mul_705", "Add_730"):
        segment = sr._fp32_binary_segment_for(records[name], model)
        assert segment is not None
        assert not segment.prefer_fp32
        segment.in_q = [
            sr.qparams_of(calib, tensor) if tensor in qtable else None
            for tensor in segment.inputs
        ]
        segment.out_q = [
            sr.qparams_of(calib, tensor) if tensor in qtable else None
            for tensor in segment.outputs
        ]
        segment.quantize_device_io = True
        segments.append(segment)

    reference = sr.load_reference()
    with axcl_session.AXSession(subdir="critical_elementwise_fp32_probe") as session:
        runner = sr.StepRunner(model, segments, session, health_every=1)
        outputs, stats = runner.run(reference["feeds"], "npu")
    assert [stat.segment for stat in stats] == ["Mul_705", "Add_730"]
    assert all(sr.segment_passed(vars(stat)) for stat in stats)
    assert float(np.ravel(outputs["distill__add_27"])[0]) == pytest.approx(
        float(np.ravel(reference["ref"]["distill__add_27"])[0]), abs=3e-6
    )


def _mask_mul_model():
    shape = [16, 1000]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["x", "z"], ["y"], name="mask_mul")],
        "mask_mul",
        [
            onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, shape)
            for t in ("x", "z")
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model


def _mask_mul_calibration():
    return {
        "tensors": {
            "x": {"scale": 0.00012254902685526758, "zero_point": 255, "signed": False},
            "z": {"scale": 0.003921568859368563, "zero_point": 0, "signed": False},
            "y": {"scale": 0.00012254902685526758, "zero_point": 255, "signed": False},
        }
    }


def test_exact_mask_mul_template_requires_its_measured_calibration():
    model = _mask_mul_model()
    rec = {
        "op": "Mul",
        "name": "mask_mul",
        "attrs": {"form": "same_shape"},
        "inputs": ["x", "z"],
        "outputs": ["y"],
    }
    calib = _mask_mul_calibration()
    segment = sr._exact_mask_mul_segment_for(rec, calib, model)
    assert segment is not None and segment.kind == "mul_mask_exact"
    assert not sr._exact_mask_mul_segment_for(
        rec,
        {
            "tensors": {
                **calib["tensors"],
                "y": {**calib["tensors"]["y"], "scale": 0.001},
            }
        },
        model,
    )


def _loss_sub_model():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sub", ["x", "z"], ["y"], name="loss_sub")],
        "loss_sub",
        [
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [16, 1000]),
            onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [16, 1]),
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [16, 1000])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model


def _loss_sub_calibration():
    return {
        "tensors": {
            "x": {"scale": 6.96580696105957, "zero_point": 255, "signed": False},
            "z": {
                "scale": 0.00012254902685526758,
                "zero_point": 255,
                "signed": False,
            },
            "y": {"scale": 6.96580696105957, "zero_point": 255, "signed": False},
        }
    }


def test_exact_loss_sub_template_requires_its_measured_calibration():
    rec = {
        "op": "Sub",
        "name": "loss_sub",
        "inputs": ["x", "z"],
        "outputs": ["y"],
    }
    model = _loss_sub_model()
    calibration = _loss_sub_calibration()
    segment = sr._exact_loss_sub_segment_for(rec, calibration, model)
    assert segment is not None and segment.kind == "sub_loss_exact"
    assert segment.output_shape == (16, 1000)
    changed = {
        "tensors": {
            **calibration["tensors"],
            "y": {**calibration["tensors"]["y"], "zero_point": 128},
        }
    }
    assert sr._exact_loss_sub_segment_for(rec, changed, model) is None


def _div2_model():
    shape = [16, 1000]
    two = numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), "two")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Div", ["x", "two"], ["y"], name="div2")],
        "div2",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        [two],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model


def _div2_calibration(zp):
    scales = {
        98: (0.05171579495072365, 0.025857897475361824),
        101: (0.0531538687646389, 0.02657693438231945),
        211: (2.16483895201236e-05, 1.08241947600618e-05),
    }
    sx, sy = scales[zp]
    return {
        "tensors": {
            "x": {"scale": sx, "zero_point": zp, "signed": False},
            "y": {"scale": sy, "zero_point": zp, "signed": False},
        }
    }


@pytest.mark.parametrize("zp", [98, 101, 211])
def test_exact_div2_template_requires_its_measured_calibration(zp):
    model = _div2_model()
    rec = {
        "op": "Div",
        "name": "div2",
        "attrs": {"form": "const", "constant_input": 1},
        "inputs": ["x", "two"],
        "outputs": ["y"],
    }
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    segment = sr._exact_div2_segment_for(rec, _div2_calibration(zp), model, inits)
    assert segment is not None and segment.kind == "div2_exact"
    assert segment.inputs == ["x"] and segment.outputs == ["y"]


@needs_device
def test_exact_mask_mul_template_matches_runner_simulation_on_axcl_vm():
    import axcl_session

    model = _mask_mul_model()
    rec = {
        "op": "Mul",
        "name": "mask_mul",
        "attrs": {"form": "same_shape"},
        "inputs": ["x", "z"],
        "outputs": ["y"],
    }
    segment = sr._exact_mask_mul_segment_for(rec, _mask_mul_calibration(), model)
    assert segment is not None
    x = np.linspace(-0.03125, 0.0, 16000, dtype=np.float32).reshape(16, 1000)
    z = np.zeros((16, 1000), dtype=np.float32)
    z[:, ::2] = 1.0
    runner = sr.StepRunner(model, [segment])
    simulated = runner._sim(segment, {"x": x, "z": z})[0]
    with axcl_session.AXSession(subdir="step_runner_exact_mask_test") as sess:
        runner.session = sess
        actual = runner._device(segment, {"x": x, "z": z})[0]
    np.testing.assert_array_equal(actual, simulated)


@needs_device
def test_exact_loss_sub_template_matches_runner_simulation_on_axcl_vm():
    import axcl_session

    model = _loss_sub_model()
    rec = {"op": "Sub", "name": "loss_sub", "inputs": ["x", "z"], "outputs": ["y"]}
    segment = sr._exact_loss_sub_segment_for(rec, _loss_sub_calibration(), model)
    assert segment is not None
    x = np.linspace(-6.96580696105957 * 255, 0.0, 16000, dtype=np.float32).reshape(
        16, 1000
    )
    z = np.linspace(-0.03125, 0.0, 16, dtype=np.float32).reshape(16, 1)
    runner = sr.StepRunner(model, [segment])
    simulated = runner._sim(segment, {"x": x, "z": z})[0]
    with axcl_session.AXSession(subdir="step_runner_exact_loss_sub_test") as sess:
        runner.session = sess
        actual = runner._device(segment, {"x": x, "z": z})[0]
    np.testing.assert_array_equal(actual, simulated)


@needs_device
@pytest.mark.parametrize("zp", [98, 101, 211])
def test_exact_div2_template_matches_runner_simulation_on_axcl_vm(zp):
    import axcl_session

    model = _div2_model()
    rec = {
        "op": "Div",
        "name": "div2",
        "attrs": {"form": "const", "constant_input": 1},
        "inputs": ["x", "two"],
        "outputs": ["y"],
    }
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    segment = sr._exact_div2_segment_for(rec, _div2_calibration(zp), model, inits)
    assert segment is not None
    sx = _div2_calibration(zp)["tensors"]["x"]["scale"]
    x = np.linspace(-zp * sx, (255 - zp) * sx, 16000, dtype=np.float32).reshape(
        16, 1000
    )
    runner = sr.StepRunner(model, [segment])
    simulated = runner._sim(segment, {"x": x})[0]
    with axcl_session.AXSession(subdir=f"step_runner_div2_z{zp}_test") as sess:
        runner.session = sess
        actual = runner._device(segment, {"x": x})[0]
    np.testing.assert_array_equal(actual, simulated)


@needs_step
def test_float_mode_reproduces_the_reference_step():
    model = sr.load_step()
    ref = sr.load_reference()
    outs, _ = sr.StepRunner(model, []).run(ref["feeds"], "float")
    loss = float(np.ravel(outs["distill__add_27"])[0])
    assert loss == pytest.approx(
        float(np.ravel(ref["ref"]["distill__add_27"])[0]), rel=1e-5
    )


@needs_device
def test_session_health_check_on_device():
    import axcl_session

    with axcl_session.AXSession() as s:
        assert axcl_session.health_check(s) <= 1.01


@needs_device
@needs_step
def test_one_segment_of_each_kind_matches_its_simulation_on_device():
    import axcl_session

    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segs, _ = sr.build_plan(model, sr.load_records(), calib)
    first: dict[str, sr.Segment] = {}
    for s in segs:
        first.setdefault(s.kind, s)
    ref = sr.load_reference()
    with axcl_session.AXSession() as sess:
        runner = sr.StepRunner(model, list(first.values()), sess, health_every=1)
        _, stats = runner.run(ref["feeds"], "npu")
    assert {st.kind for st in stats} == set(first)
    for st in stats:
        assert sr.segment_passed(vars(st)), st


@needs_device
@needs_step
def test_recentered_add_sub_segment_matches_simulation_on_axcl_vm():
    """Arbitrary residual zero points can use the native x128 binary frame."""
    import axcl_session

    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segments, _ = sr.build_plan(model, sr.load_records(), calib)
    seg = next(s for s in segments if "recentered from" in s.detail)
    reference = sr.load_reference()
    with axcl_session.AXSession(subdir="recentered_binary") as session:
        _, stats = sr.StepRunner(model, [seg], session, health_every=0).run(
            reference["feeds"], "npu"
        )
    assert len(stats) == 1
    # Reusing a fixed zero-point binary frame for arbitrary calibrated
    # residual zero-points is a bounded quantized approximation; it must
    # execute natively and remain close to the float operation.
    assert not stats[0].error, stats[0]
    assert stats[0].max_lsb <= 16.0, stats[0]
    assert stats[0].float_rel <= 0.03, stats[0]


@needs_device
@needs_step
def test_shape_expanded_add_segment_runs_on_axcl_vm():
    """The scalar decomposed Add uses a replicated native lane template."""
    import axcl_session

    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segments, _ = sr.build_plan(model, sr.load_records(), calib)
    seg = next(s for s in segments if "shape-expanded" in s.detail)
    reference = sr.load_reference()
    with axcl_session.AXSession(subdir="shape_expanded_binary") as session:
        _, stats = sr.StepRunner(model, [seg], session, health_every=0).run(
            reference["feeds"], "npu"
        )
    assert len(stats) == 1
    assert not stats[0].error, stats[0]
    assert stats[0].max_lsb <= 2.01, stats[0]


@needs_device
@needs_step
def test_constant_mul_fixed_frame_runs_on_axcl_vm():
    """Unsigned constant Mul uses staged data with the native x128 frame."""
    import axcl_session

    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segments, _ = sr.build_plan(model, sr.load_records(), calib)
    seg = next(s for s in segments if "constant from" in s.detail)
    reference = sr.load_reference()
    with axcl_session.AXSession(subdir="constant_mul_fixed_frame") as session:
        _, stats = sr.StepRunner(model, [seg], session, health_every=0).run(
            reference["feeds"], "npu"
        )
    assert len(stats) == 1
    assert not stats[0].error, stats[0]
    assert stats[0].max_lsb <= 16.0, stats[0]


@needs_device
@needs_step
def test_constant_mul_broadcast_output_shape_runs_on_axcl_vm():
    """A constant broadcast uses the validated full output-shape template."""
    import axcl_session

    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segments, _ = sr.build_plan(model, sr.load_records(), calib)
    seg = next(s for s in segments if s.name == "Mul_43")
    assert seg.output_shape == (16, 512, 7, 7)
    reference = sr.load_reference()
    with axcl_session.AXSession(subdir="constant_mul_broadcast_output") as session:
        _, stats = sr.StepRunner(model, [seg], session, health_every=0).run(
            reference["feeds"], "npu"
        )
    assert len(stats) == 1
    assert not stats[0].error, stats[0]
    # This calibrated boundary is a fixed-frame approximation; retain the
    # native path only within its measured device error budget.
    assert stats[0].max_lsb <= 40.0, stats[0]
    assert stats[0].float_rel <= 0.03, stats[0]


@needs_device
@needs_step
def test_mask_mul_tiled_native_shapes_runs_on_axcl_vm():
    """Mask multiplies can use each validated native spatial tile."""
    import axcl_session

    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segments, _ = sr.build_plan(model, sr.load_records(), calib)
    selected = [
        s
        for s in segments
        if s.name
        in {
            "Mul_58",
            "Mul_125",
            "Mul_160",
            "Mul_227",
            "Mul_262",
            "Mul_329",
            "Mul_364",
            "Mul_450",
            "Mul_451",
            "Mul_464",
        }
    ]
    assert len(selected) == 10
    reference = sr.load_reference()
    with axcl_session.AXSession(subdir="mask_mul_tiled_native_shapes") as session:
        _, stats = sr.StepRunner(model, selected, session, health_every=0).run(
            reference["feeds"], "npu"
        )
    assert len(stats) == 10
    for stat in stats:
        assert not stat.error, stat
        assert stat.max_lsb <= 2.01, stat


@needs_device
@needs_step
def test_resnet18_training_graph_runs_pulsar_free_on_axcl_vm(tmp_path):
    """Run the complete calibrated ResNet18 training graph on the AX8850."""
    import axcl_session

    model = sr.load_step()
    calibration = sr.axb.load_calibration(sr.STEP_CALIB)
    segments, _ = sr.build_plan(model, sr.load_records(), calibration)
    reference = sr.load_reference()
    with axcl_session.AXSession(subdir=f"resnet18_training_{tmp_path.name}") as session:
        runner = sr.StepRunner(
            model,
            segments,
            session,
            # The dedicated health-check test exercises this probe.  Keeping
            # it out of the long graph session avoids AXCL VM metadata reuse
            # masking a successful training-segment execution.
            health_every=0,
        )
        outputs, stats = runner.run(reference["feeds"], "npu")

    assert outputs
    assert stats
    assert {stat.kind for stat in stats} >= {
        "matmul_chain",
        "elementwise",
        "misc",
    }
    # This is a whole-graph transport smoke test.  The focused segment test
    # above keeps the strict <=2-LSB contract; a full training step also
    # contains known calibration/emitter outliers, which must not turn a
    # successful AXCL execution into a false device failure.
    for stat in stats:
        assert not stat.error, stat


@needs_device
def test_tinygrad_ax_device_runs_a_relu_and_a_matmul_chain():
    tinygrad = pytest.importorskip("tinygrad")
    import tinygrad_ax_backend as axb
    from tinygrad.device import Buffer, Device

    axb.register_ax_device()
    dev = Device["AX"]
    classes = axb.tinygrad_classes()
    try:
        # Relu: an AXCompiler request (template + ElementwiseScaleEdit)
        s = 0.01
        key = axb.TemplateKey("Relu", ((16, 512, 7, 7),), calibration_class="x0,y0")
        src = axb.build_request(key, [axb.ElementwiseScaleEdit({"x": s, "y": s})])
        prog = classes["AXProgram"](dev, classes["AXCompiler"]().compile(src))
        x = np.random.default_rng(0).uniform(0, 2, (16, 512, 7, 7)).astype(np.float32)
        xb = Buffer("AX", x.size, tinygrad.dtypes.float32, initial_value=x.tobytes())
        yb = Buffer("AX", x.size, tinygrad.dtypes.float32).allocate()
        prog(yb._buf, xb._buf, wait=True)
        want = np.clip(np.rint(x / np.float32(s)), 0, 255) * np.float32(s)
        assert np.abs(yb.numpy() - want.ravel()).max() <= s * 1.01

        # a live-operand MatMul chain: the step's fc dX (matmul_record_emit)
        if _HAVE_STEP:
            model = sr.load_step()
            calib = sr.axb.load_calibration(sr.STEP_CALIB)
            segs, _ = sr.build_plan(
                model, sr.load_records(), calib, kinds={"matmul_chain"}
            )
            seg = next(g for g in segs if g.name == "MatMul_36")
            prog = classes["AXProgram"](dev, seg.emit().SerializeToString())
            rng = np.random.default_rng(1)
            ins = [
                rng.uniform(-0.02, 0.02, sp.shape).astype(np.float32)
                for sp in prog.model.inputs
            ]
            bufs = [
                Buffer("AX", a.size, tinygrad.dtypes.float32, initial_value=a.tobytes())
                for a in ins
            ]
            out = prog.model.outputs[0]
            ob = Buffer(
                "AX", int(np.prod(out.shape)), tinygrad.dtypes.float32
            ).allocate()
            prog(ob._buf, *[b._buf for b in bufs], wait=True)
            env = dict(zip(seg.inputs, ins))
            sim = sr.StepRunner(model, [seg])._sim(seg, env)[0]
            lsb = np.abs(ob.numpy() - sim.ravel()).max() / seg.out_q[0][0]
            assert lsb <= 2.01
    finally:
        axb.close_ax_session()


@needs_device
def test_onnx_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Exercise the replacement path on the AX8850, including its schedule."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import tinygrad_ax_backend as axb

    shape = numpy_helper.from_array(np.asarray([1, 1, 8, 16], dtype=np.int64), "shape")
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Reshape", ["x", "shape"], ["r"]),
                onnx.helper.make_node("Relu", ["r"], ["y"]),
            ],
            "onnx_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, [1, 8, 4, 4]
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, [1, 1, 8, 16]
                )
            ],
            [shape],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    schedule = tmp_path / "onnx_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule))
    rng = np.random.default_rng(1965)
    x = rng.uniform(-0.5, 0.5, (1, 8, 4, 4)).astype(np.float32)

    with axcl_session.AXSession() as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x])
        finally:
            session.unload(loaded)

    np.testing.assert_allclose(
        got, np.maximum(x.reshape(1, 1, 8, 16), 0.0), atol=0.02, rtol=0
    )


@needs_device
@pytest.mark.parametrize("zero_point", [0, 128])
def test_standalone_relu_uop_to_mcode_runs_on_axcl_vm(tmp_path, zero_point):
    """Run an unfused standalone ReLU UOp through the AXCL VM."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import elementwise_scale_emit as ew
    import tinygrad_ax_backend as axb

    shape = (16, 64, 56, 56)
    _, meta = ew.load_template("Relu", shape, {"x": zero_point, "y": zero_point})
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Relu", ["x"], ["y"])],
            "onnx_standalone_relu_to_uop_vm",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    schedule = tmp_path / f"standalone_relu_uop_z{zero_point}.schedule.json"
    axmodel = axb.compile_onnx(
        model,
        str(schedule),
        {"scales": meta["scales"], "zero_points": meta["zero_points"]},
    )
    rng = np.random.default_rng(1965)
    x = rng.uniform(-1.0, 1.0, shape).astype(np.float32)

    with axcl_session.AXSession(subdir=f"uop_relu_{tmp_path.name}") as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x])
        finally:
            session.unload(loaded)

    np.testing.assert_allclose(
        got, np.maximum(x, 0.0), atol=meta["scales"]["y"] * 2, rtol=0
    )


@needs_device
def test_onnx_transpose_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Run a verified real-shape Transpose through the replacement path."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import tinygrad_ax_backend as axb

    input_shape, perm = (16, 1, 256, 49), (0, 1, 3, 2)
    output_shape = tuple(input_shape[axis] for axis in perm)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Transpose", ["x"], ["y"], perm=list(perm))],
            "onnx_transpose_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, input_shape
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, output_shape
                )
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    schedule = tmp_path / "onnx_transpose_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule))
    x = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)

    with axcl_session.AXSession(subdir=f"transpose_{tmp_path.name}") as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x])
        finally:
            session.unload(loaded)

    np.testing.assert_array_equal(got, np.transpose(x, perm))


@needs_device
def test_onnx_matmul_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Run a calibrated standalone MatMul emitted from an imported ONNX UOp."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import matmul_record_emit as mre
    import tinygrad_ax_backend as axb

    a_shape, b_shape = (16, 1000), (1000, 512)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["x", "z"], ["y"])],
            "onnx_matmul_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, a_shape
                ),
                onnx.helper.make_tensor_value_info(
                    "z", onnx.TensorProto.FLOAT, b_shape
                ),
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, (16, 512)
                )
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, quant = mre.STANDALONE_MATMUL_TEMPLATES[(a_shape, b_shape)]
    old = mre.load_scales(os.path.join(mre.STEP_TEMPLATE_DIR, quant))
    names = list(old)
    scales = {
        "x": old[names[0]][0],
        "z": old[names[1]][0],
        "y": old[names[2]][0],
    }
    zero_points = {
        "x": old[names[0]][1],
        "z": old[names[1]][1],
        "y": old[names[2]][1],
    }
    calibration = {"scales": scales, "zero_points": zero_points}
    schedule = tmp_path / "onnx_matmul_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), calibration)
    rng = np.random.default_rng(1965)
    x = rng.uniform(-0.02, 0.02, a_shape).astype(np.float32)
    z = rng.uniform(-0.02, 0.02, b_shape).astype(np.float32)

    with axcl_session.AXSession(subdir=f"uop_matmul_{tmp_path.name}") as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, z])
        finally:
            session.unload(loaded)

    want = x @ z
    np.testing.assert_allclose(got, want, atol=scales["y"] * 1.5, rtol=0)


@needs_device
def test_onnx_gemm_beta_zero_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Run beta-zero Gemm after its bias-free MatMul canonicalization."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import matmul_record_emit as mre
    import tinygrad_ax_backend as axb

    a_shape, b_shape = (16, 1000), (1000, 512)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "Gemm",
                    ["x", "z", "b"],
                    ["y"],
                    alpha=1.0,
                    beta=0.0,
                    transA=0,
                    transB=0,
                )
            ],
            "onnx_gemm_beta_zero_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, a_shape
                ),
                onnx.helper.make_tensor_value_info(
                    "z", onnx.TensorProto.FLOAT, b_shape
                ),
                onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [512]),
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, (16, 512)
                )
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, quant = mre.STANDALONE_MATMUL_TEMPLATES[(a_shape, b_shape)]
    old = mre.load_scales(os.path.join(mre.STEP_TEMPLATE_DIR, quant))
    names = list(old)
    calibration = {
        "scales": {"x": old[names[0]][0], "z": old[names[1]][0], "y": old[names[2]][0]},
        "zero_points": {
            "x": old[names[0]][1],
            "z": old[names[1]][1],
            "y": old[names[2]][1],
        },
    }
    schedule = tmp_path / "onnx_gemm_beta_zero.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), calibration)
    rng = np.random.default_rng(1965)
    x = rng.uniform(-0.02, 0.02, a_shape).astype(np.float32)
    z = rng.uniform(-0.02, 0.02, b_shape).astype(np.float32)
    with axcl_session.AXSession(
        subdir=f"uop_gemm_beta_zero_{tmp_path.name}"
    ) as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, z])
        finally:
            session.unload(loaded)
    np.testing.assert_allclose(
        got, x @ z, atol=calibration["scales"]["y"] * 1.5, rtol=0
    )


@needs_device
@needs_step
def test_training_step_matmul_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Run one live-operand MatMul shape taken from the training step."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import tinygrad_ax_backend as axb

    step = sr.load_step()
    calibration = axb.load_calibration(sr.STEP_CALIB)
    records = sr.load_records()
    segments, _ = sr.build_plan(step, records, calibration, kinds={"matmul_chain"})
    segment = next(seg for seg in segments if seg.name == "MatMul_36")
    node = next(node for node in step.graph.node if node.name == "MatMul_36")
    if len(node.input) != 2 or len(node.output) != 1:
        pytest.skip("MatMul_36 is not a two-input training MatMul in this step")

    values = {
        value.name: tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        for value in (*step.graph.input, *step.graph.value_info, *step.graph.output)
    }
    a_shape, b_shape = (values[name] for name in node.input)
    output_shape = values[node.output[0]]
    if not a_shape or not b_shape or not output_shape:
        pytest.skip("MatMul_36 has no static shapes")

    # graph_generator's standalone MatMul contract deliberately uses x/z/y;
    # retain the training node's shapes and calibration, but normalize names.
    imported = onnx.NodeProto()
    imported.CopyFrom(node)
    imported.input[:] = ["x", "z"]
    imported.output[:] = ["y"]
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [imported],
            "training_matmul_36_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, a_shape
                ),
                onnx.helper.make_tensor_value_info(
                    "z", onnx.TensorProto.FLOAT, b_shape
                ),
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, output_shape
                )
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    if len(segment.in_q) != 2 or len(segment.out_q) != 1:
        pytest.skip("MatMul_36 calibration is not a two-input/single-output form")
    matmul_calibration = {
        "scales": {
            "x": segment.in_q[0][0],
            "z": segment.in_q[1][0],
            "y": segment.out_q[0][0],
        },
        "zero_points": {
            "x": segment.in_q[0][1],
            "z": segment.in_q[1][1],
            "y": segment.out_q[0][1],
        },
    }
    schedule = tmp_path / "training_matmul_36.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), matmul_calibration)
    rng = np.random.default_rng(1965)
    # Keep the product inside this training segment's uint8 output range;
    # the step calibration has a nonzero output zero point and saturates on
    # the wider standalone MatMul probe range.
    x = rng.uniform(-0.002, 0.002, a_shape).astype(np.float32)
    z = rng.uniform(-0.002, 0.002, b_shape).astype(np.float32)

    with axcl_session.AXSession(subdir=f"training_matmul_{tmp_path.name}") as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, z])
        finally:
            session.unload(loaded)

    np.testing.assert_allclose(
        got,
        x @ z,
        atol=matmul_calibration["scales"]["y"] * 2.0,
        rtol=0,
    )


@needs_device
def test_onnx_add_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Run a calibrated two-input Add emitted from an imported ONNX UOp."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import binary_op_scale_emit as bse
    import tinygrad_ax_backend as axb

    shape = (1, 64)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["x", "z"], ["y"])],
            "onnx_add_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, shape),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, meta = bse.load_template("Add", shape, {"x": 0, "y": 0, "z": 0})
    calibration = {
        "scales": meta["scales"],
        "zero_points": meta["zero_points"],
    }
    schedule = tmp_path / "onnx_add_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), calibration)
    rng = np.random.default_rng(1965)
    x = rng.uniform(0.0, 0.3, shape).astype(np.float32)
    z = rng.uniform(0.0, 0.3, shape).astype(np.float32)

    with axcl_session.AXSession(subdir=f"uop_add_{tmp_path.name}") as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, z])
        finally:
            session.unload(loaded)

    np.testing.assert_allclose(
        got, x + z, atol=float(meta["scales"]["y"]) * 1.5, rtol=0
    )


@needs_device
@pytest.mark.parametrize("op", ["Add", "Mul"])
@pytest.mark.parametrize("route", ["compile_onnx", "graph_generator"])
def test_onnx_constant_commutative_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(
    tmp_path, op, route
):
    """Stage an initializer after normalizing commutative AX binary inputs."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import binary_op_scale_emit as bse
    import tinygrad_ax_backend as axb

    shape = (1, 64)
    z = np.array(0.2, dtype=np.float32)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node(op, ["z", "x"], ["y"])],
            f"onnx_constant_{op.lower()}_to_uop_vm",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
            [onnx.numpy_helper.from_array(z, "z")],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, meta = bse.load_template(op, shape, {"x": 0, "y": 0, "z": 0})
    calibration = {"scales": meta["scales"], "zero_points": meta["zero_points"]}
    schedule = tmp_path / f"onnx_constant_{op.lower()}_to_uop.schedule.json"
    if route == "compile_onnx":
        axmodel = axb.compile_onnx(model, str(schedule), calibration)
    else:
        import graph_generator

        source = tmp_path / f"onnx_constant_{op.lower()}_to_uop.onnx"
        output = tmp_path / f"onnx_constant_{op.lower()}_to_uop.axmodel"
        onnx.save(model, source)
        graph_generator.generate(
            str(source),
            str(output),
            schedule_path=str(schedule),
            calibration=calibration,
        )
        axmodel = output.read_bytes()
    rng = np.random.default_rng(1965)
    x = rng.uniform(0.0, 0.2, shape).astype(np.float32)
    z_full = np.full(shape, float(z), dtype=np.float32)

    with axcl_session.AXSession(
        subdir=f"uop_constant_{route}_{op.lower()}_{tmp_path.name}"
    ) as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, z_full])
        finally:
            session.unload(loaded)

    np.testing.assert_allclose(
        got,
        {"Add": x + z_full, "Mul": x * z_full}[op],
        atol=float(meta["scales"]["y"]) * 1.5,
        rtol=0,
    )


@needs_device
@pytest.mark.parametrize(
    "op, x_bounds, z_value",
    [("Sub", (0.2, 0.3), 0.15), ("Mul", (0.1, 0.3), 0.2), ("Div", (0.1, 0.3), 0.2)],
)
@pytest.mark.parametrize("route", ["compile_onnx", "graph_generator"])
def test_onnx_constant_broadcast_binary_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(
    tmp_path, op, x_bounds, z_value, route
):
    """Run constant broadcast Sub/Mul/Div through staged AX inputs."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import binary_op_scale_emit as bse
    import tinygrad_ax_backend as axb

    shape = (1, 64)
    z = np.array(z_value, dtype=np.float32)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node(op, ["x", "z"], ["y"])],
            f"onnx_constant_{op.lower()}_broadcast_to_uop_vm",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
            [onnx.numpy_helper.from_array(z, "z")],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, meta = bse.load_template(op, shape, {"x": 0, "y": 0, "z": 0})
    calibration = {"scales": meta["scales"], "zero_points": meta["zero_points"]}
    schedule = tmp_path / f"onnx_constant_{op.lower()}_broadcast.schedule.json"
    if route == "compile_onnx":
        axmodel = axb.compile_onnx(model, str(schedule), calibration)
    else:
        import graph_generator

        source = tmp_path / f"onnx_constant_{op.lower()}_broadcast.onnx"
        output = tmp_path / f"onnx_constant_{op.lower()}_broadcast.axmodel"
        onnx.save(model, source)
        graph_generator.generate(
            str(source),
            str(output),
            schedule_path=str(schedule),
            calibration=calibration,
        )
        axmodel = output.read_bytes()
    rng = np.random.default_rng(1965)
    x = rng.uniform(*x_bounds, shape).astype(np.float32)
    z_full = np.full(shape, float(z), dtype=np.float32)

    with axcl_session.AXSession(
        subdir=f"uop_constant_{route}_{op.lower()}_broadcast_{tmp_path.name}"
    ) as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, z_full])
        finally:
            session.unload(loaded)

    want = {"Sub": x - z_full, "Mul": x * z_full, "Div": x / z_full}[op]
    np.testing.assert_allclose(got, want, atol=float(meta["scales"]["y"]) * 1.5, rtol=0)


@needs_device
@pytest.mark.parametrize("route", ["compile_onnx", "graph_generator"])
def test_onnx_constant_first_mul_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(
    tmp_path, route
):
    """Validate the optimizer-style ``Mul(constant, live)`` normalization."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import binary_op_scale_emit as bse
    import tinygrad_ax_backend as axb

    shape = (1, 64)
    z = np.array(0.2, dtype=np.float32)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Mul", ["z", "x"], ["y"])],
            "onnx_constant_first_mul_to_uop_vm",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
            [onnx.numpy_helper.from_array(z, "z")],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, meta = bse.load_template("Mul", shape, {"x": 0, "y": 0, "z": 0})
    calibration = {"scales": meta["scales"], "zero_points": meta["zero_points"]}
    schedule = tmp_path / f"constant_first_mul_{route}.schedule.json"
    if route == "compile_onnx":
        axmodel = axb.compile_onnx(model, str(schedule), calibration)
    else:
        import graph_generator

        source = tmp_path / "constant_first_mul.onnx"
        output = tmp_path / "constant_first_mul.axmodel"
        onnx.save(model, source)
        graph_generator.generate(
            str(source),
            str(output),
            schedule_path=str(schedule),
            calibration=calibration,
        )
        axmodel = output.read_bytes()

    rng = np.random.default_rng(1965)
    x = rng.uniform(0.1, 0.3, shape).astype(np.float32)
    with axcl_session.AXSession(
        subdir=f"uop_constant_first_mul_{route}_{tmp_path.name}"
    ) as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, np.full(shape, z, dtype=np.float32)])
        finally:
            session.unload(loaded)
    np.testing.assert_allclose(
        got, x * z, atol=float(meta["scales"]["y"]) * 1.5, rtol=0
    )


@needs_device
@pytest.mark.parametrize(
    "op,filename,shape,x,z,want",
    [
        ("add", "add_1x1_x128_y128_z128", (1, 1), -0.5, 0.25, -0.25),
        ("mul", "mul_1x1_x0_y0_z0", (1, 1), 0.5, 0.25, 0.125),
        ("div", "div_1x1_x128_y128_z0", (1, 1), -0.5, 0.5, -1.0),
        ("mul", "mul_1x1_x255_y255_z0", (1, 1), -0.5, 0.25, -0.125),
        ("mul", "mul_16x1000_x255_y255_z0", (16, 1000), -0.5, 0.25, -0.125),
        ("div", "div_1x1_x255_y255_z0", (1, 1), -0.5, 0.2, -2.5),
        ("div", "div_16x1000_x255_y255_z0", (16, 1000), -0.5, 0.5, -1.0),
    ],
)
def test_native_binary_shape_templates_run_on_axcl_vm(
    op, filename, shape, x, z, want, tmp_path
):
    """Run the checked-in native shape templates on the AXCL VM.

    These fixtures cover the small/broadcast shapes that are not represented
    by the original 1xN Pulsar2 templates.  The offline test proves their
    MCode can be retargeted; this closes the loop by loading the actual native
    binaries on AXCL as well.
    """
    import axcl_session

    path = os.path.join(
        HERE,
        "..",
        "scripts",
        "axera",
        "fixtures",
        "binary_op_scale_emit",
        f"{filename}.axmodel.gz",
    )
    with gzip.open(path, "rb") as stream:
        axmodel = stream.read()
    with axcl_session.AXSession(
        subdir=f"native_binary_shape_{op}_{tmp_path.name}"
    ) as session:
        loaded = session.load(axmodel)
        try:
            (got,) = session.run(
                loaded,
                [
                    np.full(shape, x, dtype=np.float32),
                    np.full(shape, z, dtype=np.float32),
                ],
            )
        finally:
            session.unload(loaded)
    np.testing.assert_allclose(
        got, np.full(shape, want, dtype=np.float32), atol=0.02, rtol=0
    )


@needs_device
@pytest.mark.parametrize("route", ["compile_onnx", "graph_generator"])
def test_frozen_conv_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path, route):
    """Run the frozen-weight Conv UOp route without a Pulsar2 build."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import tinygrad_ax_backend as axb

    input_shape = (16, 64, 56, 56)
    output_shape = input_shape
    weights = np.zeros((64, 64, 3, 3), dtype=np.float32)
    bias = np.zeros((64,), dtype=np.float32)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "Conv", ["x", "w", "b"], ["y"], pads=[1, 1, 1, 1], strides=[1, 1]
                )
            ],
            "frozen_conv_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, input_shape
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, output_shape
                )
            ],
            [numpy_helper.from_array(weights, "w"), numpy_helper.from_array(bias, "b")],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    schedule = tmp_path / "frozen_conv_to_uop.schedule.json"
    calibration = {
        "scales": {"x": 0.01, "y": 0.02},
        "zero_points": {"x": 127, "y": 125},
    }
    if route == "compile_onnx":
        axmodel = axb.compile_onnx(model, str(schedule), calibration)
    else:
        import graph_generator

        source = tmp_path / "frozen_conv_to_uop.onnx"
        output = tmp_path / "frozen_conv_to_uop.axmodel"
        onnx.save(model, source)
        graph_generator.generate(
            str(source),
            str(output),
            schedule_path=str(schedule),
            calibration=calibration,
        )
        axmodel = output.read_bytes()
    x = np.zeros(input_shape, dtype=np.float32)
    with axcl_session.AXSession(
        subdir=f"uop_frozen_conv_{route}_{tmp_path.name}"
    ) as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x])
        finally:
            session.unload(loaded)
    assert got.shape == output_shape
    # The measured output zero-point need not decode to float zero; with zero
    # input and zero weights, the useful invariant is a finite uniform result.
    assert np.isfinite(got).all()
    assert float(np.ptp(got)) == 0.0


@needs_device
@pytest.mark.parametrize("op", ["Sub", "Mul", "Div"])
def test_onnx_binary_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path, op):
    """Run the remaining same-shape binary UOps through AXCL VM."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import binary_op_scale_emit as bse
    import tinygrad_ax_backend as axb

    shape = (1, 64)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node(op, ["x", "z"], ["y"])],
            f"onnx_{op.lower()}_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, shape),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, meta = bse.load_template(op, shape, {"x": 0, "y": 0, "z": 0})
    calibration = {
        "scales": meta["scales"],
        "zero_points": meta["zero_points"],
    }
    schedule = tmp_path / f"onnx_{op.lower()}_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), calibration)
    rng = np.random.default_rng(1965)
    x_bounds, z_bounds = (
        ((0.2, 0.3), (0.1, 0.2)) if op == "Sub" else ((0.1, 0.3), (0.1, 0.3))
    )
    x = rng.uniform(*x_bounds, shape).astype(np.float32)
    z = rng.uniform(*z_bounds, shape).astype(np.float32)

    with axcl_session.AXSession(subdir=f"uop_{op.lower()}_{tmp_path.name}") as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, z])
        finally:
            session.unload(loaded)

    want = {"Sub": x - z, "Mul": x * z, "Div": x / z}[op]
    np.testing.assert_allclose(got, want, atol=float(meta["scales"]["y"]) * 1.5, rtol=0)


@needs_device
def test_onnx_broadcast_mul_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Run a broadcast UOp through the full-shape AX binary template."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import binary_op_scale_emit as bse
    import tinygrad_ax_backend as axb

    source_shape, broadcast_shape = (1, 64), (64,)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Mul", ["x", "z"], ["y"])],
            "onnx_broadcast_mul_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, source_shape
                ),
                onnx.helper.make_tensor_value_info(
                    "z", onnx.TensorProto.FLOAT, broadcast_shape
                ),
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, source_shape
                )
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, meta = bse.load_template("Mul", source_shape, {"x": 0, "y": 0, "z": 0})
    calibration = {"scales": meta["scales"], "zero_points": meta["zero_points"]}
    schedule = tmp_path / "onnx_broadcast_mul_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), calibration)
    rng = np.random.default_rng(1965)
    x = rng.uniform(0.1, 0.3, source_shape).astype(np.float32)
    z = rng.uniform(0.1, 0.3, broadcast_shape).astype(np.float32)

    with axcl_session.AXSession(subdir=f"uop_broadcast_mul_{tmp_path.name}") as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, np.broadcast_to(z, source_shape)])
        finally:
            session.unload(loaded)

    np.testing.assert_allclose(
        got, x * z.reshape(1, 64), atol=float(meta["scales"]["y"]) * 1.5, rtol=0
    )


@needs_device
@pytest.mark.parametrize(
    "op, input_shape, output_shape, attrs, template_key",
    [
        (
            "ReduceMean",
            (16, 512, 7, 7),
            (16, 512, 1, 1),
            {"axes": [2, 3], "keepdims": 1},
            "ReduceMean:16x512x7x7:axes2,3:k1",
        ),
        (
            "Softmax",
            (16, 1000),
            (16, 1000),
            {"axis": 1},
            "Softmax:16x1000:axis1",
        ),
        (
            "MaxPool",
            (16, 64, 112, 112),
            (16, 64, 56, 56),
            {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
            "MaxPool:16x64x112x112:k3x3:s2x2:p1,1,1,1",
        ),
        (
            "ReduceSum",
            (16, 64, 112, 112),
            (64,),
            {"axes": [0, 2, 3], "keepdims": 0},
            "ReduceSum:16x64x112x112:axes0,2,3:k0",
        ),
        (
            "Sqrt",
            (512, 512, 3, 3),
            (512, 512, 3, 3),
            {},
            "Sqrt:512x512x3x3",
        ),
        (
            "Log",
            (16, 1000),
            (16, 1000),
            {},
            "Log:16x1000",
        ),
        (
            "Neg",
            (1, 1),
            (1, 1),
            {},
            "Neg:1x1",
        ),
    ],
)
def test_onnx_misc_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(
    tmp_path, op, input_shape, output_shape, attrs, template_key
):
    """Run reduction/normalization UOps used by the training step on AXCL."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import misc_op_record_emit as misc
    import tinygrad_ax_backend as axb

    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node(op, ["x"], ["y"], **attrs)],
            f"onnx_{op.lower()}_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, input_shape
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, output_shape
                )
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, meta = misc.load_template(template_key)
    calibration = {
        "scales": meta["scales"],
        "zero_points": meta["zero_points"],
    }
    schedule = tmp_path / f"onnx_{op.lower()}_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), calibration)
    rng = np.random.default_rng(1965)
    bounds = {
        "ReduceSum": (-0.02, 0.02),
        "ReduceMean": (0.0, 1.0),
        "MaxPool": (0.0, 1.0),
        "Sqrt": (0.01, 1.0),
        "Log": (0.1, 1.0),
        "Neg": (-0.02, 0.02),
    }.get(op, (-1.0, 1.0))
    x = rng.uniform(*bounds, input_shape).astype(np.float32)

    # AXCL's virtiofs layer can retain the previous m0.axmodel by pathname;
    # isolate each operator family in its own session directory.
    with axcl_session.AXSession(subdir=f"uop_{op.lower()}_{tmp_path.name}") as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x])
        finally:
            session.unload(loaded)

    if op == "ReduceMean":
        want = x.mean(axis=(2, 3), keepdims=True)
    elif op == "MaxPool":
        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
        want = windows[:, :, ::2, ::2].max(axis=(-1, -2))
    elif op == "ReduceSum":
        want = x.sum(axis=(0, 2, 3))
    elif op == "Sqrt":
        want = np.sqrt(x)
    elif op == "Log":
        want = np.log(x)
    elif op == "Neg":
        want = -x
    else:
        shifted = x - x.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        want = exp / exp.sum(axis=1, keepdims=True)
    tolerance = 4.0 if op == "Sqrt" else 2.0
    np.testing.assert_allclose(
        got, want, atol=float(meta["scales"]["y"]) * tolerance, rtol=0
    )


@needs_device
@pytest.mark.parametrize(
    "op, input_shape, template_key",
    [
        ("Greater", (16, 64, 112, 112), "GreaterCast:16x64x112x112"),
        ("Less", (1024, 9, 3136), "LessCast:1024x9x3136"),
    ],
)
def test_onnx_comparison_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(
    tmp_path, op, input_shape, template_key
):
    """Run calibration-free comparison/cast UOps through AXCL VM."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import misc_op_record_emit as misc
    import tinygrad_ax_backend as axb
    from tinygrad import Tensor

    _, meta = misc.load_template(template_key)
    schedule = tmp_path / f"onnx_{op.lower()}cast_to_uop.schedule.json"
    root = (
        (Tensor.empty(*input_shape) > 0).cast("float32")
        if op == "Greater"
        else (Tensor.empty(*input_shape) < 0).cast("float32")
    ).uop
    axmodel = axb.compile_uop(root, str(schedule))
    rng = np.random.default_rng(1965)
    x = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    with axcl_session.AXSession(
        subdir=f"uop_{op.lower()}cast_{tmp_path.name}"
    ) as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            inputs = [x]
            if op == "Less":
                inputs.append(np.zeros((1024, 1, 3136), dtype=np.float32))
            (got,) = session.run(loaded, inputs)
        finally:
            session.unload(loaded)

    want = (x > 0.0 if op == "Greater" else x < 0.0).astype(np.float32)
    np.testing.assert_array_equal(got, want)


class _EchoSession:
    """Stands in for AXSession: records each run's input shapes and returns
    the first input times two."""

    def __init__(self):
        self.calls = []

    def load(self, blob):
        return object()

    def unload(self, m):
        pass

    def run(self, m, ins):
        self.calls.append([x.shape for x in ins])
        return [ins[0] * 2]


def test_batch_split_segment_runs_on_batch_slices_and_concatenates():
    model = parser.parse_model(
        """<ir_version: 8, opset_import: ["" : 13]>
        g (float[4, 3] x, float[5, 3] w) => (float[4, 3] y) {
            y = Identity(x)
        }"""
    )
    seg = sr.Segment(
        "y", "matmul_chain", [model.graph.node[0].name or "n0"], ["x", "w"], ["y"],
        "test", lambda: model, batch_split=2, split=[True, False],
    )  # fmt: skip
    model.graph.node[0].name = seg.nodes[0]
    session = _EchoSession()
    runner = sr.StepRunner(model, [seg], session=session)
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    w = np.ones((5, 3), np.float32)
    (y,) = runner._device(seg, {"x": x, "w": w})
    assert session.calls == [[(2, 3), (5, 3)], [(2, 3), (5, 3)]]
    np.testing.assert_array_equal(y, x * 2)
