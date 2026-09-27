import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

from xdna_backend import (
    XDNAUnavailable,
    analyze_model,
    execute,
    load_kernel_manifest,
    optimize_partitions,
    partition_model,
    select_matmul_tile,
    matmul_kernel_key,
    resolve_kernel_artifact,
    XDNAArtifactExecutor,
    dispatch_matmul,
    dispatch_matmul_batch,
    load_tuning_profile,
    plan_matmul,
    validate_matmul_buffers,
    plan_transfer,
    plan_pipeline_transfer,
)


def _model(*ops):
    nodes = [SimpleNamespace(op_type=op, name=f"{op}_{index}") for index, op in enumerate(ops)]
    return SimpleNamespace(graph=SimpleNamespace(node=nodes))


def test_core_ops_are_xdna_partitioned_and_unknown_ops_fallback():
    parts = partition_model(_model("Relu", "MatMul", "UnsupportedOp", "Add"))
    assert [part.device for part in parts] == ["XDNA", "FALLBACK", "XDNA"]
    assert [node.op_type for node in parts[0].nodes] == ["Relu", "MatMul"]
    assert parts[1].nodes[0].supported is False


def test_analysis_has_stable_node_names():
    info = analyze_model(_model("Gemm"))
    assert info[0].name == "Gemm_0"
    assert info[0].supported is True


def test_manifest_requires_kernel_object(tmp_path):
    path = tmp_path / "kernels.json"
    path.write_text(json.dumps({"version": 1, "kernels": {"gemm_f16": {}}}))
    assert "gemm_f16" in load_kernel_manifest(path)["kernels"]

    path.write_text(json.dumps({"version": 1}))
    with pytest.raises(ValueError, match="kernels"):
        load_kernel_manifest(path)


def test_execution_is_explicitly_unavailable():
    with pytest.raises(XDNAUnavailable, match="not enabled"):
        execute()


def test_fuses_compute_post_ops_but_not_two_producers():
    groups = optimize_partitions(partition_model(_model("MatMul", "Add", "Relu", "MatMul")))
    assert [group.op_types for group in groups] == [
        ("MatMul", "Add", "Relu"),
        ("MatMul",),
    ]
    assert groups[0].kernel_kind == "matmul_fused"


def test_matmul_tile_selection_uses_measured_dtype_hints():
    assert select_matmul_tile("i8", 512, 512, 512, 8) == (64, 64, 64)
    assert select_matmul_tile("bf16", 512, 512, 512, 8) == (64, 32, 32)
    # The wide i8 N tile is illegal here, so the selector falls back to the
    # narrower measured candidate.
    assert select_matmul_tile("i8", 512, 512, 256, 8) == (64, 32, 32)


def test_profile_overrides_default_tile_when_shape_is_legal():
    profile = {"i8:512x512x512:c8": [64, 32, 32]}
    assert select_matmul_tile("i8", 512, 512, 512, 8, profile) == (64, 32, 32)
    assert matmul_kernel_key("i8", 512, 512, 512, 8, profile) == "matmul_i8_m64k32n32_c8"


def test_load_tuning_profile_validates_benchmark_report(tmp_path):
    path = tmp_path / "tuning.json"
    path.write_text('{"profile": {"i8:512x512x512:c8": [64, 32, 64]}}')
    assert load_tuning_profile(path)["i8:512x512x512:c8"] == [64, 32, 64]
    path.write_text('{"profile": {"bad": [0, 32, 64]}}')
    with pytest.raises(ValueError, match="invalid XDNA tuning profile"):
        load_tuning_profile(path)


def test_matmul_plan_infers_onnx_shapes_and_profile_tile():
    plan = plan_matmul((512, 512), (512, 512), "i8", 8, output_dtype="i32")
    assert plan.shape == (512, 512, 512)
    assert plan.output_dtype == "i32"
    assert plan.tile == (64, 64, 64)
    assert plan.kernel == "matmul_i8_oi32_m64k64n64_c8"
    with pytest.raises(ValueError, match="incompatible MatMul shapes"):
        plan_matmul((512, 256), (512, 512), "i8")


def test_matmul_buffers_are_validated_before_launch():
    plan = plan_matmul((4, 8), (8, 16), "i16", 1, output_dtype="i32")
    a = SimpleNamespace(shape=(4, 8), dtype="int16")
    b = SimpleNamespace(shape=(8, 16), dtype="int16")
    c = SimpleNamespace(shape=(4, 16), dtype="int32")
    assert validate_matmul_buffers(plan, a, b, c)[0].shape == (4, 8)
    with pytest.raises(ValueError, match="output buffer"):
        validate_matmul_buffers(plan, a, b, SimpleNamespace(shape=(4, 8), dtype="int16"))
    with pytest.raises(ValueError, match="contiguous"):
        validate_matmul_buffers(
            plan,
            SimpleNamespace(shape=(4, 8), dtype="int16", is_contiguous=False),
            b,
            c,
        )


def test_transfer_planner_selects_resident_and_double_buffered_modes():
    resident = plan_transfer(4096, 4096, reuse_count=4)
    assert resident.strategy == "weight_resident"
    assert resident.double_buffered is False
    streamed = plan_transfer(1 << 20, 1 << 20)
    assert streamed.strategy == "double_buffered_stream"
    assert streamed.double_buffered is True
    internal = plan_transfer(1 << 20, 1 << 20, input_resident=True, output_resident=True)
    assert internal.strategy == "device_resident"
    assert internal.estimated_bytes == 0
    pipeline = plan_pipeline_transfer(1 << 20, 1 << 20, [1 << 20, 1 << 18])
    assert pipeline.strategy == "resident_pipeline_double_buffered"
    assert pipeline.estimated_bytes == 2 << 20


def test_kernel_manifest_resolves_tuned_artifact():
    key = matmul_kernel_key("i8", 512, 512, 512, 8)
    manifest = {"kernels": {key: {"xclbin": "gemm.xclbin", "insts": "gemm.bin"}}}
    assert resolve_kernel_artifact(manifest, key)["xclbin"] == "gemm.xclbin"
    assert resolve_kernel_artifact(manifest, "missing") is None


def test_matmul_kernel_key_separates_output_dtypes():
    assert matmul_kernel_key("i8", 512, 512, 512, 8, output_dtype="i8") != matmul_kernel_key(
        "i8", 512, 512, 512, 8, output_dtype="i32"
    )


def test_kernel_manifest_rejects_incomplete_artifact():
    with pytest.raises(ValueError, match="insts"):
        resolve_kernel_artifact({"kernels": {"gemm": {"xclbin": "x"}}}, "gemm")


def test_artifact_executor_resolves_relative_paths(tmp_path):
    executor = XDNAArtifactExecutor(
        {"xclbin": "kernels/gemm.xclbin", "insts": "kernels/gemm.insts.bin"},
        base_dir=tmp_path,
    )
    assert executor.xclbin == tmp_path / "kernels/gemm.xclbin"
    assert executor.insts == tmp_path / "kernels/gemm.insts.bin"
    assert executor._kernel is None


def test_dispatch_reports_fallback_when_tuned_artifact_is_missing():
    result = dispatch_matmul(
        {"kernels": {}},
        object(),
        object(),
        object(),
        dtype="i8",
        m=512,
        k=512,
        n=512,
    )
    assert result["execution"] == "fallback"
    assert result["kernel"] == "matmul_i8_m64k64n64_c8"


def test_batch_dispatch_reports_fallback_without_artifact():
    result = dispatch_matmul_batch(
        {"kernels": {}},
        [],
        dtype="i8",
        m=512,
        k=512,
        n=512,
    )
    assert result["execution"] == "fallback"
    assert "calls" not in result
