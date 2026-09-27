"""Dependency-light AMD XDNA backend helpers.

The Phase 1 backend is deliberately an offline graph planner.  Runtime
execution is added once precompiled IRON/MLIR-AIE artifacts are available.
"""

from .bottleneck_runtime import BottleneckBinding, bind_bottleneck_block
from .conv_reference import execute_conv_reference, im2col_nchw
from .qdq_runtime import QDQEdge, QuantParams, extract_qdq_edges, qdq_edge_map
from .resnet_bottleneck import BottleneckBlockPlan, plan_bottleneck_blocks
from .resnet_codegen import (
    DispatchSpec,
    ResNetCodegenPlan,
    build_codegen_plan,
    codegen_plan_to_dict,
)
from .resnet_coverage import (
    CoverageEntry,
    ResNetCoverage,
    build_resnet_coverage,
    coverage_to_dict,
)
from .resnet_emitter import (
    BottleneckArtifactSpec,
    KernelArtifactSpec,
    emit_bottleneck_specs,
    emit_kernel_specs,
    render_build_manifest,
    write_build_manifest,
)
from .tinygrad_bridge import (
    numpy_to_iron,
    numpy_to_tinygrad,
    plan_tinygrad_matmul,
    run_tinygrad_matmul,
    tinygrad_dtype_to_numpy,
    tinygrad_to_iron,
    tinygrad_to_numpy,
    xdna_dtype_to_numpy,
)
from .xdna_backend import (
    CORE_OPS,
    XDNA_AVAILABLE,
    XDNAArtifactExecutor,
    XDNAUnavailable,
    analyze_model,
    dispatch_matmul,
    dispatch_matmul_batch,
    load_tuning_profile,
    matmul_kernel_key,
    optimize_partitions,
    partition_model,
    plan_matmul,
    plan_pipeline_transfer,
    plan_transfer,
    resolve_kernel_artifact,
    select_matmul_tile,
    validate_matmul_buffers,
)

__all__ = [
    "CORE_OPS",
    "XDNA_AVAILABLE",
    "XDNAUnavailable",
    "analyze_model",
    "optimize_partitions",
    "partition_model",
    "select_matmul_tile",
    "matmul_kernel_key",
    "resolve_kernel_artifact",
    "XDNAArtifactExecutor",
    "dispatch_matmul",
    "dispatch_matmul_batch",
    "load_tuning_profile",
    "plan_matmul",
    "validate_matmul_buffers",
    "plan_transfer",
    "plan_pipeline_transfer",
    "DispatchSpec",
    "ResNetCodegenPlan",
    "build_codegen_plan",
    "codegen_plan_to_dict",
    "KernelArtifactSpec",
    "BottleneckArtifactSpec",
    "emit_bottleneck_specs",
    "emit_kernel_specs",
    "render_build_manifest",
    "write_build_manifest",
    "QDQEdge",
    "QuantParams",
    "extract_qdq_edges",
    "qdq_edge_map",
    "CoverageEntry",
    "ResNetCoverage",
    "build_resnet_coverage",
    "coverage_to_dict",
    "execute_conv_reference",
    "im2col_nchw",
    "BottleneckBlockPlan",
    "plan_bottleneck_blocks",
    "BottleneckBinding",
    "bind_bottleneck_block",
    "numpy_to_iron",
    "numpy_to_tinygrad",
    "plan_tinygrad_matmul",
    "run_tinygrad_matmul",
    "tinygrad_dtype_to_numpy",
    "tinygrad_to_iron",
    "tinygrad_to_numpy",
    "xdna_dtype_to_numpy",
]
