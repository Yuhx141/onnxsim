from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "xdna"))

from benchmark import benchmark_workload  # noqa: E402


def test_benchmark_separates_fallback_coverage():
    result = benchmark_workload(
        type("Workload", (), {
            "name": "mixed", "ops": ("MatMul", "Unknown"), "macs": 16,
            "bytes_moved": 32, "kernel": "gemm_f16",
        })(),
        runs=2,
    )
    assert result["xdna_nodes"] == 1
    assert result["coverage_percent"] == 50.0
    assert result["execution"] == "not_available"
    assert result["arithmetic_intensity_macs_per_byte"] == 0.5
    assert result["kernel_artifact"] is False
    assert result["planner_us"]["median"] >= 0
