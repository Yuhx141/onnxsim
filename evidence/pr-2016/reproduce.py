#!/usr/bin/env python3
"""Replay and verify the runtime evidence for onnxsim PR #2016."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_entries():
    for folder in sorted(CASES.iterdir(), key=lambda path: int(path.name)):
        if not folder.is_dir():
            continue
        for index, case in enumerate(json.loads((folder / "cases.json").read_text())):
            yield folder.name, index, case


def summarize(values):
    import numpy as np

    result = {}
    for name, value in values.items():
        item = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256_bytes": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        if np.issubdtype(value.dtype, np.number):
            item.update(
                nan=int(np.isnan(value).sum()),
                posinf=int(np.isposinf(value).sum()),
                neginf=int(np.isneginf(value).sum()),
            )
        if value.size <= 16:
            item["values_repr"] = repr(value.tolist())
        result[name] = item
    return result


def compare(left, right):
    import numpy as np

    result = {
        "names_equal": left.keys() == right.keys(),
        "shape_dtype_equal": True,
        "exact": True,
        "allclose_rtol_1e-5_atol_1e-6": True,
        "nonfinite_masks_equal": True,
        "max_abs_finite": 0.0,
    }
    if not result["names_equal"]:
        result["exact"] = result["allclose_rtol_1e-5_atol_1e-6"] = False
        return result
    for name, a in left.items():
        b = right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            result["shape_dtype_equal"] = False
            result["exact"] = result["allclose_rtol_1e-5_atol_1e-6"] = False
            continue
        result["exact"] &= bool(np.array_equal(a, b, equal_nan=True))
        result["allclose_rtol_1e-5_atol_1e-6"] &= bool(
            np.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True)
        )
        for mask in (np.isnan, np.isposinf, np.isneginf):
            result["nonfinite_masks_equal"] &= bool(np.array_equal(mask(a), mask(b)))
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            result["max_abs_finite"] = max(
                result["max_abs_finite"],
                float(np.max(np.abs(a[finite].astype("float64") - b[finite].astype("float64")))),
            )
    return result


def run_ort(model, feeds):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    outputs = session.run(None, {value.name: feeds[value.name] for value in session.get_inputs()})
    return dict(zip([value.name for value in session.get_outputs()], outputs))


def worker(issue: str, index: int, mode: str, revision: str, output: Path) -> int:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import onnxsim

    folder = CASES / issue
    case = json.loads((folder / "cases.json").read_text())[index]
    target_dir = output / issue / str(index)
    target_dir.mkdir(parents=True, exist_ok=True)
    result_path = target_dir / "result.json"
    source_path = folder / case["source"]
    input_path = folder / case["input"]
    source = onnx.load(source_path)
    extension = next(Path(onnxsim.__file__).parent.glob("onnxsim_cpp2py_export*.so"), None)
    result = {
        "issue": int(issue),
        "case": case["name"],
        "pass": case["pass"],
        "mode": mode,
        "revision": revision,
        "source_sha256": sha256(source_path),
        "input_sha256": sha256(input_path),
        "python": sys.version,
        "onnxsim": onnxsim.__version__,
        "onnxsim_extension_sha256": sha256(extension) if extension else None,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "runtime": "CPUExecutionProvider / ORT_DISABLE_ALL / one thread",
    }
    with np.load(input_path, allow_pickle=False) as archive:
        feeds = dict(archive)

    try:
        onnx.checker.check_model(source, full_check=True)
        result["source_checker_full"] = "passed"
        expected = run_ort(source, feeds)
        result["source_ort"] = "passed"
        result["source_outputs"] = summarize(expected)
    except Exception as error:
        result["source_error"] = f"{type(error).__name__}: {error}"
        dump(result_path, result)
        return 10

    # Save source preconditions before simplify: a native assertion may terminate this process.
    dump(result_path, result)
    kwargs = {"skipped_optimizers": [case["pass"]]} if mode == "disabled" else {}
    try:
        target, check_ok = onnxsim.simplify(source, check_n=0, **kwargs)
    except Exception as error:
        result["simplify_error"] = f"{type(error).__name__}: {error}"
        dump(result_path, result)
        return 11

    target_path = target_dir / "target.onnx"
    onnx.save(target, target_path)
    result["simplifier_check_ok"] = bool(check_ok)
    result["target_sha256"] = sha256(target_path)
    result["target_ops"] = [node.op_type for node in target.graph.node]
    for name, full_check in (("target_checker_basic", False), ("target_checker_full", True)):
        try:
            onnx.checker.check_model(target, full_check=full_check)
            result[name] = "passed"
        except Exception as error:
            result[name] = f"{type(error).__name__}: {error}"

    try:
        observed = run_ort(target, feeds)
        result["target_ort"] = "passed"
        result["target_outputs"] = summarize(observed)
        result["comparison"] = compare(expected, observed)
    except Exception as error:
        result["target_ort_error"] = f"{type(error).__name__}: {error}"
        dump(result_path, result)
        return 12

    dump(result_path, result)
    comparison = result["comparison"]
    return 0 if (
        result["target_checker_full"] == "passed"
        and comparison["allclose_rtol_1e-5_atol_1e-6"]
        and comparison["nonfinite_masks_equal"]
    ) else 13


def run_all(mode: str, revision: str, output: Path) -> int:
    runs = []
    for issue, index, case in case_entries():
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_worker",
            issue,
            str(index),
            "--mode",
            mode,
            "--revision",
            revision,
            "--output",
            str(output.resolve()),
        ]
        process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        folder = output / issue / str(index)
        folder.mkdir(parents=True, exist_ok=True)
        if process.stdout:
            (folder / "stdout.log").write_text(process.stdout)
        if process.stderr:
            (folder / "stderr.log").write_text(process.stderr)
        record = {
            "issue": int(issue),
            "case_index": index,
            "case": case["name"],
            "pass": case["pass"],
            "mode": mode,
            "revision": revision,
            "returncode": process.returncode,
        }
        dump(folder / "process.json", record)
        runs.append(record)
        print(f"#{issue} [{index}] {case['pass']}: exit {process.returncode}", flush=True)
    dump(output / "run-manifest.json", {"mode": mode, "revision": revision, "runs": runs})
    return 0


def read_case(root: Path, label: str, issue: str, index: int):
    folder = root / label / issue / str(index)
    return (
        json.loads((folder / "process.json").read_text()),
        json.loads((folder / "result.json").read_text()),
    )


def verify(root: Path) -> int:
    failures = []
    exact = {"head": 0, "disabled": 0}
    cases = list(case_entries())
    for label in ("base", "disabled", "head"):
        manifest = json.loads((root / label / "run-manifest.json").read_text())
        if len(manifest["runs"]) != len(cases):
            failures.append(f"{label}: expected {len(cases)} runs")

    for issue, index, case in cases:
        for label in ("disabled", "head"):
            process, result = read_case(root, label, issue, index)
            comparison = result.get("comparison", {})
            if process["returncode"] != 0:
                failures.append(f"{label} #{issue}/{index}: exit {process['returncode']}")
            if result.get("target_checker_full") != "passed" or result.get("target_ort") != "passed":
                failures.append(f"{label} #{issue}/{index}: target validation failed")
            if not comparison.get("allclose_rtol_1e-5_atol_1e-6"):
                failures.append(f"{label} #{issue}/{index}: not allclose")
            if not comparison.get("nonfinite_masks_equal"):
                failures.append(f"{label} #{issue}/{index}: non-finite masks differ")
            exact[label] += int(bool(comparison.get("exact")))

        process, result = read_case(root, "base", issue, index)
        if issue == "1995" and "bias must be a 1D tensor" not in result.get("target_ort_error", ""):
            failures.append("base #1995: expected ORT Conv-bias rejection")
        elif issue == "1996" and "Product cannot be 0" not in result.get("target_checker_full", ""):
            failures.append("base #1996: expected full-checker Reshape rejection")
        elif issue == "1998":
            stderr_path = root / "base" / issue / str(index) / "stderr.log"
            stderr = stderr_path.read_text() if stderr_path.exists() else ""
            failure = stderr + result.get("simplify_error", "")
            if process["returncode"] == 0 or "eraseOutput" not in failure:
                failures.append("base #1998: expected eraseOutput process failure")
        elif issue == "1999" and "topologically" not in result.get("simplify_error", ""):
            failures.append("base #1999: expected topological simplify failure")
        elif issue not in {"1995", "1996", "1998", "1999"}:
            if result.get("comparison", {}).get("exact", True):
                failures.append(f"base #{issue}/{index}: expected semantic difference")

    if exact != {"head": 10, "disabled": 11}:
        failures.append(f"unexpected exact counts: {exact}")
    verification = {
        "status": "passed" if not failures else "failed",
        "cases": len(cases),
        "base_failures_or_differences": len(cases),
        "disabled_exact": exact["disabled"],
        "head_exact": exact["head"],
        "head_allclose_not_exact": len(cases) - exact["head"],
        "failures": failures,
    }
    dump(root / "verification.json", verification)
    print(json.dumps(verification, indent=2))
    return int(bool(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("default", "disabled"), required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--output", type=Path, required=True)
    worker_parser = commands.add_parser("_worker")
    worker_parser.add_argument("issue")
    worker_parser.add_argument("index", type=int)
    worker_parser.add_argument("--mode", choices=("default", "disabled"), required=True)
    worker_parser.add_argument("--revision", required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        return run_all(args.mode, args.revision, args.output)
    if args.command == "_worker":
        return worker(args.issue, args.index, args.mode, args.revision, args.output)
    return verify(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
