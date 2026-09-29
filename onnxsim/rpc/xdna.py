"""Server-side XDNA ResNet compile and run operations for onnxsim RPC."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

from . import _protocol as proto


def _script(name: str) -> Path:
    path = Path(__file__).resolve().parents[2] / "scripts" / "xdna" / name
    if not path.is_file():
        raise proto.RPCError(f"XDNA RPC needs the source checkout script {path}")
    return path


def _xdna_python(header: Dict[str, Any]) -> str:
    return str(header.get("_xdna_python") or sys.executable)


def _run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, env=env)
    except OSError as error:
        raise proto.RPCError(f"could not start XDNA command {command[1]}: {error}") from error
    if result.returncode:
        detail = (result.stdout + "\n" + result.stderr)[-16000:]
        raise proto.RPCError(
            f"XDNA command failed (exit {result.returncode}): {' '.join(command[:2])}\n{detail}"
        )
    return result


def _run_inprocess(command: list[str]) -> None:
    """Run an XRT graph request in the RPC host so contexts survive requests."""
    import runpy

    script = Path(command[1]).resolve()
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    namespace = runpy.run_path(str(script), run_name="onnxsim_rpc_xdna_runner")
    status = namespace["main"](command[2:], emit_json=False)
    if status:
        raise proto.RPCError(f"XDNA runner returned status {status}")


def compile_resnet(header: Dict[str, Any], blobs: list[bytes], work_dir: str):
    """Compile the supported ResNet artifacts on the RPC server's XDNA toolchain."""
    kind = header.get("kind")
    if kind not in ("resnet", "fused_bottleneck", "fused_stage", "maxpool_u8"):
        raise proto.RPCError(f"unsupported XDNA compile kind {kind!r}")
    if not blobs:
        raise proto.RPCError("XDNA compile requires an ONNX model blob")
    options = header.get("options") or {}
    if not isinstance(options, dict):
        raise proto.RPCError("XDNA compile options must be an object")
    root = Path(work_dir) / "xdna-rpc" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    model_path = root / "model.onnx"
    model_path.write_bytes(blobs[0])
    artifacts = root / "artifacts"
    artifacts.mkdir()

    if kind == "resnet":
        example = Path(str(options.get("example", ""))).expanduser()
        if not example.is_file():
            raise proto.RPCError("resnet compile requires an existing server-side IRON example path")
        manifest_path = root / "manifest.json"
        command = [
            _xdna_python(header), str(_script("compile_resnet_kernels.py")), str(model_path),
            str(example), str(manifest_path), "--artifact-dir", str(artifacts),
            "--device", str(options.get("device", "npu2")),
            "--columns", str(int(options.get("columns", 8))),
        ]
        if options.get("compile_all"):
            command.append("--compile-all")
        if options.get("optimize_small_m", True):
            command.append("--optimize-small-m")
        _run(command)
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {"kind": kind, "manifest": result, "artifact_dir": str(artifacts)}, []

    xclbin, insts = artifacts / f"{kind}.xclbin", artifacts / f"{kind}.insts.bin"
    if kind == "fused_bottleneck":
        block = str(options.get("block", ""))
        if not block:
            raise proto.RPCError("fused_bottleneck compile requires options.block")
        command = [
            _xdna_python(header), str(_script("fused_bottleneck_design.py")), "--dev",
            str(options.get("device", "npu2")), "--model", str(model_path), "--block", block,
            "--xclbin-path", str(xclbin), "--insts-path", str(insts),
        ]
    elif kind == "fused_stage":
        blocks = options.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != 3 or not all(isinstance(value, str) for value in blocks):
            raise proto.RPCError("fused_stage compile requires options.blocks with exactly three block prefixes")
        command = [
            _xdna_python(header), str(_script("linked_bottleneck_stage_design.py")), "--dev",
            str(options.get("device", "npu2")), "--model", str(model_path), "--blocks", *blocks,
            "--xclbin-path", str(xclbin), "--insts-path", str(insts),
        ]
    else:
        required = (
            "channels", "input_height", "input_width", "output_height", "output_width",
            "kernel_height", "kernel_width", "stride_height", "stride_width",
        )
        missing = [key for key in required if key not in options]
        if missing:
            raise proto.RPCError(f"maxpool_u8 compile missing options: {', '.join(missing)}")
        command = [
            _xdna_python(header), str(_script("maxpool_design.py")), "--dev",
            str(options.get("device", "npu2")),
        ]
        cli_names = {
            "channels": "channels", "input_height": "input-height", "input_width": "input-width",
            "output_height": "output-height", "output_width": "output-width",
            "kernel_height": "kernel-height", "kernel_width": "kernel-width",
            "stride_height": "stride-height", "stride_width": "stride-width",
            "tile_output_rows": "tile-output-rows", "tile_channels": "tile-channels",
            "pad_top": "pad-top", "pad_left": "pad-left", "pad_bottom": "pad-bottom",
            "pad_right": "pad-right",
        }
        defaults = {"tile_output_rows": 8, "tile_channels": 4,
                    "pad_top": 1, "pad_left": 1, "pad_bottom": 1, "pad_right": 1}
        for key, cli_name in cli_names.items():
            if key in options or key in defaults:
                command.extend([f"--{cli_name}", str(int(options.get(key, defaults.get(key))))])
        command += ["--uint8", "--xclbin-path", str(xclbin), "--insts-path", str(insts)]

    _run(command)
    absent = [str(path) for path in (xclbin, insts) if not path.is_file()]
    if absent:
        raise proto.RPCError(f"XDNA compiler succeeded but did not create: {', '.join(absent)}")
    return {
        "kind": kind, "xclbin": str(xclbin), "insts": str(insts),
        "artifact_dir": str(artifacts),
    }, []


def run_resnet(header: Dict[str, Any], blobs: list[bytes], work_dir: str):
    """Run and profile the XDNA graph on the RPC server; return the runner's JSON report."""
    if len(blobs) != 2:
        raise proto.RPCError("XDNA ResNet run requires model and manifest JSON blobs")
    options = header.get("options") or {}
    if not isinstance(options, dict):
        raise proto.RPCError("XDNA run options must be an object")
    root = Path(work_dir) / "xdna-rpc" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    model_path, manifest_path, report_path = root / "model.onnx", root / "manifest.json", root / "report.json"
    model_path.write_bytes(blobs[0])
    manifest_path.write_bytes(blobs[1])
    command = [
        _xdna_python(header), str(_script("run_resnet_xdna.py")), str(model_path), str(manifest_path),
        "--warmup", str(max(int(options.get("warmup", 1)), 0)),
        "--iters", str(max(int(options.get("iters", 5)), 1)),
        "--seed", str(int(options.get("seed", 0))),
        "--cpu-backend", str(options.get("cpu_backend", "numpy")),
        "--cpu-threads", str(max(int(options.get("cpu_threads", 2)), 1)),
        "--json", str(report_path),
    ]
    capture_path = None
    if options.get("capture_outputs"):
        capture_path = root / "xdna-captures.npz"
        command += ["--capture-npz", str(capture_path)]
    if options.get("cpu_small_m") is not None:
        command += ["--cpu-small-m", str(max(int(options["cpu_small_m"]), 0))]
    for block in options.get("fused_blocks", []):
        if not isinstance(block, dict) or not all(key in block for key in ("prefix", "xclbin", "insts")):
            raise proto.RPCError("each fused_blocks entry needs prefix, xclbin, and insts")
        command += ["--fused-block", str(block["prefix"]), str(block["xclbin"]), str(block["insts"])]
    for stage in options.get("fused_stages", []):
        if not isinstance(stage, dict) or not all(key in stage for key in ("blocks", "xclbin", "insts")):
            raise proto.RPCError("each fused_stages entry needs blocks, xclbin, and insts")
        blocks = stage["blocks"]
        if not isinstance(blocks, list) or len(blocks) != 3 or not all(isinstance(value, str) for value in blocks):
            raise proto.RPCError("each fused stage needs exactly three block prefixes")
        command += ["--fused-stage", *blocks, str(stage["xclbin"]), str(stage["insts"])]
    pool = options.get("maxpool_uint8")
    runtime_backend = options.get("runtime_backend", options.get("maxpool_runtime", "iron"))
    if runtime_backend not in ("iron", "xrt"):
        raise proto.RPCError("runtime_backend must be 'iron' or 'xrt'")
    if runtime_backend == "xrt":
        command += ["--runtime-backend", "xrt"]
    if pool is not None:
        if not isinstance(pool, dict) or not all(key in pool for key in ("xclbin", "insts")):
            raise proto.RPCError("maxpool_uint8 needs xclbin and insts")
        command += ["--maxpool-uint8-xclbin", str(pool["xclbin"]), "--maxpool-uint8-insts", str(pool["insts"])]
    runtime_backend = str(options.get("runtime_backend", options.get("maxpool_runtime", "iron")))
    if runtime_backend == "xrt" and Path(_xdna_python(header)).resolve() == Path(sys.executable).resolve():
        _run_inprocess(command)
    else:
        _run(command)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise proto.RPCError(f"XDNA runner did not produce a valid JSON report: {error}") from error
    if capture_path is not None and capture_path.is_file():
        report["capture_npz"] = str(capture_path)
    if runtime_backend == "xrt" and "xdna_xrt_runtime" in sys.modules:
        runtime_module = sys.modules["xdna_xrt_runtime"]
        report["xrt_kernel_cache"] = runtime_module.load_kernel.cache_info()._asdict()
    return {"report": report}, []


def compare_resnet(header: Dict[str, Any], blobs: list[bytes], work_dir: str):
    """Run XDNA and Vitis AI sequentially with matched inputs and timing settings."""
    if len(blobs) != 2:
        raise proto.RPCError("XDNA/Vitis comparison requires model and manifest JSON blobs")
    options = header.get("options") or {}
    if not isinstance(options, dict):
        raise proto.RPCError("comparison options must be an object")
    xdna_result, _ = run_resnet(header, blobs, work_dir)
    root = Path(work_dir) / "xdna-rpc" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    model_path, report_path = root / "model.onnx", root / "vitis.json"
    model_path.write_bytes(blobs[0])
    warmup = max(int(options.get("warmup", 1)), 0)
    iters = max(int(options.get("iters", 5)), 1)
    seed = int(options.get("seed", 0))
    vitis_python = str(header.get("_vitis_python") or _xdna_python(header))
    command = [
        vitis_python,
        str(_script("benchmark_vitis_resnet.py")), str(model_path),
        "--warmup", str(warmup), "--iters", str(iters), "--seed", str(seed),
        "--json", str(report_path),
    ]
    profile_path = None
    if options.get("profile_vitis", True):
        profile_path = root / "vitis-profile.json"
        command += ["--profile-json", str(profile_path)]
    capture_vitis_path = None
    capture_names = options.get("capture_vitis_outputs", [])
    if capture_names:
        if not isinstance(capture_names, list) or not all(isinstance(name, str) for name in capture_names):
            raise proto.RPCError("capture_vitis_outputs must be a list of ONNX value names")
        capture_vitis_path = root / "vitis-captures.npz"
        command += ["--capture-npz", str(capture_vitis_path)]
        for name in capture_names:
            command += ["--capture-output-name", name]
    vitis_env = os.environ.copy()
    venv_root = Path(vitis_python).expanduser().resolve().parent.parent
    inferred_installation = venv_root if (venv_root / "quicktest").is_dir() else None
    if not vitis_env.get("RYZEN_AI_INSTALLATION_PATH") and inferred_installation:
        vitis_env["RYZEN_AI_INSTALLATION_PATH"] = str(inferred_installation)
    installation = (
        Path(vitis_env["RYZEN_AI_INSTALLATION_PATH"])
        if vitis_env.get("RYZEN_AI_INSTALLATION_PATH") else inferred_installation
    )
    if installation is not None:
        xrt = Path(vitis_env["XILINX_XRT"]) if vitis_env.get("XILINX_XRT") else None
        runtime_libs = [xrt / "lib"] if xrt is not None else []
        runtime_libs += [
            *installation.glob("lib/python*/site-packages/voe/lib"),
            installation / "deployment/lib",
            installation / "onnxruntime/lib",
        ]
        existing = [str(path) for path in runtime_libs if path.is_dir()]
        inherited = [path for path in vitis_env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if path]
        vitis_env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(existing + inherited))
    vitis_process = _run(command, env=vitis_env)
    try:
        vitis = json.loads(report_path.read_text(encoding="utf-8"))
        if profile_path is not None:
            vitis["profile"] = json.loads(profile_path.read_text(encoding="utf-8"))
        if capture_vitis_path is not None and capture_vitis_path.is_file():
            vitis["capture_npz"] = str(capture_vitis_path)
    except (OSError, json.JSONDecodeError) as error:
        raise proto.RPCError(f"Vitis AI benchmark did not produce valid reports: {error}") from error
    if vitis.get("execution") != "real_npu":
        vitis["provider_log_tail"] = (vitis_process.stdout + "\n" + vitis_process.stderr)[-8000:]

    xdna = xdna_result["report"]
    same_input_shape = xdna.get("input_shape") == vitis.get("input_shape")
    same_input_seed = xdna.get("input_seed") == vitis.get("input_seed")
    xdna_npu = xdna.get("execution") in {
        "full_graph_xdna_conv_host_ops", "full_graph_hybrid_conv_host_ops",
        "full_graph_with_fused_bottleneck",
    }
    valid = (
        xdna_npu
        and same_input_shape
        and same_input_seed
        and vitis.get("execution") == "real_npu"
        and int(vitis.get("vitis_npu_node_count") or 0) > 0
    )
    result = {
        "comparison_valid": valid,
        "input_seed": seed,
        "warmup": warmup,
        "iters": iters,
        "same_server_sequential_runs": True,
        "same_input_shape": same_input_shape,
        "same_input_seed": same_input_seed,
        "xdna": xdna,
        "vitis": vitis,
        "comparison_note": (
            "Both reports are full-graph timings, and the Vitis profile confirms NPU-assigned nodes. XDNA may include the configured CPU Conv fallback."
            if valid else
            "Not a verified matched NPU comparison: check input shapes, XDNA graph execution, VitisAIExecutionProvider, and Vitis NPU node events (enable profile_vitis)."
        ),
    }
    if valid and vitis.get("avg_ms"):
        result["xdna_vs_vitis_latency_ratio"] = float(xdna["avg_ms"]) / float(vitis["avg_ms"])
        result["xdna_speedup_vs_vitis"] = float(vitis["avg_ms"]) / float(xdna["avg_ms"])
    return result, []
