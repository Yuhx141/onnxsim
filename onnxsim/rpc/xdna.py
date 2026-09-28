"""Server-side XDNA ResNet compile and run operations for onnxsim RPC."""

from __future__ import annotations

import json
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


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as error:
        raise proto.RPCError(f"could not start XDNA command {command[1]}: {error}") from error
    if result.returncode:
        detail = (result.stdout + "\n" + result.stderr)[-16000:]
        raise proto.RPCError(
            f"XDNA command failed (exit {result.returncode}): {' '.join(command[:2])}\n{detail}"
        )
    return result


def compile_resnet(header: Dict[str, Any], blobs: list[bytes], work_dir: str):
    """Compile the supported ResNet artifacts on the RPC server's XDNA toolchain."""
    kind = header.get("kind")
    if kind not in ("resnet", "fused_bottleneck", "maxpool_u8"):
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
            sys.executable, str(_script("compile_resnet_kernels.py")), str(model_path),
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
            sys.executable, str(_script("fused_bottleneck_design.py")), "--dev",
            str(options.get("device", "npu2")), "--model", str(model_path), "--block", block,
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
            sys.executable, str(_script("maxpool_design.py")), "--dev",
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
        sys.executable, str(_script("run_resnet_xdna.py")), str(model_path), str(manifest_path),
        "--warmup", str(max(int(options.get("warmup", 1)), 0)),
        "--iters", str(max(int(options.get("iters", 5)), 1)),
        "--cpu-backend", str(options.get("cpu_backend", "numpy")),
        "--cpu-threads", str(max(int(options.get("cpu_threads", 2)), 1)),
        "--json", str(report_path),
    ]
    if options.get("cpu_small_m") is not None:
        command += ["--cpu-small-m", str(max(int(options["cpu_small_m"]), 0))]
    for block in options.get("fused_blocks", []):
        if not isinstance(block, dict) or not all(key in block for key in ("prefix", "xclbin", "insts")):
            raise proto.RPCError("each fused_blocks entry needs prefix, xclbin, and insts")
        command += ["--fused-block", str(block["prefix"]), str(block["xclbin"]), str(block["insts"])]
    pool = options.get("maxpool_uint8")
    if pool is not None:
        if not isinstance(pool, dict) or not all(key in pool for key in ("xclbin", "insts")):
            raise proto.RPCError("maxpool_uint8 needs xclbin and insts")
        command += ["--maxpool-uint8-xclbin", str(pool["xclbin"]), "--maxpool-uint8-insts", str(pool["insts"])]
    _run(command)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise proto.RPCError(f"XDNA runner did not produce a valid JSON report: {error}") from error
    return {"report": report}, []
