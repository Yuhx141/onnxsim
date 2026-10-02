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
from . import xdna_cache as cache


def _script(name: str) -> Path:
    path = Path(__file__).resolve().parents[2] / "scripts" / "xdna" / name
    if not path.is_file():
        raise proto.RPCError(f"XDNA RPC needs the source checkout script {path}")
    return path


def _xdna_python(header: Dict[str, Any]) -> str:
    return str(header.get("_xdna_python") or sys.executable)


def _run(
    command: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, check=False, env=env
        )
    except OSError as error:
        raise proto.RPCError(
            f"could not start XDNA command {command[1]}: {error}"
        ) from error
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


def _body_groups(groups: Any) -> list[Any]:
    """Validate ``options.groups``.

    Each entry is a non-empty list of block prefixes (one core column per group) or
    ``{"blocks": [...], "chunk_cap": bytes, "depth": weight_fifo_depth}``.
    """

    def blocks_of(group: Any) -> Any:
        return group.get("blocks") if isinstance(group, dict) else group

    if not isinstance(groups, list) or not 1 <= len(groups) <= 8:
        raise proto.RPCError(
            "resnet_body needs options.groups: 1-8 block-prefix groups"
        )
    for group in groups:
        blocks = blocks_of(group)
        if not (
            isinstance(blocks, list)
            and blocks
            and all(isinstance(prefix, str) for prefix in blocks)
        ):
            raise proto.RPCError(
                "each resnet_body group needs a non-empty list of block prefixes"
            )
    return groups


_KINDS = (
    "resnet",
    "fused_bottleneck",
    "fused_stage",
    "resnet_body",
    "resnet_network",
    "resnet_engine",
    "graph_engine",
    "maxpool_u8",
)


def _compile_command(
    header: Dict[str, Any], kind: str, options: Dict[str, Any], root: Path
) -> tuple[list[str], Path | None, Path | None, Path | None]:
    """Validate options and build the compiler command for ``root``.

    Returns ``(command, xclbin, insts, manifest_path)``; ``manifest_path`` is set only for the
    ``resnet`` kind and ``xclbin``/``insts`` only for the others.  Pure (no filesystem writes), so
    the cache can call it with a placeholder root to derive the key.
    """
    model_path = root / "model.onnx"
    artifacts = root / "artifacts"

    if kind == "resnet":
        example = Path(str(options.get("example", ""))).expanduser()
        if not example.is_file():
            raise proto.RPCError(
                "resnet compile requires an existing server-side IRON example path"
            )
        manifest_path = root / "manifest.json"
        command = [
            _xdna_python(header),
            str(_script("compile_resnet_kernels.py")),
            str(model_path),
            str(example),
            str(manifest_path),
            "--artifact-dir",
            str(artifacts),
            "--device",
            str(options.get("device", "npu2")),
            "--columns",
            str(int(options.get("columns", 8))),
        ]
        if options.get("compile_all"):
            command.append("--compile-all")
        if options.get("optimize_small_m", True):
            command.append("--optimize-small-m")
        return command, None, None, manifest_path

    xclbin, insts = artifacts / f"{kind}.xclbin", artifacts / f"{kind}.insts.bin"
    if kind == "fused_bottleneck":
        block = str(options.get("block", ""))
        if not block:
            raise proto.RPCError("fused_bottleneck compile requires options.block")
        command = [
            _xdna_python(header),
            str(_script("fused_bottleneck_design.py")),
            "--dev",
            str(options.get("device", "npu2")),
            "--model",
            str(model_path),
            "--block",
            block,
            "--xclbin-path",
            str(xclbin),
            "--insts-path",
            str(insts),
        ]
    elif kind == "fused_stage":
        blocks = options.get("blocks")
        limit = 8 if options.get("blocked") else 3
        if (
            not isinstance(blocks, list)
            or not 1 <= len(blocks) <= limit
            or not all(isinstance(value, str) for value in blocks)
        ):
            raise proto.RPCError(
                f"fused_stage compile requires options.blocks with 1-{limit} block prefixes"
            )
        command = [
            _xdna_python(header),
            str(_script("linked_bottleneck_stage_design.py")),
            "--dev",
            str(options.get("device", "npu2")),
            "--model",
            str(model_path),
            "--blocks",
            *blocks,
            "--xclbin-path",
            str(xclbin),
            "--insts-path",
            str(insts),
        ]
        if options.get("blocked"):
            command.append("--blocked")
    elif kind == "resnet_network":
        # Stem Conv + MaxPool + every bottleneck stage (one core column per stage) as ONE xclbin.
        stages = _body_groups(options.get("stages"))
        command = [
            _xdna_python(header),
            str(_script("resnet_stage_design.py")),
            "--dev",
            str(options.get("device", "npu2")),
            "--model",
            str(model_path),
            "--xclbin-path",
            str(xclbin),
            "--insts-path",
            str(insts),
        ]
        if options.get("stem", True):
            command.append("--stem")
        if options.get("cols"):
            command += ["--cols", str(int(options["cols"]))]
        for stage in stages:
            command += [
                "--stage",
                *(stage["blocks"] if isinstance(stage, dict) else stage),
            ]
        # Per-stage tuning (lists with one entry per stage): per-core weight streams and weight
        # FIFO depths. Weight layout does not depend on either, so a run needs no matching options.
        for option, flag in (
            ("split_weights", "--split-weights"),
            ("weight_depths", "--weight-depths"),
        ):
            values = options.get(option)
            if values is not None:
                if not isinstance(values, list) or len(values) != len(stages):
                    raise proto.RPCError(
                        f"{option} needs one entry per stage ({len(stages)})"
                    )
                command += [flag, ",".join(str(int(v)) for v in values)]
    elif kind == "resnet_engine":
        # Layer-sequential engine: stem Conv + MaxPool + every conv layer as jobs over all 32 cores.
        # The artifact depends only on the ResNet-50 job structure (weights are packed at run time).
        command = [
            _xdna_python(header),
            str(_script("layer_engine_design.py")),
            "--dev",
            str(options.get("device", "npu2")),
            "--net",
            "full" if options.get("stem", True) else "bodyr",
            "--slot",
            "4096",  # must match layer_engine.ENGINE_SLOT_BYTES, which the runner packs with
            "--looped",
            "--l2",
            str(int(options.get("l2", 2))),
            "--xclbin-path",
            str(xclbin),
            "--insts-path",
            str(insts),
        ]
    elif kind == "graph_engine":
        # Layer engine for an arbitrary QDQ CNN (YOLO, MobileNet-style, ...): the job structure is compiled from the
        # model itself (layer_engine_graph.compile_graph), so the artifact is model specific.
        command = [
            _xdna_python(header),
            str(_script("layer_engine_design.py")),
            "--dev",
            str(options.get("device", "npu2")),
            "--net",
            f"onnx:{model_path}",
            "--slot",
            "4096",
            "--l2",
            str(int(options.get("l2", 2))),
            "--xclbin-path",
            str(xclbin),
            "--insts-path",
            str(insts),
        ]
    elif kind == "resnet_body":
        groups = _body_groups(options.get("groups"))
        command = [
            _xdna_python(header),
            str(_script("resnet_body_design.py")),
            "--dev",
            str(options.get("device", "npu2")),
            "--model",
            str(model_path),
            "--xclbin-path",
            str(xclbin),
            "--insts-path",
            str(insts),
        ]
        if options.get("rt"):
            command.append(
                "--rt"
            )  # runtime-shaped kernels (geometry from a per-chunk descriptor)
        for group in groups:
            command += [
                "--group",
                *(group["blocks"] if isinstance(group, dict) else group),
            ]
        caps = [
            int(g.get("chunk_cap") or 0) if isinstance(g, dict) else 0 for g in groups
        ]
        depths = [int(g.get("depth", 1)) if isinstance(g, dict) else 1 for g in groups]
        if any(caps):
            command += ["--chunk-caps", ",".join(map(str, caps))]
        if any(depth != 1 for depth in depths):
            command += ["--weight-depths", ",".join(map(str, depths))]
    else:
        required = (
            "channels",
            "input_height",
            "input_width",
            "output_height",
            "output_width",
            "kernel_height",
            "kernel_width",
            "stride_height",
            "stride_width",
        )
        missing = [key for key in required if key not in options]
        if missing:
            raise proto.RPCError(
                f"maxpool_u8 compile missing options: {', '.join(missing)}"
            )
        command = [
            _xdna_python(header),
            str(_script("maxpool_design.py")),
            "--dev",
            str(options.get("device", "npu2")),
        ]
        cli_names = {
            "channels": "channels",
            "input_height": "input-height",
            "input_width": "input-width",
            "output_height": "output-height",
            "output_width": "output-width",
            "kernel_height": "kernel-height",
            "kernel_width": "kernel-width",
            "stride_height": "stride-height",
            "stride_width": "stride-width",
            "tile_output_rows": "tile-output-rows",
            "tile_channels": "tile-channels",
            "pad_top": "pad-top",
            "pad_left": "pad-left",
            "pad_bottom": "pad-bottom",
            "pad_right": "pad-right",
        }
        defaults = {
            "tile_output_rows": 8,
            "tile_channels": 4,
            "pad_top": 1,
            "pad_left": 1,
            "pad_bottom": 1,
            "pad_right": 1,
        }
        for key, cli_name in cli_names.items():
            if key in options or key in defaults:
                value = options[key] if key in options else defaults[key]
                command.extend([f"--{cli_name}", str(int(value))])
        command += ["--uint8", "--xclbin-path", str(xclbin), "--insts-path", str(insts)]

    return command, xclbin, insts, None


def _execute_compile(
    command: list[str],
    kind: str,
    root: Path,
    xclbin: Path | None,
    insts: Path | None,
    manifest_path: Path | None,
) -> Dict[str, Any]:
    artifacts = root / "artifacts"
    _run(command)
    if manifest_path is not None:
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {"kind": kind, "manifest": result, "artifact_dir": str(artifacts)}
    assert xclbin is not None and insts is not None
    absent = [str(path) for path in (xclbin, insts) if not path.is_file()]
    if absent:
        raise proto.RPCError(
            f"XDNA compiler succeeded but did not create: {', '.join(absent)}"
        )
    return {
        "kind": kind,
        "xclbin": str(xclbin),
        "insts": str(insts),
        "artifact_dir": str(artifacts),
    }


def _scripts_dir() -> Path:
    return _script("compile_resnet_kernels.py").parent


def _cache_key(
    header: Dict[str, Any],
    kind: str,
    options: Dict[str, Any],
    model: bytes,
    command: list[str],
    root: Path,
) -> str:
    """Content-addressed key; see docs/rpc.md ("XDNA compile cache")."""
    example = str(Path(str(options.get("example", ""))).expanduser())
    example_hash = ""
    if kind == "resnet":
        example_hash = cache.sha256_file(Path(example))
    normalized = [
        "@EXAMPLE@"
        if kind == "resnet" and arg == example
        else arg.replace(str(root), "@ROOT@")
        for arg in command
    ]
    scripts = _scripts_dir()
    # The command (device, columns, blocks/groups, blocked, chunk caps, pool geometry, ...)
    # captures every option that reaches the compiler; the example file is hashed by content.
    # This module is hashed too, so changing how options map to commands invalidates entries.
    sources = cache.source_digest([scripts]) + cache.sha256_file(Path(__file__))
    return cache.make_key(
        kind,
        model,
        normalized,
        sources,
        cache.toolchain_identity(_xdna_python(header)),
        cache.compile_env(),
        extra={"example": example_hash},
    )


def compile_resnet(header: Dict[str, Any], blobs: list[bytes], work_dir: str):
    """Compile the supported ResNet artifacts on the RPC server's XDNA toolchain.

    Results are cached content-addressed (see :mod:`onnxsim.rpc.xdna_cache`); the reply gains
    ``"cache": "hit" | "miss" | "bypass"`` and, when cached, ``"cache_key"``.
    ``options["no_cache"]`` or ``ONNXSIM_XDNA_CACHE=0`` compiles into a fresh directory under
    ``work_dir`` as before.
    """
    kind = header.get("kind")
    if kind not in _KINDS:
        raise proto.RPCError(f"unsupported XDNA compile kind {kind!r}")
    if not blobs:
        raise proto.RPCError("XDNA compile requires an ONNX model blob")
    options = header.get("options") or {}
    if not isinstance(options, dict):
        raise proto.RPCError("XDNA compile options must be an object")
    model = blobs[0]

    def build(root: Path) -> Dict[str, Any]:
        command, xclbin, insts, manifest = _compile_command(header, kind, options, root)
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        (root / "model.onnx").write_bytes(model)
        try:
            return _execute_compile(command, kind, root, xclbin, insts, manifest)
        finally:
            (root / "model.onnx").unlink(missing_ok=True)

    if not cache.cache_enabled(options):
        root = Path(work_dir) / "xdna-rpc" / uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=False)
        command, xclbin, insts, manifest = _compile_command(header, kind, options, root)
        (root / "model.onnx").write_bytes(model)
        (root / "artifacts").mkdir()
        result = _execute_compile(command, kind, root, xclbin, insts, manifest)
        return {**result, "cache": "bypass"}, []

    probe_root = Path("@ROOT@")
    command = _compile_command(header, kind, options, probe_root)[0]
    key = _cache_key(header, kind, options, model, command, probe_root)

    def required(_root: Path) -> list[str]:
        if kind == "resnet":
            return []
        return [f"artifacts/{kind}.xclbin", f"artifacts/{kind}.insts.bin"]

    result, status = cache.get_or_build(
        cache.cache_root(options, work_dir),
        key,
        build,
        required,
        cache.max_entries(options),
    )
    return {**result, "cache": status, "cache_key": key}, []


def run_resnet(header: Dict[str, Any], blobs: list[bytes], work_dir: str):
    """Run and profile the XDNA graph on the RPC server; return the runner's JSON report."""
    if len(blobs) != 2:
        raise proto.RPCError("XDNA ResNet run requires model and manifest JSON blobs")
    options = header.get("options") or {}
    if not isinstance(options, dict):
        raise proto.RPCError("XDNA run options must be an object")
    root = Path(work_dir) / "xdna-rpc" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    model_path, manifest_path, report_path = (
        root / "model.onnx",
        root / "manifest.json",
        root / "report.json",
    )
    model_path.write_bytes(blobs[0])
    manifest_path.write_bytes(blobs[1])
    graph_engine = options.get("graph_engine")
    if graph_engine is not None:
        if not isinstance(graph_engine, dict) or not all(
            key in graph_engine for key in ("xclbin", "insts")
        ):
            raise proto.RPCError("graph_engine needs xclbin and insts")
        command = [
            _xdna_python(header),
            str(_script("run_graph_engine.py")),
            str(model_path),
            str(graph_engine["xclbin"]),
            str(graph_engine["insts"]),
            "--warmup",
            str(max(int(options.get("warmup", 3)), 0)),
            "--iters",
            str(max(int(options.get("iters", 20)), 1)),
            "--seed",
            str(int(options.get("seed", 0))),
            "--json",
            str(report_path),
        ]
        if options.get("check"):
            command.append("--check")
        _run(command)
        try:
            return {"report": json.loads(report_path.read_text(encoding="utf-8"))}, []
        except (OSError, json.JSONDecodeError) as error:
            raise proto.RPCError(
                f"graph engine runner did not produce a valid JSON report: {error}"
            ) from error
    command = [
        _xdna_python(header),
        str(_script("run_resnet_xdna.py")),
        str(model_path),
        str(manifest_path),
        "--warmup",
        str(max(int(options.get("warmup", 1)), 0)),
        "--iters",
        str(max(int(options.get("iters", 5)), 1)),
        "--seed",
        str(int(options.get("seed", 0))),
        "--cpu-backend",
        str(options.get("cpu_backend", "numpy")),
        "--cpu-threads",
        str(max(int(options.get("cpu_threads", 2)), 1)),
        "--json",
        str(report_path),
    ]
    capture_path = None
    if options.get("capture_outputs"):
        capture_path = root / "xdna-captures.npz"
        command += ["--capture-npz", str(capture_path)]
    if options.get("cpu_small_m") is not None:
        command += ["--cpu-small-m", str(max(int(options["cpu_small_m"]), 0))]
    for block in options.get("fused_blocks", []):
        if not isinstance(block, dict) or not all(
            key in block for key in ("prefix", "xclbin", "insts")
        ):
            raise proto.RPCError(
                "each fused_blocks entry needs prefix, xclbin, and insts"
            )
        command += [
            "--fused-block",
            str(block["prefix"]),
            str(block["xclbin"]),
            str(block["insts"]),
        ]
    for stage in options.get("fused_stages", []):
        if not isinstance(stage, dict) or not all(
            key in stage for key in ("blocks", "xclbin", "insts")
        ):
            raise proto.RPCError(
                "each fused_stages entry needs blocks, xclbin, and insts"
            )
        blocks = stage["blocks"]
        limit = 8 if stage.get("blocked") else 3
        if (
            not isinstance(blocks, list)
            or not 1 <= len(blocks) <= limit
            or not all(isinstance(value, str) for value in blocks)
        ):
            raise proto.RPCError(f"each fused stage needs 1-{limit} block prefixes")
        command += ["--fused-stage", *blocks, str(stage["xclbin"]), str(stage["insts"])]
        if stage.get("blocked") and "--fused-stage-blocked" not in command:
            command.append("--fused-stage-blocked")
    body = options.get("fused_body")
    if body is not None:
        if not isinstance(body, dict) or not all(
            key in body for key in ("xclbin", "insts", "groups")
        ):
            raise proto.RPCError("fused_body needs xclbin, insts, and groups")
        command += [
            "--fused-body",
            str(body["xclbin"]),
            str(body["insts"]),
            json.dumps(_body_groups(body["groups"])),
        ]
        if body.get("rt"):
            command.append("--fused-body-rt")
    network = options.get("device_network")
    if network is not None:
        if not isinstance(network, dict) or not all(
            key in network for key in ("xclbin", "insts", "stages")
        ):
            raise proto.RPCError("device_network needs xclbin, insts, and stages")
        stages = [
            s["blocks"] if isinstance(s, dict) else s
            for s in _body_groups(network["stages"])
        ]
        command += [
            "--device-network",
            str(network["xclbin"]),
            str(network["insts"]),
            json.dumps(stages),
        ]
    engine = options.get("layer_engine")
    if engine is not None:
        if not isinstance(engine, dict) or not all(
            key in engine for key in ("xclbin", "insts", "stages")
        ):
            raise proto.RPCError("layer_engine needs xclbin, insts, and stages")
        stages = [
            s["blocks"] if isinstance(s, dict) else s
            for s in _body_groups(engine["stages"])
        ]
        command += [
            "--layer-engine",
            str(engine["xclbin"]),
            str(engine["insts"]),
            json.dumps(stages),
        ]
        if engine.get("stem", True):
            command.append("--layer-engine-stem")
    if options.get("host_maxpool"):
        command.append("--host-maxpool")
    pool = options.get("maxpool_uint8")
    runtime_backend = options.get(
        "runtime_backend", options.get("maxpool_runtime", "iron")
    )
    if runtime_backend not in ("iron", "xrt"):
        raise proto.RPCError("runtime_backend must be 'iron' or 'xrt'")
    if runtime_backend == "xrt":
        command += ["--runtime-backend", "xrt"]
    if pool is not None:
        if not isinstance(pool, dict) or not all(
            key in pool for key in ("xclbin", "insts")
        ):
            raise proto.RPCError("maxpool_uint8 needs xclbin and insts")
        command += [
            "--maxpool-uint8-xclbin",
            str(pool["xclbin"]),
            "--maxpool-uint8-insts",
            str(pool["insts"]),
        ]
    runtime_backend = str(
        options.get("runtime_backend", options.get("maxpool_runtime", "iron"))
    )
    if (
        runtime_backend == "xrt"
        and Path(_xdna_python(header)).resolve() == Path(sys.executable).resolve()
    ):
        _run_inprocess(command)
    else:
        _run(command)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise proto.RPCError(
            f"XDNA runner did not produce a valid JSON report: {error}"
        ) from error
    if capture_path is not None and capture_path.is_file():
        report["capture_npz"] = str(capture_path)
    if runtime_backend == "xrt" and "xdna_xrt_runtime" in sys.modules:
        runtime_module = sys.modules["xdna_xrt_runtime"]
        report["xrt_kernel_cache"] = runtime_module.load_kernel.cache_info()._asdict()
    return {"report": report}, []


def compare_resnet(header: Dict[str, Any], blobs: list[bytes], work_dir: str):
    """Run XDNA and Vitis AI sequentially with matched inputs and timing settings."""
    if len(blobs) != 2:
        raise proto.RPCError(
            "XDNA/Vitis comparison requires model and manifest JSON blobs"
        )
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
        str(_script("benchmark_vitis_resnet.py")),
        str(model_path),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--seed",
        str(seed),
        "--json",
        str(report_path),
    ]
    profile_path = None
    if options.get("profile_vitis", True):
        profile_path = root / "vitis-profile.json"
        command += ["--profile-json", str(profile_path)]
    capture_vitis_path = None
    capture_names = options.get("capture_vitis_outputs", [])
    if capture_names:
        if not isinstance(capture_names, list) or not all(
            isinstance(name, str) for name in capture_names
        ):
            raise proto.RPCError(
                "capture_vitis_outputs must be a list of ONNX value names"
            )
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
        if vitis_env.get("RYZEN_AI_INSTALLATION_PATH")
        else inferred_installation
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
        inherited = [
            path
            for path in vitis_env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if path
        ]
        vitis_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            dict.fromkeys(existing + inherited)
        )
    vitis_process = _run(command, env=vitis_env)
    try:
        vitis = json.loads(report_path.read_text(encoding="utf-8"))
        if profile_path is not None:
            vitis["profile"] = json.loads(profile_path.read_text(encoding="utf-8"))
        if capture_vitis_path is not None and capture_vitis_path.is_file():
            vitis["capture_npz"] = str(capture_vitis_path)
    except (OSError, json.JSONDecodeError) as error:
        raise proto.RPCError(
            f"Vitis AI benchmark did not produce valid reports: {error}"
        ) from error
    if vitis.get("execution") != "real_npu":
        vitis["provider_log_tail"] = (
            vitis_process.stdout + "\n" + vitis_process.stderr
        )[-8000:]

    xdna = xdna_result["report"]
    same_input_shape = xdna.get("input_shape") == vitis.get("input_shape")
    same_input_seed = xdna.get("input_seed") == vitis.get("input_seed")
    xdna_npu = xdna.get("execution") in {
        "full_graph_xdna_conv_host_ops",
        "full_graph_hybrid_conv_host_ops",
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
            if valid
            else "Not a verified matched NPU comparison: check input shapes, XDNA graph execution, VitisAIExecutionProvider, and Vitis NPU node events (enable profile_vitis)."
        ),
    }
    if valid and vitis.get("avg_ms"):
        result["xdna_vs_vitis_latency_ratio"] = float(xdna["avg_ms"]) / float(
            vitis["avg_ms"]
        )
        result["xdna_speedup_vs_vitis"] = float(vitis["avg_ms"]) / float(xdna["avg_ms"])
    return result, []
