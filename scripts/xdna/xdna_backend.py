"""Phase 1 AMD XDNA backend boundary.

This module intentionally has no import-time dependency on MIGraphX, ROCm,
XRT, or mlir-aie.  It provides the part that is useful before a physical NPU
is available: operator legality, graph partitioning, and artifact discovery.

The eventual execution path is expected to consume precompiled XDNA artifacts
(``xclbin`` plus an instruction stream) through XRT.  Keeping that boundary
explicit prevents CPU fallback from being confused with NPU execution.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

# These are intentionally conservative.  MatMul and Conv are the first
# compute kernels; the remaining operators are cheap to implement as fused
# vector kernels or as graph-level layout operations.
CORE_OPS = frozenset(
    {
        "Add",
        "Mul",
        "Sub",
        "Div",
        "Relu",
        "Gelu",
        "Sigmoid",
        "Max",
        "Min",
        "MatMul",
        "Gemm",
        "Conv",
        "Reshape",
        "Transpose",
        "Identity",
        "Cast",
        "ReduceSum",
        "Softmax",
    }
)

FUSIBLE_ELEMENTWISE_OPS = frozenset(
    {"Add", "Mul", "Sub", "Div", "Relu", "Gelu", "Sigmoid", "Max", "Min"}
)
FUSIBLE_PRODUCERS = frozenset({"MatMul", "Gemm", "Conv"})

# Empirical starting points from the real Strix Halo sweep.  These are hints,
# not architectural guarantees; select_matmul_tile validates divisibility and
# falls back to smaller legal tiles.
MATMUL_TILE_HINTS = {
    "i8": ((64, 64, 64), (64, 32, 64), (64, 32, 32), (32, 32, 32)),
    "int8": ((64, 64, 64), (64, 32, 64), (64, 32, 32), (32, 32, 32)),
    "i16": ((64, 32, 32), (32, 32, 32)),
    "int16": ((64, 32, 32), (32, 32, 32)),
    "bf16": ((64, 32, 32), (32, 32, 32)),
    "float16": ((64, 32, 32), (32, 32, 32)),
}

_DTYPE_ALIASES = {
    "int8": "i8",
    "uint8": "u8",
    "int16": "i16",
    "int32": "i32",
    "uint16": "u16",
    "float16": "f16",
    "float32": "f32",
    "char": "i8",
    "signed char": "i8",
    "uchar": "u8",
    "unsigned char": "u8",
    "short": "i16",
    "int": "i32",
    "long": "i64",
    "half": "f16",
    "float": "f32",
    "double": "f64",
    "dtypes.char": "i8",
    "dtypes.uchar": "u8",
    "dtypes.short": "i16",
    "dtypes.int": "i32",
    "dtypes.long": "i64",
    "dtypes.half": "f16",
    "dtypes.float": "f32",
    "dtypes.double": "f64",
    "dtypes.bool": "bool",
}


def _normalize_dtype(dtype: Any) -> str:
    # ``.dtype`` is not always a string here.  NumPy *dtype instances* have
    # ``name`` (``np.dtype(np.int8).name == 'int8'``), NumPy *scalar-type
    # classes* have ``__name__`` (``np.int8.__name__ == 'int8'``), and
    # tinygrad ``DType`` objects have a C-ish ``name``
    # (``dtypes.int8.name == 'signed char'``).  Plain ``str()`` is the last
    # resort: ``str(np.int8)`` is ``"<class 'numpy.int8'>"``.
    name = getattr(dtype, "name", None)
    if not isinstance(name, str):
        name = getattr(dtype, "__name__", None)
    if isinstance(name, str):
        value = name.lower()
        if value.startswith("numpy."):
            value = value[len("numpy.") :]
    else:
        value = str(dtype).lower().replace("numpy.", "")
    return _DTYPE_ALIASES.get(value, value)


class XDNAUnavailable(RuntimeError):
    """Raised when execution was requested without a usable XDNA artifact."""


@dataclass(frozen=True)
class NodeInfo:
    index: int
    name: str
    op_type: str
    supported: bool


@dataclass(frozen=True)
class Partition:
    """A contiguous graph segment assigned to XDNA or fallback execution."""

    device: str
    nodes: Tuple[NodeInfo, ...]

    @property
    def supported(self) -> bool:
        return self.device == "XDNA"


@dataclass(frozen=True)
class FusedGroup:
    """One planned device dispatch after legal post-op fusion."""

    nodes: Tuple[NodeInfo, ...]
    kernel_kind: str

    @property
    def op_types(self) -> Tuple[str, ...]:
        return tuple(node.op_type for node in self.nodes)


@dataclass(frozen=True)
class MatmulDispatchPlan:
    dtype: str
    output_dtype: str
    shape: Tuple[int, int, int]
    columns: int
    tile: Tuple[int, int, int]
    kernel: str


@dataclass(frozen=True)
class TensorDescriptor:
    shape: Tuple[int, ...]
    dtype: str
    contiguous: Optional[bool]


@dataclass(frozen=True)
class TransferPlan:
    strategy: str
    chunk_bytes: int
    double_buffered: bool
    host_round_trips: int
    estimated_bytes: int


def plan_transfer(
    input_bytes: int,
    output_bytes: int,
    *,
    reuse_count: int = 1,
    input_resident: bool = False,
    output_resident: bool = False,
    local_capacity_bytes: int = 64 * 1024,
    chunk_bytes: int = 32 * 1024,
) -> TransferPlan:
    """Choose a conservative XDNA DMA strategy from traffic and reuse.

    ``weight_resident`` is selected only when the reusable input fits in local
    storage. Larger tensors use ping-pong (double-buffered) streaming to hide
    DMA behind compute. The byte estimate counts host traffic, not internal
    tile-to-tile movement.
    """
    if (
        min(input_bytes, output_bytes, reuse_count, local_capacity_bytes, chunk_bytes)
        <= 0
    ):
        raise ValueError("transfer sizes, reuse count, and capacities must be positive")
    host_input = 0 if input_resident else input_bytes
    host_output = 0 if output_resident else output_bytes
    if input_bytes <= local_capacity_bytes and reuse_count > 1:
        return TransferPlan(
            "weight_resident", input_bytes, False, 1, host_input + host_output
        )
    total = host_input + host_output
    if total == 0:
        return TransferPlan("device_resident", 0, False, 0, 0)
    if total >= 2 * chunk_bytes:
        chunks = (total + chunk_bytes - 1) // chunk_bytes
        return TransferPlan("double_buffered_stream", chunk_bytes, True, chunks, total)
    return TransferPlan("stream", total, False, 1, total)


def plan_pipeline_transfer(
    input_bytes: int,
    output_bytes: int,
    intermediate_bytes: Sequence[int],
    *,
    chunk_bytes: int = 32 * 1024,
) -> TransferPlan:
    """Plan host traffic for a chain whose intermediates stay on the NPU."""
    if min(input_bytes, output_bytes, chunk_bytes) <= 0 or any(
        value <= 0 for value in intermediate_bytes
    ):
        raise ValueError("pipeline transfer sizes must be positive")
    host_bytes = input_bytes + output_bytes
    if host_bytes >= 2 * chunk_bytes:
        chunks = (host_bytes + chunk_bytes - 1) // chunk_bytes
        strategy = "resident_pipeline_double_buffered"
        buffered = True
    else:
        chunks = 1
        strategy = "resident_pipeline"
        buffered = False
    return TransferPlan(strategy, chunk_bytes, buffered, chunks, host_bytes)


def describe_tensor(tensor: Any) -> TensorDescriptor:
    """Read the minimal shape/dtype protocol shared by numpy/tinygrad/IRON."""
    if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
        raise TypeError("tensor must expose shape and dtype")
    contiguous = None
    if hasattr(tensor, "flags") and hasattr(tensor.flags, "c_contiguous"):
        contiguous = bool(tensor.flags.c_contiguous)
    elif hasattr(tensor, "is_contiguous"):
        value = tensor.is_contiguous
        contiguous = bool(value() if callable(value) else value)
    return TensorDescriptor(
        shape=tuple(int(value) for value in tensor.shape),
        dtype=_normalize_dtype(tensor.dtype),
        contiguous=contiguous,
    )


def validate_matmul_buffers(
    plan: MatmulDispatchPlan,
    a: Any,
    b: Any,
    c: Any,
) -> Tuple[TensorDescriptor, TensorDescriptor, TensorDescriptor]:
    """Validate buffers before XDNA dispatch; no device access is performed."""
    desc_a, desc_b, desc_c = (describe_tensor(value) for value in (a, b, c))
    expected = plan.shape
    if desc_a.shape != (expected[0], expected[1]) or desc_b.shape != (
        expected[1],
        expected[2],
    ):
        raise ValueError(f"buffers do not match planned MatMul shape {expected}")
    if desc_c.shape != (expected[0], expected[2]):
        raise ValueError("output buffer does not match planned MatMul shape")
    input_dtype = _normalize_dtype(plan.dtype)
    output_dtype = _normalize_dtype(plan.output_dtype)
    if desc_a.dtype != input_dtype or desc_b.dtype != input_dtype:
        raise ValueError(f"XDNA MatMul inputs require dtype {plan.dtype!r}")
    if desc_c.dtype != output_dtype:
        raise ValueError(f"XDNA MatMul output requires dtype {plan.output_dtype!r}")
    if any(desc.contiguous is False for desc in (desc_a, desc_b, desc_c)):
        raise ValueError("XDNA MatMul requires row-major contiguous buffers")
    return desc_a, desc_b, desc_c


def select_matmul_tile(
    dtype: str,
    m: int,
    k: int,
    n: int,
    columns: int = 8,
    profile: Optional[Mapping[str, Any]] = None,
) -> Tuple[int, int, int]:
    """Choose the best measured legal starting tile for a GEMM.

    ``m``, ``k``, and ``n`` are the logical matrix dimensions.  A candidate is
    considered legal only when it tiles the dimensions and distributes the N
    dimension across the requested column count.  Unknown dtypes use the
    conservative i16 schedule.
    """
    if min(m, k, n, columns) <= 0:
        raise ValueError("GEMM dimensions and columns must be positive")
    candidates = MATMUL_TILE_HINTS.get(dtype.lower(), MATMUL_TILE_HINTS["i16"])
    if profile:
        measured = profile.get(f"{dtype.lower()}:{m}x{k}x{n}:c{columns}")
        if isinstance(measured, Sequence) and len(measured) == 3:
            candidate = tuple(int(value) for value in measured)
            candidates = (candidate,) + tuple(
                tile for tile in candidates if tile != candidate
            )
    for tile_m, tile_k, tile_n in candidates:
        if m % (tile_m * 4) == 0 and k % tile_k == 0 and n % (tile_n * columns) == 0:
            return tile_m, tile_k, tile_n
    # Last-resort small tile.  The caller can still reject it for a specific
    # AIE dialect, but it is preferable to silently selecting an invalid
    # measured configuration.
    return 32, 32, 32


def matmul_kernel_key(
    dtype: str,
    m: int,
    k: int,
    n: int,
    columns: int = 8,
    profile: Optional[Mapping[str, Any]] = None,
    output_dtype: Optional[str] = None,
) -> str:
    """Return the stable manifest key for a tuned GEMM artifact."""
    tile_m, tile_k, tile_n = select_matmul_tile(dtype, m, k, n, columns, profile)
    normalized = _normalize_dtype(dtype)
    output = _normalize_dtype(output_dtype or dtype)
    output_suffix = "" if output == normalized else f"_o{output}"
    return f"matmul_{normalized}{output_suffix}_m{tile_m}k{tile_k}n{tile_n}_c{columns}"


def plan_matmul(
    shape_a: Sequence[int],
    shape_b: Sequence[int],
    dtype: str,
    columns: int = 8,
    profile: Optional[Mapping[str, Any]] = None,
    output_dtype: Optional[str] = None,
) -> MatmulDispatchPlan:
    """Infer and validate a 2-D ONNX MatMul dispatch plan."""
    if len(shape_a) != 2 or len(shape_b) != 2:
        raise ValueError("XDNA core MatMul currently requires rank-2 inputs")
    m, k_a = (int(value) for value in shape_a)
    k_b, n = (int(value) for value in shape_b)
    if min(m, k_a, k_b, n) <= 0 or k_a != k_b:
        raise ValueError(
            f"incompatible MatMul shapes: {tuple(shape_a)} × {tuple(shape_b)}"
        )
    tile = select_matmul_tile(dtype, m, k_a, n, columns, profile)
    return MatmulDispatchPlan(
        dtype=dtype,
        output_dtype=output_dtype or dtype,
        shape=(m, k_a, n),
        columns=columns,
        tile=tile,
        kernel=matmul_kernel_key(dtype, m, k_a, n, columns, profile, output_dtype),
    )


def resolve_kernel_artifact(
    manifest: Mapping[str, Any],
    key: str,
) -> Optional[Mapping[str, Any]]:
    """Resolve one manifest entry without claiming that it is executable."""
    kernels = manifest.get("kernels") if isinstance(manifest, Mapping) else None
    if not isinstance(kernels, Mapping):
        raise ValueError("XDNA manifest must contain a 'kernels' object")
    artifact = kernels.get(key)
    if artifact is None:
        return None
    if not isinstance(artifact, Mapping):
        raise ValueError(f"XDNA manifest kernel {key!r} must be an object")
    for required in ("xclbin", "insts"):
        if required not in artifact:
            raise ValueError(f"XDNA manifest kernel {key!r} lacks {required!r}")
    return artifact


class XDNAArtifactExecutor:
    """Lazy IRON/XRT launcher for one manifest-selected artifact.

    ``inputs`` and ``outputs`` are IRON tensors.  The launcher intentionally
    does not convert numpy arrays or tinygrad tensors yet; that conversion is
    part of the tensor-planning layer and must preserve layout and dtype.
    """

    def __init__(
        self,
        artifact: Mapping[str, Any],
        *,
        base_dir: Optional[os.PathLike[str] | str] = None,
    ):
        for required in ("xclbin", "insts"):
            if required not in artifact:
                raise ValueError(f"XDNA artifact lacks {required!r}")
        root = Path(base_dir) if base_dir is not None else None
        self.xclbin = self._resolve_path(artifact["xclbin"], root)
        self.insts = self._resolve_path(artifact["insts"], root)
        self._kernel = None

    @staticmethod
    def _resolve_path(value: Any, root: Optional[Path]) -> Path:
        path = Path(os.fspath(value))
        return path if path.is_absolute() or root is None else root / path

    def __call__(self, inputs: Sequence[Any], outputs: Sequence[Any]) -> None:
        if not self.xclbin.is_file():
            raise XDNAUnavailable(f"XDNA xclbin not found: {self.xclbin}")
        if not self.insts.is_file():
            raise XDNAUnavailable(f"XDNA instruction stream not found: {self.insts}")
        try:
            from aie.utils import NPUKernel
        except Exception as exc:
            raise XDNAUnavailable(
                "IRON is required to launch XDNA artifacts; install a compatible "
                "mlir-aie wheel and Peano"
            ) from exc
        try:
            if self._kernel is None:
                self._kernel = NPUKernel(str(self.xclbin), str(self.insts))
            self._kernel(*inputs, *outputs)
        except Exception as exc:
            raise XDNAUnavailable(f"XDNA artifact launch failed: {exc}") from exc


@functools.lru_cache(maxsize=32)
def _cached_executor(xclbin: str, insts: str) -> XDNAArtifactExecutor:
    return XDNAArtifactExecutor({"xclbin": xclbin, "insts": insts})


def dispatch_matmul(
    manifest: Mapping[str, Any],
    a: Any,
    b: Any,
    c: Any,
    *,
    dtype: str,
    output_dtype: Optional[str] = None,
    m: int,
    k: int,
    n: int,
    columns: int = 8,
    base_dir: Optional[os.PathLike[str] | str] = None,
    profile: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Dispatch one tuned GEMM artifact and return its selection metadata.

    The tensor objects are deliberately opaque here.  The caller owns the
    tinygrad↔IRON tensor conversion and must provide buffers matching the
    selected artifact's layout.  Keeping conversion outside this function
    makes the dispatch contract testable without a device runtime.
    """
    key = matmul_kernel_key(dtype, m, k, n, columns, profile, output_dtype)
    artifact = resolve_kernel_artifact(manifest, key)
    if artifact is None:
        return {
            "execution": "fallback",
            "reason": "missing_kernel_artifact",
            "kernel": key,
        }
    plan = plan_matmul((m, k), (k, n), dtype, columns, profile, output_dtype)
    validate_matmul_buffers(plan, a, b, c)
    root = Path(base_dir) if base_dir is not None else None
    executor = _cached_executor(
        str(XDNAArtifactExecutor._resolve_path(artifact["xclbin"], root)),
        str(XDNAArtifactExecutor._resolve_path(artifact["insts"], root)),
    )
    executor([a, b], [c])
    return {
        "execution": "real_npu",
        "kernel": key,
        "tile": plan.tile,
    }


def dispatch_matmul_batch(
    manifest: Mapping[str, Any],
    calls: Sequence[Tuple[Any, Any, Any]],
    *,
    dtype: str,
    m: int,
    k: int,
    n: int,
    columns: int = 8,
    base_dir: Optional[os.PathLike[str] | str] = None,
    profile: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Launch repeated same-shape GEMMs while reusing one cached executor."""
    key = matmul_kernel_key(dtype, m, k, n, columns, profile)
    artifact = resolve_kernel_artifact(manifest, key)
    if artifact is None:
        return {
            "execution": "fallback",
            "reason": "missing_kernel_artifact",
            "kernel": key,
        }
    plan = plan_matmul((m, k), (k, n), dtype, columns, profile)
    root = Path(base_dir) if base_dir is not None else None
    executor = _cached_executor(
        str(XDNAArtifactExecutor._resolve_path(artifact["xclbin"], root)),
        str(XDNAArtifactExecutor._resolve_path(artifact["insts"], root)),
    )
    for a, b, c in calls:
        validate_matmul_buffers(plan, a, b, c)
        executor([a, b], [c])
    return {
        "execution": "real_npu",
        "kernel": key,
        "tile": plan.tile,
        "calls": len(calls),
    }


def fuse_partition(partition: Partition) -> List[FusedGroup]:
    """Fuse safe elementwise post-ops into a compute producer.

    This first optimizer is intentionally conservative: it never fuses two
    producers, and only fuses an elementwise chain immediately following a
    GEMM or convolution.  Shape, aliasing, and dtype checks belong in the
    later tensor planner; this function only operates on an already-legal
    contiguous partition.
    """
    if not partition.supported:
        return [FusedGroup((node,), "fallback") for node in partition.nodes]
    groups: List[FusedGroup] = []
    index = 0
    while index < len(partition.nodes):
        node = partition.nodes[index]
        group = [node]
        if node.op_type in FUSIBLE_PRODUCERS:
            index += 1
            while index < len(partition.nodes):
                post_op = partition.nodes[index]
                if post_op.op_type not in FUSIBLE_ELEMENTWISE_OPS:
                    break
                group.append(post_op)
                index += 1
            groups.append(FusedGroup(tuple(group), node.op_type.lower() + "_fused"))
        else:
            index += 1
            groups.append(FusedGroup((node,), "elementwise"))
    return groups


def optimize_partitions(partitions: Sequence[Partition]) -> List[FusedGroup]:
    """Return the dispatch plan after conservative post-op fusion."""
    groups: List[FusedGroup] = []
    for partition in partitions:
        groups.extend(fuse_partition(partition))
    return groups


def _nodes(model: Any) -> Iterable[Any]:
    """Return graph nodes without importing ONNX at module import time."""
    try:
        return model.graph.node
    except AttributeError as exc:
        raise TypeError("expected an ONNX ModelProto-like object") from exc


def analyze_model(model: Any) -> List[NodeInfo]:
    """Describe operator support without touching hardware or external SDKs."""
    result: List[NodeInfo] = []
    for index, node in enumerate(_nodes(model)):
        op_type = str(node.op_type)
        result.append(
            NodeInfo(
                index=index,
                name=str(getattr(node, "name", "")) or f"{op_type}_{index}",
                op_type=op_type,
                supported=op_type in CORE_OPS,
            )
        )
    return result


def partition_model(model: Any) -> List[Partition]:
    """Partition a graph into contiguous XDNA and fallback segments.

    This is intentionally a conservative first pass.  A later planner can
    fuse compatible nodes and account for tensor lifetimes, tile memory, and
    DMA costs without changing this public result shape.
    """
    partitions: List[Partition] = []
    current_device: Optional[str] = None
    current: List[NodeInfo] = []
    for info in analyze_model(model):
        device = "XDNA" if info.supported else "FALLBACK"
        if current and device != current_device:
            partitions.append(Partition(current_device or "FALLBACK", tuple(current)))
            current = []
        current_device = device
        current.append(info)
    if current:
        partitions.append(Partition(current_device or "FALLBACK", tuple(current)))
    return partitions


def load_kernel_manifest(path: os.PathLike[str] | str) -> Mapping[str, Any]:
    """Load an offline kernel manifest, validating its basic shape.

    A manifest describes artifacts produced offline by IRON/MLIR-AIE.  It is
    not an executable format and must never be treated as proof that hardware
    is present.
    """
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("kernels"), dict):
        raise ValueError("XDNA manifest must contain a 'kernels' object")
    return manifest


def load_tuning_profile(path: os.PathLike[str] | str) -> Mapping[str, Any]:
    """Load and validate a profile emitted by ``benchmark_gemm.py``."""
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    profile = data.get("profile") if isinstance(data, Mapping) else None
    if not isinstance(profile, Mapping):
        raise ValueError("XDNA tuning report must contain a 'profile' object")
    for key, tile in profile.items():
        if (
            not isinstance(key, str)
            or not isinstance(tile, Sequence)
            or isinstance(tile, (str, bytes))
            or len(tile) != 3
            or any(int(value) <= 0 for value in tile)
        ):
            raise ValueError(f"invalid XDNA tuning profile entry: {key!r}")
    return profile


def execute(*_: Any, **__: Any) -> None:
    """Reserved execution boundary for the Phase 2 XRT launcher."""
    raise XDNAUnavailable(
        "XDNA execution is not enabled yet; provide Phase 2 XRT artifacts "
        "and launcher support"
    )


# Hardware probing is opt-in.  Importing this module must remain safe on CI,
# developer machines, and systems without AMD NPU libraries.
XDNA_AVAILABLE = False
