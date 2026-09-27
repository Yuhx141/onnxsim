"""tinygrad tensor bridge for the XDNA backend.

This is the conversion layer that :mod:`xdna_backend` deliberately leaves to
the caller: tinygrad tensors (or NumPy arrays) become IRON NPU tensors, and a
planned MatMul dispatches through the manifest-selected artifact via
:func:`xdna_backend.dispatch_matmul`.

``tinygrad`` and IRON/``mlir-aie`` are both optional dependencies.  Nothing
here imports them at module import time; :func:`plan_tinygrad_matmul` only
reads ``.shape`` and works with any tensor-like object.  Device execution
(:func:`run_tinygrad_matmul`) additionally requires XRT and a compiled
``xclbin``/``insts`` artifact for the planned shape.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np

try:
    from .xdna_backend import (
        MatmulDispatchPlan,
        XDNAUnavailable,
        dispatch_matmul,
        matmul_kernel_key,
        plan_matmul,
        resolve_kernel_artifact,
    )
except ImportError:  # direct script-directory imports used by tooling/tests
    from xdna_backend import (
        MatmulDispatchPlan,
        XDNAUnavailable,
        dispatch_matmul,
        matmul_kernel_key,
        plan_matmul,
        resolve_kernel_artifact,
    )


TINYGRAD_DTYPE_TO_NUMPY = {
    "dtypes.char": "int8",
    "dtypes.uchar": "uint8",
    "dtypes.short": "int16",
    "dtypes.int": "int32",
    "dtypes.long": "int64",
    "dtypes.half": "float16",
    "dtypes.float": "float32",
    "dtypes.double": "float64",
    "dtypes.bool": "bool",
}

XDNA_DTYPE_TO_NUMPY = {
    "i8": "int8",
    "int8": "int8",
    "u8": "uint8",
    "uint8": "uint8",
    "i16": "int16",
    "int16": "int16",
    "i32": "int32",
    "int32": "int32",
    "f16": "float16",
    "float16": "float16",
    "f32": "float32",
    "float32": "float32",
}


def tinygrad_dtype_to_numpy(dtype: Any) -> np.dtype:
    """Map a tinygrad ``DType`` to the NumPy dtype holding its values."""
    text = str(dtype)
    try:
        return np.dtype(TINYGRAD_DTYPE_TO_NUMPY[text])
    except KeyError:
        raise ValueError(
            f"unsupported tinygrad dtype for XDNA transfer: {text!r}"
        ) from None


def xdna_dtype_to_numpy(dtype: str) -> np.dtype:
    """Map a backend plan dtype (``i8``, ``int16``, ...) to NumPy."""
    try:
        return np.dtype(XDNA_DTYPE_TO_NUMPY[str(dtype).lower()])
    except KeyError:
        raise ValueError(
            f"dtype {dtype!r} has no NumPy representation for XDNA staging"
        ) from None


def tinygrad_to_numpy(tensor: Any) -> np.ndarray:
    """Materialize a tinygrad ``Tensor`` as a row-major NumPy array.

    The tensor may be an unrealized lazy graph; ``.numpy()`` realizes it.
    """
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        raise TypeError("tensor must expose shape and dtype")
    tinygrad_dtype_to_numpy(dtype)
    if hasattr(tensor, "contiguous"):
        tensor = tensor.contiguous()
    return np.ascontiguousarray(tensor.numpy())


def numpy_to_tinygrad(array: Any, *, device: Optional[str] = None) -> Any:
    """Wrap a NumPy array as a tinygrad ``Tensor`` on an optional device."""
    from tinygrad import Tensor

    buffer = np.ascontiguousarray(array)
    if device is None:
        return Tensor(buffer)
    return Tensor(buffer, device=device)


def numpy_to_iron(array: Any, *, dtype: Any = None, device: str = "cpu") -> Any:
    """Stage a NumPy array as an IRON tensor on the given device.

    The default is ``"cpu"``, which works without XRT and keeps the
    conversion unit-testable everywhere; pass ``device="npu"`` to allocate
    device buffers ahead of a real dispatch.
    """
    import aie.iron as iron

    buffer = np.ascontiguousarray(array)
    np_dtype = np.dtype(dtype) if dtype is not None else np.dtype(buffer.dtype)
    return iron.tensor(
        buffer.astype(np_dtype, copy=False), dtype=np_dtype, device=device
    )


def tinygrad_to_iron(tensor: Any, *, dtype: Any = None, device: str = "cpu") -> Any:
    """Stage a tinygrad ``Tensor`` (or NumPy array) as an IRON tensor."""
    if isinstance(tensor, np.ndarray):
        return numpy_to_iron(tensor, dtype=dtype, device=device)
    return numpy_to_iron(tinygrad_to_numpy(tensor), dtype=dtype, device=device)


def plan_tinygrad_matmul(
    a: Any,
    b: Any,
    *,
    dtype: str,
    output_dtype: Optional[str] = None,
    columns: int = 8,
    profile: Optional[Mapping[str, Any]] = None,
) -> MatmulDispatchPlan:
    """Infer a MatMul dispatch plan from two tensor-likes' ``.shape``."""
    return plan_matmul(
        tuple(int(value) for value in a.shape),
        tuple(int(value) for value in b.shape),
        dtype,
        columns,
        profile,
        output_dtype,
    )


def run_tinygrad_matmul(
    manifest: Mapping[str, Any],
    a: Any,
    b: Any,
    *,
    dtype: str,
    output_dtype: Optional[str] = None,
    columns: int = 8,
    base_dir: Any = None,
    profile: Optional[Mapping[str, Any]] = None,
    device: str = "npu",
) -> Tuple[np.ndarray, Mapping[str, Any]]:
    """Run one 2-D MatMul on the NPU, returning the output and dispatch metadata.

    ``a`` and ``b`` are tinygrad tensors or NumPy arrays.  A missing kernel
    artifact raises :class:`XDNAUnavailable` before any device tensor is
    allocated, so fallback is never silently returned as NPU output.
    """
    plan = plan_tinygrad_matmul(
        a, b, dtype=dtype, output_dtype=output_dtype, columns=columns, profile=profile
    )
    key = matmul_kernel_key(dtype, *plan.shape, columns, profile, output_dtype)
    if resolve_kernel_artifact(manifest, key) is None:
        raise XDNAUnavailable(f"no XDNA artifact for {key!r}; cannot run on the NPU")
    m, k, n = plan.shape
    iron_a = tinygrad_to_iron(a, device=device)
    iron_b = tinygrad_to_iron(b, device=device)
    iron_c = numpy_to_iron(
        np.empty((m, n), dtype=xdna_dtype_to_numpy(plan.output_dtype)),
        device=device,
    )
    result = dispatch_matmul(
        manifest,
        iron_a,
        iron_b,
        iron_c,
        dtype=dtype,
        output_dtype=output_dtype,
        m=m,
        k=k,
        n=n,
        columns=columns,
        base_dir=base_dir,
        profile=profile,
    )
    if result.get("execution") != "real_npu":
        raise XDNAUnavailable(f"XDNA dispatch did not reach the NPU: {result!r}")
    # ``.numpy()`` is a view into the tensor's host buffer; copy so the
    # result outlives device teardown.
    return np.array(iron_c.numpy(), copy=True), result
