"""Small direct-XRT runtime for launching precompiled MLIR-AIE artifacts.

This module intentionally contains no IRON or MLIR imports. It follows the
same artifact ABI as IRON's XRT host runtime: opcode, instruction BO, byte
count, then the artifact's runtime-sequence BOs.
"""

from __future__ import annotations

import hashlib
import gc
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

import numpy as np


def _pyxrt() -> Any:
    try:
        import pyxrt
    except ImportError as exc:
        raise RuntimeError("direct XDNA runtime requires the XRT pyxrt Python binding") from exc
    return pyxrt


class XRTTensor:
    """A typed, mapped XRT BO with IRON-compatible overwrite/numpy methods."""

    def __init__(self, runtime: "XRTKernel", shape: tuple[int, ...], dtype: Any, group: int):
        self._device = runtime.device
        self._pyxrt = runtime.pyxrt
        self.group = int(group)
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        if self.nbytes <= 0:
            raise ValueError(f"XRT buffer must be non-empty, got shape {self.shape}")
        pyxrt = self._pyxrt
        # Device-owned BOs can be passed across hardware contexts, matching
        # the installed IRON XRTTensor implementation.
        self.bo = pyxrt.bo(runtime.device, self.nbytes, pyxrt.bo.host_only, self.group)
        self._host = np.frombuffer(self.bo.map(), dtype=self.dtype, count=self.nbytes // self.dtype.itemsize)
        self._host_dirty = False
        self._device_dirty = False
        self._constant_digest: bytes | None = None

    @contextmanager
    def overwrite(self) -> Iterator[np.ndarray]:
        if self._device_dirty:
            self.sync_from_device()
        try:
            yield self._host.reshape(self.shape)
        finally:
            self._host_dirty = True
            self._device_dirty = False
            self.sync_to_device()

    def sync_to_device(self) -> None:
        if self._host_dirty:
            self.bo.sync(self._pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._host_dirty = False

    def sync_from_device(self) -> None:
        if self._device_dirty:
            self.bo.sync(self._pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            self._device_dirty = False

    def numpy(self) -> np.ndarray:
        self.sync_from_device()
        return self._host.reshape(self.shape)

    def load_constant(self, value: np.ndarray) -> None:
        """Upload an immutable parameter BO once for the life of its context."""
        array = np.asarray(value, dtype=self.dtype).reshape(self.shape)
        digest = hashlib.blake2b(array.tobytes(order="C"), digest_size=16).digest()
        if digest == self._constant_digest:
            return
        with self.overwrite() as host:
            np.copyto(host, array)
        self._constant_digest = digest


class XRTKernel:
    """Persistent XRT context/kernel/instruction BO for one artifact pair."""

    def __init__(self, xclbin_path: str, insts_path: str, device_index: int = 0, kernel_name: str = "MLIR_AIE"):
        self.pyxrt = _pyxrt()
        self.device = _device(int(device_index))
        self.xclbin = self.pyxrt.xclbin(str(Path(xclbin_path)))
        uuid = self.device.register_xclbin(self.xclbin)
        self.context = self.pyxrt.hw_context(self.device, uuid)
        self.kernel = self.pyxrt.kernel(self.context, kernel_name)
        self.kernel_name = kernel_name
        self._tensor_cache: dict[tuple[tuple[int, ...], str, int], XRTTensor] = {}
        insts = Path(insts_path).read_bytes()
        if not insts:
            raise ValueError(f"instruction stream is empty: {insts_path}")
        self.instruction_bytes = len(insts)
        group = int(self.kernel.group_id(1))
        self.instruction_bo = self.pyxrt.bo(
            self.device, self.instruction_bytes, self.pyxrt.bo.cacheable, group
        )
        self.instruction_bo.write(insts, 0)
        self.instruction_bo.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def tensor(self, shape: tuple[int, ...], dtype: Any, argument_index: int = 3) -> XRTTensor:
        shape = tuple(int(dim) for dim in shape)
        dtype = np.dtype(dtype)
        argument_index = int(argument_index)
        key = (shape, dtype.str, argument_index)
        tensor = self._tensor_cache.get(key)
        if tensor is None:
            tensor = XRTTensor(self, shape, dtype, self.kernel.group_id(argument_index))
            self._tensor_cache[key] = tensor
        return tensor

    def __call__(
        self, *tensors: XRTTensor, timeout_ms: int = 120_000,
        output_indices: tuple[int, ...] | None = None,
    ) -> None:
        if not tensors:
            raise ValueError("XDNA kernel launch needs at least one runtime BO")
        for index, tensor in enumerate(tensors):
            if tensor._device is not self.device:
                raise ValueError("all runtime BOs must belong to this XRT device")
            expected_group = int(self.kernel.group_id(3 + index))
            if tensor.group != expected_group:
                raise ValueError(
                    f"XDNA argument {3 + index} needs BO group {expected_group}, got {tensor.group}"
                )
            tensor.sync_to_device()
        run = self.kernel(3, self.instruction_bo, self.instruction_bytes, *(tensor.bo for tensor in tensors))
        state = run.wait(int(timeout_ms))
        if state != self.pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"XDNA kernel {self.kernel_name} finished in state {state}")
        output_indices = output_indices if output_indices is not None else (len(tensors) - 1,)
        for index in output_indices:
            if index < 0 or index >= len(tensors):
                raise ValueError(f"output tensor index {index} is outside {len(tensors)} launch arguments")
            tensor = tensors[index]
            tensor._device_dirty = True


@lru_cache(maxsize=1)
def _device(device_index: int) -> Any:
    return _pyxrt().device(device_index)


@lru_cache(maxsize=16)
def _load_kernel_cached(
    xclbin_path: str,
    insts_path: str,
    device_index: int,
    xclbin_signature: tuple[int, int],
    insts_signature: tuple[int, int],
) -> XRTKernel:
    return XRTKernel(xclbin_path, insts_path, device_index=device_index)


_CACHE_LOCK = RLock()
_CACHE_KEYS: set[tuple[Any, ...]] = set()


def _context_cache_limit() -> int:
    return max(1, min(16, int(os.environ.get("XRT_CONTEXT_CACHE_SIZE", "16"))))


def load_kernel(xclbin_path: str, insts_path: str, device_index: int = 0) -> XRTKernel:
    """Reuse contexts and instruction BOs while detecting replaced artifacts."""
    xclbin_path = str(Path(xclbin_path).resolve())
    insts_path = str(Path(insts_path).resolve())
    xclbin_stat = Path(xclbin_path).stat()
    insts_stat = Path(insts_path).stat()
    xclbin_signature = (int(xclbin_stat.st_mtime_ns), int(xclbin_stat.st_size))
    insts_signature = (int(insts_stat.st_mtime_ns), int(insts_stat.st_size))
    key = (xclbin_path, insts_path, int(device_index), xclbin_signature, insts_signature)
    with _CACHE_LOCK:
        # Strix Halo allows 16 contexts per process/device. Clear the old
        # request's working set before opening an unseen 17th context; tensor
        # wrappers deliberately do not retain their owning kernel/context.
        if key not in _CACHE_KEYS and _load_kernel_cached.cache_info().currsize >= _context_cache_limit():
            _load_kernel_cached.cache_clear()
            _CACHE_KEYS.clear()
            gc.collect()
        kernel = _load_kernel_cached(*key)
        _CACHE_KEYS.add(key)
        return kernel


load_kernel.cache_info = _load_kernel_cached.cache_info  # type: ignore[attr-defined]
