"""Wire format shared by the onnxsim RPC server, client and tracker.

A message is one length-prefixed JSON header plus zero or more raw binary blobs::

    <u32 header_len> <u32 blob_count> <header JSON, utf-8>  (<u64 blob_len> <blob bytes>)*

Tensors travel as a ``"tensors"`` list of ``{"name", "dtype", "shape"}`` records in the header with
one little-endian, C-contiguous blob each. There is no pickle anywhere: a server never
deserializes anything but JSON and raw buffers. This is *not* TVM's wire protocol; it is a much
smaller one with the same session shape (handshake with a key, upload, load, run, time).
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

MAGIC = b"ONNXSIM-RPC/1\n"
MAX_HEADER_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_BLOB_BYTES = 4 * 1024**3

_HEADER = struct.Struct("<II")
_BLOB = struct.Struct("<Q")

# Tensor dtypes that cross the wire (the ones numpy/onnxruntime exchange as plain buffers).
DTYPES = (
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
)


class RPCError(RuntimeError):
    """An error reported by the remote side (or a protocol violation)."""


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed while reading a message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(
    sock: socket.socket, header: Dict[str, Any], blobs: Sequence[bytes] = ()
) -> None:
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    parts = [_HEADER.pack(len(payload), len(blobs)), payload]
    for blob in blobs:
        parts.append(_BLOB.pack(len(blob)))
        parts.append(blob)
    sock.sendall(b"".join(parts))


def recv_message(
    sock: socket.socket, max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES
) -> Tuple[Dict[str, Any], List[bytes]]:
    header_len, blob_count = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if header_len > MAX_HEADER_BYTES or blob_count > 1_000_000:
        raise RPCError("message header exceeds the protocol limits")
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))
    blobs = []
    for _ in range(blob_count):
        (length,) = _BLOB.unpack(_recv_exact(sock, _BLOB.size))
        if length > max_blob_bytes:
            raise RPCError(
                f"blob of {length} bytes exceeds the {max_blob_bytes} byte limit"
            )
        blobs.append(_recv_exact(sock, length))
    return header, blobs


def encode_tensors(
    tensors: Dict[str, np.ndarray],
) -> Tuple[List[Dict[str, Any]], List[bytes]]:
    specs = encode_tensor_specs(tensors)
    blobs = []
    for name, value in tensors.items():
        array = np.asarray(value)
        array = np.ascontiguousarray(
            array.astype(array.dtype.newbyteorder("<"), copy=False)
        )
        blobs.append(array.tobytes())
    return specs, blobs


def encode_tensor_specs(tensors: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Describe tensor inputs without serializing their values."""
    specs = []
    for name, value in tensors.items():
        array = np.asarray(value)
        if array.dtype.name not in DTYPES:
            raise RPCError(f"tensor {name!r} has unsupported dtype {array.dtype.name}")
        specs.append(
            {"name": name, "dtype": array.dtype.name, "shape": list(array.shape)}
        )
    return specs


def decode_tensors(
    specs: Sequence[Dict[str, Any]], blobs: Sequence[bytes]
) -> Dict[str, np.ndarray]:
    if len(specs) > len(blobs):
        raise RPCError("fewer tensor blobs than tensor descriptions")
    tensors = {}
    for spec, blob in zip(specs, blobs):
        dtype = spec["dtype"]
        if dtype not in DTYPES:
            raise RPCError(f"unsupported tensor dtype {dtype!r}")
        shape = tuple(int(d) for d in spec["shape"])
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if len(blob) != expected:
            raise RPCError(
                f"tensor {spec['name']!r}: {len(blob)} bytes, expected {expected}"
            )
        array = np.frombuffer(blob, dtype=np.dtype(dtype).newbyteorder("<")).reshape(
            shape
        )
        tensors[spec["name"]] = array.astype(np.dtype(dtype), copy=True)
    return tensors
