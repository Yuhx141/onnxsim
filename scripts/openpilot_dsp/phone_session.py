"""An ORT-style session that runs an ONNX model through the phone path: the onnx-remote compiler service (COMPILE), then the
Hexagon runner on the phone (load_compiled once, run_compiled per call), over the onnx-remote v5 wire protocol
(tools/onnx-remote/remote_transport.cpp). It has the `get_inputs()` / `get_outputs()` / `run(None, feeds)` surface
`run_models.run_driving` / `run_dm` and `evaluate.py` use, so the phone's own numerics (tinygrad's v65 integer path, the integer
heads' run-time activation quantization) go through the same route-frame scoring as ONNX Runtime.

  python evaluate.py driving name=model.onnx --backend phone --compiler 127.0.0.1:39504 --runner 127.0.0.1:39520

The compiler service and the phone's runner must be up (scripts/android/tinygrad_hexagon_bridge/openpilot_v65/run.sh compiler ...,
start_worker.sh). A recurrent-state input fed with the previous call's `next_` output stays on the phone: its input is sent empty
and its output comes back empty (the runner's `state` records), so a call moves only the camera frame.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

import numpy as np
import onnx

_MAGIC, _VERSION, _KIND_RUN, _KIND_OK = 0x4F525452, 5, 1, 2
_ONNX_NP = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
}


@dataclass(frozen=True)
class ValueInfo:
    name: str
    type: str
    shape: tuple[int, ...]
    elem_type: int = 1


def _u32(v):
    return struct.pack(">I", v)


def _u64(v):
    return struct.pack(">Q", v)


def _string(s: str | bytes):
    s = s.encode() if isinstance(s, str) else s
    return _u32(len(s)) + s


def _tensor(dtype: int, shape, data: bytes, elements: int) -> bytes:
    return (
        _u32(dtype)
        + _u32(len(shape))
        + b"".join(_u64(d) for d in shape)
        + _u64(elements)
        + data
    )


def _request(
    op: str,
    artifact_id: str = "",
    model: bytes = b"",
    artifact: bytes = b"",
    tensors=(),
    profiling: int = 0,
) -> bytes:
    body = (
        _u64(0)
        + _string(op)
        + _string(artifact_id)
        + _u64(len(model))
        + model
        + _u64(len(artifact))
        + artifact
        + _u32(profiling)
        + _u32(len(tensors))
        + b"".join(tensors)
    )
    return (
        _u32(_MAGIC) + struct.pack(">HH", _VERSION, _KIND_RUN) + _u64(len(body)) + body
    )


def _exchange(host: str, port: int, payload: bytes) -> tuple[bytes, int]:
    with socket.create_connection((host, port), timeout=3600) as s:
        s.sendall(payload)
        head = _recv(s, 16)
        magic, version, kind, n = struct.unpack(">IHHQ", head)
        if magic != _MAGIC or version != _VERSION:
            raise RuntimeError("invalid response header")
        return _recv(s, n), kind


def _recv(s: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(min(n - len(buf), 1 << 22))
        if not chunk:
            raise RuntimeError("connection closed")
        buf += chunk
    return bytes(buf)


class _Reader:
    def __init__(self, b: bytes):
        self.b, self.at = b, 0

    def take(self, n: int) -> bytes:
        out = self.b[self.at : self.at + n]
        self.at += n
        return out

    def u32(self):
        return struct.unpack(">I", self.take(4))[0]

    def u64(self):
        return struct.unpack(">Q", self.take(8))[0]

    def string(self):
        return self.take(self.u32())


def _response(body: bytes, kind: int):
    r = _Reader(body)
    r.u64()  # request id
    if kind != _KIND_OK:
        raise RuntimeError(f"remote error: {r.string().decode(errors='replace')}")
    outputs = []
    for _ in range(r.u32()):
        dtype, rank = r.u32(), r.u32()
        shape = tuple(r.u64() for _ in range(rank))
        n = r.u64()
        nbytes = n * np.dtype(_ONNX_NP[dtype]).itemsize
        outputs.append(
            np.frombuffer(r.take(nbytes), _ONNX_NP[dtype]).reshape(shape).copy()
        )
    for _ in range(r.u32()):  # profile events
        r.string(), r.string(), r.u64(), r.u64(), r.string()
    artifact_id, manifest = r.string().decode(), r.string().decode()
    artifact = r.take(r.u64())
    return outputs, artifact_id, manifest, artifact


class PhoneSession:
    def __init__(
        self,
        path: str,
        compiler: tuple[str, int] = ("127.0.0.1", 39502),
        runner: tuple[str, int] = ("127.0.0.1", 39520),
    ):
        self.compiler, self.runner = compiler, runner
        m = onnx.load(path, load_external_data=False)

        def info(v):
            t = v.type.tensor_type
            return ValueInfo(
                v.name,
                onnx.TensorProto.DataType.Name(t.elem_type).lower(),
                tuple(d.dim_value for d in t.shape.dim),
                t.elem_type,
            )

        self._inputs = tuple(info(v) for v in m.graph.input)
        self._outputs = tuple(info(v) for v in m.graph.output)
        # recurrent state: an input X with an output next_X of the same shape and dtype (the runner keeps it resident)
        self._state = {
            v.name
            for v in self._inputs
            if any(
                o.name == "next_" + v.name
                and o.shape == v.shape
                and o.elem_type == v.elem_type
                for o in self._outputs
            )
        }
        with open(path, "rb") as f:
            model = f.read()
        _, self.artifact_id, self.manifest, artifact = _response(
            *_exchange(*compiler, _request("compile", model=model))
        )
        _response(
            *_exchange(
                *runner,
                _request(
                    "load_compiled", artifact_id=self.artifact_id, artifact=artifact
                ),
            )
        )
        self._resident: dict[
            str, np.ndarray
        ] = {}  # name -> the array returned for next_<name>

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _output_names, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        tensors = []
        for v in self._inputs:
            a = np.asarray(feeds[v.name])
            if v.name in self._state and self._resident.get(v.name) is feeds[v.name]:
                # exactly the placeholder the last call returned: the state is still on the phone
                tensors.append(_tensor(v.elem_type, (0,), b"", 0))
                continue
            a = np.ascontiguousarray(a.astype(_ONNX_NP[v.elem_type], copy=False))
            tensors.append(_tensor(v.elem_type, a.shape, a.tobytes(), a.size))
        outputs, _, _, _ = _response(
            *_exchange(
                *self.runner,
                _request("run_compiled", artifact_id=self.artifact_id, tensors=tensors),
            )
        )
        result = []
        for v, o in zip(self._outputs, outputs):
            name = v.name[5:] if v.name.startswith("next_") else None
            if name in self._state:
                if (
                    o.size == 0
                ):  # resident: hand back the placeholder the next call will recognize
                    o = self._resident[name]
                else:
                    self._resident[name] = (
                        o  # the runner holds this call's output as the state; feeding `o` back keeps it there
                    )
            result.append(o)
        return result
