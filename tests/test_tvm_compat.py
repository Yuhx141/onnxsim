"""TVM compatibility wrapper tests without requiring the optional TVM wheel."""

import importlib.util
import sys
import types
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "tvm_compat", Path(__file__).parents[1] / "onnxsim" / "rpc" / "tvm_compat.py"
)
assert _SPEC and _SPEC.loader
tvm_compat = importlib.util.module_from_spec(_SPEC)
sys.modules["tvm_compat"] = tvm_compat
_SPEC.loader.exec_module(tvm_compat)


class _Session:
    def __init__(self):
        self.uploads = []

    def upload(self, path):
        self.uploads.append(path)

    def load_module(self, path):
        return ("module", path)

    def get_function(self, name):
        return lambda *args: (name, args)


def test_connect_falls_back_for_old_tvm_signature(monkeypatch):
    session = _Session()

    class OldRpc:
        @staticmethod
        def connect(host, port, key):
            assert (host, port, key) == ("runner", 9090, "pixel")
            return session

    monkeypatch.setitem(sys.modules, "tvm", types.SimpleNamespace(rpc=OldRpc))
    wrapped = tvm_compat.connect("runner", 9090, key="pixel")
    wrapped.upload("model.so", target="cpu")
    assert session.uploads == ["model.so"]
    assert wrapped.load_module("model.so") == ("module", "model.so")


def test_tracker_falls_back_for_old_request_signature(monkeypatch):
    session = _Session()

    class Tracker:
        @staticmethod
        def request(key):
            assert key == "hexagon"
            return session

    class Rpc:
        @staticmethod
        def connect_tracker(host, port):
            assert (host, port) == ("tracker", 9190)
            return Tracker()

    monkeypatch.setitem(sys.modules, "tvm", types.SimpleNamespace(rpc=Rpc))
    assert tvm_compat.connect_tracker("tracker", 9190, "hexagon").raw is session


def test_loaded_tvm_ffi_module_functions_are_forwarded_unchanged(monkeypatch):
    # Use an opaque sentinel: the wrapper must leave FFI object marshalling to
    # the installed TVM client rather than inspecting or converting the value.
    ffi_tensor = object()

    class Module:
        def get_function(self, name):
            assert name == "add_one"

            def function(value):
                assert value is ffi_tensor
                return ffi_tensor

            return function

    module = Module()

    class Session(_Session):
        def load_module(self, path):
            assert path == "kernel.so"
            return module

    session = Session()

    class Rpc:
        @staticmethod
        def connect(host, port, **kwargs):
            return session

    monkeypatch.setitem(sys.modules, "tvm", types.SimpleNamespace(rpc=Rpc))
    wrapped = tvm_compat.connect("runner", 9090)
    assert wrapped.load_module("kernel.so") is module
    assert (
        wrapped.load_module("kernel.so").get_function("add_one")(ffi_tensor)
        is ffi_tensor
    )


def test_session_forwards_time_evaluator_and_closes_on_context_exit(monkeypatch):
    evaluator = object()
    calls = []

    class Session(_Session):
        def time_evaluator(self, function_name, device, **kwargs):
            calls.append((function_name, device, kwargs))
            return evaluator

        def close(self):
            calls.append(("close",))

    session = Session()

    class Rpc:
        @staticmethod
        def connect(host, port, *, key, timeout):
            assert (host, port, key, timeout) == ("runner", 9090, "hexagon", 30)
            return session

    monkeypatch.setitem(sys.modules, "tvm", types.SimpleNamespace(rpc=Rpc))
    with tvm_compat.connect("runner", 9090, key="hexagon", timeout=30) as wrapped:
        device = object()
        assert wrapped.time_evaluator("run", device, number=5) is evaluator
        assert calls == [("run", device, {"number": 5})]
    assert calls[-1] == ("close",)
