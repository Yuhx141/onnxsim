"""TVM compatibility wrapper tests without requiring the optional TVM wheel."""

import sys
import types

import pytest

pytest.importorskip("onnx")

from onnxsim.rpc import tvm_compat


class _Session:
    def __init__(self):
        self.uploads = []

    def upload(self, path):
        self.uploads.append(path)

    def load_module(self, path):
        return ("module", path)


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
