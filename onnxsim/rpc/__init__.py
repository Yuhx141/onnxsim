"""TVM-style remote execution for onnxsim: ``connect``, ``upload``, ``load_model``, ``time_evaluator``.

See ``docs/rpc.md``. Heavy pieces (server, tracker, executor) import lazily.
"""

from __future__ import annotations

from typing import Any

from ._protocol import RPCError
from .client import (
    ProfileResult,
    RemoteModel,
    Session,
    TrackerClient,
    connect,
    connect_tracker,
    remote_executor,
)

__all__ = [
    "ProfileResult",
    "RPCError",
    "RPCServer",
    "RemoteModel",
    "RemoteModelExecutor",
    "Session",
    "Tracker",
    "TrackerClient",
    "connect",
    "connect_tracker",
    "remote_executor",
    "tvm_compat",
]


def __getattr__(name: str) -> Any:
    if name == "RPCServer":
        from .server import RPCServer

        return RPCServer
    if name == "Tracker":
        from .tracker import Tracker

        return Tracker
    if name == "RemoteModelExecutor":
        from .executor import RemoteModelExecutor

        return RemoteModelExecutor
    raise AttributeError(name)
