"""Optional wrapper around Apache TVM's native RPC client.

The wire protocol is intentionally delegated to the installed TVM package;
TVM's PackedFunc and handshake details vary between releases.
"""

from __future__ import annotations

from typing import Any, Optional


def _tvm_rpc():
    try:
        from tvm import rpc  # type: ignore
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("TVM RPC compatibility requires apache-tvm") from error
    return rpc


class TVMCompatSession:
    """Small stable wrapper over a native ``tvm.rpc.RPCSession``."""

    def __init__(self, session: Any):
        self._session = session

    @property
    def raw(self) -> Any:
        return self._session

    def upload(self, path: str, target: Optional[str] = None) -> None:
        try:
            self._session.upload(path, **({} if target is None else {"target": target}))
        except TypeError:  # TVM versions without target=
            self._session.upload(path)

    def load_module(self, path: str) -> Any:
        return self._session.load_module(path)

    def get_function(self, name: str) -> Any:
        return self._session.get_function(name)

    def time_evaluator(self, function_name: str, device: Any, **kwargs: Any) -> Any:
        return self._session.time_evaluator(function_name, device, **kwargs)

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "TVMCompatSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def connect(host: str, port: int, key: str = "", timeout: int = 10) -> TVMCompatSession:
    """Connect through TVM's native RPC handshake."""
    rpc = _tvm_rpc()
    try:
        session = rpc.connect(host, port, key=key, timeout=timeout)
    except TypeError:  # older TVM clients
        session = rpc.connect(host, port, key=key)
    return TVMCompatSession(session)


def connect_tracker(
    host: str, port: int, key: str, priority: int = 1, timeout: int = 100
) -> TVMCompatSession:
    """Request a device from a TVM tracker across API generations."""
    tracker = _tvm_rpc().connect_tracker(host, port)
    try:
        session = tracker.request(key, priority=priority, session_timeout=timeout)
    except TypeError:
        try:
            session = tracker.request(key, priority=priority, timeout=timeout)
        except TypeError:
            session = tracker.request(key)
    return TVMCompatSession(session)


__all__ = ["TVMCompatSession", "connect", "connect_tracker"]
