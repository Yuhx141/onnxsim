"""Content-addressed artifact cache for XDNA RPC compiles.

Kept free of heavy imports (no onnx, no numpy) so it can be used and tested alone.

An entry lives in ``<cache_root>/<sha256-key>/`` and holds the compiler outputs plus an
``entry.json`` manifest (the compile result with the entry path replaced by a placeholder, and
the size of every file). Entries are published by building in ``<cache_root>/.tmp-*`` and
renaming, under a per-key ``flock``, so concurrent requests for one key compile once and a crash
never leaves a half-written entry under a real key.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

try:  # POSIX only; on Windows the per-key lock is in-process (the XDNA toolchain is Linux-only)
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()

# Bump when the entry layout or key composition changes.
KEY_VERSION = 1
ENTRY_FILE = "entry.json"
PLACEHOLDER = "@ENTRY@"
DEFAULT_MAX_ENTRIES = 20

ENV_CACHE_DIR = "ONNXSIM_XDNA_CACHE_DIR"
ENV_CACHE_ENABLE = "ONNXSIM_XDNA_CACHE"
ENV_MAX_ENTRIES = "ONNXSIM_XDNA_CACHE_MAX_ENTRIES"

# Environment variables that can change what the compiler produces (or which compiler runs).
_ENV_PREFIXES = ("XDNA_", "AIE_", "IRON_", "PEANO", "MLIR_AIE")
_ENV_NAMES = ("XILINX_XRT", "PYTHONPATH", "PATH", "LD_LIBRARY_PATH")
_SOURCE_SUFFIXES = (".py", ".cc", ".h")


def cache_enabled(
    options: Mapping[str, Any], env: Optional[Mapping[str, str]] = None
) -> bool:
    env = os.environ if env is None else env
    if options.get("no_cache"):
        return False
    return env.get(ENV_CACHE_ENABLE, "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def cache_root(
    options: Mapping[str, Any], work_dir: str, env: Optional[Mapping[str, str]] = None
) -> Path:
    env = os.environ if env is None else env
    chosen = options.get("cache_dir") or env.get(ENV_CACHE_DIR)
    return Path(str(chosen)).expanduser() if chosen else Path(work_dir) / "xdna-cache"


def max_entries(
    options: Mapping[str, Any], env: Optional[Mapping[str, str]] = None
) -> int:
    env = os.environ if env is None else env
    raw = options.get("cache_max_entries", env.get(ENV_MAX_ENTRIES))
    try:
        return max(int(raw), 1) if raw is not None else DEFAULT_MAX_ENTRIES
    except (TypeError, ValueError):
        return DEFAULT_MAX_ENTRIES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(directories: Iterable[Path]) -> str:
    """Hash every ``.py``/``.cc``/``.h`` file (path and bytes) below the directories."""
    digest = hashlib.sha256()
    for directory in sorted(Path(d) for d in directories):
        files = sorted(
            p
            for p in directory.rglob("*")
            if p.is_file()
            and p.suffix in _SOURCE_SUFFIXES
            and "__pycache__" not in p.parts
        )
        for path in files:
            digest.update(str(path.relative_to(directory)).encode() + b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def compile_env(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment variables that influence compilation."""
    env = os.environ if env is None else env
    return {
        name: value
        for name, value in sorted(env.items())
        if (name.startswith(_ENV_PREFIXES) or name in _ENV_NAMES)
        and not name.startswith("ONNXSIM_")
    }


_PROBE = r"""
import json, os, sys
info = {"version": sys.version}
try:
    from importlib import metadata
    for name in ("mlir_aie", "mlir-aie", "aie", "llvm-aie", "llvm_aie", "pyxrt"):
        try:
            info[name] = metadata.version(name)
        except Exception:
            pass
except Exception:
    pass
try:
    import aie
    p = os.path.dirname(aie.__file__)
    st = os.stat(p)
    info["aie_path"] = p
    info["aie_mtime_ns"] = st.st_mtime_ns
except Exception:
    pass
peano = os.environ.get("PEANO_INSTALL_DIR")
if peano:
    clang = os.path.join(peano, "bin", "clang")
    try:
        st = os.stat(clang)
        info["peano"] = [peano, st.st_size, st.st_mtime_ns]
    except OSError:
        info["peano"] = [peano]
xrt = os.environ.get("XILINX_XRT")
if xrt:
    try:
        with open(os.path.join(xrt, "version.info")) as handle:
            info["xrt"] = handle.read()
    except OSError:
        info["xrt"] = xrt
print(json.dumps(info, sort_keys=True))
"""

_toolchain_memo: Dict[str, str] = {}
_toolchain_lock = threading.Lock()


def toolchain_identity(python: str) -> str:
    """Identity of the IRON python (path + mlir_aie/peano/XRT versions), memoized per process."""
    with _toolchain_lock:
        if python in _toolchain_memo:
            return _toolchain_memo[python]
    try:
        out = subprocess.run(
            [python, "-c", _PROBE],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        detail = (
            out.stdout.strip()
            if out.returncode == 0
            else f"probe-failed:{out.returncode}"
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = f"probe-error:{type(error).__name__}"
    identity = f"{python}|{detail}"
    with _toolchain_lock:
        _toolchain_memo[python] = identity
    return identity


def make_key(
    kind: str,
    model: bytes,
    command: Any,
    sources: str,
    toolchain: str,
    env: Mapping[str, str],
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """SHA-256 over every input that can change the compile output."""
    digest = hashlib.sha256()
    digest.update(f"onnxsim-xdna-cache-v{KEY_VERSION}\0{kind}\0".encode())
    digest.update(hashlib.sha256(model).digest())
    meta = {
        "command": command,
        "sources": sources,
        "toolchain": toolchain,
        "env": dict(env),
        "extra": dict(extra or {}),
    }
    digest.update(json.dumps(meta, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _relocate(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_relocate(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _relocate(v, old, new) for k, v in value.items()}
    return value


def _file_sizes(entry: Path) -> Dict[str, int]:
    return {
        str(p.relative_to(entry)): p.stat().st_size
        for p in sorted(entry.rglob("*"))
        if p.is_file() and p.name != ENTRY_FILE
    }


def load_entry(entry: Path) -> Optional[Dict[str, Any]]:
    """Return the cached result for ``entry`` if intact, else None (corrupt/partial)."""
    try:
        meta = json.loads((entry / ENTRY_FILE).read_text(encoding="utf-8"))
        files = meta["files"]
        required = meta["required"]
        if not files or not isinstance(files, dict):
            return None
        for name, size in files.items():
            path = entry / name
            if not path.is_file() or size <= 0 or path.stat().st_size != size:
                return None
        if any(name not in files for name in required):
            return None
        return _relocate(meta["result"], PLACEHOLDER, str(entry))
    except (OSError, ValueError, KeyError, TypeError):
        return None


@contextmanager
def _key_lock(root: Path, key: str):
    """Serialize compiles of one key: an in-process lock plus, where available, a cross-process flock."""
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(f"{root}:{key}", threading.Lock())
    with thread_lock:
        if fcntl is None:
            yield
            return
        lock_dir = root / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_dir / f"{key}.lock", "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _remove(path: Path) -> None:
    trash = path.with_name(f".trash-{uuid.uuid4().hex}")
    try:
        os.rename(path, trash)
    except OSError:
        trash = path
    shutil.rmtree(trash, ignore_errors=True)


def evict(root: Path, limit: int, keep: Iterable[str] = ()) -> list[str]:
    """Drop least-recently-used entries (by ``entry.json`` mtime) beyond ``limit``."""
    keep = set(keep)
    entries = []
    for path in root.iterdir() if root.is_dir() else ():
        if len(path.name) == 64 and path.is_dir():
            try:
                entries.append((os.stat(path / ENTRY_FILE).st_mtime_ns, path))
            except OSError:
                entries.append((0, path))  # partial entry: evict first
    entries.sort(key=lambda item: item[0])
    removed = []
    for _, path in entries[: max(len(entries) - limit, 0)]:
        if path.name in keep:
            continue
        _remove(path)
        removed.append(path.name)
    return removed


def get_or_build(
    root: Path,
    key: str,
    build: Callable[[Path], Dict[str, Any]],
    required: Callable[[Path], Iterable[str]],
    limit: int = DEFAULT_MAX_ENTRIES,
) -> tuple[Dict[str, Any], str]:
    """Return ``(result, "hit"|"miss")``, running ``build(tmp_dir)`` at most once per key.

    ``build`` compiles into the given directory and returns the result dict (paths inside it
    are relocated on publish). ``required(dir)`` lists files (relative to the entry) that must
    exist and be non-empty.
    """
    root.mkdir(parents=True, exist_ok=True)
    entry = root / key

    def hit() -> Optional[Dict[str, Any]]:
        result = load_entry(entry)
        if result is not None:
            try:
                os.utime(entry / ENTRY_FILE)
            except OSError:
                pass
        return result

    found = hit()
    if found is not None:
        return found, "hit"
    with _key_lock(root, key):
        found = hit()
        if found is not None:
            return found, "hit"
        if entry.exists():
            _remove(entry)  # corrupt or partial
        tmp = root / f".tmp-{uuid.uuid4().hex}"
        tmp.mkdir()
        try:
            result = build(tmp)
            need = list(required(tmp))
            files = _file_sizes(tmp)
            for name in need:
                if files.get(name, 0) <= 0:
                    raise RuntimeError(f"XDNA cache: compiler did not produce {name}")
            meta = {
                "key": key,
                "version": KEY_VERSION,
                "required": need,
                "files": files,
                "result": _relocate(result, str(tmp), PLACEHOLDER),
            }
            (tmp / ENTRY_FILE).write_text(
                json.dumps(meta, sort_keys=True), encoding="utf-8"
            )
            os.rename(tmp, entry)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    evict(root, limit, keep=(key,))
    return _relocate(result, str(tmp), str(entry)), "miss"
