"""Device-free tests for the XDNA RPC compile artifact cache (compiler subprocess is mocked)."""

import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("onnx")

from onnxsim.rpc import xdna, xdna_cache  # noqa: E402


class Env:
    """Fake compiler environment: counts compiler runs, writes fake artifacts."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.calls = []
        self.delay = 0.0
        self.scripts = tmp_path / "scripts"
        (self.scripts / "kernels").mkdir(parents=True)
        (self.scripts / "compile_resnet_kernels.py").write_text("# compiler v1\n")
        (self.scripts / "kernels" / "k.cc").write_text("// kernel v1\n")
        self.toolchain = "iron-1.0"
        self.work = tmp_path / "work"
        self.work.mkdir()
        self.example = tmp_path / "whole_array.py"
        self.example.write_text("# example v1\n")
        monkeypatch.setattr(xdna, "_scripts_dir", lambda: self.scripts)
        monkeypatch.setattr(
            xdna, "_script", lambda name: self.scripts / name, raising=True
        )
        monkeypatch.setattr(xdna, "_xdna_python", lambda header: "/fake/iron/python")
        monkeypatch.setattr(
            xdna_cache,
            "toolchain_identity",
            lambda python: f"{python}|{self.toolchain}",
        )
        monkeypatch.setattr(xdna, "_run", self.fake_run)
        for name in list(__import__("os").environ):
            if name.startswith(("ONNXSIM_XDNA", "XDNA_")):
                monkeypatch.delenv(name)
        monkeypatch.setenv("ONNXSIM_XDNA_CACHE_DIR", str(tmp_path / "cache"))
        self.monkeypatch = monkeypatch
        self.cache_dir = tmp_path / "cache"

    def fake_run(self, command, env=None):
        self.calls.append(list(command))
        time.sleep(self.delay)

        def arg(flag):
            return Path(command[command.index(flag) + 1])

        if "--xclbin-path" in command:
            arg("--xclbin-path").write_bytes(b"XCLBIN" * 10)
            arg("--insts-path").write_bytes(b"INSTS" * 10)
        else:  # resnet: script model example manifest --artifact-dir DIR
            arg("--artifact-dir").joinpath("gemm.xclbin").write_bytes(b"G" * 8)
            Path(command[4]).write_text(
                json.dumps({"artifacts": str(arg("--artifact-dir"))})
            )

    def compile(self, kind="fused_bottleneck", options=None, model=b"model-bytes"):
        if options is None:
            options = {"block": "/layer1/layer1.0"}
        header = {"kind": kind, "options": options}
        reply, blobs = xdna.compile_resnet(header, [model], str(self.work))
        assert blobs == []
        return reply


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


def test_second_identical_compile_is_hit_without_compiler(env):
    first = env.compile()
    second = env.compile()
    assert first["cache"] == "miss" and second["cache"] == "hit"
    assert len(env.calls) == 1
    assert first["cache_key"] == second["cache_key"]
    assert Path(second["xclbin"]).read_bytes() == b"XCLBIN" * 10
    assert Path(second["insts"]).is_file()
    assert second["xclbin"].startswith(str(env.cache_dir))
    assert {k: v for k, v in second.items() if k != "cache"} == {
        k: v for k, v in first.items() if k != "cache"
    }


def test_resnet_kind_hit_relocates_manifest_paths(env):
    options = {"example": str(env.example)}
    first = env.compile("resnet", options)
    second = env.compile("resnet", options)
    assert (first["cache"], second["cache"]) == ("miss", "hit")
    assert len(env.calls) == 1
    assert second["manifest"]["artifacts"] == second["artifact_dir"]
    assert second["artifact_dir"].startswith(str(env.cache_dir / second["cache_key"]))
    assert (Path(second["artifact_dir"]) / "gemm.xclbin").is_file()


def test_different_options_miss_and_model_change_misses(env):
    a = env.compile(options={"block": "/layer1/layer1.0"})
    b = env.compile(options={"block": "/layer1/layer1.1"})
    c = env.compile(options={"block": "/layer1/layer1.0", "device": "npu1"})
    d = env.compile(model=b"other-model")
    assert [r["cache"] for r in (a, b, c, d)] == ["miss"] * 4
    assert len({r["cache_key"] for r in (a, b, c, d)}) == 4
    assert len(env.calls) == 4


def test_irrelevant_options_do_not_change_key(env):
    a = env.compile(options={"block": "/b", "cache_max_entries": 5, "unrelated": 1})
    b = env.compile(options={"block": "/b"})
    assert b["cache"] == "hit" and a["cache_key"] == b["cache_key"]


def test_key_changes_with_sources_toolchain_env_and_example(env):
    base = env.compile(options={"example": str(env.example)}, kind="resnet")
    assert (
        env.compile(options={"example": str(env.example)}, kind="resnet")["cache"]
        == "hit"
    )

    (env.scripts / "kernels" / "k.cc").write_text("// kernel v2\n")
    after_src = env.compile(options={"example": str(env.example)}, kind="resnet")
    assert after_src["cache"] == "miss" and after_src["cache_key"] != base["cache_key"]

    (env.scripts / "__pycache__").mkdir()
    (env.scripts / "__pycache__" / "x.py").write_text("junk")
    (env.scripts / "notes.txt").write_text("not a source")
    assert (
        env.compile(options={"example": str(env.example)}, kind="resnet")["cache"]
        == "hit"
    )

    env.toolchain = "iron-2.0"
    after_tc = env.compile(options={"example": str(env.example)}, kind="resnet")
    assert after_tc["cache"] == "miss"

    env.monkeypatch.setenv("XDNA_BLOCKED_MAX_CHUNK", "4096")
    after_env = env.compile(options={"example": str(env.example)}, kind="resnet")
    assert after_env["cache"] == "miss"
    env.monkeypatch.setenv(
        "ONNXSIM_XDNA_CACHE_MAX_ENTRIES", "50"
    )  # cache knobs: not in key
    assert (
        env.compile(options={"example": str(env.example)}, kind="resnet")["cache"]
        == "hit"
    )

    env.example.write_text("# example v2\n")  # same path, new content
    after_ex = env.compile(options={"example": str(env.example)}, kind="resnet")
    assert after_ex["cache"] == "miss"
    assert len(env.calls) == 5


def test_key_is_stable_across_calls(env):
    header = {"kind": "fused_bottleneck", "options": {"block": "/b"}}
    keys = set()
    for _ in range(3):
        root = Path("@ROOT@")
        command = xdna._compile_command(
            header, "fused_bottleneck", header["options"], root
        )[0]
        keys.add(
            xdna._cache_key(
                header, "fused_bottleneck", header["options"], b"m", command, root
            )
        )
    assert len(keys) == 1


@pytest.mark.parametrize("damage", ["truncate", "delete", "empty", "entry_json"])
def test_corrupt_entry_is_rebuilt(env, damage):
    first = env.compile()
    xclbin = Path(first["xclbin"])
    if damage == "truncate":
        xclbin.write_bytes(b"XC")
    elif damage == "delete":
        xclbin.unlink()
    elif damage == "empty":
        xclbin.write_bytes(b"")
    else:
        (xclbin.parents[1] / xdna_cache.ENTRY_FILE).write_text("{not json")
    second = env.compile()
    assert second["cache"] == "miss" and len(env.calls) == 2
    assert Path(second["xclbin"]).read_bytes() == b"XCLBIN" * 10
    assert env.compile()["cache"] == "hit" and len(env.calls) == 2


def test_failed_compile_leaves_no_entry(env):
    def boom(command, env=None):
        raise xdna.proto.RPCError("compiler exploded")

    env.monkeypatch.setattr(xdna, "_run", boom)
    with pytest.raises(xdna.proto.RPCError):
        env.compile()
    assert [p for p in env.cache_dir.iterdir() if p.name != "locks"] == []
    env.monkeypatch.setattr(xdna, "_run", env.fake_run)
    assert env.compile()["cache"] == "miss"


def test_no_cache_bypasses_and_uses_fresh_work_dir(env):
    env.compile()
    reply = env.compile(options={"block": "/layer1/layer1.0", "no_cache": True})
    assert reply["cache"] == "bypass" and "cache_key" not in reply
    assert reply["xclbin"].startswith(str(env.work / "xdna-rpc"))
    assert len(env.calls) == 2
    reply2 = env.compile(options={"block": "/layer1/layer1.0", "no_cache": True})
    assert reply2["artifact_dir"] != reply["artifact_dir"]
    assert len(env.calls) == 3


def test_env_disable(env):
    env.monkeypatch.setenv("ONNXSIM_XDNA_CACHE", "0")
    assert env.compile()["cache"] == "bypass"
    assert env.compile()["cache"] == "bypass"
    assert len(env.calls) == 2


def test_default_cache_dir_is_under_work_dir(env):
    env.monkeypatch.delenv("ONNXSIM_XDNA_CACHE_DIR")
    reply = env.compile()
    assert reply["xclbin"].startswith(str(env.work / "xdna-cache"))


def test_lru_eviction_keeps_recent_entries(env):
    env.monkeypatch.setenv("ONNXSIM_XDNA_CACHE_MAX_ENTRIES", "2")
    replies = []
    for i in range(3):
        replies.append(env.compile(options={"block": f"/b{i}"}))
        if i == 1:
            # touch entry 0 so entry 1 becomes the least recently used
            time.sleep(0.02)
            assert env.compile(options={"block": "/b0"})["cache"] == "hit"
        time.sleep(0.02)
    entries = {p.name for p in env.cache_dir.iterdir() if len(p.name) == 64}
    assert entries == {replies[0]["cache_key"], replies[2]["cache_key"]}
    assert env.compile(options={"block": "/b1"})["cache"] == "miss"


def test_concurrent_same_key_compiles_once(env):
    env.delay = 0.3
    replies = []

    def worker():
        replies.append(env.compile())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(env.calls) == 1
    assert sorted(r["cache"] for r in replies) == ["hit", "hit", "hit", "miss"]
    assert len({r["xclbin"] for r in replies}) == 1


def test_source_digest_ignores_pycache_and_non_sources(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    (root / "a.py").write_text("x")
    before = xdna_cache.source_digest([root])
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "a.py").write_text("y")
    (root / "README.md").write_text("z")
    assert xdna_cache.source_digest([root]) == before
    (root / "a.py").write_text("changed")
    assert xdna_cache.source_digest([root]) != before
