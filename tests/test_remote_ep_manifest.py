"""Dependency-free contract checks for the optional ORT EP package metadata."""

import json
from pathlib import Path

MANIFEST = (
    Path(__file__).parents[1]
    / "tools"
    / "onnx-remote"
    / "ort_plugin"
    / "onnxsim_remote_ep.manifest.json"
)


def test_remote_ep_manifest_is_explicit_about_legacy_abi():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["status"] == "public_plugin_and_legacy_source_adapter"
    assert manifest["registration"] == "internal_cpp"
    assert manifest["public_plugin_entrypoints"] == [
        "CreateEpFactories",
        "ReleaseEpFactory",
    ]
    assert manifest["ort_version"] == "public_plugin_requires_ort_1.29_plus"
    assert manifest["public_plugin"]["api"] == "onnxruntime_ep_c_api.h"
    assert manifest["transport"]["protocol"] == "onnx-remote-v5"
    assert set(manifest["profiling"]["levels"]) == {"off", "summary", "detailed"}
