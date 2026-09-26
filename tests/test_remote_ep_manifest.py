"""Dependency-free contract checks for the optional ORT EP package metadata."""

import json
from pathlib import Path


MANIFEST = Path(__file__).parents[1] / "tools" / "onnx-remote" / "ort_plugin" / "onnxsim_remote_ep.manifest.json"


def test_remote_ep_manifest_is_explicit_about_legacy_abi():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["status"] == "legacy_source_abi_adapter"
    assert manifest["registration"] == "internal_cpp"
    assert manifest["public_plugin_entrypoints"] == []
    assert manifest["ort_version"] == "requires_legacy_internal_ep_headers"
    assert manifest["public_plugin_migration"]["api"] == "onnxruntime_ep_c_api.h"
    assert manifest["transport"]["protocol"] == "onnx-remote-v5"
    assert set(manifest["profiling"]["levels"]) == {"off", "summary", "detailed"}
