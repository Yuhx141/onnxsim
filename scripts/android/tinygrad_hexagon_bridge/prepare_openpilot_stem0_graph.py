#!/usr/bin/env python3
"""Build a v65 graph with only the packed, qfloat stem.0 kernel retargeted to v68."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("base_bundle", type=Path, help="validated v65 export bundle")
  p.add_argument("optimized_bundle", type=Path, help="v68 export bundle made with --optimize-openpilot-stem0")
  p.add_argument("bindings", type=Path, help="verified dynamic buffer binding JSON")
  p.add_argument("baseline_graph", type=Path, help="v65 graph directory containing validated kernel objects")
  p.add_argument("--out-bundle", type=Path, required=True)
  p.add_argument("--out-graph", type=Path, required=True)
  p.add_argument("--tinygrad-bridge", type=Path, default=Path(__file__).parent)
  p.add_argument("--hexagon-clang", type=Path, required=True)
  p.add_argument("--v68-call", type=int, action="append", default=[0],
                 help="also select a captured v68 kernel call; repeat to select multiple calls (default: 0)")
  a = p.parse_args()

  base_manifest = json.loads((a.base_bundle / "manifest.json").read_text())
  opt_manifest = json.loads((a.optimized_bundle / "manifest.json").read_text())
  if len(base_manifest["calls"]) != len(opt_manifest["calls"]): raise ValueError("graph call count differs")
  stem_call = next((c for c in opt_manifest["calls"] if c["name"] == "r_2_64_128_32_24_3_3"), None)
  base_stem_call = next((c for c in base_manifest["calls"] if c["name"] == "r_2_64_128_32_24_3_3"), None)
  if stem_call is None or base_stem_call is None or stem_call["call"] != base_stem_call["call"]:
    raise ValueError("stem.0 call is missing or has moved")
  selected_calls = set(a.v68_call)
  if base_stem_call["call"] not in selected_calls or any(i < 0 or i >= len(base_manifest["calls"]) for i in selected_calls):
    raise ValueError("selected calls must include stem.0 call 0 and stay within the graph")

  shutil.copytree(a.base_bundle, a.out_bundle, dirs_exist_ok=True)
  shutil.copytree(a.baseline_graph, a.out_graph, dirs_exist_ok=True)
  manifest_path = a.out_bundle / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  # The exporter can map multiple graph calls to the same C source (for this
  # model, calls 6 and 10 invoke the same convolution with different buffers).
  # A source is one compiled object, so selecting any such call must update all
  # manifest entries that refer to that source. Leaving a mixture of old and
  # new source hashes makes the dispatcher link the same symbol twice.
  selected_sources = {base_manifest["calls"][i]["kernel_c"] for i in selected_calls}
  for call_index, base_call in enumerate(base_manifest["calls"]):
    if base_call["kernel_c"] not in selected_sources:
      continue
    opt_call = opt_manifest["calls"][call_index]
    if opt_call["kernel_c"] != base_call["kernel_c"]:
      raise ValueError(f"kernel source mapping differs for call {call_index}")
    opt_source = (a.optimized_bundle / opt_call["kernel_c"]).read_bytes()
    source_path = a.out_bundle / base_call["kernel_c"]
    source_path.write_bytes(opt_source)
    manifest_call = manifest["calls"][call_index]
    manifest_call["source_sha256"] = hashlib.sha256(opt_source).hexdigest()
    manifest_call["exported_name"] = opt_call["exported_name"]
  manifest["hvx_arch"] = "v65+v68-selected"
  manifest["weights"] = opt_manifest["weights"]
  manifest["weights_nbytes"] = opt_manifest["weights_nbytes"]
  manifest["packed_optimizations"] = opt_manifest.get("packed_optimizations", [])
  if not manifest["packed_optimizations"]: raise ValueError("optimized export lacks packed stem metadata")
  shutil.copyfile(a.optimized_bundle / "weights.bin", a.out_bundle / "weights.bin")
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

  driver_script = a.tinygrad_bridge / "generate_openpilot_graph_driver.py"
  subprocess.run([sys.executable, str(driver_script), str(a.out_bundle), str(a.bindings), "--out", str(a.out_graph)], check=True)

  old_layout = json.loads((a.baseline_graph / "input_layout.json").read_text())
  layout_path = a.out_graph / "input_layout.json"
  layout = json.loads(layout_path.read_text())
  layout["batch_calls"] = old_layout.get("batch_calls", 8)
  old_objects = {item["source"]: item["object"] for item in old_layout["objects"]}
  new_objects = {item["source"]: item["object"] for item in layout["objects"]}
  base_call_by_index = {c["call"]: c for c in base_manifest["calls"]}
  merged_call_by_source = {}
  for c in manifest["calls"]: merged_call_by_source.setdefault(c["kernel_c"], c)

  for source, new_obj in new_objects.items():
    call = merged_call_by_source[source]
    out_obj = a.out_graph / new_obj
    if source in selected_sources:
      subprocess.run([
        str(a.hexagon_clang), "--target=hexagon", "-c", "-O2", "-Wall", "-Werror", "-fno-stack-protector",
        "-x", "c", "-fPIC", "-ffreestanding", "-nostdlib", "-mcpu=hexagonv68", "-mhvx=v68",
        "-mhvx-length=128b", "-o", str(out_obj), str(a.out_bundle / source),
      ], check=True)
    else:
      old_call = base_call_by_index[call["call"]]
      old_obj = old_objects[old_call["kernel_c"]]
      if not (a.out_graph / old_obj).is_file(): raise FileNotFoundError(a.out_graph / old_obj)
      if (a.out_graph / old_obj).resolve() != out_obj.resolve(): shutil.copyfile(a.out_graph / old_obj, out_obj)
  layout_path.write_text(json.dumps(layout, indent=2) + "\n")
  print(f"prepared {len(new_objects)} objects: v68 calls {sorted(selected_calls)} plus validated v65 kernels")


if __name__ == "__main__": main()
