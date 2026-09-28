#!/usr/bin/env python3
"""Tile openpilot's shared calls 6/10 convolution to reuse activation loads."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("bundle", type=Path, help="captured openpilot kernel bundle to update in place")
  p.add_argument("--tile", type=int, choices=(2, 4, 8, 16, 32), default=32,
                 help="number of adjacent output-channel groups to reuse each activation tile")
  args = p.parse_args()

  manifest_path = args.bundle / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  if len(manifest.get("calls", [])) <= 10 or any(
    manifest["calls"][i]["name"] != "r_64_64_32_48_4" for i in (6, 10)
  ):
    raise ValueError("calls 6/10 are missing or have changed")
  source_name = manifest["calls"][6]["kernel_c"]
  if manifest["calls"][10]["kernel_c"] != source_name:
    raise ValueError("calls 6 and 10 no longer share one kernel source")
  source_path = args.bundle / source_name
  original = source_path.read_text()
  match = re.search(r"__attribute__\(\(noinline\)\) void\s+tg_call_6_r_64_64_32_48_4\s*\(.*?\)\s*\{",
                    original, re.S)
  if match is None:
    raise ValueError("shared call 6/10 generated signature is missing or has changed")
  signature = match.group(0)[:-1].rstrip()
  expected_args = ("data0_131072", "data1_131072", "data2_393216", "data3_12288",
                   "data4_64", "data5_64")
  if any(name not in signature for name in expected_args):
    raise ValueError("calls 6/10 argument layout has changed")

  tile = args.tile
  lines = [signature + " {", f"  for (int l1b=0; l1b<64; l1b+={tile}) {{",
           "    for (int l2=0; l2<64; l2++) {"]
  for q in range(tile):
    for c in range(32):
      lines.append(f"      float a{q}_{c}=0.0f;")
  lines.append("      for (int r=0; r<48; r++) {")
  # The activation values are independent of l1 and are shared by every output
  # group in this tile. Keep them scalar so v68's qfloat path cannot alter rounding.
  for tap in range(4):
    for c in range(32):
      lines.append(f"        float x{tap}_{c}=data2_393216[(r<<13)+(l2<<5)+{tap * 2048 + c}];")
  for q in range(tile):
    weight_index = f"((l1b+{q})*192)+(r<<2)"
    for k in range(4):
      lines.append(f"        __fp16 h{q}_{k}=data3_12288[{weight_index}+{k}];")
      lines.append(f"        float w{q}_{k}=(float)h{q}_{k};")
    for c in range(32):
      terms = [f"a{q}_{c}"] + [f"(x{tap}_{c}*w{q}_{tap})" for tap in range(4)]
      lines.append(f"        a{q}_{c} = " + " + ".join(terms) + ";")
  lines.append("      }")
  for q in range(tile):
    lines.extend((f"      __fp16 res{q}=data4_64[l1b+{q}];",
                  f"      __fp16 scale{q}=data5_64[l1b+{q}];",
                  f"      float b{q}=(float)res{q};",
                  f"      float s{q}=(float)scale{q};"))
    for c in range(32):
      index = f"(((l1b+{q})<<11)+(l2<<5)+{c})"
      lines.append(f"      data0_131072[{index}] = data1_131072[{index}] + ((a{q}_{c}+b{q})*s{q});")
  lines.extend(("    }", "  }", "}", ""))
  source = "\n".join(lines)
  source_path.write_text(source)
  digest = hashlib.sha256(source.encode()).hexdigest()
  for call in manifest["calls"]:
    if call["kernel_c"] == source_name:
      call["source_sha256"] = digest
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"rewrote shared calls 6/10 with tile={tile}; per-output reduction order is unchanged")


if __name__ == "__main__":
  main()
