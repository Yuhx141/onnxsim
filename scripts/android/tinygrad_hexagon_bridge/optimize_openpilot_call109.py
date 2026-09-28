#!/usr/bin/env python3
"""Rewrite openpilot call 109 to share weight loads across adjacent positions."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("bundle", type=Path, help="captured openpilot kernel bundle to update in place")
  p.add_argument("--tile", type=int, choices=(2, 4, 8, 16, 32, 64), default=32,
                 help="number of adjacent positions processed together")
  args = p.parse_args()

  manifest_path = args.bundle / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  if len(manifest.get("calls", [])) <= 109 or manifest["calls"][109]["name"] != "r_512_2048_10":
    raise ValueError("call 109 is missing or has changed")
  call = manifest["calls"][109]
  source_path = args.bundle / call["kernel_c"]
  original = source_path.read_text()
  match = re.search(r"__attribute__\(\(noinline\)\) void\s+tg_call_109_r_512_2048_10\s*\(.*?\)\s*\{",
                    original, re.S)
  if match is None:
    raise ValueError("call 109 generated signature is missing or has changed")
  signature = match.group(0)[:-1].rstrip()
  expected_args = ("data0_512", "data1_512", "data2_20480", "data3_1048576", "data4_5120")
  if any(name not in signature for name in expected_args):
    raise ValueError("call 109 argument layout has changed")

  tile = args.tile
  lines = [signature + " {", f"  for (int base = 0; base < 512; base += {tile}) {{",
           f"    float acc[{tile}][10] = {{{{0.0f}}}};", "    for (int r = 0; r < 2048; r++) {"]
  for k in range(10):
    lines.append(f"      float w{k} = data2_20480[r + {k * 2048}];")
  for t in range(tile):
    lines.append(f"      float x{t} = (float)data3_1048576[(r << 9) + base + {t}];")
  for t in range(tile):
    for k in range(10):
      lines.append(f"      acc[{t}][{k}] = acc[{t}][{k}] + w{k} * x{t};")
  lines.append("    }")
  for t in range(tile):
    lines.append(f"    __fp16 residual{t} = data1_512[base + {t}];")
    for k in range(10):
      lines.append(f"    float bias{t}_{k} = data4_5120[base + {t} + {k * 512}];")
    epilogue = []
    for k in range(10):
      epilogue.extend((f"(float)residual{t}", f"acc[{t}][{k}]", f"bias{t}_{k}"))
    lines.append(f"    data0_512[base + {t}] = " + " + ".join(epilogue) + ";")
  lines.extend(("  }", "}", ""))
  source = "\n".join(lines)
  source_path.write_text(source)
  call["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"rewrote call 109 with tile={tile}; each position keeps its original reduction order")


if __name__ == "__main__":
  main()
