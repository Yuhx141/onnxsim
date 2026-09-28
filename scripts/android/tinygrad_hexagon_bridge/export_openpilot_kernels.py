#!/usr/bin/env python3
"""Export the DSP programs captured by tinygrad's openpilot compile3.py as inspectable artifacts.

The resulting C files and captured ELF images are per-kernel codegen outputs. The manifest keeps
call order plus backing storage slots and byte offsets for view arguments; the optional ONNX blob
contains weights. A phone-side graph runner is still a separate runtime component.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tinygrad.uop.ops import Ops

OPENPILOT_STEM0 = "r_2_64_128_32_24_3_3"
OPENPILOT_STEM0_WEIGHT = "supercombo.vision._en.stem.0.reparam_conv.weight"


def optimize_stem0_kernel(source: str) -> str:
  """Use contiguous output-channel weight vectors in the v68 stem convolution."""
  cast_re = re.compile(r"float32 cast(\d+) = __builtin_convertvector\(\(__fp1632\)\{[^\n]*?\}, float32\);")
  taps: list[int] = []
  def replace_cast(m: re.Match[str]) -> str:
    tap = int(m.group(1))
    taps.append(tap)
    return (f"__fp1632 packed{tap} = *((__fp1632*)(data3_13824 + ((Ridx0*9+{tap})*64+(Lidx3*32))));\n"
            f"          float32 cast{tap} = __builtin_convertvector(packed{tap}, float32);")
  source, count = cast_re.subn(replace_cast, source)
  if count != 9 or taps != list(range(9)):
    raise ValueError("call 0 source is not the expected vectorized stem convolution")
  source, removed = re.subn(r"^\s*__(?:fp1632|fp168|fp16) val(?:[1-9]|[1-5][0-9]|6[0-4]) = .*;\n", "", source, flags=re.M)
  if removed < 64: raise ValueError(f"expected 64 obsolete stem filter loads, removed {removed}")
  source, removed_alu = re.subn(r"^\s*int alu7 = \(\(Ridx0\*9\)\+\(Lidx3\*6912\)\);\n", "", source, flags=re.M)
  if removed_alu != 1: raise ValueError(f"expected one obsolete stem filter index, removed {removed_alu}")
  return source


def pack_stem0_weights(blob: bytearray, weights: list[dict], np) -> dict:
  record = next((w for w in weights if w["name"] == OPENPILOT_STEM0_WEIGHT), None)
  if record is None or record["shape"] != [64, 24, 3, 3] or record["dtype"] != "float16":
    raise ValueError("missing expected FP16 openpilot stem.0 convolution weights")
  start, end = record["offset"], record["offset"] + record["nbytes"]
  tensor = np.frombuffer(blob[start:end], dtype=np.float16).copy().reshape(64, 24, 3, 3)
  packed = np.ascontiguousarray(tensor.transpose(1, 2, 3, 0)).view(np.uint8).reshape(-1)
  if len(packed) != end - start: raise ValueError("packed stem weight size changed")
  blob[start:end] = packed.tobytes()
  record["source_shape"] = record["shape"]
  record["shape"] = [24, 3, 3, 64]
  record["layout"] = "IHW_O"
  record["sha256"] = hashlib.sha256(packed).hexdigest()
  return {"name": record["name"], "source_shape": record["source_shape"], "packed_shape": record["shape"],
          "offset": start, "nbytes": len(packed), "layout": "IHW_O"}


def describe_arg(uop) -> dict:
  arg = uop.arg
  param = getattr(arg, "slot", None)
  if param is None: param = getattr(arg, "idx", None)
  out = {"op": uop.op.name, "dtype": str(uop.dtype)}
  if param is not None: out["slot"] = param
  if hasattr(arg, "device"): out["device"] = arg.device
  if hasattr(arg, "size"): out["size"] = arg.size
  if getattr(uop, "_shape", None) is not None:
    out["shape"] = [int(x) if isinstance(x, int) else str(x) for x in uop.shape]
  # CALL arguments can be BITCAST/SHRINK views into a shared linear buffer.
  # Recording only the outer op loses the arena slot and byte offset needed
  # to replay calls in order on a phone.
  view, byte_offset = uop, 0
  while view.op in (Ops.BITCAST, Ops.SHRINK, Ops.RESHAPE):
    if view.op is Ops.SHRINK and len(view.src) >= 3:
      start = view.src[1].arg
      if isinstance(start, int): byte_offset += start * view.src[0].dtype.itemsize
    view = view.src[0]
  if view.op in (Ops.BUFFER, Ops.PARAM):
    backing = getattr(view.arg, "slot", getattr(view.arg, "idx", None))
    if backing is not None:
      out["storage"] = {"op": view.op.name, "slot": backing, "byte_offset": byte_offset,
                        "dtype": str(view.dtype),
                        "shape": [int(x) if isinstance(x, int) else str(x) for x in view.shape]}
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("pickle", type=Path, help="output pickle from examples/openpilot/compile3.py")
  parser.add_argument("--tinygrad-root", type=Path, required=True, help="tinygrad checkout used to create the pickle")
  parser.add_argument("--out", type=Path, required=True, help="directory for per-call C, ELF, and manifest")
  parser.add_argument("--hexagon-clang", type=Path, help="also compile kernel-only C to v65 ELF objects")
  parser.add_argument("--hex-arch", default="v65", help="HVX architecture for optional object compilation")
  parser.add_argument("--onnx", type=Path, help="also pack ONNX initializer tensors into an aligned weights blob")
  parser.add_argument("--optimize-openpilot-stem0", action="store_true",
                      help="pack stem.0 weights by input/tap/output channel and vectorize call 0 (v68+ only)")
  args = parser.parse_args()
  if args.optimize_openpilot_stem0 and (int(args.hex_arch.lstrip("v")) < 68 or not args.onnx):
    parser.error("--optimize-openpilot-stem0 requires --hex-arch v68+ and --onnx")

  tinygrad_root = args.tinygrad_root.resolve()
  sys.path.insert(0, str(tinygrad_root / "examples" / "openpilot"))
  import compile3

  args.out.mkdir(parents=True, exist_ok=True)
  kernels_dir = args.out / "kernels"
  kernels_dir.mkdir(exist_ok=True)
  with args.pickle.open("rb") as f: jit = compile3.load_pickle(f)
  captured = jit.captured
  calls = [u for u in captured.linear.toposort(gate=lambda x: x.op is not Ops.PROGRAM)
           if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM]

  records, artifact_by_hash = [], {}
  for i, call in enumerate(calls):
    program = call.src[0]
    source_uop = next(s for s in program.src if s.op is Ops.SOURCE)
    binary_uop = next(s for s in program.src if s.op is Ops.BINARY)
    source, binary = source_uop.arg, binary_uop.arg
    kernel_input = source
    if args.optimize_openpilot_stem0 and program.arg.name == OPENPILOT_STEM0:
      kernel_input = optimize_stem0_kernel(source)
    source_hash = hashlib.sha256(kernel_input.encode()).hexdigest()
    if source_hash not in artifact_by_hash:
      stem = f"{i:03d}_{program.arg.name}"
      sim_c_path, sim_elf_path = kernels_dir / f"{stem}.sim.c", kernels_dir / f"{stem}.sim.elf"
      sim_c_path.write_text(source)
      sim_elf_path.write_bytes(binary)
    else:
      stem = artifact_by_hash[source_hash]["stem"]
      sim_c_path = kernels_dir / f"{stem}.sim.c"
      sim_elf_path = kernels_dir / f"{stem}.sim.elf"
    func = re.search(r"__attribute__\(\(noinline\)\) void ([rE]_[A-Za-z0-9_]+)\(", kernel_input)
    if func is None: raise ValueError(f"can't find generated kernel function in {program.arg.name}")
    brace = kernel_input.index("{", func.end())
    depth, end = 1, brace+1
    while depth:
      if kernel_input[end] == "{": depth += 1
      elif kernel_input[end] == "}": depth -= 1
      end += 1
    if source_hash not in artifact_by_hash:
      exported_name = f"tg_call_{i}_{program.arg.name}"
      kernel_source = kernel_input[:func.start()] + kernel_input[func.start():end].replace(func.group(1), exported_name, 1)
      # Keep generated typedefs and intrinsic helpers, but omit _start and mock syscalls.
      kernel_source_path = kernels_dir / f"{stem}.kernel.c"
      kernel_source_path.write_text(kernel_source)
      object_path = None
      if args.hexagon_clang:
        object_path = kernels_dir / f"{stem}.o"
        if not object_path.exists():
          subprocess.run([
            str(args.hexagon_clang), "--target=hexagon", "-c", "-O2", "-Wall", "-Werror", "-fno-stack-protector",
            "-x", "c", "-fPIC", "-ffreestanding", "-nostdlib", f"-mcpu=hexagon{args.hex_arch}",
            f"-mhvx={args.hex_arch}", "-mhvx-length=128b", "-o", str(object_path), str(kernel_source_path),
          ], check=True)
      artifact_by_hash[source_hash] = {"stem":stem, "exported_name":exported_name, "kernel_source_path":kernel_source_path,
                                       "object_path":object_path}
    artifact = artifact_by_hash[source_hash]
    records.append({
      "call": i, "name": program.arg.name, "exported_name": artifact["exported_name"], "target": str(program.arg.target),
      "kernel_c": str(artifact["kernel_source_path"].relative_to(args.out)),
      **({"object": str(artifact["object_path"].relative_to(args.out))} if artifact["object_path"] else {}),
      "simulator_c": str(sim_c_path.relative_to(args.out)),
      "simulator_elf": str(sim_elf_path.relative_to(args.out)),
      "source_sha256": source_hash,
      "elf_sha256": hashlib.sha256(binary).hexdigest(),
      "arguments": [describe_arg(u) for u in call.src[1:]],
    })

  weights, packed_optimizations = [], []
  if args.onnx:
    import numpy as np
    from tinygrad.nn.onnx import OnnxRunner
    runner = OnnxRunner(args.onnx)
    blob = bytearray()
    for name, tensor in runner.graph_values.items():
      if tensor is None or not hasattr(tensor, "numpy"): continue
      array = np.ascontiguousarray(tensor.numpy())
      padding = (-len(blob)) % 128
      blob.extend(b"\0" * padding)
      payload = array.tobytes()
      weights.append({"name":name, "dtype":str(array.dtype), "shape":list(array.shape), "offset":len(blob),
                      "nbytes":len(payload), "sha256":hashlib.sha256(payload).hexdigest()})
      blob.extend(payload)
    if args.optimize_openpilot_stem0:
      packed_optimizations.append(pack_stem0_weights(blob, weights, np))
    weight_path = args.out / "weights.bin"
    weight_path.write_bytes(blob)
  storage = {}
  for call in records:
    for arg_index, arg in enumerate(call["arguments"]):
      backing = arg.get("storage")
      if backing is None: continue
      key = f'{backing["op"]}:{backing["slot"]}'
      item = storage.setdefault(key, {**backing, "key":key, "call_arguments":[]})
      item["call_arguments"].append([call["call"], arg_index])
  manifest = {
    "format": 2, "model_pickle": str(args.pickle.resolve()), "tinygrad_root": str(tinygrad_root),
    "hvx_arch": args.hex_arch if args.hexagon_clang else os.getenv("HVX_ARCH", "v65"), "kernel_calls": len(records),
    "input_names": captured.expected_names, "calls": records, "storage": list(storage.values()),
  }
  if args.onnx:
    manifest["onnx_model"] = str(args.onnx.resolve())
    manifest["weights_file"] = "weights.bin"
    manifest["weights_nbytes"] = len(blob)
    manifest["weights"] = weights
  if packed_optimizations: manifest["packed_optimizations"] = packed_optimizations
  (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"exported {len(records)} DSP calls to {args.out}")


if __name__ == "__main__": main()
