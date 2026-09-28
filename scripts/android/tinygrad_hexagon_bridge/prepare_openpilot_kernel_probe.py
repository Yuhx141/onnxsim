#!/usr/bin/env python3
"""Prepare a one-call phone probe from an exported openpilot Tinygrad kernel bundle.

The default call is the captured supercombo stem convolution. Inputs are synthetic except for
its real ONNX weight and bias. The reference output is produced by that call's captured QEMU ELF.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import subprocess
from pathlib import Path

import numpy as np


DTYPES = {"dtypes.float": np.dtype("float32"), "dtypes.half": np.dtype("float16"),
          "dtypes.uint8": np.dtype("uint8"), "dtypes.int": np.dtype("int32")}
C_TYPES = {"dtypes.float": "float", "dtypes.half": "__fp16", "dtypes.uint8": "unsigned char",
           "dtypes.int": "int"}


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("bundle", type=Path)
  p.add_argument("--call", type=int, default=0)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--seed", type=int, default=23)
  p.add_argument("--synthetic-weights", action="store_true",
                 help="use seeded random values for BUFFER args; useful for isolating kernel execution")
  p.add_argument("--relax-half-vector-alignment", action="store_true",
                 help="treat generated FP16 vector loads as naturally half-aligned for fault isolation")
  p.add_argument("--weight-map", action="append", default=[], metavar="BUFFER_SLOT=ONNX_NAME",
                 help="map a captured BUFFER argument to a named weight; defaults map slots 140/163 to stem.0 weight/bias")
  args = p.parse_args()

  manifest = json.loads((args.bundle / "manifest.json").read_text())
  call = next(c for c in manifest["calls"] if c["call"] == args.call)
  weight_map = {140:"supercombo.vision._en.stem.0.reparam_conv.weight",
                163:"supercombo.vision._en.stem.0.reparam_conv.bias"}
  for m in args.weight_map:
    slot, name = m.split("=", 1)
    weight_map[int(slot)] = name
  weight_index = {w["name"]:w for w in manifest.get("weights", [])}
  weight_blob = (args.bundle / manifest["weights_file"]).read_bytes()
  call_args = call["arguments"]
  if len(call_args) < 2: raise ValueError("call must have an output and at least one input")
  out_meta = call_args[0]
  out_dtype = DTYPES[out_meta["dtype"]]
  out_bytes = int(np.prod(out_meta["shape"])) * out_dtype.itemsize
  rng = np.random.default_rng(args.seed)
  packed = bytearray()
  packed_inputs, qemu_inputs, offsets = [], [], []
  for meta in call_args[1:]:
    dtype = DTYPES[meta["dtype"]]
    nbytes = int(np.prod(meta["shape"])) * dtype.itemsize
    packed.extend(b"\0" * ((-len(packed)) % 128))
    offset = len(packed)
    slot = meta.get("slot")
    if meta["op"] == "BUFFER":
      if args.synthetic_weights:
        array = rng.normal(0, 0.25, meta["shape"]).astype(dtype)
        data, label = array.tobytes(), f"synthetic_weight_slot_{slot}"
      elif slot not in weight_map or weight_map[slot] not in weight_index:
        raise ValueError(f"no ONNX weight mapping for buffer slot {slot}; pass --weight-map {slot}=NAME")
      else:
        weight = weight_index[weight_map[slot]]
        if weight["nbytes"] != nbytes or np.dtype(weight["dtype"]) != dtype:
          raise ValueError(f"weight {weight['name']} dtype/size does not match call argument {slot}")
        data = weight_blob[weight["offset"]:weight["offset"]+weight["nbytes"]]
        label = weight["name"]
    else:
      array = rng.normal(0, 0.25, meta["shape"]).astype(dtype)
      data, label = array.tobytes(), f"synthetic_slot_{slot}"
    packed.extend(data)
    packed_inputs.append({"label":label, "slot":slot, "offset":offset, "nbytes":nbytes,
                          "dtype":meta["dtype"], "shape":meta["shape"]})
    qemu_inputs.append(data)
    offsets.append(offset)

  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / "probe_input.bin").write_bytes(packed)
  (args.out / "probe_expected.bin").write_bytes(b"\0" * out_bytes)
  sim_elf = args.bundle / call["simulator_elf"]
  os.chmod(sim_elf, 0o755)  # DSPProgram does this before launching QEMU as well.
  qemu_in = b"\0" * out_bytes + b"".join(qemu_inputs)
  proc = subprocess.run(["qemu-hexagon-static", str(sim_elf)], input=qemu_in, stdout=subprocess.PIPE, check=True)
  expected = proc.stdout[4:4+out_bytes]
  if len(expected) != out_bytes: raise ValueError("QEMU reference returned a truncated output")
  (args.out / "probe_expected.bin").write_bytes(expected)

  kernel_c = (args.bundle / call["kernel_c"]).resolve()
  source = kernel_c.read_text()
  if args.relax_half_vector_alignment:
    source = re.sub(r"typedef __fp16 (\w+) __attribute__\(\(aligned\(\d+\),ext_vector_type\((\d+)\)\)\);",
                    r"typedef __fp16 \1 __attribute__((aligned(2),ext_vector_type(\2)));", source)
    kernel_c = args.out / "relaxed_half_vector_kernel.c"
    args.out.mkdir(parents=True, exist_ok=True)
    kernel_c.write_text(source)
  fn_name = call["exported_name"]
  signature = re.search(rf"void\s+{re.escape(fn_name)}\((.*?)\)\s*\{{", source, re.S)
  if signature is None: raise ValueError(f"missing function {fn_name} in {kernel_c}")
  params = [x.strip() for x in signature.group(1).split(",")]
  if len(params) != len(call_args): raise ValueError("manifest call args do not match generated function signature")
  cast_args = [f"({C_TYPES[x['dtype']]}*)" + ("output_aligned" if i == 0 else f"(input_aligned+{offsets[i-1]})")
               for i,x in enumerate(call_args)]
  adapter = f'''#include <stddef.h>
#include <string.h>
#include "{kernel_c}"
static unsigned char input_aligned[{len(packed)}] __attribute__((aligned(128)));
static unsigned char output_aligned[{out_bytes}] __attribute__((aligned(128)));
void tinygrad_openpilot_probe(const unsigned char *input_ptr, unsigned char *output_ptr) {{
  memcpy(input_aligned, input_ptr, {len(packed)});
  {fn_name}({', '.join(cast_args)});
  memcpy(output_ptr, output_aligned, {out_bytes});
}}
'''
  (args.out / "probe_adapter.c").write_text(adapter)
  desc = {"call":args.call, "kernel":call["name"], "exported_name":fn_name,
          "packed_input_bytes":len(packed), "output_bytes":out_bytes, "inputs":packed_inputs,
          "expected_file":"probe_expected.bin", "expected_sha256":__import__('hashlib').sha256(expected).hexdigest(),
          "qemu_instruction_count":int.from_bytes(proc.stdout[:4], "little")}
  (args.out / "probe.json").write_text(json.dumps(desc, indent=2)+"\n")
  print(f"prepared call {args.call} ({call['name']}): input {len(packed)} B, output {out_bytes} B, "
        f"QEMU instruction count {desc['qemu_instruction_count']}")


if __name__ == "__main__": main()
