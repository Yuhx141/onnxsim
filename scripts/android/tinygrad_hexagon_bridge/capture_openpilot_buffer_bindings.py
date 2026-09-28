#!/usr/bin/env python3
"""Replay a dynamic Tinygrad capture under QEMU and identify static BUFFER slots by content.

The exported manifest has call/argument storage identities but not initializer names. This tool
replays the graph once, hashes the actual bytes passed to each DSP kernel, and matches them to
the ONNX initializer blob. Non-initializer buffers are saved as binary constants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("pickle", type=Path)
  p.add_argument("bundle", type=Path, help="exported dynamic bundle with manifest.json and weights.bin")
  p.add_argument("--tinygrad-root", type=Path, required=True)
  p.add_argument("--out", type=Path, required=True)
  args = p.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)

  sys.path.insert(0, str(args.tinygrad_root.resolve()))
  sys.path.insert(0, str((args.tinygrad_root / "examples" / "openpilot").resolve()))
  from tinygrad import Tensor
  import tinygrad.runtime.ops_dsp as dsp
  import compile3

  manifest = json.loads((args.bundle / "manifest.json").read_text())
  weight_blob = (args.bundle / manifest["weights_file"]).read_bytes()
  weights_by_hash: dict[str, list[dict]] = {}
  for weight in manifest["weights"]:
    payload = weight_blob[weight["offset"]:weight["offset"] + weight["nbytes"]]
    if len(payload) != weight["nbytes"]: raise ValueError(f"truncated ONNX weight {weight['name']}")
    weights_by_hash.setdefault(hashlib.sha256(payload).hexdigest(), []).append(weight)

  with args.pickle.open("rb") as f: jit = compile3.load_pickle(f)
  expected = jit.captured.expected_input_info
  if len(expected) != len(manifest["input_names"]): raise ValueError("capture input count differs from manifest")
  # Zero inputs suffice: only constant weight bytes and call/storage identities are recorded.
  inputs = {name: Tensor(np.zeros(tuple(view.shape), dtype=np.float32), device=device)
            for name, (view, _vars, _dtype, device) in zip(manifest["input_names"], expected, strict=True)}

  calls: list[list[dict]] = []
  original_call = dsp.MockDSPProgram.__call__
  def recording_call(self, *bufs, **kw):
    args = []
    for i, (signature, buf) in enumerate(zip(self.signature, bufs, strict=True)):
      _name, _program_slot, dtype, shape = signature
      payload = bytes(dsp.to_mv(buf.va_addr, buf.size))
      args.append({"arg": i, "shape": list(shape), "dtype": str(dtype), "nbytes": len(payload),
                   "sha256": hashlib.sha256(payload).hexdigest()})
      if args[-1]["sha256"] not in weights_by_hash: args[-1]["data"] = payload
    calls.append(args)
    return original_call(self, *bufs, **kw)

  dsp.MockDSPProgram.__call__ = recording_call
  try:
    output = jit(**inputs).numpy()
  finally:
    dsp.MockDSPProgram.__call__ = original_call
  if output.shape != (1, 6504): raise ValueError(f"unexpected replay output shape {output.shape}")
  if len(calls) < len(manifest["calls"]): raise ValueError(f"recorded only {len(calls)} DSP calls")
  graph_calls = calls[-len(manifest["calls"]):]

  params = {s["slot"]:s for s in manifest["storage"] if s["op"] == "PARAM"}
  packed_layout, cursor = [], 0
  for slot, name in enumerate(manifest["input_names"]):
    if slot not in params: raise ValueError(f"missing PARAM slot {slot} for {name}")
    cursor = (cursor + 127) & ~127
    nbytes = int(np.prod(params[slot]["shape"])) * 4
    packed_layout.append({"name":name,"slot":slot,"offset":cursor,"nbytes":nbytes,
                         "shape":params[slot]["shape"],"dtype":"float32"})
    cursor += nbytes
  (args.out / "zero_input.bin").write_bytes(bytes(cursor))
  (args.out / "expected_zero_output.bin").write_bytes(np.ascontiguousarray(output).tobytes())
  (args.out / "input_layout.json").write_text(json.dumps({"input_bytes":cursor,"inputs":packed_layout,
      "output_bytes":output.nbytes,"calls":len(graph_calls)},indent=2)+"\n")

  slot_map: dict[int, dict] = {}
  constants: dict[int, dict] = {}
  for call_index, (record, call_args) in enumerate(zip(manifest["calls"], graph_calls, strict=True)):
    if len(record["arguments"]) != len(call_args):
      raise ValueError(f"call {call_index}: manifest has {len(record['arguments'])} args, replay has {len(call_args)}")
    for arg_index, (metadata, runtime_arg) in enumerate(zip(record["arguments"], call_args, strict=True)):
      storage = metadata.get("storage")
      if storage is None or storage["op"] != "BUFFER" or storage["slot"] == 753: continue
      digest = runtime_arg["sha256"]
      candidates = weights_by_hash.get(digest, [])
      if candidates:
        weight = candidates[0]
        binding = {"name": weight["name"], "offset": weight["offset"], "nbytes": weight["nbytes"],
                   "sha256": digest}
        old = slot_map.get(storage["slot"])
        if old is not None and old != binding:
          raise ValueError(f"BUFFER:{storage['slot']} matched two different initializer payloads")
        slot_map[storage["slot"]] = binding
      elif storage["slot"] != 605:
        payload = runtime_arg.pop("data")
        constants[storage["slot"]] = {"nbytes": len(payload), "sha256": digest,
                                       "file": f"buffer_{storage['slot']}.bin"}
        (args.out / f"buffer_{storage['slot']}.bin").write_bytes(payload)

  weight_names = {w["name"] for w in manifest["weights"]}
  bound_names = {v["name"] for v in slot_map.values()}
  if bound_names != weight_names:
    raise ValueError(f"initializer mapping incomplete: missing={len(weight_names-bound_names)}, "
                     f"unexpected={len(bound_names-weight_names)}")
  buffer_slots = {s["slot"] for s in manifest["storage"] if s["op"] == "BUFFER" and s["slot"] not in (753, 605)}
  if set(slot_map) | set(constants) != buffer_slots:
    raise ValueError(f"BUFFER mapping incomplete: missing slots={sorted(buffer_slots-set(slot_map)-set(constants))}")
  result = {"format": 1, "calls": len(graph_calls), "output_shape": list(output.shape),
            "weight_slots": {str(k): v for k, v in sorted(slot_map.items())},
            "constant_slots": {str(k): v for k, v in sorted(constants.items())}}
  (args.out / "buffer_bindings.json").write_text(json.dumps(result, indent=2) + "\n")
  print(f"verified {len(slot_map)} initializer slots and {len(constants)} constant slots across "
        f"{len(graph_calls)} calls; output {output.shape}")


if __name__ == "__main__": main()
