#!/usr/bin/env python3
"""Generate C dispatch code for a captured Tinygrad graph and its verified buffer bindings."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


DTYPE_BYTES = {"dtypes.float":4, "dtypes.half":2, "dtypes.int":4, "dtypes.uint8":1, "dtypes.char":1}
BATCH_CALLS = 8
C_TYPE = {"dtypes.float":"float", "dtypes.half":"__fp16", "dtypes.int":"int32_t",
          "dtypes.uint8":"uint8_t", "dtypes.char":"int8_t"}


def elements(shape: list) -> int:
  return math.prod(int(x) for x in shape)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("bundle", type=Path)
  p.add_argument("bindings", type=Path)
  p.add_argument("--out", type=Path, required=True)
  args = p.parse_args()
  manifest = json.loads((args.bundle / "manifest.json").read_text())
  binding_doc = json.loads(args.bindings.read_text())
  if binding_doc["calls"] != len(manifest["calls"]): raise ValueError("binding/call count mismatch")
  weight_slots = {int(k):v for k,v in binding_doc["weight_slots"].items()}
  constant_slots = {int(k):v for k,v in binding_doc["constant_slots"].items()}
  buffer_meta = {s["slot"]:s for s in manifest["storage"] if s["op"] == "BUFFER"}
  param_meta = {s["slot"]:s for s in manifest["storage"] if s["op"] == "PARAM"}
  if set(weight_slots) != {s for s in buffer_meta if s not in (753,605,*constant_slots)}:
    raise ValueError("weight slot map does not cover exactly the static weight buffers")
  if set(constant_slots) != set(buffer_meta)-set(weight_slots)-{753,605}:
    raise ValueError("constant slot map does not cover exactly the remaining static buffers")

  args.out.mkdir(parents=True, exist_ok=True)
  input_layout, input_offset = [], 0
  for name, slot in zip(manifest["input_names"], range(len(manifest["input_names"])), strict=True):
    if slot not in param_meta: raise ValueError(f"missing PARAM slot for input {name}: {slot}")
    input_offset = (input_offset + 127) & ~127
    storage = param_meta[slot]
    nbytes = elements(storage["shape"]) * DTYPE_BYTES[storage["dtype"]]
    input_layout.append({"name":name, "slot":slot, "offset":input_offset, "nbytes":nbytes,
                         "shape":storage["shape"], "dtype":storage["dtype"]})
    input_offset += nbytes
  input_bytes = input_offset
  output_meta = buffer_meta[605]
  output_bytes = elements(output_meta["shape"]) * DTYPE_BYTES[output_meta["dtype"]]
  weight_bytes = manifest["weights_nbytes"]

  prototypes, object_sources, dispatch = {}, {}, []
  for call in manifest["calls"]:
    cpath = (args.bundle / call["kernel_c"]).resolve()
    source = cpath.read_text()
    fn = call["exported_name"]
    match = re.search(rf"\bvoid\s+{re.escape(fn)}\s*\((.*?)\)\s*\{{", source, re.S)
    if match is None: raise ValueError(f"missing generated function {fn}")
    proto = f"void {fn}({match.group(1)});"
    prototypes[fn] = proto
    object_sources[call["source_sha256"]] = {"source":call["kernel_c"], "object":f"kernel_{call['source_sha256'][:16]}.o"}
    expressions=[]
    if len(call["arguments"]) != len(match.group(1).split(",")):
      raise ValueError(f"call {call['call']} signature and manifest argument count differ")
    for arg in call["arguments"]:
      storage=arg.get("storage")
      if storage is None: raise ValueError(f"call {call['call']} argument has no storage binding")
      slot, offset = storage["slot"], int(storage["byte_offset"])
      if storage["op"] == "PARAM":
        base=f"(uint8_t*)tg_param_{slot}"
      elif slot == 753:
        base="tg_scratch"
      elif slot == 605:
        base="(uint8_t*)tg_output"
      elif slot in weight_slots:
        weight=weight_slots[slot]
        actual=next(w for w in manifest["weights"] if w["name"] == weight["name"])
        if weight["offset"] != actual["offset"] or weight["nbytes"] != actual["nbytes"]:
          raise ValueError(f"stale weight binding for BUFFER:{slot}")
        base=f"tg_weights + {weight['offset']}"
      elif slot in constant_slots:
        base=f"tg_constant_{slot}"
      else: raise ValueError(f"no pointer source for {storage['op']}:{slot}")
      expressions.append(f"({C_TYPE[arg['dtype']]}*)({base} + {offset})")
    dispatch.append(f"  {fn}({', '.join(expressions)});")

  lines=["#include <stdint.h>", "#include <stddef.h>", "",
         "static void tg_copy(void *dst, const void *src, size_t n) __attribute__((noinline));",
         "static void tg_copy(void *dst, const void *src, size_t n) {",
         "  volatile uint8_t *d = (volatile uint8_t*)dst;",
         "  const volatile uint8_t *s = (const volatile uint8_t*)src;",
         "  for (size_t i=0; i<n; i++) d[i]=s[i];", "}", "",
         *prototypes.values(), "",
         "static uint8_t tg_weights[%d] __attribute__((aligned(128)));" % weight_bytes,
         "static size_t tg_weights_loaded = 0;",
         "static uint8_t tg_scratch[%d] __attribute__((aligned(128)));" % elements(buffer_meta[753]["shape"]),
         "static float tg_output[%d] __attribute__((aligned(128)));" % elements(output_meta["shape"])]
  for item in input_layout:
    lines.append(f"static float tg_param_{item['slot']}[{elements(item['shape'])}] __attribute__((aligned(128)));")
  for slot, item in constant_slots.items():
    payload=(args.bindings.parent / item["file"]).read_bytes()
    if len(payload) != item["nbytes"]: raise ValueError(f"bad constant bytes for BUFFER:{slot}")
    if __import__("hashlib").sha256(payload).hexdigest() != item["sha256"]: raise ValueError(f"bad constant hash for BUFFER:{slot}")
    lines.append(f"static const uint8_t tg_constant_{slot}[{len(payload)}] __attribute__((aligned(128))) = "
                 "{" + ",".join(str(x) for x in payload) + "};")
  lines += ["", "int tg_openpilot_load_weights(int offset, const uint8_t *data, int size) {",
            "  if (offset < 0 || size < 0 || (size_t)offset + (size_t)size > sizeof(tg_weights)) return -1;",
            "  if ((size_t)offset != tg_weights_loaded) return -2;",
            "  tg_copy(tg_weights + offset, data, (size_t)size);",
            "  tg_weights_loaded += (size_t)size;",
            "  return 0;", "}", "",
            f"int tg_openpilot_begin(const uint8_t *input, int input_bytes) {{",
            f"  if (input == NULL || input_bytes != {input_bytes}) return -1;",
            f"  if (tg_weights_loaded != {weight_bytes}) return -2;"]
  for item in input_layout:
    lines.append(f"  tg_copy(tg_param_{item['slot']}, input + {item['offset']}, {item['nbytes']});")
  lines += ["  return 0;", "}", "", "int tg_openpilot_step(int start, int count) {",
            f"  if (start < 0 || count <= 0 || start + count > {len(dispatch)}) return -1;"]
  for i, call in enumerate(dispatch):
    lines += [f"  if (start <= {i} && {i} < start + count) {{", call.strip(), "  }"]
  lines += ["  return 0;", "}", "", "int tg_openpilot_finish(uint8_t *output, int output_capacity) {",
            f"  if (output == NULL || output_capacity < {output_bytes}) return -1;",
            f"  tg_copy(output, tg_output, {output_bytes});", "  return 0;", "}", ""]
  (args.out / "openpilot_graph_driver.c").write_text("\n".join(lines))
  (args.out / "input_layout.json").write_text(json.dumps({"input_bytes":input_bytes,"inputs":input_layout,
      "output_bytes":output_bytes,"weight_bytes":weight_bytes,"calls":len(dispatch),"batch_calls":BATCH_CALLS,
      "objects":list(object_sources.values())},indent=2)+"\n")
  print(f"generated dispatcher: {len(dispatch)} calls, {len(object_sources)} kernel objects, "
        f"{input_bytes} input bytes, {output_bytes} output bytes, {weight_bytes} weight bytes")


if __name__ == "__main__": main()
