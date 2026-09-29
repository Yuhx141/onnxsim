"""Which execution provider ran each node? Reads ORT profile JSON (probe.cc with PROFILE=<prefix>)
from ./prof and reports nodes that ran on the CPU EP, i.e. ops the WebGPU EP silently fell back on.
(int64 Casts show up here unless the EP option enableInt64 is on.)"""

import collections
import glob
import json
import os
import re

tot = collections.Counter()
cpu_ops = collections.defaultdict(set)
for f in sorted(glob.glob("prof/*.json")):
    model = re.sub(r"_\d{4}-.*$", "", os.path.basename(f))
    for e in json.load(open(f)):
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            p = e["args"].get("provider", "?")
            tot[p] += 1
            if p == "CPUExecutionProvider":
                cpu_ops[e["args"]["op_name"]].add(model)
print("node events by provider:", dict(tot))
for op, models in sorted(cpu_ops.items()):
    print(f"CPU-EP op {op}: {len(models)} models, e.g. {sorted(models)[:3]}")
