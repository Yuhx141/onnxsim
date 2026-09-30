#!/usr/bin/env python3
"""Run a pow2-QDQ ResNet through the XDNA layer engine and (optionally) Vitis AI; report time + exactness.

Two interpreters are needed (see docs/xdna-subgraph-dispatch.md): ``--xdna-python`` (IRON + XRT) drives
our runner, ``--ort-python`` (Ryzen AI venv with onnxruntime-vitisai) makes the reference logits and,
with its environment file ``--vitis-env``, times the Vitis AI EP.

    compare_models.py MODEL.onnx --artifact-prefix e_r101 --arch 3,4,23,3
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def stages_of(model_path: Path) -> list[list[str]]:
    import onnx

    stages: dict[int, list[tuple[int, str]]] = {}
    for node in onnx.load(str(model_path)).graph.node:
        m = re.match(r"(/layer(\d)/layer\d\.(\d+))/conv1/Conv$", node.name)
        if m:
            stages.setdefault(int(m.group(2)), []).append((int(m.group(3)), m.group(1)))
    return [[p for _, p in sorted(items)] for _, items in sorted(stages.items())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--artifact-prefix", required=True, help="engine xclbin/insts prefix (layer_engine_design.py output)")
    parser.add_argument("--xdna-python", default="/home/takecheeze/iron-1.4.3-py312-venv/bin/python")
    parser.add_argument("--ort-python", default="/home/takecheeze/ryzen_ai-1.8.0/venv/bin/python")
    parser.add_argument("--work", type=Path, default=Path("/mnt/data/cache/claude-work/xdna/models"))
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    stem = args.model.stem
    stages = stages_of(args.model)
    (args.work / f"{stem}.stages.json").write_text(json.dumps(stages))
    manifest = args.work / f"{stem}.manifest.json"
    if not manifest.exists():
        subprocess.run([args.xdna_python, str(HERE / "emit_resnet_manifest.py"), str(args.model), str(manifest)], check=True)
    out_json = args.work / f"{stem}.xdna.json"
    logits = args.work / f"{stem}.xdna.logits.npy"
    scratch = Path("/mnt/data/cache/claude-work/xdna/scratch")
    cmd = [args.xdna_python, str(HERE / "run_resnet_xdna.py"), str(args.model), str(manifest),
           "--layer-engine", str(scratch / f"{args.artifact_prefix}.xclbin"), str(scratch / f"{args.artifact_prefix}.insts.bin"),
           json.dumps(stages), "--layer-engine-stem", "--cpu-small-m", "100000", "--iters", str(args.iters), "--warmup", "5",
           "--json", str(out_json), "--dump-output", str(logits)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    result = json.loads(out_json.read_text())
    check = (
        "import sys, numpy as np, onnxruntime as ort\n"
        "s = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']); i = s.get_inputs()[0]\n"
        "x = np.random.default_rng(0).random([1, 3, 32, 32], dtype=np.float32)\n"
        "ref = s.run(None, {i.name: x})[0]; got = np.load(sys.argv[2])\n"
        "print(float(abs(ref - got).max()), int(ref.argmax()) == int(got.argmax()))\n"
    )
    ref = subprocess.run([args.ort_python, "-c", check, str(args.model), str(logits)], capture_output=True, text=True, check=True).stdout.split()
    print(f"{stem}: max abs error vs ORT CPU {ref[0]}, argmax match {ref[1]}")
    print(f"{stem}: xdna layer engine avg {result['avg_ms']:.3f} ms (device call {result['profile_ms'].get('fused_stage_kernel_call_ms', 0):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
