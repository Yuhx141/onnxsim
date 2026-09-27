#!/usr/bin/env python3
"""Upload a compiled RKNN model and benchmark it on a Luckfox RV1106 board.

Authentication is delegated to the user's normal ``ssh``/``scp`` setup.  For
the stock Buildroot image, configure a key or use an SSH agent first; the
documented password is ``luckfox`` for ``root``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "luckfox_rknn_runner.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="host-side compiled .rknn model")
    ap.add_argument("--host", default="192.168.0.238")
    ap.add_argument("--user", default="root")
    ap.add_argument("--remote-dir", default="/tmp/onnxsim-rknn")
    ap.add_argument("--input-shape", default="1,3,224,224")
    ap.add_argument("--input-type", choices=("int8", "float32"), default="int8")
    ap.add_argument("--input-format", choices=("nchw", "nhwc"), default="nhwc")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--ssh-option", action="append", default=[])
    args = ap.parse_args()
    model = os.path.abspath(args.model)
    if not os.path.isfile(model):
        ap.error(f"model does not exist: {model}")
    target = f"{args.user}@{args.host}"
    opts = [item for value in args.ssh_option for item in ("-o", value)]
    remote_model = f"{args.remote_dir}/model.rknn"
    remote_runner = f"{args.remote_dir}/luckfox_rknn_runner.py"
    subprocess.run(["ssh", *opts, target, "mkdir", "-p", args.remote_dir], check=True)
    subprocess.run(["scp", *opts, model, f"{target}:{remote_model}"], check=True)
    subprocess.run(["scp", *opts, RUNNER, f"{target}:{remote_runner}"], check=True)
    command = " ".join([
        "python3", shlex.quote(remote_runner), shlex.quote(remote_model),
        "--input-shape", shlex.quote(args.input_shape),
        "--input-type", args.input_type,
        "--input-format", args.input_format,
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
    ])
    proc = subprocess.run(["ssh", *opts, target, command], check=True,
                          capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Luckfox runner returned no JSON")
    result = json.loads(lines[-1])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
