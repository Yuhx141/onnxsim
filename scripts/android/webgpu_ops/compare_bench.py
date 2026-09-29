"""Compare the first-run outputs of bench (out/<model>.cpu/N.bin vs out/<model>.webgpu/N.bin), float32."""

import glob
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "out"
for cdir in sorted(glob.glob(f"{root}/*.cpu")):
    name = os.path.basename(cdir)[:-4]
    gdir = f"{root}/{name}.webgpu"
    worst = 0.0
    desc = []
    for cf in sorted(
        glob.glob(f"{cdir}/*.bin"), key=lambda p: int(os.path.basename(p)[:-4])
    ):
        gf = os.path.join(gdir, os.path.basename(cf))
        if not os.path.exists(gf):
            desc.append("missing")
            continue
        c, g = np.fromfile(cf, "f4"), np.fromfile(gf, "f4")
        if c.size != g.size:
            desc.append(f"size {c.size}!={g.size}")
            continue
        scale = max(np.abs(c).max(), 1e-6)
        err = np.abs(c - g).max() / scale
        worst = max(worst, err)
        desc.append(f"{err:.1e}")
    print(
        f"{name:14s} worst max-abs-err/max|cpu| = {worst:.2e}   per-output: {' '.join(desc[:8])}"
    )
