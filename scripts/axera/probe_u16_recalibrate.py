#!/usr/bin/env python3
"""Can a 16-bit step chain move to a new calibration without a Pulsar2 build?

Builds one real step node's chain at U16 twice (calibration A on the reference
batch's tensors; B on the same tensors scaled per input, so every scale and
zero point moves), then runs ``matmul_record_emit.check`` both ways: the
records and ``npu_params`` recalibrated from A onto B must equal the native B
build, and the reverse. Usage: probe_u16_recalibrate.py WORKDIR NODE...
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matmul_record_emit as mre  # noqa: E402
import step_runner as sr  # noqa: E402
import u16_chain  # noqa: E402

FACTORS = (1.3, 0.8, 1.1)  # per input, cycling


def main() -> None:
    work, names = sys.argv[1], sys.argv[2:]
    model = sr.load_step()
    by = {n.name: n for n in model.graph.node}
    inits = {i.name for i in model.graph.initializer}
    plan = {}
    for name in names:
        node = by[name]
        plan[name] = [t for t in node.input if t and t not in inits]
    need = sorted({t for ins in plan.values() for t in ins})
    outs, _ = sr.StepRunner(model, []).run(
        sr.load_reference()["feeds"], "float", keep=need
    )
    for name, ins in plan.items():
        sub, _ = u16_chain.chain_model(model, ins, list(by[name].output))
        a = {t: np.asarray(outs[t], np.float32) for t in ins}
        b = {t: v * np.float32(FACTORS[k % 3]) for k, (t, v) in enumerate(a.items())}
        for tag, data in (("A", a), ("B", b)):
            res = u16_chain.build_chain(work, f"{name}_{tag}", sub, data, "U16")
            print(name, tag, "built" if res.success else "FAILED", flush=True)
        ra = os.path.join(work, f"{name}_A")
        rb = os.path.join(work, f"{name}_B")
        for src, dst in ((ra, rb), (rb, ra)):
            try:
                r = mre.check(src, dst)
                print(
                    name,
                    os.path.basename(src)[-1],
                    "->",
                    os.path.basename(dst)[-1],
                    "record_diffs",
                    len(r["record_diffs"]),
                    "params_diff_bytes",
                    r["params_diff_bytes"],
                    flush=True,
                )
            except Exception as exc:
                print(name, "check failed:", type(exc).__name__, str(exc)[:160], flush=True)


if __name__ == "__main__":
    main()
