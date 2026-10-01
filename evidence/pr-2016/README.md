# Runtime and C++ IR evidence for onnxsim PR #2016

This package records the requested reference executions for issues #1995, #1996, and #1998–#2003. Every case uses the public `onnxsim.simplify` entry point. Source and target models are checked and then executed with ONNX Runtime 1.30.0 using `CPUExecutionProvider`, disabled graph optimization, one thread, and fixed inputs.

## Revisions

- pre-fix base: `cbbd9bc3d13cb98a6b926ff0b191f2e1fa227d79`
- PR head: `f508c227e8968ec224009cba6678a58e8cbc8cf1`
- custom ONNX IR: `281e5e6accfd6113fd57d819ce47f32a937bb106`
- onnx-optimizer submodule: `8c13b168c37e74b610d0cccba4e9e01618c612a7`

Both revisions were built from clean source trees. Wheel and extension hashes are in `environment.json`.

## Recorded results

| Issue | Exact-base behavior through `onnxsim.simplify` |
|---|---|
| #1995 | The custom IR and full checker accept the rewritten model; ORT rejects the rank-4 Conv bias. |
| #1996 | The IR can export the rewritten model; the full checker and ORT reject `Reshape [0, -1]` with `allowzero=1`. |
| #1998 | The process fails in C++ IR mutation at `eraseOutput` before a target can be serialized. |
| #1999 | Simplification leaves `v_in` unresolved and fails topological validation before returning a target. |
| #2000 | Attention and GQA differ from their sources by maximum absolute values `6.5055865` and `1.8620088`. |
| #2001 | LayerNorm and RMSNorm each differ by about `0.4951630`. |
| #2002 | The finite additive-mask case differs by `5.0`. |
| #2003 | GELU differs by `0.0359993`; LayerNorm changes the finite/NaN mask. |

Disabling only the affected pass restores exact output in all 11 cases. On the PR head, all 11 targets pass the full checker and execute in ORT; ten are exact, and RoPE is allclose with maximum finite absolute difference `1.1920928955078125e-07` and identical non-finite masks.

The native C++ IR round-trip results are under `native/`. They distinguish IR import/export acceptance from full-checker and runtime rejection.

## Package layout

- `cases/`: fixed source models, inputs, and pass names.
- `reproduce.py`: isolated actual-path replay and result verifier.
- `recorded/`: raw stdout/stderr, models, and JSON results for exact base, pass-disabled base, and PR head.
- `native/`: standalone IR round-trip source, build notes, and recorded results.
- `environment.json`: dependency versions and build hashes.
- `SHA256SUMS`: hashes for every published evidence file.

## Reproduce

Use Python 3.11 with `numpy==2.4.6`, `onnx==1.22.0`, and `onnxruntime==1.30.0`. From a checkout of this evidence commit:

```bash
python -m venv /tmp/onnxsim-pr2016-venv
/tmp/onnxsim-pr2016-venv/bin/pip install \
  'setuptools>=77' wheel 'nanobind>=2.12.0' cmake ninja \
  numpy==2.4.6 onnx==1.22.0 onnxruntime==1.30.0

git worktree add /tmp/onnxsim-pr2016-base cbbd9bc3d13cb98a6b926ff0b191f2e1fa227d79
git worktree add /tmp/onnxsim-pr2016-head f508c227e8968ec224009cba6678a58e8cbc8cf1
git -C /tmp/onnxsim-pr2016-base submodule update --init --recursive
git -C /tmp/onnxsim-pr2016-head submodule update --init --recursive

ONNXSIM_NON_CORE_FEATURES=0 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 \
  /tmp/onnxsim-pr2016-venv/bin/pip wheel /tmp/onnxsim-pr2016-base --no-build-isolation -w /tmp/base-wheel
ONNXSIM_NON_CORE_FEATURES=0 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 \
  /tmp/onnxsim-pr2016-venv/bin/pip wheel /tmp/onnxsim-pr2016-head --no-build-isolation -w /tmp/head-wheel

/tmp/onnxsim-pr2016-venv/bin/pip install --no-deps --target /tmp/base-site /tmp/base-wheel/onnxsim-*.whl
/tmp/onnxsim-pr2016-venv/bin/pip install --no-deps --target /tmp/head-site /tmp/head-wheel/onnxsim-*.whl

EVIDENCE="$PWD/evidence/pr-2016"
PYTHONPATH=/tmp/base-site /tmp/onnxsim-pr2016-venv/bin/python "$EVIDENCE/reproduce.py" run \
  --mode default --revision cbbd9bc3d13cb98a6b926ff0b191f2e1fa227d79 --output /tmp/pr2016-rerun/base
PYTHONPATH=/tmp/base-site /tmp/onnxsim-pr2016-venv/bin/python "$EVIDENCE/reproduce.py" run \
  --mode disabled --revision cbbd9bc3d13cb98a6b926ff0b191f2e1fa227d79 --output /tmp/pr2016-rerun/disabled
PYTHONPATH=/tmp/head-site /tmp/onnxsim-pr2016-venv/bin/python "$EVIDENCE/reproduce.py" run \
  --mode default --revision f508c227e8968ec224009cba6678a58e8cbc8cf1 --output /tmp/pr2016-rerun/head
/tmp/onnxsim-pr2016-venv/bin/python "$EVIDENCE/reproduce.py" verify --root /tmp/pr2016-rerun
```

The verifier expects 11 base failures or differences, 11 exact pass-disabled controls, and ten exact plus one allclose PR-head comparison.
