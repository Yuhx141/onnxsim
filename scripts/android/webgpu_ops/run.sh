#!/usr/bin/env bash
# Sweep single-op models on the phone's ORT WebGPU EP and compare to host-CPU ORT.
#   NDK=<ndk toolchain root> ORT_BUILD=<ort build dir with libonnxruntime.so> ORT_SRC=<ort checkout> \
#     PYTHON=<python with onnx, onnxruntime, numpy> ./run.sh
# Run it from a scratch directory (it writes ./m ./out ./err ./prof there); it needs `adb`.
# Steps: generate models (gen_ops.py + gen_new_op_tests.py), write x.bin, build probe.cc against the ORT
# C API, push + run under phone-run (shared phone lock), pull outputs, then compare.py + lrn_ref.py.
# The probe reads its input from x.bin (X_BIN), so phone and host see identical bits; without that,
# exact-tie ops (Equal/Greater/Cast-to-int at a threshold) differ by 1 ulp on any backend.
# Extra provider options: pass `key value` pairs after the out path in probe.cc's argv (e.g. preferredLayout NCHW).
set -euo pipefail
: "${NDK:?}" "${ORT_BUILD:?}" "${ORT_SRC:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
rm -rf m out err prof
$PY "$HERE/gen_ops.py" && $PY "$HERE/gen_new_op_tests.py"
$PY -c "
import numpy as np
np.array([((i*2654435761)%1000)/1000-.5 for i in range(3*32*32)], dtype='f').tofile('x.bin')"
"$NDK"/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ -O2 \
  -I "$ORT_SRC/include/onnxruntime/core/session" "$HERE/probe.cc" -L "$ORT_BUILD" -lonnxruntime -o probe
D=/data/local/tmp/wg-ops
~/.cache/android-phone/phone-run bash -c "
adb shell 'rm -rf $D; mkdir -p $D/m $D/out $D/err $D/prof'
adb push probe m x.bin '$ORT_BUILD/libonnxruntime.so' $D/ >/dev/null
adb shell \"cd $D && for f in m/*.onnx; do n=\\\$(basename \\\$f .onnx); X_BIN=x.bin LD_LIBRARY_PATH=. timeout 60 ./probe \\\$f webgpu 0 out/\\\$n.bin >/dev/null 2>err/\\\$n.txt; PROFILE=prof/\\\$n X_BIN=x.bin LD_LIBRARY_PATH=. timeout 60 ./probe \\\$f webgpu 0 /dev/null >/dev/null 2>&1; done\"
rm -rf out err prof; adb pull $D/out out >/dev/null; adb pull $D/err err >/dev/null; adb pull $D/prof prof >/dev/null"
$PY "$HERE/compare.py"
$PY "$HERE/lrn_ref.py" | tail -1
$PY "$HERE/placement.py"
