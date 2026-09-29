#!/usr/bin/env bash
# Run every model in ./m on the phone's ORT WebGPU EP, then compare against host CPU ORT.
#   NDK=... ORT_BUILD=<ort build dir with libonnxruntime.so> ORT_SRC=<ort checkout> ./run.sh
# Steps: python gen_ops.py (writes ./m); build probe.cc against the ORT C API; push + run under
# phone-run (shared phone lock); pull outputs to ./out and ./err; python compare.py.
# Extra probe args after the out path are provider option key/value pairs, e.g. `preferredLayout NCHW`.
# Note: the probe builds its input in float32 (n/1000.f-.5f); compare.py builds it in float64 then
# rounds, so exact-tie ops (Equal/Greater/Cast-to-int at a threshold) differ by 1 ulp on any
# backend, including the phone CPU EP. Ignore those.
set -euo pipefail
: "${NDK:?}" "${ORT_BUILD:?}" "${ORT_SRC:?}"
"$NDK"/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ -O2 \
  -I "$ORT_SRC/include/onnxruntime/core/session" probe.cc -L "$ORT_BUILD" -lonnxruntime -o probe
D=/data/local/tmp/wg-ops
~/.cache/android-phone/phone-run bash -c "
adb shell 'rm -rf $D; mkdir -p $D/m $D/out $D/err'
adb push probe m '$ORT_BUILD/libonnxruntime.so' $D/ >/dev/null
adb shell \"cd $D && for f in m/*.onnx; do n=\\\$(basename \\\$f .onnx); LD_LIBRARY_PATH=. timeout 60 ./probe \\\$f webgpu 0 out/\\\$n.bin >/dev/null 2>err/\\\$n.txt; done\"
rm -rf out err; adb pull $D/out out >/dev/null; adb pull $D/err err >/dev/null"
python compare.py
