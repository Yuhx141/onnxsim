#!/usr/bin/env bash
# Latency sweep on the phone: every model in $MODELS_DIR on the CPU EP (4 threads) and the WebGPU EP,
# with the first run's outputs dumped for a correctness check (compare_bench.py).
#   NDK=... ORT_BUILD=... ORT_SRC=... MODELS_DIR=<dir with the float models> ./bench_all.sh
# Models and per-model settings are listed in MODELS below: name:extra-args:warmup:iters.
# Runs under phone-run (the shared phone lock), so other agents' phone jobs queue with it.
set -euo pipefail
: "${NDK:?}" "${ORT_BUILD:?}" "${ORT_SRC:?}" "${MODELS_DIR:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
MODELS=${MODELS:-"resnet50:shape=pixel_values:1,3,224,224:3:20
yolo11n::3:20 yolo26n::3:20 rtdetr_pre::3:10 rtdetr_mid0::3:30 rtdetr_mid1::3:30 rtdetr_post::3:30
sam_l0_enc::2:8 sam_l0_dec::3:20"}
"$NDK"/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ -std=c++17 -O2 \
  -I "$ORT_SRC/include/onnxruntime/core/session" "$HERE/bench.cc" -L "$ORT_BUILD" -lonnxruntime -o bench
D=/data/local/tmp/wg-bench
rm -rf out && mkdir out
{
  echo "adb shell 'mkdir -p $D/m $D/out'"
  echo "adb push bench '$ORT_BUILD/libonnxruntime.so' $D/ >/dev/null"
  echo "adb push $MODELS_DIR/. $D/m/ >/dev/null"
  echo "adb shell 'cd $D && rm -rf out && mkdir out && cat /sys/class/kgsl/kgsl-3d0/gpuclk /sys/class/kgsl/kgsl-3d0/max_gpuclk 2>/dev/null | tr \"\\n\" \" \"; echo'"
  for spec in $MODELS; do
    IFS=: read -r name extra warm iters <<<"$spec"
    # extra may itself contain ':' (shape=in:1,3,...): re-split on the last two ':'
    extra=${spec#*:}; extra=${extra%:*:*}; warm=${spec%:*}; warm=${warm##*:}; iters=${spec##*:}
    for p in cpu webgpu; do
      echo "adb shell 'cd $D && mkdir -p out/$name.$p && LD_LIBRARY_PATH=. timeout 600 ./bench m/$name.onnx $p $warm $iters threads=4 dump=out/$name.$p $extra 2>&1 | grep -E \"^RESULT|ERR\"'"
    done
  done
  echo "adb shell 'cat /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null'"
  echo "adb pull $D/out/. out/ >/dev/null"
} > run_on_phone.sh
~/.cache/android-phone/phone-run bash run_on_phone.sh | tee bench_results.txt
