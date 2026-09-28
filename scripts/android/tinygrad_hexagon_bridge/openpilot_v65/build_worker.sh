#!/bin/bash
# Build onnx-remote-hexagon-worker (tools/onnx-remote/remote_hexagon_worker.cpp) for Android aarch64: the runner side of the
# compiler/runner RPC for tghx-v65 artifacts. Its FastRPC stub comes from the tinygrad fork's tg_graph.idl (the skels in the
# artifacts implement it). Needs TINYGRAD_ROOT, HEXAGON_SDK_ROOT and the NDK (NDK_ROOT or the Debian android-ndk package).
#   build_worker.sh [OUT_DIR]      -> OUT_DIR/onnx-remote-hexagon-worker (default: ./hexagon_worker_build)
set -eo pipefail
: "${TINYGRAD_ROOT:?}" "${HEXAGON_SDK_ROOT:?}"
here=$(cd "$(dirname "$0")" && pwd)
remote="$here/../../../../tools/onnx-remote"
out=$(mkdir -p "${1:-hexagon_worker_build}" && cd "${1:-hexagon_worker_build}" && pwd)
ndk_bin="${NDK_ROOT:-/usr/lib/android-ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin"
cp "$TINYGRAD_ROOT/tinygrad/runtime/support/dsp_graph_files/v65/tg_graph.idl" "$out/"
(cd "$out" && "$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" tg_graph.idl)
inc=(-I "$out" -I "$remote" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
"$ndk_bin/aarch64-linux-android29-clang" -O2 -c "${inc[@]}" -o "$out/tg_graph_stub.o" "$out/tg_graph_stub.c"
"$ndk_bin/aarch64-linux-android29-clang++" -std=c++17 -O2 -Wall "${inc[@]}" -o "$out/onnx-remote-hexagon-worker" \
  "$remote/remote_hexagon_worker.cpp" "$remote/remote_transport.cpp" "$out/tg_graph_stub.o" \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc -static-libstdc++
echo "$out/onnx-remote-hexagon-worker"
