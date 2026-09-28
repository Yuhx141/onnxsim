#!/bin/bash
# Builds and runs the from-scratch, TVM-free Hexagon FastRPC transport PoC end to end:
#   mini_rpc.idl --(qaic)--> mini_rpc_{skel,stub}.c, mini_rpc.h
#   mini_rpc_skel.c + mini_rpc_impl.c --(hexagon-clang + hexagon-link)--> mini_rpc.so   (DSP side)
#   client_main.c + mini_rpc_stub.c   --(Android NDK aarch64 clang)-->    mini_client  (host side)
#   adb push both to the device, run mini_client, which drives mini_rpc.so purely via
#   libcdsprpc.so's remote_handle64_open/invoke/close -- no tvm.rpc, no tvm.contrib.hexagon,
#   no MinRPC, no libhexagon_rpc_skel.so anywhere in the loop. See ../README.md for the
#   two real bugs this hit and how they were root-caused.
#
# Requires: HEXAGON_SDK_ROOT, HEXAGON_TOOLCHAIN (Hexagon v73 clang/link), the Android NDK
# (aarch64-linux-android<api>-clang on PATH or NDK_CLANG set), and an adb-reachable device
# (DEVICE_SERIAL, default 239dbd8f).
set -euo pipefail

: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT to the Hexagon SDK root (contains incs/, tools/)}"
: "${HEXAGON_TOOLCHAIN:?set HEXAGON_TOOLCHAIN to .../tools/HEXAGON_Tools/<ver>/Tools}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
HEX_ARCH="${HEX_ARCH:-v73}"
LINK_ARCH="${LINK_ARCH:-$HEX_ARCH}"
PROBE_DIR="${OPENPILOT_PROBE_DIR:-}"
PROBE_CPPFLAGS=()
PROBE_OBJECTS=()
PROBE_CLIENT_DEFS=()
GRAPH_DIR="${OPENPILOT_GRAPH_DIR:-}"
GRAPH_BUNDLE_DIR="${OPENPILOT_GRAPH_BUNDLE_DIR:-}"
GRAPH_BINDINGS_DIR="${OPENPILOT_GRAPH_BINDINGS_DIR:-}"
GRAPH_CPPFLAGS=()
GRAPH_OBJECTS=()
GRAPH_CLIENT_DEFS=()

if [ -n "$PROBE_DIR" ] && [ -n "$GRAPH_DIR" ]; then
  echo "OPENPILOT_PROBE_DIR and OPENPILOT_GRAPH_DIR cannot be used together" >&2
  exit 2
fi

if [ -n "$PROBE_DIR" ]; then
  PROBE_DIR="$(realpath "$PROBE_DIR")"
  PROBE_INPUT_BYTES="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["packed_input_bytes"])' "$PROBE_DIR/probe.json")"
  PROBE_OUTPUT_BYTES="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_bytes"])' "$PROBE_DIR/probe.json")"
  MODEL_ARCH="${MODEL_ARCH:-v65}"
  MODEL_CLANG="${MODEL_CLANG:-/usr/bin/clang-19}"
  PROBE_CPPFLAGS=(-DTG_OPENPILOT_PROBE -DTG_PROBE_INPUT_BYTES="$PROBE_INPUT_BYTES" -DTG_PROBE_OUTPUT_BYTES="$PROBE_OUTPUT_BYTES")
  if [ "${PROBE_TRANSPORT_ONLY:-0}" = 1 ]; then PROBE_CPPFLAGS+=("-DTG_PROBE_TRANSPORT_ONLY"); fi
  PROBE_CLIENT_DEFS=("${PROBE_CPPFLAGS[@]}")
  PROBE_OBJECTS=(openpilot_probe.o)
fi

if [ -n "$GRAPH_DIR" ]; then
  GRAPH_DIR="$(realpath "$GRAPH_DIR")"
  GRAPH_BUNDLE_DIR="$(realpath "${GRAPH_BUNDLE_DIR:?set OPENPILOT_GRAPH_BUNDLE_DIR to the exported kernel bundle}")"
  GRAPH_BINDINGS_DIR="$(realpath "${GRAPH_BINDINGS_DIR:?set OPENPILOT_GRAPH_BINDINGS_DIR to the captured binding directory}")"
  GRAPH_INPUT_FILE="${OPENPILOT_GRAPH_INPUT:-$GRAPH_BINDINGS_DIR/zero_input.bin}"
  GRAPH_EXPECTED_FILE="$GRAPH_BINDINGS_DIR/expected_zero_output.bin"
  GRAPH_LAYOUT="$GRAPH_DIR/input_layout.json"
  GRAPH_INPUT_BYTES="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["input_bytes"])' "$GRAPH_LAYOUT")"
  GRAPH_OUTPUT_BYTES="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_bytes"])' "$GRAPH_LAYOUT")"
  GRAPH_WEIGHT_BYTES="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["weight_bytes"])' "$GRAPH_LAYOUT")"
  GRAPH_CALLS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["calls"])' "$GRAPH_LAYOUT")"
  GRAPH_BATCH_CALLS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch_calls"])' "$GRAPH_LAYOUT")"
  GRAPH_CPPFLAGS=(-DTG_OPENPILOT_GRAPH -DTG_GRAPH_INPUT_BYTES="$GRAPH_INPUT_BYTES"
                  -DTG_GRAPH_OUTPUT_BYTES="$GRAPH_OUTPUT_BYTES")
  GRAPH_CLIENT_DEFS=("${GRAPH_CPPFLAGS[@]}" -DTG_GRAPH_WEIGHT_BYTES="$GRAPH_WEIGHT_BYTES"
                     -DTG_GRAPH_WEIGHT_CHUNK=8388608 -DTG_GRAPH_CALLS="$GRAPH_CALLS"
                     -DTG_GRAPH_BATCH_CALLS="$GRAPH_BATCH_CALLS")
  MODEL_ARCH="${MODEL_ARCH:-v65}"
  MODEL_CLANG="${MODEL_CLANG:-/usr/bin/clang-19}"
fi

cd "$(dirname "$0")"

QAIC="$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic"
"$QAIC" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" mini_rpc.idl

"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -Wall \
  -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  -o mini_rpc_skel.o mini_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
  -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" "${PROBE_CPPFLAGS[@]}" "${GRAPH_CPPFLAGS[@]}" \
  -o mini_rpc_impl.o mini_rpc_impl.c

if [ -n "$PROBE_DIR" ]; then
  "$MODEL_CLANG" --target=hexagon -c -O2 -Wall -Werror -fno-stack-protector -x c -fPIC -ffreestanding -nostdlib \
    -mcpu=hexagon"$MODEL_ARCH" -mhvx="$MODEL_ARCH" -mhvx-length=128b \
    -o openpilot_probe.o "$PROBE_DIR/probe_adapter.c"
fi

if [ -n "$GRAPH_DIR" ]; then
  "$MODEL_CLANG" --target=hexagon -c -O2 -Wall -Werror -fno-stack-protector -x c -fPIC -ffreestanding -nostdlib \
    -mcpu=hexagon"$MODEL_ARCH" -mhvx="$MODEL_ARCH" -mhvx-length=128b \
    -o "$GRAPH_DIR/openpilot_graph_driver.o" "$GRAPH_DIR/openpilot_graph_driver.c"
  GRAPH_OBJECTS+=("$GRAPH_DIR/openpilot_graph_driver.o")
  while IFS=$'\t' read -r kernel_src kernel_obj; do
    src="$GRAPH_BUNDLE_DIR/$kernel_src"
    obj="$GRAPH_DIR/$kernel_obj"
    if [ "${OPENPILOT_GRAPH_PREBUILT:-0}" = 1 ]; then
      if [ ! -f "$obj" ]; then echo "missing prebuilt graph object: $obj" >&2; exit 2; fi
    elif [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
      "$MODEL_CLANG" --target=hexagon -c -O2 -Wall -Werror -fno-stack-protector -x c -fPIC -ffreestanding -nostdlib \
        -mcpu=hexagon"$MODEL_ARCH" -mhvx="$MODEL_ARCH" -mhvx-length=128b -o "$obj" "$src"
    fi
    GRAPH_OBJECTS+=("$obj")
  done < <(python3 - "$GRAPH_LAYOUT" <<'PY'
import json,sys
for obj in json.load(open(sys.argv[1]))["objects"]: print(obj["source"]+"\t"+obj["object"])
PY
  )
fi

LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$LINK_ARCH/G0"
# Tinygrad-generated FP16 kernels can lower half-to-float conversions to
# __extendhfsf2. The SDK shared libgcc leaves this helper unresolved in the
# FastRPC module, so extract the routine from its archive and link it directly.
HELPER_DIR="${TMPDIR:-/tmp}/tinygrad_hexagon_helpers"
mkdir -p "$HELPER_DIR"
"$HEXAGON_TOOLCHAIN/bin/hexagon-ar" p "$LIBPATH/pic/libgcc.a" extendhfsf2.o > "$HELPER_DIR/extendhfsf2.o"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic \
  -o mini_rpc.so mini_rpc_skel.o mini_rpc_impl.o "${PROBE_OBJECTS[@]}" "${GRAPH_OBJECTS[@]}" \
  "$HELPER_DIR/extendhfsf2.o" "$LIBPATH/pic/libgcc.so"

"$NDK_CLANG" -O2 "${PROBE_CLIENT_DEFS[@]}" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  "${GRAPH_CLIENT_DEFS[@]}" \
  -o mini_client client_main.c mini_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc

adb -s "$DEVICE_SERIAL" shell "mkdir -p /data/local/tmp/native_transport"
adb -s "$DEVICE_SERIAL" push mini_client mini_rpc.so /data/local/tmp/native_transport/
adb -s "$DEVICE_SERIAL" shell "chmod 755 /data/local/tmp/native_transport/mini_client"
# If gen_gemm_test_data.py's output is present, push it too so client_main.c's real-kernel test
# (cin=64,cout=256,m=54400, hex_gemm_kernel.py's own default shape) runs instead of being skipped.
if [ -f gemm_a.bin ] && [ -f gemm_bp.bin ]; then
  adb -s "$DEVICE_SERIAL" push gemm_a.bin gemm_bp.bin /data/local/tmp/native_transport/
fi
if [ -f boxhead_a.bin ] && [ -f boxhead_bp.bin ]; then
  adb -s "$DEVICE_SERIAL" push boxhead_a.bin boxhead_bp.bin /data/local/tmp/native_transport/
fi
if [ -n "$PROBE_DIR" ]; then
  adb -s "$DEVICE_SERIAL" push "$PROBE_DIR/probe_input.bin" /data/local/tmp/native_transport/openpilot_probe_input.bin
fi
if [ -n "$GRAPH_DIR" ]; then
  adb -s "$DEVICE_SERIAL" push "$GRAPH_BUNDLE_DIR/weights.bin" /data/local/tmp/native_transport/openpilot_graph_weights.bin
  adb -s "$DEVICE_SERIAL" push "$GRAPH_INPUT_FILE" /data/local/tmp/native_transport/openpilot_graph_input.bin
  adb -s "$DEVICE_SERIAL" push "$GRAPH_EXPECTED_FILE" /data/local/tmp/native_transport/openpilot_graph_expected.bin
  adb -s "$DEVICE_SERIAL" shell "rm -f /data/local/tmp/native_transport/openpilot_graph_output.bin"
fi
PROBE_STATUS=0
adb -s "$DEVICE_SERIAL" shell "cd /data/local/tmp/native_transport && \
  LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=/data/local/tmp/native_transport \
  ./mini_client 'file:///mini_rpc.so?mini_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp'" || PROBE_STATUS=$?
if [ -n "$PROBE_DIR" ]; then
  adb -s "$DEVICE_SERIAL" pull /data/local/tmp/native_transport/openpilot_probe_output.bin "$PROBE_DIR/phone_output.bin"
fi
if [ -n "$GRAPH_DIR" ]; then
  adb -s "$DEVICE_SERIAL" pull /data/local/tmp/native_transport/openpilot_graph_output.bin "$GRAPH_BINDINGS_DIR/phone_output.bin"
fi
if [ "$PROBE_STATUS" -ne 0 ]; then exit "$PROBE_STATUS"; fi
