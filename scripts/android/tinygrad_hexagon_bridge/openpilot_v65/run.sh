#!/bin/bash
# Generate every openpilot modeld program with tinygrad as a standalone Hexagon v65 (SDM845 cDSP) program, check it bit for bit
# under qemu, build the FastRPC skel + client, and optionally run it on an attached phone. See ../README.md, "openpilot on v65".
#
# ONNX models run through onnxsim's compiler/runner RPC (tools/onnx-remote): compile_v65.sh behind onnx-remote-compiler makes
# the artifact, onnx-remote-hexagon-worker on the phone loads and runs it.
#
#   run.sh compiler [KEY=VAL...]                              start onnx-remote-compiler (COMPILER_PORT, default 39502) for
#                                                             hexagon-v65; KEY=VAL extras go into compile_v65.sh's command
#                                                             (e.g. ONNX_QDQ_INT_CONV=1 ONNX_QDQ_LUT=1 TC_OPT=1 for a QDQ model)
#   run.sh rpc <model.onnx> [client args]                     COMPILE on it, then load_compiled + run_compiled on the phone
#                                                             (e2e.sh; e.g. --input-raw ... --expect ref.bin --iters 5 --profile)
#   run.sh model <name> <model.onnx>                          the same capture locally, kept as g_<name> for inspection
#   run.sh warp  <name> <compile_warp.py args...>             the camera warps modeld builds (compile_warp.py in the fork).
#                                                             They are tinygrad programs with no ONNX form, so they can't use
#                                                             the ONNX compiler contract:
#   run.sh phone <name> [iters] [threads] [batch] [prof]      push g_<name> and its own client to the phone and run it directly
#
# Environment: TINYGRAD_ROOT (the onnxsim/tinygrad fork, branch openpilot-v65-graph), HEXAGON_SDK_ROOT, HEXAGON_TOOLCHAIN,
# CC (a clang that targets hexagonv65; the SDK 6.x compiler starts at v68), WORK (output dir, default ./openpilot_v65_work),
# DSP_THREADS (default 4), DEVICE_SERIAL, PHONE_LOCK (default ~/.cache/android-phone/phone-run, if present),
# REMOTE_BUILD (the tools/onnx-remote CMake build dir with onnx-remote-compiler and onnx-remote-client),
# WORKER (onnx-remote-hexagon-worker; built by build_worker.sh into $WORK/worker when unset).
set -eo pipefail
: "${TINYGRAD_ROOT:?set TINYGRAD_ROOT to the tinygrad fork checkout}"
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
CC="${CC:-clang-19}"; WORK="${WORK:-$PWD/openpilot_v65_work}"; DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
PHONE_LOCK="${PHONE_LOCK:-$HOME/.cache/android-phone/phone-run}"
mkdir -p "$WORK"; cd "$WORK"
ENV=(MOCKDSP=1 DEV=DSP CONV_PAD_MATERIALIZE=1 DSP_V65_HW=1 DSP_THREADS="${DSP_THREADS:-4}" NOLOCALS=1 BEAM=0 CC="$CC" HEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT"
     HEXAGON_TOOLCHAIN="$HEXAGON_TOOLCHAIN" PYTHONUNBUFFERED=1 PYTHONPATH="$TINYGRAD_ROOT/examples/openpilot:$TINYGRAD_ROOT")
MODEL_ENV=(FLOAT16=0 ONNX_FP16_AS_FP32=1 BENCH_RUNS=1 DSP_ALL_INPUTS=1 ALL_OUTPUTS=1)
# heavy (qemu runs every kernel): one job at a time, memory-capped when systemd-run is available
run() { if command -v systemd-run >/dev/null; then
          systemd-run --user --wait --collect --pipe -q -p MemoryMax=24G -p MemorySwapMax=0 --working-directory="$PWD" -E PATH="$PATH" \
            $(for e in "${ENV[@]}" "${EXTRA[@]}"; do printf -- "-E %s " "$e"; done) "$@"
        else env "${ENV[@]}" "${EXTRA[@]}" "$@"; fi; }
export_graph() {  # <name> [--inputs npz]
  local name=$1; shift
  run python3 "$TINYGRAD_ROOT/examples/openpilot/dsp_graph_v65.py" "$WORK/$name.pkl" "$WORK/g_$name" "$@" --qemu --build > "g_$name.log" 2>&1 \
    || { grep -v -i warning "g_$name.log" | tail -20; exit 1; }
  grep -E "^(reference|emitted|built)" "g_$name.log"
}
cmd=${1:-}; name=${2:-}; shift $(( $# < 2 ? $# : 2 ))
case "$cmd" in
  model)
    EXTRA=("${MODEL_ENV[@]}")
    [ -f "$name.pkl" ] || run python3 "$TINYGRAD_ROOT/examples/openpilot/compile3.py" "$1" "$WORK/$name.pkl" > "$name.capture.log" 2>&1 \
      || { tail -20 "$name.capture.log"; exit 1; }
    export_graph "$name" ;;
  warp)
    EXTRA=()
    [ -f "$name.pkl" ] || run python3 "$TINYGRAD_ROOT/examples/openpilot/compile_warp.py" "$@" --output "$WORK/$name.pkl" > "$name.capture.log" 2>&1 \
      || { tail -20 "$name.capture.log"; exit 1; }
    export_graph "$name" --inputs "$WORK/${name}_inputs.npz" ;;
  compiler)
    : "${REMOTE_BUILD:?set REMOTE_BUILD to the tools/onnx-remote build dir}"
    here=$(cd "$(dirname "$0")" && pwd)
    id="tinygrad-$(git -C "$TINYGRAD_ROOT" rev-parse --short HEAD)"
    # $name is the first extra, if any: everything after "compiler" is KEY=VAL
    extras=("$name" "$@"); [ -z "$name" ] && extras=()
    mkdir -p "$WORK/tmp"
    exec env TINYGRAD_ROOT="$TINYGRAD_ROOT" HEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" HEXAGON_TOOLCHAIN="$HEXAGON_TOOLCHAIN" CC="$CC" \
      TMPDIR="$WORK/tmp" "$REMOTE_BUILD/onnx-remote-compiler" --port "${COMPILER_PORT:-39502}" --cache-dir "$WORK/compiler-cache" \
      --target hexagon-v65 --compiler-id "$id" --command "$here/compile_v65.sh {input} {output} {manifest} ${extras[*]}" ;;
  rpc)
    : "${REMOTE_BUILD:?set REMOTE_BUILD to the tools/onnx-remote build dir}"
    here=$(cd "$(dirname "$0")" && pwd)
    worker="${WORKER:-$WORK/worker/onnx-remote-hexagon-worker}"
    [ -x "$worker" ] || "$here/build_worker.sh" "$WORK/worker" > /dev/null
    CLIENT="$REMOTE_BUILD/onnx-remote-client" exec "$here/e2e.sh" "$worker" "$name" "$@" ;;
  phone)
    d="$WORK/g_$name"; R=/data/local/tmp/openpilot_v65/$name
    job="adb -s $DEVICE_SERIAL shell mkdir -p $R && adb -s $DEVICE_SERIAL push -q $d/client $d/tg_graph.so $d/blob.bin $d/input.bin $d/ref.bin $R/ >/dev/null && \
adb -s $DEVICE_SERIAL shell 'cd $R && chmod 755 client && ADSP_LIBRARY_PATH=. ./client \"file:///tg_graph.so?tg_graph_skel_handle_invoke&_modver=1.0&_dom=cdsp\" . $*'"
    if [ -x "$PHONE_LOCK" ]; then PHONE_LOCK_OWNER="${PHONE_LOCK_OWNER:-openpilot-v65}" "$PHONE_LOCK" bash -c "$job"; else bash -c "$job"; fi ;;
  *) sed -n '2,27p' "$0"; exit 2 ;;
esac
