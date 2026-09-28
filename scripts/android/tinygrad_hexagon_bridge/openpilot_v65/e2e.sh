#!/bin/bash
# Compiler/runner RPC end to end on an attached phone: onnx-remote-hexagon-worker is pushed and started on the device, reached
# through adb forward, and onnx-remote-client --compile-run drives COMPILE (on the compiler service, e.g. compile_v65.sh behind
# onnx-remote-compiler), load_compiled and run_compiled. The phone part holds the shared phone lock when it exists.
#   e2e.sh WORKER_BINARY MODEL.onnx [client args: --input-raw ... --expect/--dump ... --iters N --profile]
# Env: CLIENT (onnx-remote-client), COMPILER_HOST/COMPILER_PORT (127.0.0.1:39502), RUNNER_PORT (39520), DEVICE_SERIAL,
# THREADS (worker --threads override), PHONE_LOCK.
set -eo pipefail
worker=$1; model=$2; shift 2
: "${CLIENT:?set CLIENT to onnx-remote-client}"
serial="${DEVICE_SERIAL:-239dbd8f}"; port="${RUNNER_PORT:-39520}"; lock="${PHONE_LOCK:-$HOME/.cache/android-phone/phone-run}"
chost="${COMPILER_HOST:-127.0.0.1}"; cport="${COMPILER_PORT:-39502}"
# compile first, outside the phone lock (a cache miss can take a long time; later calls hit the compiler's cache)
"$CLIENT" --compile "$chost" "$cport" "$model" > /dev/null
R=/data/local/tmp/onnx-remote-hexagon
args=$(printf "%q " "$@")
job="set -e
adb -s $serial shell mkdir -p $R
adb -s $serial push -q '$worker' $R/onnx-remote-hexagon-worker >/dev/null
adb -s $serial shell 'kill \$(pidof onnx-remote-hexagon-worker) 2>/dev/null; chmod 755 $R/onnx-remote-hexagon-worker' || true
# adb shell waits for anything started under it, so the host backgrounds the adb shell instead
adb -s $serial shell \"cd $R && exec ./onnx-remote-hexagon-worker --port $port --cache-dir $R/cache ${THREADS:+--threads $THREADS}\" > worker.log 2>&1 &
adb -s $serial forward tcp:$port tcp:$port >/dev/null
sleep 2
status=0
'$CLIENT' --compile-run $chost $cport 127.0.0.1 $port '$model' $args || status=\$?
adb -s $serial shell 'kill \$(pidof onnx-remote-hexagon-worker) 2>/dev/null' || true
adb -s $serial forward --remove tcp:$port || true
exit \$status"
if [ -x "$lock" ]; then PHONE_LOCK_OWNER="${PHONE_LOCK_OWNER:-onnx-remote-hexagon}" "$lock" bash -c "$job"; else bash -c "$job"; fi
