#!/bin/bash
# Start / stop onnx-remote-hexagon-worker on the attached phone and forward its port, for clients that talk to the runner
# themselves (scripts/openpilot_dsp/evaluate.py --backend phone). e2e.sh does the same around a single --compile-run.
#   phone_worker.sh start WORKER_BINARY      push, start, adb forward tcp:$RUNNER_PORT
#   phone_worker.sh stop
# Env: DEVICE_SERIAL, RUNNER_PORT (39520), THREADS (worker --threads override). Hold the phone lock around the whole session, e.g.
#   ~/.cache/android-phone/phone-run bash -c 'phone_worker.sh start W; python evaluate.py ...; phone_worker.sh stop'
set -eo pipefail
cmd=$1
serial="${DEVICE_SERIAL:-239dbd8f}"; port="${RUNNER_PORT:-39520}"; R=/data/local/tmp/onnx-remote-hexagon
case "$cmd" in
  start)
    worker=${2:?worker binary}
    adb -s "$serial" shell mkdir -p $R
    adb -s "$serial" push -q "$worker" $R/onnx-remote-hexagon-worker > /dev/null
    adb -s "$serial" shell 'kill $(pidof onnx-remote-hexagon-worker) 2>/dev/null; chmod 755 '"$R"'/onnx-remote-hexagon-worker' || true
    # adb shell waits for what it starts, so the host backgrounds it
    adb -s "$serial" shell "cd $R && exec ./onnx-remote-hexagon-worker --port $port --cache-dir $R/cache ${THREADS:+--threads $THREADS}" \
      > "${WORKER_LOG:-worker.log}" 2>&1 &
    adb -s "$serial" forward tcp:"$port" tcp:"$port" > /dev/null
    sleep 2 ;;
  stop)
    adb -s "$serial" shell 'kill $(pidof onnx-remote-hexagon-worker) 2>/dev/null' || true
    adb -s "$serial" forward --remove tcp:"$port" || true ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac
