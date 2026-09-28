#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 WORKER DORA_NODE" >&2
  exit 2
fi

worker=$1
node=$2
port=39573
log_dir=${TMPDIR:-/tmp}/onnx-remote-dora-e2e-$$
mkdir -p "$log_dir"
worker_pid=
cleanup() {
  if [[ -n "$worker_pid" ]]; then
    kill "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" 2>/dev/null || true
  fi
  if [[ ${KEEP_DORA_E2E_LOGS:-0} == 1 ]]; then
    echo "DORA adapter logs: $log_dir"
  else
    rm -rf "$log_dir"
  fi
}
trap cleanup EXIT

"$worker" --port "$port" >"$log_dir/worker.log" 2>&1 &
worker_pid=$!
ONNXSIM_DORA_STUB_RUNTIME=1 \
ONNXSIM_DORA_ANNOUNCE=1 \
ONNXSIM_DORA_PUBLISH_PROFILE_EVENTS=1 \
ONNXSIM_DORA_REMOTE_HOST=127.0.0.1 \
ONNXSIM_DORA_REMOTE_PORT="$port" \
  "$node" >"$log_dir/node.log" 2>&1
