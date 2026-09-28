#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 WORKER BRIDGE SERVICE_SMOKE DISCOVERY_SMOKE FAILOVER_SMOKE" >&2
  exit 2
fi

worker=$1
bridge=$2
service_smoke=$3
discovery_smoke=$4
failover_smoke=$5
port=39571
fallback_port=39572
log_dir=${TMPDIR:-/tmp}/onnx-remote-ros2-e2e-$$
mkdir -p "$log_dir"
export ROS_DOMAIN_ID=$((100 + $$ % 100))

worker_pid=
announcer_pid=
consumer_pid=
cleanup() {
  for pid in "$consumer_pid" "$announcer_pid" "$worker_pid"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  done
  for pid in "$consumer_pid" "$announcer_pid" "$worker_pid"; do
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
  if [[ ${KEEP_ROS2_E2E_LOGS:-0} == 1 ]]; then
    echo "ROS2 discovery logs: $log_dir"
  else
    rm -rf "$log_dir"
  fi
}
trap cleanup EXIT

"$worker" --port "$port" >"$log_dir/worker.log" 2>&1 &
worker_pid=$!
"$bridge" --ros-args \
  -p remote_host:=127.0.0.1 -p remote_port:="$port" \
  -p runner_id:=runner-a -p discovery_target:=target \
  -p advertise_host:=127.0.0.1 -p announce_period_ms:=100 \
  -p discovery_timeout_ms:=1200 \
  >"$log_dir/announcer.log" 2>&1 &
announcer_pid=$!
"$bridge" --ros-args \
  -p remote_host:=127.0.0.1 -p remote_port:="$fallback_port" \
  -p runner_id:=consumer -p auto_discover:=true \
  -p discovery_target:=target -p announce_period_ms:=100 \
  -p discovery_timeout_ms:=1200 \
  >"$log_dir/consumer.log" 2>&1 &
consumer_pid=$!

"$discovery_smoke"
"$service_smoke"

# The consumer keeps the selected endpoint while runner-a's transient-local
# announcement is live, then reports lease expiry and restores its fallback.
"$failover_smoke" >"$log_dir/failover.log" 2>&1 &
failover_pid=$!
sleep 0.5
kill "$announcer_pid"
wait "$announcer_pid" 2>/dev/null || true
announcer_pid=
wait "$failover_pid"
