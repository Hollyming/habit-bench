#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

METHOD="${1:?usage: scripts/run_shard_with_server.sh METHOD DATASET METHOD_OUTPUT_ROOT SHARD_INDEX SHARD_COUNT}"
DATASET="${2:?usage: scripts/run_shard_with_server.sh METHOD DATASET METHOD_OUTPUT_ROOT SHARD_INDEX SHARD_COUNT}"
METHOD_OUTPUT_ROOT="${3:?usage: scripts/run_shard_with_server.sh METHOD DATASET METHOD_OUTPUT_ROOT SHARD_INDEX SHARD_COUNT}"
SHARD_INDEX="${4:?usage: scripts/run_shard_with_server.sh METHOD DATASET METHOD_OUTPUT_ROOT SHARD_INDEX SHARD_COUNT}"
SHARD_COUNT="${5:?usage: scripts/run_shard_with_server.sh METHOD DATASET METHOD_OUTPUT_ROOT SHARD_INDEX SHARD_COUNT}"

printf -v SHARD_NAME 'shard_%03d_of_%03d' "$SHARD_INDEX" "$SHARD_COUNT"
OUTPUT_DIR="$METHOD_OUTPUT_ROOT/$SHARD_NAME"
mkdir -p "$OUTPUT_DIR"

if [[ "${HABITBENCH_FORCE_RERUN:-0}" != "1" && -f "$OUTPUT_DIR/metrics.json" ]]; then
  echo "skip_completed method=$METHOD shard=$SHARD_INDEX/$SHARD_COUNT output=$OUTPUT_DIR"
  exit 0
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
PORT="${HABITBENCH_VLLM_PORT:-8000}"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export HABITBENCH_METHOD_CUDA_VISIBLE_DEVICES="${HABITBENCH_METHOD_CUDA_VISIBLE_DEVICES:-0}"
SERVER_LOG="$OUTPUT_DIR/vllm_server.log"

bash "$PROJECT_ROOT/scripts/lumia/start_vllm_openai_server.sh" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 "${HABITBENCH_SERVER_READY_ATTEMPTS:-180}"); do
  if curl -fsS "$OPENAI_BASE_URL/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -100 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep "${HABITBENCH_SERVER_READY_SLEEP_SEC:-2}"
done
if [[ "$READY" != "1" ]]; then
  echo "vLLM server did not become ready at $OPENAI_BASE_URL" >&2
  tail -100 "$SERVER_LOG" >&2
  exit 1
fi

bash "$PROJECT_ROOT/scripts/run_user_shard.sh" \
  "$METHOD" \
  "$DATASET" \
  "$METHOD_OUTPUT_ROOT" \
  "$SHARD_INDEX" \
  "$SHARD_COUNT" \
  --base-url "$OPENAI_BASE_URL" \
  --progress-every "${HABITBENCH_PROGRESS_EVERY:-25}"
