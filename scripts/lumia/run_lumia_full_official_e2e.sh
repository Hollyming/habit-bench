#!/usr/bin/env bash
set -euo pipefail

# End-to-end Lumia runner for HABIT-Bench full official subset experiments.
# It downloads/caches models, starts a local vLLM OpenAI-compatible endpoint,
# waits until the endpoint passes the structured-output smoke test, runs the
# full official suite, and then stops the server it started.

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

DATA="${1:-${HABITBENCH_DATASET:-./runs/habit_bench_balanced_v0_3_official_subset_90}}"
OUT="${2:-${HABITBENCH_OFFICIAL_OUT:-$DATA/full_official_results}}"
RUN_ID="${HABITBENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${HABITBENCH_LUMIA_LOG_DIR:-$OUT/run_manifests/$RUN_ID}"
WAIT_SEC="${HABITBENCH_ENDPOINT_WAIT_SEC:-900}"
POLL_SEC="${HABITBENCH_ENDPOINT_POLL_SEC:-15}"
SKIP_DOWNLOAD="${HABITBENCH_SKIP_MODEL_DOWNLOAD:-0}"
REUSE_SERVER="${HABITBENCH_REUSE_SERVER:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "No python3 or python executable found on PATH" >&2
    exit 127
  fi
fi
export PYTHON_BIN

mkdir -p "$LOG_DIR" "$OUT"
export HABITBENCH_RUN_ID="$RUN_ID"
export HABITBENCH_RUN_MANIFEST_DIR="${HABITBENCH_RUN_MANIFEST_DIR:-$LOG_DIR}"

SERVER_PID=""
FINALIZED=0

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "Stopping vLLM server pid=$SERVER_PID" | tee -a "$LOG_DIR/e2e.log"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}

finalize() {
  local exit_code="$1"
  if [[ "$FINALIZED" == "1" ]]; then
    return
  fi
  FINALIZED=1
  "$PYTHON_BIN" ./scripts/lumia/write_run_manifest.py \
    --out "$LOG_DIR/e2e_end_manifest.json" \
    --stage "e2e_end" \
    --dataset-dir "$DATA" \
    --results-dir "$OUT" \
    --command "bash ./scripts/lumia/run_lumia_full_official_e2e.sh $DATA $OUT" \
    --extra "run_id=$RUN_ID" \
    --extra "server_pid=${SERVER_PID:-reused}" \
    --extra "exit_code=$exit_code" \
    --extra "reuse_server=$REUSE_SERVER" \
    --extra "skip_download=$SKIP_DOWNLOAD" \
    --note "E2E Lumia full official run final manifest. Nonzero exit_code means the run did not complete successfully." \
    >/dev/null 2>"$LOG_DIR/e2e_end_manifest.stderr" || true
}

on_exit() {
  local exit_code="$?"
  finalize "$exit_code"
  cleanup
  exit "$exit_code"
}
trap on_exit EXIT

"$PYTHON_BIN" ./scripts/lumia/write_run_manifest.py \
  --out "$LOG_DIR/e2e_start_manifest.json" \
  --stage "e2e_start" \
  --dataset-dir "$DATA" \
  --results-dir "$OUT" \
  --command "bash ./scripts/lumia/run_lumia_full_official_e2e.sh $DATA $OUT" \
  --extra "run_id=$RUN_ID" \
  --extra "reuse_server=$REUSE_SERVER" \
  --extra "skip_download=$SKIP_DOWNLOAD"

PREFLIGHT_ARGS=(
  --dataset-dir "$DATA"
  --results-dir "$OUT"
  --out "$LOG_DIR/lumia_preflight_manifest.json"
)
if [[ "$REUSE_SERVER" == "1" ]]; then
  PREFLIGHT_ARGS+=(--allow-no-gpu)
fi
"$PYTHON_BIN" ./scripts/lumia/preflight_lumia_run.py "${PREFLIGHT_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/lumia_preflight.log"

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  bash ./scripts/lumia/download_open_models.sh \
    2>&1 | tee "$LOG_DIR/model_download.log"
fi

if [[ "$REUSE_SERVER" == "1" ]]; then
  echo "Reusing existing OpenAI-compatible endpoint at ${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}" \
    | tee -a "$LOG_DIR/e2e.log"
else
  echo "Starting vLLM endpoint; logs: $LOG_DIR/vllm_server.log" | tee -a "$LOG_DIR/e2e.log"
  bash ./scripts/lumia/start_vllm_openai_server.sh \
    >"$LOG_DIR/vllm_server.log" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$LOG_DIR/vllm_server.pid"
fi

deadline=$((SECONDS + WAIT_SEC))
until "$PYTHON_BIN" ./scripts/lumia/check_openai_endpoint.py \
  >"$LOG_DIR/openai_endpoint_check.json" 2>"$LOG_DIR/openai_endpoint_check.stderr"; do
  if (( SECONDS >= deadline )); then
    echo "Endpoint did not become ready within ${WAIT_SEC}s" | tee -a "$LOG_DIR/e2e.log"
    tail -n 80 "$LOG_DIR/vllm_server.log" || true
    exit 1
  fi
  echo "Waiting for endpoint... (${POLL_SEC}s)" | tee -a "$LOG_DIR/e2e.log"
  sleep "$POLL_SEC"
done

bash ./scripts/run_full_official_subset_suite.sh "$DATA" "$OUT" \
  2>&1 | tee "$LOG_DIR/full_official_suite.log"

finalize 0

echo "E2E Lumia full official run finished. Results: $OUT"
echo "Logs/manifests: $LOG_DIR"
