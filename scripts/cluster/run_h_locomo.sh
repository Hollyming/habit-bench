#!/usr/bin/env bash
set -euo pipefail

# H-cluster worker wrapper for the LoCoMo method/sample suite. One RJob
# Replica owns eight H200s; the Python runner starts one Qwen3-8B server per
# GPU and joins all Replicas through the GPFS atomic task queue.

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PLAN=""
ENV_FILE=""
PORT_BASE=8100

usage() {
  echo "Usage: scripts/cluster/run_h_locomo.sh --plan PATH --env-file PATH [--port-base N]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) PLAN="${2:?missing value for --plan}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?missing value for --env-file}"; shift 2 ;;
    --port-base) PORT_BASE="${2:?missing value for --port-base}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$PLAN" ]] || { echo "LoCoMo plan not found: $PLAN" >&2; exit 1; }
[[ -f "$ENV_FILE" ]] || { echo "H environment file not found: $ENV_FILE" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN must be set by the H environment file}"
VLLM_PYTHON="${HABITBENCH_VLLM_PYTHON:?HABITBENCH_VLLM_PYTHON must be set by the H environment file}"

REPLICA_INDEX="${NODE_RANK:-0}"
REPLICA_COUNT="${NODE_COUNT:-1}"
if ! [[ "$REPLICA_INDEX" =~ ^[0-9]+$ && "$REPLICA_COUNT" =~ ^[1-9][0-9]*$ ]] || (( REPLICA_INDEX >= REPLICA_COUNT )); then
  echo "Invalid RJob NODE_RANK/NODE_COUNT: $REPLICA_INDEX/$REPLICA_COUNT" >&2
  exit 1
fi
if [[ "$REPLICA_COUNT" != "1" && -z "${JOB_ID:-}" ]]; then
  echo "Multi-Replica LoCoMo evaluation requires RJob JOB_ID" >&2
  exit 1
fi

# The H base image places its system libstdc++ ahead of persistent Conda
# environments.  vLLM imports sqlite/diskcache during API-server startup;
# the frozen vLLM environment's ICU requires a newer CXXABI than the image
# provides.  Put the matching persistent runtimes first for every child
# process (vLLM first, then the method environment), as in run_h_eval.sh.
METHOD_ENV_LIB="$(dirname "$(dirname "$PYTHON_BIN")")/lib"
VLLM_ENV_LIB="$(dirname "$(dirname "$VLLM_PYTHON")")/lib"
for env_lib in "$METHOD_ENV_LIB" "$VLLM_ENV_LIB"; do
  if [[ ! -d "$env_lib" ]]; then
    echo "Persistent Conda runtime directory does not exist: $env_lib" >&2
    exit 1
  fi
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$env_lib:"*) ;;
    *) export LD_LIBRARY_PATH="$env_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
done
if ! "$VLLM_PYTHON" -c 'import sqlite3' >/dev/null 2>&1; then
  echo "vLLM Python cannot load sqlite3 with its persistent Conda runtime: $VLLM_ENV_LIB" >&2
  exit 1
fi

RUN_ROOT="$(dirname "$PLAN")"
COORDINATOR_ID="${JOB_ID:-manual-$(sha256sum "$PLAN" | awk '{print substr($1,1,12)}')-$$}"
RUN_LOG_ROOT="$RUN_ROOT/h_locomo_logs/$COORDINATOR_ID"
RUN_LOG="$RUN_LOG_ROOT/worker.replica-$(printf '%03d-of-%03d' "$REPLICA_INDEX" "$REPLICA_COUNT").log"
mkdir -p "$RUN_LOG_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "H LoCoMo runner start replica=$REPLICA_INDEX/$REPLICA_COUNT job_id=${JOB_ID:-unset} plan=$PLAN"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable inside the RJob container" >&2
  exit 1
fi
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//' | sed '/^$/d')
if (( ${#GPU_NAMES[@]} != 8 )); then
  echo "LoCoMo RJob Replica must expose exactly 8 GPUs, found ${#GPU_NAMES[@]}" >&2
  exit 1
fi
for gpu_name in "${GPU_NAMES[@]}"; do
  [[ "$gpu_name" == *H200* ]] || { echo "Expected H200, found $gpu_name" >&2; exit 1; }
done

GPU_LIST="0,1,2,3,4,5,6,7"
echo "H200 preflight passed: replica=$REPLICA_INDEX/$REPLICA_COUNT gpus=$GPU_LIST names=${GPU_NAMES[*]}"

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_locomo_plan.py" \
  --plan "$PLAN" \
  --gpus "$GPU_LIST" \
  --env-file "$ENV_FILE" \
  --port-base "$PORT_BASE" \
  --replica-index "$REPLICA_INDEX" \
  --replica-count "$REPLICA_COUNT" \
  --coordinator-id "$COORDINATOR_ID" \
  --task-attempts "${HABITBENCH_LOCOMO_TASK_ATTEMPTS:-2}" \
  --coordination-root "$RUN_ROOT/locomo_queue" \
  --runtime-output "$RUN_ROOT/locomo_runtime.replica-$(printf '%03d-of-%03d' "$REPLICA_INDEX" "$REPLICA_COUNT").json" \
  --log-root "$RUN_ROOT/locomo_vllm_logs/$COORDINATOR_ID/replica-$(printf '%03d-of-%03d' "$REPLICA_INDEX" "$REPLICA_COUNT")"
