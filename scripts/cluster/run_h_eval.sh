#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PLAN=""
GPUS=""
ENV_FILE=""
PORT_BASE=8100
CONTINUE_ON_GROUP_ERROR=0
POST_SUPPLEMENTARY_ANALYSIS=0
EXPECTED_REPLICAS=""

usage() {
  echo "Usage: scripts/cluster/run_h_eval.sh --plan PATH --gpus 4|8 --env-file PATH [options]"
  echo "  --port-base N"
  echo "  --expected-replicas N"
  echo "  --continue-on-group-error"
  echo "  --post-supplementary-analysis"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) PLAN="${2:?missing value for --plan}"; shift 2 ;;
    --gpus) GPUS="${2:?missing value for --gpus}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?missing value for --env-file}"; shift 2 ;;
    --port-base) PORT_BASE="${2:?missing value for --port-base}"; shift 2 ;;
    --expected-replicas) EXPECTED_REPLICAS="${2:?missing value for --expected-replicas}"; shift 2 ;;
    --continue-on-group-error) CONTINUE_ON_GROUP_ERROR=1; shift ;;
    --post-supplementary-analysis) POST_SUPPLEMENTARY_ANALYSIS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$GPUS" != "4" && "$GPUS" != "8" ]]; then
  echo "--gpus must be 4 or 8 for the H200 evaluator, got: ${GPUS:-unset}" >&2
  exit 2
fi
if [[ ! "$PORT_BASE" =~ ^[1-9][0-9]*$ ]] || (( PORT_BASE + GPUS > 65535 )); then
  echo "--port-base must leave room for every local vLLM worker" >&2
  exit 2
fi
for required in "$PLAN" "$ENV_FILE"; do
  if [[ ! -f "$required" ]]; then
    echo "Required H evaluation input does not exist: ${required:-unset}" >&2
    exit 1
  fi
done

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN must be set by the H environment file}"
VLLM_PYTHON="${HABITBENCH_VLLM_PYTHON:?HABITBENCH_VLLM_PYTHON must be set by the H environment file}"
REPLICA_INDEX="${NODE_RANK:-0}"
REPLICA_COUNT="${NODE_COUNT:-1}"
if ! [[ "$REPLICA_INDEX" =~ ^[0-9]+$ && "$REPLICA_COUNT" =~ ^[1-9][0-9]*$ ]] \
  || (( REPLICA_INDEX >= REPLICA_COUNT )); then
  echo "Invalid RJob NODE_RANK/NODE_COUNT: $REPLICA_INDEX/$REPLICA_COUNT" >&2
  exit 1
fi
if [[ -n "$EXPECTED_REPLICAS" && "$REPLICA_COUNT" != "$EXPECTED_REPLICAS" ]]; then
  echo "RJob exposed NODE_COUNT=$REPLICA_COUNT, expected $EXPECTED_REPLICAS" >&2
  exit 1
fi
REPLICA_TAG=$(printf '%03d-of-%03d' "$REPLICA_INDEX" "$REPLICA_COUNT")
RUN_ROOT="$(dirname "$PLAN")"
if [[ "$REPLICA_COUNT" != "1" && -z "${JOB_ID:-}" ]]; then
  echo "Multi-Replica H evaluation requires the RJob-injected JOB_ID" >&2
  exit 1
fi
COORDINATOR_ID="${JOB_ID:-manual-$(sha256sum "$PLAN" | awk '{print substr($1,1,12)}')-$$}"
COORDINATION_ROOT="$RUN_ROOT/distributed_queue"
RUNTIME_ROOT="$RUN_ROOT/replica_runtime/$COORDINATOR_ID"
RUN_LOG_ROOT="$RUN_ROOT/h_rjob_logs/$COORDINATOR_ID"
RUN_LOG="$RUN_LOG_ROOT/h_rjob_worker.replica-${REPLICA_TAG}.log"
mkdir -p "$RUNTIME_ROOT" "$RUN_LOG_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "H runner start: replica=$REPLICA_INDEX/$REPLICA_COUNT job_id=${JOB_ID:-unset} plan=$PLAN"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Method Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -x "$VLLM_PYTHON" ]]; then
  echo "vLLM Python is not executable: $VLLM_PYTHON" >&2
  exit 1
fi

# The H base image exports its system libstdc++ ahead of persistent Conda
# environments. Packages installed in those environments (notably ICU, loaded
# by sqlite/diskcache during vLLM startup) may require a newer CXXABI. Give each
# child process access to the matching Conda runtime, with the vLLM environment
# first because every local server is launched from it.
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
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable inside the RJob container" >&2
  exit 1
fi

mapfile -t GPU_NAMES < <(
  nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/[[:space:]]*$//' \
    | sed '/^$/d'
)
if (( ${#GPU_NAMES[@]} != GPUS )); then
  echo "RJob exposed ${#GPU_NAMES[@]} GPUs, but the plan requested $GPUS" >&2
  printf 'visible_gpu=%s\n' "${GPU_NAMES[@]}" >&2
  exit 1
fi
for gpu_name in "${GPU_NAMES[@]}"; do
  if [[ "$gpu_name" != *H200* ]]; then
    echo "H200 validation failed; visible GPU is: $gpu_name" >&2
    exit 1
  fi
done

GPU_LIST=""
for ((index = 0; index < GPUS; index++)); do
  [[ -z "$GPU_LIST" ]] || GPU_LIST+=","
  GPU_LIST+="$index"
done

echo "H200 preflight passed: replica=$REPLICA_INDEX/$REPLICA_COUNT gpus=$GPUS names=${GPU_NAMES[*]} plan=$PLAN"

RUN_ARGS=(
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_multigpu_plan.py"
  --plan "$PLAN"
  --gpus "$GPU_LIST"
  --env-file "$ENV_FILE"
  --port-base "$PORT_BASE"
  --replica-index "$REPLICA_INDEX"
  --replica-count "$REPLICA_COUNT"
  --coordination-root "$COORDINATION_ROOT"
  --coordinator-id "$COORDINATOR_ID"
  --runtime-output "$RUNTIME_ROOT/suite_runtime.replica-${REPLICA_TAG}.json"
  --log-root "$RUN_ROOT/vllm_logs/$COORDINATOR_ID/replica-${REPLICA_TAG}"
)
if [[ "$CONTINUE_ON_GROUP_ERROR" == "1" ]]; then
  RUN_ARGS+=(--continue-on-group-error)
fi

child_pid=""
terminated=0
forward_termination() {
  terminated=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
}
trap forward_termination TERM INT

"${RUN_ARGS[@]}" &
child_pid=$!
set +e
wait "$child_pid"
run_status=$?
set -e
child_pid=""
trap - TERM INT

if [[ "$terminated" == "1" ]]; then
  replica_status="interrupted"
  run_status=143
elif (( run_status == 0 )); then
  replica_status="succeeded"
else
  replica_status="failed"
fi

# One atomic terminal marker per Replica is the only coordinator checkpoint.
# It is written once after the local shard runner exits, not per session/probe.
STATUS_ROOT="$RUN_ROOT/replica_status/$COORDINATOR_ID"
STATUS_FILE="$STATUS_ROOT/replica-${REPLICA_TAG}.json"
mkdir -p "$STATUS_ROOT"
"$PYTHON_BIN" - "$STATUS_FILE" "$replica_status" "$run_status" "$REPLICA_INDEX" "$REPLICA_COUNT" "$RUN_LOG" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "returncode": int(sys.argv[3]),
    "replica_index": int(sys.argv[4]),
    "replica_count": int(sys.argv[5]),
    "worker_log": sys.argv[6],
    "finished_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
echo "replica_terminal status=$replica_status returncode=$run_status marker=$STATUS_FILE"

merge_status=0
terminal_count=$(find "$STATUS_ROOT" -maxdepth 1 -type f -name 'replica-*.json' | wc -l)
success_count=$(grep -l '"status": "succeeded"' "$STATUS_ROOT"/replica-*.json 2>/dev/null | wc -l || true)
if (( terminal_count == REPLICA_COUNT && success_count == REPLICA_COUNT )); then
  # GPFS atomic directory creation elects exactly one merge owner. Do not use
  # cross-node flock here: the task-queue incident demonstrated that it does
  # not provide a safe shared-state protocol in this runtime.
  if mkdir "$STATUS_ROOT/merge.claim" 2>/dev/null; then
    echo "all_replicas_succeeded replicas=$REPLICA_COUNT; starting global merge"
    set +e
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/merge_shard_plan.py" \
      --plan "$PLAN" \
      --replica-runtime-root "$RUNTIME_ROOT"
    merge_status=$?
    if (( merge_status == 0 )) && [[ "$POST_SUPPLEMENTARY_ANALYSIS" == "1" ]]; then
      SUPPLEMENTARY_SUITE_ROOT="$($PYTHON_BIN - "$PLAN" <<'PY'
import json
import sys
from pathlib import Path

plan = Path(sys.argv[1]).resolve()
manifest = json.loads(plan.with_suffix(".manifest.json").read_text(encoding="utf-8"))
print(Path(manifest["output_root"]).resolve())
PY
)"
      echo "global_merge_succeeded; starting non-human supplementary analysis suite=$SUPPLEMENTARY_SUITE_ROOT"
      "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_supplementary_analysis.py" \
        --suite-root "$SUPPLEMENTARY_SUITE_ROOT" \
        --output-root "$SUPPLEMENTARY_SUITE_ROOT/supplementary" \
        --bootstrap-samples 10000 \
        --seed 42
      merge_status=$?
    fi
    set -e
    if (( merge_status == 0 )); then
      "$PYTHON_BIN" - "$STATUS_ROOT/merge.succeeded.json" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
temporary.write_text(
    json.dumps({"status": "succeeded", "finished_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
    encoding="utf-8",
)
temporary.replace(path)
PY
      echo "global_merge_succeeded"
    else
      echo "global_merge_failed returncode=$merge_status" >&2
    fi
  elif [[ -f "$STATUS_ROOT/merge.succeeded.json" ]]; then
    echo "global_merge_already_succeeded"
  else
    echo "global_merge_claimed_by_other_replica"
  fi
else
  echo "replica_coordinator terminal=$terminal_count/$REPLICA_COUNT succeeded=$success_count/$REPLICA_COUNT; merge deferred to last successful Replica"
fi

if (( run_status != 0 )); then
  echo "H evaluation replica failed/interrupted; successful shard checkpoints remain reusable"
  exit "$run_status"
fi
if (( merge_status != 0 )); then
  exit "$merge_status"
fi
