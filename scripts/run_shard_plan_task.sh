#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

PLAN="${1:?usage: scripts/run_shard_plan_task.sh PLAN TASK_ID [ENV_FILE]}"
TASK_ID="${2:?usage: scripts/run_shard_plan_task.sh PLAN TASK_ID [ENV_FILE]}"
ENV_FILE="${3:-}"

if [[ -n "$ENV_FILE" ]]; then
  source "$ENV_FILE"
fi

ROW=$(awk -F '\t' -v task="$TASK_ID" 'NR > 1 && $1 == task {print; exit}' "$PLAN")
if [[ -z "$ROW" ]]; then
  echo "Task $TASK_ID was not found in $PLAN" >&2
  exit 2
fi

IFS=$'\t' read -r _ METHOD DATASET_NAME DATASET_DIR METHOD_OUTPUT_ROOT SHARD_INDEX SHARD_COUNT <<<"$ROW"
echo "start_task id=$TASK_ID method=$METHOD dataset=$DATASET_NAME shard=$SHARD_INDEX/$SHARD_COUNT"

bash "$PROJECT_ROOT/scripts/run_shard_with_server.sh" \
  "$METHOD" \
  "$DATASET_DIR" \
  "$METHOD_OUTPUT_ROOT" \
  "$SHARD_INDEX" \
  "$SHARD_COUNT"
