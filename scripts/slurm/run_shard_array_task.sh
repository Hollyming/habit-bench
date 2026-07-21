#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PLAN="${1:?usage: scripts/slurm/run_shard_array_task.sh PLAN [ENV_FILE]}"
ENV_FILE="${2:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

export HABITBENCH_VLLM_PORT="$(( ${HABITBENCH_PORT_BASE:-8100} + TASK_ID % 1000 ))"
bash "$PROJECT_ROOT/scripts/run_shard_plan_task.sh" "$PLAN" "$TASK_ID" "$ENV_FILE"
