#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PLAN="${1:?usage: scripts/slurm/merge_shard_array.sh PLAN [ENV_FILE]}"
ENV_FILE="${2:-}"
if [[ -n "$ENV_FILE" ]]; then
  source "$ENV_FILE"
fi
cd "$PROJECT_ROOT"
"${PYTHON_BIN:-python}" scripts/merge_shard_plan.py --plan "$PLAN"
