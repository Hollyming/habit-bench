#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_ROOT="${HABITBENCH_RESULTS_ROOT:-$PROJECT_ROOT/results}"
OUTPUT_ROOT="${HABITBENCH_Q14_OUTPUT_ROOT:-$RESULTS_ROOT/habit-h200-main-qwen3-14b-v1}"
JOB_NAME="${HABITBENCH_Q14_JOB_NAME:-zjm-main-q14b-2x8-v1}"

exec bash "$PROJECT_ROOT/scripts/submit_h_cluster.sh" \
  --job-type reserved \
  --creator-type group \
  --creator-ad linzhouhan \
  --gpus 8 \
  --replicas 2 \
  --shards 16 \
  --methods full_memory,mem0,amem,memos,memrl,lightmem,letta,mirix,secom \
  --datasets food,finance,software,travel \
  --output-root "$OUTPUT_ROOT" \
  --job-name "$JOB_NAME" \
  --env-file "$PROJECT_ROOT/scripts/cluster/env.h.qwen3_14b.sh" \
  "$@"
