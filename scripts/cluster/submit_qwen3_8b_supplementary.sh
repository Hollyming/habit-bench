#!/usr/bin/env bash
set -euo pipefail

# Current four-domain Qwen3-8B supplementary evaluation on 2 x 8 H200.
# Human audit is intentionally excluded.  The regular main-suite sidecars are
# already offline analyses; this job supplies the missing No-Memory and two
# private-label Oracle controls, merges all 16 user shards, then computes the
# same non-human supplementary metrics for those controls.

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CREATOR_AD="${HABITBENCH_CREATOR_AD:?Set HABITBENCH_CREATOR_AD to the actual authenticated Group-AD}"
OUTPUT_ROOT="${HABITBENCH_SUPPLEMENTARY_OUTPUT_ROOT:-$PROJECT_ROOT/results/habit-h200-supplementary-qwen3-8b-v1}"
JOB_NAME="${HABITBENCH_SUPPLEMENTARY_JOB_NAME:-hb-q8b-supp-oracle-v16}"

exec bash "$PROJECT_ROOT/scripts/submit_h_cluster.sh" \
  --job-type reserved \
  --creator-type group \
  --creator-ad "$CREATOR_AD" \
  --gpus 8 \
  --replicas 2 \
  --shards 16 \
  --methods no_memory,oracle_evidence,oracle_habit_state \
  --datasets food,finance,software,travel \
  --output-root "$OUTPUT_ROOT" \
  --job-name "$JOB_NAME" \
  --post-supplementary-analysis \
  "$@"
