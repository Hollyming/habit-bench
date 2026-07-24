#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

METHOD="${1:?usage: scripts/run_user_shard.sh METHOD DATASET METHOD_OUTPUT_ROOT SHARD_INDEX SHARD_COUNT [eval args]}"
DATASET="${2:?usage: scripts/run_user_shard.sh METHOD DATASET METHOD_OUTPUT_ROOT SHARD_INDEX SHARD_COUNT [eval args]}"
METHOD_OUTPUT_ROOT="${3:?usage: scripts/run_user_shard.sh METHOD DATASET METHOD_OUTPUT_ROOT SHARD_INDEX SHARD_COUNT [eval args]}"
SHARD_INDEX="${4:?usage: scripts/run_user_shard.sh METHOD DATASET METHOD_OUTPUT_ROOT SHARD_INDEX SHARD_COUNT [eval args]}"
SHARD_COUNT="${5:?usage: scripts/run_user_shard.sh METHOD DATASET METHOD_OUTPUT_ROOT SHARD_INDEX SHARD_COUNT [eval args]}"
shift 5

printf -v SHARD_NAME 'shard_%03d_of_%03d' "$SHARD_INDEX" "$SHARD_COUNT"
OUTPUT_DIR="$METHOD_OUTPUT_ROOT/$SHARD_NAME"

if [[ "${HABITBENCH_FORCE_RERUN:-0}" != "1" && -f "$OUTPUT_DIR/metrics.json" ]]; then
  echo "skip_completed method=$METHOD shard=$SHARD_INDEX/$SHARD_COUNT output=$OUTPUT_DIR"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
bash "$PROJECT_ROOT/scripts/run_method.sh" \
  "$METHOD" \
  "$DATASET" \
  "$OUTPUT_DIR" \
  --user-shard-index "$SHARD_INDEX" \
  --user-shard-count "$SHARD_COUNT" \
  "$@"
