#!/usr/bin/env bash
set -euo pipefail

# Run all currently implemented full-path official scaffolds on the official
# subset. This assumes the OpenAI-compatible endpoint is already running.

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

DATA="${1:-${HABITBENCH_DATASET:-./runs/habit_bench_balanced_v0_3_official_subset_90}}"
OUT="${2:-${HABITBENCH_OFFICIAL_OUT:-$DATA/full_official_results}}"
CHECK_ENDPOINT="${HABITBENCH_CHECK_ENDPOINT:-1}"
RUN_ID="${HABITBENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MANIFEST_DIR="${HABITBENCH_RUN_MANIFEST_DIR:-$OUT/run_manifests/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUT"
mkdir -p "$MANIFEST_DIR"
FINALIZED=0

finalize() {
  local exit_code="$1"
  if [[ "$FINALIZED" == "1" ]]; then
    return
  fi
  FINALIZED=1
  "$PYTHON_BIN" ./scripts/lumia/write_run_manifest.py \
    --out "$MANIFEST_DIR/suite_end_manifest.json" \
    --stage "suite_end" \
    --dataset-dir "$DATA" \
    --results-dir "$OUT" \
    --command "bash ./scripts/run_full_official_subset_suite.sh $DATA $OUT" \
    --extra "run_id=$RUN_ID" \
    --extra "exit_code=$exit_code" \
    --note "After running full official subset suite. Nonzero exit_code means the suite did not complete successfully." \
    >/dev/null 2>"$MANIFEST_DIR/suite_end_manifest.stderr" || true
}

on_exit() {
  local exit_code="$?"
  finalize "$exit_code"
  exit "$exit_code"
}
trap on_exit EXIT

"$PYTHON_BIN" ./scripts/lumia/write_run_manifest.py \
  --out "$MANIFEST_DIR/suite_start_manifest.json" \
  --stage "suite_start" \
  --dataset-dir "$DATA" \
  --results-dir "$OUT" \
  --command "bash ./scripts/run_full_official_subset_suite.sh $DATA $OUT" \
  --extra "run_id=$RUN_ID" \
  --note "Before running full official subset suite."

if [[ "$CHECK_ENDPOINT" == "1" ]]; then
  "$PYTHON_BIN" ./scripts/lumia/check_openai_endpoint.py \
    | tee "$MANIFEST_DIR/openai_endpoint_check.json"
fi

bash ./scripts/run_full_official_subset_mem0.sh "$DATA" "$OUT"
bash ./scripts/run_full_official_subset_graphiti.sh "$DATA" "$OUT"

"$PYTHON_BIN" ./eval/collect_official_results.py \
  --dataset-dir "$DATA" \
  --results-dir "$OUT"

"$PYTHON_BIN" ./eval/audit_full_official_results.py \
  --dataset-dir "$DATA" \
  --results-dir "$OUT"

finalize 0

echo "Full official subset suite finished. Results: $OUT"
echo "Run manifests: $MANIFEST_DIR"
