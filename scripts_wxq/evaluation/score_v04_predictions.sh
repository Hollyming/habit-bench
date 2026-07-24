#!/usr/bin/env bash
# Strictly score externally produced choice predictions against the v0.4 private key.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${HABITBENCH_V04_DATASET:-$ROOT/runs_wxq/taskmaster_planning_defaults_v0_4}"
METHOD="${1:?usage: score_v04_predictions.sh METHOD PREDICTIONS_JSONL RUN_NAME [score arguments]}"
PREDICTIONS="${2:?usage: score_v04_predictions.sh METHOD PREDICTIONS_JSONL RUN_NAME [score arguments]}"
RUN_NAME="${3:?usage: score_v04_predictions.sh METHOD PREDICTIONS_JSONL RUN_NAME [score arguments]}"
shift 3

if [[ ! -f "$PREDICTIONS" ]]; then
  echo "Prediction file not found: $PREDICTIONS" >&2
  exit 2
fi
OUTPUT_ROOT="${HABITBENCH_V04_RESULTS:-$DATASET/evaluation_results}"
OUTPUT="$OUTPUT_ROOT/$RUN_NAME"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing run: $OUTPUT" >&2
  exit 2
fi

python "$ROOT/scripts_wxq/evaluation/validate_v04.py" >/dev/null
exec python -m eval.score \
  --dataset-dir "$DATASET" \
  --predictions "$PREDICTIONS" \
  --output-dir "$OUTPUT" \
  --method-name "$METHOD" \
  "$@"
