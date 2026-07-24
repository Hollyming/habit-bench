#!/usr/bin/env bash
# Run one repository-standard HABIT-Bench adapter on the finalized v0.4 release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${HABITBENCH_V04_DATASET:-$ROOT/runs_wxq/taskmaster_planning_defaults_v0_4}"
METHOD="${1:?usage: run_v04.sh METHOD RUN_NAME [eval arguments]}"
RUN_NAME="${2:?usage: run_v04.sh METHOD RUN_NAME [eval arguments]}"
shift 2

case "$METHOD" in
  no_memory|full_history|mem0|amem|graphiti|secom|omem) ;;
  *) echo "Unsupported method: $METHOD" >&2; exit 2 ;;
esac

OUTPUT_ROOT="${HABITBENCH_V04_RESULTS:-$DATASET/evaluation_results}"
OUTPUT="$OUTPUT_ROOT/$RUN_NAME"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing run: $OUTPUT" >&2
  echo "Choose a new RUN_NAME or move the completed run deliberately." >&2
  exit 2
fi

python "$ROOT/scripts_wxq/evaluation/validate_v04.py" >/dev/null
exec bash "$ROOT/scripts/run_method.sh" "$METHOD" "$DATASET" "$OUTPUT" "$@"
