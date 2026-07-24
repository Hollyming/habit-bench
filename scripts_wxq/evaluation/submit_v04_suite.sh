#!/usr/bin/env bash
# Submit a resumable, user-sharded full v0.4 benchmark suite to Slurm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${HABITBENCH_V04_DATASET:-$ROOT/runs_wxq/taskmaster_planning_defaults_v0_4}"
METHODS="no_memory,full_history,mem0,amem,graphiti,secom,omem"
SHARDS=6
MAX_CONCURRENT=6
PARTITION="ADA6000"
TIME_LIMIT="24:00:00"
CPUS=12
MEMORY="96G"
PORT_BASE=8300
ENV_FILE="$ROOT/scripts_wxq/evaluation/v04_lumia_env.sh"
RUN_NAME="suite_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=0

usage() {
  echo "usage: submit_v04_suite.sh [options]"
  echo "  --methods LIST        comma-separated explicit methods"
  echo "  --shards N            user shards, 1..6 (default: 6)"
  echo "  --max-concurrent N    concurrent GPU tasks (default: 6)"
  echo "  --partition NAME      Slurm GPU partition (default: ADA6000)"
  echo "  --time HH:MM:SS       per-shard limit (default: 24:00:00)"
  echo "  --run-name NAME       output directory name"
  echo "  --env-file PATH       runtime environment file"
  echo "  --dry-run             write plan and print sbatch commands only"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods) METHODS="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then echo "Environment file not found: $ENV_FILE" >&2; exit 2; fi
if (( SHARDS < 1 || SHARDS > 6 )); then echo "--shards must be in 1..6" >&2; exit 2; fi
if (( MAX_CONCURRENT < 1 )); then echo "--max-concurrent must be positive" >&2; exit 2; fi

python "$ROOT/scripts_wxq/evaluation/validate_v04.py" >/dev/null
OUTPUT_ROOT="$DATASET/evaluation_results/$RUN_NAME"
PLAN="$OUTPUT_ROOT/shard_plan.tsv"
mkdir -p "$OUTPUT_ROOT/slurm"
python "$ROOT/scripts/create_shard_plan.py" \
  --methods "$METHODS" \
  --datasets taskmaster_v04 \
  --dataset "taskmaster_v04=$DATASET" \
  --shards "$SHARDS" \
  --output-root "$OUTPUT_ROOT" \
  --plan "$PLAN" \
  --force

TASKS=$(( $(wc -l < "$PLAN") - 1 ))
LAST_TASK=$(( TASKS - 1 ))
ARRAY=(sbatch --parsable --job-name habit-v04 --array "0-${LAST_TASK}%${MAX_CONCURRENT}" \
  --partition "$PARTITION" --gres gpu:1 --cpus-per-task "$CPUS" --mem "$MEMORY" --time "$TIME_LIMIT" \
  --export "ALL,HABITBENCH_PORT_BASE=$PORT_BASE" \
  --output "$OUTPUT_ROOT/slurm/%A_%a.out" --error "$OUTPUT_ROOT/slurm/%A_%a.err" \
  --wrap "bash '$ROOT/scripts/slurm/run_shard_array_task.sh' '$PLAN' '$ENV_FILE'")

if (( DRY_RUN )); then
  printf 'DRY RUN array: '; printf '%q ' "${ARRAY[@]}"; printf '\n'
  echo "DRY RUN merge: sbatch --dependency=afterok:<array_job_id> --wrap 'bash $ROOT/scripts/slurm/merge_shard_array.sh $PLAN $ENV_FILE'"
  exit 0
fi

ARRAY_JOB_ID="$("${ARRAY[@]}")"
MERGE_JOB_ID="$(sbatch --parsable --dependency "afterok:$ARRAY_JOB_ID" --job-name habit-v04-merge \
  --cpus-per-task 4 --mem 32G --time 02:00:00 \
  --output "$OUTPUT_ROOT/slurm/%A_merge.out" --error "$OUTPUT_ROOT/slurm/%A_merge.err" \
  --wrap "bash '$ROOT/scripts/slurm/merge_shard_array.sh' '$PLAN' '$ENV_FILE'")"
printf 'submitted suite=%s array_job=%s tasks=%s merge_job=%s plan=%s\n' "$RUN_NAME" "$ARRAY_JOB_ID" "$TASKS" "$MERGE_JOB_ID" "$PLAN"
