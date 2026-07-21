#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

METHODS="mem0"
DATASETS="food,finance_software"
SHARDS=8
MAX_CONCURRENT=8
OUTPUT_ROOT="$PROJECT_ROOT/results/full_scale"
PLAN=""
ENV_FILE="$PROJECT_ROOT/scripts/lumia/lumia_env_example.sh"
PARTITION=""
TIME_LIMIT="24:00:00"
CPUS=12
MEMORY="96G"
PORT_BASE="${HABITBENCH_PORT_BASE:-8100}"
DRY_RUN=0
NO_MERGE_JOB=0

usage() {
  echo "usage: scripts/slurm/submit_sharded_suite.sh [options]"
  echo "  --methods LIST          comma-separated; default mem0"
  echo "  --datasets LIST         default food,finance_software"
  echo "  --shards N              user shards per method/dataset; default 8"
  echo "  --max-concurrent N      maximum simultaneous GPU tasks; default 8"
  echo "  --output-root PATH      experiment output root"
  echo "  --plan PATH             plan path; defaults under output root"
  echo "  --env-file PATH         cluster environment file"
  echo "  --partition NAME        Slurm partition"
  echo "  --time HH:MM:SS         per-shard time limit"
  echo "  --cpus N                CPUs per GPU task"
  echo "  --mem SIZE              memory per GPU task"
  echo "  --port-base N           localhost port base for this array"
  echo "  --no-merge-job          do not submit dependent CPU merge job"
  echo "  --dry-run               create plan and print sbatch commands"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods) METHODS="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --plan) PLAN="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --port-base) PORT_BASE="$2"; shift 2 ;;
    --no-merge-job) NO_MERGE_JOB=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

OUTPUT_ROOT=$(realpath -m "$OUTPUT_ROOT")
PLAN="${PLAN:-$OUTPUT_ROOT/shard_plan.tsv}"
mkdir -p "$OUTPUT_ROOT/slurm"
if [[ -n "$ENV_FILE" ]]; then
  source "$ENV_FILE"
fi

"${PYTHON_BIN:-python}" scripts/create_shard_plan.py \
  --methods "$METHODS" \
  --datasets "$DATASETS" \
  --shards "$SHARDS" \
  --output-root "$OUTPUT_ROOT" \
  --plan "$PLAN" \
  --force

TASKS=$(( $(wc -l < "$PLAN") - 1 ))
LAST_TASK=$(( TASKS - 1 ))
SBATCH_ARGS=(
  --parsable
  --job-name habitbench-shards
  --array "0-${LAST_TASK}%${MAX_CONCURRENT}"
  --gres gpu:1
  --cpus-per-task "$CPUS"
  --mem "$MEMORY"
  --time "$TIME_LIMIT"
  --export "ALL,HABITBENCH_PORT_BASE=$PORT_BASE"
  --output "$OUTPUT_ROOT/slurm/%A_%a.out"
  --error "$OUTPUT_ROOT/slurm/%A_%a.err"
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition "$PARTITION")
fi

ARRAY_COMMAND=(sbatch "${SBATCH_ARGS[@]}" scripts/slurm/run_shard_array_task.sh "$PLAN" "$ENV_FILE")
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY RUN array: '; printf '%q ' "${ARRAY_COMMAND[@]}"; printf '\n'
  echo "DRY RUN merge: sbatch --dependency=afterok:<array_job_id> scripts/slurm/merge_shard_array.sh $PLAN $ENV_FILE"
  exit 0
fi

ARRAY_JOB_ID=$("${ARRAY_COMMAND[@]}")
echo "submitted_array job_id=$ARRAY_JOB_ID tasks=$TASKS plan=$PLAN"

if [[ "$NO_MERGE_JOB" != "1" ]]; then
  MERGE_JOB_ID=$(sbatch \
    --parsable \
    --dependency "afterok:$ARRAY_JOB_ID" \
    --job-name habitbench-merge \
    --cpus-per-task 4 \
    --mem 32G \
    --time 02:00:00 \
    --output "$OUTPUT_ROOT/slurm/%A_merge.out" \
    --error "$OUTPUT_ROOT/slurm/%A_merge.err" \
    scripts/slurm/merge_shard_array.sh "$PLAN" "$ENV_FILE")
  echo "submitted_merge job_id=$MERGE_JOB_ID dependency=afterok:$ARRAY_JOB_ID"
fi
