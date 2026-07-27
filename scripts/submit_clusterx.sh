#!/usr/bin/env bash
set -euo pipefail

# Build a deterministic three-domain shard plan and submit one multi-GPU
# HABIT-Bench evaluation job through ClusterX.

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark/bin/python}"
CLUSTERX_BIN="${CLUSTERX_BIN:-/plm-shared/zhangjunming/miniconda3/envs/clusterx/bin/clusterx}"
ENV_FILE="${HABITBENCH_ENV_FILE:-$PROJECT_ROOT/scripts/cluster/env.example.sh}"
METHODS="mem0,amem,memos,memrl,lightmem,letta,mirix,graphiti,secom,omem"
DATASETS="food,finance,software"
SHARDS=8
GPUS=8
OUTPUT_ROOT=""
JOB_NAME=""
QUEUE="queue-t-reserved-plm"
CLUSTER="cluster-t"
MACHINE_TYPE="n3ls.ii.i60a"
IMAGE="registry.pjlab.org.cn/ccr-lepton-official-images/ngc-pytorch:26.04-cu13.2-py3.12-ubuntu24.04"
CPUS=""
MEMORY_GIB=""
SHM_GIB=""
PORT_BASE=8100
DRY_RUN=0
FORCE_PLAN=0
CONTINUE_ON_GROUP_ERROR=0

usage() {
  echo "Usage: scripts/submit_clusterx.sh [options]"
  echo "  --methods CSV       default: all ten memory methods; controls are explicit"
  echo "  --datasets CSV      default: food,finance,software"
  echo "  --shards N          user shards per method/domain, default: 8"
  echo "  --gpus N            GPUs on one ClusterX node, default: 8"
  echo "  --output-root PATH  default: results/bge_m3_<UTC timestamp>"
  echo "  --job-name NAME     default: habit-bge-m3-<UTC timestamp>"
  echo "  --continue-on-group-error"
  echo "                      record failed groups and keep running the remaining plan"
  echo "  --dry-run           create and print the submission without submitting"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods) METHODS="${2:?missing value for --methods}"; shift 2 ;;
    --datasets) DATASETS="${2:?missing value for --datasets}"; shift 2 ;;
    --shards) SHARDS="${2:?missing value for --shards}"; shift 2 ;;
    --gpus) GPUS="${2:?missing value for --gpus}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
    --job-name) JOB_NAME="${2:?missing value for --job-name}"; shift 2 ;;
    --queue) QUEUE="${2:?missing value for --queue}"; shift 2 ;;
    --cluster) CLUSTER="${2:?missing value for --cluster}"; shift 2 ;;
    --machine-type) MACHINE_TYPE="${2:?missing value for --machine-type}"; shift 2 ;;
    --image) IMAGE="${2:?missing value for --image}"; shift 2 ;;
    --cpus) CPUS="${2:?missing value for --cpus}"; shift 2 ;;
    --memory-gib) MEMORY_GIB="${2:?missing value for --memory-gib}"; shift 2 ;;
    --shm-gib) SHM_GIB="${2:?missing value for --shm-gib}"; shift 2 ;;
    --port-base) PORT_BASE="${2:?missing value for --port-base}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?missing value for --env-file}"; shift 2 ;;
    --force-plan) FORCE_PLAN=1; shift ;;
    --continue-on-group-error) CONTINUE_ON_GROUP_ERROR=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$SHARDS" =~ ^[1-9][0-9]*$ && "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--shards and --gpus must be positive integers" >&2
  exit 2
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/bge_m3_$STAMP}"
JOB_NAME="${JOB_NAME:-habit-bge-m3-$STAMP}"
if (( ${#JOB_NAME} > 32 )); then
  echo "--job-name must be at most 32 characters for ClusterX: $JOB_NAME" >&2
  exit 2
fi
CPUS="${CPUS:-$((GPUS * 8))}"
MEMORY_GIB="${MEMORY_GIB:-$((GPUS * 64))}"
SHM_GIB="${SHM_GIB:-$((GPUS * 8))}"
PLAN="$OUTPUT_ROOT/shard_plan.tsv"

for required in "$PYTHON_BIN" "$CLUSTERX_BIN" "$ENV_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path does not exist: $required" >&2
    exit 1
  fi
done
NEEDS_EMBED=0
IFS=',' read -r -a METHOD_LIST <<< "$METHODS"
for method in "${METHOD_LIST[@]}"; do
  case "$method" in
    mem0|amem|memos|memrl|lightmem|letta|mirix|graphiti|secom|omem)
      NEEDS_EMBED=1
      ;;
  esac
done
if [[ "$NEEDS_EMBED" == "1" && ! -f /plm-shared/zhangjunming/Workspace/models/bge-m3/config.json ]]; then
  echo "BGE-M3 is missing at /plm-shared/zhangjunming/Workspace/models/bge-m3" >&2
  exit 1
fi
if [[ ! -s /plm-shared/zhangjunming/.cache/tiktoken/fb374d419588a4632f3f557e76b4b70aebbca790 ]]; then
  echo "Offline o200k_base tiktoken cache is missing under /plm-shared/zhangjunming/.cache/tiktoken" >&2
  exit 1
fi

PLAN_ARGS=(
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/create_shard_plan.py"
  --methods "$METHODS"
  --datasets "$DATASETS"
  --shards "$SHARDS"
  --output-root "$OUTPUT_ROOT"
  --plan "$PLAN"
  --metadata "clusterx_job_name=$JOB_NAME"
  --metadata "cluster=$CLUSTER"
  --metadata "queue=$QUEUE"
  --metadata "machine_type=$MACHINE_TYPE"
  --metadata "image=$IMAGE"
  --metadata "gpus=$GPUS"
  --metadata "cpus=$CPUS"
  --metadata "memory_gib=$MEMORY_GIB"
  --metadata "shm_gib=$SHM_GIB"
)
if [[ "$FORCE_PLAN" == "1" ]]; then
  PLAN_ARGS+=(--force)
fi
"${PLAN_ARGS[@]}"

GPU_LIST=""
for ((index = 0; index < GPUS; index++)); do
  if [[ -n "$GPU_LIST" ]]; then
    GPU_LIST+=","
  fi
  GPU_LIST+="$index"
done

printf -v NODE_COMMAND \
  'set -euo pipefail; cd %q; %q %q --plan %q --gpus %q --env-file %q --port-base %q; %q %q --plan %q' \
  "$PROJECT_ROOT" \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_multigpu_plan.py" \
  "$PLAN" "$GPU_LIST" "$ENV_FILE" "$PORT_BASE" \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/merge_shard_plan.py" "$PLAN"
if [[ "$CONTINUE_ON_GROUP_ERROR" == "1" ]]; then
  printf -v NODE_COMMAND \
    'set -euo pipefail; cd %q; %q %q --plan %q --gpus %q --env-file %q --port-base %q --continue-on-group-error; %q %q --plan %q' \
    "$PROJECT_ROOT" \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_multigpu_plan.py" \
    "$PLAN" "$GPU_LIST" "$ENV_FILE" "$PORT_BASE" \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/merge_shard_plan.py" "$PLAN"
fi

CLUSTER_COMMAND=(
  "$CLUSTERX_BIN" run
  -J "$JOB_NAME"
  -q "$QUEUE"
  -C "$CLUSTER"
  --machine-type "$MACHINE_TYPE"
  --gpus-per-task "$GPUS"
  --cpus-per-task "$CPUS"
  --memory-per-task "$MEMORY_GIB"
  --shm-size-gib "$SHM_GIB"
  --no-env
  --image "$IMAGE"
  bash -lc "$NODE_COMMAND"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY RUN: '
  printf '%q ' env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy "${CLUSTER_COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  "${CLUSTER_COMMAND[@]}" 2>&1 | tee "$OUTPUT_ROOT/clusterx_submit.log"
