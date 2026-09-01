#!/usr/bin/env bash
set -euo pipefail

# Submit the food final four-scale Qwen3 run as one 4 x 8-H200 reserved RJob.
# Replicas 0..3 run Qwen3-4B/8B/14B/32B respectively.  Replica 1 then runs
# the Qwen3-8B non-human supplementary controls on the same node.

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SPEC_PATH="/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/H集群architecture分区RJob任务提交规范.md"
KUBEBRAIN_CLUSTER_ENTRY_REQUIRED="http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451"
IMAGE="${HABITBENCH_H_IMAGE:-registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20251117}"
FOOD_FINAL_DATASET="$PROJECT_ROOT/domain/food/food_habit_lifelines_final_check"
MAIN_ROOT_PREFIX="$PROJECT_ROOT/results/habit-h200-main-food-final-qwen3"
SUPPLEMENTARY_ROOT="$PROJECT_ROOT/results/habit-h200-supplementary-food-final-qwen3-8b-v1"
JOB_NAME="${HABITBENCH_FOOD_FINAL_JOB_NAME:-zjm-qwen3-food-final-4x8-v1}"
DRY_RUN="${HABITBENCH_FOOD_FINAL_DRY_RUN:-0}"
SHARDS=16
GPUS=8
REPLICAS=4
CPUS=64
MEMORY_MIB=524288
MAIN_METHODS="full_memory,recency_5,recency_10,bm25_rag,dense_rag,temporal_hybrid_rag,mem0,amem,memos,memrl,lightmem,letta,mirix,secom"
SUPPLEMENTARY_METHODS="no_memory,oracle_evidence,oracle_habit_state"
# This job is intentionally scoped to the newly released food final dataset.
# Other domains keep their own independent plans/jobs and are not re-run here.
DATASETS="food"

if [[ ! -r "$SPEC_PATH" ]]; then
  echo "Cannot read H llmarchitecture specification: $SPEC_PATH" >&2
  exit 1
fi
if [[ ! -d "$FOOD_FINAL_DATASET" ]]; then
  echo "Food final dataset does not exist: $FOOD_FINAL_DATASET" >&2
  exit 1
fi

# Resolve identity before plan creation and verify it again immediately before
# the state-changing rjob call.  P5 reserved is valid only for the Group-AD.
# shellcheck disable=SC1091
source /etc/profile.d/ssh-init.sh
export KUBEBRAIN_CLUSTER_ENTRY="$KUBEBRAIN_CLUSTER_ENTRY_REQUIRED"
ACTUAL_CREATOR_AD="${BRAIN_USERNAME:-}"
if [[ -z "$ACTUAL_CREATOR_AD" || "$ACTUAL_CREATOR_AD" == "brainpp" ]]; then
  FQDN="$(hostname -f)"
  IFS='.' read -r -a FQDN_PARTS <<< "$FQDN"
  (( ${#FQDN_PARTS[@]} > 1 )) && ACTUAL_CREATOR_AD="${FQDN_PARTS[1]}"
fi
CREATOR_AD="${HABITBENCH_CREATOR_AD:-$ACTUAL_CREATOR_AD}"
if [[ -z "$ACTUAL_CREATOR_AD" || "$CREATOR_AD" != "$ACTUAL_CREATOR_AD" ]]; then
  echo "HABITBENCH_CREATOR_AD must match the authenticated creator: $ACTUAL_CREATOR_AD" >&2
  exit 2
fi
if [[ "$CREATOR_AD" != "linzhouhan" ]]; then
  echo "Reserved P5 requires the llmarchitecture Group-AD linzhouhan; authenticated creator=$CREATOR_AD" >&2
  exit 2
fi

ENV_DIR="$PROJECT_ROOT/scripts/cluster"
PYTHON_BIN=""
set -a
# shellcheck disable=SC1091
source "$ENV_DIR/env.h.example.sh"
set +a
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN must be provided by env.h.example.sh}"
VLLM_PYTHON="${HABITBENCH_VLLM_PYTHON:?HABITBENCH_VLLM_PYTHON must be provided by env.h.example.sh}"
for required in "$PYTHON_BIN" "$VLLM_PYTHON" "$PROJECT_ROOT/scripts/create_shard_plan.py" "$PROJECT_ROOT/scripts/cluster/run_h_eval.sh" "$HABITBENCH_EMBED_MODEL" "$HABITBENCH_LIGHTMEM_MODEL" "$HABITBENCH_CHAT_TEMPLATE" "$TRITON_PTXAS_PATH"; do
  [[ -e "$required" ]] || { echo "Required H path is missing: $required" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" && -x "$VLLM_PYTHON" ]] || { echo "H Python environments are not executable" >&2; exit 1; }

MAIN_ROOTS=()
MODEL_SLUGS=(4b 8b 14b 32b)
MODEL_ENVS=(
  "$ENV_DIR/env.h.qwen3_4b.sh"
  "$ENV_DIR/env.h.example.sh"
  "$ENV_DIR/env.h.qwen3_14b.sh"
  "$ENV_DIR/env.h.qwen3_32b.sh"
)

create_or_validate_plan() {
  local slug="$1" env_file="$2" output_root="$3" methods="$4"
  local plan="$output_root/shard_plan.tsv"
  local model_py model_path served_model model_id model_revision embed lightmem secom
  mkdir -p "$output_root"
  # Model-specific profiles deliberately override the common Qwen3-8B
  # defaults.  Clear inherited values so sourcing env.h.example for the 8B
  # Replica cannot accidentally retain the preceding 4B/14B/32B profile.
  unset HABITBENCH_LLM_MODEL HABITBENCH_SERVED_MODEL HABITBENCH_LLM_MODEL_ID HABITBENCH_LLM_MODEL_REVISION
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
  model_py="$PYTHON_BIN"
  model_path="$HABITBENCH_LLM_MODEL"
  served_model="$HABITBENCH_SERVED_MODEL"
  model_id="$HABITBENCH_LLM_MODEL_ID"
  model_revision="$HABITBENCH_LLM_MODEL_REVISION"
  embed="$HABITBENCH_EMBED_MODEL"
  lightmem="$HABITBENCH_LIGHTMEM_MODEL"
  secom="$HABITBENCH_SECOM_COMPRESSOR"
  [[ -d "$model_path" ]] || { echo "Qwen3-$slug model path missing: $model_path" >&2; exit 1; }
  [[ -f "$model_path/config.json" && -f "$model_path/HABIT_MODEL_INFO.json" ]] || { echo "Incomplete Qwen3-$slug model snapshot: $model_path" >&2; exit 1; }
  if [[ -f "$plan" ]]; then
    echo "Reusing existing plan: $plan"
  else
    "$model_py" "$PROJECT_ROOT/scripts/create_shard_plan.py" \
      --methods "$methods" \
      --datasets "$DATASETS" \
      --dataset "food=$FOOD_FINAL_DATASET" \
      --shards "$SHARDS" \
      --embedding-model-path "$embed" \
      --llm-model-path "$model_path" \
      --served-model-name "$served_model" \
      --lightmem-model-path "$lightmem" \
      --secom-compressor-path "$secom" \
      --output-root "$output_root" \
      --plan "$plan" \
      --metadata "launcher=h-rjob-food-final-4x8" \
      --metadata "food_dataset=$FOOD_FINAL_DATASET" \
      --metadata "model_scale=$slug" \
      --metadata "llm_model=$model_path" \
      --metadata "llm_model_id=$model_id" \
      --metadata "llm_model_revision=$model_revision" \
      --metadata "served_model=$served_model" \
      --metadata "gpu_model=h200" \
      --metadata "gpus_per_replica=8" \
      --metadata "replicas=1" \
      --metadata "total_gpus=8" \
      --metadata "shards=$SHARDS" \
      --metadata "human_audit=excluded_for_supplementary_only"
  fi
  "$model_py" - "$plan" "$output_root" "$methods" "$DATASETS" "$SHARDS" "$FOOD_FINAL_DATASET" <<'PY'
import csv
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1]).resolve()
output_root = Path(sys.argv[2]).resolve()
expected_methods = [item for item in sys.argv[3].split(",") if item]
expected_datasets = [item for item in sys.argv[4].split(",") if item]
expected_shards = int(sys.argv[5])
food_final = str(Path(sys.argv[6]).resolve())
manifest_path = plan_path.with_suffix(".manifest.json")
if not manifest_path.is_file():
    raise SystemExit(f"plan manifest is missing: {manifest_path}")
with plan_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if not rows:
    raise SystemExit(f"plan is empty: {plan_path}")
methods = list(dict.fromkeys(row["method"] for row in rows))
datasets = list(dict.fromkeys(row["dataset_name"] for row in rows))
if methods != expected_methods:
    raise SystemExit(f"method mismatch: expected={expected_methods}, actual={methods}")
if datasets != expected_datasets:
    raise SystemExit(f"dataset mismatch: expected={expected_datasets}, actual={datasets}")
for row in rows:
    if int(row["shard_count"]) != expected_shards:
        raise SystemExit(f"shard-count mismatch in row: {row}")
    method_root = Path(row["method_output_root"]).resolve()
    method_root.relative_to(output_root)
    if row["dataset_name"] == "food" and str(Path(row["dataset_dir"]).resolve()) != food_final:
        raise SystemExit(f"food plan is not using final dataset: {row['dataset_dir']}")
expected_tasks = len(expected_methods) * len(expected_datasets) * expected_shards
if len(rows) != expected_tasks:
    raise SystemExit(f"task-count mismatch: expected={expected_tasks}, actual={len(rows)}")
print(f"validated plan={plan_path} tasks={len(rows)} food_final={food_final}")
PY
}

for index in "${!MODEL_SLUGS[@]}"; do
  slug="${MODEL_SLUGS[$index]}"
  root="$MAIN_ROOT_PREFIX-$slug-v1"
  MAIN_ROOTS+=("$root")
  create_or_validate_plan "$slug" "${MODEL_ENVS[$index]}" "$root" "$MAIN_METHODS"
done
create_or_validate_plan "8b-supp" "$ENV_DIR/env.h.example.sh" "$SUPPLEMENTARY_ROOT" "$SUPPLEMENTARY_METHODS"

# Verify every selected adapter import once on the submission host.  This is
# the same preflight used by the regular H launcher and catches broken vendor
# imports before reserving 32 H200s.
HABITBENCH_SELECTED_METHODS="$MAIN_METHODS,$SUPPLEMENTARY_METHODS" \
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from scripts.run_multigpu_plan import _preflight_method_imports

methods = {item.strip() for item in os.environ["HABITBENCH_SELECTED_METHODS"].split(",") if item.strip()}
print("method import preflight: " + json.dumps(
    _preflight_method_imports(Path(os.environ["PYTHON_BIN"]), dict(os.environ), methods),
    sort_keys=True,
))
PY

MOUNT_CONFIG="gpfs://gpfs2/plm-gpfs/jmzhang:/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang"
for required in "$PROJECT_ROOT" "$FOOD_FINAL_DATASET" "$MAIN_ROOT_PREFIX" "$SUPPLEMENTARY_ROOT" "$PYTHON_BIN" "$VLLM_PYTHON" "$HABITBENCH_EMBED_MODEL" "$HABITBENCH_LIGHTMEM_MODEL" "$HABITBENCH_CHAT_TEMPLATE" "$TRITON_PTXAS_PATH"; do
  [[ "$required" == /mnt/shared-storage-* ]] || { echo "Required path is outside persistent H storage: $required" >&2; exit 1; }
done

RUNTIME_SCRIPT="$PROJECT_ROOT/scripts/cluster/run_qwen3_all_food_final.sh"
[[ -x "$RUNTIME_SCRIPT" || -f "$RUNTIME_SCRIPT" ]] || { echo "Missing runtime launcher: $RUNTIME_SCRIPT" >&2; exit 1; }
RJOB_COMMAND=(
  rjob submit
  --name "$JOB_NAME"
  --namespace ailab-llmarchitecture
  --task-type normal
  --priority 5
  --charged-group llmarchitecture_gpu
  --private-machine group
  --gpu "$GPUS"
  --cpu "$CPUS"
  --memory "$MEMORY_MIB"
  -P "$REPLICAS"
  --image "$IMAGE"
  --image-pull-policy IfNotPresent
  --share-host-shm True
  --host-network=true
  -e DISTRIBUTED_JOB=true
  -e "HABITBENCH_PROJECT_ROOT=$PROJECT_ROOT"
  -e "HABITBENCH_FOOD_FINAL_DATASET=$FOOD_FINAL_DATASET"
  -e "HABITBENCH_FOOD_MAIN_ROOT_PREFIX=$MAIN_ROOT_PREFIX"
  -e "HABITBENCH_FOOD_SUPPLEMENTARY_ROOT=$SUPPLEMENTARY_ROOT"
  --mount "$MOUNT_CONFIG"
)
if [[ "$DRY_RUN" == "1" ]]; then
  RJOB_COMMAND+=(--dry-run True)
fi
RJOB_COMMAND+=(
  --
  bash -lc
  "exec $RUNTIME_SCRIPT"
)

echo "RJob type=reserved creator=$CREATOR_AD/group resources=8 H200 x 4 Replicas = 32 total GPUs; cpu=$CPUS memory_mib=$MEMORY_MIB per Replica"
echo "food final dataset=$FOOD_FINAL_DATASET"
printf 'RJob command: '; printf '%q ' "${RJOB_COMMAND[@]}"; printf '\n'

# The highest H specification and SSH route are checked immediately before the
# only state-changing operation, as required for the new scheduler entry.
LATEST_SPEC_SHA256="$(sha256sum "$SPEC_PATH" | awk '{print $1}')"
[[ -n "$LATEST_SPEC_SHA256" ]] || { echo "Unable to hash H specification" >&2; exit 1; }
sed -n '1,$p' "$SPEC_PATH" >/dev/null
# shellcheck disable=SC1091
source /etc/profile.d/ssh-init.sh
export KUBEBRAIN_CLUSTER_ENTRY="$KUBEBRAIN_CLUSTER_ENTRY_REQUIRED"
[[ "$KUBEBRAIN_CLUSTER_ENTRY" == "$KUBEBRAIN_CLUSTER_ENTRY_REQUIRED" ]] || { echo "Invalid llmarchitecture scheduler entry" >&2; exit 1; }
FINAL_CREATOR="${BRAIN_USERNAME:-}"
[[ "$FINAL_CREATOR" == "$CREATOR_AD" ]] || { echo "Authenticated creator changed before submit: $FINAL_CREATOR (expected $CREATOR_AD)" >&2; exit 1; }
"${RJOB_COMMAND[@]}"
