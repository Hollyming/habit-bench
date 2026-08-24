#!/usr/bin/env bash
set -euo pipefail

# Submit 4- or 8-H200-per-Replica HABIT-Bench evaluation to the H cluster.
# Resource values are per Replica; shards use a cross-Replica dynamic queue.

LAUNCHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC_PATH="/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/H集群architecture分区RJob任务提交规范.md"
KUBEBRAIN_CLUSTER_ENTRY_REQUIRED="http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451"
# New llmarchitecture jobs must go through the partition scheduler rather than
# the legacy platform entry.  Pin it here so every wrapper using this launcher
# has the same routing contract before its first rjob invocation.
export KUBEBRAIN_CLUSTER_ENTRY="$KUBEBRAIN_CLUSTER_ENTRY_REQUIRED"
ENV_FILE="${HABITBENCH_ENV_FILE:-$LAUNCHER_ROOT/scripts/cluster/env.h.example.sh}"
METHODS="full_memory,mem0,amem,memos,memrl,lightmem,letta,mirix,secom"
DATASETS="food,finance,software,travel"
GPUS=8
REPLICAS=1
SHARDS=""
CPUS=""
MEMORY_MIB=""
OUTPUT_ROOT=""
JOB_NAME=""
JOB_TYPE="managed-spot"
CREATOR_TYPE="${HABITBENCH_CREATOR_TYPE:-}"
CREATOR_AD="${HABITBENCH_CREATOR_AD:-}"
IMAGE="${HABITBENCH_H_IMAGE:-registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20251117}"
MOUNT_CONFIGS=()
MOUNTS_FROM_CLI=0
PORT_BASE=8100
MAX_USERS=""
MAX_PROBES=""
DRY_RUN=0
FORCE_PLAN=0
CONTINUE_ON_GROUP_ERROR=0

usage() {
  echo "Usage: scripts/submit_h_cluster.sh [options]"
  echo "  --job-type TYPE      managed-spot (default), reserved, or idle"
  echo "  --creator-type TYPE  actual authenticated identity: user or group (required)"
  echo "  --creator-ad AD      actual authenticated AD/account name (required)"
  echo "  --gpus N             4 or 8 H200 GPUs per Replica, default: 8"
  echo "  --replicas N         independent RJob Replicas, default: 1"
  echo "  --shards N           user shards per method/domain, default: GPU x Replicas"
  echo "  --methods CSV        default: compact full_memory plus eight memory methods"
  echo "  --datasets CSV       default: food v5, finance/software v1.4, travel v13"
  echo "  --output-root PATH   default: PROJECT_ROOT/results/h-...; must be persistent H storage"
  echo "  --job-name NAME      lowercase letters, digits, hyphens; at most 32 chars"
  echo "  --env-file PATH      default: scripts/cluster/env.h.example.sh"
  echo "  --image IMAGE        explicit H registry image"
  echo "  --mount CONFIG       explicit GPFS mount; repeat for multiple user/shared roots"
  echo "  --cpus N             per Replica; minimum/default: 8 per GPU"
  echo "  --memory-mib N       per Replica; minimum/default: 65536 MiB per GPU"
  echo "  --max-users N        recorded smoke-test user prefix"
  echo "  --max-probes N       recorded smoke-test probe prefix"
  echo "  --continue-on-group-error"
  echo "  --force-plan         replace an existing shard plan intentionally"
  echo "  --dry-run            ask rjob to render only; no RJob is created"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-type) JOB_TYPE="${2:?missing value for --job-type}"; shift 2 ;;
    --creator-type) CREATOR_TYPE="${2:?missing value for --creator-type}"; shift 2 ;;
    --creator-ad) CREATOR_AD="${2:?missing value for --creator-ad}"; shift 2 ;;
    --gpus) GPUS="${2:?missing value for --gpus}"; shift 2 ;;
    --replicas) REPLICAS="${2:?missing value for --replicas}"; shift 2 ;;
    --shards) SHARDS="${2:?missing value for --shards}"; shift 2 ;;
    --methods) METHODS="${2:?missing value for --methods}"; shift 2 ;;
    --datasets) DATASETS="${2:?missing value for --datasets}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
    --job-name) JOB_NAME="${2:?missing value for --job-name}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?missing value for --env-file}"; shift 2 ;;
    --image) IMAGE="${2:?missing value for --image}"; shift 2 ;;
    --mount)
      MOUNT_CONFIGS+=("${2:?missing value for --mount}")
      MOUNTS_FROM_CLI=1
      shift 2
      ;;
    --cpus) CPUS="${2:?missing value for --cpus}"; shift 2 ;;
    --memory-mib) MEMORY_MIB="${2:?missing value for --memory-mib}"; shift 2 ;;
    --port-base) PORT_BASE="${2:?missing value for --port-base}"; shift 2 ;;
    --max-users) MAX_USERS="${2:?missing value for --max-users}"; shift 2 ;;
    --max-probes) MAX_PROBES="${2:?missing value for --max-probes}"; shift 2 ;;
    --continue-on-group-error) CONTINUE_ON_GROUP_ERROR=1; shift ;;
    --force-plan) FORCE_PLAN=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$GPUS" != "4" && "$GPUS" != "8" ]]; then
  echo "--gpus must be 4 or 8 for the H200 evaluator, got: $GPUS" >&2
  exit 2
fi
if ! [[ "$REPLICAS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--replicas must be a positive integer, got: $REPLICAS" >&2
  exit 2
fi
SHARDS="${SHARDS:-$((GPUS * REPLICAS))}"
CPUS="${CPUS:-$((GPUS * 8))}"
MEMORY_MIB="${MEMORY_MIB:-$((GPUS * 65536))}"
for value_name in REPLICAS SHARDS CPUS MEMORY_MIB PORT_BASE; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  fi
done
if (( CPUS < GPUS * 8 )); then
  echo "--cpus must be at least 8 cores per requested H200" >&2
  exit 2
fi
if (( MEMORY_MIB < GPUS * 65536 )); then
  echo "--memory-mib must be at least 65536 MiB per requested H200" >&2
  exit 2
fi
for smoke_value in "$MAX_USERS" "$MAX_PROBES"; do
  if [[ -n "$smoke_value" && ! "$smoke_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-users and --max-probes must be positive integers" >&2
    exit 2
  fi
done
case "$JOB_TYPE" in
  managed-spot)
    [[ "$CREATOR_TYPE" == "user" ]] || {
      echo "managed-spot requires the actual personal User-AD identity: --creator-type user" >&2
      exit 2
    }
    ;;
  reserved)
    [[ "$CREATOR_TYPE" == "group" ]] || {
      echo "reserved requires the team's actual Group-AD identity: --creator-type group" >&2
      exit 2
    }
    ;;
  idle)
    [[ "$CREATOR_TYPE" == "user" || "$CREATOR_TYPE" == "group" ]] || {
      echo "idle still requires an explicit actual identity: --creator-type user|group" >&2
      exit 2
    }
    ;;
  *) echo "--job-type must be managed-spot, reserved, or idle" >&2; exit 2 ;;
esac
if [[ -z "$CREATOR_AD" || ! "$CREATOR_AD" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--creator-ad must name the actual authenticated AD/account" >&2
  exit 2
fi

# Resolve the authenticated platform creator from the same environment used by
# brainpp. Group workspaces may expose BRAIN_USERNAME=brainpp, in which case
# BrainPP itself derives the effective user from the workspace FQDN.
# shellcheck disable=SC1091
source /etc/profile.d/ssh-init.sh
ACTUAL_CREATOR_AD="${BRAIN_USERNAME:-}"
if [[ -z "$ACTUAL_CREATOR_AD" || "$ACTUAL_CREATOR_AD" == "brainpp" ]]; then
  WORKSPACE_FQDN="$(hostname -f)"
  IFS='.' read -r -a FQDN_PARTS <<< "$WORKSPACE_FQDN"
  if (( ${#FQDN_PARTS[@]} > 1 )); then
    ACTUAL_CREATOR_AD="${FQDN_PARTS[1]}"
  fi
fi
if [[ -z "$ACTUAL_CREATOR_AD" || "$ACTUAL_CREATOR_AD" == "brainpp" ]]; then
  echo "Cannot resolve the actual RJob creator from the authenticated workspace" >&2
  exit 1
fi
if [[ "$CREATOR_AD" != "$ACTUAL_CREATOR_AD" ]]; then
  echo "--creator-ad=$CREATOR_AD does not match the authenticated RJob creator: $ACTUAL_CREATOR_AD" >&2
  exit 2
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "H environment file does not exist: $ENV_FILE" >&2
  exit 1
fi
ENV_FILE="$(realpath "$ENV_FILE")"
if [[ ! -r "$SPEC_PATH" ]]; then
  echo "Cannot read the H llmarchitecture specification: $SPEC_PATH" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$LAUNCHER_ROOT}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN must be set by the H environment file}"
for required_name in \
  HABITBENCH_VLLM_PYTHON \
  HABITBENCH_LLM_MODEL \
  HABITBENCH_SERVED_MODEL \
  HABITBENCH_LLM_MODEL_ID \
  HABITBENCH_LLM_MODEL_REVISION \
  HABITBENCH_EMBED_MODEL \
  HABITBENCH_LIGHTMEM_MODEL \
  HABITBENCH_SECOM_COMPRESSOR \
  HABITBENCH_CHAT_TEMPLATE \
  HF_HOME \
  XDG_CACHE_HOME \
  VLLM_CACHE_ROOT \
  TORCH_HOME \
  TORCHINDUCTOR_CACHE_DIR \
  TRITON_CACHE_DIR \
  TIKTOKEN_CACHE_DIR \
  TRITON_PTXAS_PATH
do
  if [[ ! -v "$required_name" || -z "${!required_name}" ]]; then
    echo "$required_name must be set by the H environment file" >&2
    exit 2
  fi
done
for field_name in IMAGE PROJECT_ROOT ENV_FILE; do
  field_value="${!field_name}"
  if [[ -z "$field_value" || "$field_value" == *'<'* || "$field_value" == *'>'* ]]; then
    echo "$field_name is empty or still contains a placeholder: $field_value" >&2
    exit 2
  fi
done
if ! command -v rjob >/dev/null 2>&1; then
  echo "rjob CLI is unavailable; install or activate brainpp before submission" >&2
  exit 1
fi

STAMP="$(date -u +%m%d-%H%M%S)"
JOB_NAME="${JOB_NAME:-habit-h200-${GPUS}g-$STAMP}"
if (( ${#JOB_NAME} > 32 )) || [[ ! "$JOB_NAME" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]]; then
  echo "--job-name must be at most 32 lowercase letters/digits/hyphens: $JOB_NAME" >&2
  exit 2
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/h-${JOB_TYPE}-${GPUS}g-$STAMP}"
if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$PROJECT_ROOT/$OUTPUT_ROOT"
fi
if [[ "$OUTPUT_ROOT" != /mnt/shared-storage-* ]]; then
  echo "--output-root must be persistent H storage under /mnt/shared-storage-*: $OUTPUT_ROOT" >&2
  exit 2
fi
PLAN="$OUTPUT_ROOT/shard_plan.tsv"

# The immutable environments/models may live in a shared owner's tree while
# the clone, writable caches and results belong to the actual evaluator.  The
# RJob CLI accepts multiple configs after one --mount.  If the caller did not
# specify mounts, infer the narrow per-owner GPFS2 roots needed by every path.
if (( MOUNTS_FROM_CLI == 0 )); then
  if [[ -n "${HABITBENCH_H_MOUNTS:-}" ]]; then
    # Mount configs cannot contain whitespace; split the documented list.
    read -r -a MOUNT_CONFIGS <<< "$HABITBENCH_H_MOUNTS"
  elif [[ -n "${HABITBENCH_H_MOUNT:-}" ]]; then
    MOUNT_CONFIGS=("$HABITBENCH_H_MOUNT")
  fi
fi

REQUIRED_MOUNT_PATHS=(
  "$PROJECT_ROOT"
  "$ENV_FILE"
  "$PYTHON_BIN"
  "$HABITBENCH_VLLM_PYTHON"
  "$HABITBENCH_LLM_MODEL"
  "$HABITBENCH_EMBED_MODEL"
  "$HABITBENCH_LIGHTMEM_MODEL"
  "$HABITBENCH_SECOM_COMPRESSOR"
  "$TRITON_PTXAS_PATH"
  "$HABITBENCH_CHAT_TEMPLATE"
  "$TIKTOKEN_CACHE_DIR"
  "$HF_HOME"
  "$XDG_CACHE_HOME"
  "$VLLM_CACHE_ROOT"
  "$TORCH_HOME"
  "$TORCHINDUCTOR_CACHE_DIR"
  "$TRITON_CACHE_DIR"
  "$OUTPUT_ROOT"
)

append_unique_mount() {
  local candidate="$1"
  local existing
  for existing in "${MOUNT_CONFIGS[@]}"; do
    [[ "$existing" == "$candidate" ]] && return 0
  done
  MOUNT_CONFIGS+=("$candidate")
}

if (( ${#MOUNT_CONFIGS[@]} == 0 )); then
  GPFS2_PLM_TARGET_PREFIX="/mnt/shared-storage-gpfs2/plm-gpfs"
  for mounted_path in "${REQUIRED_MOUNT_PATHS[@]}"; do
    if [[ "$mounted_path" != "$GPFS2_PLM_TARGET_PREFIX/"* ]]; then
      echo "Cannot infer an H mount for required path: $mounted_path" >&2
      echo "Pass one or more --mount configs explicitly" >&2
      exit 2
    fi
    owner_relative="${mounted_path#"$GPFS2_PLM_TARGET_PREFIX/"}"
    owner_name="${owner_relative%%/*}"
    if [[ -z "$owner_name" ]]; then
      echo "Cannot infer the GPFS owner root for: $mounted_path" >&2
      exit 2
    fi
    append_unique_mount \
      "gpfs://gpfs2/plm-gpfs/$owner_name:$GPFS2_PLM_TARGET_PREFIX/$owner_name"
  done
fi

MOUNT_TARGETS=()
for mount_config in "${MOUNT_CONFIGS[@]}"; do
  if [[ ! "$mount_config" =~ ^gpfs://[^:]+:/mnt/shared-storage- ]]; then
    echo "Invalid GPFS-to-H persistent mount config: $mount_config" >&2
    exit 2
  fi
  MOUNT_TARGETS+=("${mount_config##*:}")
done

path_is_mounted() {
  local candidate="$1"
  local target
  for target in "${MOUNT_TARGETS[@]}"; do
    if [[ "$candidate" == "$target" || "$candidate" == "$target/"* ]]; then
      return 0
    fi
  done
  return 1
}

for mounted_path in "${REQUIRED_MOUNT_PATHS[@]}"; do
  if ! path_is_mounted "$mounted_path"; then
    echo "No RJob mount target covers required path: $mounted_path" >&2
    printf 'mount_target=%s\n' "${MOUNT_TARGETS[@]}" >&2
    exit 2
  fi
done

MOUNT_METADATA="$(IFS=' '; echo "${MOUNT_CONFIGS[*]}")"

# These locations are intentionally per clone/user and writable.  Tiktoken is
# excluded because it is a verified immutable cache in the shared asset tree.
mkdir -p \
  "$OUTPUT_ROOT" \
  "$HF_HOME" \
  "$XDG_CACHE_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR"

for required in \
  "$PYTHON_BIN" \
  "$HABITBENCH_VLLM_PYTHON" \
  "$HABITBENCH_LLM_MODEL/config.json" \
  "$HABITBENCH_LLM_MODEL/HABIT_MODEL_INFO.json" \
  "$TIKTOKEN_CACHE_DIR/fb374d419588a4632f3f557e76b4b70aebbca790" \
  "$PROJECT_ROOT/scripts/create_shard_plan.py" \
  "$PROJECT_ROOT/scripts/cluster/run_h_eval.sh"
do
  if [[ ! -e "$required" ]]; then
    echo "Required H evaluation path does not exist: $required" >&2
    exit 1
  fi
done
for python_path in "$PYTHON_BIN" "$HABITBENCH_VLLM_PYTHON"; do
  if [[ ! -x "$python_path" ]]; then
    echo "Required H Python is not executable: $python_path" >&2
    exit 1
  fi
done
if [[ ! -x "$TRITON_PTXAS_PATH" ]]; then
  echo "Pinned Triton ptxas is not executable: $TRITON_PTXAS_PATH" >&2
  exit 1
fi
if [[ ! -f "$HABITBENCH_CHAT_TEMPLATE" ]]; then
  echo "Qwen chat template does not exist: $HABITBENCH_CHAT_TEMPLATE" >&2
  exit 1
fi
if ! find "$HABITBENCH_LLM_MODEL" -maxdepth 1 -type f \
  \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) \
  -print -quit | grep -q .
then
  echo "Qwen checkpoint has no local weight files: $HABITBENCH_LLM_MODEL" >&2
  exit 1
fi
if [[ ! -f "$HABITBENCH_LLM_MODEL/tokenizer.json" && ! -f "$HABITBENCH_LLM_MODEL/tokenizer.model" ]]; then
  echo "Qwen checkpoint has no local tokenizer asset: $HABITBENCH_LLM_MODEL" >&2
  exit 1
fi

NEEDS_EMBED=0
NEEDS_LIGHTMEM=0
NEEDS_SECOM=0
NEEDS_GPT2_CACHE=0
NEEDS_CL100K_CACHE=0
IFS=',' read -r -a METHOD_LIST <<< "$METHODS"
for method in "${METHOD_LIST[@]}"; do
  case "$method" in
    mem0|amem|memos|memrl|lightmem|letta|mirix|secom) NEEDS_EMBED=1 ;;
  esac
  case "$method" in
    lightmem) NEEDS_LIGHTMEM=1 ;;
    secom) NEEDS_SECOM=1 ;;
    memrl) NEEDS_GPT2_CACHE=1 ;;
    letta) NEEDS_CL100K_CACHE=1 ;;
  esac
done
if [[ "$NEEDS_EMBED" == "1" ]]; then
  for required in \
    "$HABITBENCH_EMBED_MODEL/config.json" \
    "$HABITBENCH_EMBED_MODEL/HABIT_MODEL_INFO.json" \
    "$HABITBENCH_EMBED_MODEL/pytorch_model.bin"
  do
    [[ -f "$required" ]] || { echo "Incomplete H BGE-M3 snapshot: $required" >&2; exit 1; }
  done
fi
if [[ "$NEEDS_LIGHTMEM" == "1" ]]; then
  for required in \
    "$HABITBENCH_LIGHTMEM_MODEL/config.json" \
    "$HABITBENCH_LIGHTMEM_MODEL/HABIT_MODEL_INFO.json" \
    "$HABITBENCH_LIGHTMEM_MODEL/model.safetensors"
  do
    [[ -f "$required" ]] || { echo "Incomplete H LLMLingua2 snapshot: $required" >&2; exit 1; }
  done
fi
if [[ "$NEEDS_SECOM" == "1" ]]; then
  for required in \
    "$HABITBENCH_SECOM_COMPRESSOR/config.json" \
    "$HABITBENCH_SECOM_COMPRESSOR/HABIT_MODEL_INFO.json" \
    "$HABITBENCH_SECOM_COMPRESSOR/model.safetensors"
  do
    [[ -f "$required" ]] || { echo "Incomplete H SeCom compressor snapshot: $required" >&2; exit 1; }
  done
fi
if [[ "$NEEDS_GPT2_CACHE" == "1" ]]; then
  for cache_key in \
    6d1cbeee0f20b3d9449abfede4726ed8212e3aee \
    6c7ea1a7e38e3a7f062df639a5b80947f075ffe6
  do
    [[ -s "$TIKTOKEN_CACHE_DIR/$cache_key" ]] || {
      echo "MemRL GPT-2 tokenizer cache is missing: $TIKTOKEN_CACHE_DIR/$cache_key" >&2
      exit 1
    }
  done
fi
if [[ "$NEEDS_CL100K_CACHE" == "1" ]]; then
  CL100K_CACHE_KEY=9b5ad71b2ce5302211f9c61530b329a4922fc6a4
  if [[ ! -s "$TIKTOKEN_CACHE_DIR/$CL100K_CACHE_KEY" ]]; then
    echo "Letta cl100k_base tokenizer cache is missing: $TIKTOKEN_CACHE_DIR/$CL100K_CACHE_KEY" >&2
    exit 1
  fi
fi

IDENTITY_ARGS=(
  qwen "$HABITBENCH_LLM_MODEL/HABIT_MODEL_INFO.json"
  "$HABITBENCH_LLM_MODEL_ID" "$HABITBENCH_LLM_MODEL_REVISION"
)
if [[ "$NEEDS_EMBED" == "1" ]]; then
  IDENTITY_ARGS+=(
    bge "$HABITBENCH_EMBED_MODEL/HABIT_MODEL_INFO.json"
    BAAI/bge-m3 5617a9f61b028005a4858fdac845db406aefb181
  )
fi
if [[ "$NEEDS_LIGHTMEM" == "1" ]]; then
  IDENTITY_ARGS+=(
    lightmem "$HABITBENCH_LIGHTMEM_MODEL/HABIT_MODEL_INFO.json"
    microsoft/llmlingua-2-xlm-roberta-large-meetingbank
    ebaba9b0e874dadd3003ffcff828e4397e568089
  )
fi
if [[ "$NEEDS_SECOM" == "1" ]]; then
  IDENTITY_ARGS+=(
    secom "$HABITBENCH_SECOM_COMPRESSOR/HABIT_MODEL_INFO.json"
    microsoft/llmlingua-2-xlm-roberta-large-meetingbank
    ebaba9b0e874dadd3003ffcff828e4397e568089
  )
fi
"$PYTHON_BIN" - "${IDENTITY_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

items = sys.argv[1:]
if len(items) % 4:
    raise SystemExit("invalid model identity preflight arguments")
for offset in range(0, len(items), 4):
    name, raw_path, expected_id, expected_revision = items[offset : offset + 4]
    path = Path(raw_path)
    marker = json.loads(path.read_text(encoding="utf-8"))
    actual = (marker.get("model_id"), marker.get("revision"))
    expected = (expected_id, expected_revision)
    if actual != expected:
        raise SystemExit(
            f"{name} identity mismatch at {path}: expected={expected}, actual={actual}"
        )
    print(f"model identity ready: {name}={expected_id}@{expected_revision}")
PY

# Exercise every selected adapter in isolated method-runtime subprocesses before
# requesting H200s. MemOS/MemRL traverse their vendored service import graph.
HABITBENCH_SELECTED_METHODS="$METHODS" \
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from scripts.run_multigpu_plan import _preflight_method_imports

methods = {
    item.strip()
    for item in os.environ["HABITBENCH_SELECTED_METHODS"].split(",")
    if item.strip()
}
results = _preflight_method_imports(
    Path(os.environ["PYTHON_BIN"]),
    dict(os.environ),
    methods,
)
print("method import preflight ready: " + json.dumps(results, sort_keys=True))
PY

SPEC_SHA256="$(sha256sum "$SPEC_PATH" | awk '{print $1}')"
PLAN_ARGS=(
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/create_shard_plan.py"
  --methods "$METHODS"
  --datasets "$DATASETS"
  --shards "$SHARDS"
  --embedding-model-path "$HABITBENCH_EMBED_MODEL"
  --llm-model-path "$HABITBENCH_LLM_MODEL"
  --served-model-name "$HABITBENCH_SERVED_MODEL"
  --lightmem-model-path "$HABITBENCH_LIGHTMEM_MODEL"
  --secom-compressor-path "$HABITBENCH_SECOM_COMPRESSOR"
  --output-root "$OUTPUT_ROOT"
  --plan "$PLAN"
  --metadata "launcher=h-rjob"
  --metadata "job_name=$JOB_NAME"
  --metadata "job_type=$JOB_TYPE"
  --metadata "namespace=ailab-llmarchitecture"
  --metadata "creator_type=$CREATOR_TYPE"
  --metadata "creator_ad=$ACTUAL_CREATOR_AD"
  --metadata "gpu_model=h200"
  --metadata "gpus_per_replica=$GPUS"
  --metadata "replicas=$REPLICAS"
  --metadata "total_gpus=$((GPUS * REPLICAS))"
  --metadata "cpus_per_replica=$CPUS"
  --metadata "memory_mib_per_replica=$MEMORY_MIB"
  --metadata "image=$IMAGE"
  --metadata "mount=$MOUNT_METADATA"
  --metadata "kubebrain_cluster_entry=$KUBEBRAIN_CLUSTER_ENTRY"
  --metadata "llm_model=$HABITBENCH_LLM_MODEL"
  --metadata "llm_model_id=$HABITBENCH_LLM_MODEL_ID"
  --metadata "llm_model_revision=$HABITBENCH_LLM_MODEL_REVISION"
  --metadata "served_model=$HABITBENCH_SERVED_MODEL"
  --metadata "embedding_model=$HABITBENCH_EMBED_MODEL"
  --metadata "lightmem_model=$HABITBENCH_LIGHTMEM_MODEL"
  --metadata "secom_compressor=$HABITBENCH_SECOM_COMPRESSOR"
  --metadata "h_cluster_spec_sha256=$SPEC_SHA256"
)
if [[ "$FORCE_PLAN" == "1" ]]; then
  PLAN_ARGS+=(--force)
fi
if [[ -n "$MAX_USERS" ]]; then
  PLAN_ARGS+=(--max-users "$MAX_USERS")
fi
if [[ -n "$MAX_PROBES" ]]; then
  PLAN_ARGS+=(--max-probes "$MAX_PROBES")
fi
if [[ -f "$PLAN" && "$FORCE_PLAN" != "1" ]]; then
  PLAN_MANIFEST="${PLAN%.tsv}.manifest.json"
  if [[ ! -f "$PLAN_MANIFEST" ]]; then
    echo "Existing plan has no integrity manifest: $PLAN_MANIFEST" >&2
    exit 1
  fi
  echo "Reusing existing persistent plan: $PLAN"
else
  "${PLAN_ARGS[@]}"
fi

# A resumed launch may change Replica count, but it must not silently change
# the deterministic experiment plan or output mapping.
"$PYTHON_BIN" - "$PLAN" "$OUTPUT_ROOT" "$METHODS" "$DATASETS" "$SHARDS" <<'PY'
import csv
import sys
from pathlib import Path

plan_path = Path(sys.argv[1]).resolve()
output_root = Path(sys.argv[2]).resolve()
expected_methods = [item for item in sys.argv[3].split(",") if item]
expected_datasets = [item for item in sys.argv[4].split(",") if item]
expected_shards = int(sys.argv[5])
with plan_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if not rows:
    raise SystemExit(f"resume plan is empty: {plan_path}")
methods = list(dict.fromkeys(row["method"] for row in rows))
datasets = list(dict.fromkeys(row["dataset_name"] for row in rows))
if methods != expected_methods:
    raise SystemExit(f"resume method mismatch: expected={expected_methods}, actual={methods}")
if datasets != expected_datasets:
    raise SystemExit(f"resume dataset mismatch: expected={expected_datasets}, actual={datasets}")
groups = {}
for row in rows:
    shard_count = int(row["shard_count"])
    if shard_count != expected_shards:
        raise SystemExit(
            f"resume shard-count mismatch: expected={expected_shards}, actual={shard_count}"
        )
    method_root = Path(row["method_output_root"]).resolve()
    try:
        method_root.relative_to(output_root)
    except ValueError as exc:
        raise SystemExit(f"resume output escapes root: {method_root}") from exc
    key = (row["method"], row["dataset_name"], str(method_root))
    groups.setdefault(key, []).append(int(row["shard_index"]))
for key, indices in groups.items():
    if sorted(indices) != list(range(expected_shards)):
        raise SystemExit(f"resume group has incomplete shard indices: {key}: {sorted(indices)}")
expected_tasks = len(expected_methods) * len(expected_datasets) * expected_shards
if len(rows) != expected_tasks:
    raise SystemExit(f"resume task-count mismatch: expected={expected_tasks}, actual={len(rows)}")
print(
    f"resume plan validated: tasks={len(rows)} methods={len(methods)} "
    f"datasets={len(datasets)} shards={expected_shards}"
)
PY

WORKER_COMMAND=(
  bash "$PROJECT_ROOT/scripts/cluster/run_h_eval.sh"
  --plan "$PLAN"
  --gpus "$GPUS"
  --env-file "$ENV_FILE"
  --port-base "$PORT_BASE"
  --expected-replicas "$REPLICAS"
)
if [[ "$CONTINUE_ON_GROUP_ERROR" == "1" ]]; then
  WORKER_COMMAND+=(--continue-on-group-error)
fi

RJOB_COMMAND=(
  rjob submit
  --name "$JOB_NAME"
  --namespace ailab-llmarchitecture
  --gpu "$GPUS"
  --cpu "$CPUS"
  --memory "$MEMORY_MIB"
  -P "$REPLICAS"
  --image "$IMAGE"
  --image-pull-policy IfNotPresent
  --share-host-shm True
  --mount "${MOUNT_CONFIGS[@]}"
)
if (( REPLICAS > 1 )); then
  RJOB_COMMAND+=(
    --host-network=true
    -e DISTRIBUTED_JOB=true
  )
fi
case "$JOB_TYPE" in
  managed-spot)
    RJOB_COMMAND+=(
      --task-type normal
      --priority 1
      --charged-group llmarchitecture_gpu
      --private-machine group
    )
    ;;
  reserved)
    RJOB_COMMAND+=(
      --task-type normal
      --priority 5
      --charged-group llmarchitecture_gpu
      --private-machine group
    )
    ;;
  idle)
    RJOB_COMMAND+=(
      --task-type idle
      --restart-policy never
      --termination-grace-period-seconds 30
    )
    ;;
esac
if [[ "$DRY_RUN" == "1" ]]; then
  RJOB_COMMAND+=(--dry-run True)
fi
RJOB_COMMAND+=(-- "${WORKER_COMMAND[@]}")

mkdir -p "$OUTPUT_ROOT"
echo "RJob type=$JOB_TYPE creator=$ACTUAL_CREATOR_AD/$CREATOR_TYPE resources=$GPUS H200 x $REPLICAS Replicas = $((GPUS * REPLICAS)) total GPUs; cpu=$CPUS memory_mib=$MEMORY_MIB per Replica"
printf 'RJob command: '
printf '%q ' "${RJOB_COMMAND[@]}"
printf '\n'

# The highest llmarchitecture specification is deliberately re-read and SSH
# credentials initialized immediately before the only RJob state operation.
LATEST_SPEC_SHA256="$(sha256sum "$SPEC_PATH" | awk '{print $1}')"
if [[ "$LATEST_SPEC_SHA256" != "$SPEC_SHA256" ]]; then
  echo "H llmarchitecture specification changed during preflight; rerun the launcher" >&2
  exit 1
fi
sed -n '1,$p' "$SPEC_PATH" >/dev/null
# shellcheck disable=SC1091
source /etc/profile.d/ssh-init.sh
# ssh-init may refresh the BrainPP environment; restore and verify the required
# llmarchitecture scheduler route immediately before invoking rjob.
export KUBEBRAIN_CLUSTER_ENTRY="$KUBEBRAIN_CLUSTER_ENTRY_REQUIRED"
if [[ "$KUBEBRAIN_CLUSTER_ENTRY" != "$KUBEBRAIN_CLUSTER_ENTRY_REQUIRED" ]]; then
  echo "Invalid llmarchitecture KUBEBRAIN_CLUSTER_ENTRY: $KUBEBRAIN_CLUSTER_ENTRY" >&2
  exit 1
fi
"${RJOB_COMMAND[@]}" 2>&1 | tee -a "$OUTPUT_ROOT/h_rjob_submit.log"
