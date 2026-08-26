#!/usr/bin/env bash
set -euo pipefail

# Submit all registered methods over the current four domains for three
# external API models in one single-node 8-H200 RJob.

LAUNCHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC_PATH="/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/H集群architecture分区RJob任务提交规范.md"
KUBEBRAIN_CLUSTER_ENTRY_REQUIRED="http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451"
export KUBEBRAIN_CLUSTER_ENTRY="$KUBEBRAIN_CLUSTER_ENTRY_REQUIRED"

ENV_FILE="${HABITBENCH_ENV_FILE:-$LAUNCHER_ROOT/scripts/cluster/env.h.example.sh}"
CREDENTIAL_FILE="${HABITBENCH_API_CREDENTIAL_FILE:-}"
MODELS="deepseek-v4-pro-0813,glm-5.2,kimi-k3"
GPU_ALLOCATIONS="3,3,2"
METHODS="no_memory,full_memory,full_history,recency_5,recency_10,bm25_rag,dense_rag,temporal_hybrid_rag,mem0,amem,memos,memrl,lightmem,letta,mirix,secom"
DATASETS="food,finance,software,travel"
SHARDS=8
RPM=60
TPM=50000000
GATEWAY_PORT=8090
CPUS=64
MEMORY_MIB=524288
OUTPUT_ROOT=""
JOB_NAME=""
JOB_TYPE="managed-spot"
CREATOR_TYPE="${HABITBENCH_CREATOR_TYPE:-}"
CREATOR_AD="${HABITBENCH_CREATOR_AD:-}"
IMAGE="${HABITBENCH_H_IMAGE:-registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20251117}"
MOUNT_CONFIGS=()
MOUNTS_FROM_CLI=0
FORCE_PLANS=0
DRY_RUN=0

usage() {
  echo "Usage: scripts/submit_h_api_suite.sh [options]"
  echo "  --job-type TYPE       managed-spot (default), reserved, or idle"
  echo "  --creator-type TYPE   actual authenticated identity: user or group (required)"
  echo "  --creator-ad AD       actual authenticated account name (required)"
  echo "  --output-root PATH    defaults below the clone's ./results"
  echo "  --job-name NAME       lowercase letters/digits/hyphens, max 32 chars"
  echo "  --env-file PATH       H method environment"
  echo "  --credential-file PATH  required mode-600 external API credential file"
  echo "  --models CSV          default: deepseek-v4-pro-0813,glm-5.2,kimi-k3"
  echo "  --gpu-allocations CSV default: 3,3,2; must sum to 8"
  echo "  --methods CSV         default: all 16 registered methods"
  echo "  --datasets CSV        default: current four domains"
  echo "  --shards N            per method/domain/model, default: 8"
  echo "  --rpm N               shared API RPM, default: 60"
  echo "  --tpm N               shared API TPM, default: 50000000"
  echo "  --cpus N              per Replica, default: 64"
  echo "  --memory-mib N        per Replica, default: 524288"
  echo "  --image IMAGE         explicit H image"
  echo "  --mount CONFIG        explicit GPFS mount; repeatable"
  echo "  --force-plans         intentionally replace existing plans"
  echo "  --dry-run             render RJob only"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-type) JOB_TYPE="${2:?missing value for --job-type}"; shift 2 ;;
    --creator-type) CREATOR_TYPE="${2:?missing value for --creator-type}"; shift 2 ;;
    --creator-ad) CREATOR_AD="${2:?missing value for --creator-ad}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
    --job-name) JOB_NAME="${2:?missing value for --job-name}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?missing value for --env-file}"; shift 2 ;;
    --credential-file) CREDENTIAL_FILE="${2:?missing value for --credential-file}"; shift 2 ;;
    --models) MODELS="${2:?missing value for --models}"; shift 2 ;;
    --gpu-allocations) GPU_ALLOCATIONS="${2:?missing value for --gpu-allocations}"; shift 2 ;;
    --methods) METHODS="${2:?missing value for --methods}"; shift 2 ;;
    --datasets) DATASETS="${2:?missing value for --datasets}"; shift 2 ;;
    --shards) SHARDS="${2:?missing value for --shards}"; shift 2 ;;
    --rpm) RPM="${2:?missing value for --rpm}"; shift 2 ;;
    --tpm) TPM="${2:?missing value for --tpm}"; shift 2 ;;
    --gateway-port) GATEWAY_PORT="${2:?missing value for --gateway-port}"; shift 2 ;;
    --cpus) CPUS="${2:?missing value for --cpus}"; shift 2 ;;
    --memory-mib) MEMORY_MIB="${2:?missing value for --memory-mib}"; shift 2 ;;
    --image) IMAGE="${2:?missing value for --image}"; shift 2 ;;
    --mount) MOUNT_CONFIGS+=("${2:?missing value for --mount}"); MOUNTS_FROM_CLI=1; shift 2 ;;
    --force-plans) FORCE_PLANS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in SHARDS RPM TPM GATEWAY_PORT CPUS MEMORY_MIB; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer: $value" >&2
    exit 2
  fi
done
if (( CPUS < 64 || MEMORY_MIB < 524288 )); then
  echo "An 8-H200 API suite requires at least 64 CPU and 524288 MiB memory" >&2
  exit 2
fi
if (( GATEWAY_PORT >= 65536 )); then
  echo "--gateway-port must be below 65536" >&2
  exit 2
fi

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a ALLOCATION_LIST <<< "$GPU_ALLOCATIONS"
if (( ${#MODEL_LIST[@]} != ${#ALLOCATION_LIST[@]} || ${#MODEL_LIST[@]} == 0 )); then
  echo "--models and --gpu-allocations must have equal nonzero lengths" >&2
  exit 2
fi
allocation_total=0
for index in "${!MODEL_LIST[@]}"; do
  model="${MODEL_LIST[$index]}"
  allocation="${ALLOCATION_LIST[$index]}"
  if [[ ! "$model" =~ ^[a-z0-9][a-z0-9._-]*$ || ! "$allocation" =~ ^[1-9][0-9]*$ ]]; then
    echo "Unsafe model/allocation: model=$model allocation=$allocation" >&2
    exit 2
  fi
  allocation_total=$((allocation_total + allocation))
done
if (( allocation_total != 8 )); then
  echo "Model allocations must sum to the single Replica's 8 H200s" >&2
  exit 2
fi

case "$JOB_TYPE" in
  managed-spot)
    [[ "$CREATOR_TYPE" == "user" ]] || { echo "managed-spot requires --creator-type user" >&2; exit 2; }
    ;;
  reserved)
    [[ "$CREATOR_TYPE" == "group" ]] || { echo "reserved requires --creator-type group" >&2; exit 2; }
    ;;
  idle)
    [[ "$CREATOR_TYPE" == "user" || "$CREATOR_TYPE" == "group" ]] || {
      echo "idle requires --creator-type user|group" >&2; exit 2;
    }
    ;;
  *) echo "--job-type must be managed-spot, reserved, or idle" >&2; exit 2 ;;
esac
if [[ -z "$CREATOR_AD" || ! "$CREATOR_AD" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--creator-ad must name the actual authenticated account" >&2
  exit 2
fi

# Resolve the same BrainPP identity that will own the final RJob.
# shellcheck disable=SC1091
source /etc/profile.d/ssh-init.sh
export KUBEBRAIN_CLUSTER_ENTRY="$KUBEBRAIN_CLUSTER_ENTRY_REQUIRED"
ACTUAL_CREATOR_AD="${BRAIN_USERNAME:-}"
if [[ -z "$ACTUAL_CREATOR_AD" || "$ACTUAL_CREATOR_AD" == "brainpp" ]]; then
  WORKSPACE_FQDN="$(hostname -f)"
  IFS='.' read -r -a FQDN_PARTS <<< "$WORKSPACE_FQDN"
  if (( ${#FQDN_PARTS[@]} > 1 )); then
    ACTUAL_CREATOR_AD="${FQDN_PARTS[1]}"
  fi
fi
if [[ -z "$ACTUAL_CREATOR_AD" || "$ACTUAL_CREATOR_AD" == "brainpp" ]]; then
  echo "Cannot resolve the actual authenticated RJob creator" >&2
  exit 1
fi
if [[ "$CREATOR_AD" != "$ACTUAL_CREATOR_AD" ]]; then
  echo "--creator-ad=$CREATOR_AD does not match authenticated creator $ACTUAL_CREATOR_AD" >&2
  exit 2
fi

if [[ -z "$CREDENTIAL_FILE" ]]; then
  echo "--credential-file (or HABITBENCH_API_CREDENTIAL_FILE) is required" >&2
  exit 2
fi
for required_file in "$SPEC_PATH" "$ENV_FILE" "$CREDENTIAL_FILE"; do
  [[ -f "$required_file" ]] || { echo "Required file is missing: $required_file" >&2; exit 1; }
done
ENV_FILE="$(realpath "$ENV_FILE")"
CREDENTIAL_FILE="$(realpath "$CREDENTIAL_FILE")"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$LAUNCHER_ROOT}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN must be set by the H environment file}"

STAMP="$(date -u +%m%d-%H%M%S)"
JOB_NAME="${JOB_NAME:-zjm-api-main-$STAMP}"
if (( ${#JOB_NAME} > 32 )) || [[ ! "$JOB_NAME" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]]; then
  echo "--job-name must be <=32 lowercase letters/digits/hyphens: $JOB_NAME" >&2
  exit 2
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/api-main-$STAMP}"
if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$PROJECT_ROOT/$OUTPUT_ROOT"
fi
if [[ "$OUTPUT_ROOT" != /mnt/shared-storage-* ]]; then
  echo "--output-root must be persistent H storage: $OUTPUT_ROOT" >&2
  exit 2
fi

for required_name in HABITBENCH_LLM_MODEL HABITBENCH_EMBED_MODEL HABITBENCH_LIGHTMEM_MODEL HABITBENCH_SECOM_COMPRESSOR TIKTOKEN_CACHE_DIR HF_HOME XDG_CACHE_HOME VLLM_CACHE_ROOT TORCH_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR; do
  [[ -n "${!required_name:-}" ]] || { echo "$required_name is required" >&2; exit 2; }
done
for required in \
  "$PYTHON_BIN" \
  "$HABITBENCH_LLM_MODEL/config.json" \
  "$HABITBENCH_EMBED_MODEL/config.json" \
  "$HABITBENCH_EMBED_MODEL/pytorch_model.bin" \
  "$HABITBENCH_EMBED_MODEL/HABIT_MODEL_INFO.json" \
  "$HABITBENCH_LIGHTMEM_MODEL/model.safetensors" \
  "$HABITBENCH_SECOM_COMPRESSOR/model.safetensors" \
  "$PROJECT_ROOT/scripts/create_api_suite_plans.py" \
  "$PROJECT_ROOT/scripts/cluster/run_h_api_suite.sh" \
  "$PROJECT_ROOT/eval/api_gateway.py"
do
  [[ -e "$required" ]] || { echo "Required API evaluation path is missing: $required" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Method Python is not executable: $PYTHON_BIN" >&2; exit 1; }
if [[ ! -f "$HABITBENCH_LLM_MODEL/tokenizer.json" && ! -f "$HABITBENCH_LLM_MODEL/tokenizer.model" ]]; then
  echo "Shared context-budget tokenizer is missing under $HABITBENCH_LLM_MODEL" >&2
  exit 1
fi

API_ORIGIN="$($PYTHON_BIN - "$CREDENTIAL_FILE" "${MODEL_LIST[@]}" <<'PY'
import json
import stat
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

path = Path(sys.argv[1])
mode = stat.S_IMODE(path.stat().st_mode)
if mode & 0o077:
    raise SystemExit(f"credential file must have mode 600 or stricter, got {mode:o}")
values = {}
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line and not line.startswith("#"):
        if "=" not in line:
            raise SystemExit("credential file contains a malformed line")
        name, value = line.split("=", 1)
        values[name.strip()] = value
key = values.get("OPENAI_API_KEY")
base = values.get("HABITBENCH_EXTERNAL_API_BASE_URL", "").rstrip("/")
if not key or urlsplit(base).scheme != "https":
    raise SystemExit("credential file has no usable key/HTTPS base URL")
request = urllib.request.Request(
    base + "/models", headers={"Authorization": "Bearer " + key}
)
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.loads(response.read())
available = {
    str(item.get("id"))
    for item in payload.get("data", [])
    if isinstance(item, dict)
}
missing = [model for model in sys.argv[2:] if model not in available]
if missing:
    raise SystemExit(f"external endpoint is missing requested models: {missing}")
print(urlsplit(base).netloc)
PY
)"
echo "external API preflight passed: origin=$API_ORIGIN models=$MODELS credential=redacted"

if (( MOUNTS_FROM_CLI == 0 )); then
  if [[ -n "${HABITBENCH_H_MOUNTS:-}" ]]; then
    read -r -a MOUNT_CONFIGS <<< "$HABITBENCH_H_MOUNTS"
  elif [[ -n "${HABITBENCH_H_MOUNT:-}" ]]; then
    MOUNT_CONFIGS=("$HABITBENCH_H_MOUNT")
  fi
fi
REQUIRED_MOUNT_PATHS=(
  "$PROJECT_ROOT"
  "$ENV_FILE"
  "$CREDENTIAL_FILE"
  "$PYTHON_BIN"
  "$HABITBENCH_LLM_MODEL"
  "$HABITBENCH_EMBED_MODEL"
  "$HABITBENCH_LIGHTMEM_MODEL"
  "$HABITBENCH_SECOM_COMPRESSOR"
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
  GPFS2_PREFIX="/mnt/shared-storage-gpfs2/plm-gpfs"
  for mounted_path in "${REQUIRED_MOUNT_PATHS[@]}"; do
    if [[ "$mounted_path" != "$GPFS2_PREFIX/"* ]]; then
      echo "Cannot infer H mount for: $mounted_path" >&2
      exit 2
    fi
    relative="${mounted_path#"$GPFS2_PREFIX/"}"
    owner="${relative%%/*}"
    append_unique_mount "gpfs://gpfs2/plm-gpfs/$owner:$GPFS2_PREFIX/$owner"
  done
fi
MOUNT_TARGETS=()
for mount_config in "${MOUNT_CONFIGS[@]}"; do
  if [[ ! "$mount_config" =~ ^gpfs://[^:]+:/mnt/shared-storage- ]]; then
    echo "Invalid persistent mount: $mount_config" >&2
    exit 2
  fi
  MOUNT_TARGETS+=("${mount_config##*:}")
done
for mounted_path in "${REQUIRED_MOUNT_PATHS[@]}"; do
  covered=0
  for target in "${MOUNT_TARGETS[@]}"; do
    if [[ "$mounted_path" == "$target" || "$mounted_path" == "$target/"* ]]; then
      covered=1
      break
    fi
  done
  if (( covered != 1 )); then
    echo "No RJob mount covers required path: $mounted_path" >&2
    exit 2
  fi
done
MOUNT_METADATA="$(IFS=' '; echo "${MOUNT_CONFIGS[*]}")"

mkdir -p \
  "$OUTPUT_ROOT" \
  "$HF_HOME" \
  "$XDG_CACHE_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR"

SPEC_SHA256="$(sha256sum "$SPEC_PATH" | awk '{print $1}')"
PLAN_ARGS=(
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/create_api_suite_plans.py"
  --models "$MODELS"
  --gpu-allocations "$GPU_ALLOCATIONS"
  --methods "$METHODS"
  --datasets "$DATASETS"
  --shards "$SHARDS"
  --output-root "$OUTPUT_ROOT"
  --embedding-model-path "$HABITBENCH_EMBED_MODEL"
  --tokenizer-model-path "$HABITBENCH_LLM_MODEL"
  --lightmem-model-path "$HABITBENCH_LIGHTMEM_MODEL"
  --secom-compressor-path "$HABITBENCH_SECOM_COMPRESSOR"
  --job-name "$JOB_NAME"
  --job-type "$JOB_TYPE"
  --api-origin "$API_ORIGIN"
  --rpm "$RPM"
  --tpm "$TPM"
  --metadata "launcher=h-api-rjob"
  --metadata "namespace=ailab-llmarchitecture"
  --metadata "creator_type=$CREATOR_TYPE"
  --metadata "creator_ad=$ACTUAL_CREATOR_AD"
  --metadata "gpu_model=h200"
  --metadata "gpus_per_replica=8"
  --metadata "replicas=1"
  --metadata "total_gpus=8"
  --metadata "cpus_per_replica=$CPUS"
  --metadata "memory_mib_per_replica=$MEMORY_MIB"
  --metadata "image=$IMAGE"
  --metadata "mount=$MOUNT_METADATA"
  --metadata "kubebrain_cluster_entry=$KUBEBRAIN_CLUSTER_ENTRY"
  --metadata "h_cluster_spec_sha256=$SPEC_SHA256"
)
if (( FORCE_PLANS == 1 )); then
  PLAN_ARGS+=(--force)
fi
"${PLAN_ARGS[@]}"

WORKER_COMMAND=(
  bash "$PROJECT_ROOT/scripts/cluster/run_h_api_suite.sh"
  --suite-root "$OUTPUT_ROOT"
  --env-file "$ENV_FILE"
  --credential-file "$CREDENTIAL_FILE"
  --models "$MODELS"
  --gpu-allocations "$GPU_ALLOCATIONS"
  --gateway-port "$GATEWAY_PORT"
  --rpm "$RPM"
  --tpm "$TPM"
)

RJOB_COMMAND=(
  rjob submit
  --name "$JOB_NAME"
  --namespace ailab-llmarchitecture
  --gpu 8
  --cpu "$CPUS"
  --memory "$MEMORY_MIB"
  -P 1
  --image "$IMAGE"
  --image-pull-policy IfNotPresent
  --share-host-shm True
  --mount "${MOUNT_CONFIGS[@]}"
)
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
if (( DRY_RUN == 1 )); then
  RJOB_COMMAND+=(--dry-run True)
fi
RJOB_COMMAND+=(-- "${WORKER_COMMAND[@]}")

echo "RJob type=$JOB_TYPE creator=$ACTUAL_CREATOR_AD/$CREATOR_TYPE resources=8 H200 x 1 Replica = 8 total GPUs; cpu=$CPUS memory_mib=$MEMORY_MIB"
echo "API allocation models=$MODELS gpu_allocations=$GPU_ALLOCATIONS shared_limit=${RPM}RPM/${TPM}TPM"
printf 'RJob command: '
printf '%q ' "${RJOB_COMMAND[@]}"
printf '\n'

# Re-read the highest specification and reinitialize SSH immediately before
# the sole RJob state-changing operation.
LATEST_SPEC_SHA256="$(sha256sum "$SPEC_PATH" | awk '{print $1}')"
if [[ "$LATEST_SPEC_SHA256" != "$SPEC_SHA256" ]]; then
  echo "H llmarchitecture specification changed during preflight; rerun" >&2
  exit 1
fi
sed -n '1,$p' "$SPEC_PATH" >/dev/null
# shellcheck disable=SC1091
source /etc/profile.d/ssh-init.sh
export KUBEBRAIN_CLUSTER_ENTRY="$KUBEBRAIN_CLUSTER_ENTRY_REQUIRED"
if [[ "$KUBEBRAIN_CLUSTER_ENTRY" != "$KUBEBRAIN_CLUSTER_ENTRY_REQUIRED" ]]; then
  echo "Invalid llmarchitecture scheduler entry" >&2
  exit 1
fi
"${RJOB_COMMAND[@]}" 2>&1 | tee -a "$OUTPUT_ROOT/h_rjob_submit.log"
