#!/usr/bin/env bash
set -euo pipefail

# Run three external-API model tracks inside one single-node 8-H200 RJob.
# The GPUs remain assigned to local embedding/compression adapters; all LLM
# traffic passes through one credential-owning, globally rate-limited gateway.

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SUITE_ROOT=""
ENV_FILE=""
CREDENTIAL_FILE=""
MODELS="deepseek-v4-pro-0813,glm-5.2,kimi-k3"
GPU_ALLOCATIONS="3,3,2"
GATEWAY_PORT=8090
RPM=60
TPM=50000000

usage() {
  echo "Usage: scripts/cluster/run_h_api_suite.sh --suite-root PATH --env-file PATH --credential-file PATH [options]"
  echo "  --models CSV            default: deepseek-v4-pro-0813,glm-5.2,kimi-k3"
  echo "  --gpu-allocations CSV   default: 3,3,2; must sum to 8"
  echo "  --gateway-port N        default: 8090"
  echo "  --rpm N                 shared upstream limit, default: 60"
  echo "  --tpm N                 shared upstream limit, default: 50000000"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-root) SUITE_ROOT="${2:?missing value for --suite-root}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?missing value for --env-file}"; shift 2 ;;
    --credential-file) CREDENTIAL_FILE="${2:?missing value for --credential-file}"; shift 2 ;;
    --models) MODELS="${2:?missing value for --models}"; shift 2 ;;
    --gpu-allocations) GPU_ALLOCATIONS="${2:?missing value for --gpu-allocations}"; shift 2 ;;
    --gateway-port) GATEWAY_PORT="${2:?missing value for --gateway-port}"; shift 2 ;;
    --rpm) RPM="${2:?missing value for --rpm}"; shift 2 ;;
    --tpm) TPM="${2:?missing value for --tpm}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in "$SUITE_ROOT" "$ENV_FILE" "$CREDENTIAL_FILE"; do
  if [[ -z "$required" ]]; then
    echo "--suite-root, --env-file and --credential-file are required" >&2
    exit 2
  fi
done
if [[ "$SUITE_ROOT" != /mnt/shared-storage-* ]]; then
  echo "--suite-root must use persistent H storage: $SUITE_ROOT" >&2
  exit 2
fi
for required_file in "$ENV_FILE" "$CREDENTIAL_FILE"; do
  [[ -f "$required_file" ]] || { echo "Required file is missing: $required_file" >&2; exit 1; }
done
for value_name in GATEWAY_PORT RPM TPM; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer: $value" >&2
    exit 2
  fi
done
if (( GATEWAY_PORT >= 65536 )); then
  echo "--gateway-port must be below 65536" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN must be set by the H environment file}"
[[ -x "$PYTHON_BIN" ]] || { echo "Method Python is not executable: $PYTHON_BIN" >&2; exit 1; }

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a ALLOCATION_LIST <<< "$GPU_ALLOCATIONS"
if (( ${#MODEL_LIST[@]} != ${#ALLOCATION_LIST[@]} || ${#MODEL_LIST[@]} == 0 )); then
  echo "--models and --gpu-allocations must have the same nonzero length" >&2
  exit 2
fi
allocation_total=0
for allocation in "${ALLOCATION_LIST[@]}"; do
  if ! [[ "$allocation" =~ ^[1-9][0-9]*$ ]]; then
    echo "Every GPU allocation must be positive: $allocation" >&2
    exit 2
  fi
  allocation_total=$((allocation_total + allocation))
done
if (( allocation_total != 8 )); then
  echo "API suite allocations must sum to the requested 8 H200s: $GPU_ALLOCATIONS" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable inside the RJob container" >&2
  exit 1
fi
mapfile -t GPU_NAMES < <(
  nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/[[:space:]]*$//' \
    | sed '/^$/d'
)
if (( ${#GPU_NAMES[@]} != 8 )); then
  echo "RJob exposed ${#GPU_NAMES[@]} GPUs; expected exactly 8" >&2
  exit 1
fi
for gpu_name in "${GPU_NAMES[@]}"; do
  [[ "$gpu_name" == *H200* ]] || { echo "Expected H200, found: $gpu_name" >&2; exit 1; }
done

mkdir -p "$SUITE_ROOT/api_gateway" "$SUITE_ROOT/api_runner_logs" "$SUITE_ROOT/api_runtime"
MASTER_LOG="$SUITE_ROOT/h_api_suite.log"
exec > >(tee -a "$MASTER_LOG") 2>&1
echo "H external API suite start: models=$MODELS allocations=$GPU_ALLOCATIONS rpm=$RPM tpm=$TPM"
echo "H200 preflight passed: names=${GPU_NAMES[*]}"

credential_mode="$($PYTHON_BIN - "$CREDENTIAL_FILE" <<'PY'
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = stat.S_IMODE(path.stat().st_mode)
if mode & 0o077:
    raise SystemExit(f"credential file must not be group/world readable: mode={mode:o}")
names = set()
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line and not line.startswith("#") and "=" in line:
        names.add(line.split("=", 1)[0].strip())
required = {"OPENAI_API_KEY", "HABITBENCH_EXTERNAL_API_BASE_URL"}
if not required <= names:
    raise SystemExit(f"credential file is missing required names: {sorted(required - names)}")
print(f"{mode:o}")
PY
)"
echo "credential preflight passed: path=$CREDENTIAL_FILE mode=$credential_mode values=redacted"

GATEWAY_BASE_URL="http://127.0.0.1:$GATEWAY_PORT/v1"
GATEWAY_LOG="$SUITE_ROOT/api_gateway/gateway.log"
GATEWAY_METRICS="$SUITE_ROOT/api_gateway/metrics.json"
"$PYTHON_BIN" -m eval.api_gateway \
  --credential-file "$CREDENTIAL_FILE" \
  --host 127.0.0.1 \
  --port "$GATEWAY_PORT" \
  --rpm "$RPM" \
  --tpm "$TPM" \
  --max-upstream-retries 120 \
  --metrics-path "$GATEWAY_METRICS" \
  --metrics-every 25 \
  > >(sed -u 's/^/[api-gateway] /' | tee -a "$GATEWAY_LOG") 2>&1 &
gateway_pid=$!

runner_pids=()
terminated=0
stop_children() {
  terminated=1
  for pid in "${runner_pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  kill -TERM "$gateway_pid" 2>/dev/null || true
}
trap stop_children TERM INT

"$PYTHON_BIN" - "$GATEWAY_PORT" <<'PY'
import json
import sys
import time
import urllib.request

url = f"http://127.0.0.1:{int(sys.argv[1])}/healthz"
last = None
for _ in range(60):
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read())
        if payload.get("status") == "ok":
            print("external API gateway health check passed")
            break
    except Exception as exc:
        last = exc
    time.sleep(1)
else:
    raise SystemExit(f"gateway did not become ready: {last}")
PY

slug_for_model() {
  local model="$1"
  model="${model//./-}"
  printf '%s' "$model" | tr -c 'a-z0-9-' '-'
}

run_model() {
  local model="$1"
  local gpu_list="$2"
  local model_index="$3"
  local slug plan model_root coordinator runner_log
  slug="$(slug_for_model "$model")"
  model_root="$SUITE_ROOT/$slug"
  plan="$model_root/shard_plan.tsv"
  coordinator="${JOB_ID:-api-main}-${slug}"
  runner_log="$SUITE_ROOT/api_runner_logs/$slug.log"
  if [[ ! -f "$plan" || ! -f "${plan%.tsv}.manifest.json" ]]; then
    echo "[$model] persistent plan is missing: $plan" >&2
    return 1
  fi
  mkdir -p "$model_root/api_runtime"
  echo "[$model] runner start: gpus=$gpu_list plan=$plan"
  set -o pipefail
  env \
    OPENAI_API_KEY=dummy \
    OPENAI_BASE_URL="$GATEWAY_BASE_URL" \
    HABITBENCH_INFERENCE_BACKEND=external-openai-compatible \
    HABITBENCH_EXTERNAL_API_PROVIDER=pjlab-token \
    HABITBENCH_SERVED_MODEL="$model" \
    HABITBENCH_ANSWER_MAX_TOKENS=256 \
    HABITBENCH_ANSWER_TIMEOUT_SEC=900 \
    HABITBENCH_ANSWER_MAX_RETRIES=5 \
    HABITBENCH_MED_USER_WORKERS=2 \
    HABITBENCH_MEM0_USER_WORKERS=2 \
    HABITBENCH_AMEM_USER_WORKERS=2 \
    HABITBENCH_MEMOS_USER_WORKERS=2 \
    HABITBENCH_MEMRL_USER_WORKERS=2 \
    HABITBENCH_LETTA_USER_WORKERS=2 \
    HABITBENCH_LIGHTMEM_USER_WORKERS=1 \
    HABITBENCH_MIRIX_USER_WORKERS=1 \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_multigpu_plan.py" \
      --plan "$plan" \
      --gpus "$gpu_list" \
      --env-file "$ENV_FILE" \
      --external-base-url "$GATEWAY_BASE_URL" \
      --port-base "$((8200 + model_index * 100))" \
      --replica-index 0 \
      --replica-count 1 \
      --coordination-root "$model_root/distributed_queue" \
      --coordinator-id "$coordinator" \
      --runtime-output "$model_root/api_runtime/suite_runtime.json" \
      --log-root "$model_root/api_runtime/unused-vllm-logs" \
      --continue-on-group-error \
    2>&1 | sed -u "s/^/[$model] /" | tee -a "$runner_log"
  runner_status=${PIPESTATUS[0]}
  if (( runner_status != 0 )); then
    echo "[$model] runner failed: returncode=$runner_status" >&2
    return "$runner_status"
  fi
  echo "[$model] runner terminal; merging completed shards"
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/merge_shard_plan.py" --plan "$plan" \
    2>&1 | sed -u "s/^/[$model merge] /" | tee -a "$runner_log"
}

gpu_cursor=0
MODEL_SLUGS=()
for model_index in "${!MODEL_LIST[@]}"; do
  model="${MODEL_LIST[$model_index]}"
  allocation="${ALLOCATION_LIST[$model_index]}"
  gpu_list=""
  for ((offset = 0; offset < allocation; offset++)); do
    [[ -z "$gpu_list" ]] || gpu_list+=","
    gpu_list+="$((gpu_cursor + offset))"
  done
  gpu_cursor=$((gpu_cursor + allocation))
  MODEL_SLUGS+=("$(slug_for_model "$model")")
  run_model "$model" "$gpu_list" "$model_index" &
  runner_pids+=("$!")
done

runner_statuses=()
set +e
for pid in "${runner_pids[@]}"; do
  wait "$pid"
  runner_statuses+=("$?")
done
set -e

kill -TERM "$gateway_pid" 2>/dev/null || true
set +e
wait "$gateway_pid"
gateway_status=$?
set -e
trap - TERM INT

overall_status=0
for status in "${runner_statuses[@]}"; do
  if (( status != 0 )); then
    overall_status=1
  fi
done
if (( terminated != 0 )); then
  overall_status=143
fi

STATUS_FILE="$SUITE_ROOT/api_runtime/suite_terminal.json"
"$PYTHON_BIN" - "$STATUS_FILE" "$overall_status" "$MODELS" "$GPU_ALLOCATIONS" "${runner_statuses[*]}" "$gateway_status" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
models = sys.argv[3].split(",")
allocations = [int(value) for value in sys.argv[4].split(",")]
statuses = [int(value) for value in sys.argv[5].split()]
payload = {
    "contract_version": "habitbench.h_api_suite.v1",
    "status": "succeeded" if int(sys.argv[2]) == 0 else "failed",
    "returncode": int(sys.argv[2]),
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "models": [
        {"model": model, "gpus": gpu, "returncode": status}
        for model, gpu, status in zip(models, allocations, statuses, strict=True)
    ],
    "gateway_returncode": int(sys.argv[6]),
}
temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
echo "H external API suite terminal: returncode=$overall_status status_file=$STATUS_FILE"
exit "$overall_status"
