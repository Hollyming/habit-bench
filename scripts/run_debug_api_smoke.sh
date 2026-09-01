#!/usr/bin/env bash
set -euo pipefail

# Run a small external-API smoke suite from the already allocated 8-H200
# debug RJob. This is intentionally separate from formal API result roots.

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/workspace/habit-bench}"
ENV_FILE="${HABITBENCH_ENV_FILE:-$PROJECT_ROOT/scripts/cluster/env.h.example.sh}"
CREDENTIAL_FILE="${HABITBENCH_API_CREDENTIAL_FILE:-/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/.secrets/habitbench-pjlab-api.env}"
OUTPUT_ROOT=""
MODELS="deepseek-v4-pro-0813,glm-5.2"
METHODS="no_memory,full_memory,full_history,recency_5,recency_10,bm25_rag,dense_rag,temporal_hybrid_rag"
DATASETS="food,finance,software,travel"
MAX_USERS=1
MAX_PROBES=4
SHARDS=1
GATEWAY_PORT=8190
RPM=60
TPM=50000000

usage() {
  cat <<'EOF'
Usage: scripts/run_debug_api_smoke.sh --output-root PATH [options]
  --output-root PATH       persistent smoke output root (required)
  --models CSV             default: deepseek-v4-pro-0813,glm-5.2
  --methods CSV            fast methods used by this smoke test
  --datasets CSV           default: food,finance,software,travel
  --max-users N            default: 1
  --max-probes N           default: 4 per domain/method
  --gateway-port N         default: 8190
  --credential-file PATH   mode-600 external API credential file
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
    --models) MODELS="${2:?missing value for --models}"; shift 2 ;;
    --methods) METHODS="${2:?missing value for --methods}"; shift 2 ;;
    --datasets) DATASETS="${2:?missing value for --datasets}"; shift 2 ;;
    --max-users) MAX_USERS="${2:?missing value for --max-users}"; shift 2 ;;
    --max-probes) MAX_PROBES="${2:?missing value for --max-probes}"; shift 2 ;;
    --gateway-port) GATEWAY_PORT="${2:?missing value for --gateway-port}"; shift 2 ;;
    --credential-file) CREDENTIAL_FILE="${2:?missing value for --credential-file}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$OUTPUT_ROOT" ]] || { echo "--output-root is required" >&2; exit 2; }
[[ "$OUTPUT_ROOT" == /mnt/shared-storage-* ]] || {
  echo "--output-root must be persistent H storage: $OUTPUT_ROOT" >&2
  exit 2
}
[[ -f "$ENV_FILE" ]] || { echo "missing H environment file: $ENV_FILE" >&2; exit 1; }
[[ -f "$CREDENTIAL_FILE" ]] || { echo "missing credential file: $CREDENTIAL_FILE" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN must be set by the H environment file}"
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 1; }
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for value in MAX_USERS MAX_PROBES SHARDS GATEWAY_PORT RPM TPM; do
  [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || { echo "$value must be positive" >&2; exit 2; }
done

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a METHOD_LIST <<< "$METHODS"
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
(( ${#MODEL_LIST[@]} > 0 )) || { echo "no models selected" >&2; exit 2; }
(( ${#METHOD_LIST[@]} > 0 )) || { echo "no methods selected" >&2; exit 2; }
(( ${#DATASET_LIST[@]} > 0 )) || { echo "no datasets selected" >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT/api_gateway" "$OUTPUT_ROOT/api_runner_logs" "$OUTPUT_ROOT/api_runtime"
if [[ -e "$OUTPUT_ROOT/api_runtime/active-suite.lock" ]]; then
  echo "smoke output root already exists; choose a new --output-root" >&2
  exit 1
fi
exec 9>"$OUTPUT_ROOT/api_runtime/active-suite.lock"
flock -n 9 || { echo "smoke output root is already active" >&2; exit 1; }

credential_summary="$($PYTHON_BIN - "$CREDENTIAL_FILE" <<'PY'
import stat
import sys
from pathlib import Path
from eval.api_gateway import load_credentials

path = Path(sys.argv[1])
mode = stat.S_IMODE(path.stat().st_mode)
if mode & 0o077:
    raise SystemExit(f"credential file must be mode 600 or stricter, got {mode:o}")
keys, _ = load_credentials(path)
if not keys:
    raise SystemExit("credential file contains no API keys")
print(f"{mode:o}:{len(keys)}")
PY
)"
echo "debug smoke start: output_root=$OUTPUT_ROOT models=$MODELS methods=$METHODS datasets=$DATASETS max_users=$MAX_USERS max_probes=$MAX_PROBES credential_slots=${credential_summary##*:}"

slug_for_model() {
  local model="$1"
  model="${model//./-}"
  printf '%s' "$model" | tr -c 'a-z0-9-' '-'
}

for model in "${MODEL_LIST[@]}"; do
  slug="$(slug_for_model "$model")"
  model_root="$OUTPUT_ROOT/$slug"
  plan="$model_root/shard_plan.tsv"
  mkdir -p "$model_root"
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/create_shard_plan.py" \
    --methods "$METHODS" \
    --datasets "$DATASETS" \
    --shards "$SHARDS" \
    --max-users "$MAX_USERS" \
    --max-probes "$MAX_PROBES" \
    --embedding-model-path "$HABITBENCH_EMBED_MODEL" \
    --llm-model-path "$HABITBENCH_LLM_MODEL" \
    --served-model-name "$model" \
    --llm-provider external-openai-compatible \
    --lightmem-model-path "$HABITBENCH_LIGHTMEM_MODEL" \
    --secom-compressor-path "$HABITBENCH_SECOM_COMPRESSOR" \
    --output-root "$model_root" \
    --plan "$plan" \
    --metadata "debug_smoke=true" \
    --metadata "api_model=$model" \
    --metadata "excluded_model=kimi-k3" \
    --metadata "max_users=$MAX_USERS" \
    --metadata "max_probes=$MAX_PROBES"
done

GATEWAY_BASE_URL="http://127.0.0.1:$GATEWAY_PORT/v1"
GATEWAY_LOG="$OUTPUT_ROOT/api_gateway/gateway.log"
GATEWAY_METRICS="$OUTPUT_ROOT/api_gateway/metrics.json"
"$PYTHON_BIN" -m eval.api_gateway \
  --credential-file "$CREDENTIAL_FILE" \
  --host 127.0.0.1 \
  --port "$GATEWAY_PORT" \
  --rpm "$RPM" \
  --tpm "$TPM" \
  --max-upstream-retries 8 \
  --max-empty-response-retries 2 \
  --metrics-path "$GATEWAY_METRICS" \
  --metrics-every 10 \
  > >(sed -u 's/^/[api-gateway] /' | tee -a "$GATEWAY_LOG") 2>&1 &
gateway_pid=$!
runner_pids=()
terminated=0

stop_children() {
  terminated=1
  for pid in "${runner_pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
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
            print("debug API gateway health check passed", flush=True)
            break
    except Exception as exc:
        last = exc
    time.sleep(1)
else:
    raise SystemExit(f"gateway did not become ready: {last}")
PY

run_model() {
  local model="$1" index="$2" slug model_root plan runner_log
  slug="$(slug_for_model "$model")"
  model_root="$OUTPUT_ROOT/$slug"
  plan="$model_root/shard_plan.tsv"
  runner_log="$OUTPUT_ROOT/api_runner_logs/$slug.log"
  echo "[$model] runner start: plan=$plan" | tee -a "$runner_log"
  set +e
  env \
    OPENAI_API_KEY=dummy \
    OPENAI_BASE_URL="$GATEWAY_BASE_URL" \
    HABITBENCH_INFERENCE_BACKEND=external-openai-compatible \
    HABITBENCH_EXTERNAL_API_PROVIDER=pjlab-token \
    HABITBENCH_SERVED_MODEL="$model" \
    HABITBENCH_ANSWER_MAX_TOKENS=1024 \
    HABITBENCH_ANSWER_TIMEOUT_SEC=300 \
    HABITBENCH_ANSWER_MAX_RETRIES=3 \
    HABITBENCH_MED_USER_WORKERS=1 \
    HABITBENCH_ADAPTER_CPU_THREADS=2 \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_multigpu_plan.py" \
      --plan "$plan" \
      --gpus 0 \
      --env-file "$ENV_FILE" \
      --external-base-url "$GATEWAY_BASE_URL" \
      --port-base "$((8300 + index * 100))" \
      --replica-index 0 \
      --replica-count 1 \
      --coordination-root "$model_root/distributed_queue" \
      --coordinator-id "debug-smoke-${slug}-$$" \
      --runtime-output "$model_root/api_runtime/suite_runtime.json" \
      --log-root "$model_root/api_runtime/unused-vllm-logs" \
      --continue-on-group-error \
    2>&1 | sed -u "s/^/[$model] /" | tee -a "$runner_log"
  local status=${PIPESTATUS[0]}
  if (( status == 0 )); then
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/merge_shard_plan.py" --plan "$plan" \
      2>&1 | sed -u "s/^/[$model merge] /" | tee -a "$runner_log"
    status=${PIPESTATUS[0]}
  fi
  echo "[$model] smoke terminal: returncode=$status" | tee -a "$runner_log"
  return "$status"
}

for index in "${!MODEL_LIST[@]}"; do
  run_model "${MODEL_LIST[$index]}" "$index" &
  runner_pids+=("$!")
done

overall_status=0
for pid in "${runner_pids[@]}"; do
  wait "$pid" || overall_status=1
done
kill -TERM "$gateway_pid" 2>/dev/null || true
set +e
wait "$gateway_pid"
gateway_status=$?
set -e
trap - TERM INT
if (( terminated != 0 )); then overall_status=143; fi

"$PYTHON_BIN" - "$OUTPUT_ROOT/api_runtime/suite_terminal.json" "$overall_status" "$gateway_status" "$MODELS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "contract_version": "habitbench.debug_api_smoke.v1",
    "status": "succeeded" if int(sys.argv[2]) == 0 else "failed",
    "returncode": int(sys.argv[2]),
    "gateway_returncode": int(sys.argv[3]),
    "models": sys.argv[4].split(","),
    "finished_at": datetime.now(timezone.utc).isoformat(),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
echo "debug API smoke terminal: returncode=$overall_status status_file=$OUTPUT_ROOT/api_runtime/suite_terminal.json"
exit "$overall_status"
