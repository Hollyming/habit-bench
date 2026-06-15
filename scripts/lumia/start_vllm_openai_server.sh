#!/usr/bin/env bash
set -euo pipefail

# Start a local OpenAI-compatible vLLM server for full official-method runs.
#
# Usage:
#   bash ./scripts/lumia/start_vllm_openai_server.sh
#
# Environment:
#   HABITBENCH_LLM_MODEL       HuggingFace model id or local path.
#   HABITBENCH_SERVED_MODEL    Name exposed through /v1/models.
#   HABITBENCH_VLLM_HOST       Bind host, default 0.0.0.0.
#   HABITBENCH_VLLM_PORT       Bind port, default 8000.
#   HABITBENCH_TENSOR_PARALLEL Tensor parallel size, default 1.
#   HABITBENCH_GPU_MEMORY_UTIL vLLM GPU memory utilization, default 0.90.
#   HABITBENCH_MAX_MODEL_LEN   vLLM max model length, default 16384.
#   HABITBENCH_VLLM_EXTRA_ARGS Optional extra args passed to vLLM.

MODEL="${HABITBENCH_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL="${HABITBENCH_SERVED_MODEL:-habitbench-open-llm}"
HOST="${HABITBENCH_VLLM_HOST:-0.0.0.0}"
PORT="${HABITBENCH_VLLM_PORT:-8000}"
TP="${HABITBENCH_TENSOR_PARALLEL:-1}"
GPU_UTIL="${HABITBENCH_GPU_MEMORY_UTIL:-0.90}"
MAX_MODEL_LEN="${HABITBENCH_MAX_MODEL_LEN:-16384}"
EXTRA_ARGS="${HABITBENCH_VLLM_EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "No python3 or python executable found on PATH" >&2
    exit 127
  fi
fi
export PYTHON_BIN

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import vllm
PY
then
  "$PYTHON_BIN" -m pip install -U vllm
fi

"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  $EXTRA_ARGS
