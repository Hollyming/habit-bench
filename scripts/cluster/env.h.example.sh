#!/usr/bin/env bash

# H-cluster paths. Copy this file to an ignored env.h.local.sh when the
# persistent environment or model layout differs from these defaults.

HABITBENCH_H_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HABITBENCH_H_ROOT="${HABITBENCH_H_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang}"
export HABITBENCH_PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$HABITBENCH_H_ENV_DIR/../.." && pwd)}"
export HABITBENCH_H_ENV_ROOT="${HABITBENCH_H_ENV_ROOT:-$HABITBENCH_H_ROOT/envs}"
export HABITBENCH_H_MODEL_ROOT="${HABITBENCH_H_MODEL_ROOT:-$HABITBENCH_H_ROOT/models/habitbench}"

export PYTHON_BIN="${PYTHON_BIN:-$HABITBENCH_H_ENV_ROOT/habitbenchmark/bin/python}"
export HABITBENCH_VLLM_PYTHON="${HABITBENCH_VLLM_PYTHON:-$HABITBENCH_H_ENV_ROOT/habitbenchmark-vllm/bin/python}"
export HABITBENCH_LLM_MODEL="${HABITBENCH_LLM_MODEL:-$HABITBENCH_H_MODEL_ROOT/Qwen3-8B}"
export HABITBENCH_EMBED_MODEL="${HABITBENCH_EMBED_MODEL:-$HABITBENCH_H_MODEL_ROOT/bge-m3}"
export HABITBENCH_LIGHTMEM_MODEL="${HABITBENCH_LIGHTMEM_MODEL:-$HABITBENCH_H_MODEL_ROOT/llmlingua-2-xlm-roberta-large-meetingbank}"
export HABITBENCH_SECOM_COMPRESSOR="${HABITBENCH_SECOM_COMPRESSOR:-$HABITBENCH_LIGHTMEM_MODEL}"

export HF_HOME="${HF_HOME:-$HABITBENCH_H_ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HABITBENCH_H_ROOT/.cache}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HABITBENCH_H_ROOT/.cache/tiktoken}"
export HABITBENCH_TRITON_PTXAS_PATH="${HABITBENCH_TRITON_PTXAS_PATH:-$HABITBENCH_H_ENV_ROOT/habitbenchmark-vllm/lib/python3.10/site-packages/triton/backends/nvidia/bin/ptxas}"

# Reuse the frozen evaluation and vLLM settings; every path above is exported
# first, so the legacy ClusterX defaults in the shared profile cannot replace
# the H-cluster values.
# shellcheck disable=SC1091
source "$HABITBENCH_H_ENV_DIR/env.example.sh"

unset HABITBENCH_H_ENV_DIR
