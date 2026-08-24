#!/usr/bin/env bash

# Qwen3-32B scale-ablation profile for the H200 evaluator. The BF16 model fits
# on one 141-GB H200, so each GPU still hosts one independent vLLM worker.

HABITBENCH_Q32_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HABITBENCH_H_SHARED_ROOT="${HABITBENCH_H_SHARED_ROOT:-${HABITBENCH_H_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang}}"
export HABITBENCH_H_ROOT="$HABITBENCH_H_SHARED_ROOT"
export HABITBENCH_H_MODEL_ROOT="${HABITBENCH_H_MODEL_ROOT:-$HABITBENCH_H_SHARED_ROOT/models/habitbench}"
export HABITBENCH_LLM_MODEL="$HABITBENCH_H_MODEL_ROOT/Qwen3-32B"
export HABITBENCH_SERVED_MODEL="Qwen3-32B"
export HABITBENCH_LLM_MODEL_ID="Qwen/Qwen3-32B"
export HABITBENCH_LLM_MODEL_REVISION="9216db5781bf21249d130ec9da846c4624c16137"

# shellcheck disable=SC1091
source "$HABITBENCH_Q32_ENV_DIR/env.h.example.sh"

unset HABITBENCH_Q32_ENV_DIR
