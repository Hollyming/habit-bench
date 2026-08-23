#!/usr/bin/env bash

# Qwen3-4B scale-ablation profile for the H200 evaluator. Auxiliary models,
# context length and inference settings intentionally match the 8B main run.

HABITBENCH_Q4_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HABITBENCH_H_ROOT="${HABITBENCH_H_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang}"
export HABITBENCH_H_MODEL_ROOT="${HABITBENCH_H_MODEL_ROOT:-$HABITBENCH_H_ROOT/models/habitbench}"
export HABITBENCH_LLM_MODEL="$HABITBENCH_H_MODEL_ROOT/Qwen3-4B"
export HABITBENCH_SERVED_MODEL="Qwen3-4B"
export HABITBENCH_LLM_MODEL_ID="Qwen/Qwen3-4B"
export HABITBENCH_LLM_MODEL_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"

# shellcheck disable=SC1091
source "$HABITBENCH_Q4_ENV_DIR/env.h.example.sh"

unset HABITBENCH_Q4_ENV_DIR
