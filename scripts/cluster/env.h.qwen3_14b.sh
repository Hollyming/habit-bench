#!/usr/bin/env bash

# Qwen3-14B scale-ablation profile for the H200 evaluator. Keep the auxiliary
# embedding/compression snapshots and inference policy identical to the 8B
# main run so model scale is the only intended experimental variable.

HABITBENCH_Q14_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HABITBENCH_H_ROOT="${HABITBENCH_H_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang}"
export HABITBENCH_H_MODEL_ROOT="${HABITBENCH_H_MODEL_ROOT:-$HABITBENCH_H_ROOT/models/habitbench}"
export HABITBENCH_LLM_MODEL="$HABITBENCH_H_MODEL_ROOT/Qwen3-14B"
export HABITBENCH_SERVED_MODEL="Qwen3-14B"
export HABITBENCH_LLM_MODEL_ID="Qwen/Qwen3-14B"
export HABITBENCH_LLM_MODEL_REVISION="40c069824f4251a91eefaf281ebe4c544efd3e18"

# shellcheck disable=SC1091
source "$HABITBENCH_Q14_ENV_DIR/env.h.example.sh"

unset HABITBENCH_Q14_ENV_DIR
