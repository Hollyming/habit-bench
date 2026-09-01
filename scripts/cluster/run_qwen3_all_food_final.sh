#!/usr/bin/env bash
set -euo pipefail

# One four-Replica RJob uses one 8-H200 node per Qwen3 scale.  Each model has
# its own immutable plan/output root, so a Replica can resume independently and
# the 8B Replica can run the supplementary controls after its main suite.

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MAIN_ROOT_PREFIX="${HABITBENCH_FOOD_MAIN_ROOT_PREFIX:-$PROJECT_ROOT/results/habit-h200-main-food-final-qwen3}"
SUPPLEMENTARY_ROOT="${HABITBENCH_FOOD_SUPPLEMENTARY_ROOT:-$PROJECT_ROOT/results/habit-h200-supplementary-food-final-qwen3-8b-v1}"
FOOD_FINAL_DATASET="${HABITBENCH_FOOD_FINAL_DATASET:-$PROJECT_ROOT/domain/food/food_habit_lifelines_final_check}"
H_EVAL="$PROJECT_ROOT/scripts/cluster/run_h_eval.sh"

REPLICA_INDEX="${NODE_RANK:-0}"
REPLICA_COUNT="${NODE_COUNT:-1}"
if ! [[ "$REPLICA_INDEX" =~ ^[0-9]+$ && "$REPLICA_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid RJob NODE_RANK/NODE_COUNT: $REPLICA_INDEX/$REPLICA_COUNT" >&2
  exit 1
fi
if [[ "$REPLICA_COUNT" != "4" ]]; then
  echo "This launcher requires exactly four RJob Replicas (4 x 8 H200); got $REPLICA_COUNT" >&2
  exit 1
fi

case "$REPLICA_INDEX" in
  0)
    MODEL_SLUG="4b"
    ENV_FILE="$PROJECT_ROOT/scripts/cluster/env.h.qwen3_4b.sh"
    PORT_BASE=8100
    ;;
  1)
    MODEL_SLUG="8b"
    # env.h.example is the frozen Qwen3-8B profile.
    ENV_FILE="$PROJECT_ROOT/scripts/cluster/env.h.example.sh"
    PORT_BASE=8200
    ;;
  2)
    MODEL_SLUG="14b"
    ENV_FILE="$PROJECT_ROOT/scripts/cluster/env.h.qwen3_14b.sh"
    PORT_BASE=8300
    ;;
  3)
    MODEL_SLUG="32b"
    ENV_FILE="$PROJECT_ROOT/scripts/cluster/env.h.qwen3_32b.sh"
    PORT_BASE=8400
    ;;
  *)
    echo "Replica index must be in [0, 3], got $REPLICA_INDEX" >&2
    exit 1
    ;;
esac

MAIN_ROOT="$MAIN_ROOT_PREFIX-$MODEL_SLUG-v1"
MAIN_PLAN="$MAIN_ROOT/shard_plan.tsv"
if [[ ! -f "$MAIN_PLAN" ]]; then
  echo "Missing pre-created $MODEL_SLUG main plan: $MAIN_PLAN" >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing H model environment: $ENV_FILE" >&2
  exit 1
fi
if [[ ! -d "$FOOD_FINAL_DATASET" ]]; then
  echo "Missing food final dataset: $FOOD_FINAL_DATASET" >&2
  exit 1
fi

BASE_JOB_ID="${JOB_ID:-zjm-qwen3-food-final-$$}"
echo "food-final main start: replica=$REPLICA_INDEX/4 model=Qwen3-$MODEL_SLUG plan=$MAIN_PLAN"
echo "food-final dataset: $FOOD_FINAL_DATASET"

# The RJob injects NODE_COUNT=4 for the four model workers.  Every model plan
# is intentionally a one-Replica local queue; override those variables only
# for this child so run_h_eval does not try to make four workers share a model
# plan.  The parent Replica still owns its independent 8 visible H200s.
NODE_RANK=0 NODE_COUNT=1 JOB_ID="${BASE_JOB_ID}-qwen3-${MODEL_SLUG}" \
  bash "$H_EVAL" \
    --plan "$MAIN_PLAN" \
    --gpus 8 \
    --env-file "$ENV_FILE" \
    --port-base "$PORT_BASE" \
    --expected-replicas 1

# Supplementary controls are evaluated on the same 8B node after its main
# suite.  Other model Replicas may still be finishing in parallel; no extra
# GPUs are requested.  Human audit is deliberately absent from this plan.
if [[ "$MODEL_SLUG" == "8b" ]]; then
  SUP_PLAN="$SUPPLEMENTARY_ROOT/shard_plan.tsv"
  if [[ ! -f "$SUP_PLAN" ]]; then
    echo "Missing pre-created 8B supplementary plan: $SUP_PLAN" >&2
    exit 1
  fi
  echo "food-final supplementary start: Qwen3-8B controls=no_memory,oracle_evidence,oracle_habit_state plan=$SUP_PLAN"
  NODE_RANK=0 NODE_COUNT=1 JOB_ID="${BASE_JOB_ID}-qwen3-8b-supp" \
    bash "$H_EVAL" \
      --plan "$SUP_PLAN" \
      --gpus 8 \
      --env-file "$PROJECT_ROOT/scripts/cluster/env.h.example.sh" \
      --port-base 8600 \
      --expected-replicas 1 \
      --post-supplementary-analysis
fi

echo "food-final Replica complete: model=Qwen3-$MODEL_SLUG main_root=$MAIN_ROOT"
