#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

METHOD="${1:?usage: scripts/run_two_domains.sh METHOD [OUTPUT_ROOT]}"
OUTPUT_ROOT="${2:-$PROJECT_ROOT/results}"

DATASETS=(
  "$PROJECT_ROOT/domain/food/food_habit_lifelines_stress"
  "$PROJECT_ROOT/domain/finance-software/habit_bench_multidogo_finance_software_long_hard_diverse_v0_5"
)

for dataset in "${DATASETS[@]}"; do
  bash "$PROJECT_ROOT/scripts/run_method.sh" \
    "$METHOD" \
    "$dataset" \
    "$OUTPUT_ROOT/$(basename "$dataset")/$METHOD"
done
