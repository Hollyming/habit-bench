#!/usr/bin/env bash
set -euo pipefail

# Run Mem0's LLM-backed official memory path on the HABIT-Bench official subset.
# Requires an OpenAI-compatible endpoint, e.g. vLLM started with
# scripts/lumia/start_vllm_openai_server.sh.

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

DATA="${1:-${HABITBENCH_DATASET:-./runs/habit_bench_balanced_v0_3_official_subset_90}}"
OUT="${2:-${HABITBENCH_OFFICIAL_OUT:-$DATA/full_official_results}}"
METHOD="official_mem0_full_llm_openai"
RUN_ID="${HABITBENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MANIFEST_DIR="${HABITBENCH_RUN_MANIFEST_DIR:-$OUT/run_manifests/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OFFICIAL_TIMEOUT_SEC="${HABITBENCH_OFFICIAL_TIMEOUT_SEC:-21600}"

mkdir -p "$OUT"
mkdir -p "$MANIFEST_DIR"

"$PYTHON_BIN" ./scripts/lumia/write_run_manifest.py \
  --out "$MANIFEST_DIR/${METHOD}_start_manifest.json" \
  --stage "${METHOD}_start" \
  --dataset-dir "$DATA" \
  --results-dir "$OUT/mem0_full_llm_openai" \
  --command "bash ./scripts/run_full_official_subset_mem0.sh $DATA $OUT" \
  --extra "run_id=$RUN_ID"

"$PYTHON_BIN" ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/mem0_full_llm_openai" \
  --method-name "$METHOD" \
  --timeout-sec "$OFFICIAL_TIMEOUT_SEC" \
  --command "python ./eval/official_adapters/official_mem0_full_llm_adapter.py --input {input} --output {output} --topk 5 --threshold 0.0" \
  --adapter-note "Official Mem0 Memory.add infer=True path with OpenAI-compatible local LLM endpoint for fact extraction/update, followed by official Memory.search retrieval. The answer head remains HABIT-Bench lexical scoring over retrieved memories."

"$PYTHON_BIN" ./eval/collect_official_results.py \
  --dataset-dir "$DATA" \
  --results-dir "$OUT"

"$PYTHON_BIN" ./scripts/lumia/write_run_manifest.py \
  --out "$MANIFEST_DIR/${METHOD}_end_manifest.json" \
  --stage "${METHOD}_end" \
  --dataset-dir "$DATA" \
  --results-dir "$OUT/mem0_full_llm_openai" \
  --command "bash ./scripts/run_full_official_subset_mem0.sh $DATA $OUT" \
  --extra "run_id=$RUN_ID"
