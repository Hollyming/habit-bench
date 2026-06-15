#!/usr/bin/env bash
set -euo pipefail

# Run Graphiti's LLM-backed add_episode path on the HABIT-Bench official subset.
# Requires an OpenAI-compatible /chat/completions endpoint, e.g. vLLM started
# with scripts/lumia/start_vllm_openai_server.sh.

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

DATA="${1:-${HABITBENCH_DATASET:-./runs/habit_bench_balanced_v0_3_official_subset_90}}"
OUT="${2:-${HABITBENCH_OFFICIAL_OUT:-$DATA/full_official_results}}"
METHOD="official_graphiti_full_llm_episode_kuzu"
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
  --results-dir "$OUT/graphiti_full_llm_episode_kuzu" \
  --command "bash ./scripts/run_full_official_subset_graphiti.sh $DATA $OUT" \
  --extra "run_id=$RUN_ID"

"$PYTHON_BIN" ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/graphiti_full_llm_episode_kuzu" \
  --method-name "$METHOD" \
  --timeout-sec "$OFFICIAL_TIMEOUT_SEC" \
  --command "python ./eval/official_adapters/official_graphiti_full_llm_adapter.py --input {input} --output {output} --topk 5 --continue-on-add-error" \
  --adapter-note "Official Graphiti add_episode LLM extraction/KG resolution path with Kuzu backend and OpenAIGenericClient for local OpenAI-compatible chat-completions endpoints; retrieval uses Graphiti.search_ edge cosine because local Kuzu BM25 full-text index is unavailable."

"$PYTHON_BIN" ./eval/collect_official_results.py \
  --dataset-dir "$DATA" \
  --results-dir "$OUT"

"$PYTHON_BIN" ./scripts/lumia/write_run_manifest.py \
  --out "$MANIFEST_DIR/${METHOD}_end_manifest.json" \
  --stage "${METHOD}_end" \
  --dataset-dir "$DATA" \
  --results-dir "$OUT/graphiti_full_llm_episode_kuzu" \
  --command "bash ./scripts/run_full_official_subset_graphiti.sh $DATA $OUT" \
  --extra "run_id=$RUN_ID"
