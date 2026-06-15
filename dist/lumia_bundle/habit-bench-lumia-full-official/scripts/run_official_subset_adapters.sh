#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

DATA="${1:-./runs/habit_bench_balanced_v0_3_official_subset_90}"
OUT="${2:-$DATA/official_results}"

mkdir -p "$OUT"

python ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/mem0_infer_false_hf_qdrant" \
  --method-name official_mem0_infer_false_hf_qdrant \
  --command "python ./eval/official_adapters/official_mem0_adapter.py --input {input} --output {output}" \
  --adapter-note "Official Mem0 Memory.add infer=False and Memory.search retrieval adapter."

python ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/amem_search_agentic_no_evolution" \
  --method-name official_amem_search_agentic_no_evolution \
  --command "python ./eval/official_adapters/official_amem_adapter.py --input {input} --output {output}" \
  --adapter-note "Official A-MEM add_note/search_agentic retrieval adapter without LLM evolution."

python ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/secom_bm25_session" \
  --method-name official_secom_bm25_session \
  --command "python ./eval/official_adapters/official_secom_adapter.py --input {input} --output {output}" \
  --adapter-note "Official SeCom.retrieve_external_memory session BM25 adapter."

python ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/graphiti_kuzu_edge_cosine" \
  --method-name official_graphiti_kuzu_edge_cosine \
  --command "python ./eval/official_adapters/official_graphiti_adapter.py --input {input} --output {output}" \
  --adapter-note "Official Graphiti Kuzu EntityEdge storage and edge-cosine search adapter."

python ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/omem_retrieval_injected_memory" \
  --method-name official_omem_retrieval_injected_memory \
  --command "python ./eval/official_adapters/official_omem_adapter.py --input {input} --output {output} --topn 12 --drop-threshold 0.0" \
  --adapter-note "Official O-Mem MemoryManager.retrieve_from_memory_soft_segmentation with injected visible sessions."

python ./eval/collect_official_results.py \
  --dataset-dir "$DATA" \
  --results-dir "$OUT"
