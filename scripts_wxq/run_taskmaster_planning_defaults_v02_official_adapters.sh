#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-/mnt/petrelfs/linzhouhan/xqwang/project/habit-bench}"
DATA="${1:-$PROJECT_ROOT/runs_wxq/taskmaster_planning_defaults_v0_2}"
OUT="${2:-$DATA/official_results}"
PYTHON_BIN="${HABIT_OFFICIAL_PYTHON:-/mnt/petrelfs/linzhouhan/xqwang/conda_envs/habit-official/bin/python}"
OFFICIAL_REPO_ROOT="${OFFICIAL_BASELINES_REPO_ROOT:-/mnt/petrelfs/linzhouhan/xqwang/project/official-baselines}"
SHIM_DIR="$PROJECT_ROOT/scripts_wxq/official_shims"
NLTK_DATA_DIR="$DATA/official_deps/nltk_data"

export AMEM_REPO="${AMEM_REPO:-$OFFICIAL_REPO_ROOT/a-mem}"
export SECOM_REPO="${SECOM_REPO:-$OFFICIAL_REPO_ROOT/SeCom}"
export OMEM_REPO="${OMEM_REPO:-$OFFICIAL_REPO_ROOT/O-Mem}"
export PYTHONPATH="$SHIM_DIR:$AMEM_REPO:$SECOM_REPO:$OMEM_REPO:${PYTHONPATH:-}"
export NLTK_DATA="$NLTK_DATA_DIR:${NLTK_DATA:-}"
export POSTHOG_DISABLED=true

cd "$PROJECT_ROOT"
mkdir -p "$OUT" "$NLTK_DATA_DIR"

"$PYTHON_BIN" ./eval/official_adapter_status.py \
  --out-dir "$DATA/official_adapter_status"

"$PYTHON_BIN" ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/mem0_infer_false_hf_qdrant" \
  --method-name official_mem0_infer_false_hf_qdrant \
  --command "$PYTHON_BIN ./eval/official_adapters/official_mem0_adapter.py --input {input} --output {output}" \
  --adapter-note "Official Mem0 Memory.add infer=False and Memory.search retrieval adapter."

"$PYTHON_BIN" ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/amem_search_agentic_no_evolution" \
  --method-name official_amem_search_agentic_no_evolution \
  --command "$PYTHON_BIN ./eval/official_adapters/official_amem_adapter.py --input {input} --output {output}" \
  --adapter-note "Official A-MEM add_note/search_agentic retrieval adapter without LLM evolution."

"$PYTHON_BIN" ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/secom_bm25_session" \
  --method-name official_secom_bm25_session \
  --command "$PYTHON_BIN ./eval/official_adapters/official_secom_adapter.py --input {input} --output {output}" \
  --adapter-note "Official SeCom.retrieve_external_memory session BM25 adapter."

"$PYTHON_BIN" ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/graphiti_kuzu_edge_cosine" \
  --method-name official_graphiti_kuzu_edge_cosine \
  --command "$PYTHON_BIN ./eval/official_adapters/official_graphiti_adapter.py --input {input} --output {output}" \
  --adapter-note "Official Graphiti Kuzu EntityEdge storage and edge-cosine search adapter."

"$PYTHON_BIN" ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/omem_retrieval_injected_memory" \
  --method-name official_omem_retrieval_injected_memory \
  --command "$PYTHON_BIN ./eval/official_adapters/official_omem_adapter.py --input {input} --output {output} --topn 12 --drop-threshold 0.0" \
  --adapter-note "Official O-Mem MemoryManager.retrieve_from_memory_soft_segmentation with injected visible sessions."

"$PYTHON_BIN" ./eval/collect_official_results.py \
  --dataset-dir "$DATA" \
  --results-dir "$OUT" \
  --out-dir "$OUT/collected"
