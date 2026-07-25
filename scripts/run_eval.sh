#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

METHOD="${1:?usage: scripts/run_eval.sh METHOD DATASET_DIR [OUTPUT_DIR] [eval args]}"
DATASET="${2:?usage: scripts/run_eval.sh METHOD DATASET_DIR [OUTPUT_DIR] [eval args]}"
OUTPUT="${3:-$PROJECT_ROOT/results/$(basename "$DATASET")/$METHOD}"
PYTHON_BIN="${PYTHON_BIN:-/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "${HABITBENCH_FORCE_RERUN:-0}" != "1" && -f "$OUTPUT/metrics.json" ]]; then
  echo "skip_completed method=$METHOD output=$OUTPUT"
  exit 0
fi

EXTRA_EVAL_ARGS=()

case "$METHOD" in
  no_memory)
    KIND="control"
    SOURCE="HABIT-Bench"
    REVISION="memory_context.v1"
    COMMAND="$PYTHON_BIN -m eval.controls --input {input} --output {output} --mode $METHOD"
    NOTE="Shared Qwen3-8B control with no historical or memory context."
    CONFIG_ARGS=()
    ;;
  full_memory|full_history)
    KIND="control"
    SOURCE="HABIT-Bench"
    REVISION="memory_context.v3"
    WINDOW_TIER="${HABITBENCH_CONTEXT_WINDOW_TIER:-auto}"
    MODEL_CONTEXT_TOKENS="${HABITBENCH_MAX_MODEL_LEN:-40960}"
    WINDOW_RESOLVER_ARGS=(
      --tier "$WINDOW_TIER"
      --model-context-tokens "$MODEL_CONTEXT_TOKENS"
    )
    WINDOW_ADAPTER_ARGS="--context-window-tier $WINDOW_TIER --model-context-tokens $MODEL_CONTEXT_TOKENS"
    if [[ -n "${HABITBENCH_MAX_INPUT_TOKENS:-}" ]]; then
      WINDOW_RESOLVER_ARGS+=(--custom-max-input-tokens "$HABITBENCH_MAX_INPUT_TOKENS")
      WINDOW_ADAPTER_ARGS+=" --custom-max-input-tokens $HABITBENCH_MAX_INPUT_TOKENS"
    fi
    if [[ -n "${HABITBENCH_FULL_MEMORY_RESERVED_TOKENS:-}" ]]; then
      WINDOW_RESOLVER_ARGS+=(--reserved-prompt-tokens "$HABITBENCH_FULL_MEMORY_RESERVED_TOKENS")
      WINDOW_ADAPTER_ARGS+=" --reserved-prompt-tokens $HABITBENCH_FULL_MEMORY_RESERVED_TOKENS"
    fi
    if [[ -n "${HABITBENCH_FULL_MEMORY_MAX_TOKENS:-}" ]]; then
      WINDOW_RESOLVER_ARGS+=(--max-history-tokens "$HABITBENCH_FULL_MEMORY_MAX_TOKENS")
      WINDOW_ADAPTER_ARGS+=" --max-history-tokens $HABITBENCH_FULL_MEMORY_MAX_TOKENS"
    fi
    for eval_arg in "${@:4}"; do
      if [[ "$eval_arg" == "--max-input-tokens" || "$eval_arg" == --max-input-tokens=* ]]; then
        echo "For full_memory, select the window with HABITBENCH_CONTEXT_WINDOW_TIER (or the custom-window environment variables), not --max-input-tokens." >&2
        exit 2
      fi
    done
    RESOLVED_MAX_INPUT_TOKENS=$(
      "$PYTHON_BIN" -m eval.context_windows \
        "${WINDOW_RESOLVER_ARGS[@]}" \
        --field max_input_tokens
    )
    COMMAND="$PYTHON_BIN -m eval.controls --input {input} --output {output} --mode $METHOD --tokenizer-path ${HABITBENCH_LLM_MODEL:-/plm-shared/zhangjunming/Workspace/models/Qwen3-8B} $WINDOW_ADAPTER_ARGS"
    EXTRA_EVAL_ARGS=(--max-input-tokens "$RESOLVED_MAX_INPUT_TOKENS")
    CONFIG_PATH="$PROJECT_ROOT/configs/methods/full_memory.yaml"
    CONFIG_ARGS=(
      --method-config-name "full_memory"
      --method-config-path "$CONFIG_PATH"
    )
    NOTE="Capacity-aware long-context control: auto/explicit window tier, all visible history when it fits, otherwise recent complete sessions under the resolved tokenizer budget. full_history is a compatibility alias."
    ;;
  mem0|amem|memos|memrl|lightmem|letta|mirix)
    case "$METHOD" in
      mem0) MED_CONFIG="mem0_qwen3-8b_adapted" ;;
      amem) MED_CONFIG="amem_qwen3-8b_adapted" ;;
      memos) MED_CONFIG="memos_qwen3-8b_adapted" ;;
      memrl) MED_CONFIG="memrl_qwen3-8b_adapted" ;;
      lightmem) MED_CONFIG="lightmem_qwen3-8b_adapted" ;;
      letta) MED_CONFIG="letta_qwen3-8b_adapted" ;;
      mirix) MED_CONFIG="mirix_qwen3-8b_adapted" ;;
    esac
    MED_ROOT="${HABITBENCH_MEDMEMORYBENCH_ROOT:-$PROJECT_ROOT/third_party/medmemorybench}"
    KIND="benchmark_reproduction"
    SOURCE="https://github.com/AQ-MedAI/MedMemoryBench"
    REVISION="6591eb3251402f26535846ea4a95f5b4478ae35a"
    COMMAND="$PYTHON_BIN -m eval.medmemorybench_adapters.structured_memory --input {input} --output {output} --med-repo $MED_ROOT --method-config $MED_CONFIG --user-workers ${HABITBENCH_MED_USER_WORKERS:-1} --progress-every ${HABITBENCH_PROGRESS_EVERY:-25}"
    CONFIG_PATH="$MED_ROOT/configs/method_config/$MED_CONFIG.yaml"
    if [[ ! -f "$CONFIG_PATH" ]]; then
      echo "Method config not found: $CONFIG_PATH" >&2
      exit 1
    fi
    CONFIG_ARGS=(
      --method-config-name "$MED_CONFIG"
      --method-config-path "$CONFIG_PATH"
    )
    NOTE="Vendored MedMemoryBench memory lifecycle with shared local BGE-M3 retrieval and retrieval-only output passed to the shared HABIT answerer."
    ;;
  graphiti)
    KIND="official_adapted"
    SOURCE="https://github.com/getzep/graphiti"
    REVISION="graphiti-core==0.29.2"
    COMMAND="$PYTHON_BIN -m eval.official_adapters.graphiti --input {input} --output {output}"
    CONFIG_PATH="$PROJECT_ROOT/configs/methods/graphiti_bge_m3_qwen3.yaml"
    CONFIG_ARGS=(
      --method-config-name "graphiti_bge_m3_qwen3"
      --method-config-path "$CONFIG_PATH"
    )
    NOTE="Official Graphiti add_episode/search_ lifecycle with a local Kuzu backend and documented edge-cosine/RRF retrieval adaptation."
    ;;
  secom)
    KIND="official_adapted"
    SOURCE="https://github.com/microsoft/SeCom"
    REVISION="1738e563b5dc7c51df762247e3d0379f1132ad23"
    COMMAND="$PYTHON_BIN -m eval.official_adapters.secom --input {input} --output {output}"
    CONFIG_PATH="$PROJECT_ROOT/configs/methods/secom_bge_m3_qwen3.yaml"
    CONFIG_ARGS=(
      --method-config-name "secom_bge_m3_qwen3"
      --method-config-path "$CONFIG_PATH"
    )
    NOTE="Official SeCom segmentation, LLMLingua compression and FAISS retrieval adapted to online chronological session ingestion."
    ;;
  omem)
    KIND="official_adapted"
    SOURCE="https://github.com/OPPO-PersonalAI/O-Mem"
    REVISION="46e131ac39af55d456304c61dfb881717044528e"
    COMMAND="$PYTHON_BIN -m eval.official_adapters.omem --input {input} --output {output}"
    CONFIG_PATH="$PROJECT_ROOT/configs/methods/omem_bge_m3_qwen3.yaml"
    CONFIG_ARGS=(
      --method-config-name "omem_bge_m3_qwen3"
      --method-config-path "$CONFIG_PATH"
    )
    NOTE="Official O-Mem SimpleMemory lifecycle with documented local JSON-output compatibility handling."
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    echo "Available: no_memory full_memory full_history mem0 amem memos memrl lightmem letta mirix graphiti secom omem" >&2
    exit 2
    ;;
esac

exec "$PYTHON_BIN" -m eval.run \
  --dataset-dir "$DATASET" \
  --output-dir "$OUTPUT" \
  --method-name "$METHOD" \
  --adapter-command "$COMMAND" \
  --implementation-kind "$KIND" \
  --implementation-source "$SOURCE" \
  --implementation-revision "$REVISION" \
  --adapter-note "$NOTE" \
  --timeout-sec "${HABITBENCH_OFFICIAL_TIMEOUT_SEC:-172800}" \
  "${CONFIG_ARGS[@]}" \
  "${@:4}" \
  "${EXTRA_EVAL_ARGS[@]}"
