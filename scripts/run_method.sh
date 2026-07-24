#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

METHOD="${1:?usage: scripts/run_method.sh METHOD DATASET_DIR [OUTPUT_DIR]}"
DATASET="${2:?usage: scripts/run_method.sh METHOD DATASET_DIR [OUTPUT_DIR]}"
OUTPUT="${3:-$PROJECT_ROOT/results/$(basename "$DATASET")/$METHOD}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$METHOD" in
  no_memory|full_history)
    KIND="control"
    SOURCE="HABIT-Bench"
    REVISION="memory_context.v1"
    COMMAND="$PYTHON_BIN -m eval.official_adapters.controls --input {input} --output {output} --mode $METHOD"
    NOTE="Shared Qwen3-8B control; memory context mode=$METHOD."
    ;;
  mem0)
    KIND="official_adapted"
    SOURCE="https://github.com/mem0ai/mem0"
    REVISION="mem0ai==1.0.2"
    COMMAND="$PYTHON_BIN -m eval.official_adapters.mem0 --input {input} --output {output} --topk 5 --threshold 0.0"
    NOTE="Official Memory.add(infer=True) and Memory.search; no benchmark-specific extraction prompt."
    ;;
  amem)
    KIND="official_adapted"
    SOURCE="https://github.com/agiresearch/a-mem"
    REVISION="ceffb860f0712bbae97b184d440df62bc910ca8d"
    COMMAND="$PYTHON_BIN -m eval.official_adapters.amem --input {input} --output {output} --topk 5"
    NOTE="Official AgenticMemorySystem add_note/search_agentic path."
    ;;
  graphiti)
    KIND="official_adapted"
    SOURCE="https://github.com/getzep/graphiti"
    REVISION="graphiti-core==0.29.2"
    COMMAND="$PYTHON_BIN -m eval.official_adapters.graphiti_sharded --input {input} --output {output} --topk 5 --shard-count ${HABITBENCH_GRAPHITI_SHARDS:-4}"
    NOTE="Official add_episode/search_ APIs with local Kuzu and documented retrieval adaptation."
    ;;
  secom)
    KIND="official_adapted"
    SOURCE="https://github.com/microsoft/SeCom"
    REVISION="1738e563b5dc7c51df762247e3d0379f1132ad23"
    COMMAND="env CUDA_VISIBLE_DEVICES=${HABITBENCH_METHOD_CUDA_VISIBLE_DEVICES:-0} $PYTHON_BIN -m eval.official_adapters.secom --input {input} --output {output} --topk 5 --compress-rate 0.9"
    NOTE="Official SeCom modules adapted to chronological per-session ingestion."
    ;;
  omem)
    KIND="official_adapted"
    SOURCE="https://github.com/OPPO-PersonalAI/O-Mem"
    REVISION="46e131ac39af55d456304c61dfb881717044528e"
    COMMAND="$PYTHON_BIN -m eval.official_adapters.omem_sharded --input {input} --output {output} --topn 12 --drop-threshold 0.0 --shard-count ${HABITBENCH_OMEM_SHARDS:-4}"
    NOTE="Official SimpleMemory lifecycle with documented local-backend compatibility patches."
    ;;
  medmemorybench_mem0|medmemorybench_amem|medmemorybench_memos|medmemorybench_memrl|medmemorybench_lightmem|medmemorybench_letta|medmemorybench_mirix)
    SHORT_NAME="${METHOD#medmemorybench_}"
    case "$SHORT_NAME" in
      mem0) MED_CONFIG="mem0_qwen3-8b_adapted" ;;
      amem) MED_CONFIG="amem_qwen3-8b_adapted" ;;
      memos) MED_CONFIG="memos_qwen3-8b_adapted" ;;
      memrl) MED_CONFIG="memrl_qwen3-8b_adapted" ;;
      lightmem) MED_CONFIG="lightmem_qwen3-8b_adapted" ;;
      letta) MED_CONFIG="letta_qwen3-8b_adapted" ;;
      mirix) MED_CONFIG="mirix_qwen3-8b_adapted" ;;
    esac
    MED_ROOT="${HABITBENCH_MEDMEMORYBENCH_ROOT:-$PROJECT_ROOT/../MedMemoryBench}"
    KIND="benchmark_reproduction"
    SOURCE="https://github.com/AQ-MedAI/MedMemoryBench"
    REVISION="fe44743d72b6118d96e793891630a259ddfc8b45"
    COMMAND="$PYTHON_BIN -m eval.medmemorybench_adapters.structured_memory --input {input} --output {output} --med-repo $MED_ROOT --method-config $MED_CONFIG"
    NOTE="Vendored MedMemoryBench source implementation with retrieval-only output passed to the shared HABIT answerer."
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    echo "Available: no_memory full_history mem0 amem graphiti secom omem medmemorybench_{mem0,amem,memos,memrl,lightmem,letta,mirix}" >&2
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
  "${@:4}"
