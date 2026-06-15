#!/usr/bin/env bash
set -euo pipefail

# Download/cache the open-weight models used by Lumia experiments.
#
# Usage:
#   bash ./scripts/lumia/download_open_models.sh
#
# Override defaults:
#   HABITBENCH_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
#   HABITBENCH_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
#   bash ./scripts/lumia/download_open_models.sh

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

LLM_MODEL="${HABITBENCH_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EMBED_MODEL="${HABITBENCH_EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
OUT_DIR="${HABITBENCH_MODEL_MANIFEST_DIR:-./runs/lumia_manifests}"
MANIFEST="$OUT_DIR/model_download_manifest.json"
PYTHON_BIN="${PYTHON_BIN:-}"
export HABITBENCH_MODEL_DOWNLOAD_MANIFEST="${HABITBENCH_MODEL_DOWNLOAD_MANIFEST:-$MANIFEST}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "No python3 or python executable found on PATH" >&2
    exit 127
  fi
fi
export PYTHON_BIN

"$PYTHON_BIN" -m pip install -U "huggingface_hub[cli]>=0.34.0,<1.0"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" ./scripts/lumia/preflight_open_models.py \
  --out "$OUT_DIR/model_preflight_manifest.json"

"$PYTHON_BIN" ./scripts/lumia/download_open_models.py \
  --llm-model "$LLM_MODEL" \
  --embed-model "$EMBED_MODEL" \
  --out "$HABITBENCH_MODEL_DOWNLOAD_MANIFEST"

echo "Downloaded models:"
echo "  LLM:   $LLM_MODEL"
echo "  Embed: $EMBED_MODEL"
echo "  Manifest: $HABITBENCH_MODEL_DOWNLOAD_MANIFEST"
