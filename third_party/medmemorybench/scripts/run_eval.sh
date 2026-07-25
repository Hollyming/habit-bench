#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Evaluation Framework"
echo "=========================================="

METHOD="${1:-letta_gpt-5.1}"
DATASET="${2:-locomo}"
shift 2 2>/dev/null || true

echo "Method: $METHOD"
echo "Dataset: $DATASET"
echo "=========================================="

python main.py -m "$METHOD" -d "$DATASET" "$@"