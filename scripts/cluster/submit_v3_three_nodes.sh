#!/usr/bin/env bash
set -euo pipefail

# Compatibility alias only.  The old three-domain ClusterX/A800 launcher has
# been retired; current supplementary evaluation is four-domain H200 RJob.
PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ "${1:-}" == "--resume-existing" ]]; then
  # The current shard runner validates and reuses successful persistent
  # checkpoints automatically, so no separate resume mode is needed.
  shift
fi
echo "warning: submit_v3_three_nodes.sh is deprecated; forwarding to the current four-domain H-cluster supplementary launcher" >&2
exec bash "$PROJECT_ROOT/scripts/cluster/submit_qwen3_8b_supplementary.sh" "$@"
