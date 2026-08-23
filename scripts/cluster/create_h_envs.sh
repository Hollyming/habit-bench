#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
ENV_ROOT="${HABITBENCH_H_ENV_ROOT:-/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/envs}"
CONDA_CHANNEL="${HABITBENCH_CONDA_CHANNEL:-conda-forge}"
PIP_INDEX_URL="${HABITBENCH_PIP_INDEX_URL:-https://pypi.org/simple}"
PYTORCH_INDEX_URL="${HABITBENCH_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-}"
INSTALL_METHOD=1
INSTALL_VLLM=1

usage() {
  cat <<'EOF'
Usage: scripts/cluster/create_h_envs.sh [options]
  --conda-bin PATH       Conda executable (default: /root/miniconda3/bin/conda)
  --env-root PATH        Persistent GPFS environment root
  --method-only          Create/update only the Python 3.11 method environment
  --vllm-only            Create/update only the Python 3.10 vLLM environment

Environment overrides:
  HABITBENCH_CONDA_CHANNEL   Conda channel, default conda-forge
  HABITBENCH_PIP_INDEX_URL  Official Python index, default https://pypi.org/simple
  HABITBENCH_PYTORCH_INDEX_URL
                              Official PyTorch CUDA index, default cu128
  TIKTOKEN_CACHE_DIR         Persistent offline tokenizer cache
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-bin) CONDA_BIN="${2:?missing value for --conda-bin}"; shift 2 ;;
    --env-root) ENV_ROOT="${2:?missing value for --env-root}"; shift 2 ;;
    --method-only) INSTALL_METHOD=1; INSTALL_VLLM=0; shift ;;
    --vllm-only) INSTALL_METHOD=0; INSTALL_VLLM=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$(dirname "$ENV_ROOT")/.cache/tiktoken}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Conda executable is unavailable: $CONDA_BIN" >&2
  exit 1
fi
if [[ "$ENV_ROOT" != /mnt/shared-storage-* ]]; then
  echo "--env-root must be persistent H storage under /mnt/shared-storage-*: $ENV_ROOT" >&2
  exit 2
fi
mkdir -p "$ENV_ROOT"

install_environment() {
  local name="$1"
  local python_version="$2"
  local requirements_file="$3"
  local extra_index_url="${4:-}"
  local prefix="$ENV_ROOT/$name"
  local python_bin="$prefix/bin/python"

  if [[ ! -x "$python_bin" ]]; then
    "$CONDA_BIN" create -y \
      --override-channels \
      -c "$CONDA_CHANNEL" \
      -p "$prefix" \
      "python=$python_version" \
      pip=26.1.2 \
      setuptools=83.0.0 \
      wheel=0.47.0
  fi

  local actual_python
  actual_python="$($python_bin -c 'import platform; print(platform.python_version())')"
  if [[ "$actual_python" != "$python_version" ]]; then
    echo "$prefix has Python $actual_python; expected exactly $python_version" >&2
    exit 1
  fi

  local pip_args=(
    --no-build-isolation
    --timeout 120
    --retries 10
    --index-url "$PIP_INDEX_URL"
  )
  if [[ -n "$extra_index_url" ]]; then
    pip_args+=(--extra-index-url "$extra_index_url")
  fi
  pip_args+=(-r "$PROJECT_ROOT/$requirements_file")

  # Ignore both pip configuration files and inherited index environment
  # variables. Network routing is provided by the caller's proxy_on/proxy_off;
  # package resolution uses only the official indexes above.
  env \
    -u PIP_INDEX_URL \
    -u PIP_EXTRA_INDEX_URL \
    -u PIP_TRUSTED_HOST \
    PIP_CONFIG_FILE=/dev/null \
    "$python_bin" -m pip install "${pip_args[@]}" \
    2>&1 | tee "$prefix/pip-install.log"
  "$python_bin" -m pip check | tee "$prefix/pip-check.log"
  echo "ready environment=$name python=$actual_python prefix=$prefix"
}

prepare_tiktoken_cache() {
  local python_bin="$ENV_ROOT/habitbenchmark/bin/python"
  mkdir -p "$TIKTOKEN_CACHE_DIR"
  TIKTOKEN_CACHE_DIR="$TIKTOKEN_CACHE_DIR" "$python_bin" - <<'PY'
import tiktoken

for encoding_name in ("o200k_base", "cl100k_base", "gpt2"):
    encoding = tiktoken.get_encoding(encoding_name)
    print(f"ready tiktoken={encoding_name} vocab={encoding.n_vocab}")
PY
  for cache_key in \
    fb374d419588a4632f3f557e76b4b70aebbca790 \
    9b5ad71b2ce5302211f9c61530b329a4922fc6a4 \
    6d1cbeee0f20b3d9449abfede4726ed8212e3aee \
    6c7ea1a7e38e3a7f062df639a5b80947f075ffe6
  do
    if [[ ! -s "$TIKTOKEN_CACHE_DIR/$cache_key" ]]; then
      echo "Failed to populate tiktoken cache: $TIKTOKEN_CACHE_DIR/$cache_key" >&2
      exit 1
    fi
  done
}

if [[ "$INSTALL_METHOD" == "1" ]]; then
  install_environment habitbenchmark 3.11.15 requirements.txt
  prepare_tiktoken_cache
fi
if [[ "$INSTALL_VLLM" == "1" ]]; then
  install_environment \
    habitbenchmark-vllm \
    3.10.20 \
    requirements-vllm.txt \
    "$PYTORCH_INDEX_URL"
fi
