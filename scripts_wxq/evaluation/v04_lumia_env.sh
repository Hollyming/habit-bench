#!/usr/bin/env bash
# Runtime settings for the finalized v0.4 evaluation on Lumia Slurm GPU nodes.
# This file contains no credentials and may be sourced by every shard task.

export HABITBENCH_PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-/home/xqwang/project/habit-bench}"
export PYTHON_BIN="${PYTHON_BIN:-/home/xqwang/conda_envs/habit-official/bin/python3}"
export HABITBENCH_LLM_MODEL="${HABITBENCH_LLM_MODEL:-/data1/public/hf/Qwen/Qwen3-8B}"
export HABITBENCH_SERVED_MODEL="${HABITBENCH_SERVED_MODEL:-habitbench-qwen3-8b}"
export HABITBENCH_EMBED_MODEL="${HABITBENCH_EMBED_MODEL:-/home/jmzhang/models/e5-base-v2}"
export HABITBENCH_EMBED_DIMS="${HABITBENCH_EMBED_DIMS:-768}"
export HABITBENCH_SECOM_EMBED_MODEL="${HABITBENCH_SECOM_EMBED_MODEL:-$HABITBENCH_EMBED_MODEL}"
export HABITBENCH_SECOM_COMPRESSOR="${HABITBENCH_SECOM_COMPRESSOR:-/home/jmzhang/models/llmlingua-2-xlm-roberta-large-meetingbank}"
export HABITBENCH_GPU_MEMORY_UTIL="${HABITBENCH_GPU_MEMORY_UTIL:-0.82}"
export HABITBENCH_MAX_MODEL_LEN="${HABITBENCH_MAX_MODEL_LEN:-40960}"
export HABITBENCH_MEMORY_LLM_MAX_TOKENS="${HABITBENCH_MEMORY_LLM_MAX_TOKENS:-4096}"
export HABITBENCH_OFFICIAL_TIMEOUT_SEC="${HABITBENCH_OFFICIAL_TIMEOUT_SEC:-172800}"
export HABITBENCH_PROGRESS_EVERY="${HABITBENCH_PROGRESS_EVERY:-25}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
