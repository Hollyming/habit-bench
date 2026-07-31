#!/usr/bin/env bash

# Source this file inside a ClusterX job, or copy it to a private shared path
# and pass that path to the multi-GPU launcher with --env-file.

export HABITBENCH_PROJECT_ROOT="${HABITBENCH_PROJECT_ROOT:-/plm-shared/zhangjunming/Workspace/HABIT-bench}"
# Keep the MedMemoryBench adapters in the dependency-rich method environment,
# but run vLLM from a small dedicated environment matching the WJR runtime.
export PYTHON_BIN="${PYTHON_BIN:-/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark/bin/python}"
export HABITBENCH_VLLM_PYTHON="${HABITBENCH_VLLM_PYTHON:-/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark-vllm/bin/python}"
export HABITBENCH_MEDMEMORYBENCH_ROOT="${HABITBENCH_MEDMEMORYBENCH_ROOT:-$HABITBENCH_PROJECT_ROOT/third_party/medmemorybench}"
export HABITBENCH_SECOM_REPO="${HABITBENCH_SECOM_REPO:-$HABITBENCH_PROJECT_ROOT/third_party/official-baselines/vendor/SeCom}"

export HABITBENCH_LLM_MODEL="${HABITBENCH_LLM_MODEL:-/plm-shared/zhangjunming/Workspace/models/Qwen3-8B}"
export HABITBENCH_SERVED_MODEL="${HABITBENCH_SERVED_MODEL:-Qwen3-8B}"
export HABITBENCH_CHAT_TEMPLATE="${HABITBENCH_CHAT_TEMPLATE:-$HABITBENCH_PROJECT_ROOT/configs/chat_templates/qwen3_no_thinking.jinja}"
export HABITBENCH_EMBED_MODEL="${HABITBENCH_EMBED_MODEL:-/plm-shared/zhangjunming/Workspace/models/bge-m3}"
export HABITBENCH_EMBED_DIMS="${HABITBENCH_EMBED_DIMS:-1024}"
# WJR's fast layout reserves each visible GPU for vLLM and runs retrieval
# encoders in CPU-only adapter workers, avoiding GPU memory/compute contention.
export HABITBENCH_EMBED_DEVICE="${HABITBENCH_EMBED_DEVICE:-cpu}"
export HABITBENCH_ADAPTER_CUDA_VISIBLE_DEVICES="${HABITBENCH_ADAPTER_CUDA_VISIBLE_DEVICES:-}"
export HABITBENCH_ADAPTER_CPU_THREADS="${HABITBENCH_ADAPTER_CPU_THREADS:-2}"
export HABITBENCH_MED_USER_WORKERS="${HABITBENCH_MED_USER_WORKERS:-4}"
# WJR full-run profiles: independent-user parallelism helps these five
# methods, while LightMem and MIRIX become slower on complete lifelines.
export HABITBENCH_MEM0_USER_WORKERS="${HABITBENCH_MEM0_USER_WORKERS:-7}"
export HABITBENCH_AMEM_USER_WORKERS="${HABITBENCH_AMEM_USER_WORKERS:-7}"
export HABITBENCH_MEMOS_USER_WORKERS="${HABITBENCH_MEMOS_USER_WORKERS:-7}"
export HABITBENCH_MEMRL_USER_WORKERS="${HABITBENCH_MEMRL_USER_WORKERS:-7}"
export HABITBENCH_LETTA_USER_WORKERS="${HABITBENCH_LETTA_USER_WORKERS:-7}"
export HABITBENCH_LIGHTMEM_USER_WORKERS="${HABITBENCH_LIGHTMEM_USER_WORKERS:-1}"
export HABITBENCH_MIRIX_USER_WORKERS="${HABITBENCH_MIRIX_USER_WORKERS:-1}"
export HABITBENCH_LIGHTMEM_MODEL="${HABITBENCH_LIGHTMEM_MODEL:-/plm-shared/zhangjunming/Workspace/models/llmlingua-2-xlm-roberta-large-meetingbank}"
export HABITBENCH_SECOM_COMPRESSOR="${HABITBENCH_SECOM_COMPRESSOR:-$HABITBENCH_LIGHTMEM_MODEL}"
# full_memory selects the largest standard tier supported by MAX_MODEL_LEN.
# Set this to 8k/16k/32k/40k/64k/128k, or to custom together with
# HABITBENCH_MAX_INPUT_TOKENS. Optional RESERVED/MAX variables override the
# tier's history budget and normally should remain unset.
export HABITBENCH_CONTEXT_WINDOW_TIER="${HABITBENCH_CONTEXT_WINDOW_TIER:-auto}"

export HABITBENCH_GPU_MEMORY_UTIL="${HABITBENCH_GPU_MEMORY_UTIL:-0.85}"
export HABITBENCH_MAX_MODEL_LEN="${HABITBENCH_MAX_MODEL_LEN:-40960}"
export HABITBENCH_ENABLE_PREFIX_CACHING="${HABITBENCH_ENABLE_PREFIX_CACHING:-1}"
# MIRIX's local JSON-schema bridge requires compact structured output. Without
# disable_any_whitespace, xgrammar permits unbounded whitespace between bounded
# JSON fields and Qwen can waste the full completion budget before closing the
# object. The option is harmless for requests without response_format and also
# keeps other schema-constrained adapters finite.
export HABITBENCH_VLLM_EXTRA_ARGS="${HABITBENCH_VLLM_EXTRA_ARGS:---dtype bfloat16 --max-num-seqs 32 --reasoning-parser qwen3 --generation-config vllm --enable-auto-tool-choice --tool-call-parser hermes --default-chat-template-kwargs '{\"enable_thinking\": false}' --structured-outputs-config '{\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}' --attention-backend FLASH_ATTN}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
# Gate on aggregate decode throughput at MED_USER_WORKERS concurrent requests.
export HABITBENCH_VLLM_MIN_TOKENS_PER_SEC="${HABITBENCH_VLLM_MIN_TOKENS_PER_SEC:-60}"
export HABITBENCH_VLLM_BENCHMARK_TOKENS="${HABITBENCH_VLLM_BENCHMARK_TOKENS:-128}"
export HABITBENCH_VLLM_BENCHMARK_TIMEOUT_SEC="${HABITBENCH_VLLM_BENCHMARK_TIMEOUT_SEC:-120}"
export HABITBENCH_VLLM_BENCHMARK_CONCURRENCY="${HABITBENCH_VLLM_BENCHMARK_CONCURRENCY:-4}"
export HABITBENCH_MEMORY_LLM_MAX_TOKENS="${HABITBENCH_MEMORY_LLM_MAX_TOKENS:-4096}"
export HABITBENCH_PROGRESS_EVERY="${HABITBENCH_PROGRESS_EVERY:-25}"
export HABITBENCH_OFFICIAL_TIMEOUT_SEC="${HABITBENCH_OFFICIAL_TIMEOUT_SEC:-172800}"
export HABITBENCH_STRUCTURED_OUTPUT_MODE="${HABITBENCH_STRUCTURED_OUTPUT_MODE:-json_schema}"
export HABITBENCH_MEMORY_LLM_TEMPERATURE="${HABITBENCH_MEMORY_LLM_TEMPERATURE:-0.0}"
export HABITBENCH_MEMORY_LLM_SEED="${HABITBENCH_MEMORY_LLM_SEED:-42}"
export HABITBENCH_SERVER_READY_ATTEMPTS="${HABITBENCH_SERVER_READY_ATTEMPTS:-180}"
export HABITBENCH_SERVER_READY_SLEEP_SEC="${HABITBENCH_SERVER_READY_SLEEP_SEC:-2}"

export HF_HOME="${HF_HOME:-/plm-shared/zhangjunming/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/plm-shared/zhangjunming/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$VLLM_CACHE_ROOT/torch_inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$VLLM_CACHE_ROOT/triton}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-/plm-shared/zhangjunming/.cache/tiktoken}"
# The ClusterX base image exposes CUDA 13.2's system ptxas. Force the dedicated
# vLLM environment's CUDA 12.8 assembler for its Torch 2.10/Triton 3.6 runtime.
export TRITON_PTXAS_PATH="${HABITBENCH_TRITON_PTXAS_PATH:-/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark-vllm/lib/python3.10/site-packages/triton/backends/nvidia/bin/ptxas}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
