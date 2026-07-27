# Multi-GPU evaluation with ClusterX

HABIT-Bench parallelizes by user. A shard owns complete user lifelines, so
method state never crosses workers and DDP synchronization is unnecessary.

## Active execution model

On one ClusterX node:

1. `create_shard_plan.py` creates deterministic user shards for every selected
   method/domain pair.
2. `run_multigpu_plan.py` starts one persistent Qwen3-8B vLLM server per GPU.
3. One method/domain group runs at a time; its user shards run concurrently
   across the persistent GPU workers.
4. GPUs are primarily reserved for persistent vLLM processes. Most adapter
   processes receive `CUDA_VISIBLE_DEVICES=""` and run BGE-M3 on CPU; MIRIX
   and SeCom retain their native CUDA paths and are explicitly bound to the
   same worker GPU.
5. MedMemoryBench shards preserve chronological execution inside one user but
   process independent users concurrently. Frozen method profiles use up to 7
   workers for Mem0/A-MEM/MemOS/MemRL/Letta, and 1 for
   LightMem/MIRIX.
6. Graphiti uses the same safe boundary: up to 4 independent user processes
   per GPU, one isolated Kuzu store per process, and one shared vLLM endpoint.
   Each user's official `add_episode` chain remains chronological; bulk or
   concurrent same-user ingestion is not used.
7. `merge_shard_plan.py` checks shard coverage and dataset/config consistency,
   merges predictions, rescoring the complete domain.

The Graphiti concurrency profile passed a real Food-v2 acceptance gate on
2026-07-27. Four workers each completed 25 episodes in at most 2,021.2 seconds,
with no fatal/add failure. The previous serial implementation required
7,175.4–7,538.3 seconds for the same aggregate 100-episode milestone across
six comparable shards (mean 7,361.9 seconds), giving a measured 3.64x mean
wall-clock speedup on one A800. The gate was stopped after this fixed-work
milestone; it is an efficiency validation, not a benchmark result.

Running groups sequentially makes method/domain wall-clock measurements
interpretable and avoids repeatedly starting vLLM for every shard.

## Environment

The job uses these Conda environments:

```text
/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark
/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark-vllm
/plm-shared/zhangjunming/miniconda3/envs/clusterx
```

`habitbenchmark` runs adapters and scoring. `habitbenchmark-vllm` is a
dedicated WJR-aligned serving runtime and is checked before GPU allocation is
used:

```text
Python 3.10.20
vLLM 0.17.1
PyTorch 2.10.0+cu128
Triton 3.6.0
Transformers 4.57.6
FlashInfer 0.6.4
xgrammar 0.1.29
```

Default model paths are recorded in `scripts/cluster/env.example.sh`:

```text
Qwen3-8B: /plm-shared/zhangjunming/Workspace/models/Qwen3-8B
BGE-M3:   /plm-shared/zhangjunming/Workspace/models/bge-m3
```

The environment also pins offline Hugging Face mode and
`TIKTOKEN_CACHE_DIR=/plm-shared/zhangjunming/.cache/tiktoken`. The latter
contains the `o200k_base` fallback used by the vendored base class for the
Qwen model name; the submit preflight rejects a missing cache instead of
allowing an evaluation job to attempt a network download.

The current ClusterX image exposes a CUDA 13.2 system `ptxas`, whereas the
dedicated vLLM environment uses PyTorch cu128 and Triton 3.6.
`TRITON_PTXAS_PATH` therefore pins that environment's bundled CUDA 12.8
assembler; this path is checked before the eight servers are started.

`HABITBENCH_CHAT_TEMPLATE` points vLLM to
`configs/chat_templates/qwen3_no_thinking.jinja`. This makes non-thinking the
default for all MedMemoryBench method clients, including clients that cannot
send Qwen-specific `chat_template_kwargs`, and avoids truncated structured
memory writes without changing the native memory algorithms.

The default structured-output backend is xgrammar with
`disable_any_whitespace=true`. MIRIX's JSON-tool bridge already bounds fields
and arrays; this matching WJR launch option prevents unbounded whitespace
between those fields from consuming the complete 8,192-token child budget.
Requests without a structured response format are unaffected.

`VLLM_BATCH_INVARIANT=1` and `--attention-backend FLASH_ATTN` keep outputs
stable when several user workers share one endpoint. After startup, every
server runs one single-stream sample plus a representative four-request
sample. The job records both rates and refuses to begin a
method group when aggregate decode throughput is below
`HABITBENCH_VLLM_MIN_TOKENS_PER_SEC` (60 by default).

## Submit

The single supported submit entry is:

```bash
cd /plm-shared/zhangjunming/Workspace/HABIT-bench

bash scripts/submit_clusterx.sh \
  --methods mem0,amem,memos,memrl,lightmem,letta,mirix,graphiti,secom,omem \
  --datasets food,finance,software \
  --shards 8 \
  --gpus 8 \
  --output-root results/habit_bge_m3_v1
```

This submits through `clusterx run` to `cluster-t` /
`queue-t-reserved-plm`. Use `--dry-run` to create the plan and inspect the exact
submission without creating a ClusterX job.
For endpoint gates only, `--max-users N --max-probes N` creates an explicitly
recorded smoke subset; those outputs are not formal full-dataset results.

Controls are never added implicitly:

```bash
bash scripts/submit_clusterx.sh \
  --methods no_memory,full_memory \
  --shards 8 \
  --gpus 8 \
  --output-root results/controls_bge_m3_v1
```

`full_history` is accepted as a backward-compatible alias of `full_memory`.
The long-context control defaults to
`HABITBENCH_CONTEXT_WINDOW_TIER=auto`, which selects the largest standard
8k/16k/32k/40k/64k/128k tier supported by `HABITBENCH_MAX_MODEL_LEN`. Each tier
reserves prompt space and uses the remainder for history. Use
`HABITBENCH_CONTEXT_WINDOW_TIER=custom` with
`HABITBENCH_MAX_INPUT_TOKENS` for a non-standard model window.
`HABITBENCH_FULL_MEMORY_MAX_TOKENS` remains an explicit history-budget
override for ablations.
Prefix caching is enabled by default (`HABITBENCH_ENABLE_PREFIX_CACHING=1`) so
repeated probes for the same user can reuse their identical history prefix.

After preemption, resubmit the same output root. Completed shards are skipped.
Use `--force-plan` only when intentionally replacing the plan and
`HABITBENCH_FORCE_RERUN=1` only when intentionally replacing completed shard
outputs.

## Output and timing

```text
results/habit_bge_m3_v1/
├── shard_plan.tsv
├── shard_plan.manifest.json
├── clusterx_submit.log
├── suite_runtime.json
├── evaluation_summary.json
├── vllm_logs/
├── food/<method>/
│   ├── shard_000_of_008/
│   │   ├── run_manifest.json
│   │   ├── worker_runtime.json
│   │   └── ...
│   └── merged/
│       ├── merge_manifest.json
│       ├── metrics.json
│       └── ...
├── finance/<method>/
└── software/<method>/
```

Timing fields have distinct meanings:

- `run_manifest.execution.wall_clock_sec`: evaluator wall time for one shard;
- `worker_runtime.wall_clock_sec`: outer task wall time on its GPU worker;
- `merge_manifest.timing.shard_wall_clock_sum_sec`: total shard compute;
- `merge_manifest.timing.shard_wall_clock_max_sec`: ideal concurrent lower
  bound;
- `suite_runtime.groups[].wall_clock_sec`: observed end-to-end wall time for a
  method/domain group;
- `suite_runtime.servers[].throughput_gate`: single-stream and concurrent
  aggregate decode rate for each GPU;
- `suite_runtime.wall_clock_sec`: complete node job runtime, including one-time
  vLLM startup;
- `evaluation_summary.json`: score, shard count, config snapshot and timing for
  every method/domain pair.

Only `merged/metrics.json` is a complete-domain score. Partial shard metrics
must not be compared with complete-domain results.
