# Multi-GPU evaluation with ClusterX or H RJob

HABIT-Bench parallelizes by user. A shard owns complete user lifelines, so
method state never crosses workers and DDP synchronization is unnecessary.
The execution model is shared by ClusterX and H RJob; only the scheduler,
persistent paths, image and resource declaration differ.

## Active execution model

On one single scheduler node:

1. `create_shard_plan.py` creates deterministic user shards for every selected
   method/domain pair.
2. `run_multigpu_plan.py` starts one persistent Qwen3-8B vLLM server per GPU.
3. Every persistent GPU worker repeatedly claims the next shard from one flat
   task queue. On multi-Replica H jobs the queue lives on GPFS and atomically
   creates one directory per task, so either node can claim any still-unassigned
   shard but no task can have two owners.
4. GPUs are primarily reserved for persistent vLLM processes. Most adapter
   processes receive `CUDA_VISIBLE_DEVICES=""` and run BGE-M3 on CPU; MIRIX
   and SeCom retain their native CUDA paths and are explicitly bound to the
   same worker GPU.
5. MedMemoryBench shards preserve chronological execution inside one user but
   process independent users concurrently. Frozen method profiles use up to 7
   workers for Mem0/A-MEM/MemOS/MemRL/Letta, and 1 for
   LightMem/MIRIX.
6. `merge_shard_plan.py` checks shard coverage and dataset/config consistency,
   merges predictions, rescoring the complete domain.

There is no method/domain barrier: when one GPU finishes a fast shard it can
start the next plan row while a slower shard from the previous group continues.
This removes node-to-node straggler waits without introducing DDP/NCCL, because
evaluation shards have no model gradients or shared method state. Group timing
is therefore an observed first-claim-to-last-completion span, while per-shard
timing remains the unit for method efficiency analysis.

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

Following MIRIX's upstream vLLM integration, the server keeps
`--enable-auto-tool-choice --tool-call-parser hermes` but does not install a
reasoning parser. Since this protocol disables thinking, a reasoning parser is
both unnecessary and unsafe for MIRIX's local `response_format` bridge: it can
move the constrained JSON away from `message.content`. MIRIX preflight rejects
profiles that reintroduce one.

The default structured-output backend is xgrammar with
`disable_any_whitespace=true` and `disable_fallback=true`. MIRIX's local bridge
first selects one memory tool with a tiny schema and then generates only that
tool's exact argument schema. The first option prevents unbounded whitespace
from consuming the complete 8,192-token child budget; the second prevents an
unsupported constraint from silently degrading to unconstrained text. Selector
and arguments are sequential within one child call, while independent official
MIRIX memory children remain concurrent. If local serving still ends a JSON
object early, the bridge reuses MIRIX's official tolerant parser
(`json.loads` / `demjson3` / `json-repair`) and then applies the same strict
tool schema before execution. For a parser-repaired truncated object only,
unknown fields are projected away when every declared required field is already
present. If truncation starts one more object in an array, the bridge keeps a
schema-valid nonempty prefix and discards only that incomplete final item when
the shortened array still satisfies its schema. Complete invalid JSON, a lone
incomplete item, an invalid prefix, and declared-field type errors are still
rejected. This covers the upstream `tree_path` drift documented in
[MIRIX issue #103](https://github.com/Mirix-AI/MIRIX/issues/103) without changing
its tool contract. Corrective generations use
deterministic per-attempt seeds and light sampling, so a retry does not replay
identical malformed bytes. Tool validation, execution, storage, and child
lifecycle remain unchanged. Requests without a structured response format are
unaffected.

`VLLM_BATCH_INVARIANT=1` and `--attention-backend FLASH_ATTN` keep outputs
stable when several user workers share one endpoint. After startup, every
server runs one single-stream sample plus a representative four-request
sample. The job records both rates and refuses to begin a
method group when aggregate decode throughput is below
`HABITBENCH_VLLM_MIN_TOKENS_PER_SEC` (60 by default).

## Submit with ClusterX

The single supported submit entry is:

```bash
cd /plm-shared/zhangjunming/Workspace/HABIT-bench

bash scripts/submit_clusterx.sh \
  --methods full_memory,mem0,amem,memos,memrl,lightmem,letta,mirix,secom \
  --datasets food,finance,software,travel \
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

`full_memory` is the online compact-history control; `full_history` is the
separate raw recency-truncation control. Both default to
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
The checkpoint granularity is one complete user shard, not a session or probe:
the runner writes one atomic `worker_runtime.json` only after the run manifest,
contexts, predictions and metrics all succeed. An incomplete/failed shard is
removed before rerun so partially written memory-backend state is never
ingested twice and checkpoint writes stay infrequent.
Use `--force-plan` only when intentionally replacing the plan and
`HABITBENCH_FORCE_RERUN=1` only when intentionally replacing completed shard
outputs.

The launch queue is scoped to one coordinator, so a second RJob can still
target the same output root. Before checkpoint inspection or partial-output
cleanup, every worker therefore acquires the persistent shard lock in
`.habitbench-shard-locks/` and holds it through final marker publication. A
second coordinator waits, then revalidates the checkpoint after acquiring the
lock. POSIX locks are released by the kernel when a worker is terminated, so a
preemption cannot leave a permanent lock claim.

For smoke plans, `--max-users` and `--max-probes` define one global ordered
dataset prefix before users are split into shards. Therefore, the disjoint
shard union is exactly the unsharded subset recorded in the plan manifest;
`max-probes` is not multiplied by the shard count.

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

## Submit with H RJob

The H launcher supports 4 or 8 H200 GPUs per Replica. Multi-Replica runs use
`JOB_ID` to create one launch-scoped GPFS task queue; `NODE_RANK` identifies the
claim owner but does not statically partition shard indices. Atomic GPFS
directory creation serializes each task claim, and each plan row is issued once
per launch. The separate shard-output lock prevents different launches from
mutating the same persistent shard concurrently. No DDP/NCCL synchronization is used. Per-Replica runtime/log files
avoid shared writes, and an atomic merge-claim directory elects the Replica that
performs the global merge. The launcher keeps managed spot/reserved/idle parameters separate and
verifies GPU model names inside each worker before vLLM starts. See
[`h_cluster_evaluation.md`](h_cluster_evaluation.md) for persistent environment
layout, creator identity requirements, commands, and interruption recovery.
