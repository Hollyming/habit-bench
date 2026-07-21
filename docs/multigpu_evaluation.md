# Multi-GPU Full Evaluation

This runbook executes memory methods over every user while preserving complete
per-user lifelines. It supports a single multi-GPU host and Slurm job arrays.

## Execution Unit

HABIT-Bench shards by sorted `user_id`, never by session or probe. For shard
index `i` of `N`, the evaluator selects `users[i::N]`. Every selected user's
complete visible lifeline is ingested chronologically by the memory method.

Each GPU task owns:

- one user shard;
- one method and one dataset;
- one local Qwen3-8B vLLM server;
- one independent method-native memory store;
- one result directory named `shard_XXX_of_NNN`.

The Mem0 adapter also assigns a shard-local `MEM0_DIR`. Mem0 1.0.2 otherwise
creates a shared telemetry migration store under `~/.mem0`, which is unsafe
when multiple GPU workers initialize concurrently.

The merger requires all shard indices, validates dataset hashes and probe
coverage, rejects duplicate predictions, and rescores the combined predictions
against the complete private key.

## Install On A New Cluster

```bash
git clone --recurse-submodules https://github.com/Hollyming/habit-bench.git
cd habit-bench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use shared model storage where possible. Adapt
`scripts/cluster/env.example.sh` to the cluster paths. Qwen3-8B, E5, and the
SeCom compressor should be local before an offline run. Hugging Face downloads
must be performed without HTTP/HTTPS proxy variables on Lumia.

## Select Methods Explicitly

The plan has no implicit controls. This command creates only Mem0 tasks:

```bash
python scripts/create_shard_plan.py \
  --methods mem0 \
  --datasets food,finance_software \
  --shards 8 \
  --output-root results/mem0_full \
  --plan results/mem0_full/shard_plan.tsv
```

To run several memory methods without controls:

```bash
--methods mem0,amem,graphiti,secom,omem
```

`no_memory` and `full_history` run only when their names are explicitly present
in `--methods`. The active raw-context control is named `full_history`; there is
no method named `full_memory`.

The task count is:

```text
number of methods x number of datasets x number of user shards
```

Do not request more shards than users: food has 30 users and
finance/software has 45.

## Slurm Job Array

The submitter creates the plan, submits one-GPU array tasks, and submits a CPU
merge job with an `afterok` dependency:

```bash
bash scripts/slurm/submit_sharded_suite.sh \
  --methods mem0 \
  --datasets food,finance_software \
  --shards 8 \
  --max-concurrent 8 \
  --partition GPU_PARTITION \
  --time 16:00:00 \
  --cpus 12 \
  --mem 96G \
  --port-base 8100 \
  --env-file /path/to/habitbench-cluster-env.sh \
  --output-root results/mem0_full
```

Choose a different `--port-base` for concurrent experiment arrays that may
share a node.

Use `--dry-run` to inspect the plan and `sbatch` commands. Use
`--no-merge-job` when the target cluster does not permit dependent CPU jobs;
merge manually after every shard succeeds:

```bash
python scripts/merge_shard_plan.py \
  --plan results/mem0_full/shard_plan.tsv
```

Completed shards are skipped when `metrics.json` exists. Set
`HABITBENCH_FORCE_RERUN=1` to replace a completed shard. Failed or preempted
array indices can therefore be resubmitted without repeating successful work.

## One Multi-GPU Node Without Slurm

Create the same plan, then launch one sequential worker per GPU:

```bash
python scripts/run_multigpu_plan.py \
  --plan results/mem0_full/shard_plan.tsv \
  --gpus 0,1,2,3,4,5,6,7 \
  --env-file /path/to/habitbench-cluster-env.sh \
  --port-base 8100

python scripts/merge_shard_plan.py \
  --plan results/mem0_full/shard_plan.tsv
```

Each worker processes its assigned task ids sequentially, so two vLLM servers
never share one GPU. Different workers use different localhost ports.

## Output Layout

```text
results/mem0_full/
  shard_plan.tsv
  food/mem0/
    shard_000_of_008/
    ...
    shard_007_of_008/
    merged/
      merge_manifest.json
      memory_contexts.jsonl
      predictions.jsonl
      scored_predictions.jsonl
      metrics.json
      metrics_by_group.csv
  finance_software/mem0/
    ...
```

Only `merged/metrics.json` is the complete-dataset score. A shard's `overall`
metric covers only that shard's users.

## Observed Mem0 Cost

On Lumia with one RTX 6000 Ada and Qwen3-8B:

| dataset | one-user lifeline | Mem0 write time |
| --- | ---: | ---: |
| food | 47 sessions | about 17 minutes |
| finance/software | 320 sessions | about 93 minutes |

These are pilot measurements, not guaranteed throughput. A sequential full
finance run would take roughly 70 GPU-hours. With eight balanced user shards,
the expected wall time is approximately 8-10 hours plus queue and server
startup time. Use at least a 12-16 hour per-shard limit and preserve logs.

## Fairness And Reporting

- Keep the Qwen model, thinking mode, temperature, retrieval budget, and method
  revision fixed across shards.
- Report total GPU-hours, per-user write latency, storage size, retrieved
  context tokens, and answer Accuracy.
- A-MEM, Graphiti, SeCom, O-Mem, and Mem0 are `official_adapted` integrations;
  do not describe them as unchanged paper configurations.
- Do not compare an individual shard or one-user pilot with a full-domain
  no-memory score.
