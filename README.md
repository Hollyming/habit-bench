# HABIT-Bench

HABIT-Bench evaluates whether a longitudinal assistant memory system can learn
and apply user habits over long interaction histories. The active evaluator is
an end-to-end multiple-choice protocol. Retrieval similarity is never used to
select an answer.

## Active Datasets

| dataset | domains | users | sessions | probes | public lifeline format |
| --- | --- | ---: | ---: | ---: | --- |
| `domain/food/food_habit_lifelines_stress` | food | 30 | 1,410 | 1,260 | one session per JSONL row |
| `domain/finance-software/habit_bench_multidogo_finance_software_long_hard_diverse_v0_5` | finance, software | 45 | 14,400 | 810 | one user with nested sessions per JSONL row |

Both formats are normalized by `eval/core/dataset.py`. A memory method receives
only public sessions, public probe text and choices, and the history cutoff.
Gold labels and hidden habit annotations are loaded only by the scorer.

## Evaluation Contract

For each user and probe, the formal pipeline is:

1. Feed visible sessions to the memory method in chronological order.
2. Let the method update its native memory state.
3. Query the method through its native retrieval interface.
4. Return `memory_context` to the shared evaluator.
5. Ask Qwen3-8B to select one `choice_id` from `memory_context + query + choices`.
6. Score exact-choice Accuracy against `private/probe_key.jsonl`.

Memory adapters are forbidden from returning `choice_id` or choice `scores`.
The primary metric is Accuracy. Reports also include Wilson 95% confidence
intervals and breakdowns by domain, probe type, capability group, habit family,
stress variant, and split when those annotations are available.

The fixed base model is:

```text
/data1/public/hf/Qwen/Qwen3-8B
served model name: habitbench-qwen3-8b
context length: 40960
thinking mode: disabled
temperature: 0
```

Methods that require an LLM for memory construction also use this served model.
E5 or another embedding model may be used inside a method for retrieval, but an
embedding model never chooses the final answer.

## Methods

Active adapters and pinned upstream revisions are recorded in
`eval/methods.json`.

| method | source path | status |
| --- | --- | --- |
| Mem0 | official `Memory.add(infer=True)` and `Memory.search` | official API integration |
| A-MEM | official `AgenticMemorySystem` | official-code integration |
| Graphiti | official `add_episode` and `search_` | official API with documented Kuzu retrieval adaptation |
| SeCom | official segmentation, compression, retrieval modules | official code adapted to online session ingestion |
| O-Mem | official `SimpleMemory` lifecycle | official code with documented local-backend compatibility patches |
| no-memory / full-history | HABIT-Bench controls | not memory methods |

These are controlled HABIT-Bench integrations, not claims that every original
paper configuration has been exactly reproduced.

## Quick Start On Lumia

Allocate a Slurm compute node before running tests or model evaluation:

```bash
srun -p debug --gres=gpu:1 --cpus-per-task=8 --mem=64G --pty bash
cd /home/jmzhang/Workspace/habit-bench
source scripts/lumia/lumia_env_example.sh
```

Start Qwen3-8B on the allocated node:

```bash
bash scripts/lumia/start_vllm_openai_server.sh
```

In another shell attached to the same node, validate both datasets:

```bash
python -m eval.validate \
  domain/food/food_habit_lifelines_stress \
  domain/finance-software/habit_bench_multidogo_finance_software_long_hard_diverse_v0_5
```

Run a small end-to-end check:

```bash
bash scripts/run_method.sh no_memory \
  domain/food/food_habit_lifelines_stress \
  results/dev/food_no_memory \
  --max-users 1 --max-probes 4
```

Run one method on both domains:

```bash
bash scripts/run_two_domains.sh mem0
```

For full multi-GPU evaluation, create a user-sharded plan and explicitly select
the methods to run. This example runs Mem0 only and skips both controls:

```bash
bash scripts/slurm/submit_sharded_suite.sh \
  --methods mem0 \
  --shards 8 \
  --max-concurrent 8 \
  --partition GPU_PARTITION \
  --env-file /path/to/habitbench-cluster-env.sh \
  --output-root results/mem0_full
```

See `docs/multigpu_evaluation.md` for Slurm, single-node multi-GPU, resume, and
strict shard-merge instructions. `no_memory` and `full_history` are never added
implicitly; include them in `--methods` only when those controls should run.

Hugging Face access must be used with HTTP/HTTPS proxy variables unset. The
current Qwen3-8B model is already in `/data1/public/hf`, so no download is
needed.

## Output Layout

Each run writes:

```text
run_manifest.json
method_input.json
memory_contexts.jsonl
predictions.jsonl
scored_predictions.jsonl
metrics.json
metrics_by_group.csv
adapter.stdout.log
adapter.stderr.log
```

`method_input.json` is safe for evaluated methods and contains no gold labels.
`memory_contexts.jsonl` is the method boundary. `predictions.jsonl` is produced
only after the shared Qwen3-8B answer stage.

## Project Layout

| path | purpose |
| --- | --- |
| `domain/` | current domain datasets |
| `eval/core/` | normalization, Qwen answering, scoring, and I/O |
| `eval/official_adapters/` | official-code memory context adapters |
| `eval/run.py` | one complete memory-to-answer-to-Accuracy run |
| `eval/score.py` | strict scoring of existing predictions |
| `eval/validate.py` | public/private dataset contract validation |
| `schema/` | normalized session, probe, and memory-context schemas |
| `scripts/run_method.sh` | method launcher |
| `scripts/run_two_domains.sh` | two-domain launcher |
| `scripts/create_shard_plan.py` | deterministic user-shard task planner |
| `scripts/slurm/submit_sharded_suite.sh` | selectable Slurm multi-GPU suite |
| `tests/evaluation/` | evaluator unit tests |
| `third_party/official-baselines/` | pinned official repositories |
| `docs/evaluation_protocol.md` | formal protocol and leakage boundary |

Historical experiments under `runs/` are not part of the active evaluation
contract and should not be mixed with new `results/` outputs.
