# MedMemoryBench integration

This directory documents the source-level MedMemoryBench integration used to
evaluate seven structured memory methods on HABIT-Bench. It does not call a
hosted Mem0, A-MEM, or other full memory workflow API.

## Repository layout

Clone the two repositories as siblings:

```text
workspace/
├── habit-bench/
└── MedMemoryBench/
```

Use branch `wjr` in both repositories. The HABIT method registry pins
MedMemoryBench commit `6591eb3251402f26535846ea4a95f5b4478ae35a`.

Initialize MedMemoryBench submodules and apply its recorded LightMem
compatibility patch:

```bash
cd MedMemoryBench
git submodule update --init --recursive
bash scripts/apply_lightmem_patch.sh
```

The patch command is idempotent. Mem0, A-MEM, MemOS, MemRL, Letta and MIRIX
changes are committed directly in the MedMemoryBench `wjr` branch.

## Environments

The validated internal environments are:

```text
core:  /plm-shared/wangjiarui/anaconda3/envs/habit_medmemorybench
Letta: /plm-shared/wangjiarui/anaconda3/envs/habit_medmemory_letta
MIRIX: /plm-shared/wangjiarui/anaconda3/envs/habit_medmemory_mirix
```

They are cluster-specific references, not portable defaults. On another
machine, create equivalent environments and select their Python executable
through `PYTHON_BIN`.

## Lightweight verification

Run the MedMemoryBench tests:

```bash
cd MedMemoryBench
python -m pytest -q tests
```

Run the HABIT adapter tests:

```bash
cd habit-bench
python -m pytest -q tests/evaluation/test_medmemorybench_adapter.py
```

List the available MedMemoryBench configurations:

```bash
cd MedMemoryBench
python main.py --list-methods
python main.py --list-datasets
```

## MedMemoryBench smoke run

The smoke dataset uses one persona and one evaluation unit:

```bash
cd MedMemoryBench
python main.py \
  --method mem0_qwen3-8b_smoke \
  --dataset medmemorybench_smoke_efficient \
  --output-dir outputs/mem0-smoke
```

Replace the method with `amem`, `memos`, `memrl`, `lightmem`, `letta`, or
`mirix` and the matching `*_qwen3-8b_smoke` configuration.

## HABIT-Bench smoke run

The two repositories must remain siblings, or
`HABITBENCH_MEDMEMORYBENCH_ROOT` must point to the MedMemoryBench checkout:

```bash
cd habit-bench
export HABITBENCH_MEDMEMORYBENCH_ROOT=../MedMemoryBench
export PYTHON_BIN=/path/to/the/method/environment/bin/python

bash scripts/run_method.sh \
  medmemorybench_mem0 \
  domain/food/food_habit_lifelines_stress \
  results/dev/food-medmemorybench-mem0 \
  --max-users 1 \
  --max-probes 4
```

Supported names are:

```text
medmemorybench_mem0
medmemorybench_amem
medmemorybench_memos
medmemorybench_memrl
medmemorybench_lightmem
medmemorybench_letta
medmemorybench_mirix
```

The adapter incrementally writes public sessions, calls only the method's
native retrieval path, and passes `memory_context` to the shared HABIT answerer.
Private labels and gold habit annotations are never passed to a memory method.

## Formal-run requirements

A formal result must include:

- exact repository commits and submodule revisions;
- data, model, prompt and metric hashes;
- complete user/persona and query/probe coverage;
- per-item predictions and retrieved contexts;
- strict shard-merge evidence and a failure scan;
- a label distinguishing method-native, common-reader and adapted results.

See [changes.md](changes.md) for the implementation summary and
[experiment_notes.md](experiment_notes.md) for the compact result record.
