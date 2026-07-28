# Evaluation code

The evaluator has one strict boundary:

```text
public history + public probe
          |
          v
memory method or evaluator control
          |
          v
memory_context.jsonl
          |
          v
shared Qwen3-8B answerer
          |
          v
choice_id -> private scorer -> Accuracy
```

The primary seven methods are `mem0`, `amem`, `memos`, `memrl`, `lightmem`,
`letta`, and `mirix`. They all enter through
`medmemorybench_adapters/structured_memory.py`, which calls the method-native
memory-build and retrieval lifecycle in `third_party/medmemorybench`.
SeCom enters through a thin adapter in `official_adapters/`. Graphiti and
O-Mem are explicitly excluded pending bounded, full-domain implementations;
their blockers are recorded in `unsupported_methods.json`. `controls.py`
implements `no_memory` and the
capacity-aware, token-bounded `full_memory` long-context control;
`context_windows.py` resolves its standard/custom window tier, and
`full_history` is its compatibility alias.

## Memory method input

`eval.run` normalizes either active domain format into one JSON object:

```json
{
  "contract_version": "habitbench.memory_context.v1",
  "sessions_by_user": {"user_id": []},
  "probes": [],
  "output_contract": {
    "required_fields": ["probe_id", "memory_context"],
    "optional_fields": ["evidence_session_ids", "debug", "cost"],
    "forbidden_fields": ["choice_id", "scores"]
  }
}
```

Gold labels, gold evidence, hidden habit graphs, and policy variants are not
included. A history cutoff may be copied into the method-visible probe only
when a source dataset omitted this required evaluation-time field.

## Memory method output

One JSONL row is required for every probe:

```json
{
  "probe_id": "...",
  "memory_context": "retrieved method-native memory",
  "evidence_session_ids": ["..."],
  "debug": {},
  "cost": {}
}
```

The runner rejects missing or extra probes, duplicate probe IDs, non-string
contexts, invalid evidence IDs, and any top-level `choice_id` or `scores`.

## Entry points

```bash
python -m eval.validate DATASET_DIR [DATASET_DIR ...]
python -m eval.run --help
python -m eval.score --help
bash scripts/run_eval.sh METHOD DATASET_DIR OUTPUT_DIR [eval args]
bash scripts/submit_clusterx.sh --help
python -m unittest discover -s tests/evaluation -p 'test_*.py'
```

Formal method profiles pin local BGE-M3 at revision
`5617a9f61b028005a4858fdac845db406aefb181` with 1024-dimensional dense
embeddings. See `docs/multigpu_evaluation.md` for the single-node ClusterX
launcher and the timing/provenance files produced by a run.
