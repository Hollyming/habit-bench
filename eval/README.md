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
their blockers are recorded in `unsupported_methods.json`. `compact_history.py`
implements the `memory_context.v5` query-independent online compact
`full_memory` control. `controls.py` retains `no_memory` and the raw
`full_history` recency control; `context_windows.py` resolves their shared
standard/custom window tier. `retrieval_baselines.py` provides fixed-session
Recency-5/10, lexical BM25-RAG, BGE-M3 Dense-RAG, and as-of-aware Temporal
Hybrid-RAG without memory extraction, summarization, or an agentic lifecycle.

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
bash scripts/submit_h_cluster.sh --help
python -m unittest discover -s tests/evaluation -p 'test_*.py'
```

Formal method profiles pin local BGE-M3 at revision
`5617a9f61b028005a4858fdac845db406aefb181` with 1024-dimensional dense
embeddings. See `docs/multigpu_evaluation.md` for the shared single-node
execution model and `docs/h_cluster_evaluation.md` for the 4/8-card H200 RJob
launcher, including the timing/provenance files produced by a run.
