# Evaluation Code

The evaluator has one strict boundary:

```text
public history + public probe
          |
          v
official memory adapter
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

## Memory Method Input

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

Gold labels, evidence labels, hidden habit graphs, and policy variants are not
included. A private history cutoff may be copied into the public-facing probe
only when a source dataset omitted this necessary evaluation-time field.

## Memory Method Output

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

The runner rejects missing/extra probes, duplicate probe ids, non-string
contexts, invalid evidence ids, and any top-level `choice_id` or `scores`.

## Entry Points

```bash
python -m eval.validate DATASET_DIR [DATASET_DIR ...]
python -m eval.run --help
python -m eval.score --help
python -m unittest discover -s tests -p 'test_*.py'
```
