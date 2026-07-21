# MultiDoGO Finance + Software HABIT-Bench candidate v0.5

This is a **source-grounded, synthetic-longitudinal** candidate built from MultiDoGO finance and software customer-agent conversations. It is not a claim that MultiDoGO contains natural same-user longitudinal identities.

## What changed from v0.4

1. **Identity coherence by construction:** raw customer dialogues are not concatenated. Source conversations are converted to sanitized task events and rewritten under one stable persona.
2. **Longer histories:** 320 model-visible sessions per user across nearly three years.
3. **Multi-habit users:** six active habits, two updated habits, one conditionally scoped habit, and one deliberately tentative one-off signal per user.
4. **Harder probes:** normal end-user requests; no `What should the assistant do?` meta-question. Probes include two/three-habit composition, priority conflict, drift, nested exceptions, false personalization, and insufficient evidence.
5. **Diversity:** 810/810 unique normalized queries and 810/810 unique choice sets.

## Dataset size

- Domains: finance, software
- Pseudo-users: 45
- Sessions: 14400 (320 per user)
- Retained habits: 15
- Probes: 810
- Median full prompt: 84760 tokens
- Identity audit: 45/45 passed

## Evaluation

Give a method only `public/lifelines.jsonl` and `public/probes.jsonl`. The method outputs one JSONL row per probe:

```json
{"probe_id":"mdgo_v05_probe_000000","choice_id":"A"}
```

Score with:

```bash
python scripts/score_predictions.py --dataset-dir . --predictions predictions.jsonl --output-dir runs/my_eval --method-name my_method
```

Private files contain gold answers, evidence links, persona profiles, and source provenance. This candidate must still receive human review before paper-scale use.
