# Reference Alignment Audit

## Verdict

- Structural alignment: **pass**
- Accepted by the reference evaluator loader: **True**
- Content memory-dependence gate: **pass**
- Reference evaluation ready: **True**
- Human review complete: **False**
- Release ready: **False**

Structural alignment and benchmark difficulty are intentionally separate. A dataset may be fully readable by the reference evaluator while still allowing question/choice-only shortcuts.

## Counts

- Users: 30
- Sessions: 1080
- Probes/keys: 120/120
- Probe types: `{"boundary": 30, "direct_use": 30, "exception": 30, "explicit_retrieval": 30}`
- Review statuses: `{"taskmaster_planning_defaults_v02_gpt55_xhigh_needs_human_review": 84, "taskmaster_planning_defaults_v02_gpt55_xhigh_repaired_needs_human_review": 36}`

## Structural Errors

- None.

## Content Warnings

- None.

## No-memory Diagnostic

- Overall accuracy: 0.4167
- By probe type: `{"boundary": 0.6, "direct_use": 0.3, "exception": 0.5, "explicit_retrieval": 0.2667}`

## Formal Evaluation Contract

Formal results must be produced only through the unchanged repository reference modules:

- `eval/evaluate_baselines.py`
- `eval/run_external_baseline.py`
- `eval/official_adapters/*`
- `eval/collect_official_results.py`
