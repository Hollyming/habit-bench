# Dataset Card: Taskmaster Planning Defaults v0.2

## Scope

- Habit family: `planning_defaults`.
- Source dataset: Taskmaster-2 flights and hotels.
- Purpose: evaluate whether an agent can infer and apply a user's scoped default planning preference over long travel-planning histories.
- Habit template bank: 12 global planning-default types, assigned across users.
- Generation mode: `llm`.
- Session/probe generator: `gpt-5.5` with `reasoning_effort=xhigh`.

## Size

- Users: 30.
- Sessions: 1080.
- Probes: 120 total, with one `direct_use`, one `boundary`, one `exception`, and one `explicit_retrieval` probe per user.
- Support sessions: 150.
- Boundary sessions: 60.
- Exception sessions: 30.
- Distractor sessions: 840.

## Length Contract

- Minimum chars/session: 1500.
- Minimum messages/session: 12.
- Average chars/session: 3331.244.
- Median chars/session: 3277.
- P10 chars/session: 2732.
- Average messages/session: 18.977.
- Median messages/session: 18.

## Files

- `public/lifelines.jsonl`: public session histories without hidden habit annotations.
- `public/probes.jsonl`: public multiple-choice probes.
- `private/sessions_with_annotations.jsonl`: private sessions with signal annotations.
- `private/habit_graphs.jsonl`: hidden habit graphs.
- `private/probe_key.jsonl`: gold answers and evidence session IDs.
- `review/planning_defaults_review_queue_all.csv`: full human-review queue.
- `reports/auto_validation_summary.json`: validation and length statistics.

## Evaluation Note

The public probe format is accepted unchanged by the repository reference
loader and uses the same `choice_equals` scoring contract as the reference
runs. Formal results must be produced through `eval/evaluate_baselines.py` or
`eval/run_external_baseline.py` with the repository official adapters.

The current slice is structurally aligned. Its reference no-memory lexical
control is evaluated against a 0.45 ceiling calibrated to the reference
curated-v0.2 result (0.3986) with allowance for this slice's smaller sample.
See `reports/reference_alignment_audit.md`. A separate Qwen full-history or
Qwen answer-head matrix is not part of the aligned evaluation.
