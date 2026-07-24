# Taskmaster Planning Defaults v0.2

- Goal: build a longer multi-habit planning_defaults benchmark slice with stronger distractors.
- Generation mode: llm.
- LLM generation: gpt-5.5 with reasoning_effort=xhigh.
- Habit templates: 12 global planning-default types.
- Users: 30
- Sessions: 1080
- Probes: 120
- Avg messages/session: 18.977
- Median messages/session: 18
- Avg chars/session: 3331.244
- Median chars/session: 3277
- P10 chars/session: 2732
- Avg words/session: 587.348
- Length contract: at least 1500 chars and 12 messages per session.

Primary human-review file: `review/planning_defaults_review_queue_all.csv`.

Evaluation status:

- Structural format: compatible with the unchanged reference evaluator.
- Formal lightweight results: `baseline_results/`, produced by
  `eval/evaluate_baselines.py`.
- Formal official-adapter results: `official_results/`, produced by the same
  commands as `scripts/run_official_subset_adapters.sh`.
- Content alignment gate: evaluated only with the unchanged reference
  no-memory baseline; see `reports/reference_alignment_audit.md`.
- Non-reference Qwen full-history/memory-matrix results are intentionally not
  part of this run.
