# Human Review Guidelines

Use `review/planning_defaults_review_queue_all.csv` for full review.

## Decision Labels

- `accept`: the probe is answerable, the gold choice is correct, evidence is sufficient, and distractors are plausible.
- `revise`: the idea is usable but wording, evidence strength, or distractors need edits.
- `reject`: the probe is not usable for this habit family or cannot be repaired locally.

## What To Check

- The query should not reveal the answer without reading history, except `explicit_retrieval`, which intentionally asks for the remembered preference.
- `boundary` probes should reward not applying the work-trip default to leisure or relaxed travel.
- `exception` probes should reward following the current trip's explicit constraint.
- Distractors should be plausible travel recommendations, not absurd or format-only answers.
- Evidence sessions listed in `evidence_preview_json` should support the proposed gold action.
