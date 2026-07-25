# Food Content Constraints v0.3 Candidate

Status: auto-validated candidate generated from the food content-constraints
pilot; human review is still required before release claims.

## Source

- Seed source: local `data/dialog` and `data/recipe`.
- Input package: `runs_new/food_habit_lifelines_v1_naturalized`.
- Habit family: `content_constraints`.
- Controlled components: hidden habit graphs, injected preference evidence,
  answer choices, gold labels, and evidence links.

## Contents

- Users: 30
- Sessions: 3600
- Hidden habits: 210
- Probes: 1470
- Probe types: direct_use, boundary, exception, explicit_retrieval.
- Stress variants: original_food_v03, unseen_paraphrase.

## Important Boundary

Public files are intended for evaluated systems. Private files contain hidden
graphs, gold labels, evidence ids, and memory annotations.

## Files

- `public/lifelines.jsonl`
- `public/probes.jsonl`
- `private/sessions_with_annotations.jsonl`
- `private/probe_key.jsonl`
- `private/habit_graphs.jsonl`
- `reports/food_v03_manifest.json`
- `review/food_v03_review_queue_sample.csv`
