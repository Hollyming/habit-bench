# Food Habit Lifelines v2 Stress Dataset

Status: auto-validated candidate generated from the food content-constraints
pilot; human review is still required before release claims.

## Source

- Seed source: local `data/dialog` and `data/recipe`.
- Input package: `runs_v3/food_habit_lifelines_final_naturalized_v2`.
- Habit family: `content_constraints`.
- Controlled components: hidden habit graphs, injected preference evidence,
  answer choices, gold labels, and evidence links.

## Contents

- Users: 30
- Sessions: 3900
- Hidden habits: 210
- Probes: 630
- Probe types: direct_use, boundary, exception.
- Stress variants: original_food_v2.

## Important Boundary

Public files are intended for evaluated systems. Private files contain hidden
graphs, gold labels, evidence ids, and memory annotations.

## Files

- `public/lifelines.jsonl`
- `public/probes.jsonl`
- `private/sessions_with_annotations.jsonl`
- `private/probe_key.jsonl`
- `private/habit_graphs.jsonl`
- `reports/food_v2_manifest.json`
- `review/food_v2_review_queue_sample.csv`
