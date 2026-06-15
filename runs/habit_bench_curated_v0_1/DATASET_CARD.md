# HABIT-Bench Curated v0.1

Status: small-scale curated dataset after senior review.

This dataset is built from the HABIT-Bench pilot v0 pre-review package. It is
intended to test evaluator code and method-inspired baselines before scaling.

## Contents

- Users: 78
- Sessions: 4438
- Probes: 161
- Reviewed pilot rows: 277
- Selected reviewed probes: 125
- Added explicit retrieval probes: 36

## Review Policy

Evidence and ask/act probes from pilot v0 were rejected for this version.
Direct-use, boundary, exception, drift, and privacy probes were retained after
systematic revisions. The visible histories are cleaned to remove noisy real-log
seed tails while preserving controlled user-agent evidence.

## Files

- `public/lifelines.jsonl`: histories for evaluated memory systems.
- `public/probes.jsonl`: public queries and answer choices.
- `private/probe_key.jsonl`: gold labels, evidence ids, and capability groups.
- `review/senior_review_decisions.csv`: review decisions for the pilot sample.
