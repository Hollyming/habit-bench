# HABIT-Bench Pilot v0 Construction Note

Status: pre-human-review

## What Was Built

- 200 pseudo-users.
- 12,000 user-agent sessions.
- 604 hidden habit graphs.
- 2,785 public probes.
- 2,785 private gold-label rows.
- 2,785 full review rows.
- 277 stratified review-sample rows.

## Source Data

The builder used 5,000 sanitized prompt seeds from `allenai/WildChat`
(`data/train-00000-of-00006.parquet`) and stitched them into controlled
pseudo-user lifelines. Real prompts are used only as task/context seeds;
hidden habits, feedback, and probes are synthetic.

The source loader rejected rows for non-English language, moderation/redaction,
length, PII patterns, or non-Latin/garbled text. See
`reports/build_manifest.json` for exact rejection counts.

No GPT-5.5/API generation was used in this run. The script supports optional
OpenAI-compatible template naturalization, but this pilot was generated
deterministically for reproducibility.

## Main Artifacts

- `public/lifelines.jsonl`: visible histories for memory systems.
- `public/probes.jsonl`: opaque probe ids, queries, choices, and history scope.
- `private/habit_graphs.jsonl`: hidden per-user habit rules.
- `private/probe_key.jsonl`: gold choices, public/private probe-id mapping, and
  evidence session ids.
- `review/review_queue_all.csv`: full pre-human-review queue.
- `review/review_queue_sample.csv`: first stratified audit sample.
- `reports/auto_validation_summary.md`: validation and distribution summary.
- `reports/build_manifest.json`: exact run settings and provenance.

## Automatic Checks Passed

- All 12,000 sessions passed schema/PII checks.
- All 2,785 probes passed schema, gold-choice, duplicate-id, and evidence-link checks.
- Public probe ids are opaque hashes.
- Public files omit hidden habit ids, gold labels, explicit signal types, and support counts.
- Every public probe id maps to one private key row and one review row.

## Human Review Boundary

Human review has not been performed. The next step is to review
`review/review_queue_sample.csv`, mark accept/reject/revise, inspect ambiguity
and leakage notes, then decide whether to freeze or regenerate specific splits.
