# HABIT-Bench Pilot Dataset Card

This pilot benchmark package was generated automatically and has not yet
undergone human review.

## Intended Use

Evaluate whether long-term memory agents can infer and use implicit,
context-scoped user habits while avoiding false personalization.

## Construction

- Seed source: cache
- Users: 200
- Sessions: 12000
- Probes: 2719
- Habit graphs: 585

Real user-assistant prompts are used only as sanitized task seeds. Hidden habit
graphs, feedback, and probes are synthetic and controlled.

## Human Review Status

Human review has not been performed. Use `review/review_queue_sample.csv` for a
first audit and `review/review_queue_all.csv` for full audit.

## Leakage Controls

Public files omit hidden habit graphs, gold labels, and explicit signal types.
Private files contain labels and should not be exposed to benchmarked systems.

## Known Limitations

- Lifelines are pseudo-users stitched from real prompt seeds and synthetic
  controlled feedback.
- The current pilot is English-only unless the seed loader is extended.
- Automatic validators check structure, labels, evidence links, and obvious PII;
  they do not replace human audit.
