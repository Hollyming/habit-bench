# Senior Review Summary: HABIT-Bench Curated v0.1

## Review Decision

I reviewed the stratified pilot sample using the rubric in `HUMAN_REVIEW_GUIDELINES.md`.
The current `evidence` and `ask_act` probe templates were rejected for v0.1 because they are too meta-level and too easy to answer from wording alone.
Direct-use, boundary, exception, drift, and privacy probes were retained only after applying two systematic revisions: removing noisy real-log seed tails from visible histories and softening obviously artificial distractor wording.

## Counts

- users: 78
- sessions: 4438
- probes: 161
- reviewed_sample_rows: 277
- selected_reviewed_probes: 125
- explicit_retrieval_probes_added: 36

## Probe Types

- boundary: 36
- direct_use: 36
- drift: 8
- exception: 36
- explicit_retrieval: 36
- privacy: 9

## Capability Groups

- counterevidence_exception: 36
- explicit_fact_preference_retrieval: 36
- false_personalization_privacy: 9
- habit_boundary_false_personalization: 36
- habit_direct_use: 36
- habit_drift: 8

## Remaining Risks

- This is a small curated set intended for evaluator and baseline testing, not a final benchmark release.
- The explicit retrieval split is generated from reviewed direct-use habits to create a sanity-control task.
- A second human pass should manually inspect accepted rows before a paper-scale release.
