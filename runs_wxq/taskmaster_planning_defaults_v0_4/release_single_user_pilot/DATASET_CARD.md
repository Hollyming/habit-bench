# HABIT-Bench Travel Planning Defaults v0.4 — Single-User Pilot

## Release status

This is a frozen, evaluation-ready **single-user pilot**, not the final
multi-user HABIT-Bench release.  It is intended for internal review,
end-to-end evaluator integration, and preliminary model/memory experiments.

## Scope and construction

- Domain: travel planning defaults grounded in Taskmaster flights/hotels.
- User: `tm_pd_v04_user_000`.
- History: 126 chronological sessions in 30 recurring trip episodes.
- Dialogue size: 1,778 messages and about 93,984 Qwen3-8B tokens.
- Construction: GPT-5.5 with `reasoning_effort=xhigh` directly authored the
  dossier, event arc, sessions, and probes. No utterance templates, slot-filled
  dialogue skeletons, or paraphrase/rewrite pipeline were used.
- Five latent habits are tested. Two additional habits are retained as
  background realism and are not probe targets.

## Probe set

- 40 four-choice probes in total.
- 30 positive latent-habit probes: exactly 6 for each tested habit.
- 10 false-personalization negative controls spanning 7 decision dimensions.
- Gold answer positions are balanced: A/B/C/D each occur 10 times.
- No released query near-duplicates were detected by the release audit.

The positive probes test repeated weak-evidence induction, scope/boundary
calibration, cross-context transfer, conflict resolution, evidence
disambiguation, and supported exceptions. The negative controls test whether
noisy or trip-specific history causes an agent to invent a stable preference.

## Files

### Public model input

- `public/lifelines.jsonl`: chronological user-assistant sessions.
- `public/probes.jsonl`: multiple-choice benchmark questions without labels.

### Private ground truth

- `private/probe_key.jsonl`: gold answers, evidence citations, target
  habits/negative controls, and independent adjudications.
- `private/user_dossiers.jsonl`: hidden controlled user dossier and
  habit graph.
- `private/sessions_with_annotations.jsonl`: sessions plus verified
  post-hoc habit evidence.
- `private/habit_version_history.jsonl`: temporal habit states.
- `private/false_personalization_controls.jsonl`: adjudicated non-habit
  controls.

The `private/` directory contains labels and is not model input.

## Evaluation unit

For each probe, provide the evaluated system with the public user history up
to `visible_history_scope.max_session_index`, followed by the probe query and
choices. Score exact equality between the returned choice ID and
`gold_choice_id` in the private key.

Positive latent-habit accuracy and false-personalization accuracy should be
reported separately as well as jointly. For memory systems, also report the
change relative to a query-only baseline rather than interpreting raw accuracy
alone.

## Known limitations

- This pilot contains only one synthetic longitudinal user, so it cannot
  measure cross-user variance or support statistically strong model rankings.
- The frozen history has no verified durable habit revision. It must not be
  used to claim drift-handling performance.
- Five false-personalization probes are deliberately easy for a query-only
  solver. Their purpose is to measure whether noisy memory degrades an
  otherwise resolvable decision.
- Results from this release should be labeled preliminary/pilot results.

## Release gate

The release contains 126 sessions and 40 probes; public/private IDs align,
all evidence citations are traceable, all 10 negative controls were
independently judged as not supporting a stable preference, and answer
positions are balanced. See `reports/generation_release.json` for the machine
readable summary.
