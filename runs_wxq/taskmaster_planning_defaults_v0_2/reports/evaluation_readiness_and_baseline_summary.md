# Evaluation Readiness and Baseline Summary

Date: 2026-07-12

## Dataset Status

- Source: Taskmaster-2 flights + hotels, rewritten/generated with gpt-5.5 xhigh for the planning_defaults habit family.
- Users: 30
- Sessions: 1080
- Probes: 120
- Hidden habit graphs: 30
- Probe distribution: 30 direct_use, 30 boundary, 30 exception, 30 explicit_retrieval
- Habit template distribution: 12 global planning-default templates, with 2-3 users per template.
- Session length: mean 3331.2 characters, median 3276.5, min 1824, max 5277; mean 19.0 messages.
- Model review: 120 accept, 0 revise, 0 reject.
- Structural audit: 0 ID/key/capability consistency errors after capability_group normalization.

## Compatibility Fix

The data was structurally valid, but `private/probe_key.jsonl` used short capability names. The existing `runs` workflow expects standard capability names for report aggregation.

Applied mapping:

- direct_use -> habit_direct_use
- boundary -> habit_boundary_false_personalization
- exception -> counterevidence_exception
- explicit_retrieval -> explicit_fact_preference_retrieval

This changes evaluation metadata only; it does not change sessions, probe text, choices, gold answers, or evidence.

## Surface Cue Audit

- Gold choice distribution is balanced: A=28, B=30, C=33, D=29.
- Contrast marker high-risk cases from the previous audit remain resolved: no probe has two or more contrast-marked distractors while the gold choice lacks that marker.
- Final contrast marker ratio: gold 15/120, distractors 41/360.
- A broader caveat/risk-word scan still finds some travel-risk vocabulary, which is expected in flights/hotels planning. These are not the earlier obvious `but/though` distractor artifacts.

## Lightweight Baseline Evaluation

Command used:

```bash
python eval/evaluate_baselines.py \
  --dataset-dir runs_wxq/taskmaster_planning_defaults_v0_2 \
  --output-dir runs_wxq/taskmaster_planning_defaults_v0_2/baseline_results
```

Outputs:

- `baseline_results/metrics_summary.csv`
- `baseline_results/diagnostic_summary.csv`
- `baseline_results/baseline_report.md`
- one `*_predictions.jsonl` file per lightweight baseline

Overall accuracy:

| baseline | overall |
| --- | ---: |
| no_memory_lexical | 0.4167 |
| full_history_segment_retrieval | 0.4917 |
| mem0_like_fact_memory | 0.3333 |
| zep_like_temporal_graph | 0.3750 |
| a_mem_like_note_linking | 0.5333 |
| secom_like_segment_memory | 0.4417 |
| rmm_like_reflective_summary | 0.3333 |
| o_mem_like_user_profile | 0.2583 |

Interpretation:

- The slice is not trivially solved by current lightweight baselines; the strongest proxy baseline is 0.5333.
- The no-memory lexical baseline is above random on boundary/exception cases, so future expansion should continue strengthening non-memory distractors and possibly add paraphrased or open-ended variants.
- For the current 120-probe v0.2 slice, the data is suitable to proceed with evaluator-loop experiments and human spot review before claiming official benchmark results.
