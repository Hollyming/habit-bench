# Taskmaster Planning Defaults v0.1

This run is a single-family HABIT-Bench slice:

- family: `planning_defaults`
- representative domain: `travel`
- real seed source: Taskmaster-2 `flights` and `hotels`
- hidden habit: business/client-meeting travel prefers early arrival and about a 90-minute meeting buffer

## Final Files To Review

Use this file for human review:

```text
review/planning_defaults_review_queue_all_gpt55_xhigh_labeled.csv
```

It contains the full 120-row review queue with `gpt-5.5` `xhigh` pre-labels.
The original blank queue is kept at:

```text
review/planning_defaults_review_queue_all.csv
```

The quick calibration sample is:

```text
review/planning_defaults_review_queue_sample.csv
```

## Dataset Files

```text
public/lifelines.jsonl
public/probes.jsonl
private/sessions_with_annotations.jsonl
private/probe_key.jsonl
private/habit_graphs.jsonl
```

## Key Reports

```text
reports/build_manifest.json
reports/planning_defaults_summary.md
reports/auto_validation_summary.json
reports/model_label_gpt55_xhigh_summary.json
```

## Archive

Temporary test labels, heuristic labels, default-effort labels, and single-row
retry files were moved under:

```text
archive/debug_labeling/
```

They are kept only for provenance/debugging and are not the handoff files.
