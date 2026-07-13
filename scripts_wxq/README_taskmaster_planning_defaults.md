# Taskmaster-2 Planning Defaults Slice

This directory contains the builder for a HABIT-Bench `planning_defaults`
slice grounded in Taskmaster-2 `flights` and `hotels` dialogs.

## Builder

```bash
python /mnt/petrelfs/linzhouhan/xqwang/project/habit-bench/scripts_wxq/build_taskmaster_planning_defaults.py
```

Default output:

```text
/mnt/petrelfs/linzhouhan/xqwang/project/habit-bench/runs_wxq/taskmaster_planning_defaults_v0_1
```

Expected raw files if automatic download is blocked:

```text
runs_wxq/taskmaster_planning_defaults_v0_1/data/raw_taskmaster/flights.json
runs_wxq/taskmaster_planning_defaults_v0_1/data/raw_taskmaster/hotels.json
```

Then rerun the same command. Or pass another directory:

```bash
python /mnt/petrelfs/linzhouhan/xqwang/project/habit-bench/scripts_wxq/build_taskmaster_planning_defaults.py \
  --raw-data-dir /path/to/taskmaster/TM-2-2020/data
```

## Output Layout

```text
data/
  raw_taskmaster/
  filtered_taskmaster_travel_seeds.jsonl
public/
  lifelines.jsonl
  probes.jsonl
private/
  sessions_with_annotations.jsonl
  habit_graphs.jsonl
  probe_key.jsonl
review/
  planning_defaults_review_queue_all.csv
  planning_defaults_review_queue_sample.csv
reports/
  build_manifest.json
  planning_defaults_summary.md
  auto_validation_summary.json
```

## Current Run Note

The first run on this machine could not download from GitHub/HuggingFace because
the configured Squid proxy returned HTTP 403 and direct no-proxy access timed
out. After manually placing `flights.json` and `hotels.json` under
`data/raw_taskmaster/`, the builder completed successfully and wrote:

```text
runs_wxq/taskmaster_planning_defaults_v0_1/reports/build_manifest.json
runs_wxq/taskmaster_planning_defaults_v0_1/reports/planning_defaults_summary.md
```

The builder now removes a stale `reports/download_failure.json` after a
successful run.

## Model/Heuristic Review Labeling

Model pre-labeling script:

```bash
bash -ic 'proxy_on; unset all_proxy ALL_PROXY; \
HABITBENCH_BASE_URL="https://queqiao.online/v1" \
HABITBENCH_API_KEY="<api key>" \
HABITBENCH_LABEL_MODEL="gpt-5.5" \
python /mnt/petrelfs/linzhouhan/xqwang/project/habit-bench/scripts_wxq/model_label_planning_defaults_review.py \
  --transport curl \
  --reasoning-effort xhigh \
  --timeout-sec 240 \
  --max-retries 5'
```

The script writes a new labeled CSV and does not overwrite the blank review
queue:

```text
review/planning_defaults_review_queue_all_model_labeled.csv
```

On the current machine, plain proxy settings use the wrong proxy and fail. The
working pattern is interactive `proxy_on` plus `unset all_proxy ALL_PROXY`.
Python `urllib` chat requests triggered HTTP 403 `error code: 1010`, so the
labeler uses `--transport curl`, which succeeds with `gpt-5.5`.

The completed `gpt-5.5` default-effort pre-review artifacts are:

```text
review/planning_defaults_review_queue_all_gpt55_labeled.csv
review/planning_defaults_review_queue_all_gpt55_labeled_raw.jsonl
reports/model_label_gpt55_summary.json
```

The completed `gpt-5.5` `xhigh` pre-review artifacts are:

```text
review/planning_defaults_review_queue_all_gpt55_xhigh_labeled.csv
review/planning_defaults_review_queue_all_gpt55_xhigh_labeled_raw.jsonl
reports/model_label_gpt55_xhigh_summary.json
```

The xhigh run produced 59 `accept` and 61 `revise` labels after retrying and
merging one transient proxy failure.

For comparison, I also generated a clearly marked rule-based prelabel file:

```text
review/planning_defaults_review_queue_all_heuristic_labeled.csv
reports/heuristic_label_summary.json
```

Those heuristic labels are for triage only and should not be treated as model
review output.
