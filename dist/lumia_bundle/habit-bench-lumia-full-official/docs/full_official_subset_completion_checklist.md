# Full Official Subset Completion Checklist

Use this checklist before claiming that the Lumia full official subset
experiment is complete.

## Required Dataset Evidence

- Dataset path exists:
  `./runs/habit_bench_balanced_v0_3_official_subset_90`
- Manifest exists:
  `reports/official_subset_manifest.json`
- Manifest counts match:
  - probes: 90
  - users: 67
  - sessions: 3,815
  - families: 9
  - probes per family: 10
  - privacy probes: at least 8
  - drift probes: at least 8
- Lightweight sanity baseline exists:
  `baseline_results/diagnostic_summary.csv`

## Required Environment Evidence

Record in the run log:

- Lumia host identifier or job id.
- GPU type and count.
- Python version and virtualenv path.
- Installed dependency file or exact install command.
- LLM model id/path.
- Embedding model id/path.
- OpenAI-compatible endpoint URL.
- vLLM or serving backend version.
- Model download manifest exists:
  `runs/lumia_manifests/model_download_manifest.json` or the path set by
  `HABITBENCH_MODEL_DOWNLOAD_MANIFEST`.
- Model download manifest is real completion evidence:
  - top-level `status=pass`
  - top-level `dry_run=false`
  - at least two model rows
  - every model row has `status=pass` and a non-empty `cache_path`
- Full-suite run manifests exist under:
  `full_official_results/run_manifests/<run_id>/`.
- `lumia_preflight_manifest.json` exists under the run-manifest directory and
  has `status=pass`.
- `suite_end_manifest.json` and `e2e_end_manifest.json` include
  `extra.exit_code=0`. A nonzero or missing exit code is diagnostic evidence,
  not completion evidence.

## Adapter Baseline Evidence

Current official-code storage/retrieval adapters are complete when all expected
directories exist under:

`official_results/`

Expected adapter rows:

- `mem0_infer_false_hf_qdrant`
- `amem_search_agentic_no_evolution`
- `secom_bm25_session`
- `graphiti_kuzu_edge_cosine`
- `omem_retrieval_injected_memory`

For each row:

- `*_raw_predictions.jsonl` exists.
- `*_scored_predictions.jsonl` exists.
- `*_metrics_summary.csv` exists.
- `*_diagnostic_summary.csv` exists.
- `*_baseline_report.md` exists.
- `*_stderr.txt` is inspected; warnings are acceptable if documented.

Then run:

```bash
python ./eval/collect_official_results.py \
  --dataset-dir ./runs/habit_bench_balanced_v0_3_official_subset_90
```

Expected collection outputs:

- `official_results/collected/official_results_collected.csv`
- `official_results/collected/official_results_collected.md`
- `official_results/collected/official_results_collected.json`

## Full Official Evidence

A method can be marked full official only when its core LLM-backed write/update
path is enabled.

| method | required full-path evidence |
| --- | --- |
| Mem0 | LLM extraction/update enabled; run config shows `infer=True` or equivalent official extraction path; visible-history filtering preserved |
| A-MEM | `process_memory` / evolution / linking enabled with local LLM endpoint |
| SeCom | segmentation and compression path enabled, not session-only BM25 |
| Graphiti | `add_episode` or equivalent LLM extraction/KG resolution path enabled; graph backend, `OpenAIGenericClient`/endpoint, and structured-output mode documented |
| O-Mem | active message understanding/persona update and generation path enabled |
| RMM | official or author-provided implementation path recorded |

Current implemented full-path scaffold:

- `official_mem0_full_llm_openai`, launched with
  `scripts/run_full_official_subset_mem0.sh`.
- `official_mem0_full_llm_openai_dryrun` is only a config/runner contract
  check. Do not cite its accuracy or treat it as a method result.
- `official_graphiti_full_llm_episode_kuzu`, launched with
  `scripts/run_full_official_subset_graphiti.sh`.
- `official_graphiti_full_llm_episode_kuzu_dryrun` is only a config/runner
  contract check. Do not cite its accuracy or treat it as a method result.

For every full official method:

- Save command in run log.
- Save config/env snapshot.
- Save per-method start/end run manifests.
- Save raw/scored predictions and diagnostic summary.
- Save token/call/latency/cost evidence if available.
- Label disabled components explicitly.

After collecting results, run:

```bash
python ./eval/audit_full_official_results.py \
  --dataset-dir ./runs/habit_bench_balanced_v0_3_official_subset_90 \
  --results-dir ./runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results
```

Expected audit outputs:

- `full_official_results/audit/full_official_audit.json`
- `full_official_results/audit/full_official_audit.md`
- audit status: `pass`

## Do Not Claim Completion If

- Only retrieval adapters were run.
- Full official methods used different LLMs or embeddings without recording it.
- The method answered with access to hidden labels/private keys.
- The subset was changed after results were produced.
- Human-audit status is omitted.
- RMM is called official without a released or author-provided implementation.
- `audit_full_official_results.py` fails.
- The detached launcher reports `status=not_running` but no
  `habitbench_remote_e2e.exitcode` file.
- Any end manifest has missing or nonzero `extra.exit_code`.
- `lumia_preflight_manifest.json` is missing or has `status=fail`.
- `model_download_manifest.json` is missing, has `dry_run=true`, has
  `status` other than `pass`, or any model row lacks a `cache_path`.
- Dry-run directories are collected successfully but audit reports
  `dry_run_config_present_in_full_results`.
