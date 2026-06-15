# Lumia Full Official Handoff

Generated: 2026-06-15T10:39:03.643140+00:00

## Current Artifact

- Bundle: `dist\lumia_bundle\habit-bench-lumia-full-official.tar.gz`
- SHA256: `4e47ba480bd144496fc146c851aa108b8843e4c6c1d0da6f4831d447b0cc81bb`
- Bytes: 259993
- Bundle files: 60

## Dataset

- Dataset: `runs\habit_bench_balanced_v0_3_official_subset_90`
- Probes: 90
- Users: 67
- Sessions: 3815
- Family counts: coding_review: 10, content_constraints: 10, drift_seasonality: 10, format_style: 10, meeting_prep: 10, planning_defaults: 10, privacy_consent: 10, risk_threshold: 10, tool_action: 10
- Probe types: boundary: 20, direct_use: 16, drift: 8, exception: 24, explicit_retrieval: 14, privacy: 8
- Stress variants: original_balanced: 50, unseen_paraphrase: 40
- Domain provenance: `pass`
- Family-domain contract: `nine_unique_representative_domains`

## Launch

Preferred single-command guarded cycle:

```bash
python ./scripts/lumia/run_lumia_guarded_full_cycle.py \
  --host lumia \
  --remote-dir /home/jmzhang/habitbench-lumia \
  --scp "scp -O" \
  --remote-env PYTHON_BIN=/home/jmzhang/miniconda3/bin/python \
  --remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct \
  --remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2 \
  --remote-env HABITBENCH_EMBED_DIMS=768 \
  --remote-env HABITBENCH_MAX_MODEL_LEN=16384 \
  --remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256 \
  --remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600 \
  --remote-env HABITBENCH_PROGRESS_EVERY=100 \
  --slurm-detached \
  --slurm-partition L40S \
  --slurm-gres gpu:1 \
  --slurm-time 06:00:00 \
  --slurm-job-name habitbench \
  --wait-timeout-sec 86400 \
  --wait-poll-sec 300 \
  --execute
```

Run it without `--execute` first to inspect both child launcher plans.

Dry-run first:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host lumia \
  --remote-dir /home/jmzhang/habitbench-lumia \
  --scp "scp -O" \
  --remote-env PYTHON_BIN=/home/jmzhang/miniconda3/bin/python \
  --remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct \
  --remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2 \
  --remote-env HABITBENCH_EMBED_DIMS=768 \
  --remote-env HABITBENCH_MAX_MODEL_LEN=16384 \
  --remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256 \
  --remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600 \
  --remote-env HABITBENCH_PROGRESS_EVERY=100 \
  --remote-run-prefix "srun --partition=L40S --gres=gpu:1 --time=06:00:00 bash -lc"
```

For the currently configured `lumia` SSH alias, `scp -O` is required
because the JumpServer SFTP-mode scp cannot see the same home path;
`PYTHON_BIN=/home/jmzhang/miniconda3/bin/python` avoids the login
node's missing `python3-venv`; `--remote-run-prefix` can route short
preflight/download runs through Slurm GPU nodes; and `--slurm-detached`
submits the long full run with `sbatch` so Slurm owns the process
lifetime. The current Lumia offline path uses local
`/home/jmzhang/models/Qwen2.5-14B-Instruct` and
`/home/jmzhang/models/e5-base-v2`; the latter requires
`HABITBENCH_EMBED_DIMS=768`.

Run remote preflight only before starting the long job:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host lumia \
  --remote-dir /home/jmzhang/habitbench-lumia \
  --scp "scp -O" \
  --remote-env PYTHON_BIN=/home/jmzhang/miniconda3/bin/python \
  --remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct \
  --remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2 \
  --remote-env HABITBENCH_EMBED_DIMS=768 \
  --remote-env HABITBENCH_MAX_MODEL_LEN=16384 \
  --remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256 \
  --remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600 \
  --remote-env HABITBENCH_PROGRESS_EVERY=100 \
  --remote-run-prefix "srun --partition=L40S --gres=gpu:1 --time=06:00:00 bash -lc" \
  --preflight-only \
  --execute
```

This fetches the returned preflight manifests and writes local
`runs/lumia_preflight_import_audit.json` and `.md`; require JSON
`status=pass`
before launching the detached full run.

Download/cache open models only and audit the returned manifest:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host lumia \
  --remote-dir /home/jmzhang/habitbench-lumia \
  --scp "scp -O" \
  --remote-env PYTHON_BIN=/home/jmzhang/miniconda3/bin/python \
  --remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct \
  --remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2 \
  --remote-env HABITBENCH_EMBED_DIMS=768 \
  --remote-env HABITBENCH_MAX_MODEL_LEN=16384 \
  --remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256 \
  --remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600 \
  --remote-env HABITBENCH_PROGRESS_EVERY=100 \
  --remote-run-prefix "srun --partition=L40S --gres=gpu:1 --time=06:00:00 bash -lc" \
  --download-models-only \
  --execute
```

Preferred one-command option: preflight, download/cache open models,
fetch/audit both returned manifest sets locally, then start the detached
full run only if both audits pass:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host lumia \
  --remote-dir /home/jmzhang/habitbench-lumia \
  --scp "scp -O" \
  --remote-env PYTHON_BIN=/home/jmzhang/miniconda3/bin/python \
  --remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct \
  --remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2 \
  --remote-env HABITBENCH_EMBED_DIMS=768 \
  --remote-env HABITBENCH_MAX_MODEL_LEN=16384 \
  --remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256 \
  --remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600 \
  --remote-env HABITBENCH_PROGRESS_EVERY=100 \
  --slurm-detached \
  --slurm-partition L40S \
  --slurm-gres gpu:1 \
  --slurm-time 06:00:00 \
  --slurm-job-name habitbench \
  --preflight-download-then-detached \
  --execute
```

After those audits pass, this mode reuses the same remote workspace for
the detached full run and sets `HABITBENCH_SKIP_MODEL_DOWNLOAD=1`, so the
audited `model_download_manifest.json` is preserved for final import.

Preflight-only one-command option for hosts where model download has
already been audited separately:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host lumia \
  --remote-dir /home/jmzhang/habitbench-lumia \
  --scp "scp -O" \
  --remote-env PYTHON_BIN=/home/jmzhang/miniconda3/bin/python \
  --remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct \
  --remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2 \
  --remote-env HABITBENCH_EMBED_DIMS=768 \
  --remote-env HABITBENCH_MAX_MODEL_LEN=16384 \
  --remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256 \
  --remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600 \
  --remote-env HABITBENCH_PROGRESS_EVERY=100 \
  --slurm-detached \
  --slurm-partition L40S \
  --slurm-gres gpu:1 \
  --slurm-time 06:00:00 \
  --slurm-job-name habitbench \
  --preflight-then-detached \
  --execute
```

Start the real detached Lumia run with an already-audited model/cache
manifest and an existing remote workspace:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host lumia \
  --remote-dir /home/jmzhang/habitbench-lumia \
  --scp "scp -O" \
  --remote-env PYTHON_BIN=/home/jmzhang/miniconda3/bin/python \
  --remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct \
  --remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2 \
  --remote-env HABITBENCH_EMBED_DIMS=768 \
  --remote-env HABITBENCH_MAX_MODEL_LEN=16384 \
  --remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256 \
  --remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600 \
  --remote-env HABITBENCH_PROGRESS_EVERY=100 \
  --slurm-detached \
  --slurm-partition L40S \
  --slurm-gres gpu:1 \
  --slurm-time 06:00:00 \
  --slurm-job-name habitbench \
  --reuse-remote-workspace \
  --detached \
  --execute
```

Check status and log tail:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host lumia \
  --remote-dir /home/jmzhang/habitbench-lumia \
  --scp "scp -O" \
  --status-only \
  --execute
```

Preferred collector after the detached job starts: wait for
`habitbench_remote_e2e.exitcode=0`, then fetch and import/audit locally:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host lumia \
  --remote-dir /home/jmzhang/habitbench-lumia \
  --scp "scp -O" \
  --wait-and-fetch \
  --wait-timeout-sec 86400 \
  --wait-poll-sec 300 \
  --execute
```

This mode does not import partial results when the detached job is still
running, missing an exit-code file, or exits nonzero.

Fetch and audit after the job stops and `exit_code=0` is present:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host lumia \
  --remote-dir /home/jmzhang/habitbench-lumia \
  --scp "scp -O" \
  --fetch-only \
  --execute
```

## Current Lumia Evidence

- SSH alias `lumia` resolves to login/storage host `storage-hdd`; GPU
  work must be routed through Slurm.
- Verified Slurm GPU smoke: `srun --partition=RTX4090 --gres=gpu:1`
  reached `gpu-4090-2` with one RTX 4090. The recommended full-run
  partition is L40S because the detected local LLM is a 14B model.
- Remote readiness and GPU/dataset/import preflight pass on the RTX4090
  node when using the conda Python bootstrap above.
- Hugging Face downloads on Lumia require proxy to be disabled first:
  run `proxy_off` from the account bashrc before model download; use
  `proxy_on` only when another network path explicitly needs it.
- Current full-run submission path uses `--slurm-detached`, which writes
  `habitbench_remote_e2e.sbatch`, records `habitbench_remote_e2e.jobid`,
  and lets Slurm own the long job lifecycle.

## Completion Gate

Do not claim completion until all of the following are true:

- `habitbench_remote_e2e.exitcode` exists on Lumia and contains `0`.
- `full_official_results/run_manifests/<run_id>/lumia_preflight_manifest.json` has `status=pass`.
- `full_official_results/run_manifests/<run_id>/suite_end_manifest.json` has `extra.exit_code=0`.
- If present, `full_official_results/run_manifests/<run_id>/e2e_end_manifest.json` has `extra.exit_code=0`.
- `runs/lumia_manifests/model_download_manifest.json` exists with top-level `status=pass` and `dry_run=false`.
- Every model row in `model_download_manifest.json` has `status=pass` and a non-empty `cache_path`.
- `full_official_results/mem0_full_llm_openai/` exists and contains raw/scored predictions plus config/runtime/report files.
- `full_official_results/graphiti_full_llm_episode_kuzu/` exists and contains raw/scored predictions plus config/runtime/report files.
- `full_official_results/collected/official_results_collected.csv` exists.
- `full_official_results/audit/full_official_audit.json` has `status=pass`.

## Current Missing Items


## Notes

- The benchmark uses `allenai/WildChat` as the single real prompt seed source.
- The official subset is real-prompt-seeded, domain-grounded, and synthetic-longitudinal.
- The nine habit families use unique representative domains; they are not nine separate external datasets.
- This handoff is a sidecar document next to the bundle; it is intentionally not included inside the tarball because it records the tarball hash.
