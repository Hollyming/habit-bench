# Lumia Full Official Subset Runbook

Purpose: run the expensive/full official-method configurations on a small,
stratified HABIT-Bench subset before scaling to v0.3 or a paper-ready audited
split.

## Dataset

Prepared subset:

`./runs/habit_bench_balanced_v0_3_official_subset_90`

Current manifest:

- probes: 90
- users: 67
- sessions: 3,815
- capability coverage:
  - explicit retrieval: 14
  - direct habit use: 16
  - boundary / false personalization: 20
  - exception / counterevidence: 24
  - drift: 8
  - privacy / false personalization: 8
- family coverage: 10 probes per habit family
- stress variants:
  - original balanced: 50
  - unseen paraphrase: 40

This subset is selected from the larger v0.3 balanced candidate, which is
auto-validated but still pending human audit. Use it for method feasibility,
cost, and failure-mode diagnosis; do not treat it as final benchmark evidence.

## Sync To Lumia

Preferred: create a minimal bundle that excludes local dry-run stores and old
pilot/result artifacts:

```bash
python ./scripts/lumia/make_lumia_bundle.py
```

Upload:

```bash
scp ./dist/lumia_bundle/habit-bench-lumia-full-official.tar.gz \
  USER@LUMIA:/path/to/
scp ./dist/lumia_bundle/habit-bench-lumia-full-official.tar.gz.sha256 \
  USER@LUMIA:/path/to/
scp ./dist/lumia_bundle/habit-bench-lumia-full-official.tar.gz.manifest.json \
  USER@LUMIA:/path/to/
scp ./dist/lumia_bundle/habit-bench-lumia-full-official.handoff.md \
  USER@LUMIA:/path/to/
```

On Lumia:

```bash
cd /path/to
sha256sum -c habit-bench-lumia-full-official.tar.gz.sha256
tar -xzf habit-bench-lumia-full-official.tar.gz
cd habit-bench-lumia-full-official
```

Local dry-run for the full upload/run/fetch command sequence:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host USER@LUMIA \
  --remote-dir /path/to/habitbench-lumia
```

Execute the remote launch after checking the printed commands:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host USER@LUMIA \
  --remote-dir /path/to/habitbench-lumia \
  --execute
```

The launcher uploads the tarball, sha256, manifest, and handoff sidecar when
present; verifies the sha256 remotely; runs the e2e script; fetches only the
returned `runs/lumia_manifests` and `full_official_results` subdirectories
instead of the remote virtualenv; and calls
`import_lumia_results.py` locally.

Current JumpServer/Slurm settings for the configured `lumia` SSH alias:

```bash
--scp "scp -O" \
--remote-env PYTHON_BIN=/home/jmzhang/miniconda3/bin/python \
--remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct \
--remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2 \
--remote-env HABITBENCH_EMBED_DIMS=768 \
--remote-env HABITBENCH_MAX_MODEL_LEN=16384 \
--remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256 \
--remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600 \
--remote-env HABITBENCH_PROGRESS_EVERY=100 \
```

Rationale:

- `scp -O` is required because Windows OpenSSH scp defaults to SFTP mode, while
  the Lumia JumpServer exposes the expected home path through legacy scp.
- `PYTHON_BIN=/home/jmzhang/miniconda3/bin/python` avoids the login node's
  system Python missing `ensurepip` / `python3-venv`.
- The `lumia` alias lands on `storage-hdd`; GPU work must go through Slurm.
  Short preflight/download checks can use `--remote-run-prefix "srun ..."`.
  Long full runs should use `--slurm-detached` so `sbatch` owns the process
  lifecycle.
- A smoke check reached `gpu-4090-2` with one RTX 4090 via
  `srun --partition=RTX4090 --gres=gpu:1`; the current full run uses L40S
  because the available local LLM is `Qwen2.5-14B-Instruct`.

Preferred single-command guarded cycle: start the gated detached full run, then
wait for `habitbench_remote_e2e.exitcode=0`, fetch, import, and run the final
audit:

```bash
python ./scripts/lumia/run_lumia_guarded_full_cycle.py \
  --host USER@LUMIA \
  --remote-dir /path/to/habitbench-lumia \
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

Run the same command without `--execute` first to inspect both child launcher
plans. The two explicit launcher commands below are useful when you want to
start the detached job and collect it in separate terminal sessions.

Before starting a long full-suite job, run remote preflight only. This uploads
the bundle, creates the venv, installs dependencies, checks dataset/source
contract readiness, checks GPU/disk/imports, and performs HuggingFace model
metadata preflight, but does not start vLLM or the method suite. By default it
fetches the returned bundle tree and runs `audit_lumia_preflight.py` locally:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host USER@LUMIA \
  --remote-dir /path/to/habitbench-lumia \
  --scp "scp -O" \
  --remote-env PYTHON_BIN=/home/jmzhang/miniconda3/bin/python \
  --remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct \
  --remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2 \
  --remote-env HABITBENCH_EMBED_DIMS=768 \
  --remote-env HABITBENCH_MAX_MODEL_LEN=16384 \
  --remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256 \
  --remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600 \
  --remote-env HABITBENCH_PROGRESS_EVERY=100 \
  --remote-run-prefix "srun --partition=RTX4090 --gres=gpu:1 --time=00:20:00 bash -lc" \
  --preflight-only \
  --execute
```

Require `./runs/lumia_preflight_import_audit.json` to
have `status=pass` before launching the detached full run. The companion
`./runs/lumia_preflight_import_audit.md` is the
human-readable handoff. Use `--skip-fetch` only when you intentionally want to
inspect the remote files manually.

To prove the open models can be downloaded/cached before starting vLLM or any
method run, use the model-download-only mode:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host USER@LUMIA \
  --remote-dir /path/to/habitbench-lumia \
  --scp "scp -O" \
  --remote-env PYTHON_BIN=/home/jmzhang/miniconda3/bin/python \
  --remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct \
  --remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2 \
  --remote-env HABITBENCH_EMBED_DIMS=768 \
  --remote-env HABITBENCH_MAX_MODEL_LEN=16384 \
  --remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256 \
  --remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600 \
  --remote-env HABITBENCH_PROGRESS_EVERY=100 \
  --remote-run-prefix "srun --partition=RTX4090 --gres=gpu:1 --time=02:00:00 bash -lc" \
  --download-models-only \
  --execute
```

Require `./runs/lumia_model_download_audit.json` to have
`status=pass`. The companion `.md` report summarizes downloaded model ids and
cache paths. A dry-run manifest is not enough: the full official audit requires
the download manifest to have top-level `status=pass`, `dry_run=false`, at
least two model rows, and `status=pass` plus a non-empty `cache_path` for every
row.

Preferred guarded launch: run remote preflight, download/cache open models,
fetch the returned manifests, audit both locally, and start the detached full
run only if both local audits pass:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host USER@LUMIA \
  --remote-dir /path/to/habitbench-lumia \
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

This is the safest one-command path when the open-weight models are not already
known to be cached on Lumia. After the two local audits pass, the detached full
run reuses the same remote workspace and sets `HABITBENCH_SKIP_MODEL_DOWNLOAD=1`
so the audited `model_download_manifest.json` is preserved for the final import
and full official audit.

You can also run the preflight and start the detached full run in one guarded
invocation. The launcher starts the detached full run only after the local
preflight audit command succeeds; use this only when model download has already
been audited separately:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host USER@LUMIA \
  --remote-dir /path/to/habitbench-lumia \
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

For long Lumia jobs, prefer detached mode so the SSH session can close after
the job is started. If the remote workspace and model/cache manifest are
already audited, this is the shortest direct submission path:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host USER@LUMIA \
  --remote-dir /path/to/habitbench-lumia \
  --scp "scp -O" \
  --remote-env HABITBENCH_LLM_MODEL=/home/jmzhang/models/Qwen2.5-14B-Instruct \
  --remote-env HABITBENCH_EMBED_MODEL=/home/jmzhang/models/e5-base-v2 \
  --remote-env HABITBENCH_EMBED_DIMS=768 \
  --remote-env HABITBENCH_MAX_MODEL_LEN=16384 \
  --remote-env HABITBENCH_MEMORY_LLM_MAX_TOKENS=256 \
  --remote-env HABITBENCH_OFFICIAL_TIMEOUT_SEC=21600 \
  --remote-env HABITBENCH_PROGRESS_EVERY=100 \
  --reuse-remote-workspace \
  --detached \
  --slurm-detached \
  --slurm-partition L40S \
  --slurm-gres gpu:1 \
  --slurm-time 06:00:00 \
  --slurm-job-name habitbench \
  --execute
```

Check whether the detached job is still running and tail the remote log:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host USER@LUMIA \
  --remote-dir /path/to/habitbench-lumia \
  --scp "scp -O" \
  --status-only \
  --execute
```

The status output also prints `exit_code=<N>` after the detached script exits.
Treat `status=not_running` without an exit-code file as incomplete evidence;
the process may have been killed before the wrapper wrote its completion code.

Preferred completion collector: poll the detached job, fetch the returned tree
only after `habitbench_remote_e2e.exitcode` contains `0`, and run the local
import/full official audit:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host USER@LUMIA \
  --remote-dir /path/to/habitbench-lumia \
  --scp "scp -O" \
  --wait-and-fetch \
  --wait-timeout-sec 86400 \
  --wait-poll-sec 300 \
  --execute
```

If the job is still running, has no exit-code file, or exits nonzero, this mode
does not import partial results.

After the status command shows the job is no longer running, fetch the returned
bundle tree and run the local import/audit:

```bash
python ./scripts/lumia/launch_lumia_remote.py \
  --host USER@LUMIA \
  --remote-dir /path/to/habitbench-lumia \
  --scp "scp -O" \
  --fetch-only \
  --execute
```

Current Lumia evidence as of 2026-06-13:

- Remote readiness manifest: pass.
- Remote GPU/dataset/import preflight on Slurm GPU nodes: pass.
- Hugging Face downloads require proxy to be disabled first on the known Lumia
  account. Run `proxy_off` from bashrc before model download; `proxy_on`
  restores the `192.168.102.101:7890` proxy when another network path needs it.
- Remote local-path model preflight/download audit: pass for
  `/home/jmzhang/models/Qwen2.5-14B-Instruct` and
  `/home/jmzhang/models/e5-base-v2`; the manifest is not dry-run.
- Full official subset results were imported locally and
  `runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results/audit/full_official_audit.json`
  has `status=pass`.

Alternative: rsync the working tree manually from the local workspace root.
`third_party/official-baselines` is optional for retrieval-adapter experiments that
depend on local cloned repos; the current full official suite only needs
`.`.

```bash
rsync -av ./ USER@LUMIA:/path/to/benchmark/habit-bench/
# Optional for retrieval-adapter experiments:
rsync -av third_party/official-baselines/ USER@LUMIA:/path/to/benchmark/third_party/official-baselines/
```

On Lumia:

```bash
cd /path/to/benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r ./requirements-official.txt
python ./scripts/lumia/check_lumia_readiness.py
python ./scripts/lumia/preflight_lumia_run.py
```

If `requirements-official.txt` has not been created yet, install the known
dependencies used by the current adapters:

```bash
python -m pip install mem0ai graphiti-core kuzu rank_bm25 tiktoken \
  sentence-transformers==2.7.0 chromadb==0.4.24 onnxruntime==1.17.3 \
  omegaconf langchain-core langchain-community llmlingua litellm ollama
```

## Open-Weight Model Setup

Use a single local model endpoint for all full official runs so memory systems
are compared under the same model budget.

Recommended first pass:

- generator / memory LLM: `Qwen/Qwen2.5-7B-Instruct` or
  `meta-llama/Llama-3.1-8B-Instruct` if available under local policy.
- embedding: `sentence-transformers/all-MiniLM-L6-v2` for continuity with
  current adapters; optionally repeat with `BAAI/bge-small-en-v1.5`.

Example vLLM server:

```bash
python -m pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b-instruct \
  --host 0.0.0.0 \
  --port 8000
```

Common environment:

```bash
source ./scripts/lumia/lumia_env_example.sh
```

Preflight model access before a long download:

```bash
python ./scripts/lumia/preflight_open_models.py
```

Download/cache models:

```bash
bash ./scripts/lumia/download_open_models.sh
```

This writes a model cache manifest to:

`./runs/lumia_manifests/model_download_manifest.json`

The download helper writes this manifest at the start and updates each model
row as it progresses. If a download fails, fetch the manifest anyway; its
`status=fail` row is the evidence needed to diagnose model-access or cache
problems. For completion evidence, the returned manifest must not be a dry-run
manifest and every required model row must report a cache path.

Start the OpenAI-compatible local model endpoint:

```bash
bash ./scripts/lumia/start_vllm_openai_server.sh
```

One-command alternative on Lumia:

```bash
source ./scripts/lumia/lumia_env_example.sh
bash ./scripts/lumia/run_lumia_full_official_e2e.sh
```

This downloads models unless `HABITBENCH_SKIP_MODEL_DOWNLOAD=1`, starts vLLM
unless `HABITBENCH_REUSE_SERVER=1`, runs a structured Lumia preflight, waits
for the endpoint, runs the full official suite, audits results, and stops the
vLLM process it started.

In a second shell, verify that `/v1/models` and `/v1/chat/completions` work
before launching full runs:

```bash
source ./scripts/lumia/lumia_env_example.sh
python ./scripts/lumia/check_openai_endpoint.py
```

## Current Adapter Baseline Commands

These commands reproduce current official-code storage/retrieval adapters on
the subset. They are not full paper reproductions, but they verify the runner
and provide comparable retrieval-layer numbers.

Convenience script:

```bash
bash ./scripts/run_official_subset_adapters.sh \
  ./runs/habit_bench_balanced_v0_3_official_subset_90
```

```bash
DATA=./runs/habit_bench_balanced_v0_3_official_subset_90
OUT=$DATA/official_results

python ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/mem0_infer_false_hf_qdrant" \
  --method-name official_mem0_infer_false_hf_qdrant \
  --command "python ./eval/official_adapters/official_mem0_adapter.py --input {input} --output {output}" \
  --adapter-note "Official Mem0 Memory.add infer=False and Memory.search retrieval adapter."

python ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/amem_search_agentic_no_evolution" \
  --method-name official_amem_search_agentic_no_evolution \
  --command "python ./eval/official_adapters/official_amem_adapter.py --input {input} --output {output}" \
  --adapter-note "Official A-MEM add_note/search_agentic retrieval adapter without LLM evolution."

python ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/secom_bm25_session" \
  --method-name official_secom_bm25_session \
  --command "python ./eval/official_adapters/official_secom_adapter.py --input {input} --output {output}" \
  --adapter-note "Official SeCom.retrieve_external_memory session BM25 adapter."

python ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/graphiti_kuzu_edge_cosine" \
  --method-name official_graphiti_kuzu_edge_cosine \
  --command "python ./eval/official_adapters/official_graphiti_adapter.py --input {input} --output {output}" \
  --adapter-note "Official Graphiti Kuzu EntityEdge storage and edge-cosine search adapter."

python ./eval/run_external_baseline.py \
  --dataset-dir "$DATA" \
  --output-dir "$OUT/omem_retrieval_injected_memory" \
  --method-name official_omem_retrieval_injected_memory \
  --command "python ./eval/official_adapters/official_omem_adapter.py --input {input} --output {output} --topn 12 --drop-threshold 0.0" \
  --adapter-note "Official O-Mem MemoryManager.retrieve_from_memory_soft_segmentation with injected visible sessions."
```

Collect current adapter results:

```bash
python ./eval/collect_official_results.py \
  --dataset-dir "$DATA" \
  --results-dir "$OUT"
```

## Full Official Config Target

A result should be labeled "full official subset" only if the method's core
write/update/reasoning path is enabled:

| method | full official subset requirement |
| --- | --- |
| Mem0 | enable LLM extraction/update, configured local LLM endpoint, same embedding model, and visible-history filtering |
| A-MEM | enable `process_memory` / evolution / linking with local LLM backend; do not use no-evolution adapter |
| SeCom | enable paper-style segmentation and compression dependencies, not just BM25 session retrieval |
| Zep/Graphiti | use `add_episode` or equivalent LLM extraction/KG resolution path, temporal invalidation where available, and a production graph backend if Kuzu blocks full hybrid search |
| O-Mem | use message understanding, active persona update, retrieval, and answer generation; do not inject memory structures directly |
| RMM | only run if official code is released or author code is provided via `RMM_REPO` |

## First Full Official Scaffold: Mem0

Mem0 now has a separate full-path scaffold that enables `Memory.add(...,
infer=True)` against the local OpenAI-compatible endpoint.

Run after the vLLM endpoint is live:

```bash
source ./scripts/lumia/lumia_env_example.sh
bash ./scripts/run_full_official_subset_mem0.sh \
  ./runs/habit_bench_balanced_v0_3_official_subset_90
```

Optional config dry-run before starting vLLM:

```bash
python ./eval/run_external_baseline.py \
  --dataset-dir ./runs/habit_bench_balanced_v0_3_official_subset_90 \
  --output-dir ./runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results_dryrun/mem0_full_llm_openai \
  --method-name official_mem0_full_llm_openai_dryrun \
  --command "python ./eval/official_adapters/official_mem0_full_llm_adapter.py --input {input} --output {output} --dry-run-config" \
  --adapter-note "Dry-run config check for Mem0 full LLM adapter; does not create Memory or call LLM endpoint."
```

The dry-run only validates runner coverage and config generation. Its accuracy
is meaningless and must not be cited.

Expected output directory:

`./runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results/mem0_full_llm_openai`

This is the first full official subset row because Mem0's official LLM
extraction/update path is enabled. The answer head is still the shared
HABIT-Bench lexical scorer over retrieved memories, so the claim is about
Mem0's official memory write/update/retrieval path rather than a fully
personalized dialogue policy.

## Full Official Suite Convenience Command

After the vLLM endpoint passes the health check, run all currently implemented
full-path scaffolds:

```bash
source ./scripts/lumia/lumia_env_example.sh
bash ./scripts/run_full_official_subset_suite.sh \
  ./runs/habit_bench_balanced_v0_3_official_subset_90
```

This currently runs:

- `official_mem0_full_llm_openai`
- `official_graphiti_full_llm_episode_kuzu`

It writes to:

`./runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results`

The suite also writes:

- `full_official_results/run_manifests/<run_id>/suite_start_manifest.json`
- `full_official_results/run_manifests/<run_id>/lumia_preflight_manifest.json`
- `full_official_results/run_manifests/<run_id>/openai_endpoint_check.json`
- `full_official_results/run_manifests/<run_id>/*_start_manifest.json`
- `full_official_results/run_manifests/<run_id>/*_end_manifest.json`
- `full_official_results/run_manifests/<run_id>/suite_end_manifest.json`
- `full_official_results/audit/full_official_audit.json`
- `full_official_results/audit/full_official_audit.md`

The e2e runner also writes
`full_official_results/run_manifests/<run_id>/e2e_end_manifest.json`. The e2e
and suite end manifests are written on both success and failure; require
`extra.exit_code=0` before treating them as completion evidence.

Copy results back from Lumia:

```bash
rsync -av USER@LUMIA:/path/to/benchmark/habit-bench/runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results/ \
  ./runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results/
rsync -av USER@LUMIA:/path/to/benchmark/habit-bench/runs/lumia_manifests/ \
  ./runs/lumia_manifests/
```

If the returned Lumia files were copied into a local directory that preserves
the `work/` tree, import and audit them with:

```bash
python ./scripts/lumia/import_lumia_results.py \
  --returned-root path/to/returned/lumia_bundle
```

## Second Full Official Scaffold: Graphiti

Graphiti now has a full-path scaffold that calls `Graphiti.add_episode(...)`
for LLM-backed entity/edge extraction, KG resolution, embedding generation, and
graph writes. It uses Graphiti's `OpenAIGenericClient`, so it targets local
OpenAI-compatible `/chat/completions` endpoints such as vLLM rather than the
OpenAI Responses API. It uses Kuzu locally and `Graphiti.search_` edge-cosine
retrieval because the local Kuzu BM25 full-text index path is unavailable.

The default structured-output mode is `json_schema`, which is suitable for vLLM
constrained decoding. If the serving backend rejects `json_schema`, set
`HABITBENCH_STRUCTURED_OUTPUT_MODE=json_object` before launching the run.

Run after the vLLM endpoint is live:

```bash
source ./scripts/lumia/lumia_env_example.sh
bash ./scripts/run_full_official_subset_graphiti.sh \
  ./runs/habit_bench_balanced_v0_3_official_subset_90
```

Optional config dry-run before starting vLLM:

```bash
python ./eval/run_external_baseline.py \
  --dataset-dir ./runs/habit_bench_balanced_v0_3_official_subset_90 \
  --output-dir ./runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results_dryrun/graphiti_full_llm_episode_kuzu \
  --method-name official_graphiti_full_llm_episode_kuzu_dryrun \
  --command "python ./eval/official_adapters/official_graphiti_full_llm_adapter.py --input {input} --output {output} --dry-run-config" \
  --adapter-note "Dry-run config check for Graphiti full LLM adapter; does not create Graphiti or call LLM endpoint."
```

The dry-run only validates runner coverage and config generation. Its accuracy
is meaningless and must not be cited.

## Measurement Requirements

For every full official run, record:

- exact command and git commit / artifact hash;
- model name, endpoint, decoding config, and embedding model;
- write-time LLM calls and retrieval-time LLM calls;
- wall-clock runtime;
- peak GPU memory if available;
- prompt/completion tokens if the endpoint reports usage;
- method-specific disabled components, if any.

Append results to:

`research/research-runs/2026-06-11-agent-memory-long-term-user-preference-benchmark/05_run_log.csv`

## Scale Decision

Proceed from the 90-probe subset to full v0.3 only if:

- all enabled full official methods finish without adapter contract violations;
- per-method cost is measurable;
- qualitative failure cases still include boundary, exception, drift, or privacy
  errors rather than only answer-head artifacts;
- at least one reviewer inspects a sample of v0.3 rows used by the official
  run.

## Completion Checklist

Before claiming this stage is complete, audit the run against:

`./docs/full_official_subset_completion_checklist.md`
