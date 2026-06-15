# HABIT-Bench

HABIT-Bench is a real-prompt-seeded benchmark for long-horizon agent memory.
It tests whether memory systems can infer scoped user habits from many
user-agent interactions, apply them in the right context, and avoid false
personalization under boundary, exception, drift, and privacy cases.

## Current Dataset Contract

- Real prompt seed source: `allenai/WildChat`.
- Release framing: real-prompt-seeded, domain-grounded, synthetic longitudinal benchmark.
- Family/domain contract: 9 habit families mapped to 9 unique representative WildChat domain buckets.
- Synthetic controlled components: hidden habit graphs, assistant feedback, counterfactual probes, answer choices, gold labels, and evidence links.
- Claim to avoid: the 9 families are not drawn from 9 separate external datasets.

## Project Layout

| path | purpose |
| --- | --- |
| `README.md` | project overview and common commands |
| `docs/` | paper-facing taxonomy, unified 9-family table, Lumia runbook, completion checklist |
| `schema/` | JSON schemas for public sessions and probes |
| `scripts/` | dataset construction, curation, subset building, provenance audit, and method-run launchers |
| `scripts/lumia/` | Lumia packaging, remote launch, model download, vLLM serving, preflight, import, and audit helpers |
| `eval/` | scoring, baseline execution, official-result collection, and official adapter code |
| `eval/official_adapters/` | adapters for Mem0, Graphiti, A-MEM, SeCom, O-Mem, and related official-code paths |
| `runs/` | generated datasets, final experiment results, manifests, and audit summaries |
| `dist/` | generated release bundles and handoff sidecars |
| `third_party/official-baselines/` | optional local official baseline repositories |
| `research/research-runs/` | literature survey, gap map, idea bank, experiment plan, and research memo |
| `archive/2026-06-transient-smoke/` | archived smoke-test scripts, dry-run plans, returned debug payloads, and other transient development artifacts |

The root `habit-bench/` directory is now the project root. Run commands below
from this directory unless a command says otherwise.

## Key Dataset Artifacts

| artifact | status |
| --- | --- |
| `runs/habit_bench_balanced_v0_3` | balanced v0.3 candidate: 174 users, 9,924 sessions, 270 habits, 2,010 probes |
| `runs/habit_bench_balanced_v0_3_official_subset_90` | official subset: 67 users, 3,815 sessions, 90 probes |
| `runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results` | Lumia full official subset results |
| `runs/goal_status.json` | objective-level completion ledger |

## Common Commands

Check that the official subset and Lumia scripts are structurally ready:

```bash
python scripts/lumia/check_lumia_readiness.py
```

Audit the completed full official subset results:

```bash
python eval/audit_full_official_results.py \
  --dataset-dir runs/habit_bench_balanced_v0_3_official_subset_90 \
  --results-dir runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results \
  --model-manifest runs/lumia_manifests/model_download_manifest.json
```

Collect result summaries:

```bash
python eval/collect_official_results.py \
  --dataset-dir runs/habit_bench_balanced_v0_3_official_subset_90 \
  --results-dir runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results
```

Build and verify a clean Lumia bundle:

```bash
python scripts/lumia/make_lumia_bundle.py
python scripts/lumia/write_lumia_handoff.py --host-placeholder lumia --remote-dir /home/jmzhang/habitbench-lumia
python scripts/lumia/verify_lumia_bundle.py
```

Run the full official suite on a machine that already has an OpenAI-compatible
LLM endpoint available:

```bash
source scripts/lumia/lumia_env_example.sh
bash scripts/run_full_official_subset_suite.sh
```

## Lumia Notes

The current Lumia path uses vLLM plus an OpenAI-compatible local endpoint.
`scripts/lumia/lumia_env_example.sh` sets the default dataset and serving
environment. On the known Lumia account, Hugging Face download requires proxy
to be off:

```bash
proxy_off
bash scripts/lumia/download_open_models.sh
```

`proxy_on` restores the `192.168.102.101:7890` proxy if needed for other
network paths.

## Smoke-Test Cleanup Policy

Smoke-test scripts and dry-run outputs were useful during development, but they
are not part of the normal project surface. They have been moved under:

```text
archive/2026-06-transient-smoke/
```

The main project keeps only release-oriented checks:

- `scripts/lumia/check_lumia_readiness.py`
- `scripts/lumia/verify_lumia_bundle.py`
- `eval/audit_full_official_results.py`
- `eval/collect_official_results.py`

## Rebuilding Core Data

The main data-generation pipeline is:

```bash
python scripts/build_habit_bench_pilot.py \
  --out-dir runs/habit_bench_pilot_v0 \
  --n-users 200 \
  --sessions-per-user 60 \
  --seed-prompts 5000 \
  --source auto \
  --seed 20260612

python scripts/build_balanced_v03_dataset.py \
  --input-dir runs/habit_bench_pilot_v0 \
  --output-dir runs/habit_bench_balanced_v0_3 \
  --target-habits-per-family 30 \
  --review-sample-rate 0.10 \
  --seed 20260612

python scripts/make_official_subset.py \
  --input-dir runs/habit_bench_balanced_v0_3 \
  --output-dir runs/habit_bench_balanced_v0_3_official_subset_90 \
  --total-probes 90 \
  --min-per-capability 8 \
  --include-variants all \
  --seed 20260612
```

Before paper-scale claims, run the provenance checks and human audit described
in `docs/9_family_taxonomy.md` and `docs/full_official_subset_completion_checklist.md`.
