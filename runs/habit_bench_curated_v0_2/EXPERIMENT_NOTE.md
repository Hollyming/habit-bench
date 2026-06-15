# HABIT-Bench Curated v0.2 Experiment Note

Date: 2026-06-12

## What Changed From v0.1

v0.2 keeps the reviewed v0.1 histories and gold labels, then adds 125
deterministic `unseen_paraphrase` stress probes for:

- direct habit use;
- boundary / false personalization;
- exception / counterevidence;
- drift;
- privacy / false personalization.

The goal is to reduce surface overlap between test queries and visible history
episodes, so raw episode/segment retrieval cannot rely as much on exact wording.

## Dataset Size

- Original v0.1 probes: 161
- New unseen-paraphrase probes: 125
- Total v0.2 probes: 286
- Visible sessions: inherited from v0.1

## Baseline Result

The explicit-vs-habit gap is stronger for fact/profile-style memory proxies:

| baseline | explicit acc | habit stress acc | gap |
| --- | ---: | ---: | ---: |
| mem0_like_fact_memory | 0.972 | 0.180 | 0.792 |
| zep_like_temporal_graph | 0.944 | 0.225 | 0.720 |
| rmm_like_reflective_summary | 0.972 | 0.174 | 0.798 |
| o_mem_like_user_profile | 0.889 | 0.135 | 0.754 |

Unseen paraphrase lowers proxy performance:

| baseline | original acc | unseen paraphrase acc |
| --- | ---: | ---: |
| mem0_like_fact_memory | 0.485 | 0.296 |
| zep_like_temporal_graph | 0.497 | 0.320 |
| rmm_like_reflective_summary | 0.472 | 0.304 |
| o_mem_like_user_profile | 0.429 | 0.296 |
| a_mem_like_note_linking | 0.658 | 0.536 |
| full_history_segment_retrieval | 0.665 | 0.544 |

Segment/episode retrieval remains more robust than profile memories on some
boundary/exception cases, but v0.2 now exposes a clearer paraphrase sensitivity
and much higher estimated context cost:

- `full_history_segment_retrieval`: ~817 retrieved tokens/probe.
- `a_mem_like_note_linking`: ~86 retrieved tokens/probe.
- `secom_like_segment_memory`: ~76 retrieved tokens/probe, ~134 stored items.
- `mem0_like` / `zep_like`: ~102-104 retrieved tokens/probe, ~12 stored items.
- `rmm_like` / `o_mem_like`: ~105 retrieved tokens/probe, 1 profile item.

## Claim Status

Supported:

- Fact/profile style memory proxies are strong on explicit preference retrieval
  and weak on boundary, exception, drift, and privacy stress cases.
- Unseen paraphrase makes the benchmark harder and reduces reliance on exact
  episode matching.
- Cost proxies distinguish full-history/segment-heavy methods from compact
  profile memories.
- Official-code adapters for Mem0, A-MEM, SeCom, Zep/Graphiti, and O-Mem now
  reproduce the key explicit-vs-habit stress gap or adjacent failure modes
  under their adapter contracts.
- Zep/Graphiti is feasible as an official graph storage/search adapter, but
  raw session facts stored as graph edges do not improve scoped habit policy
  induction over raw semantic memory.
- O-Mem is feasible as an official retrieval adapter and is more balanced on
  direct/boundary/drift than raw semantic retrieval, but remains weak on
  exception and privacy controls.

Still not supported:

- Full paper-reproduction claims for Mem0, A-MEM, SeCom, Zep/Graphiti, RMM, or
  O-Mem. The current official-code adapters exercise official storage/retrieval
  code, but disable or omit LLM extraction, evolution, segmentation,
  compression, KG resolution, temporal invalidation, active profiling, or
  generation where noted.
- Any official-code claim for RMM. No public official RMM implementation was
  found; only a method-inspired proxy is available unless authors release code
  or an `RMM_REPO` path is provided.
- A final claim that all segment-retrieval systems fail habit boundary or
  counterevidence. Some remain competitive on v0.2 and need stronger tests.

## New Code

- `scripts/derive_v02_stress_dataset.py`: derives v0.2 unseen-paraphrase probes.
- `scripts/build_balanced_v03_dataset.py`: builds a larger 9-family balanced
  v0.3 candidate from the auto-validated pilot pool.
- `scripts/make_official_subset.py`: creates a stratified subset for expensive
  full official-method runs.
- `eval/evaluate_baselines.py`: now reports stress variant and cost proxies.
- `eval/official_adapter_status.py`: checks official baseline dependency/repo readiness.
- `eval/run_external_baseline.py`: generic runner for official repo/package commands.
- `eval/official_adapters/official_mem0_adapter.py`: official Mem0 API retrieval adapter.
- `eval/official_adapters/official_amem_adapter.py`: official A-MEM retrieval adapter.
- `eval/official_adapters/official_secom_adapter.py`: official SeCom retrieval adapter.
- `eval/official_adapters/official_graphiti_adapter.py`: official Graphiti Kuzu edge storage/search adapter.
- `eval/official_adapters/official_omem_adapter.py`: official O-Mem retrieval adapter with injected visible sessions.
- `docs/9_family_taxonomy.md`: unified 9-family implementation taxonomy and
  domain mapping.
- `docs/lumia_full_official_subset_runbook.md`: runbook for full official
  subset experiments on Lumia/server hardware.
- `OFFICIAL_RESULTS_SUMMARY.md`: unified official-code adapter result summary.

## Next Split Prepared

A larger balanced candidate split now exists at:

`work/habit-bench-builder/runs/habit_bench_balanced_v0_3`

It contains 165 users, 9,390 sessions, and 2,010 probes, with 30 selected
habits per implementation family. This split is auto-validated and balanced,
but still pending human audit; v0.2 remains the stronger reviewed evidence set.

A 90-probe stratified subset for full official-method runs now exists at:

`work/habit-bench-builder/runs/habit_bench_balanced_v0_3_official_subset_90`

It is intended for Lumia/server runs with full LLM-backed official
configurations before scaling to the full v0.3 candidate.

## Official Baseline Status

The current environment has runnable official-code adapters for Mem0, A-MEM,
SeCom, Zep/Graphiti, and O-Mem. RMM remains unavailable as an official-code run
because no public implementation was found. The unified result summary is at:

`work/habit-bench-builder/runs/habit_bench_curated_v0_2/OFFICIAL_RESULTS_SUMMARY.md`

The latest dependency/repo readiness report is at:

`work/habit-bench-builder/runs/official_adapter_status/official_adapter_status.md`

The next decisive step for full official claims is to run LLM-backed/full
configuration adapters for Mem0, A-MEM, SeCom, Zep/Graphiti, and O-Mem on a
small stratified subset before scaling to the full v0.2 split.
