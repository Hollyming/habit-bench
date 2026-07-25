# HABIT-Bench MultiDoGO Finance & Software v1.2.1

This is a compatibility and audit patch over v1.2.

## Core scale

- 54 coherent pseudo-users: 36 finance, 18 software.
- 540 sessions per user; 29,160 sessions total.
- 2,048 multiple-choice probes with exactly balanced A/B/C/D gold labels.
- 2,048 private multi-session evidence chains.

## v1.2.1 changes

1. The complete synthetic timeline was shifted by exactly 1,461 days, from 2025–2029 to 2021–2025. Relative ordering and every evidence gap are preserved. This avoids a needless conflict with an evaluator whose real current date is 2026.
2. All 960 as-of probes now state an exact date and minute. The previous month-only/day-only wording made 168 probes temporally ambiguous when a policy update occurred inside the displayed month/day.
3. Year-looking archive IDs such as `V11-SW-2033-10000` were replaced with neutral IDs such as `V12-SW-R10000`.
4. Every one of the 2,048 evidence chains is revalidated against visible shortlist text, distant ordinal resolution, temporal state, reference-case state, policy signatures, and the hidden exact-match gold.

## Evaluation inputs

Give a method only:

- `public/lifelines.jsonl`
- `public/probes.jsonl`

Do not expose the files under `private/` during the standard benchmark.

## Evidence-chain files

- `private/probes_with_evidence.jsonl`
- `private/probe_evidence_chains.jsonl`
- `private/probe_evidence_chain_edges.csv`

The private `session_id` list contains the decisive multi-session evidence chain.

## Scoring

Predictions are strict choice-ID exact match:

```json
{"probe_id":"mdgo_v11_probe_000000","choice_id":"A"}
```

```bash
python scripts/score_predictions.py --dataset-dir . --predictions predictions.jsonl --output-dir runs/eval --method-name my_method
```
