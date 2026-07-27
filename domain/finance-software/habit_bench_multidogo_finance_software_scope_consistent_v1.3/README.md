# HABIT-Bench MultiDoGO Finance & Software v1.3

v1.3 is a small quality patch over v1.2.1. It retains 54 coherent pseudo-users, 29,160 sessions, 2,048 probes, all answer choices, all gold labels, and the existing evidence topology. The patch addresses two reviewer-identified ambiguities and adds a decision-chain-balanced exact-match view.

## Core scale

| Item | v1.3 |
|---|---:|
| Pseudo-users | 54 |
| Finance users | 36 |
| Software users | 18 |
| Sessions per user | 540 |
| Total sessions | 29,160 |
| Probes | 2,048 |
| Private evidence chains | 2,048 |
| Gold A/B/C/D | 512 each |

## v1.3 changes

1. **64 `scope_temporal_pair` probes now bind time scopes explicitly.** The first workstream uses the user's current standing process; the second uses the process in force at the stated historical timestamp. Historical deictic phrases such as “stands now” are rewritten to “stood at that time.” No workflow variant is revealed.
2. **The account-review scope for `mdgo_v05_fin_user_0006` is clarified.** Visible history now states that the habit covers balance, statement, transaction-history, and card-activity reconciliation. The later `POLR-winter-docket-giisxq` decision explicitly supersedes the earlier `POLB-winter-docket-giisxq` decision across that whole habit scope, rather than only for one statement example.
3. **Decision-unit metadata and chain-balanced exact-match scoring are added.** A decision unit is a unique `(user, habit, decisive evidence pair)`. The scorer continues to report ordinary per-probe exact-match accuracy and additionally reports `decision_unit_macro_accuracy`, so one repeatedly reused latent decision cannot dominate the aggregate score.
4. **All evidence-chain excerpts, review rows, and private enriched probes are regenerated after the history patch.**

## Evaluation inputs

Give a benchmarked method only:

- `public/lifelines.jsonl`
- `public/probes.jsonl`

Do not expose `private/` files during standard evaluation.

## Exact-match scoring

```json
{"probe_id":"mdgo_v11_probe_000000","choice_id":"A"}
```

```bash
python scripts/score_predictions.py --dataset-dir . --predictions predictions.jsonl --output-dir runs/eval --method-name my_method
```

The scorer uses strict choice-ID equality only. It reports:

- `probe_micro_accuracy`: backward-compatible exact accuracy over all 2,048 probes;
- `decision_unit_macro_accuracy`: exact accuracy macro-averaged over unique user–habit decision chains;
- `decision_bundle_macro_accuracy`: exact accuracy macro-averaged over unique multi-habit decision bundles.

No similarity score or partial credit is used.

## Evidence and audit files

- `private/probe_evidence_chains.jsonl`
- `private/probe_evidence_chain_edges.csv`
- `private/decision_unit_index.jsonl`
- `private/probe_decision_units.jsonl`
- `reports/scope_temporal_pair_binding_audit.csv`
- `reports/user0006_balance_scope_revalidation.json`
- `reports/user0006_balance_scope_probe_audit.csv`
- `reports/decision_unit_reuse_audit.csv`
- `reports/v13_quality_patch.md`

## Compatibility

Existing `user_id`, `session_id`, `probe_id`, gold labels, and evidence-chain IDs are preserved. Record IDs inside choices retain the inherited `V12-*` prefix because changing all 2,048 answer strings would be unrelated to this small patch.

## Status

`auto_validated_pending_target_model_baseline_and_human_audit`
