# official_results_habit_official_env_offline_hash_v2 Baseline Report

This is an external official-code adapter run, not a lightweight proxy baseline.
Adapter note: Official Mem0 Memory.add infer=False and Memory.search retrieval adapter with offline deterministic embedding fallback.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_mem0_infer_false_hf_qdrant | 0.600 | 0.467 | 0.633 | 0.500 | nan | nan |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| official_mem0_infer_false_hf_qdrant | 0.600 | 0.567 | 0.033 | 0.633 |

## Variant Sensitivity

| baseline | taskmaster_seeded_gpt55_xhigh_sessionwise_multihabit_v02 acc |
| --- | ---: |
| official_mem0_infer_false_hf_qdrant | 0.550 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| official_mem0_infer_false_hf_qdrant | 1956.4 | 36.0 |

## Interpretation

The table reports HABIT-Bench metrics for `official_mem0_infer_false_hf_qdrant` under the adapter contract above. Interpret results according to the adapter note rather than as a full paper-reproduction claim.
Official conclusions should be made only for the exact method configuration represented by this adapter run.
