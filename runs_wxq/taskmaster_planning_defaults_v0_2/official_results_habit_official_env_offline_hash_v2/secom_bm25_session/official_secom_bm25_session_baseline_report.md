# official_results_habit_official_env_offline_hash_v2 Baseline Report

This is an external official-code adapter run, not a lightweight proxy baseline.
Adapter note: Official SeCom.retrieve_external_memory session BM25 adapter with offline tokenizer/compressor shims.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_secom_bm25_session | 0.600 | 0.533 | 0.667 | 0.367 | nan | nan |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| official_secom_bm25_session | 0.600 | 0.517 | 0.083 | 0.667 |

## Variant Sensitivity

| baseline | taskmaster_seeded_gpt55_xhigh_sessionwise_multihabit_v02 acc |
| --- | ---: |
| official_secom_bm25_session | 0.542 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| official_secom_bm25_session | 3761.5 | 36.0 |

## Interpretation

The table reports HABIT-Bench metrics for `official_secom_bm25_session` under the adapter contract above. Interpret results according to the adapter note rather than as a full paper-reproduction claim.
Official conclusions should be made only for the exact method configuration represented by this adapter run.
