# official_results_habit_official_env_offline_hash_v2 Baseline Report

This is an external official-code adapter run, not a lightweight proxy baseline.
Adapter note: Official O-Mem MemoryManager.retrieve_from_memory_soft_segmentation with injected visible sessions and offline deterministic embedding fallback.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_omem_retrieval_injected_memory | 0.533 | 0.400 | 0.433 | 0.300 | nan | nan |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| official_omem_retrieval_injected_memory | 0.533 | 0.367 | 0.167 | 0.433 |

## Variant Sensitivity

| baseline | taskmaster_seeded_gpt55_xhigh_sessionwise_multihabit_v02 acc |
| --- | ---: |
| official_omem_retrieval_injected_memory | 0.417 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| official_omem_retrieval_injected_memory | 7911.3 | 36.0 |

## Interpretation

The table reports HABIT-Bench metrics for `official_omem_retrieval_injected_memory` under the adapter contract above. Interpret results according to the adapter note rather than as a full paper-reproduction claim.
Official conclusions should be made only for the exact method configuration represented by this adapter run.
