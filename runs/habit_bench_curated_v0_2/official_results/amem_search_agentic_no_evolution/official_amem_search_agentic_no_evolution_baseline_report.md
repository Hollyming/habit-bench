# official_results Baseline Report

This is an external official-code adapter run, not a lightweight proxy baseline.
Adapter note: Official A-MEM repository code is used for AgenticMemorySystem.add_note and search_agentic with Chroma retrieval. LLM-based process_memory/evolution is disabled because it requires a live backend during memory writes; this is an official-code retrieval adapter, not full A-MEM evolution reproduction.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_amem_search_agentic_no_evolution | 0.944 | 0.778 | 0.306 | 0.389 | 0.250 | 0.556 |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| official_amem_search_agentic_no_evolution | 0.944 | 0.360 | 0.585 | 0.356 |

## Variant Sensitivity

| baseline | original reviewed acc | unseen paraphrase acc |
| --- | ---: | ---: |
| official_amem_search_agentic_no_evolution | 0.609 | 0.448 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| official_amem_search_agentic_no_evolution | 114.9 | 56.8 |

## Interpretation

The table reports HABIT-Bench metrics for `official_amem_search_agentic_no_evolution` under the adapter contract above. Interpret results according to the adapter note rather than as a full paper-reproduction claim.
Official conclusions should be made only for the exact method configuration represented by this adapter run.
