# full_official_results Baseline Report

This is an external official-code adapter run, not a lightweight proxy baseline.
Adapter note: Official Mem0 Memory.add infer=True path with OpenAI-compatible local LLM endpoint for fact extraction/update, followed by official Memory.search retrieval. The answer head remains HABIT-Bench lexical scoring over retrieved memories.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_mem0_full_llm_openai | 0.857 | 0.562 | 0.200 | 0.625 | 1.000 | 0.000 |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| official_mem0_full_llm_openai | 0.857 | 0.450 | 0.407 | 0.143 |

## Variant Sensitivity

| baseline | original_balanced acc | unseen_paraphrase acc |
| --- | ---: | ---: |
| official_mem0_full_llm_openai | 0.640 | 0.400 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| official_mem0_full_llm_openai | 49.1 | 3815.0 |

## Interpretation

The table reports HABIT-Bench metrics for `official_mem0_full_llm_openai` under the adapter contract above. Interpret results according to the adapter note rather than as a full paper-reproduction claim.
Official conclusions should be made only for the exact method configuration represented by this adapter run.
