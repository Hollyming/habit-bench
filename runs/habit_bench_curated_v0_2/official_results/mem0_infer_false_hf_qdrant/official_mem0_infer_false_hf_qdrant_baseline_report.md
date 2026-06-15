# official_results Baseline Report

This is an external official-code adapter run, not a lightweight proxy baseline.
Adapter note: Official Mem0 OSS Python API is used for Memory.add(..., infer=False) and Memory.search with local HuggingFace all-MiniLM-L6-v2 embeddings and local Qdrant metadata filtering. LLM fact extraction/update is disabled, so this is an official API retrieval adapter, not full Mem0 memory-extraction reproduction.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_mem0_infer_false_hf_qdrant | 0.972 | 0.778 | 0.319 | 0.431 | 0.250 | 0.611 |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| official_mem0_infer_false_hf_qdrant | 0.972 | 0.388 | 0.585 | 0.378 |

## Variant Sensitivity

| baseline | original reviewed acc | unseen paraphrase acc |
| --- | ---: | ---: |
| official_mem0_infer_false_hf_qdrant | 0.627 | 0.472 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| official_mem0_infer_false_hf_qdrant | 115.8 | 56.8 |

## Interpretation

The table reports HABIT-Bench metrics for `official_mem0_infer_false_hf_qdrant` under the adapter contract above. Interpret results according to the adapter note rather than as a full paper-reproduction claim.
Official conclusions should be made only for the exact method configuration represented by this adapter run.
