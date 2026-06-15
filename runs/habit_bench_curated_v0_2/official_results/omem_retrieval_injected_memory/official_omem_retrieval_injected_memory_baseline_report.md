# official_results Baseline Report

This is an external official-code adapter run, not a lightweight proxy baseline.
Adapter note: Official O-Mem repository code is used for SimpleMemory/MemoryChain/MemoryManager.retrieve_from_memory_soft_segmentation. LLM-based message understanding, active persona update, and response generation are bypassed by injecting HABIT-Bench visible sessions into official working/episodic/persona memory structures; this is an official-code retrieval adapter, not full O-Mem active-profiling reproduction.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_omem_retrieval_injected_memory | 0.667 | 0.681 | 0.556 | 0.347 | 0.688 | 0.111 |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| official_omem_retrieval_injected_memory | 0.667 | 0.438 | 0.229 | 0.467 |

## Variant Sensitivity

| baseline | original reviewed acc | unseen paraphrase acc |
| --- | ---: | ---: |
| official_omem_retrieval_injected_memory | 0.571 | 0.472 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| official_omem_retrieval_injected_memory | 351.8 | 56.8 |

## Interpretation

The table reports HABIT-Bench metrics for `official_omem_retrieval_injected_memory` under the adapter contract above. Interpret results according to the adapter note rather than as a full paper-reproduction claim.
Official conclusions should be made only for the exact method configuration represented by this adapter run.
