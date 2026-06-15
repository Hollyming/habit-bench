# full_official_results Baseline Report

This is an external official-code adapter run, not a lightweight proxy baseline.
Adapter note: Official Graphiti add_episode LLM extraction/KG resolution path with Kuzu backend and OpenAIGenericClient, run as four user shards and merged; retrieval uses Graphiti.search_ edge cosine because local Kuzu BM25 full-text index is unavailable.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_graphiti_full_llm_episode_kuzu | 0.643 | 0.562 | 0.300 | 0.375 | 1.000 | 0.000 |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| official_graphiti_full_llm_episode_kuzu | 0.643 | 0.383 | 0.260 | 0.214 |

## Variant Sensitivity

| baseline | original_balanced acc | unseen_paraphrase acc |
| --- | ---: | ---: |
| official_graphiti_full_llm_episode_kuzu | 0.540 | 0.350 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| official_graphiti_full_llm_episode_kuzu | 29.2 | 932.2 |

## Interpretation

The table reports HABIT-Bench metrics for `official_graphiti_full_llm_episode_kuzu` under the adapter contract above. Interpret results according to the adapter note rather than as a full paper-reproduction claim.
Official conclusions should be made only for the exact method configuration represented by this adapter run.
