# official_results Baseline Report

This is an external official-code adapter run, not a lightweight proxy baseline.
Adapter note: Official Graphiti package code is used for Kuzu graph storage, EntityNode/EntityEdge writes, and Graphiti advanced search. LLM episode extraction and KG resolution are bypassed by storing each visible HABIT-Bench session as an EntityEdge fact with local sentence-transformers embeddings; search uses edge cosine only because the local Kuzu backend did not expose the BM25 full-text index needed by Graphiti's default hybrid search. This is an official-code graph storage/search adapter, not a full Zep/Graphiti reproduction.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_graphiti_kuzu_edge_cosine | 0.972 | 0.778 | 0.319 | 0.431 | 0.250 | 0.611 |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| official_graphiti_kuzu_edge_cosine | 0.972 | 0.388 | 0.585 | 0.378 |

## Variant Sensitivity

| baseline | original reviewed acc | unseen paraphrase acc |
| --- | ---: | ---: |
| official_graphiti_kuzu_edge_cosine | 0.627 | 0.472 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| official_graphiti_kuzu_edge_cosine | 115.8 | 56.8 |

## Interpretation

The table reports HABIT-Bench metrics for `official_graphiti_kuzu_edge_cosine` under the adapter contract above. Interpret results according to the adapter note rather than as a full paper-reproduction claim.
Official conclusions should be made only for the exact method configuration represented by this adapter run.
