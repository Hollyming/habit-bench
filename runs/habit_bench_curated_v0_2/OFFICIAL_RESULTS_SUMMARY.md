# HABIT-Bench Curated v0.2 Official-Code Adapter Summary

Date: 2026-06-12

This note upgrades the v0.2 evidence ledger from proxy-only baselines to
official-code adapter runs where available. These runs are still not full
paper-reproduction runs: each adapter uses official storage/retrieval code but
keeps the HABIT-Bench answer head fixed to a simple lexical multiple-choice
scorer so that differences are attributable to the memory interface and
retrieved evidence.

## Dataset

- Dataset: `work/habit-bench-builder/runs/habit_bench_curated_v0_2`
- Users: 78
- Sessions: 4,438
- Probes: 286
- Probe variants: 161 reviewed originals + 125 unseen paraphrases

## Official-Code Adapter Scope

| method | official code used | disabled / not reproduced |
| --- | --- | --- |
| Mem0 | `Memory.add(..., infer=False)` and `Memory.search` from the official OSS Python package; local HuggingFace `all-MiniLM-L6-v2` embeddings; local Qdrant filtering by visible session index | LLM fact extraction, memory update/delete decisions, graph/entity enrichment |
| A-MEM | official `AgenticMemorySystem.add_note` and `search_agentic` from `agiresearch/a-mem`; Chroma retrieval | LLM-based `process_memory` evolution/linking, because it requires a live LLM backend during writes |
| SeCom | official `SeCom.retrieve_external_memory`; session-level BM25 retriever | LLM segmentation, LLMLingua compression, full paper configuration; an unused `vllm` import is shimmed on Windows |
| Zep/Graphiti | official Kuzu driver, `EntityNode`/`EntityEdge` writes, and Graphiti `search_` with edge cosine retrieval | LLM episode extraction, KG resolution/deduplication, temporal invalidation, production graph backend; local Kuzu BM25 full-text index was unavailable |
| O-Mem | official `SimpleMemory`, `MemoryChain`, `MemoryManager`, and `retrieve_from_memory_soft_segmentation`; visible HABIT-Bench sessions injected into working/episodic/persona memory structures | LLM message understanding, active persona update, response generation |
| RMM | no public official code found after local and web search | proxy only; cannot claim official-code run until authors release code or provide `RMM_REPO` |

## Accuracy By Capability

| official-code adapter | overall | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mem0 infer-false HF+Qdrant | 0.559 | 0.972 | 0.778 | 0.319 | 0.431 | 0.250 | 0.611 |
| Graphiti Kuzu edge cosine | 0.559 | 0.972 | 0.778 | 0.319 | 0.431 | 0.250 | 0.611 |
| A-MEM search-agentic no-evolution | 0.539 | 0.944 | 0.778 | 0.306 | 0.389 | 0.250 | 0.556 |
| SeCom BM25 session retrieval | 0.678 | 0.833 | 0.764 | 0.569 | 0.819 | 0.562 | 0.000 |
| O-Mem injected retrieval | 0.528 | 0.667 | 0.681 | 0.556 | 0.347 | 0.688 | 0.111 |

## Diagnostic Gap

| official-code adapter | explicit acc | habit stress acc | explicit-minus-stress gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| Mem0 infer-false HF+Qdrant | 0.972 | 0.388 | 0.585 | 0.378 |
| Graphiti Kuzu edge cosine | 0.972 | 0.388 | 0.585 | 0.378 |
| A-MEM search-agentic no-evolution | 0.944 | 0.360 | 0.585 | 0.356 |
| SeCom BM25 session retrieval | 0.833 | 0.612 | 0.221 | 0.456 |
| O-Mem injected retrieval | 0.667 | 0.438 | 0.229 | 0.467 |

Habit stress accuracy is the weighted aggregate over boundary/false
personalization, counterevidence exception, drift, and privacy/false
personalization probes.

## Variant Sensitivity

| official-code adapter | reviewed original acc | unseen paraphrase acc | drop |
| --- | ---: | ---: | ---: |
| Mem0 infer-false HF+Qdrant | 0.627 | 0.472 | 0.155 |
| Graphiti Kuzu edge cosine | 0.627 | 0.472 | 0.155 |
| A-MEM search-agentic no-evolution | 0.609 | 0.448 | 0.161 |
| SeCom BM25 session retrieval | 0.702 | 0.648 | 0.054 |
| O-Mem injected retrieval | 0.571 | 0.472 | 0.099 |

## Cost Proxy

| official-code adapter | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| Mem0 infer-false HF+Qdrant | 115.8 | 56.8 |
| Graphiti Kuzu edge cosine | 115.8 | 56.8 |
| A-MEM search-agentic no-evolution | 114.9 | 56.8 |
| SeCom BM25 session retrieval | 248.4 | 56.8 |
| O-Mem injected retrieval | 351.8 | 56.8 |

## Upgraded Evidence

The central HABIT-Bench claim is now supported by official-code retrieval
adapters, not only method-inspired proxies:

1. Explicit preference retrieval is substantially easier than scoped habit
   use under boundary, exception, drift, and privacy controls.
2. Semantic raw-memory adapters based on official Mem0 and A-MEM code show
   high explicit accuracy but large stress gaps around 0.585.
3. Official Graphiti graph storage/search collapses to the same score pattern
   as Mem0 when each session is stored as a raw edge fact, showing that a graph
   container alone does not solve scoped habit policy induction.
4. Official SeCom BM25 retrieval is more robust on exception and boundary
   probes, but its privacy/false-personalization score is 0.000 under the
   fixed lexical answer head, indicating that retrieval alone does not solve
   consent-sensitive habit policy decisions.
5. Official O-Mem retrieval is more balanced across direct/boundary/drift than
   raw semantic retrieval, but remains weak on exception and privacy controls.
6. Unseen paraphrase remains a meaningful stressor, especially for semantic
   raw-memory adapters.

## New Capability Problem

HABIT-Bench should be framed as testing **Longitudinal Habit Policy
Induction**, not generic long-context retrieval. A memory system must infer a
stable but scoped user policy from repeated interactions and decide when to
apply, withhold, update, or forget that policy.

The capability decomposes into:

- support accumulation: distinguish one-off facts from repeated habits;
- scope calibration: apply a habit only inside its intended context;
- counterevidence handling: separate exceptions from genuine preference change;
- temporal drift: prefer latest sustained evidence when habits evolve;
- consent-sensitive memory: avoid durable use of sensitive one-off traits;
- paraphrase robustness: generalize beyond surface overlap;
- budgeted evidence use: solve the above without full-history context.

## Remaining Officialization Gaps

- Full Mem0 requires LLM extraction/update, optional entity/BM25 enrichment, and
  graph memory if evaluated.
- Full A-MEM requires live LLM evolution/linking, not just Chroma retrieval.
- Full SeCom requires LLM segmentation and LLMLingua compression.
- Full Zep/Graphiti requires LLM episode extraction, KG resolution, temporal
  invalidation, and a production graph backend such as Neo4j/FalkorDB; the
  local Kuzu default hybrid BM25 index path failed, so the adapter uses official
  edge cosine search.
- Full O-Mem requires LLM message understanding, active persona update, and
  generation, not just injected-memory retrieval.
- RMM still needs an official released implementation or author-provided code;
  no public official repository was found.

The next decisive experiment is a small but fully faithful LLM-backed run for
Mem0, A-MEM, SeCom, Graphiti, and O-Mem on a 30-50 probe stratified subset,
followed by the full v0.2 run if costs and latency are acceptable.
