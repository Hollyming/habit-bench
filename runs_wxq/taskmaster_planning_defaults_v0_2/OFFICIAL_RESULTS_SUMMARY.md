# Taskmaster Planning Defaults v0.2 Official-Code Adapter Summary

Date: 2026-07-13

This note mirrors the `runs/habit_bench_curated_v0_2` official-results ledger for
the Taskmaster flights/hotels `planning_defaults` slice.

Important scope note: this run uses official storage/retrieval adapter code, but
it is not yet a full paper-reproduction run. Because HuggingFace model downloads
were blocked/unstable in the current environment, embedding-backed adapters used
a deterministic offline hash embedding fallback from `scripts_wxq/official_shims`.

## Dataset

- Dataset: `runs_wxq/taskmaster_planning_defaults_v0_2`
- Users: 30
- Sessions: 1080
- Probes: 120
- Probe types: 30 direct_use, 30 boundary, 30 exception, 30 explicit_retrieval
- Source domain: Taskmaster-2 flights + hotels
- Habit family: planning_defaults

## Official-Code Adapter Scope

| method | status | official code used | disabled / fallback |
| --- | --- | --- | --- |
| Mem0 | ran | `Memory.add(..., infer=False)` and `Memory.search` from the official OSS package; local Qdrant filtering | LLM extraction/update disabled; deterministic offline hash embedding fallback |
| A-MEM | ran | official `AgenticMemorySystem.add_note` and `search_agentic` from `agiresearch/a-mem`; Chroma retrieval | LLM evolution/linking disabled; deterministic offline hash embedding fallback |
| SeCom | ran | official `SeCom.retrieve_external_memory`; session-level BM25 retriever | LLM segmentation/compression disabled; token/compression shims |
| O-Mem | ran | official `SimpleMemory`, `MemoryChain`, `MemoryManager`, and `retrieve_from_memory_soft_segmentation` with injected visible sessions | LLM message understanding / active profiling / generation disabled; deterministic offline hash embedding fallback |
| Graphiti/Zep | not completed | official adapter attempted | blocked by missing `kuzu`; source build failed because system CMake is too old |
| RMM | not run | none | no official implementation path configured |

## Accuracy By Capability

| official-code adapter | overall | explicit | direct | boundary/false-pers | exception |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mem0 infer-false Qdrant, offline hash embedding | 0.550 | 0.600 | 0.467 | 0.633 | 0.500 |
| A-MEM search-agentic no-evolution, offline hash embedding | 0.550 | 0.600 | 0.467 | 0.633 | 0.500 |
| SeCom BM25 session retrieval | 0.542 | 0.600 | 0.533 | 0.667 | 0.367 |
| O-Mem injected retrieval, offline hash embedding | 0.417 | 0.533 | 0.400 | 0.433 | 0.300 |

## Diagnostic Gap

| official-code adapter | explicit acc | habit stress acc | explicit-minus-stress gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| Mem0 infer-false Qdrant, offline hash embedding | 0.600 | 0.567 | 0.033 | 0.633 |
| A-MEM search-agentic no-evolution, offline hash embedding | 0.600 | 0.567 | 0.033 | 0.633 |
| SeCom BM25 session retrieval | 0.600 | 0.517 | 0.083 | 0.667 |
| O-Mem injected retrieval, offline hash embedding | 0.533 | 0.367 | 0.167 | 0.433 |

## Cost Proxy

| official-code adapter | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| Mem0 infer-false Qdrant, offline hash embedding | 1956.4 | 36.0 |
| A-MEM search-agentic no-evolution, offline hash embedding | 1956.4 | 36.0 |
| SeCom BM25 session retrieval | 3761.5 | 36.0 |
| O-Mem injected retrieval, offline hash embedding | 7911.3 | 36.0 |

## Result Files

- Per-method results: `official_results_habit_official_env_offline_hash_v2/*`
- Collected summary: `official_results_habit_official_env_offline_hash_v2/collected/official_results_collected.md`
- Run status: `official_results_habit_official_env_offline_hash_v2/RUN_STATUS.md`

## Remaining To Match Curated v0.2 More Closely

1. Replace deterministic offline hash embeddings with real local HuggingFace
   `sentence-transformers/all-MiniLM-L6-v2` or another fixed local embedding
   model.
2. Install or expose a working `kuzu` backend, then rerun the Graphiti/Zep
   adapter.
3. If paper-level claims are needed, add a small LLM-backed official-method
   run for methods whose full behavior depends on LLM extraction, evolution,
   compression, or active profiling.

Until those are done, the current results should be cited as full-slice
official-code retrieval adapter validation, not final full official evaluation.
