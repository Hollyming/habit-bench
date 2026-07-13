# Official Adapter Run Status

Date: 2026-07-13

## Scope

- Dataset: `runs_wxq/taskmaster_planning_defaults_v0_2`
- Full slice: 30 users, 1080 sessions, 120 probes
- Environment: `/mnt/petrelfs/linzhouhan/xqwang/conda_envs/habit-official`
- Official repos: `/mnt/petrelfs/linzhouhan/xqwang/project/official-baselines`
- Result dir: `official_results_habit_official_env_offline_hash_v2`

## Environment Handling

The official adapter dependencies were made runnable with minimal changes:

- Used the user-provided `habit-official` conda environment.
- Used cloned official repos through `AMEM_REPO`, `SECOM_REPO`, and `OMEM_REPO`.
- Added local shims under `scripts_wxq/official_shims` for unavailable or unused runtime-heavy modules:
  - `litellm`, `ollama`: no-LLM retrieval paths only.
  - `tiktoken`, `llmlingua`: SeCom retrieval-only token/compression fallback.
  - `vllm`, `FlagEmbedding`: imported by official code but unused by the adapter path.
  - `sentence_transformers`: deterministic offline hash embeddings because HuggingFace model downloads were blocked/unstable.
  - `nltk`: minimal tokenization/stopword fallback to avoid external NLTK data downloads.
- Downgraded `numpy` to `1.26.4` to make it compatible with the existing `torch==2.0.1`.

These changes do not modify the dataset or official repositories.

## Successful Runs

| method | status | overall | explicit | direct | stress | gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mem0 infer-false HF+Qdrant adapter | ok | 0.5500 | 0.6000 | 0.4667 | 0.5666 | 0.0334 |
| A-MEM search-agentic no-evolution adapter | ok | 0.5500 | 0.6000 | 0.4667 | 0.5666 | 0.0334 |
| SeCom BM25 session adapter | ok | 0.5417 | 0.6000 | 0.5333 | 0.5167 | 0.0833 |
| O-Mem injected retrieval adapter | ok | 0.4167 | 0.5333 | 0.4000 | 0.3667 | 0.1666 |

Collected summary:

- `collected/official_results_collected.md`
- `collected/official_results_collected.csv`
- `collected/official_results_collected.json`

## Failed / Not Run

| method | status | reason |
| --- | --- | --- |
| Graphiti/Zep Kuzu adapter | failed | Python package `kuzu` was missing; installing `kuzu` fell back to source build and failed because the available system CMake is too old. |
| RMM | not run | No official implementation path is configured. |

## Interpretation

These are official-code storage/retrieval adapter runs, not full paper reproduction runs. In particular:

- LLM extraction/evolution/generation paths are disabled or shimmed.
- The embedding-backed adapters used deterministic offline hash embeddings, not downloaded HuggingFace embeddings.
- SeCom uses the official BM25 retrieval path with token/compression shims.

The run is still useful as a formal evaluator-loop check on the full 120-probe planning_defaults slice. For publication-strength official claims, rerun with real HuggingFace embedding models and the Graphiti Kuzu backend once the environment supports them.
