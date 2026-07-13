# Official Results Collection

- Dataset: `/mnt/petrelfs/linzhouhan/xqwang/project/habit-bench/runs_wxq/taskmaster_planning_defaults_v0_2`
- Results dir: `/mnt/petrelfs/linzhouhan/xqwang/project/habit-bench/runs_wxq/taskmaster_planning_defaults_v0_2/official_results_habit_official_env_offline_hash_v2`
- Methods found: 4

| method | status | overall | explicit | direct | stress | gap | false-pers ctrl | retrieved toks | stored items | config | runtime | stderr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| amem_search_agentic_no_evolution | ok | 0.55 | 0.6 | 0.4667 | 0.5666 | 0.0334 | 0.6333 | 1956.44 | 36.0 | False | False | False |
| mem0_infer_false_hf_qdrant | ok | 0.55 | 0.6 | 0.4667 | 0.5666 | 0.0334 | 0.6333 | 1956.44 | 36.0 | False | False | True |
| omem_retrieval_injected_memory | ok | 0.4167 | 0.5333 | 0.4 | 0.3667 | 0.1666 | 0.4333 | 7911.29 | 36.0 | False | False | False |
| secom_bm25_session | ok | 0.5417 | 0.6 | 0.5333 | 0.5167 | 0.0833 | 0.6667 | 3761.51 | 36.0 | False | False | False |

## Interpretation Contract

Rows produced by current `official_adapters` are official-code storage/retrieval
adapter runs unless their method note explicitly states that full LLM-backed
write/update/reasoning paths were enabled. Do not cite adapter rows as full
paper reproductions.
