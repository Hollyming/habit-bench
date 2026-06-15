# Official Results Collection

- Dataset: `work/habit-bench-builder/runs/habit_bench_balanced_v0_3_official_subset_90`
- Results dir: `work/habit-bench-builder/runs/habit_bench_balanced_v0_3_official_subset_90/full_official_results`
- Methods found: 2

| method | status | overall | explicit | direct | stress | gap | false-pers ctrl | retrieved toks | stored items | config | runtime | stderr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| graphiti_full_llm_episode_kuzu | ok | 0.4556 | 0.6429 | 0.5625 | 0.3833 | 0.2596 | 0.2143 | 29.21 | 932.22 | True | True | True |
| mem0_full_llm_openai | ok | 0.5333 | 0.8571 | 0.5625 | 0.45 | 0.4071 | 0.1429 | 49.14 | 3815.0 | True | True | True |

## Interpretation Contract

Rows produced by current `official_adapters` are official-code storage/retrieval
adapter runs unless their method note explicitly states that full LLM-backed
write/update/reasoning paths were enabled. Do not cite adapter rows as full
paper reproductions.
