# Official Results Collection

- Dataset: `work\habit-bench-builder\runs\habit_bench_curated_v0_2`
- Results dir: `work\habit-bench-builder\runs\habit_bench_curated_v0_2\official_results`
- Methods found: 5

| method | status | overall | explicit | direct | stress | gap | false-pers ctrl | retrieved toks | stored items | stderr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| amem_search_agentic_no_evolution | ok | 0.5385 | 0.9444 | 0.7778 | 0.3596 | 0.5848 | 0.3556 | 114.94 | 56.78 | True |
| graphiti_kuzu_edge_cosine | ok | 0.5594 | 0.9722 | 0.7778 | 0.3876 | 0.5846 | 0.3777 | 115.77 | 56.78 | False |
| mem0_infer_false_hf_qdrant | ok | 0.5594 | 0.9722 | 0.7778 | 0.3876 | 0.5846 | 0.3777 | 115.77 | 56.78 | True |
| omem_retrieval_injected_memory | ok | 0.528 | 0.6667 | 0.6806 | 0.4382 | 0.2285 | 0.4667 | 351.83 | 56.78 | True |
| secom_bm25_session | ok | 0.6783 | 0.8333 | 0.7639 | 0.6123 | 0.221 | 0.4555 | 248.45 | 56.78 | False |

## Interpretation Contract

Rows produced by current `official_adapters` are official-code storage/retrieval
adapter runs unless their method note explicitly states that full LLM-backed
write/update/reasoning paths were enabled. Do not cite adapter rows as full
paper reproductions.
