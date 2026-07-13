# taskmaster_planning_defaults_v0_2 Baseline Report

These are lightweight method-inspired baselines, not official package integrations.
They are intended to validate the benchmark/evaluator loop before scaling.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory_lexical | 0.267 | 0.300 | 0.600 | 0.500 | nan | nan |
| full_history_segment_retrieval | 0.533 | 0.500 | 0.500 | 0.433 | nan | nan |
| mem0_like_fact_memory | 0.333 | 0.333 | 0.400 | 0.267 | nan | nan |
| zep_like_temporal_graph | 0.367 | 0.333 | 0.433 | 0.367 | nan | nan |
| a_mem_like_note_linking | 0.600 | 0.467 | 0.600 | 0.467 | nan | nan |
| secom_like_segment_memory | 0.367 | 0.300 | 0.600 | 0.500 | nan | nan |
| rmm_like_reflective_summary | 0.367 | 0.333 | 0.400 | 0.233 | nan | nan |
| o_mem_like_user_profile | 0.300 | 0.267 | 0.300 | 0.167 | nan | nan |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| a_mem_like_note_linking | 0.600 | 0.533 | 0.067 | 0.600 |
| full_history_segment_retrieval | 0.533 | 0.467 | 0.067 | 0.500 |
| mem0_like_fact_memory | 0.333 | 0.333 | -0.000 | 0.400 |
| no_memory_lexical | 0.267 | 0.550 | -0.283 | 0.600 |
| o_mem_like_user_profile | 0.300 | 0.233 | 0.067 | 0.300 |
| rmm_like_reflective_summary | 0.367 | 0.317 | 0.050 | 0.400 |
| secom_like_segment_memory | 0.367 | 0.550 | -0.183 | 0.600 |
| zep_like_temporal_graph | 0.367 | 0.400 | -0.033 | 0.433 |

## Variant Sensitivity

| baseline | taskmaster_seeded_gpt55_xhigh_sessionwise_multihabit_v02 acc |
| --- | ---: |
| no_memory_lexical | 0.417 |
| full_history_segment_retrieval | 0.492 |
| mem0_like_fact_memory | 0.333 |
| zep_like_temporal_graph | 0.375 |
| a_mem_like_note_linking | 0.533 |
| secom_like_segment_memory | 0.442 |
| rmm_like_reflective_summary | 0.333 |
| o_mem_like_user_profile | 0.258 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| a_mem_like_note_linking | 1819.3 | 36.0 |
| full_history_segment_retrieval | 14188.5 | 36.0 |
| mem0_like_fact_memory | 595.7 | 1.7 |
| no_memory_lexical | 0.0 | 0.0 |
| o_mem_like_user_profile | 595.7 | 1.0 |
| rmm_like_reflective_summary | 595.7 | 1.0 |
| secom_like_segment_memory | 1603.8 | 683.2 |
| zep_like_temporal_graph | 595.7 | 1.7 |

## Interpretation

The strongest current evidence is for an explicit-vs-habit gap: fact/profile-style memories do well on explicit retrieval but degrade on boundary, exception, drift, and privacy/false-personalization stress cases.
Full-history and segment-retrieval baselines can handle some boundary/exception cases, which means later versions should add paraphrased, unseen, and cost-controlled stress tests before making claims about official systems.
