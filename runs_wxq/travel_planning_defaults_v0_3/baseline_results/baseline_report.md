# travel_planning_defaults_v0_3 Baseline Report

These are lightweight method-inspired baselines, not official package integrations.
They are intended to validate the benchmark/evaluator loop before scaling.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory_lexical | 0.139 | 0.194 | 0.639 | 0.361 | nan | nan |
| full_history_segment_retrieval | 0.694 | 0.639 | 0.278 | 0.139 | nan | nan |
| mem0_like_fact_memory | 0.139 | 0.222 | 0.444 | 0.278 | nan | nan |
| zep_like_temporal_graph | 0.139 | 0.250 | 0.500 | 0.278 | nan | nan |
| a_mem_like_note_linking | 0.694 | 0.667 | 0.361 | 0.111 | nan | nan |
| secom_like_segment_memory | 0.639 | 0.500 | 0.528 | 0.222 | nan | nan |
| rmm_like_reflective_summary | 0.167 | 0.194 | 0.444 | 0.278 | nan | nan |
| o_mem_like_user_profile | 0.222 | 0.278 | 0.278 | 0.167 | nan | nan |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| a_mem_like_note_linking | 0.694 | 0.236 | 0.458 | 0.361 |
| full_history_segment_retrieval | 0.694 | 0.208 | 0.486 | 0.278 |
| mem0_like_fact_memory | 0.139 | 0.361 | -0.222 | 0.444 |
| no_memory_lexical | 0.139 | 0.500 | -0.361 | 0.639 |
| o_mem_like_user_profile | 0.222 | 0.222 | -0.000 | 0.278 |
| rmm_like_reflective_summary | 0.167 | 0.361 | -0.194 | 0.444 |
| secom_like_segment_memory | 0.639 | 0.375 | 0.264 | 0.528 |
| zep_like_temporal_graph | 0.139 | 0.389 | -0.250 | 0.500 |

## Variant Sensitivity

| baseline | travel_variable_profile_v03 acc |
| --- | ---: |
| no_memory_lexical | 0.343 |
| full_history_segment_retrieval | 0.423 |
| mem0_like_fact_memory | 0.293 |
| zep_like_temporal_graph | 0.313 |
| a_mem_like_note_linking | 0.443 |
| secom_like_segment_memory | 0.478 |
| rmm_like_reflective_summary | 0.293 |
| o_mem_like_user_profile | 0.274 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| a_mem_like_note_linking | 1408.2 | 72.2 |
| full_history_segment_retrieval | 20331.5 | 72.2 |
| mem0_like_fact_memory | 326.5 | 1.1 |
| no_memory_lexical | 0.0 | 0.0 |
| o_mem_like_user_profile | 326.5 | 1.0 |
| rmm_like_reflective_summary | 326.5 | 1.0 |
| secom_like_segment_memory | 1273.2 | 1248.5 |
| zep_like_temporal_graph | 326.5 | 1.1 |

## Interpretation

The strongest current evidence is for an explicit-vs-habit gap: fact/profile-style memories do well on explicit retrieval but degrade on boundary, exception, drift, and privacy/false-personalization stress cases.
Full-history and segment-retrieval baselines can handle some boundary/exception cases, which means later versions should add paraphrased, unseen, and cost-controlled stress tests before making claims about official systems.
