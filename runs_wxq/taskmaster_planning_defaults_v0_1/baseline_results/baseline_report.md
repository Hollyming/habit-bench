# taskmaster_planning_defaults_v0_1 Baseline Report

These are lightweight method-inspired baselines, not official package integrations.
They are intended to validate the benchmark/evaluator loop before scaling.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory_lexical | 0.000 | 0.000 | 0.000 | 0.000 | nan | nan |
| full_history_segment_retrieval | 0.233 | 0.333 | 0.300 | 0.067 | nan | nan |
| mem0_like_fact_memory | 1.000 | 1.000 | 0.000 | 0.200 | nan | nan |
| zep_like_temporal_graph | 1.000 | 1.000 | 0.000 | 0.200 | nan | nan |
| a_mem_like_note_linking | 0.333 | 0.500 | 0.467 | 0.167 | nan | nan |
| secom_like_segment_memory | 0.000 | 0.200 | 0.033 | 0.033 | nan | nan |
| rmm_like_reflective_summary | 1.000 | 1.000 | 0.000 | 0.200 | nan | nan |
| o_mem_like_user_profile | 1.000 | 1.000 | 0.000 | 0.200 | nan | nan |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| a_mem_like_note_linking | 0.333 | 0.317 | 0.017 | 0.467 |
| full_history_segment_retrieval | 0.233 | 0.183 | 0.050 | 0.300 |
| mem0_like_fact_memory | 1.000 | 0.100 | 0.900 | 0.000 |
| no_memory_lexical | 0.000 | 0.000 | 0.000 | 0.000 |
| o_mem_like_user_profile | 1.000 | 0.100 | 0.900 | 0.000 |
| rmm_like_reflective_summary | 1.000 | 0.100 | 0.900 | 0.000 |
| secom_like_segment_memory | 0.000 | 0.033 | -0.033 | 0.033 |
| zep_like_temporal_graph | 1.000 | 0.100 | 0.900 | 0.000 |

## Variant Sensitivity

| baseline | taskmaster_seeded_boundary acc | taskmaster_seeded_direct acc | taskmaster_seeded_exception acc | taskmaster_seeded_explicit acc |
| --- | ---: | ---: | ---: | ---: |
| no_memory_lexical | 0.000 | 0.000 | 0.000 | 0.000 |
| full_history_segment_retrieval | 0.300 | 0.333 | 0.067 | 0.233 |
| mem0_like_fact_memory | 0.000 | 1.000 | 0.200 | 1.000 |
| zep_like_temporal_graph | 0.000 | 1.000 | 0.200 | 1.000 |
| a_mem_like_note_linking | 0.467 | 0.500 | 0.167 | 0.333 |
| secom_like_segment_memory | 0.033 | 0.200 | 0.033 | 0.000 |
| rmm_like_reflective_summary | 0.000 | 1.000 | 0.200 | 1.000 |
| o_mem_like_user_profile | 0.000 | 1.000 | 0.200 | 1.000 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| a_mem_like_note_linking | 190.9 | 36.0 |
| full_history_segment_retrieval | 1111.9 | 36.0 |
| mem0_like_fact_memory | 203.1 | 5.0 |
| no_memory_lexical | 0.0 | 0.0 |
| o_mem_like_user_profile | 203.1 | 1.0 |
| rmm_like_reflective_summary | 203.1 | 1.0 |
| secom_like_segment_memory | 189.8 | 80.0 |
| zep_like_temporal_graph | 203.1 | 5.0 |

## Interpretation

The strongest current evidence is for an explicit-vs-habit gap: fact/profile-style memories do well on explicit retrieval but degrade on boundary, exception, drift, and privacy/false-personalization stress cases.
Full-history and segment-retrieval baselines can handle some boundary/exception cases, which means later versions should add paraphrased, unseen, and cost-controlled stress tests before making claims about official systems.
