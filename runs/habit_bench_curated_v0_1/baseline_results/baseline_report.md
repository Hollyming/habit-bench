# HABIT-Bench Curated v0.1 Baseline Report

These are lightweight method-inspired baselines, not official package integrations.
They are intended to validate the benchmark/evaluator loop before scaling.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory_lexical | 0.306 | 0.222 | 0.667 | 0.583 | 0.000 | 0.000 |
| full_history_segment_retrieval | 0.806 | 0.583 | 0.722 | 0.750 | 0.500 | 0.000 |
| mem0_like_fact_memory | 0.972 | 0.667 | 0.361 | 0.139 | 0.000 | 0.111 |
| zep_like_temporal_graph | 0.944 | 0.639 | 0.417 | 0.222 | 0.000 | 0.000 |
| a_mem_like_note_linking | 0.806 | 0.583 | 0.722 | 0.750 | 0.375 | 0.000 |
| secom_like_segment_memory | 0.722 | 0.333 | 0.778 | 0.611 | 1.000 | 0.000 |
| rmm_like_reflective_summary | 0.972 | 0.667 | 0.361 | 0.083 | 0.000 | 0.111 |
| o_mem_like_user_profile | 0.889 | 0.694 | 0.194 | 0.028 | 0.000 | 0.444 |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| a_mem_like_note_linking | 0.806 | 0.629 | 0.176 | 0.578 |
| full_history_segment_retrieval | 0.806 | 0.640 | 0.165 | 0.578 |
| mem0_like_fact_memory | 0.972 | 0.213 | 0.759 | 0.311 |
| no_memory_lexical | 0.306 | 0.506 | -0.200 | 0.533 |
| o_mem_like_user_profile | 0.889 | 0.135 | 0.754 | 0.244 |
| rmm_like_reflective_summary | 0.972 | 0.191 | 0.781 | 0.311 |
| secom_like_segment_memory | 0.722 | 0.652 | 0.070 | 0.622 |
| zep_like_temporal_graph | 0.944 | 0.258 | 0.686 | 0.333 |

## Interpretation

The strongest current evidence is for an explicit-vs-habit gap: fact/profile-style memories do well on explicit retrieval but degrade on boundary, exception, drift, and privacy/false-personalization stress cases.
Full-history and segment-retrieval baselines can handle some boundary/exception cases, which means later versions should add paraphrased, unseen, and cost-controlled stress tests before making claims about official systems.
