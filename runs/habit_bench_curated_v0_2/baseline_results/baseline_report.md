# habit_bench_curated_v0_2 Baseline Report

These are lightweight method-inspired baselines, not official package integrations.
They are intended to validate the benchmark/evaluator loop before scaling.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_memory_lexical | 0.306 | 0.319 | 0.667 | 0.444 | 0.000 | 0.000 |
| full_history_segment_retrieval | 0.806 | 0.542 | 0.611 | 0.778 | 0.438 | 0.000 |
| mem0_like_fact_memory | 0.972 | 0.667 | 0.347 | 0.083 | 0.000 | 0.056 |
| zep_like_temporal_graph | 0.944 | 0.639 | 0.431 | 0.125 | 0.000 | 0.000 |
| a_mem_like_note_linking | 0.806 | 0.556 | 0.597 | 0.778 | 0.312 | 0.000 |
| secom_like_segment_memory | 0.722 | 0.333 | 0.708 | 0.667 | 1.000 | 0.000 |
| rmm_like_reflective_summary | 0.972 | 0.667 | 0.361 | 0.056 | 0.000 | 0.056 |
| o_mem_like_user_profile | 0.889 | 0.694 | 0.194 | 0.028 | 0.000 | 0.444 |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| a_mem_like_note_linking | 0.806 | 0.584 | 0.221 | 0.478 |
| full_history_segment_retrieval | 0.806 | 0.601 | 0.204 | 0.489 |
| mem0_like_fact_memory | 0.972 | 0.180 | 0.792 | 0.289 |
| no_memory_lexical | 0.306 | 0.449 | -0.144 | 0.533 |
| o_mem_like_user_profile | 0.889 | 0.135 | 0.754 | 0.244 |
| rmm_like_reflective_summary | 0.972 | 0.174 | 0.798 | 0.300 |
| secom_like_segment_memory | 0.722 | 0.646 | 0.076 | 0.567 |
| zep_like_temporal_graph | 0.944 | 0.225 | 0.720 | 0.344 |

## Variant Sensitivity

| baseline | original reviewed acc | unseen paraphrase acc |
| --- | ---: | ---: |
| no_memory_lexical | 0.398 | 0.400 |
| full_history_segment_retrieval | 0.665 | 0.544 |
| mem0_like_fact_memory | 0.484 | 0.296 |
| zep_like_temporal_graph | 0.497 | 0.320 |
| a_mem_like_note_linking | 0.658 | 0.536 |
| secom_like_segment_memory | 0.596 | 0.552 |
| rmm_like_reflective_summary | 0.472 | 0.304 |
| o_mem_like_user_profile | 0.429 | 0.296 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| a_mem_like_note_linking | 85.6 | 56.8 |
| full_history_segment_retrieval | 817.4 | 56.8 |
| mem0_like_fact_memory | 102.5 | 11.9 |
| no_memory_lexical | 0.0 | 0.0 |
| o_mem_like_user_profile | 105.3 | 1.0 |
| rmm_like_reflective_summary | 105.3 | 1.0 |
| secom_like_segment_memory | 75.8 | 133.7 |
| zep_like_temporal_graph | 104.3 | 11.9 |

## Interpretation

The strongest current evidence is for an explicit-vs-habit gap: fact/profile-style memories do well on explicit retrieval but degrade on boundary, exception, drift, and privacy/false-personalization stress cases.
Full-history and segment-retrieval baselines can handle some boundary/exception cases, which means later versions should add paraphrased, unseen, and cost-controlled stress tests before making claims about official systems.
