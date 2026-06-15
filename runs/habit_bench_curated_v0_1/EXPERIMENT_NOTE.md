# HABIT-Bench Curated v0.1 Experiment Note

Date: 2026-06-12

## Scope

This is a small-scale reviewed dataset and lightweight baseline experiment. It
is meant to validate the HABIT-Bench pipeline before scaling to official method
integrations.

Important boundary: the current baselines are method-inspired proxies, not the
official Mem0, Zep/Graphiti, A-MEM, SeCom, RMM, or O-Mem repositories.

## Human Review Outcome

I reviewed the pilot sample as a senior reviewer using the rubric in
`HUMAN_REVIEW_GUIDELINES.md`.

Decisions:

- `evidence` probes rejected for v0.1: the current options are too meta-level.
- `ask_act` probes rejected for v0.1: the query framing reveals that the agent
  should ask/abstain.
- `direct_use`, `boundary`, `exception`, `drift`, and `privacy` retained only
  after revision.

Systematic revisions:

- removed noisy real-log seed tails from visible histories;
- softened obviously artificial distractors;
- removed exact boundary-counterexample sessions from visible histories so
  boundary probes test new-context overgeneralization rather than exact lookup;
- added an explicit fact/preference retrieval split as a sanity-control task.

Frozen v0.1:

- 78 users;
- 4,438 visible sessions;
- 161 probes;
- 36 explicit retrieval probes;
- 36 direct-use probes;
- 36 boundary/false-personalization probes;
- 36 exception/counterevidence probes;
- 8 drift probes;
- 9 privacy/false-personalization probes.

## Baseline Result

The strongest diagnostic pattern is an explicit-vs-habit gap for fact/profile
memory systems:

| baseline | explicit acc | habit stress acc | gap |
| --- | ---: | ---: | ---: |
| mem0_like_fact_memory | 0.972 | 0.214 | 0.759 |
| zep_like_temporal_graph | 0.944 | 0.258 | 0.686 |
| rmm_like_reflective_summary | 0.972 | 0.191 | 0.781 |
| o_mem_like_user_profile | 0.889 | 0.135 | 0.754 |

This supports the core benchmark hypothesis for fact/profile style memories:
they can retrieve explicit user preferences but collapse on scope, exception,
drift, and privacy/false-personalization stress cases.

Episode/segment retrieval is a more nuanced story:

| baseline | explicit acc | habit stress acc | gap |
| --- | ---: | ---: | ---: |
| a_mem_like_note_linking | 0.806 | 0.629 | 0.176 |
| secom_like_segment_memory | 0.722 | 0.652 | 0.071 |
| full_history_segment_retrieval | 0.806 | 0.640 | 0.165 |

These baselines handle some boundary/exception cases because retrieving raw
episodes can preserve context that profile memories compress away. They still
show weaknesses on direct habit use, drift, privacy, or cost, but v0.1 is not
yet strong enough to claim that official A-MEM/SeCom-style systems fail across
all habit stress splits.

## Current Claim Supported

Supported by this run:

- HABIT-Bench can separate explicit preference retrieval from habit stress
  behavior.
- Fact/profile-style memory proxies show a large explicit-vs-habit gap.
- Privacy/false-personalization remains hard across nearly all lightweight
  baselines.
- Exact boundary lookup is now controlled by hiding boundary-counterexample
  episodes from visible histories.

Not yet supported:

- A claim about official Mem0/Zep/A-MEM/SeCom/RMM/O-Mem implementations.
- A claim that all episode/segment-retrieval systems fail boundary/exception
  cases.
- A paper-scale statistical claim; v0.1 is still small and partly synthetic.

## Next Decisive Step

Before scaling:

1. Add paraphrased unseen boundary/exception/drift probes so raw segment
   retrieval cannot win by surface similarity.
2. Add token/latency/storage budgets to penalize full-history and segment-heavy
   methods.
3. Replace lightweight proxies with official method adapters where feasible.
4. Expand privacy and drift splits; current counts are too small.
5. Run paired bootstrap confidence intervals over users.
