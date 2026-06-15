# Research Memo: HABIT-Bench for Longitudinal Agent Memory

Date: 2026-06-11

## One-Sentence Claim

Current agent-memory systems and benchmarks mostly evaluate explicit facts, preferences, event recall, and retrieval; a top-conference-level gap remains in **longitudinal habit understanding**: learning, updating, scoping, and safely using implicit user habits over long or ultra-long user-agent interaction histories.

## Why It Matters

Real personal agents should not merely remember that "the user likes concise answers." They need to know whether that was a one-off request, a work-only habit, a time-varying habit, an exception-bearing rule, or a sensitive pattern that should not be stored. The hardest user-memory failures are often not forgetting; they are **over-personalizing from weak evidence**, **applying the right memory in the wrong context**, and **continuing stale habits after the user changes**.

This is a different capability from long-context QA. It is closer to longitudinal behavioral modeling with memory governance:

- infer habits from repeated weak evidence rather than explicit profile facts;
- distinguish stable habits from task artifacts and temporary constraints;
- infer context boundaries and exceptions;
- adapt to drift while not catastrophically overwriting history;
- decide when to ask instead of acting;
- avoid storing or using sensitive inferred traits.

## Closest Prior Work And Gap

### Long-Term Conversation / Agent Memory Benchmarks

- **LoCoMo** evaluates long-term conversational memory across long multi-session dialogues, including temporal/causal reasoning and summarization. It is foundational for long dialogue memory, but the target remains mostly recall/reasoning over events rather than habit induction and false personalization. Source: https://arxiv.org/abs/2402.17753
- **LongMemEval** evaluates chat assistants over sustained histories with information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. It is the closest long-term assistant-memory QA benchmark, but many targets are explicit facts/preferences rather than distributional habits. Source: https://arxiv.org/html/2410.10813v2
- **LongMemEval-V2** pushes to ultra-long agent experience: up to 500 web-agent trajectories and 115M tokens, with latency-aware agent-memory evaluation. It is excellent for "experienced colleague" environment memory, but not a user-habit benchmark. Source: https://xiaowu0162.github.io/longmemeval-v2/
- **MemoryAgentBench** and **MemoryArena** move beyond static recall into incremental interactions, forgetting, test-time learning, and multi-session agentic task success. They support the need for action-based evaluation, but are broader agent-memory benchmarks, not specifically habit understanding. Sources: https://arxiv.org/html/2507.05257v2 and https://arxiv.org/html/2602.16313v1
- **BEAM** shows that even very long context and RAG struggle as conversations reach 100K-10M tokens. It motivates ultra-long evaluation, but its main target is memory probing over synthetic long conversations. Source: https://openreview.net/forum?id=y59hf5lrMn

### Personalized Memory Benchmarks

- **PersonaMem** and **PersonaMem-v2** are very close: they evaluate evolving user profiles, implicit preferences, and compact agentic memory over long histories. The remaining gap is that habits are still not treated as scoped, probabilistic rules with counterevidence, boundaries, exceptions, and false-personalization cost. Sources: https://arxiv.org/html/2504.14225v1 and https://arxiv.org/html/2512.06688v1
- **PERMA** is another near neighbor: event-driven preference evolution, realistic task environments, and interactive evaluation. This means a new paper cannot claim novelty from "preferences emerge from events" alone. The new axis must be habit boundary/calibration and not merely preference retrieval. Source: https://arxiv.org/html/2603.23231v1
- **AlpsBench** uses real WildChat dialogues to build a benchmark for explicit/implicit memorization and preference alignment across extraction, update, retrieval, and utilization. It is the strongest precedent for reusing real user-agent logs, so our data pipeline should use real logs but define a different target: longitudinal habits, boundary tests, and no-personalization traps. Source: https://arxiv.org/html/2603.26680v2
- **PAL-Bench / Mem-PAL** targets personalized long-term service dialogue, especially Chinese multi-session scenarios. It is useful as a downstream comparison but does not fully cover ultra-long real-log habit inference. Source: https://ojs.aaai.org/index.php/AAAI/article/view/40385

### Memory Methods To Benchmark

Representative systems include **MemGPT/Letta** hierarchical memory, **Mem0** extraction/consolidation/retrieval and graph memory, **Zep/Graphiti** temporal knowledge graph memory, **A-MEM** agentic note-linking, **SeCom** segment/compression memory, **RMM** reflective memory management, and **O-Mem** hierarchical personalized profiling. These are strong baselines, but most represent memories as facts, notes, profiles, graph edges, or summaries. HABIT-Bench tests whether those representations support frequency, confidence, context scope, counterevidence, and exceptions.

Sources: https://arxiv.org/abs/2310.08560, https://arxiv.org/abs/2504.19413, https://arxiv.org/abs/2501.13956, https://arxiv.org/abs/2502.12110, https://arxiv.org/abs/2502.05589, https://aclanthology.org/2025.acl-long.413/, https://arxiv.org/abs/2511.13593

## Proposed Benchmark: HABIT-Bench

### Core Definition

**Habit = a latent, repeated, context-conditioned behavioral policy inferred from user-agent interactions.**

A habit memory should include:

- condition: when it applies;
- action/default: what the agent should do;
- strength: always/usually/sometimes/unknown;
- support: episodes that justify it;
- counterevidence: episodes against it;
- exceptions: explicit or inferred;
- temporal validity: start/end/drift/seasonality;
- sensitivity: whether it should be retained or used.

### Capability Splits

1. **Habit Induction**
   Infer habits from repeated weak signals, not explicit declarations.

2. **Boundary Calibration**
   Apply the habit only in the right context. Penalize false personalization heavily.

3. **Exception Retention**
   Preserve rare but decisive exceptions that summaries often erase.

4. **Drift And Seasonality**
   Detect gradual, abrupt, or cyclic changes in the user's behavior.

5. **Ask-Versus-Act Calibration**
   Choose to ask when habit evidence is insufficient or conflicted.

6. **Evidence-Provenance Memory**
   Return minimal supporting episodes and uncertainty for a personalized action.

7. **Consentful Memory**
   Do not elevate sensitive one-off events into durable inferred user traits.

## Benchmark Production Pipeline

The benchmark should be semi-automatic, with small human audit rather than full manual construction.

### Data Sources

- **WildChat**: opt-in real ChatGPT interaction logs; useful for realistic prompts, domains, timestamps, languages, and user-agent style. Source: https://arxiv.org/abs/2405.01470
- **LMSYS-Chat-1M**: large real-world multi-model chat logs; useful for task diversity and realistic user requests. Source: https://arxiv.org/abs/2309.11998
- **AlpsBench-style structured memories**: useful precedent for extraction/update/retrieval/utilization labels, but avoid copying the same task definition.
- **PersonaMem-v2 / PERMA**: use for external comparison and generator inspiration, not as the sole source of novelty.
- Optional: **REALTALK** for realistic long-term conversational texture and temporal dynamics, though it is human-human rather than user-agent task data. Source: https://arxiv.org/abs/2502.13270

### Construction Steps

1. **Mine real prompts**
   Cluster real conversations by task domain, output style, constraints, feedback, and repeated action choices. Remove PII and sensitive spans before any generation.

2. **Generate hidden habit graphs**
   Create rules like:

   `weekday_meal_planning + user_cooks_for_family -> prefer vegetarian, except birthdays/travel`

   Each rule stores support episodes, counterevidence, exception cases, timestamps, and sensitivity labels.

3. **Stitch long pseudo-user lifelines**
   Combine real prompts with generated assistant replies and controlled feedback into timelines of 50-1000 sessions per pseudo-user. Use real language from logs but controlled ground truth.

4. **Inject distractors and counterfactuals**
   For every habit-positive case, create matched negative cases where the same surface request should not trigger the habit.

5. **Generate probes**
   Use deterministic templates plus LLM generation for naturalness. Validate with rule executors and discard ambiguous items.

6. **Audit cheaply**
   Human-review 5-10% of accepted items and all high-disagreement cases. Report acceptance rate, audit time, and discarded categories.

## Evaluation Metrics

- **Habit Action Accuracy**: correct personalized response/action.
- **Boundary F1**: correctly apply vs not apply the habit.
- **False Personalization Rate**: personalized action in contexts where no habit should be used.
- **Drift Adaptation Lag**: number of sessions after a change point until behavior updates.
- **Exception Recall**: success on rare exception cases.
- **Evidence Precision/Recall**: quality of cited support episodes.
- **Ask/Act Calibration**: abstain or ask when evidence is insufficient.
- **Sensitive Memory Violation Rate**: storing/using disallowed inferred sensitive traits.
- **Cost**: prompt tokens, retrieval calls, write-time calls, p50/p95 latency, memory growth.

## Baselines And Expected Behavior

This section is **hypothesis, not experimental result**.

| Method class | Expected strength | Expected failure on HABIT-Bench |
| --- | --- | --- |
| No-memory / latest-turn | strong on local instructions | no longitudinal habit induction |
| Full-context long-context LLM | strong at small/medium horizons | cost explosion; diluted evidence; weak ultra-long drift/boundary control |
| Sliding window / summary | cheap and robust for recent changes | erases rare exceptions and distributed evidence |
| BM25 / dense RAG | finds explicit matching episodes | misses habits spread across many weak signals; poor boundary calibration |
| Temporal RAG | helps recency and updates | still treats habits as retrieved snippets, not confidence-scoped rules |
| MemGPT/Letta | good memory hierarchy baseline | write policy may miss implicit habits or over-store one-offs |
| Mem0 / Zep / graph memory | strong fact/preference extraction and temporal facts | graph/fact edges underrepresent frequency, negative evidence, and exceptions |
| A-MEM / note-linking | better organization and associative retrieval | links similarity more than causal habit scope |
| SeCom / RMM | strong personalized retrieval and reflection | may rationalize retrieved facts without calibrated habit uncertainty |
| O-Mem/profile memory | strong compact user profile | profile abstraction may overgeneralize and apply habits too broadly |

## Strong Experimental Claim

A compelling ICML/NeurIPS/ICLR paper would show:

1. Systems that look strong on LongMemEval/PersonaMem/PERMA/AlpsBench still fail on HABIT-Bench boundary/drift/no-personalization splits.
2. The failure persists under matched model, context, and retrieval budgets.
3. Error analysis shows systematic overgeneralization, exception erasure, and drift inertia.
4. A simple habit-aware representation with confidence, scope, counterevidence, and expiry improves utility without raising false personalization.
5. The benchmark can be built semi-automatically with transparent data provenance, privacy filtering, and small audit cost.

## Why This Is Not Incremental

The closest prior work already covers long dialogue memory, explicit/implicit preferences, event-driven preference evolution, and real-log memory lifecycle evaluation. HABIT-Bench changes the unit of evaluation from **remembered item** to **behavioral rule under uncertainty**. It asks whether a memory agent can decide not only "what did the user say before?" but "what stable pattern, if any, should govern this new decision, and when should I refuse to use it?"

That is a new capability definition for long-term personal agents and a harder negative-control setting than ordinary preference recall.

## Evidence Status

- Established by sources: recent memory benchmarks cover long histories, agent experience, explicit/implicit preference, event-driven preference, and real-dialogue memory lifecycle.
- Established by sources: real user-agent logs such as WildChat and LMSYS-Chat-1M can seed realistic benchmark construction.
- Plausible but untested: current memory systems will have high false-personalization and low boundary calibration on habit tasks.
- Not yet shown in this workspace: any baseline numbers on HABIT-Bench.
- Next decisive step: build HABIT-Bench-Pilot and run 5-8 baselines under equal token/retrieval budgets.

## Next Decisive Experiment

Build a pilot with:

- 200 pseudo-users;
- 50-200 sessions each;
- 8 habit families;
- 5 probe types;
- 20K-50K total test items;
- 5-10% human audit;
- baselines: no-memory, full-context, summary, dense RAG, temporal RAG, Mem0-like, Zep-like, A-MEM-like, profile memory.

The paper idea should be killed or revised if these baselines already achieve high habit action accuracy, high boundary F1, low false-personalization, and low drift lag under the same context/retrieval budget.
