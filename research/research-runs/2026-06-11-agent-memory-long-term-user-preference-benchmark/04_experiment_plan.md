# Experiment Plan: agent memory long-term user preference benchmark

## Hypothesis

**H1: Current memory agents are optimized for explicit fact/preference retrieval and will underperform on longitudinal habit understanding**, where the correct personalized action depends on aggregating repeated weak evidence, inferring scope/boundaries, retaining exceptions, detecting drift, and avoiding false personalization.

**H2: A semi-automatic benchmark built from real user-assistant logs plus controlled habit graphs can expose this gap with low manual cost**, if every test item has hidden evidence provenance, counterfactual no-personalization traps, and automatic validators.

**H3: Habit-aware memory representations with uncertainty, negative evidence, and scoped rules will improve downstream utility under matched context/retrieval budgets.**

## Minimal Refutation Experiment

- Dataset/task: HABIT-Bench-Pilot with 200 pseudo-users, 50-200 sessions per user, 8 habit families, 5 probe types. Construct from WildChat/LMSYS-style real prompts, then inject controlled assistant feedback/events and hidden habit graphs.
- Baselines: no-memory latest-turn, sliding window, full-context where possible, recursive summary, BM25, embedding RAG, RAG + temporal metadata, Mem0-like fact memory, Zep/Graphiti-like temporal KG, A-MEM-like note linking, profile-summary memory.
- Metric: habit action accuracy, boundary F1, false-personalization rate, drift adaptation lag, exception recall, evidence support precision, abstain/ask calibration, token cost, latency, memory growth.
- Expected result if hypothesis is true: baselines that retrieve explicit memories do well on direct preference probes but fail on boundary/no-personalization/drift probes; long-context helps only at small horizons and high cost.
- Result that would weaken the hypothesis: strong RAG/profile/KG memory systems achieve high habit utility and low false-personalization under equal budgets; no separate habit-aware evaluation is needed.

## Strong Top-Conference Experiment

- Benchmarks: HABIT-Bench full test; plus LoCoMo, LongMemEval, PersonaMem-v2, PERMA, and AlpsBench as external controls to show complementarity rather than replacement.
- Models: at least two frontier API models and two open-weight instruction models; keep the LLM fixed across memory systems for primary comparisons.
- Memory budgets: fixed memory item count and fixed retrieved-token budget; separate small/medium/large memory-store regimes.
- Context budgets: latest turn only, 8K, 32K, 128K, and full-context where available. Report accuracy-cost Pareto curves.
- Baselines: no-memory, full-context, window, summary, BM25, dense vector RAG, hybrid RAG, temporal RAG, recursive/tree summary, MemGPT/Letta, Mem0, Zep/Graphiti, A-MEM, SeCom/RMM-style reflective retrieval, O-Mem/profile memory when available.
- Ablations: remove timestamps, remove negative evidence, remove exception memory, remove boundary labels, remove confidence threshold, remove deletion/correction support, vary habit evidence count, vary distractor ratio.
- Stress tests: ultra-long histories, sparse evidence, conflicting habits, abrupt/gradual/seasonal drift, one-off sensitive events, adversarial memory injection, collaborator conflicts, domain transfer, irrelevant personalization traps.
- Cost measurements: prompt tokens, write-time LLM calls, retrieval calls, latency p50/p95, memory-store growth, storage bytes, human audit minutes per accepted item.
- Statistical checks: paired bootstrap over user lifelines; random seeds for data generation and retrieval; report confidence intervals by habit family and horizon bucket.

## Benchmark Definition: HABIT-Bench

### New Capability Problem

**Longitudinal habit understanding**: Given a long sequence of user-agent interactions, the agent must infer and use latent user habits as scoped, probabilistic behavioral rules. A habit is not a stated preference; it is a repeated pattern whose applicability depends on context, history, evidence strength, exceptions, and time.

The benchmark asks whether an agent can:

- infer a habit from repeated weak signals without explicit user statements;
- decide when the habit should not apply;
- update confidence as new evidence arrives;
- handle exception clauses and drift;
- cite supporting episodes or ask when uncertain;
- avoid forming sensitive or spurious user traits.

### Habit Families

- Format/style: terse vs detailed, bullets vs prose, code-first vs explanation-first.
- Planning defaults: early flights, low walking distance, calendar buffer, no weekend work.
- Tool/action choices: always check source freshness, prefer local files, ask before booking/buying.
- Content constraints: vegetarian weekday meals but flexible on weekends; kid-safe restaurant choices; citation requirements.
- Risk/stakes thresholds: draft only vs send email, recommend vs execute purchase.
- Privacy/consent: never retain medical/finance one-offs unless explicit future use is requested.
- Domain-specific habits: coding review style, writing tone, meeting-prep structure.
- Seasonal/drift habits: diet, budget, location, role, or project phase changes.

### Probe Types

- Direct use: choose the personalized action/response for a new task.
- Boundary probe: same surface task, different context where the habit should not apply.
- Counterfactual probe: asks whether behavior would change if a condition were absent.
- Drift probe: recent sustained evidence conflicts with older habit.
- Evidence/provenance probe: retrieve the minimal support set and confidence.
- Ask-vs-act probe: correct answer is to ask a clarifying question because evidence is insufficient.
- Privacy/deletion probe: correct answer is not to store/use a sensitive inferred habit.

## Implementation Plan

1. Data acquisition: download or reference WildChat and LMSYS-Chat-1M; optionally use AlpsBench/PersonaMem-v2/PERMA if licenses permit for comparison, not direct copying into the final test.
2. Real prompt mining: cluster conversations by task domain, style choice, explicit feedback, and repeated user constraints. Use privacy filters to remove PII/sensitive spans before generation.
3. Habit graph generation: create hidden rules of the form `(context conditions, action/style default, strength, support episodes, exceptions, start/end time, sensitivity label)`.
4. Lifeline construction: stitch real prompts and generated assistant replies into pseudo-user timelines with 50-1000 sessions. Preserve realistic language/noise while controlling ground truth.
5. Counterfactual balancing: for each positive habit application, generate matched negative contexts where the habit must not apply.
6. Automatic validation: use rule executors and LLM-as-judge only for candidate filtering; require deterministic checks where possible, e.g., option label, slot value, action trace, prohibited memory use.
7. Human audit: review 5-10% of items plus all high-disagreement cases; report acceptance rate and audit minutes.
8. Baseline harness: common API for memory systems: `observe(session)`, `answer_or_act(query)`, `return_evidence()`, `delete/update(request)`.
9. Evaluation: run baselines with fixed model/context/retrieval budgets; produce tables by horizon, habit family, probe type, and cost.
10. Release artifacts: dataset cards, privacy statement, generation prompts, validators, hidden/private test split, seed logs, and baseline configs.

## Risks And Confounders

- Overlap with PersonaMem-v2/PERMA/AlpsBench: avoid claiming novelty on implicit preference alone; novelty must be habit boundary + false-personalization + evidence-calibrated action.
- Synthetic artifacts: keep real prompts and styles; use automatic artifact detectors and hold out generation templates.
- Ambiguity: every item needs a hidden habit graph and validator; ambiguous items go to human audit or are discarded.
- Privacy: never release raw user identifiers; strip PII; avoid real sensitive inference unless synthetic and clearly labeled.
- Context-length confound: compare systems under matched context budgets and also report full-context upper bounds separately.
- Prompt tuning confound: freeze prompts after dev split; provide all prompts/configs.
- Baseline weakness: include simple strong retrieval and summary baselines, not only no-memory.
- Leakage: release hidden test after benchmark server or keep private labels; ensure generated probes do not expose rule wording.
