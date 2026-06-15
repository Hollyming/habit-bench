# Gap Map: agent memory long-term user preference benchmark

## Current Best Evidence

- LongMemEval/LongMemEval-V2 show that memory QA and experience retrieval degrade sharply with long histories, and that retrieval quality, latency, and context budget must be reported together.
- LoCoMo/BEAM show that ultra-long dialogue memory is not solved by long context alone; temporal/causal reasoning and summarization remain hard.
- MemoryAgentBench/MemoryArena push evaluation from static recall toward incremental learning, forgetting, and multi-session agentic task success.
- PersonaMem/PersonaMem-v2, PERMA, PAL-Bench, and AlpsBench show a fast-moving personalized memory line: explicit/implicit preferences, event-driven preference evolution, realistic dialogue sources, and lifecycle evaluation are already occupied.
- Production-style methods such as Mem0, Zep/Graphiti, MemGPT/Letta, SeCom, RMM, A-MEM, and O-Mem mainly compete on fact/preference extraction, memory organization, retrieval, and profile maintenance.

## Contradictions Across Papers

- Long-context models sometimes beat memory systems when histories are small enough, but lose on cost/latency or ultra-long histories. A new benchmark must control token budget and report cost.
- Memory systems improve retrieval benchmarks but can harm personalized action by retrieving plausible but stale or over-broad memories.
- Persona/profile methods assume a stable latent user, while real users show habits that are context-conditioned, frequency-based, and sometimes intentionally inconsistent.
- Real-log benchmarks improve ecological validity but inherit sparse histories, privacy filtering, ambiguity, and uneven user coverage.

## Under-Tested Assumptions

- A user preference can be represented as a stable declarative fact. Habits are often probabilistic policies: "usually do X in context C, except when condition E holds".
- More remembered facts means better personalization. In reality, false personalization and over-application may be worse than asking a clarifying question.
- Repeated behavior should always be compressed into a memory. Some repeated behavior is caused by external context, temporary constraints, or platform defaults.
- Retrieval relevance is enough. Habit use requires evidence aggregation, negative evidence, recency weighting, boundary discovery, and uncertainty calibration.
- User correction/deletion can be evaluated as a binary update. A corrected habit may require downgrading old evidence rather than deleting all related episodes.

## Missing Benchmarks Or Controls

- Habit induction from weak repeated evidence, not explicit statements.
- Boundary inference: when the same user habit should and should not apply across domains, devices, time, collaborators, or task stakes.
- Frequency and confidence calibration: whether the agent knows a habit is "always", "usually", "sometimes", or "unknown".
- Negative transfer / false personalization: penalize applying a remembered habit in a context where it should be irrelevant.
- Drift and seasonality: preferences that change gradually, abruptly, or cyclically.
- Counterfactual habit tests: held-out tasks where the correct action requires knowing why the habit exists, not just memorizing surface behavior.
- Privacy and consent: whether sensitive or one-off behaviors are inappropriately elevated into durable user traits.
- Evidence-provenance metrics: every personalized action should cite supporting episodes and explain uncertainty.
- Long-horizon action evaluation: downstream task success/reward, not just QA over memories.

## Memory-Specific Failure Modes To Probe

- One-shot overgeneralization: "User once requested a terse answer" becomes "user always wants terse answers".
- Context collapse: "for legal summaries use bullets" is wrongly applied to casual travel planning.
- Boundary blindness: agent cannot infer that a preference applies only on mobile, weekdays, work tasks, family meals, low-budget travel, etc.
- Exception erasure: summaries preserve the main habit but omit decisive exceptions.
- Drift inertia: old habits dominate despite sustained contrary evidence.
- Retrieval myopia: the query retrieves recent or semantically similar episodes but misses distributed evidence across many sessions.
- Spurious habit formation: agent infers a habit from dataset artifacts, assistant defaults, or repeated task requirements rather than user choice.
- Privacy leakage: agent stores and uses sensitive inferred traits when no future task requires them.
- Uncalibrated confidence: agent acts personally when it should ask, or asks repeatedly after enough evidence exists.

## Candidate Paper Claims

1. A benchmark centered on habit understanding will reveal failures not captured by current long-term memory and personalization benchmarks.
2. Habit memory should be evaluated as distributional, context-conditioned, exception-bearing, and temporally evolving behavior, not as static key-value user preferences.
3. A semi-automatic pipeline can construct high-quality habit benchmarks by mining real user-assistant logs, converting repeated choices into latent habit graphs, injecting controlled counterfactual tests, and auditing a small subset.
4. Current memory systems will show high recall on explicit memories but low boundary calibration and high false-personalization rates under habit tests.
