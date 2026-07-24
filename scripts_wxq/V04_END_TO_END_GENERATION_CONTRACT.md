# Taskmaster planning defaults v0.4: end-to-end generation contract

v0.4 is a new data-generation track. It does not inherit v0.3's dialogue
templates or rewrite pipeline.

## Required lineage

1. Independent Taskmaster-2 conversations are converted to source cards.
2. An LLM induces reusable, scoped habit candidates from at least three
   distinct Taskmaster `instruction_id` values. The source speakers are never
   represented as one real longitudinal user.
3. An LLM composes a synthetic multi-habit user dossier from grounded habit
   candidates.
4. An LLM designs that user's chronological travel-planning arc from the
   complete dossier, including grounded habits. It does not receive or produce
   support/boundary/exception labels, target signal counts, or coverage tables.
5. Given the dossier, the arc event, relevant Taskmaster excerpts, and a short
   continuity summary, an LLM directly writes every message in every session.
6. Only after each final dialogue exists, an independent xhigh call labels
   actual user evidence. Weak post-hoc coverage rejects the generated history;
   it never causes deterministic label insertion before dialogue generation.
7. After the full history exists, an independent LLM receives only the
   generated history and re-extracts the user's habits. Recovery, boundary,
   exception, revision, and unsupported-habit checks are release gates.
8. Only after history-level recovery passes, an LLM writes probes and labels them against
   explicit evidence sessions.

All formal v0.4 LLM stages use `gpt-5.5` with explicit
`reasoning_effort=xhigh`; relying on a provider default is not sufficient.

## Prohibited construction methods

- fixed user or assistant utterance templates;
- slot-filled dialogue skeletons;
- deterministic support/boundary/exception message constructors;
- padding or duplicating sessions to reach a target length;
- generating a template transcript and asking another model to paraphrase it;
- treating multiple conversations with the same Taskmaster instruction as
  independent evidence;
- treating a destination, date, route, party size, or one-off quoted price as a
  reusable habit.

The program may create identifiers, assign chronological indices, measure
post-hoc evidence coverage, and reject malformed, weak, or repetitive model
output. It may not author dialogue content or insert label-targeted events.

## Long-range and temporal minimums

- a persona-derived 100--150 sessions per synthetic user; identical history
  lengths across all users are a release error;
- micro/short/medium/long sessions are mixed naturally (4--28 messages and
  roughly 300--6,500 characters per session by class), without a fixed matrix
  per user or phase;
- each dossier chooses a 220,000--400,000 character history target, and the
  completed history must contain at least 51,200 tokens under the exact local
  Qwen3-8B tokenizer (1.25x its 40,960-token context window);
- 5--8 scoped habit instances per user across only 4--7 relevant families,
  with a non-fixed count selected for a coherent persona. A family may occur
  twice only for genuinely disjoint recurring contexts; defaults and their
  fallback tolerance remain one habit graph;
- evidence for every tested habit appears in early, middle, and late history;
- histories are organized into recurring trip episodes with irregular time
  gaps, not a flat sequence at fixed intervals;
- signal labels are per habit within a session. A session may support one habit
  while being an exception to another;
- durable revisions create a new temporal habit version with an effective
  session boundary; one-trip exceptions never update state;
- every probe cites prior evidence sessions and is placed after all cited
  evidence;
- final session signals and probe gold answers require separate xhigh calls
  that do not see the proposed labels;
- exact and semantic near-duplicate audits are release gates.
