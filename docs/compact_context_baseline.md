# Full-memory online compact-history baseline

## Naming and purpose

`full_memory` now denotes the implemented `memory_context.v5` online compact
control requested for the main experiment. The old `memory_context.v3` raw
recency behavior is retained under `full_history`. Historical artifacts that
used the old `full_memory` ID remain identifiable by their run-manifest
revision and must be reported as **40k Full-History (recency-truncated)**.

The three controls answer distinct questions:

1. `no_memory`: what can the answer model infer from the current probe alone?
2. `full_history`: what can it infer from raw recent history under a fixed
   context limit?
3. `full_memory`: what can a query-independent online compactor preserve
   from the entire visible history under a smaller state budget?

Neither `full_history` nor `full_memory` is an oracle. `oracle_evidence`
and `oracle_habit_state` remain separate diagnostic upper bounds.

The primary construction is the established **rolling/recursive summarization
+ recent verbatim buffer** pattern. [SUMM^N](https://aclanthology.org/2022.acl-long.112/)
formalizes multi-stage split-then-summarize for inputs beyond a model window;
this baseline makes that hierarchy chronological and causal by repeatedly
combining the prior compact state with the next old-history chunk. Keeping a
separate raw recent tier also follows the hierarchical working/long-term memory
motivation in [MemGPT](https://arxiv.org/abs/2310.08560). This is an adaptation,
not a claim that either paper used HABIT-Bench's exact schema.

This separation also matches current long-memory evaluation practice. The
[LongMemEval reference implementation](https://github.com/xiaowu0162/LongMemEval)
uses full history only where it fits and evaluates memory retrieval separately;
its [paper analysis](https://arxiv.org/html/2410.10813) warns that replacing raw
sessions with summaries or extracted facts can discard answer-critical detail.
[LoCoMo](https://github.com/snap-research/locomo) likewise distinguishes a
truncated-conversation input from retrieval over dialogs, observations, and
session summaries. Compact history is therefore the primary full-memory
baseline, while the raw recency control remains necessary as an ablation.

## Legacy-run audit

The persisted legacy Travel run confirms that overflow is not a rare fallback:
all 128 probes were truncated at the 40k tier. The mean visible history was
167,989.7 tokens across 135.0 sessions, while the delivered context averaged
37,306.5 tokens across 29.5 recent sessions. It dropped 105.6 sessions on
average, and mean decisive-evidence context recall was 0.181. These figures are
computed from the merged `memory_contexts.jsonl` and `scored_predictions.jsonl`
under `domain/travel/.../travel_v05_final_20260728/travel/full_memory/merged`.

## Implemented baseline

Maintain two per-user objects in chronological order:

- a bounded structured summary of compacted older sessions;
- a raw recent-session buffer.

When the raw buffer would exceed its budget, compact its oldest complete
sessions together with the previous summary. The compactor must run before any
probe-specific query or choices are exposed. It may see only sessions at or
before the current cutoff and must never see gold evidence, answer labels, or
the hidden habit graph.

Use a fixed schema so the summary does not collapse into a generic prose recap:

```text
stable_defaults
scoped_preferences
exceptions_and_one_offs
changes_and_reversals
supporting_observations: session_id, timestamp, speaker, confidence
unresolved_conflicts
```

Assistant proposals must not be promoted to user preferences unless later user
behavior confirms them. Every retained claim needs source session IDs. The
compactor should preserve scope, negation, exceptions, temporal order, and
uncertainty before incidental narrative details.

## Reproducible profile

- Compactor and answer model: the same fixed Qwen3-8B checkpoint.
- Generation: temperature 0, thinking disabled, versioned prompt and schema.
- Summary state budget: 4,096 tokens for the formal main profile.
- Normal compaction targets at most 2,048 tokens, 24 bullets, 45 words per
  bullet, and four representative citations per bullet. The 4,096-token state
  budget is an outer safety envelope, not a generation target.
- If a response reaches the completion envelope, deterministic retries tighten
  those limits to 1,024/12/35/3 and finally 640/6/30/2. This makes overflow
  handling bounded while keeping the final state auditable.
- If all three normal retries still end at `finish_reason=length`, recursively
  split that input on complete-session boundaries and compact the halves in
  chronological order. This is the operational bug fix for pathological 30k
  compactor inputs and preserves the original successful v5 path.
- If recursive splitting reaches one session and the model still ignores every
  soft limit, use a strict JSON schema with at most one fact in each of the six
  sections, at most 240 characters and two supplied session IDs per fact.
  Unknown IDs, invalid JSON, an empty state, or another length stop hard-fails;
  truncated generated text is never accepted.
- Raw recent buffer: complete sessions using the rest of the 38k history
  budget minus a 1,024-token wrapper reserve (32,880 tokens by default).
- Histories at or below 38k remain entirely raw; compaction starts only after
  the complete visible history overflows.
- On summary overflow: recursively compact old summary plus newly evicted
  sessions and apply the bounded recovery above; hard-fail rather than accept a
  truncated generated state.
- Advance state chronologically per user and cache each distinct cutoff so
  repeated cutoffs do not regenerate or leak across users.
- Record compactor calls, input/output tokens, prompt hash, state hash, source
  session coverage, latency, and truncation/overflow counts.

## Comparisons

Run both of these views:

- **budget matched:** `full_history` and `full_memory` receive the same
  memory-token budget before the shared answer prompt;
- **capacity matched:** the existing 40k recency control is compared with
  compacted state plus recent raw sessions under the same 40k model limit.

Report Accuracy together with decisive-evidence recall, retained source-session
coverage, summary compression ratio, compactor cost, and results by evidence
distance, conflict resolution, exception, and habit revision. A summary may
improve global coverage while deleting decisive details, so token savings alone
are not evidence of better memory.
