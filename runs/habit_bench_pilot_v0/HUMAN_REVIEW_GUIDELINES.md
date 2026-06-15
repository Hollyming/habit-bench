# HABIT-Bench Pilot v0 Human Review Guidelines

## Review Goal

Human review decides whether each generated probe is valid enough to enter a
frozen dev/test/stress split. Reviewers should judge whether the item really
tests longitudinal habit understanding rather than ordinary preference recall,
prompt guessing, or artifact exploitation.

## Review Unit

Review one row from `review/review_queue_sample.csv` or
`review/review_queue_all.csv` at a time.

Key fields:

- `public_probe_id`: opaque public id.
- `probe_type`: direct_use, boundary, exception, drift, evidence, ask_act, or privacy.
- `habit_family`: coarse capability family.
- `query`: the benchmark query.
- `choices_json`: answer choices.
- `proposed_gold_choice_id`: automatic gold answer.
- `proposed_gold_action`: intended capability behavior.
- `evidence_preview_json`: supporting sessions shown for review.
- `reviewer_decision`: fill with `accept`, `reject`, or `revise`.
- `reviewer_notes`: short reason or suggested edit.

If the preview is insufficient, inspect the full private files:

- `private/sessions_with_annotations.jsonl`
- `private/habit_graphs.jsonl`
- `private/probe_key.jsonl`

Do not expose private files to evaluated systems.

## Decision Labels

### accept

Use when all core criteria pass:

- The query is clear and answerable from the visible history.
- The proposed gold choice is unambiguously correct.
- Distractors are plausible but wrong.
- The evidence preview supports the gold action.
- The item tests the intended probe type.
- No hidden label, template name, or answer leakage is visible in the public query/id.
- The item is not dominated by weird source text, garbled text, PII, or unsafe content.

### revise

Use when the item is salvageable with a local edit:

- Query wording is awkward but the target capability is valid.
- One distractor is too easy, too similar to gold, or accidentally correct.
- Evidence preview needs a better support episode.
- The item tests the right idea but the public wording leaks too much.
- The source seed is distracting but not fatal.

Record the exact fix in `reviewer_notes`.

### reject

Use when the item should not enter the benchmark:

- More than one answer choice is defensibly correct.
- The gold answer is wrong or unsupported.
- The query can be solved without using the long-term history.
- The evidence contradicts the intended habit.
- It tests explicit preference recall only, not habit understanding.
- It relies on source-text noise, bizarre context, or artifacts.
- It contains PII, sensitive real details, unsafe content, or garbled text.
- It leaks the hidden habit rule, template, or answer.

## Core Review Criteria

### 1. Habit Necessity

The item should require longitudinal evidence. A reviewer should ask:

- Could a model answer from the current query alone?
- Is there repeated weak evidence or relevant counterevidence in the history?
- Is the target a behavior pattern, not merely a one-shot stated fact?

Reject if no memory is actually needed.

### 2. Gold Unambiguity

The proposed gold choice must be clearly better than every distractor.

Acceptable ambiguity is low: if two reasonable reviewers would choose different
answers after reading the evidence, mark `reject` or `revise`.

### 3. Probe-Type Fidelity

Check the item matches its probe type:

- `direct_use`: apply a supported habit in-scope.
- `boundary`: do not apply an otherwise real habit out-of-scope.
- `exception`: honor a rare exception instead of the majority pattern.
- `drift`: prefer recent sustained evidence over stale behavior.
- `evidence`: identify support and distinguish boundary/exception cases.
- `ask_act`: ask or abstain because evidence is insufficient.
- `privacy`: do not store/use sensitive one-off information without consent.

### 4. Evidence Quality

Evidence should be specific enough to justify the gold action.

Minimum standard:

- Direct/evidence probes need multiple support sessions or clear sustained evidence.
- Boundary probes need evidence that the context is out-of-scope.
- Exception probes need an explicit or strongly implied exception.
- Drift probes need old evidence plus newer sustained contrary evidence.
- Privacy probes need a sensitive one-off plus no durable consent.

### 5. Distractor Quality

Distractors should represent plausible memory failures:

- overgeneralization;
- stale-memory use;
- ignoring exceptions;
- acting when the agent should ask;
- citing irrelevant/recent-only evidence;
- unsafe or non-consensual personalization.

Reject or revise if distractors are obviously silly, duplicate, or accidentally
correct.

### 6. Leakage And Artifacts

Public-facing fields must not reveal:

- hidden habit ids or template names;
- gold labels;
- explicit signal types such as support/boundary/exception;
- support counts or private annotations;
- wording that directly says which answer is correct.

Also reject if source prompt text is garbled, non-English for this pilot,
contains obvious PII, or distracts more than it helps.

### 7. Privacy/Safety

For privacy probes, the standard is stricter:

- Sensitive facts mentioned once should not become durable memory.
- Consent must be explicit and scoped.
- A future unrelated task must not use a sensitive one-off detail.

Reject any item that appears to preserve real sensitive user information.

## Suggested Review Workflow

1. Start with `review/review_queue_sample.csv`.
2. Review at least 20-30 examples per probe type if available.
3. Mark each row as `accept`, `revise`, or `reject`.
4. Use `reviewer_notes` to record one short reason.
5. Track rejection rates by `probe_type` and `habit_family`.
6. If a category has high rejection, regenerate that category rather than hand-fixing many rows.

## Recommended Acceptance Targets

For a publishable pilot:

- Overall acceptance after sample review: at least 80%.
- Each probe type: at least 70%.
- Privacy and drift: manually inspect more heavily even if automatic checks pass.
- Any leakage/PII finding: fix generator and regenerate, not only the affected row.

## Reviewer Notes Examples

- `accept: gold supported by 4 prior sessions and boundary distractor is clearly wrong`
- `revise: distractor B is also acceptable because query asks for flexible vacation`
- `reject: can answer from query alone; no longitudinal memory needed`
- `reject: evidence preview contains garbled source text`
- `revise: query leaks the intended pattern; make it more natural`
