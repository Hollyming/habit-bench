# HABIT-Bench 9-Family Taxonomy

Status: implementation taxonomy for HABIT-Bench v0.2/v0.3.

HABIT-Bench uses real WildChat prompts as task seeds, then injects synthetic
hidden habit graphs, controlled feedback, and probes. A habit family is the
behavioral memory phenomenon being tested; a domain is the task setting used to
make the phenomenon concrete. In the current implementation the nine habit
families are mapped to nine unique representative task domains; these domains
are controlled WildChat seed buckets, not separate external datasets.

## Family Table

| family | representative domain | hidden habit template | capability focus | positive evidence | negative / stress evidence | paper-level super-family |
| --- | --- | --- | --- | --- | --- | --- |
| `format_style` | work | work updates in three crisp bullets | infer recurring response format from repeated feedback | repeated standup/status notes rewarded for concise three-bullet form | exploratory/creative writing should not be forced into bullets; deep-dive requests override compactness | format/style |
| `coding_review` | code | code review starts with risks and minimal patch | learn domain-specific review ordering | PR/debug sessions reward risks-first and minimal patch | onboarding/tutorial contexts should teach patiently instead | domain-specific workflow |
| `planning_defaults` | travel | business travel prefers early flights and buffer time | infer planning defaults under context scope | work travel sessions reward early arrival and 90-minute meeting buffer | leisure or flexible travel should not inherit business timing | planning defaults |
| `content_constraints` | food | weekday family meals are vegetarian | infer content constraints tied to routine context | weekday family meal sessions reward vegetarian defaults | birthdays/travel/restaurant exploration should not over-apply the weekday rule | content constraints |
| `tool_action` | news/current information | freshness check for high-stakes or current topics | learn when tool/source freshness is part of the user's workflow | high-stakes/current tasks reward recency checks before advice | evergreen educational questions should not require live lookup | tool/action choices |
| `risk_threshold` | commitment/action approval | confirm before bookings, sends, or irreversible actions | learn execution-risk threshold and confirmation policy | compare/draft/find tasks reward pausing before booking/submission/send | low-risk comparison/drafting should not add unnecessary confirmation friction | risk/stakes thresholds |
| `meeting_prep` | meeting | Monday meeting prep emphasizes decisions, blockers, next actions | learn recurring domain-specific document structure | Monday sync prep rewards decisions/blockers/actions | casual check-ins should stay conversational | domain-specific workflow |
| `drift_seasonality` | equipment/work gear | equipment preference drifts from cheapest to durable quality | update a habit after sustained recent counterevidence | old budget evidence and newer durable-quality evidence coexist | one-off disposable supplies should not inherit durable-equipment preference; latest sustained evidence should win | seasonal/drift habits |
| `privacy_consent` | privacy/sensitive information | sensitive one-off facts require consent before memory use | avoid durable personalization from sensitive one-off facts | one-off sensitive tasks explicitly say not to remember/use later | explicit consent is required before sensitive facts can influence future tasks | privacy/consent |

## Current Data Source Contract

- Seed source: `allenai/WildChat`.
- Real prompt usage: sanitized task/domain seed only. The current release is
  single-source in external provenance and multi-domain in representative task
  grounding.
- Domain assignment: keyword-filtered WildChat task seed buckets: work, code,
  travel, food, news/current-info, commitment/action approval, meeting,
  equipment/work gear, privacy/sensitive-info, plus general distractors.
- Synthetic controlled components: hidden habit graph, feedback, counterfactual
  contexts, answer choices, gold labels, and evidence links.
- Release framing: real-prompt-seeded synthetic longitudinal habit benchmark.
- Reviewer boundary: do not claim that each habit family is drawn from a
  different external dataset. The accurate claim is that every family is
  grounded in a unique representative task domain while sharing the same real
  prompt source. General distractors may appear in user histories but are not a
  habit-family domain.

See `docs/9_family_unified_table.md` for the paper-facing table and
`runs/*/reports/domain_provenance_summary.md` for split-level verification.

## Domain And Family Interpretation

The implementation-level 9-family taxonomy is intentionally more concrete than
the original 8-family paper outline. The original "domain-specific habits"
super-family is split into `coding_review` and `meeting_prep` because both are
important, common agent workflows with different failure modes.

For paper narration, use:

- 9 implementation families for tables, stratified sampling, and error analysis.
- 8 super-families for the conceptual taxonomy, merging `coding_review` and
  `meeting_prep` under domain-specific workflow habits.

## Probe Coverage Policy

Primary probe types for all ordinary families:

- `direct_use`: apply a supported scoped habit.
- `boundary`: avoid applying the habit outside its context.
- `exception`: respect a rare or explicit exception.
- `evidence`: cite or select the support set while distinguishing boundary and
  exception cases. Current v0.1/v0.2 evidence probes were rejected and should be
  regenerated before a final release.

Specialized probe types:

- `drift`: only required for `drift_seasonality`.
- `privacy`: only required for `privacy_consent`.
- `ask_act`: user-level negative-control probes for insufficient evidence.
  Current v0.1/v0.2 ask-act probes were rejected and should be regenerated.

## Current Counts

v0.2 is a small reviewed stress set, not a balanced final benchmark:

| family | v0.2 probes |
| --- | ---: |
| `privacy_consent` | 54 |
| `drift_seasonality` | 45 |
| `tool_action` | 32 |
| `planning_defaults` | 29 |
| `meeting_prep` | 29 |
| `content_constraints` | 25 |
| `format_style` | 25 |
| `coding_review` | 24 |
| `risk_threshold` | 23 |

The pilot pool has enough auto-validated candidates to construct a larger
balanced v0.3 candidate. That larger split must still be treated as requiring
human audit before paper-scale claims.
