# HABIT-Bench 9-Family Unified Table

Status: reviewer-facing table for the balanced v0.3 candidate and the
90-probe official subset.

## Source And Domain Contract

HABIT-Bench currently has one real external prompt source and nine controlled
habit families grounded in nine unique representative task domains.

- Real prompt seed source: `allenai/WildChat`.
- Domain assignment: keyword-filtered task seed buckets from WildChat, with one
  representative domain per habit family.
- Controlled synthetic components: hidden habit graphs, assistant feedback,
  counterfactual probe contexts, answer choices, gold labels, and evidence
  links.
- Accurate release claim: real-prompt-seeded, domain-grounded, synthetic
  longitudinal habit benchmark.
- Claim to avoid: each family is drawn from a different external dataset.

The v0.3 domain provenance reports verify 100% alignment between each habit
family and its representative domain for gold evidence sessions:

- `runs/habit_bench_balanced_v0_3/reports/domain_provenance_summary.md`
- `runs/habit_bench_balanced_v0_3_official_subset_90/reports/domain_provenance_summary.md`

## Unified Table

| family | representative domain | habit capability | core probes | specialized probes | v0.3 selected habits | v0.3 probes | official subset probes | evidence source |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `format_style` | work | infer recurring response format under a scoped work context | direct, boundary, exception, explicit retrieval | none | 30 | 210 | 10 | WildChat work seeds + controlled habit feedback |
| `coding_review` | code | infer review ordering and patch minimality for code tasks | direct, boundary, exception, explicit retrieval | none | 30 | 210 | 10 | WildChat code seeds + controlled habit feedback |
| `planning_defaults` | travel | infer planning defaults for business travel | direct, boundary, exception, explicit retrieval | none | 30 | 210 | 10 | WildChat travel seeds + controlled habit feedback |
| `content_constraints` | food | infer routine content constraints for weekday family meals | direct, boundary, exception, explicit retrieval | none | 30 | 210 | 10 | WildChat food seeds + controlled habit feedback |
| `tool_action` | news/current info | infer when freshness checks are part of the user's workflow | direct, boundary, exception, explicit retrieval | none | 30 | 210 | 10 | WildChat news seeds + controlled habit feedback |
| `risk_threshold` | commitment/action approval | infer confirmation thresholds before costly or committing actions | direct, boundary, exception, explicit retrieval | none | 30 | 210 | 10 | WildChat commitment seeds + controlled habit feedback |
| `meeting_prep` | meeting | infer recurring meeting-prep document structure | direct, boundary, exception, explicit retrieval | none | 30 | 210 | 10 | WildChat meeting seeds + controlled habit feedback |
| `drift_seasonality` | equipment/work gear | update a habit after sustained recent counterevidence | direct, boundary, exception, explicit retrieval | drift | 30 | 270 | 10 | WildChat equipment seeds + controlled pre/post-drift feedback |
| `privacy_consent` | privacy/sensitive info | avoid durable use of sensitive one-off facts without consent | direct, boundary, exception, explicit retrieval | privacy | 30 | 270 | 10 | WildChat privacy seeds + controlled consent/no-consent feedback |

## Current Counts

- Balanced v0.3: 174 users, 9,924 sessions, 270 selected habits, 2,010 probes.
- Full balanced v0.3 family policy: 30 habits per family; 210 probes for
  ordinary families and 270 probes for `drift_seasonality` /
  `privacy_consent` because each has one specialized probe type.
- Official subset: 67 users, 3,815 visible-history sessions, 90 probes.
- Official subset family policy: exactly 10 probes per family, with capability
  floors for drift and privacy retained.

## Probe-Type Interpretation

- `direct_use`: apply a supported scoped habit.
- `boundary`: avoid applying the habit outside its valid context.
- `exception`: respect rare or explicit exceptions.
- `explicit_retrieval`: answer a directly retrievable fact/preference control.
- `drift`: prefer the latest sustained habit after temporal counterevidence.
- `privacy`: avoid false personalization from sensitive one-off facts unless
  consent is explicit.

## Why This Matters

The benchmark is not intended to test whether memory systems can retrieve a
single fact. It tests whether they can induce scoped user policies across long
interaction histories, then avoid over-personalization when the surface domain
is similar but the behavioral condition changes.

The domain grounding is therefore a control variable: each family uses a
realistic task setting where the target memory behavior naturally appears, while
the hidden habit and counterfactual probes remain synthetic and auditable.
