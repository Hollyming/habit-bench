# HABIT-Bench Domain Provenance Summary

- Dataset: `work\habit-bench-builder\runs\habit_bench_balanced_v0_3_official_subset_90`
- Status: `pass`
- Users: 67
- Sessions: 3815
- Probes: 90

## Source Contract

HABIT-Bench currently uses a single real external prompt source and nine unique representative task domains. The real prompts provide task surface form; hidden habits, feedback, probes, answer choices, labels, and evidence links are controlled synthetic components.

- Real prompt seed source: allenai/WildChat: 3815
- Family-domain contract: `nine_unique_representative_domains`
- Seed domain buckets: code: 505, commitment: 485, equipment: 646, food: 483, general: 239, meeting: 481, news: 135, privacy: 223, travel: 124, work: 494
- Session domains: code: 505, commitment: 485, equipment: 646, food: 483, general: 239, meeting: 481, news: 135, privacy: 223, travel: 124, work: 494

## 9-Family Table

| family | representative domain | selected habits | probes | unique gold evidence sessions | session alignment | seed alignment | evidence source datasets | evidence domains | seed domains | capability |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `coding_review` | code | 9 | 10 | 38 | 100.0% | 100.0% | allenai/WildChat: 38 | code: 38 | code: 38 | Infer review ordering and patch minimality for code tasks. |
| `content_constraints` | food | 9 | 10 | 29 | 100.0% | 100.0% | allenai/WildChat: 29 | food: 29 | food: 29 | Infer routine content constraints for weekday family meals. |
| `drift_seasonality` | equipment | 9 | 10 | 40 | 100.0% | 100.0% | allenai/WildChat: 40 | equipment: 40 | equipment: 40 | Update a habit after sustained recent counterevidence. |
| `format_style` | work | 9 | 10 | 37 | 100.0% | 100.0% | allenai/WildChat: 37 | work: 37 | work: 37 | Infer recurring response format under a scoped work context. |
| `meeting_prep` | meeting | 10 | 10 | 26 | 100.0% | 100.0% | allenai/WildChat: 26 | meeting: 26 | meeting: 26 | Infer recurring meeting-prep document structure. |
| `planning_defaults` | travel | 10 | 10 | 42 | 100.0% | 100.0% | allenai/WildChat: 42 | travel: 42 | travel: 42 | Infer planning defaults for business travel. |
| `privacy_consent` | privacy | 7 | 10 | 35 | 100.0% | 100.0% | allenai/WildChat: 35 | privacy: 35 | privacy: 35 | Avoid durable use of sensitive one-off facts without consent. |
| `risk_threshold` | commitment | 8 | 10 | 34 | 100.0% | 100.0% | allenai/WildChat: 34 | commitment: 34 | commitment: 34 | Infer confirmation thresholds before costly or committing actions. |
| `tool_action` | news | 9 | 10 | 38 | 100.0% | 100.0% | allenai/WildChat: 38 | news: 38 | news: 38 | Infer when freshness checks are part of the user's workflow. |

## Reviewer-Facing Interpretation

Do not describe the current split as drawing each habit family from a different external dataset. The accurate claim is that all real task seeds come from WildChat, then are filtered into domain buckets that ground the nine controlled habit families with unique representative domains.
