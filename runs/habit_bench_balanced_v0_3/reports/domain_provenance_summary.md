# HABIT-Bench Domain Provenance Summary

- Dataset: `work\habit-bench-builder\runs\habit_bench_balanced_v0_3`
- Status: `pass`
- Users: 174
- Sessions: 9924
- Probes: 2010

## Source Contract

HABIT-Bench currently uses a single real external prompt source and nine unique representative task domains. The real prompts provide task surface form; hidden habits, feedback, probes, answer choices, labels, and evidence links are controlled synthetic components.

- Real prompt seed source: allenai/WildChat: 9924
- Family-domain contract: `nine_unique_representative_domains`
- Seed domain buckets: code: 1351, commitment: 1328, equipment: 1612, food: 1198, general: 650, meeting: 1311, news: 335, privacy: 483, travel: 350, work: 1306
- Session domains: code: 1351, commitment: 1328, equipment: 1612, food: 1198, general: 650, meeting: 1311, news: 335, privacy: 483, travel: 350, work: 1306

## 9-Family Table

| family | representative domain | selected habits | probes | unique gold evidence sessions | session alignment | seed alignment | evidence source datasets | evidence domains | seed domains | capability |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `coding_review` | code | 30 | 210 | 180 | 100.0% | 100.0% | allenai/WildChat: 180 | code: 180 | code: 180 | Infer review ordering and patch minimality for code tasks. |
| `content_constraints` | food | 30 | 210 | 180 | 100.0% | 100.0% | allenai/WildChat: 180 | food: 180 | food: 180 | Infer routine content constraints for weekday family meals. |
| `drift_seasonality` | equipment | 30 | 270 | 240 | 100.0% | 100.0% | allenai/WildChat: 240 | equipment: 240 | equipment: 240 | Update a habit after sustained recent counterevidence. |
| `format_style` | work | 30 | 210 | 180 | 100.0% | 100.0% | allenai/WildChat: 180 | work: 180 | work: 180 | Infer recurring response format under a scoped work context. |
| `meeting_prep` | meeting | 30 | 210 | 180 | 100.0% | 100.0% | allenai/WildChat: 180 | meeting: 180 | meeting: 180 | Infer recurring meeting-prep document structure. |
| `planning_defaults` | travel | 30 | 210 | 180 | 100.0% | 100.0% | allenai/WildChat: 180 | travel: 180 | travel: 180 | Infer planning defaults for business travel. |
| `privacy_consent` | privacy | 30 | 270 | 180 | 100.0% | 100.0% | allenai/WildChat: 180 | privacy: 180 | privacy: 180 | Avoid durable use of sensitive one-off facts without consent. |
| `risk_threshold` | commitment | 30 | 210 | 180 | 100.0% | 100.0% | allenai/WildChat: 180 | commitment: 180 | commitment: 180 | Infer confirmation thresholds before costly or committing actions. |
| `tool_action` | news | 30 | 210 | 180 | 100.0% | 100.0% | allenai/WildChat: 180 | news: 180 | news: 180 | Infer when freshness checks are part of the user's workflow. |

## Reviewer-Facing Interpretation

Do not describe the current split as drawing each habit family from a different external dataset. The accurate claim is that all real task seeds come from WildChat, then are filtered into domain buckets that ground the nine controlled habit families with unique representative domains.
