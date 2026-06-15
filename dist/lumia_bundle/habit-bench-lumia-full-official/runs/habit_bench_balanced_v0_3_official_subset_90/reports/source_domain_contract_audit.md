# Source/Domain Contract Audit

- Status: `pass`
- Dataset: `work\habit-bench-builder\runs\habit_bench_balanced_v0_3_official_subset_90`
- Expected source: `allenai/WildChat`
- Users: 67
- Sessions: 3815
- Public probes: 90
- Private keys: 90
- Families: 9
- Session source datasets: allenai/WildChat: 3815
- Session seed domains: code: 505, commitment: 485, equipment: 646, food: 483, general: 239, meeting: 481, news: 135, privacy: 223, travel: 124, work: 494

## Family Contract

| family | representative domain | probes | habits | evidence sessions | session-domain alignment | seed-domain alignment | source alignment |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coding_review | code | 10 | 9 | 38 | 1.000 | 1.000 | 1.000 |
| content_constraints | food | 10 | 9 | 29 | 1.000 | 1.000 | 1.000 |
| drift_seasonality | equipment | 10 | 9 | 40 | 1.000 | 1.000 | 1.000 |
| format_style | work | 10 | 9 | 37 | 1.000 | 1.000 | 1.000 |
| meeting_prep | meeting | 10 | 10 | 26 | 1.000 | 1.000 | 1.000 |
| planning_defaults | travel | 10 | 10 | 42 | 1.000 | 1.000 | 1.000 |
| privacy_consent | privacy | 10 | 7 | 35 | 1.000 | 1.000 | 1.000 |
| risk_threshold | commitment | 10 | 8 | 34 | 1.000 | 1.000 | 1.000 |
| tool_action | news | 10 | 9 | 38 | 1.000 | 1.000 | 1.000 |

## Errors

- none

## Warnings

- none
