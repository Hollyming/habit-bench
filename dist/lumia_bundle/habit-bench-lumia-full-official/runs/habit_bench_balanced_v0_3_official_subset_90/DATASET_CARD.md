# HABIT-Bench Official Full-Config Subset

This stratified subset is intended for expensive full official-method runs on Lumia or another GPU/server environment.

## Source And Domain Contract

- Real prompt seed source: `allenai/WildChat`.
- Domain assignment: keyword-filtered WildChat task seed buckets, with one
  representative domain per habit family.
- Controlled synthetic components: hidden habit graphs, assistant feedback,
  counterfactual probes, answer choices, gold labels, and evidence links.
- Accurate release claim: real-prompt-seeded, domain-grounded, synthetic
  longitudinal habit benchmark.
- Claim to avoid: each habit family is drawn from a different external dataset.

## Contents

- Source split: `work\habit-bench-builder\runs\habit_bench_balanced_v0_3`
- Probes: 90
- Users: 67
- Sessions: 3815
- Include variants: `all`
