# HABIT-Bench Balanced v0.3 Candidate

Status: larger balanced candidate split; automatic validation passed; human
audit still required before paper-scale claims.

## Source

- Real prompt seed source: `allenai/WildChat`.
- Domain assignment: keyword-filtered WildChat task seed buckets, with one
  representative domain per habit family.
- Controlled synthetic components: hidden habit graphs, assistant feedback,
  counterfactual probes, answer choices, gold labels, and evidence links.
- Accurate release claim: real-prompt-seeded, domain-grounded, synthetic
  longitudinal habit benchmark.
- Claim to avoid: each habit family is drawn from a different external dataset.
- Input pool: `work\habit-bench-builder\runs\habit_bench_pilot_v0`.

## Contents

- Users: 174
- Sessions: 9924
- Probes: 2010
- Selected habits per family: 30
- Stress variants: original balanced + deterministic unseen paraphrase for
  habit-stress probes.

## Important Boundary

This split is larger and more balanced than v0.2, but most rows are selected
from the auto-validated pilot pool rather than the senior-reviewed v0.2 set.
Use it for scaling experiments and review planning. Treat v0.2 as the stronger
reviewed evidence set until v0.3 receives human audit.

## Files

- `public/lifelines.jsonl`: histories for evaluated memory systems.
- `public/probes.jsonl`: public queries and answer choices.
- `private/probe_key.jsonl`: gold labels, evidence ids, and hidden habit graphs.
- `reports/balanced_v03_manifest.json`: counts, balance, validation status.
- `review/balanced_review_queue_sample.csv`: stratified human audit sample.
