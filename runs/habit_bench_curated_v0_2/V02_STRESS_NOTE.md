# HABIT-Bench Curated v0.2 Stress Note

Status: curated_v0_2_with_unseen_paraphrase_stress

v0.2 preserves the reviewed v0.1 histories and labels, then adds deterministic
`unseen_paraphrase` variants for direct-use, boundary, exception, drift, and
privacy probes.

Counts:

- Original probes: 161
- Derived unseen-paraphrase probes: 125
- Total probes: 286

Purpose:

- Reduce surface overlap between test queries and visible history episodes.
- Stress raw episode/segment retrieval methods that rely on exact wording.
- Keep gold labels and evidence stable so v0.1 and v0.2 can be compared.
