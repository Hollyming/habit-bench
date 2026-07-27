# v1.3 Release Notes

- Rephrased all 64 `scope_temporal_pair` queries so the first workstream is explicitly current and the second is explicitly historical.
- Removed historical deictic conflicts such as “stands now” in the second workstream.
- Clarified the visible scope and global replacement semantics of `finance_balance_statement_summary_first` for `mdgo_v05_fin_user_0006`.
- Regenerated evidence-chain excerpts and review previews after the four-session history patch.
- Added private decision-unit indexes and exact-match decision-unit macro scoring to prevent repeated use of one latent decision from dominating aggregate results.
- Retained 2,048 probes, 29,160 sessions, all choices, all gold labels, and all IDs.
