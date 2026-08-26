# v1.4 targeted quality patch

## Scope

This is a lightweight patch on v1.3. Historical lifelines and persona data are unchanged.

## Finance

Eight probes with incomplete visible workstream-to-policy reachability were excluded and replaced with graph-valid probes. The type profile is five `dual_asof_reversal` and one each of `reference_case_reconstruction`, `scope_temporal_pair`, and `triple_asof_interleaved`. See `finance_scope_anchor_replacements.csv`.

## Software

Four query-only placement clauses were removed because placement was not part of the memory capability and none of the four choices implemented it. See `software_query_choice_coverage_fixes.csv`.

## Surface cleanup

The full-corpus scan removed exact repeated sentences from 100 queries. This includes the Finance duplicates observed in adjudication and additional instances found outside that sample. No policy signature or gold label was changed by this cleanup.

## Release gates

All 2,048 probes pass:

1. gold–graph consistency;
2. query–choice completeness;
3. binding evidence topology;
4. surface quality and uniqueness.

The 8 regenerated Finance probes and 4 repaired Software probes remain marked for targeted human re-review.
