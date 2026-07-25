# Stratified Manual Evidence-Chain Spot Check

在 2,048 条确定性全量审计之外，按 7 种 probe type 各抽查至少 1 条，人工阅读 query、四个 choices、decisive shortlist、远端 ordinal resolution、reference/as-of 元数据和 gold signature。

| Probe type | 抽查 probe | 结果 |
|---|---|---|
| `suggestion_rejection_pair` | `mdgo_v11_probe_000000` | 通过 |
| `reference_case_reconstruction` | `mdgo_v11_probe_000001` | 通过 |
| `triple_asof_interleaved` | `mdgo_v11_probe_000002` | 通过 |
| `dual_asof_reversal` | `mdgo_v11_probe_000003` | 通过 |
| `surface_decoy_pair` | `mdgo_v11_probe_000008` | 通过 |
| `scope_temporal_pair` | `mdgo_v11_probe_000037` | 通过 |
| `provenance_weighted_triple` | `mdgo_v11_probe_000050` | 通过 |

抽查重点：

- shortlist 中 workflow 的顺序是否清楚；
- resolution 的 first/second 指代是否唯一；
- as-of 时点是否位于正确的 baseline/replacement 区间；
- reference case 的 open workstream 是否正确；
- local exception 与 assistant suggestion 是否明确非绑定；
- gold choice 是否是唯一同时满足所有 target habits 的答案。

未发现新增语义矛盾。
