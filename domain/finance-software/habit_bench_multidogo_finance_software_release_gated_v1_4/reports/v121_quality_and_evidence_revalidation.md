# v1.2.1 时间线修订与 Evidence Chain 全量复验报告

## 1. 为什么不建议保留 2029 年

v1.2 的 2025–2029 时间线在纯合成 benchmark 内部并不构成逻辑错误：只要把这些日期理解为虚构用户时间线，模型仍可进行先后顺序和 as-of 推理。

但在 2026 年运行评测时，2029 年会引入一个与目标能力无关的额外变量：

- 模型的系统日期和 benchmark 日期冲突；
- 某些模型可能把 2029 误解为现实未来，而不是离线历史记录；
- 题目中的 `2033–2036` 样式记录编号也容易被误认为年份；
- 这会把“长程 habit memory”与“如何处理未来日期”混在一起。

因此，v1.2.1 将完整合成时间线统一平移 **1,461 天**：

| 项目 | v1.2 | v1.2.1 |
|---|---|---|
| Session 时间范围 | 2025-01-05 至 2029-06-18 | 2021-01-05 至 2025-06-18 |
| Probe 时间范围 | 2029-06-12 至 2029-09-17 | 2025-06-12 至 2025-09-17 |
| 相对时间间隔 | 原始值 | 完全保留 |
| Evidence session 顺序 | 原始值 | 完全保留 |
| Gold / habit graph | 原始值 | 完全保留 |

该平移只消除现实日期冲突，不改变任务难度和时间推理结构。

## 2. 全量复验发现的实际问题

对 v1.2 的 2,048 条 probe 做逐条证据链复验时，发现：

- shortlist 与 resolution 的 evidence topology 正确；
- selected ordinal 与 variant 的映射正确；
- gold choice 与 evidence 推出的 policy signature 一致；
- 但 **168 条 temporal probe 的题面时间粒度不足**。

这些题有时只写：

```text
May 2029
```

或只写某一天，但该月或该日内部发生了 policy replacement。于是同一个题面日期可能同时对应 old policy 和 new policy，不能唯一确定 gold。

按 habit component 计，共发现 227 个这种时间粒度冲突；按 probe 去重后是 168 条。

## 3. v1.2.1 的修复

### 3.1 所有 as-of 题使用精确时间

v1.2.1 对全部 **960 条 as-of probe** 使用明确到分钟的 benchmark-local 时间，例如：

```text
as of 4:00 pm on May 16, 2023
```

这样不会再出现“更新发生在该月/该日内部，但题面没有说明更新前还是更新后”的歧义。

### 3.2 中性化记录编号

原来的：

```text
V11-SW-2033-10000
```

改为：

```text
V12-SW-R10000
```

避免把档案编号误读为未来年份。

## 4. 每条证据链复验了什么

对 2,048 条 probe 逐条执行以下检查：

1. `required_component_groups` 中的 session 是否真实存在并属于同一用户；
2. shortlist session 与 resolution session 是否属于同一个 habit；
3. 两个 session 的 `pair_ref` 是否一致；
4. shortlist 中两个 workflow 是否真的出现在模型可见的 assistant 文本中；
5. resolution 是否在模型可见的 user 文本中明确选择 first/second、former/latter 或 option one/two；
6. `selected_ordinal` 是否正确映射到 `variant_id`；
7. 对 current、as-of 和 reference-case 三类状态，独立重建当时有效的 policy；
8. evidence 推出的 variant 是否与 hidden gold 的 `choice_policy_signatures` 一致；
9. 四个 choices 中是否只有一个 choice 同时满足所有目标 habit；
10. choice 文本是否确实表达其 metadata 中声明的 policy variant；
11. one-case exception 是否明确限定为单次、不可升级为长期 habit；
12. assistant suggestion 是否明确未被用户采纳；
13. reference case 是否指向正确的 open workstream 和当时 policy；
14. 所有 evidence 是否早于 probe，且属于相同 pseudo-user；
15. as-of 时间是否在题面中精确出现，且足以唯一确定状态。

## 5. 最终结果

| Probe type | 数量 | 通过 | 失败 |
|---|---:|---:|---:|
| `dual_asof_reversal` | 384 | 384 | 0 |
| `triple_asof_interleaved` | 512 | 512 | 0 |
| `surface_decoy_pair` | 384 | 384 | 0 |
| `reference_case_reconstruction` | 320 | 320 | 0 |
| `suggestion_rejection_pair` | 256 | 256 | 0 |
| `scope_temporal_pair` | 64 | 64 | 0 |
| `provenance_weighted_triple` | 128 | 128 | 0 |
| **总计** | **2,048** | **2,048** | **0** |

最终审计结论：

```text
semantic_chain_pass_count = 2048
semantic_chain_fail_count = 0
as_of_granularity_ambiguous_count = 0
```

## 6. 证据链为何能够支持答案

每个 durable habit 的最小决定链通常由两条相隔很远的 session 构成：

```text
ordered shortlist
    +
distant ordinal resolution
    =
selected policy variant
```

多 habit probe 会组合 2–3 组这样的链。Temporal probe 还要根据精确 as-of 时间判断使用 baseline 还是 replacement 记录；reference-case probe 还要定位历史 case 中未完成的 workstream。

因此，`session_id` 不是随便挑出的相关 session，而是可以机械地重建 gold 的 decisive chain。

## 7. 非绑定证据的处理

`nonbinding_evidence_session_ids` 中的内容可能在表面上支持另一个 workflow，甚至偶尔与最终 gold 使用同一个 workflow。但它们明确标记为：

- 仅适用于当前 case 的 local exception；或
- assistant 提议但用户未采纳的 suggestion。

其作用是测试 provenance、scope 和 user ratification，而不是作为正向 gold evidence。

## 8. 复验文件

- `reports/evidence_chain_semantic_audit_summary.json`：全量审计汇总；
- `reports/evidence_chain_semantic_audit_per_probe.csv`：2,048 条逐题结果；
- `private/probe_evidence_chains.jsonl`：完整多 session 证据链；
- `private/probe_evidence_chain_edges.csv`：25,248 条 probe–session 证据边；
- `reports/temporal_consistency_audit_v121.csv`：逐题时间一致性；
- `reports/validation_report_v121.json`：最终 package validator 输出；
- `scripts/audit_evidence_chains_semantic.py`：可复现全量语义链审计。

## 9. 结论与限制

v1.2.1 已消除未来时间线混淆和 as-of 时间粒度歧义；每条 evidence chain 都能在当前 benchmark contract 下唯一推出 hidden gold。

这里的“全量复验”是基于公开 session 文本、private annotation、policy signature 和时间状态进行的确定性审计，并辅以覆盖全部 7 种题型的分层人工抽查。它证明数据内部自洽，但 paper-scale release 仍建议保留独立 reviewer 对自然性和任务合理性的抽查。
