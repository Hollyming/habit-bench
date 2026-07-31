# HABIT-Bench v3 人工审计与盲审裁决结果

审计日期：2026-07-30

审计协议：`docs/human_audit/HUMAN_AUDIT_GUIDE.md`

审计范围：Food v4、Finance v1.3、Software v1.3
抽样：固定 seed 42，按 `domain × probe_type` 分层，每层最多 50 题

## 1. 结论摘要

本轮共审计 851 题。Annotator A、B 的原始文件先被冻结；第三审查者 C
在不知道 gold 的情况下，仅对 A/B 的 204 个分歧题逐题完成语义裁决。每个分歧题都
重新阅读了完整 query、四个 choices 和 evidence packet 的全部 session，不是关键词
路由、规则匹配、标签拷贝或多数票。完成 C 后先生成完整 `adjudicated.csv` 和 SHA-256
冻结清单，之后数据管理员阶段才读取 private key。

需要准确说明人员属性：Reviewer C 是 Codex 执行的逐题完整阅读式 **AI
adjudication**，不是一名现实中的第三位人类标注员。数据管理员阶段也是在冻结盲审后
由 Codex 按预先声明的 keep/modify/exclude 规则执行。论文可以报告为“两位独立标注员
+ 一名 blind AI adjudicator”，不能在没有额外人类复核时写成“三位人类标注员”或
“fully human-verified”。

最终 release disposition：

| Domain | N | Keep | Modify | Exclude |
| --- | ---: | ---: | ---: | ---: |
| Food | 200 | 83 (41.50%) | 114 (57.00%) | 3 (1.50%) |
| Finance | 340 | 310 (91.18%) | 22 (6.47%) | 8 (2.35%) |
| Software | 311 | 307 (98.71%) | 4 (1.29%) | 0 |
| **合计** | **851** | **700 (82.26%)** | **140 (16.45%)** | **11 (1.29%)** |

最重要的发现是 Food v4 的 gold-label 映射缺陷：

- 150 个 latent-habit probes（`direct_use`、`boundary`、`exception`）中，147
  个有唯一 adjudicated choice；
- 这 147 个 choice **全部 147/147** 与源 private key 中
  `hidden_habit_graph` 的目标动作逐字一致；
- 当前数据集 gold 在全部 150 题中仅 **38/150** 与自己的 hidden graph 一致，
  112/150 不一致；
- 在 147 个可答题里有 110 个 adjudicated choice 与当前 gold ID 不同，需要修正
  answer key；
- 同一 Food 样本的 50 个 `explicit_retrieval` probes 为 **50/50** human–gold
  一致。

因此 Food 当前约 44% 的 human–gold agreement 不能解释成潜在习惯题不可答，也不能
用于支持论文核心假设；它首先证明了 stress-v4 latent probes 的
`gold_choice_id` 与 hidden graph/choice text 没有正确对齐。Food v4 上依赖当前 gold
计算的 method Accuracy 必须视为受污染结果。应生成新数据版本、修正 answer key、
复核 8 个 evidence packet 和 3 个不可答题，并重跑所有受影响方法。

## 2. 盲审完整性与冻结证据

| Domain | 总题数 | A/B 分歧题 | 分歧率 | C 覆盖 | Consensus 题 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Food | 200 | 80 | 40.00% | 80/80 | 120 |
| Finance | 340 | 115 | 33.82% | 115/115 | 225 |
| Software | 311 | 9 | 2.89% | 9/9 | 302 |
| **合计** | **851** | **204** | **23.97%** | **204/204** | **647** |

分歧定义为 `selected_best_choice_id` 或九个二元审计字段中任一字段不同。
`exclusion_reason`/`notes` 的纯文本差异本身不触发第三审。

C 文件的冻结 SHA-256：

| Domain | `adjudicator_c_review.csv` SHA-256 |
| --- | --- |
| Food | `fe0133d9b7ff85ba568af97fcfbdc9e0012303e12be644a612ee8e0c3bc6222f` |
| Finance | `8b272e64c43f868b30e947556b96d5275350956ee23548c5bb2f0ccd8526d376` |
| Software | `698e1ddb94d72b50add8da03bc05ad4f8d11e200d081aacb0897e284a9dcccaa` |

每个域的 `blind_adjudication_manifest.json` 还记录了 A、B、分歧表、C 和最终
`adjudicated.csv` 的 SHA-256。数据管理员脚本在读取 private key 前强制重新校验这些
哈希；任一文件在解盲前被改动都会终止。

## 3. Pre-adjudication：A/B 标注覆盖和 gold agreement

`gold_choice_agreement` 是数据有效性统计，不是模型 Accuracy。

| Domain | Annotator | N | 合法 choice | Gold agreement | Needs-modification rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Food | A | 200 | 200 | 44.00% | 1.50% |
| Food | B | 200 | 197 | 44.16% | 28.50% |
| Finance | A | 340 | 340 | 100.00% | 9.41% |
| Finance | B | 340 | 260 | 95.77% | 23.53% |
| Software | A | 311 | 311 | 99.68% | 1.61% |
| Software | B | 311 | 307 | 100.00% | 1.29% |

### 3.1 Food A/B agreement

| Field | n | Raw agreement | Cohen's κ |
| --- | ---: | ---: | ---: |
| selected_best_choice_id | 197 | 99.49% | 0.993 |
| answerable_from_evidence | 200 | 98.50% | 0.000 |
| evidence_sufficient | 200 | 100.00% | N/A |
| scope_condition_correct | 200 | 98.50% | 0.000 |
| boundary_exception_correct | 200 | 98.00% | 0.000 |
| choices_balanced | 200 | 100.00% | N/A |
| language_natural | 200 | 88.00% | 0.727 |
| source_grounded | 200 | 98.50% | 0.000 |
| privacy_safe | 200 | 100.00% | N/A |
| needs_modification | 200 | 71.00% | 0.005 |

### 3.2 Finance A/B agreement

| Field | n | Raw agreement | Cohen's κ |
| --- | ---: | ---: | ---: |
| selected_best_choice_id | 260 | 95.77% | 0.944 |
| answerable_from_evidence | 340 | 76.47% | 0.000 |
| evidence_sufficient | 340 | 79.12% | 0.000 |
| scope_condition_correct | 340 | 77.65% | 0.000 |
| boundary_exception_correct | 340 | 77.65% | 0.000 |
| choices_balanced | 340 | 100.00% | N/A |
| language_natural | 340 | 100.00% | 1.000 |
| source_grounded | 340 | 76.47% | 0.000 |
| privacy_safe | 340 | 100.00% | N/A |
| needs_modification | 340 | 71.76% | 0.010 |

### 3.3 Software A/B agreement

| Field | n | Raw agreement | Cohen's κ |
| --- | ---: | ---: | ---: |
| selected_best_choice_id | 307 | 100.00% | 1.000 |
| answerable_from_evidence | 311 | 98.71% | 0.000 |
| evidence_sufficient | 311 | 100.00% | N/A |
| scope_condition_correct | 311 | 98.71% | 0.000 |
| boundary_exception_correct | 311 | 100.00% | N/A |
| choices_balanced | 311 | 100.00% | N/A |
| language_natural | 311 | 100.00% | 1.000 |
| source_grounded | 311 | 98.71% | 0.000 |
| privacy_safe | 311 | 100.00% | N/A |
| needs_modification | 311 | 97.11% | -0.014 |

多个高 raw-agreement 字段的 κ 为 0 或无法定义，是类别极端偏斜造成的 prevalence
effect。例如一位标注员几乎全部填 1 时，chance agreement 会接近 observed
agreement。不能把 κ=0 单独解释为“逐题随机”，也不能只选择更好看的 raw agreement；
论文应同时报告 `n`、raw agreement 和 κ。

## 4. Adjudication 后的逐域质量指标

| Domain | Valid choice | Gold agreement | Answerable | Evidence | Scope | Boundary | Balanced | Natural | Grounded | Privacy | Needs modify |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Food | 197/200 | 44.16% | 98.50% | 100.00% | 98.50% | 98.00% | 100.00% | 62.00% | 98.50% | 100.00% | 5.50% |
| Finance | 332/340 | 100.00% | 97.65% | 97.65% | 97.65% | 97.65% | 100.00% | 90.59% | 97.65% | 100.00% | 8.82% |
| Software | 307/311 | 100.00% | 98.71% | 100.00% | 98.71% | 100.00% | 100.00% | 98.39% | 98.71% | 100.00% | 1.29% |

Food 的 `language_natural=62%` 是独立的可读性警告。很多 query 使用 taxonomy 式、
模板化或跨菜谱的措辞。盲审认为其中多数“生硬但仍可理解”，所以
`needs_modification=0`，没有把所有 naturalness fail 自动升级成 release-blocking
modify；下一数据版本仍建议统一重写并重新审计。

## 5. Keep / Modify / Exclude 的逐域依据

### 5.1 Food

- **110 个 gold-label modify**：adjudicated choice 与当前
  `gold_choice_id` 不同；其中所有有答案的 latent choices 都与 hidden graph
  对应动作一致。106 题只有 gold-label 问题，4 题同时存在 evidence packet 问题。
- **另 4 个 packet modify**：当前 gold 可对齐，但 evidence 内存在“声称选择一种
  route、随后却执行另一种 route”等内部矛盾。
- **3 个 exclude**：`AUDIT-00011`、`AUDIT-00040`、`AUDIT-00057`。query 要求
  allium-before-liquid 的 scope，但反复 ratified policy 和 offered choices 都不能
  满足该 scope；修复会改变 target condition，不应静默改成同一道题。
- **83 个 keep**：通过 release rule，且当前 gold 不需要修正。

Food 的 8 个 packet-error 题为：

```text
AUDIT-00019, AUDIT-00038, AUDIT-00060, AUDIT-00125,
AUDIT-00151, AUDIT-00165, AUDIT-00194, AUDIT-00200
```

### 5.2 Finance

- **22 个 modify**：query/context 中出现连续的完全重复句。policy 和 gold 可唯一
  恢复，只需删除重复文本并重新审计。
- **8 个 exclude**：目标 workstream 没有被任何 evidence 锚定到指定 reference，
  把 replacement-card policy 迁移到 address change 等其他 scope 没有依据。补齐
  evidence 会改变证据拓扑，因此排除而不是局部润色。
- **310 个 keep**：有效 choice 与 gold 100% 一致，核心有效性字段通过。

Finance 排除题：

```text
AUDIT-00118, AUDIT-00123, AUDIT-00176, AUDIT-00193,
AUDIT-00222, AUDIT-00233, AUDIT-00244, AUDIT-00317
```

### 5.3 Software

- **4 个 modify**：`AUDIT-00202`、`AUDIT-00221`、`AUDIT-00269`、
  `AUDIT-00289`。其 substantive policy pair 与 gold 可以恢复，但 query 额外要求
  把内容放入某个 section，而四个 choices 全部遗漏该 placement 子任务。删除这个
  非目标 clause，或给所有 choices 同等补上 placement，可在不改变 target habit、
  decision unit、evidence topology 和能力目标的情况下修复。
- **307 个 keep**。
- **0 个 exclude**。

## 6. Probe-type 层面的主要现象

Food：

| Probe type | N | Valid | Gold agreement | Keep | Modify | Exclude |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| boundary | 50 | 50 | 26.00% | 12 | 38 | 0 |
| direct_use | 50 | 47 | 17.02% | 8 | 39 | 3 |
| exception | 50 | 50 | 32.00% | 15 | 35 | 0 |
| explicit_retrieval | 50 | 50 | 100.00% | 48 | 2 | 0 |

该切分进一步定位到 latent probe 的 gold remapping，而不是一个跨所有 Food probe
type 的通用标注问题。

Finance 的 8 个 exclude 集中在需要跨 scope/as-of 组合的题：

- `dual_asof_reversal`：5/50；
- `reference_case_reconstruction`：1/50；
- `scope_temporal_pair`：1/40；
- `triple_asof_interleaved`：1/50。

Finance 的 22 个语言修改主要集中在
`provenance_weighted_triple`（12/50），其余为 suggestion/surface 等题中的重复句。

Software 的 4 个 modify 分别出现在
`reference_case_reconstruction`（2/50）和
`suggestion_rejection_pair`（2/50）。

完整 probe-type 表不在本文中重复所有字段，见各域
`scored/human_audit_adjudicated_by_probe_type.csv`。

## 7. 对论文与正式实验的影响

1. **Finance/Software 可以在处理清单后继续使用。** Finance 先移除 8 个无 scope
   锚点的题并修复 22 个重复句；Software 修复 4 个 placement clause 后重新审计。
2. **Food v4 当前 Accuracy 不可直接进论文主结果。** 110 个 sampled answer-key
   mismatch 且 112/150 latent gold 与 hidden graph 不一致，比例过高，不是可忽略的
   label noise。
3. **修复必须产生新版本。** 不要静默覆盖 Food v4 或 Finance/Software v1.3；应更新
   probe hash、review artifact、audit manifest，并重跑所有受影响 memory methods。
4. **核心假设仍需模型实验支持。** 本人工审计能证明 oracle evidence 下题目质量和
   gold 一致性，不能证明 memory agent 会失败。特别是 Food gold 修复前的低 Accuracy
   不能作为“latent habit 更难”的证据。
5. **论文的人审表述必须准确。** 当前是 A/B 独立标注 + blind AI adjudication；
   若要写成 fully human adjudicated，需要现实中的人类 C 对 204 个分歧题再次独立
   复核，且在看 gold 前冻结。

## 8. 结果文件地图

根目录：

```text
results/habit_3domain_v3/supplementary/human_audit/
├── human_audit_final_summary.json
└── <domain>/
    ├── annotations/
    │   ├── annotator_a.csv
    │   └── annotator_b.csv
    ├── adjudication/
    │   ├── disagreements.csv
    │   ├── adjudicator_c_review.csv
    │   ├── adjudicated.csv
    │   └── blind_adjudication_manifest.json
    └── scored/
        ├── human_audit_metrics.json
        ├── human_audit_by_annotator.csv
        ├── human_audit_agreement.csv
        ├── human_audit_adjudicated_metrics.json
        ├── human_audit_adjudicated_by_probe_type.csv
        ├── human_audit_dispositions.csv
        └── unblinding_manifest.json
```

文件用途：

- `human_audit_final_summary.json`：三域合并后的机器可读最终结论和 manifest 哈希；
- `disagreements.csv`：完整题目、evidence 和 A/B 判断；
- `adjudicator_c_review.csv`：C 对全部分歧题的冻结裁决；
- `adjudicated.csv`：647 个 consensus 判断加 204 个 C 判断构成的全量 851 题结果；
- `human_audit_metrics.json`：只用原始 A/B 计算的 pre-adjudication 指标；
- `human_audit_agreement.csv`：原始 A/B 的逐字段 raw agreement 与 κ；
- `human_audit_adjudicated_metrics.json`：解盲后的最终质量指标和 Food hidden-graph
  consistency；
- `human_audit_dispositions.csv`：每题 keep/modify/exclude、理由和必要动作，是数据
  修订的主清单；
- `blind_adjudication_manifest.json`：证明 C 合并在 gold 之前冻结；
- `unblinding_manifest.json`：private inputs 和 scored outputs 的 SHA-256。

## 9. 可复现工具

本次新增两个职责隔离的工具：

```text
eval/supplementary/human_audit_adjudication.py
eval/supplementary/human_audit_data_manager.py
```

前者的 `prepare/render/merge` 不接受 private audit key；`merge` 校验 C 覆盖后生成
全量盲审结果和 freeze manifest。后者只用于解盲阶段，并在打开 private key 之前校验
freeze manifest 中的所有哈希，再输出逐题 disposition 和汇总指标。两者都不修改
Annotator A/B/C 的冻结文件。
