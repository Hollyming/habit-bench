# HABIT-Bench 三域人工盲审操作手册

本文说明 HABIT-Bench v3 的 Food v4、Finance v1.3 和 Software v1.3 人工质量审计如何执行、文件放在哪里、每一列怎样填写、如何计算一致性，以及论文中怎样解释结果。

本文是数据质量审计协议，不是 memory method 的模型评测协议。人工标注员阅读的是题目、四个 choices 和私有标注生成的 evidence packet，不阅读任何 memory method 的预测。因此：

- `gold_choice_agreement` 衡量人工判断与数据集 gold 的一致程度，不能写成模型 Accuracy；
- 人工审计用于验证题目是否唯一可答、证据是否充分、scope/exception/temporal 语义是否正确、选项是否平衡、语言是否自然、内容是否有来源且隐私安全；
- memory method 的 Accuracy、Recall@5、Complete Chain 等结果仍以正式评测输出为准。

## 1. 当前审计样本已经在哪里

项目根目录：

```text
/plm-shared/zhangjunming/Workspace/HABIT-bench
```

当前 v3 人工审计根目录：

```text
/plm-shared/zhangjunming/Workspace/HABIT-bench/results/habit_3domain_v3/supplementary/human_audit
```

三个域的目录：

```text
results/habit_3domain_v3/supplementary/human_audit/
├── food/
│   ├── annotation_template.csv
│   ├── audit_key.private.jsonl
│   └── audit_manifest.json
├── finance/
│   ├── annotation_template.csv
│   ├── audit_key.private.jsonl
│   └── audit_manifest.json
└── software/
    ├── annotation_template.csv
    ├── audit_key.private.jsonl
    └── audit_manifest.json
```

实际数据版本：

```text
Food:
/plm-shared/zhangjunming/Workspace/HABIT-bench/domain/food/food_habit_lifelines_stress_v4

Finance 与 Software:
/plm-shared/zhangjunming/Workspace/HABIT-bench/domain/finance-software/habit_bench_multidogo_finance_software_scope_consistent_v1.3
```

Finance 和 Software 来自同一个物理数据包，但在审计时分别使用 `domain_filter=finance` 和 `domain_filter=software`，必须作为两个独立域报告。

## 2. 当前抽样规模

样本由 `eval/supplementary/human_audit.py` 使用固定 seed 42，按 `domain × probe_type` 均匀分层抽取；每层最多 50 题。如果某一层不足 50 题，则全取。

| Domain | 样本数 | 分层数 | 具体组成 |
| --- | ---: | ---: | --- |
| Food | 200 | 4 | boundary 50、direct_use 50、exception 50、explicit_retrieval 50 |
| Finance | 340 | 7 | 6 层各 50，scope_temporal_pair 40 |
| Software | 311 | 7 | 5 层各 50，provenance_weighted_triple 37，scope_temporal_pair 24 |
| 合计 | 851 | 18 | 每题至少两位独立标注员 |

每位标注员若覆盖全部三域，需要完成 851 题。按每题 3–5 分钟估算，每人约需 43–71 小时。可以给 Food、Finance、Software 分配不同的标注员对，但同一域内的两位标注员必须审计完全相同的全部题目，才能计算逐题一致性。

## 3. 三个输入文件分别是什么

### 3.1 `annotation_template.csv`

这是唯一允许交给普通标注员的输入文件。它包含：

- 匿名审计 ID 和原 probe ID；
- domain 和 probe type；
- 当前 query；
- 四个 choices；
- 与该题相关的 evidence packet；
- 待填写的人工判断列。

它不包含 `gold_choice_id`。

### 3.2 `audit_key.private.jsonl`

这是私有答案文件，包含：

```text
item_id
probe_id
domain
probe_type
valid_choice_ids
gold_choice_id
evidence_session_ids
```

禁止把该文件交给两位独立标注员。标注员在独立阶段也不应查看原始 `private/probe_key.jsonl`、method predictions、scored predictions 或任何带 gold 的报告。

只有数据管理员和最终 unblinding 阶段可以使用该文件。该文件不应上传到公开仓库、邮件群组或公共标注平台。

### 3.3 `audit_manifest.json`

这是审计元数据，记录：

- 数据集名称、版本、用户数、session 数和 probe 数；
- 抽样策略；
- seed；
- 每个 stratum 的可用题数和抽中题数；
- 二元字段及其编码。

论文报告 sample size 和 strata 时，以该文件为准，不要手工推测。

## 4. Evidence packet 是怎样生成的

`annotated_evidence_packet_json` 是 JSON 字符串数组。数组中的每个元素是一条渲染后的 session，形式类似：

```text
[SESSION_ID=...]
[SESSION_INDEX=...]
[TIMESTAMP=...]
user: ...
assistant: ...
```

生成器会合并：

1. 该题 private key 中的决定性 evidence；
2. 必要 temporal context；
3. `nonbinding_evidence_session_ids` 指向的局部例外、未被采纳的 assistant suggestion 或其他非绑定证据；
4. 按 session index 排序后的文本。

因此 evidence packet 不是完整 lifelong history，也不是 memory method 检索结果。它是用于验证“给定标注证据后，这道题本身是否合理且可唯一作答”的 Oracle 质量审计材料。

标注员必须根据文本中的 user ratification、时间、scope、单次例外和 replacement 关系判断哪些 evidence 有约束力，不能把 packet 中出现过的所有 workflow 都当成长期习惯。

## 5. 人员角色和盲法

最低配置：

- Annotator A：独立填写完整域；
- Annotator B：独立填写同一完整域；
- Adjudicator C：只处理 A/B 的分歧，并作最终裁决；
- Data manager：保管 private key、运行评分、冻结修改/排除清单。

独立标注阶段必须遵守：

1. A 和 B 使用各自独立的 CSV 副本；
2. 不讨论具体题目；
3. 不查看对方文件；
4. 不查看 gold、模型预测或原始私有 habit graph；
5. 只使用 query、choices 和 evidence packet；
6. 记录标注员背景、培训方式、开始/结束时间和 rubric 版本。

如果标注员拥有共享文件系统账号，仅“口头要求不看 private key”不构成严格盲法。正式论文审计建议由数据管理员把模板复制到标注员可访问、但 private key 不可访问的位置，完成后再收回结果。

## 6. 初始化标注文件

不要直接编辑 `annotation_template.csv`，要为每位标注员复制独立文件。建议目录结构：

```text
<domain>/
├── annotation_template.csv
├── audit_key.private.jsonl
├── audit_manifest.json
├── annotations/
│   ├── annotator_a.csv
│   └── annotator_b.csv
├── adjudication/
│   ├── disagreements.csv
│   └── adjudicated.csv
└── scored/
    ├── human_audit_metrics.json
    ├── human_audit_by_annotator.csv
    └── human_audit_agreement.csv
```

数据管理员可在项目根目录执行：

```bash
cd /plm-shared/zhangjunming/Workspace/HABIT-bench

AUDIT_ROOT=/plm-shared/zhangjunming/Workspace/HABIT-bench/results/habit_3domain_v3/supplementary/human_audit

for domain in food finance software; do
  mkdir -p "$AUDIT_ROOT/$domain/annotations"
  mkdir -p "$AUDIT_ROOT/$domain/adjudication"
  mkdir -p "$AUDIT_ROOT/$domain/scored"
  cp -n "$AUDIT_ROOT/$domain/annotation_template.csv" \
    "$AUDIT_ROOT/$domain/annotations/annotator_a.csv"
  cp -n "$AUDIT_ROOT/$domain/annotation_template.csv" \
    "$AUDIT_ROOT/$domain/annotations/annotator_b.csv"
done
```

`cp -n` 用于避免覆盖已经填写的文件。分发给标注员时只分发对应的 `annotator_*.csv` 和不含 gold 的 rubric，不分发整个结果目录。

## 7. CSV 中哪些列不能改

以下列属于题目身份和展示内容，禁止修改、排序后错配或删除：

```text
item_id
probe_id
domain
probe_type
query
choices_json
annotated_evidence_packet_json
```

以下列由标注员填写：

```text
selected_best_choice_id
answerable_from_evidence
evidence_sufficient
scope_condition_correct
boundary_exception_correct
choices_balanced
language_natural
source_grounded
privacy_safe
needs_modification
exclusion_reason
notes
```

不要新增或删除行。评分器要求每份 annotation CSV 与 private key 的 `item_id` 集合完全一致；缺行、多行或重复 ID 都会报错。

推荐使用 UTF-8 CSV。使用 Excel 或 LibreOffice 时：

- 把 `item_id`、`probe_id` 和 choice ID 列作为文本导入；
- 保留逗号分隔和双引号转义；
- 不要让软件把长 JSON 单元格改写成科学计数法或截断；
- 保存后重新打开抽查 `choices_json` 和 `annotated_evidence_packet_json` 是否仍为完整 JSON；
- 不要通过复制粘贴整列破坏行对应关系。

## 8. 标注顺序

每题建议按以下顺序判断：

1. 先读 query，明确当前请求包含哪些子任务、scope 和时间点；
2. 展开 `choices_json`，确认每个 choice 的 `choice_id` 和完整文本；
3. 按 session index 阅读 evidence packet；
4. 区分 user 明确选择、assistant 提议、单次例外、rehearsal、replacement 和非绑定内容；
5. 不看 gold，独立推导最受证据支持的 choice；
6. 填写 `selected_best_choice_id`；
7. 再填写九个审计字段；
8. 对任何 `0`、空 choice 或 `needs_modification=1` 写清原因。

标注员不应先猜“数据集可能想选什么”，而应回答“仅凭给出的证据，一个谨慎的人类是否能唯一推出某个 choice”。

## 9. 每个字段怎样填写

九个二元字段统一使用：

```text
1 = yes / pass
0 = no / fail
```

评分程序也接受 `yes/no`、`true/false`、`pass/fail`，但正式标注建议只使用 `1/0`，减少格式差异。

确实不适用的字段留空，并在 `notes` 中写 `N/A: 原因`。不要把“不适用”强行写成 1，否则会虚高通过率。评分结果中的每个 `*_n` 是该字段的有效标注数，论文必须同时报告 `n` 和 rate。

### 9.1 `selected_best_choice_id`

填写 `choices_json` 中真实存在的 choice ID，例如 `A`、`B`、`C` 或 `D`，不要填写 choice 文本。

判定标准：

- 该 choice 同时满足 query 中所有子任务；
- 与 binding user evidence、scope 和目标时间一致；
- 不把单次例外推广成长期规则；
- 不把 assistant suggestion 当作用户已采纳的习惯；
- 多组件题必须同时满足全部组件，不能只命中其中一个。

如果没有唯一可辩护的 choice：

- `selected_best_choice_id` 留空；
- `answerable_from_evidence=0`；
- `needs_modification=1`；
- 必须填写 `exclusion_reason` 和 `notes`。

不要为了让 CSV 看起来完整而随机选一个。

### 9.2 `answerable_from_evidence`

问题、choices 和 evidence packet 合在一起，是否能在不依赖外部知识和猜测的情况下推出唯一最佳答案。

写 0 的典型情况：

- 两个或更多 choices 同样符合证据；
- query 的时间或 scope 不明确；
- evidence 和 choices 之间无法完成必要映射；
- packet 内部冲突且没有优先级规则；
- 没有任何 choice 完整满足多组件请求。

### 9.3 `evidence_sufficient`

packet 是否包含完成推理所需的所有决定性事实、必要时间上下文和 provenance。

它与 `answerable_from_evidence` 的区别：

- evidence 足够，但 choices 重复或题面歧义：`evidence_sufficient=1`、`answerable_from_evidence=0`；
- choices 本身清晰，但 packet 缺关键 resolution/replacement：两者通常都为 0。

### 9.4 `scope_condition_correct`

题目和最佳 choice 是否把习惯应用在正确的对象、任务、workstream、设备、账户、菜品结构或其他条件范围内。

重点检查：

- 相似表面词是否属于不同 scope；
- 一条 policy 是否被错误推广到另一个 workstream；
- query 的多个子任务是否分别绑定到正确 habit；
- current scope 和 historical scope 是否被混用。

### 9.5 `boundary_exception_correct`

题目、证据和最佳 choice 是否正确处理“何时不应用默认习惯”、局部例外、用户拒绝和 policy revision。

典型失败：

- 把 one-case exception 写成新的全局默认；
- 忽略明确 exception；
- 把 assistant suggestion 当作用户习惯；
- 用最新 policy 回答历史 as-of 问题；
- 用旧 policy 回答 replacement 后的当前问题；
- 把 nonbinding rehearsal 当成决定性证据。

对完全不涉及 boundary、exception、rejection 或 temporal revision 的题，可留空并在 `notes` 标注 N/A。

### 9.6 `choices_balanced`

四个 choices 是否在长度、语言质量、任务完成度、具体程度和安全性上大体匹配，gold 是否不会因表面线索显得特殊。

写 0 的典型情况：

- gold 明显更长、更完整或更自然；
- 只有一个 choice 遵守基本安全规范；
- choice label、固定短语或词汇重合直接泄露答案；
- distractor 明显荒谬或缺少 query 要求的子任务；
- 两个 choices 实质重复。

### 9.7 `language_natural`

query、choices 和 evidence 是否像自然、可理解的用户—助手交互，而不是难以理解的模板拼接。

允许 benchmark 为控制变量而保持一定结构一致，但以下情况应写 0：

- 指代无法解析；
- 语法破坏理解；
- 多个子任务连接生硬到无法判断 scope；
- ID 或模板残片意外泄漏生成过程；
- 同一句出现互相矛盾的时间表达。

### 9.8 `source_grounded`

最佳 choice 中用于决定答案的每个关键 policy component，是否都能追溯到 packet 内的有效 source；其 provenance 是否是 user ratification，而不是无约束的 assistant 内容。

它与 `evidence_sufficient` 的区别：

- `evidence_sufficient` 问“材料够不够作答”；
- `source_grounded` 问“答案的每个关键主张是否确实来自正确来源”。

如果最佳 choice 含有 packet 未支持的额外 policy component，应写 0，即使主要 choice 方向看似正确。

### 9.9 `privacy_safe`

题目、evidence 和最佳 choice 是否避免要求、复述或传播密码、PIN、验证码、完整敏感账号、私密身份信息或其他不必要秘密；是否保留了数据集中设计的安全处理边界。

包含虚构业务标识不自动构成隐私失败。判断重点是内容是否诱导暴露真实凭据或把应留在安全通道的信息带入聊天。

### 9.10 `needs_modification`

该题在正式 benchmark 发布前是否需要人工修改或排除。

以下任一实质问题通常应写 1：

- 不可唯一作答；
- evidence 缺失或错误；
- gold 语义疑似不一致；
- scope、exception、replacement 或 as-of 绑定不清楚；
- choices 存在明显 shortcut、不平衡或重复；
- 语言问题会影响理解；
- source grounding 或 privacy 有实质问题。

轻微措辞偏好但不影响可答性时，可以在 `notes` 提建议，同时保持 `needs_modification=0`。不要把所有风格建议都升级为数据失效。

### 9.11 `exclusion_reason`

只有需要修改、排除或无法选择答案时填写。建议使用受控词表：

```text
unanswerable
insufficient_evidence
ambiguous_choices
scope_mismatch
boundary_exception_mismatch
temporal_mismatch
gold_suspected_wrong
shortcut_or_choice_artifact
unnatural_language
source_not_grounded
privacy_issue
annotation_packet_error
other
```

多个原因用分号分隔。`other` 必须在 `notes` 中解释。

### 9.12 `notes`

建议写短而可复核的理由，尤其是：

- 任何二元字段为 0；
- `selected_best_choice_id` 为空；
- `needs_modification=1`；
- 某字段为 N/A；
- evidence 中哪些 session 或文本造成冲突。

不要在 notes 中写“我猜 gold 是 A”之类关于隐藏答案的推测。

## 10. 各 probe type 的审计重点

### 10.1 Food

| Probe type | 审计重点 |
| --- | --- |
| `direct_use` | 重复弱证据是否足以归纳一个 scoped 默认行为，最佳 choice 是否正确应用该默认 |
| `boundary` | 当前请求是否位于习惯适用边界之外，choice 是否避免错误个性化 |
| `exception` | 局部例外是否被保留，同时没有污染长期默认习惯 |
| `explicit_retrieval` | 是否能从 packet 中直接找回明确事实或先前决定，不应强行做潜在习惯推断 |

### 10.2 Finance 与 Software

这些题通常要求连接 2–3 组远距离 evidence。一个 durable policy 常由“较早的有序 shortlist + 较远的 ordinal resolution”共同确定，单条 session 通常不足以推出答案。

| Probe type | 审计重点 |
| --- | --- |
| `suggestion_rejection_pair` | assistant 建议是否被用户拒绝；拒绝的建议不能成为 standing policy |
| `reference_case_reconstruction` | 是否找到指定 reference case、未完成 workstream 和当时有效 policy |
| `triple_asof_interleaved` | 三个交错组件是否都按指定历史时间恢复，不能使用 probe 时刻的最新状态 |
| `dual_asof_reversal` | 两个组件经历 reversal/replacement 后，是否恢复 as-of 时刻分别有效的版本 |
| `surface_decoy_pair` | 表面词汇相近的非绑定 workflow 是否被误当成真正 policy |
| `scope_temporal_pair` | 一个 workstream 使用当前 standing policy，另一个使用题面明确时间点的历史 policy |
| `provenance_weighted_triple` | 三个组件是否正确区分 user ratification、assistant suggestion、local exception 和其他 provenance |

## 11. 标注完成前的自检

每位标注员提交前检查：

1. 行数和 `item_id` 没有变化；
2. 所有非空 choice ID 都存在于该行 `choices_json`；
3. 二元字段只含 `0`、`1` 或有说明的空值；
4. `answerable_from_evidence=0` 时，`needs_modification=1` 且有原因；
5. `needs_modification=1` 时，有 `exclusion_reason` 或 notes；
6. 所有失败判断都有可复核理由；
7. 文件仍能被标准 CSV reader 读取。

评分命令会严格检查：

- 是否有重复 `item_id`；
- 是否缺少或多出题目；
- annotator 名称是否唯一；
- 是否至少有两份 annotation。

但当前评分器不会强制所有二元字段都填满。因此不能把“命令成功”误解为“人工标注完整”。

## 12. 运行双人评分

使用固定虚拟环境：

```text
/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark/bin/python
```

Food：

```bash
cd /plm-shared/zhangjunming/Workspace/HABIT-bench

PYTHON=/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark/bin/python
DOMAIN_ROOT=/plm-shared/zhangjunming/Workspace/HABIT-bench/results/habit_3domain_v3/supplementary/human_audit/food

"$PYTHON" -m eval.supplementary.human_audit score \
  --audit-key "$DOMAIN_ROOT/audit_key.private.jsonl" \
  --annotation "annotator_a=$DOMAIN_ROOT/annotations/annotator_a.csv" \
  --annotation "annotator_b=$DOMAIN_ROOT/annotations/annotator_b.csv" \
  --output-dir "$DOMAIN_ROOT/scored"
```

Finance 和 Software 只需把 `DOMAIN_ROOT` 末尾改成对应 domain。也可以统一执行：

```bash
cd /plm-shared/zhangjunming/Workspace/HABIT-bench

PYTHON=/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark/bin/python
AUDIT_ROOT=/plm-shared/zhangjunming/Workspace/HABIT-bench/results/habit_3domain_v3/supplementary/human_audit

for domain in food finance software; do
  DOMAIN_ROOT="$AUDIT_ROOT/$domain"
  "$PYTHON" -m eval.supplementary.human_audit score \
    --audit-key "$DOMAIN_ROOT/audit_key.private.jsonl" \
    --annotation "annotator_a=$DOMAIN_ROOT/annotations/annotator_a.csv" \
    --annotation "annotator_b=$DOMAIN_ROOT/annotations/annotator_b.csv" \
    --output-dir "$DOMAIN_ROOT/scored"
done
```

评分是 CPU 离线统计，不需要 GPU，也不需要 ClusterX。

## 13. 三个评分输出怎么看

### 13.1 `human_audit_metrics.json`

这是最完整的机器可读汇总。

顶层字段：

```text
contract_version
items
annotators
annotator_metrics
inter_annotator_agreement
interpretation
```

`annotator_metrics` 中每位标注员包含：

- `items`：模板总题数；
- `valid_best_choice_annotations`：填写了合法 choice ID 的题数；
- `gold_choice_agreement`：人工最佳 choice 与数据集 gold 的一致率；
- `<field>_n`：该二元字段的有效标注数；
- `<field>_rate`：该字段中 1 的比例。

除 `needs_modification_rate` 外，通常 rate 越高代表质量越好。`needs_modification_rate` 越高表示越多题需要修订。

### 13.2 `human_audit_by_annotator.csv`

这是每位标注员一行的扁平表，适合直接放入统计表或绘图脚本。它包含和 `annotator_metrics` 相同的 per-annotator 指标。

### 13.3 `human_audit_agreement.csv`

每一行对应：

```text
一对 annotator × 一个字段
```

字段：

- `n`：两人都给出有效值的题数；
- `raw_agreement`：直接一致率；
- `cohen_kappa`：扣除随机一致后的 Cohen’s κ。

需要同时报告 raw agreement 和 κ。二元字段高度偏向全 1 时，raw agreement 可能很高而 κ 较低或无法定义，这是 prevalence effect，不应只挑一个有利数字。

可用于内部排查的粗略参考，而不是硬性论文门槛：

- κ ≥ 0.80：强一致；
- 0.60–0.80：较好；
- 0.40–0.60：中等，需要看分歧类型；
- κ < 0.40：rubric、题目或标注培训需要重点复核。

若 κ 为 `null`，先看两人的类别分布和 raw agreement；当 expected agreement 为 1 时 κ 无法计算，不等同于标注失败。

## 14. 第三人 adjudication 怎么做

当前脚本只计算独立标注和 pairwise agreement，不自动用多数票覆盖分歧。正确流程：

1. 先冻结 A、B 原始 CSV，生成 pre-adjudication 指标；
2. 按 `item_id` 对齐两份文件，不要假设行号永远一致；
3. 找出 `selected_best_choice_id` 或任一二元字段不同的题；
4. 生成 `adjudication/disagreements.csv`，保留题目、证据、A/B 判断和 notes；
5. Adjudicator C 在不知道 gold 的情况下阅读原题和证据，给出最终判断及理由；
6. 对 A/B 一致题沿用一致判断，对分歧题使用 C 的裁决，形成完整的 `adjudicated.csv`；
7. 完成裁决后再由 data manager unblind，与 private gold 对照；
8. 冻结 `keep / modify / exclude` 决策。

第三人不应仅根据“谁更资深”选择 A 或 B，也不应先看 gold 再让结论迎合 gold。

论文中：

- raw agreement 和 κ 使用 adjudication 前的 A/B 文件；
- 数据接受率、修改率和排除率使用 adjudication 后的最终决策；
- 不用 adjudication 后被强制统一的标签重新计算 κ。

核心评分器 `eval/supplementary/human_audit.py` 不会自动覆盖分歧或执行多数票。
当前仓库另有两个职责隔离的辅助工具：

```text
eval/supplementary/human_audit_adjudication.py
eval/supplementary/human_audit_data_manager.py
```

前者只机械生成分歧表、渲染完整题目并合并已经填写的 C 裁决；它不接受 private key，
也不产生任何语义判断。后者只在 blind adjudication 已冻结后使用，先验证
`blind_adjudication_manifest.json` 中 A/B/C/合并文件的 SHA-256，再读取 private
key 并输出逐题 `keep/modify/exclude` 清单。两个工具均不得覆盖 A/B/C 原始文件。

当前 v3 已完成审计的详细结果、限制和文件地图见：

```text
docs/human_audit/HUMAN_AUDIT_RESULTS_V3.md
```

## 15. 接受、修改和排除规则

建议在看 gold 前冻结以下规则。

### Keep

题目应满足：

- 唯一可答；
- evidence 充分；
- scope/temporal 绑定正确；
- 相关 boundary/exception 处理正确；
- choices 基本平衡；
- 语言可理解；
- source grounded；
- privacy safe；
- `needs_modification=0`。

### Modify

问题可修复，且修复不改变目标 habit、decision unit、证据拓扑或目标能力。例如轻度措辞歧义、choice 长度失衡、缺少一个明确时间短语。

任何修改都应进入新的 dataset version，重新生成 probe hash、review artifact 和相关评测结果。不要静默修改已经用于 v3 实验的 Food v4 或 Finance/Software v1.3。

### Exclude

以下问题通常应排除，而不是微调措辞后继续当作同一题：

- gold 无法由 evidence 推出；
- 多个 choices 同样正确；
- 关键 evidence 错用户、错时间或不存在；
- 修复需要改变目标 habit/policy；
- 隐私或安全问题不可局部修复；
- 严重 template leakage 或 shortcut。

## 16. 论文中至少报告什么

每个 domain 单独报告：

```text
dataset version
sampling strategy and seed
number of strata
sample size
annotator count and background
valid annotation coverage
gold-choice agreement by annotator
each audit field's n and pass rate
raw agreement
Cohen's kappa
adjudicated keep / modify / exclude counts
```

建议表格：

| Domain | N | Gold agreement | Answerable | Evidence sufficient | Scope | Boundary/exception | Choice balance | Naturalness | Grounded | Privacy | Modify rate | Raw agree | κ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

不要写：

- “human-verified”但仍有未裁决分歧；
- 把 gold agreement 当模型性能；
- 只报告 κ、不报告 raw agreement 和有效 `n`；
- 把空字段默认为通过；
- 在旧版本上修题后仍沿用旧版本名和旧实验结果。

## 17. 当前协议的限制

当前模板验证的是“给 Oracle evidence packet 后能否唯一作答”。它没有自动实现以下实验：

- query/choices-only 与 evidence-present 的两阶段 memory-necessity 双盲比较；
- Krippendorff’s α；
- 自动逐题 adjudication；
- 自动修改数据集；
- 对 open-ended model response 的人工偏好评测。

如果要做 memory-necessity 双盲测试，应先让标注员在不看 evidence 的独立表上判断是否可答，再经过隔离或重新随机化后提供 evidence 版本。不能先看 evidence 再回填 no-evidence 判断，否则会产生泄漏。该扩展应作为独立协议和输出，不要混入当前 `human_audit_scores.v1`。

## 18. 如何重新生成样本

当前 v3 模板已经生成，正式审计期间不要重跑 `prepare`，否则可能覆盖冻结样本。

如创建新的实验版本，可执行：

```bash
cd /plm-shared/zhangjunming/Workspace/HABIT-bench

PYTHON=/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark/bin/python

"$PYTHON" -m eval.supplementary.human_audit prepare \
  --dataset-dir /plm-shared/zhangjunming/Workspace/HABIT-bench/domain/food/food_habit_lifelines_stress_v4 \
  --output-dir /path/to/new_audit/food \
  --per-stratum 50 \
  --seed 42

"$PYTHON" -m eval.supplementary.human_audit prepare \
  --dataset-dir /plm-shared/zhangjunming/Workspace/HABIT-bench/domain/finance-software/habit_bench_multidogo_finance_software_scope_consistent_v1.3 \
  --domain-filter finance \
  --output-dir /path/to/new_audit/finance \
  --per-stratum 50 \
  --seed 42

"$PYTHON" -m eval.supplementary.human_audit prepare \
  --dataset-dir /plm-shared/zhangjunming/Workspace/HABIT-bench/domain/finance-software/habit_bench_multidogo_finance_software_scope_consistent_v1.3 \
  --domain-filter software \
  --output-dir /path/to/new_audit/software \
  --per-stratum 50 \
  --seed 42
```

新样本必须记录新目录、seed、dataset hash 和生成时间，不能与当前 v3 审计结果混合。

## 19. 最短执行清单

数据管理员：

1. 冻结当前三个 `annotation_template.csv` 和 manifest；
2. 严格隔离 `audit_key.private.jsonl`；
3. 为每个域创建 A/B 独立副本；
4. 分发模板和 rubric，不分发 gold；
5. 收回并校验两份完整 CSV；
6. 运行 `human_audit score`；
7. 冻结 pre-adjudication 指标；
8. 生成分歧表，由第三人盲裁；
9. unblind 并确定 keep/modify/exclude；
10. 在论文中按域报告全部 n、rate、raw agreement、κ 和修改/排除数量。

标注员：

1. 只看 query、choices 和 evidence packet；
2. 独立选择最有证据支持的 choice；
3. 填写九个质量字段；
4. 对失败、空 choice 和修改建议写明理由；
5. 不改身份列、不删行、不看 gold、不与另一位标注员讨论。
