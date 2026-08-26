# HABIT-Bench ICLR 2027 补充实验方案

本文把 `docs/supplementary_exp/HABIT-Bench_ICLR27_experiment_analysis.md` 中可以可靠落地的建议，
转换为当前仓库的可执行实验。原则是：

1. 不修改现有 `eval/core/scoring.py`、retrieval scorer、主指标定义和正式运行入口。
2. 已跑出的 `metrics.json`、`retrieval_metrics.json` 仍是论文主结果的唯一来源。
3. 所有新增统计写入单独的 `supplementary_*` 文件，不与主结果混写。
4. 当前标注不足以支持的指标明确返回 `unavailable`，不使用近似标签制造结论。

新增代码全部位于：

```text
eval/supplementary/
├── oracle_controls.py   # Oracle Evidence / Oracle Habit State 诊断上界
├── merge_oracle.py      # Oracle 用户分片的严格合并与重新计分
├── analyze.py           # 单方法用户统计、难度切片、组件指标、效率
├── compare.py           # 多方法配对显著性检验与 Holm 校正
└── human_audit.py       # 分层盲审采样与一致性统计
scripts/run_supplementary_analysis.py  # 完整 suite 的批量离线分析
```

## 1. 建议加入论文正文的实验

### 1.1 Explicit-to-Habit transfer gap

问题：一个方法能否检索显式历史事实，并不等价于它能从重复弱证据中归纳潜在习惯。

Food v5 同时具有：

- `explicit_retrieval`：显式记忆任务；
- `direct_use`、`boundary`、`exception`：潜在习惯任务。

`analyze.py` 额外报告：

```text
explicit_accuracy
latent_habit_accuracy
explicit_minus_latent_gap
paired_user_gap + user-cluster bootstrap CI
```

其中 transfer gap 定义为：

```text
Acc(explicit_retrieval) - Acc(latent habit probes)
```

这个统计只是对现有 Accuracy 的重新分组，不改变任何 probe 的正确性或主指标。建议论文
同时给出两类绝对准确率，不能只给 gap；否则负 gap 可能被误读为 latent task 更容易。

Finance/Software v1.4 与 Travel v16 没有同机制的 `explicit_retrieval` 对照，因此这些数据上会返回
`unavailable`，不跨不同数据生成机制强行计算 transfer gap。

### 1.2 四级诊断对照

建议对每个域报告：

```text
No Memory
Full Memory / recency-truncated long context
Oracle Evidence
Oracle Habit State
```

四者的含义：

| 对照 | 提供给同一 answer model 的信息 | 诊断作用 |
| --- | --- | --- |
| No Memory | 无历史 | base-model prior、题面捷径 |
| Full Memory | 容量允许的全部历史，否则最近完整 session suffix | 无检索器的长上下文能力 |
| Oracle Evidence | 私有标注的决定性证据和必要 temporal context 原文 | 完美检索后，习惯归纳与 answer head 的上界 |
| Oracle Habit State | 私有受控 habit graph 或当前 policy variants | 完美习惯状态后，answer head/题目歧义的上界 |

Oracle 控制不会把 `gold_choice_id` 写入 prompt：

- Food/Travel 的单习惯题使用受控 habit graph 的 default、boundary、exception rule，并指出当前适用 rule；
- Finance/Software 及 Travel 的组合题使用 gold choice 对应的 target-habit policy variants 与
  `gold_action_text`；
- Oracle Evidence 对 Finance/Software 排除 `nonbinding_evidence_session_ids`；
- Oracle 是使用私有标签的诊断上界，不得作为可部署 memory 方法加入方法排名。

推荐用下面三个差值定位瓶颈：

```text
retrieval/state gap = Oracle Evidence - tested method
induction gap       = Oracle Habit State - Oracle Evidence
answer-head gap     = 1.0 - Oracle Habit State
```

这些差值必须与原始 Accuracy 和置信区间一起报告；不要把它们解释为彼此完全独立的因果
分解。

单卡运行示例：

```bash
conda run -n habitbenchmark python -m eval.supplementary.oracle_controls \
  --dataset-dir domain/food/food_habit_lifelines_stress_v5 \
  --output-dir results/supplementary_oracle/food/oracle_evidence \
  --mode oracle_evidence \
  --base-model Qwen3-8B \
  --base-model-path /plm-shared/zhangjunming/Workspace/models/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1

conda run -n habitbenchmark python -m eval.supplementary.oracle_controls \
  --dataset-dir \
    domain/finance-software/habit_bench_multidogo_finance_software_release_gated_v1_4 \
  --domain-filter finance \
  --output-dir results/supplementary_oracle/finance/oracle_habit_state \
  --mode oracle_habit_state \
  --base-url http://127.0.0.1:8000/v1
```

只检查上下文而不启动 answer model：

```bash
conda run -n habitbenchmark python -m eval.supplementary.oracle_controls \
  --dataset-dir domain/food/food_habit_lifelines_stress_v5 \
  --output-dir /plm-shared/zhangjunming/tmp/oracle_food_smoke \
  --mode oracle_evidence --max-users 1 --max-probes 8 --prepare-only
```

Oracle 支持与正式方法一致的 `--user-shard-index/--user-shard-count`。每个成功 shard
同时写出标准 `run_manifest.json`，并由独立的 Oracle merger 合并：

```bash
conda run -n habitbenchmark python -m eval.supplementary.merge_oracle \
  --dataset-dir domain/food/food_habit_lifelines_stress_v5 \
  --shard-root results/supplementary_oracle/food/oracle_evidence \
  --output-dir results/supplementary_oracle/food/oracle_evidence/merged \
  --mode oracle_evidence --expected-shards 8
```

必须使用专用 Oracle merger：`Oracle Habit State` 没有执行 retrieval，专用 merger
会复用原 scorer 的 no-retrieval 语义并仅重标 method name，避免制造全零 Recall@5。
这不需要修改 `eval/core/retrieval_scoring.py`。

### 1.3 用户级统计与配对显著性

主表继续报告现有 micro Accuracy。补充报告：

- user-macro Accuracy；
- 以 user 为 cluster 的 95% percentile bootstrap CI；
- 每个用户准确率的分布；
- 方法间配对 user-cluster bootstrap 差值与 CI；
- probe 级 exact McNemar test，作为次要配对检验；
- 对同一张方法比较表中的全部 pair 执行 Holm family-wise correction。

不能把同一用户的多个 probes 当作独立样本做普通 bootstrap，因为它会低估方差。

单方法旁路分析：

```bash
conda run -n habitbenchmark python -m eval.supplementary.analyze \
  --dataset-dir domain/food/food_habit_lifelines_stress_v5 \
  --scored-predictions \
    results/<suite>/food/mem0/merged/scored_predictions.jsonl \
  --artifact-root results/<suite>/food/mem0/merged \
  --output-dir results/<suite>/food/mem0/merged/supplementary \
  --bootstrap-samples 10000 --seed 42
```

Finance 与 Software 必须分别加 domain filter：

```bash
conda run -n habitbenchmark python -m eval.supplementary.analyze \
  --dataset-dir \
    domain/finance-software/habit_bench_multidogo_finance_software_release_gated_v1_4 \
  --domain-filter software \
  --scored-predictions \
    results/<suite>/software/mem0/merged/scored_predictions.jsonl \
  --output-dir results/<suite>/software/mem0/merged/supplementary
```

方法间比较：

```bash
conda run -n habitbenchmark python -m eval.supplementary.compare \
  --dataset-dir domain/food/food_habit_lifelines_stress_v5 \
  --output-dir results/<suite>/food/supplementary_comparison \
  --run mem0=results/<suite>/food/mem0/merged/scored_predictions.jsonl \
  --run amem=results/<suite>/food/amem/merged/scored_predictions.jsonl \
  --run memos=results/<suite>/food/memos/merged/scored_predictions.jsonl \
  --bootstrap-samples 10000 --seed 42
```

比较工具要求所有方法 probe coverage 完全相同；这样可以防止某个方法因漏掉难题而得到
虚高结果。

正式 suite 全部完成后，可一次处理四个域和所有已合并方法：

```bash
conda run -n habitbenchmark python scripts/run_supplementary_analysis.py \
  --suite-root results/<suite> \
  --output-root results/<suite>_supplementary \
  --domains food,finance,software,travel \
  --bootstrap-samples 10000 --seed 42
```

该批处理器同时兼容当前 `domain/method/merged` 和旧实验的
`method/domain/method/merged` 布局。指定 `--methods mem0,amem,...` 时，缺少任一
完整 merged 结果都会报错，不会静默跳过。

### 1.4 Finance/Software policy-component accuracy

Finance/Software v1.4 的一个 choice 常由 2–3 个目标习惯的 policy variant 组合而成。
只看 exact-choice Accuracy 无法区分“全部习惯都错”和“只错一个组件”。

基于已有私有 `choice_policy_signatures`，`analyze.py` 额外报告：

```text
policy_component_accuracy
mean_wrong_components
zero/one/multiple_wrong_component_rate
surface_decoy_component_selection_rate
per_habit component accuracy
```

exact-choice Accuracy 保持不变，仍是更严格的主指标。组件准确率只用于说明失败结构：

- exact-choice 低、component accuracy 高：多习惯组合是主要难点；
- component accuracy 也低：单个潜在习惯状态本身未被正确恢复；
- surface-decoy rate 高：方法偏向表面相似而非用户采纳/provenance。

Food v5 没有精确的 per-choice action taxonomy。不能使用字符串相似度把 Food choices
反推为 default/boundary/exception/wrong action；该数据的组件指标会返回
`unavailable`。

### 1.5 现有数据上的难度分层

`supplementary_metrics_by_slice.csv` 对现有结果提供以下观察性切片：

| 切片 | 定义 |
| --- | --- |
| `support_count` | 决定性证据 session 数量 |
| `history_length_bin` | probe cutoff 前可见 session 数 |
| `distractor_ratio_bin` | 非相关可见 session 数 / 决定性证据数 |
| `evidence_position_bin` | 决定性证据在可见 lifeline 中的平均相对位置 |
| `evidence_bands` | 数据集已有 early/middle/late 证据分布标签 |
| `probe_type` | 原生能力类型 |
| `domain` | food/finance/software |

每个 slice 同时给出：

```text
micro accuracy
user-macro accuracy
Evidence Recall@5
Full Evidence@5
Complete Chain@5
Component Complete Coverage@5
Joint Answer-Evidence Hit@5
Clean Grounded Answer@5
Nonbinding Intrusion Rate@5
```

这些是观察性分层，不应写成支持数、距离或干扰项的因果 intervention curve，因为不同
probe 的其他难度因素没有保持一致。要得到论文分析建议中的严格曲线，仍需生成同一
latent habit 的 matched counterfactual probes，见第 3 节。

### 1.6 效率与 Pareto frontier

`supplementary_metrics.json` 从现有逐题记录汇总：

- answer latency mean/P50/P95；
- retrieval latency mean/P50/P95；
- prompt、completion、total token 总量与均值；
- memory tokens；
- ingestion session 数；
- 可选的 run artifact 总字节数与每用户字节数。

建议画两张图：

```text
x = wall-clock / GPU-hours，y = Accuracy
x = mean retrieval latency，y = Evidence Recall@5
```

只把没有被其他点同时在“更准确且更便宜”支配的方法连成 Pareto frontier。磁盘字节
统计严格定义为 `--artifact-root` 下普通文件的递归字节数；不同方法若把缓存放在目录
外，不可直接比较，需先统一 artifact boundary。

### 1.7 Answer–retrieval 错误耦合

`answer_retrieval_error_analysis` 把可评估的 probe 分成：

```text
answer correct + evidence hit
answer correct + no evidence hit
answer wrong   + evidence hit
answer wrong   + no evidence hit
```

并对 complete evidence 重复同一交叉统计，同时报告 wrong answer 与 nonbinding
intrusion 的共现率以及非法 attribution 比例。它可用来发现：

- `correct + no hit`：可能依赖题面 shortcut 或模型 prior；
- `wrong + hit`：检索到了部分证据，但习惯状态归纳或 answer use 失败；
- `wrong + no hit`：首先存在 retrieval failure；
- `correct + complete evidence`：最严格的 grounded success。

这些是描述性共现，不是模块错误的因果归因。论文中不得把所有 `wrong + hit` 都断言为
answer-head failure。

### 1.8 分层双人盲审

自动生成 domain × probe type 的均匀分层样本：

```bash
conda run -n habitbenchmark python -m eval.supplementary.human_audit prepare \
  --dataset-dir domain/food/food_habit_lifelines_stress_v5 \
  --output-dir results/human_audit/food \
  --per-stratum 50 --seed 42
```

输出：

```text
annotation_template.csv      # 只把这个文件交给标注员
audit_key.private.jsonl      # gold choice；必须与标注员隔离
audit_manifest.json          # sample、seed、strata
```

每位标注员填写最佳 choice，并用 1/0 标注：

```text
answerable_from_evidence
evidence_sufficient
scope_condition_correct
boundary_exception_correct
choices_balanced
language_natural
source_grounded
privacy_safe
needs_modification
```

至少两名标注员独立完成后：

```bash
conda run -n habitbenchmark python -m eval.supplementary.human_audit score \
  --audit-key results/human_audit/food/audit_key.private.jsonl \
  --annotation annotator_a=results/human_audit/food/a.csv \
  --annotation annotator_b=results/human_audit/food/b.csv \
  --output-dir results/human_audit/food/scored
```

输出 gold-choice agreement、每项通过率、raw agreement 和 Cohen's kappa。论文应说明
分歧由第三人 adjudicate；本工具不自动用多数票掩盖分歧。

## 2. 补充输出结构

### 2.1 单方法分析

```text
<output-dir>/
├── supplementary_metrics.json
├── supplementary_metrics_by_slice.csv
├── supplementary_metrics_by_user.csv
└── supplementary_probe_diagnostics.jsonl
```

`supplementary_metrics.json` 的顶层结构：

```text
accuracy
explicit_to_habit_transfer
policy_components
calibration
false_personalization
answer_retrieval_error_analysis
efficiency
metric_definitions
```

逐 probe diagnostic 含私有 evidence difficulty 元数据，应与 `scored_predictions.jsonl`
采用相同访问控制，不发布到公开 test set。

### 2.2 方法比较

```text
<output-dir>/
├── supplementary_comparison.json
├── supplementary_comparison_methods.csv
└── supplementary_comparison_pairs.csv
```

### 2.3 Oracle

Oracle 保留标准输出文件，便于复用现有 scorer 和专用 merger：

```text
<output-dir>/
├── supplementary_manifest.json
├── run_manifest.json
├── memory_contexts.jsonl
├── predictions.jsonl
├── scored_predictions.jsonl
├── metrics.json
├── metrics_by_group.csv
├── retrieval_metrics.json
└── retrieval_metrics_by_group.csv
```

这里的 `metrics.json` 仍由原生 scorer 生成，指标定义没有改变；manifest 会明确写出
`diagnostic_upper_bound` 和 private-label warning。

## 3. 当前不能从现有数据可靠得到的指标

### 3.1 概率校准

当前 answerer 只保存 hard `choice_id`，没有完整 choice probability。不能由是否答对
反推置信度。`analyze.py` 仅在每题存在：

```json
{
  "answer": {
    "choice_probabilities": {
      "A": 0.1,
      "B": 0.7,
      "C": 0.1,
      "D": 0.1
    }
  }
}
```

时计算 multiclass Brier、NLL、ECE 和 AURC；缺失时明确返回 `unavailable`。正式做
probabilistic habit calibration 还需要同一情境下的随机行为概率或多次采样标签，
不能把单次 deterministic gold 当成真实用户概率。

### 3.2 真正的 false-personalization cost

现有 Food/Finance/Software 都没有 matched no-habit user，也没有完整 option-action
taxonomy。`1 - boundary accuracy` 只是边界失败率，不等于严格的 false-personalization
rate。

未来数据应为每题增加：

```text
personalization_applicable: bool
choice_action_taxonomy:
  <choice_id>: generic | applicable_habit | stale_habit |
               boundary_violation | exception_violation |
               unsupported_personalization
```

只有该 contract 对全部 probe 完整时，sidecar 才会报告：

```text
false_personalization_rate
missed_personalization_rate
stale_habit_selection_rate
Utility(lambda) = Accuracy - lambda * false_personalization_rate
```

建议至少报告 `lambda ∈ {0, 0.5, 1, 2, 5}` 的敏感性，不选择单个有利 lambda。

### 3.3 因果 stress curves

要做 Support Count、Evidence Distance、Distractor Ratio 的因果曲线，需要固定
user、habit、query 和 choices，只改变一个变量。建议新建独立 stress split：

```text
support count: 1 / 2 / 3 / 5 / 8
evidence band: early / middle / recent
distractor ratio: 0x / 10x / 50x / 100x
```

同一 matched family 使用相同 `counterfactual_group_id`，分析时以 group 为 cluster。
这属于数据扩展，不应在现有 Food v5 或 Finance/Software v1.4 标签上做代码近似。

### 3.4 外部显式记忆基准

“在显式记忆 benchmark 上强、在 HABIT 上弱”最好用同一方法、同一 LLM/embedding
设置运行外部 benchmark。仓库内保留的 MedMemoryBench 快照已经包含 LoCoMo dataset
loader/evaluator、long-context、BM25-RAG 和 embedding-RAG 配置，但当前主仓库没有把
LoCoMo 分数转换成 HABIT choice Accuracy。外部结果应保留 LoCoMo 原生指标，并通过
method/config/checkpoint hash 对齐，不应把两个 benchmark 的不同指标直接相减。

## 4. 控制变量与报告规范

### 4.1 窗口预算

`full_memory` 已支持 8k/16k/32k/40k/64k/128k 档位。补充实验建议至少跑：

```text
No Memory
Full Memory 8k
Full Memory 16k
Full Memory 32k
Full Memory 40k（当前 Qwen3-8B 上限内）
```

同一图中必须报告实际 `history_token_budget`、truncated probe 比例和平均保留 session
数。超过 checkpoint 实际容量的 64k/128k 不能仅通过修改环境变量宣称支持。

Memory methods 的 top-k 不等价于 token budget。严格 budget-matched 实验应在方法原生
retrieval 后统一做相同 tokenizer token cap，并保留完整 retrieval item 边界；这会改变
输入协议，建议作为独立 ablation，而不是覆盖当前主结果。

### 4.2 Answer model、seed 与重复实验

主比较保持同一 Qwen3-8B、temperature 0、prompt 和最大输入。为了排除 answer head
瓶颈，建议只对以下紧凑集合增加一个更强 answer model：

```text
No Memory + Full Memory + 最强两个 memory methods
+ Oracle Evidence + Oracle Habit State
```

如果仍使用 temperature 0，重复 seed 通常不会提供真正的随机方差。需要 seed 实验时，
应显式启用非零 sampling 并保存 temperature/top-p/seed；否则报告用户 cluster CI 即可。

### 4.3 推荐论文表格

主表保持现有指标：

```text
Accuracy
Evidence Recall@5 / Context Evidence Recall
Full Evidence / Complete Chain
Clean Grounded Answer
Nonbinding Intrusion
原生 capability panels
```

新增三张补充表：

1. `No Memory / Full Memory / methods / Oracle Evidence / Oracle Habit State`；
2. micro、user-macro、cluster CI、paired delta、Holm-adjusted p；
3. policy-component、stress slices、latency/tokens/storage。

新增两张图：

1. Explicit accuracy 与 latent-habit accuracy 的 paired user gap；
2. Accuracy–cost 与 Recall@5–latency Pareto 图。

人工审计单独报告 sample size、strata、gold agreement、每项通过率、raw agreement、
Cohen's kappa 和 adjudication 后修改/排除数量。

## 5. 推荐执行顺序

1. 固定当前 Food v5、Finance/Software v1.4 和 Travel v16 四域主实验口径。
2. 对每个 merged `scored_predictions.jsonl` 运行 `analyze.py`。
3. 每个域用 `compare.py` 生成统一配对统计。
4. 运行两个 Oracle control，先做 `--prepare-only` smoke test。
5. 做双人分层盲审并冻结排除规则。
6. 根据 Oracle gap 决定是否值得增加 stronger answer-head ablation。
7. 最后再生成 matched no-habit、概率标签和 causal stress split；它们应使用新 dataset
   version，不能静默修改当前 Food v5 或 Finance/Software v1.4。
