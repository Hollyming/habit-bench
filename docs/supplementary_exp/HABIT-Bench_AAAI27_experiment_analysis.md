# HABIT-Bench：AAAI-27 实验充分性审计与补强路线

更新日期：2026-07-27
审计对象：`habit-bench` 当前代码、数据包、v1 完整结果、正在运行的三域 v2，以及 `AuthorKit27` 论文初稿。

## 1. 结论先行

当前实现已经是一套有实质内容的 benchmark harness，而不是只有 Accuracy 的早期原型。它具备：

- 固定 answer model 的端到端 choice accuracy；
- Food 上的 direct use、explicit retrieval、boundary、exception 和 unseen paraphrase；
- Finance/Software 上的 decisive evidence、temporal context、non-binding evidence、required component groups 和 decision units；
- Evidence Recall/Precision/MRR/nDCG@5、完整证据链、decoy intrusion、clean grounded answer；
- `no_memory` 与容量感知的 `full_memory` 控制；
- 严格的时间 cutoff、用户隔离、gold leakage 检查、运行配置和 provenance 记录。

但是，**这些指标还不足以单独支撑一篇成熟顶会论文的核心主张**：

> “在显式用户记忆 benchmark 上表现良好的 memory agent，会在需要从重复弱证据推断潜在、概率性、情境条件化习惯时失败。”

原因不是指标数量少，而是现有实验尚未把这句话中的几个因果环节隔离出来：

1. 没有在同一批方法、同一 answer head 和尽量一致的预算下，直接测量“显式记忆成绩高、HABIT 成绩低”的跨 benchmark 落差；
2. 当前数据主要把隐藏 policy 离散化成确定性四选一答案，尚未真正测量“概率性”与置信度校准；
3. `boundary accuracy` 的错误并不自动等价于 false personalization，缺少 option-level error taxonomy 和真正的 no-habit negative controls；
4. Finance/Software 有 temporal/as-of 测试，但没有实现论文所声称的 drift adaptation lag；
5. 缺少 Oracle Evidence 上限，因此低分可能同时来自检索失败、answer model 推理失败或题目歧义；
6. Food 与 Finance/Software 数据卡目前都明确写着 human audit 尚未完成；Finance/Software 的人工 spot check 只有每类一题；
7. v1 把 Finance 与 Software 合并，三域 v2 尚未完成，Travel 尚无可评测数据。

因此，当前状态适合写成“**一个设计完整、已有强诊断信号的 benchmark candidate**”，但还不能把所有预期能力都当作已验证结论。成熟版本至少要完成下文 P0 实验。

## 2. 当前代码到底测到了什么

### 2.1 能力覆盖审计

| 论文能力主张 | 当前实现 | 判断 | 必须补的部分 |
| --- | --- | --- | --- |
| Habit induction | Food `direct_use`；Finance/Software 多组件 decisive evidence | 部分充分 | 证据条数基本固定，不能证明系统是在聚合重复弱证据；需 support-count intervention |
| Boundary calibration | Food `boundary`；Finance/Software `scope_temporal_pair` | 部分充分 | `1 - boundary accuracy` 只是 boundary error，不一定全是 false personalization |
| Drift handling | dual/triple as-of、replacement、temporal context | 部分充分 | 缺少逐步加入新证据后的 adaptation curve/lag；缺少 gradual 与 seasonal drift |
| Exception retention | Food `exception`；non-binding evidence/intrusion | 较强诊断 | 仍需测 exception 之后默认习惯是否被污染，以及 exception 与 revision 的混淆矩阵 |
| False-personalization cost | `false_personalization_cost = 1 - boundary accuracy` | 不充分 | 缺 no-habit、insufficient-evidence、irrelevant/baiting negative controls；缺真实 utility/cost 权重 |
| Probabilistic habit | 未实现概率输出或概率生成过程 | 缺失 | 增加 stochastic evidence、confidence、Brier/ECE/log loss、selective risk |
| Ask vs. act | Food manifest 显示 `ask_act` 被删除 | 缺失 | 恢复高质量 insufficient/conflicting evidence probes |
| Evidence provenance | decisive/temporal/non-binding 拆分，来源 ID 校验 | 当前强项 | 增加 Oracle 与无 provenance 方法的明确分层报告 |
| Ultra-long robustness | Food 120 sessions/user；Finance/Software 540 sessions/user | 单点充分、曲线不足 | 需要 history length、evidence distance、distractor ratio 的干预曲线 |
| Cross-domain interference | Finance 与 Software 可独立过滤 | 尚未真正测到 | 同一 pseudo-user 的 mixed-domain lifeline 与 domain-pure 对照 |
| Efficiency | token、wall-clock、存储/运行信息已有 | 可用但未成结论 | 统一预算与相同硬件上的 accuracy-cost Pareto |

### 2.2 v1 结果给出的真实信号

下列数字来自完整的 legacy v1 run；该 run 使用 Food 与合并的 Finance–Software，不应与正在运行的三域 v2 混报。

- Food：
  - `no_memory` 为 29.7%，40k recency-truncated `full_memory` 为 42.7%；
  - 最好的 memory method 是 Letta，47.1%；
  - 但系统出现明显的 apply-versus-withhold trade-off：例如 LightMem 的 direct-use 为 52.1%、boundary 为 35.2%；Letta 的 direct-use 为 35.5%、boundary 为 57.1%。
- 合并 Finance–Software：
  - `no_memory` 为 24.6%，接近四选一随机水平；
  - memory methods 只有 23.0%–24.3%，没有超过 `no_memory`；
  - 所有 memory methods 的 decisive Evidence Recall@5 只有 0%–7.7%，complete-chain rate 全为 0；
  - 40k `full_memory` 为 20.8%，其窗口内 decisive evidence recall 只有 23.8%，说明它不是“完整历史上限”，而是明确受限的 recency baseline。

这些结果已经支持两个有价值但较谨慎的结论：

1. HABIT-Bench 确实暴露了普通总体 Accuracy 看不出的“应用习惯”和“克制个性化”之间的结构性权衡；
2. 多组件、时序和 provenance-aware 的 Finance–Software 任务对当前检索式 memory 极难。

但它们尚不能直接证明“这些方法在现有显式 benchmark 上很强，所以出现了能力反转”，也不能排除固定 Qwen3-8B answer head 对复杂选项的推理瓶颈。

## 3. 与近邻 benchmark 相比，HABIT-Bench 的正确定位

当前相关工作已经覆盖了不少曾经可以单独声称为 novelty 的要素，因此论文应避免把“隐式偏好”“动态 persona”或“过度个性化”本身写成首次提出。

| 近邻工作 | 已覆盖内容 | HABIT-Bench 应强调的剩余差异 |
| --- | --- | --- |
| [LongMemEval](https://arxiv.org/abs/2410.10813) | extraction、multi-session reasoning、temporal reasoning、knowledge update、abstention；500 个精心构造问题 | 目标不是事实/事件答案，而是从重复弱证据形成 scoped behavioral policy |
| [PersonaMem-v2](https://arxiv.org/abs/2512.06688) | 大规模 implicit persona、动态偏好、128k context、MCQ/open-ended、agentic memory | 必须强调 matched boundary、exception、non-binding provenance 和“何时不应用” |
| [PERMA](https://arxiv.org/abs/2603.23231) | event-driven preference、temporal probing、跨域干扰、MCQ 与 interactive evaluation | 不能只说偏好随事件形成；要展示可控弱证据聚合、局部例外与 policy scope |
| [AlpsBench](https://arxiv.org/abs/2603.26680) | 2,500 条真实 WildChat 长期序列；extraction/update/retrieval/utilization 全生命周期；human-verified memory | HABIT 的贡献应是 behavioral-rule decision boundary，不是“使用真实日志”本身 |
| [OP-Bench](https://arxiv.org/abs/2601.13722) | 1,700 个双人复核样本；irrelevance、repetition、sycophancy 三种 over-personalization | HABIT 的 false personalization 是“相关但越界的习惯 policy 被错误执行”，需与 OP-Bench 的话题/社会性过度个性化区分 |
| [LoCoMo-Plus](https://arxiv.org/abs/2602.10715) | cue-trigger semantic disconnect 与 latent constraint consistency | HABIT 必须证明 repeated/probabilistic evidence、boundary/exception/drift 的组合难度 |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257) | incremental interaction、retrieval、test-time learning、long-range understanding、selective forgetting | HABIT 聚焦长期用户行为 policy，而不是通用 memory competence |
| [MemoryArena](https://arxiv.org/abs/2602.16313) | 多 session 中记忆与行动闭环，且已有 preference-constrained planning | HABIT 的优势是可控 ground-truth habit graph 与细粒度归因，不是“memory 影响行动”本身 |
| [LongMemEval-V2](https://arxiv.org/abs/2605.12493) | 最长 500 条 trajectory、115M tokens、evidence gathering 与 latency Pareto | 当前 90k 左右历史不宜无修饰地称为 ultra-long；应报告确切长度并用 longitudinal/long-horizon |

建议把论文 novelty 压缩成一句更经得起审稿的表述：

> HABIT-Bench isolates whether a memory agent can infer and safely apply a context-scoped behavioral default from repeated weak evidence, while preserving local exceptions, temporal validity, and non-binding provenance.

## 4. P0：决定论文主张能否成立的实验

### P0.1 同一系统的 Explicit-to-Habit Transfer Gap

这是核心假设最缺的一张表。

对相同方法、相同 answer head、相同检索预算，至少运行一个成熟显式 benchmark（优先 LongMemEval；其次 LoCoMo）和 HABIT-Bench。代码中已经 vendored MedMemoryBench/LoCoMo 适配，可优先复用。

主表应报告：

| Method | Explicit benchmark | Matched explicit control | HABIT direct | HABIT boundary | HABIT drift/exception | Habit gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| … | … | … | … | … | … | … |

其中 `Habit gap` 不能直接用两个不同量纲的分数相减。更可靠的是：

- 在 HABIT lifeline 内创建 matched pair：
  - Explicit condition：用户明确说出一次稳定规则；
  - Latent condition：相同规则只由 1/3/5/8 条弱证据表达；
- 当前 query、choices、cutoff、distractor history 和 gold action 保持匹配；
- 用 paired accuracy difference、paired bootstrap CI 和 McNemar test 评估。

跨 benchmark 结果用于 external validity；同 lifeline 的 matched intervention 用于因果隔离。

### P0.2 Oracle 分解：先证明题目可答

每个 probe 至少增加四个诊断轨：

1. `No Memory`：只有 query + choices；
2. `Recency Window`：与 memory method 相同 token budget 的最近 sessions；
3. `Oracle Evidence`：只给 decisive evidence + 必需 temporal context；
4. `Oracle Habit State`：直接给结构化的 active habit（scope、action、exception、version），用于 answer-head 上限。

建议通过条件：

- Oracle Habit State ≥ 90%；
- Oracle Evidence ≥ 80%；
- No Memory 接近 chance，且 question-only/choice-only trained classifier 不显著高于 chance；
- 若 Oracle 仍低，应先修题目或换更强 answer head，不应把低分归因于 memory。

当前 `full_memory` 在所有 legacy probes 上都发生 40k 截断，不能承担 Oracle 的角色。论文中应写成 `40k Full-History (recency-truncated)`。

### P0.3 真正测量“概率性”和 calibration

目前 hidden habit graph 更接近确定性 rule。若摘要继续使用 “probabilistic habit”，至少加入：

- 支持比例：100/0、80/20、60/40、50/50；
- evidence reliability：用户亲自选择/确认、助手建议后采纳、助手单方面建议、一次性例外；
- 输出 `choice_id` 同时输出 `confidence`，或直接要求四个选项概率；
- 指标：Brier score、negative log-likelihood、ECE、risk–coverage/AURC；
- insufficient/conflicting evidence 时正确答案为 ask/abstain；
- 比较过早个性化成本与漏掉稳定习惯的成本。

如果无法在本轮完成，应把主张降为 “latent, context-conditioned default inferred from repeated weak evidence”，暂时删去 “probabilistic”。

### P0.4 把 false personalization 从 proxy 变成可解释错误

为每个 choice 增加私有 action taxonomy，例如：

```text
correct_scoped_action
unpersonalized_safe_action
out_of_scope_habit_application
stale_policy_application
exception_overgeneralization
assistant_suggestion_as_user_habit
wrong_habit_component
```

据此报告：

- True Personalization Rate：in-scope 时应用正确习惯；
- False Personalization Rate：out-of-scope/no-habit 时错误应用任一习惯；
- Stale Personalization Rate；
- Exception Overwrite/Contamination Rate；
- Utility：`benefit(correct personalization) - λ * cost(false personalization)`，至少给 λ 的敏感性曲线。

Food 现有 `1 - boundary accuracy` 应改名为 `boundary error rate`，除非已验证所有错误选项都属于个性化越界。

### P0.5 支持强度、距离和噪声的干预曲线

核心机制是“重复弱证据”，必须通过可控 intervention 展示，而不能只有一个固定难度点。

建议使用非完全析因、平衡抽样设计：

- support count：1、3、5、8；
- evidence distance：history 的前 25%、中间 50%、最后 25%；
- distractor ratio：1×、10×、50×；
- lexical relation：direct、paraphrase、cue-trigger disconnect；
- scope：in-scope、near-boundary、no-habit；
- temporal state：stable、local exception、abrupt revision、gradual drift。

主要图：

- x 轴 support count，y 轴 calibrated habit utility；
- 每条线代表 memory method；
- 分面展示短/长 distance 或低/高 distractor；
- 同时画 Oracle 与 no-memory。

这张图能直接回答“方法是否积累证据”，而不仅是“是否碰巧检索到一条相关 session”。

### P0.6 人工质量验证与 shortcut audit

当前状态：

- Food：数据卡明确为 `auto_validated_pending_human_review`；
- Finance/Software：数据包状态为 `auto_validated_pending_target_model_baseline_and_human_audit`；
- Finance/Software 的 stratified manual spot check 只有 7 题。

最低要求：

- 每个 domain × probe type 至少随机抽样 50 题；总量建议 400–600；
- 每题两名独立 annotator，分歧由第三人 adjudicate；
- 报告 raw agreement、Cohen’s κ/ Krippendorff’s α、接受率和修改率；
- 维度：gold 唯一性、evidence sufficiency、scope 正确性、exception/revision 区分、choice 等难度、自然度、source grounding、隐私；
- 增加 memory-necessity 双盲测试：无历史时不可确定，给 Oracle evidence 后唯一可答；
- 训练型 shortcut audit：用 train/dev 上的 question-only、choices-only、长度/词法特征 classifier 测 held-out test，而不只用启发式规则。

Food 的 `no_memory=29.7%` 已高于随机 25%，需要专门解释并按模板/用户/bootstrap 给置信区间。

### P0.7 公平预算、多个 answer head 与统计推断

当前 memory context 使用量从约 70 tokens 到 3,300 tokens，不是严格预算匹配。正式实验应同时报告：

- Native setting：保留方法推荐配置；
- Budget-matched setting：统一 512/2k/8k retrieved-token budget，按方法排序后截断；
- 相同 embedding 的 controlled comparison 与方法原生 embedding 的 native comparison；
- Qwen3-8B 作为主 answer head；
- 至少一个明显更强的 answer head 在全量或预注册分层子集上复核；
- 对有随机写入/压缩的方法跑 3 seeds；确定性方法应说明并验证 bit-level 或 prediction-level 稳定性。

统计上不要只用逐 probe Wilson interval，因为同一 user/habit 下的 probes 相关：

- 主报告 user-macro accuracy；
- Finance/Software 同时报 decision-unit macro；
- user-level cluster bootstrap 95% CI；
- 方法比较使用 paired cluster bootstrap 或 permutation test；
- 多方法比较进行 Holm correction；
- 报告 effect size，而不只报 p-value。

Software 当前只有 18 个 pseudo-users，cluster-level CI 会很宽。建议最终至少 50 users/domain，或明确把 Software 作为较小诊断域。

## 5. P1：显著提升论文成熟度的实验

### P1.1 简单但有解释力的 Habit-State baseline

纯 benchmark 论文可以成立，但在当前近邻工作密集的情况下，一个透明 baseline 会明显增强贡献。

建议实现结构化状态：

```text
condition/scope
default action
support and counterevidence counts
source reliability
local exceptions
active time interval/version
confidence
source session ids
```

用 Beta-Bernoulli/Dirichlet 或轻量 LLM extractor + deterministic updater 维护。关键不是追求 SOTA，而是通过 ablation 说明哪些字段必要：

- 去掉 scope；
- 去掉 negative/non-binding evidence；
- 去掉 exception；
- 去掉 timestamps/version；
- 去掉 confidence/abstention；
- 去掉 provenance。

若这个简单 baseline 能提高 boundary/drift/exception，同时降低 false personalization，就能证明 benchmark 指标对目标机制敏感。

### P1.2 Mixed-domain interference

当前 Finance 与 Software 是物理同包、逻辑分域，但这不等于跨域干扰。

构建同一 pseudo-user 的：

- domain-pure history；
- mixed Food/Finance/Software/Travel history；
- surface-similar cross-domain decoys；
- 同一习惯在不同域中适用/不适用的 contrast。

报告 mixed-domain degradation 与 cross-domain false transfer。

### P1.3 Open-ended/interactive secondary track

四选一利于确定性评分，但可能被质疑为 option artifacts。保留 MCQ 为主指标，同时抽取 200–400 题做：

- open-ended action generation；
- 结构化 action slots 的 deterministic scoring；
- 对无法结构化的部分，用两名人工标注者或经过人工校准的 judge；
- 可选 user simulator：如果系统违反习惯，用户补充一次信息，测纠正轮数和交互负担。

这与 PERMA 的 interactive track 和 OP-Bench 的开放式输出形成可比性。

### P1.4 Efficiency Pareto

已有日志足以报告：

- write-time LLM calls/tokens；
- retrieval/answer tokens；
- p50/p95 latency；
- storage bytes/user；
- memory growth/session；
- end-to-end accuracy 与 clean grounded answer 的 Pareto。

不同方法 wall-clock 必须在同硬件、相同并发设置下比较；否则只报告 tokens/calls/storage。

### P1.5 Error analysis

在每个主方法分层抽取 50 个错误，使用 option-level taxonomy 归类：

- no induction；
- only one weak component retrieved；
- surface decoy；
- assistant suggestion treated as user evidence；
- stale policy；
- exception overwrote default；
- scope overgeneralization；
- correct retrieval but wrong arbitration；
- answer-head formatting/reasoning error。

给出 retrieval hit/miss 条件下的错误分布，这会比单纯多报几个 Recall 数字更能解释机制。

## 6. P2：后续扩展

- adversarial memory injection 与 malicious assistant suggestions；
- multilingual/idiolect robustness；
- consent/deletion 与敏感习惯；
- small real-user longitudinal pilot，用于验证 synthetic-to-real transfer；
- seasonal/cyclic drift；
- collaborator/shared-account 冲突；
- action execution environment，而不仅是选择文本动作。

## 7. 推荐的最终主实验版式

### 主表 1：数据质量与规模

四域 users、sessions、tokens、habits、probes、probe types、human acceptance、κ、source provenance。

### 主表 2：同系统的 Explicit-to-Habit Gap

同一批 memory methods 的显式 benchmark、matched explicit、HABIT user-macro、calibrated utility。

### 主图 1：Framework

真实日志/任务 grounding + controlled habit graph → pseudo-user lifeline → memory agent → fixed answer head；私有 scorer 并行评估 answer、decisive evidence、scope、drift、exception、false personalization。

### 主图 2：Capability trade-off / stress curve

优先使用 support-count × distance 曲线；如果新实验未完成，可暂时使用 legacy Food direct-use vs boundary/exception trade-off，但必须标注为 pilot。

### 主表 3：最终四域 leaderboard

Food、Finance、Software、Travel、domain macro；同时给 decision-unit/user macro CI。

### 主表 4：机制分解

Evidence Recall、complete chain、non-binding intrusion、clean grounded answer、Oracle gap。

### 主图 3：Accuracy–cost Pareto

横轴 total tokens 或 latency，纵轴 calibrated habit utility。

### 附录

完整 probe taxonomy、所有预算/seeds、分域/分类型结果、human audit rubric、prompt、许可与数据卡、所有错误分析。

## 8. 可执行的“最低可信版本”清单

如果计算和标注资源有限，优先顺序如下：

1. 完成三域 v2 严格 merge；不要混入 partial shard 或 legacy combined-domain 数字；
2. 完成 Travel，或者把论文明确改成三域，不在摘要中承诺四域；
3. 增加 Oracle Evidence 与 Oracle Habit State；
4. 对 Food/Finance/Software 完成双人 human audit；
5. 加入 matched explicit-vs-latent 对照；
6. 加入 no-habit/ask-act 与 option-level false-personalization taxonomy；
7. 同一方法在 LongMemEval 或 LoCoMo 上跑一个 matched external control；
8. 用 user-level cluster bootstrap 和 paired tests 重算主表；
9. 在一个更强 answer head 上复核 Oracle 与关键方法排名；
10. 最后才扩充更多 memory methods。

在这十项中，1–7 比再增加一个新的 memory architecture 更重要。

## 9. 论文措辞边界

在新实验完成前，建议使用：

- “source-grounded, longitudinally synthetic pseudo-users”；
- “long-horizon histories of 120–540 sessions per user”；
- “legacy pilot results suggest…”；
- “capacity-limited full-history control”；
- “boundary error” 而非未经验证的全部 “false personalization”；
- “temporal/as-of resolution” 而非已经测得 “adaptation lag”；
- “three implemented domains plus Travel under construction”，或只写三域。

暂时避免：

- “ultra-long” 而不给 token 分布和与现有 million-token benchmark 的对照；
- “probabilistic habit” 而没有 probability/calibration 任务；
- “current high-scoring memory agents fail” 而没有同配置 explicit benchmark 成绩；
- “human-verified benchmark” 而数据卡仍为 pending human audit；
- 把 40k recency-truncated baseline 称为完整 full context；
- 把 v1 合并 Finance–Software 与 v2 分域结果放在同一主表中。

## 10. 总体判断

HABIT-Bench 最有价值的部分不是又一个长上下文 QA 集，而是它已经把 decisive evidence、temporal context、non-binding evidence 和 multi-component chain 分开建模。这为回答“记忆找到了什么、为何仍做错、何时不该个性化”提供了比普通 Accuracy 更好的诊断基础。

论文能否达到成熟顶会水准，取决于是否把这一诊断基础升级为**对核心主张的直接实验检验**。最关键的不是扩大方法数量，而是：

- matched explicit-to-latent intervention；
- Oracle 分解；
- probability/abstention 与真实 false-personalization；
- controlled stress curves；
- human validity；
- user-level statistics。

完成这些后，论文可以稳定地把贡献定位为一个新的能力问题；否则它更像一套有潜力但尚未闭环的数据与评测工程。
