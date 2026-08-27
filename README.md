# HABIT-Bench

HABIT-Bench 用于评估 memory agent 能否从超长用户—助手交互历史中的重复弱证据，
归纳并正确应用潜在的、弱证据驱动的、受情境约束的用户习惯。

核心假设是：在显式用户记忆基准上表现较好的方法，不一定能够处理需要跨多次弱证据
归纳的习惯，也可能在边界、异常、漂移或证据不足时产生错误个性化。

ICLR 2027 补充实验、Oracle 对照、用户级统计和人工盲审协议见
[`docs/supplementary_exp/ICLR27_supplementary_experiments.md`](docs/supplementary_exp/ICLR27_supplementary_experiments.md)。
人工审计的逐字段 rubric、双人评分和第三人裁决流程见
[`HUMAN_AUDIT_GUIDE`](docs/human_audit/HUMAN_AUDIT_GUIDE.md)，v3 审计结论见
[`HUMAN_AUDIT_RESULTS_V3`](docs/human_audit/HUMAN_AUDIT_RESULTS_V3.md)。
H 集群单节点 4/8 卡 H200 的环境准备、RJob 三类任务和恢复流程见
[`H 集群评测`](docs/h_cluster_evaluation.md)。

本仓库的正式评测不是单纯的 embedding 相似度测试。每个 memory 方法只负责构建和
检索 `memory_context`；固定的 base model 根据：

```text
memory_context + current request + response choices
```

选择一个 `choice_id`。私有 scorer 同时评估最终答案、证据检索、证据链完整性、
干扰抑制和可追溯性，避免把“依靠模型先验碰巧答对”误当成 memory 能力。

## 1. 当前实验范围

### 1.1 数据集

| 数据集别名 | 领域 | 用户 | sessions | probes | 路径 |
| --- | --- | ---: | ---: | ---: | --- |
| `food` | food | 30 | 4,500 | 1,470 | `domain/food/food_habit_lifelines_stress_v5` |
| `finance` | finance | 36 | 19,440 | 1,368 | `domain/finance-software/habit_bench_multidogo_finance_software_release_gated_v1_4` |
| `software` | software | 18 | 9,720 | 680 | `domain/finance-software/habit_bench_multidogo_finance_software_release_gated_v1_4` |
| `travel` | travel | 30 | 4,092 | 650 | `domain/travel/release_candidate_v16_postrepair_repaired_r4` |

Finance 和 Software 在 v1.4 中共用同一个物理数据包，但正式评测把它们视为两个独立
dataset alias。`shard_plan.tsv` 分别写入 `domain_filter=finance` 和
`domain_filter=software`；loader 在用户分片之前过滤 sessions、probes 和 private
keys，合并时再次核验 domain filter。旧的 `finance_software` 无过滤别名仅为兼容已有
plan 保留，不属于默认正式 suite。

Travel v16 已加入默认正式 suite，包含 30 个用户、4,092 个 sessions 和 650 个
probes。其 post-repair validation 状态为 pass；650 个 probes 都具有可加载的 oracle evidence
和 oracle habit state。四个 alias 都由同一 plan builder 固定到上述版本。

每个数据包主要包含：

```text
public/lifelines.jsonl     # 方法允许读取的用户历史
public/probes.jsonl        # 当前请求、choices、user_id、可见历史范围
private/probe_key.jsonl    # gold choice 和评测标签，仅 scorer 读取
private/...                # habit graph、evidence chain、标注会话等审计信息
reports/...                # 数据质量、泄漏、难度和一致性审计
review/...                 # 人工复核队列
source/...                 # 数据来源与构建溯源（若该数据包提供）
```

`eval/core/dataset.py` 会把不同数据包的 public lifeline 组织形式归一化为统一
session/probe contract。Gold answer、gold evidence、habit graph、persona profile 和
policy label 不会进入 memory 方法输入。

Food v5 是当前主实验的 content-constraint 习惯域，包含 210 个受控习惯：

| Probe type | 数量 | 设计目标 | 私有正向证据 |
| --- | ---: | --- | --- |
| `direct_use` | 420 | 从重复弱证据归纳习惯并应用 | 5 个 sessions |
| `explicit_retrieval` | 210 | 显式询问历史行为，测基础检索能力 | 5 个 sessions |
| `boundary` | 420 | 情境超出习惯适用范围时避免错误套用 | 1 个 boundary session |
| `exception` | 420 | 保留并应用用户明确给出的局部例外 | 1 个 exception session |

其中 630 个 probe 使用 `unseen_paraphrase`，用于区分真正的习惯归纳和表面模板匹配。
Food 的正向检索 gold 是 `private/probe_key.jsonl` 中的
`gold_evidence_session_ids`。所有 1,470 个 probe 都有有效、同用户、早于 cutoff 的
证据链接。

Finance/Software v1.4 更强调多证据链、漂移、scope 和 provenance：

| Probe type | Finance | Software | 总数 | 设计目标 |
| --- | ---: | ---: | ---: | --- |
| `dual_asof_reversal` | 255 | 129 | 384 | 两个习惯在不同 as-of 状态下的反转 |
| `triple_asof_interleaved` | 333 | 179 | 512 | 三习惯、跨时序的交错证据归纳 |
| `scope_temporal_pair` | 40 | 24 | 64 | scope 与时间边界联合校准 |
| `surface_decoy_pair` | 266 | 118 | 384 | 拒绝表面相似但不具约束力的记忆 |
| `suggestion_rejection_pair` | 163 | 93 | 256 | 区分用户采纳与 assistant 单方面建议 |
| `provenance_weighted_triple` | 91 | 37 | 128 | 三习惯证据 provenance 加权 |
| `reference_case_reconstruction` | 220 | 100 | 320 | 重建历史未完成状态和适用 policy |

该域把证据明确拆成：

```text
decision_evidence_session_ids    # 能推出最终习惯/决策的决定性证据
temporal_context_session_ids     # 单独评估的时序上下文
nonbinding_evidence_session_ids  # 局部例外或未获用户采纳的干扰证据
required_component_groups        # 每个目标习惯必须组合的弱证据组件
decision_unit_ids                # 去除重复潜在决策带来的计权偏差
```

`gold_evidence_session_ids` 是上述相关上下文的并集，包含 nonbinding 干扰项，因此绝不能
直接作为 Finance 的正向 Recall gold。正式 scorer 使用
`decision_evidence_session_ids`，并把 temporal 与 nonbinding 分别报告。

### 1.2 方法集合

正式 registry 位于 `eval/methods.json`。

第一组是当前主要基准，即 MedMemoryBench 固定源码中的七个方法：

```text
mem0, amem, memos, memrl, lightmem, letta, mirix
```

第二组是额外的官方源码适配：

```text
secom
```

第三组是 evaluator control，不是 learned memory 方法：

```text
no_memory, full_memory
```

`full_memory` 是 query-independent 在线 compact-history 主实验控制；`full_history`
保留原始 recent-session truncation 消融。H 集群默认主计划包含 `full_memory`。

第四组是没有 agentic memory lifecycle 的标准 session-retrieval baselines：

```text
recency_5, recency_10, bm25_rag, dense_rag, temporal_hybrid_rag
```

它们分别测固定近期窗口、纯 lexical BM25、统一 BGE-M3 semantic retrieval，以及
BM25/BGE-M3 RRF + as-of-aware temporal prior。详细公式、时间语义和公平性边界见
[`docs/retrieval_baselines.md`](docs/retrieval_baselines.md)。新的 H 通用默认主计划包含
这五项；已存在的持久化计划不会被自动改写。

## 2. 统一评测协议

```text
public lifeline + public probe
              │
              ▼
     chronological ingestion
              │
              ▼
 method-native memory construction
              │
              ▼
 method-native retrieval / control context
              │
              ▼
       memory_context.jsonl
              │
              ▼
 shared Qwen choice answerer
              │
              ▼
         choice_id
              │
              ▼
 private scorer → answer + retrieval + chain/provenance metrics
```

对同一用户，probe 按 `visible_history_scope.max_session_index` 排序。每次只向方法补充
当前 cutoff 新增可见的 sessions，不允许读取未来历史。每个用户拥有独立方法状态，
不同用户之间不共享 memory。

统一 answer model 默认配置：

| 项目 | 默认值 |
| --- | --- |
| checkpoint | `/plm-shared/zhangjunming/Workspace/models/Qwen3-8B` |
| served model name | `Qwen3-8B` |
| temperature | `0.0` |
| thinking | disabled |
| answer completion | 64 tokens |
| answer format | `{"choice_id": "..."}` |
| server max model length | 40,960 |
| answer input length | 由窗口档位或 evaluator 参数决定 |

memory adapter 不允许输出 `choice_id` 或 choice score。`eval/run.py` 会验证 probe
覆盖率、重复 ID、context 类型、evidence ID 类型和禁止字段。

## 3. 方法实现与配置

### 3.1 共享设置

需要向量检索的方法统一使用本地 BGE-M3：

| 项目 | 值 |
| --- | --- |
| model | `BAAI/bge-m3` |
| revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| local path | `/plm-shared/zhangjunming/Workspace/models/bge-m3` |
| dense dimension | 1024 |
| default device | CPU；MIRIX 与 SeCom 的原生 CUDA 路径例外，见下文 |

Embedding 只参与方法内部索引或检索，不直接选择最终答案。完整 YAML 和 SHA-256 会保存
到 shard plan 与每个 run manifest 中。

大多数方法把 BGE-M3 放在 CPU，GPU 专用于每卡一个持久 vLLM server。MIRIX 的原生
memory writer 明确配置 `embedding_device=cuda`，SeCom 的官方 LLMLingua2 compressor
也默认要求 CUDA；launcher 因此只向这两个 adapter 暴露其所配对的单张 GPU，并把
实际 `adapter_cuda_visible_devices` 写入 `worker_runtime.json`。它们不会看到其他
分片的 GPU。MemRL 则显式从 YAML 传递 `embedding.dim=1024` 给本地 Qdrant，避免依赖
模型路径字符串猜测维度；并发 worker 的本地 cube identity 使用微秒级唯一后缀，防止
不同 shard 重复局部 context ID 时发生 SQLite 唯一键冲突。这些都是本地运行时兼容，
不改变方法的构建、检索或更新策略。

### 3.2 MedMemoryBench-source 方法

这七个方法共享适配入口 `eval/medmemorybench_adapters/structured_memory.py`，使用
`third_party/medmemorybench` 中固定的源码快照。适配器只负责 HABIT session 格式转换、
用户状态隔离、增量 ingestion 和 retrieval-only 输出，不替换原生 memory 表示。

| 方法 | 配置文件 | 关键配置 |
| --- | --- | --- |
| `mem0` | `mem0_qwen3-8b_adapted.yaml` | top-k 5；4,096-token 写入 chunk；32,768 context；compact prompts；strict JSON schema |
| `amem` | `amem_qwen3-8b_adapted.yaml` | top-k 5；A-MEM OpenAI backend；evolution threshold 100；2,048 max tokens；4,096-token chunk |
| `memos` | `memos_qwen3-8b_adapted.yaml` | top-k 5；general text memory；最大 memory context 24,000 tokens；输入/question 上限 4,096 |
| `memrl` | `memrl_qwen3-8b_adapted.yaml` | candidate top-k 12；similarity threshold 0.2；返回 context 1,800 tokens；Q-value/similarity 混合选择 |
| `lightmem` | `lightmem_qwen3-8b_adapted.yaml` | topic segmentation；event extraction；embedding index/retrieval；offline update；formal profile 中 pre-compress=false |
| `letta` | `letta_qwen3-8b_adapted.yaml` | top-k 5；4,096 embedding chunk；32,768 context；archival passage retrieval；用户级 persistence |
| `mirix` | `mirix_qwen3-8b_adapted.yaml` | top-k 5；multi-store memory；4,096 retrieval context；严格工具 schema、单次 meta update；本地 JSON bridge 保留官方 `finish_memory_update`；memory-child completion 上限 8,192；vLLM xgrammar 禁止 schema 字段间的任意空白，与 WJR q8a20 的 acceptance 运行模式一致 |

实际 YAML 路径：

```text
third_party/medmemorybench/configs/method_config/
├── mem0_qwen3-8b_adapted.yaml
├── amem_qwen3-8b_adapted.yaml
├── memos_qwen3-8b_adapted.yaml
├── memrl_qwen3-8b_adapted.yaml
├── lightmem_qwen3-8b_adapted.yaml
├── letta_qwen3-8b_adapted.yaml
└── mirix_qwen3-8b_adapted.yaml
```

源码和本地兼容修改的 provenance 见
`third_party/medmemorybench/VENDOR_INFO.md` 与 `docs/medmemorybench/changes.md`。

### 3.3 官方源码适配与暂不实现的方法

| 方法 | 构建接口 | 检索接口 | 主要配置 |
| --- | --- | --- | --- |
| `secom` | 官方 segmentation + LLMLingua2 compression，按 session 增量写入 | 官方 FAISS retriever | segment granularity；compression rate 0.9；top-k 5；BGE-M3 |

对应配置：

```text
configs/methods/
└── secom_bge_m3_qwen3.yaml
```

SeCom 是普通源码快照，不是 Git submodule，也不含嵌套 `.git`：

```text
third_party/official-baselines/vendor/SeCom
```

固定 revision 和兼容边界见 `third_party/official-baselines/README.md`。

Graphiti 与 O-Mem 当前标记为 `not_implemented`，不在正式 registry、默认计划、
运行脚本或依赖中，也不应报告实验结果。Graphiti 的本地 Kuzu/API 适配尚未通过
ultra-long lifeline 的稳定性和效率验收；O-Mem 的官方 lifecycle 会在 router JSON
不完整时无界重试，并且逐消息调用 LLM，早期长历史 smoke 中未通过稳定性与效率验收。
问题和恢复条件记录在 `eval/unsupported_methods.json`。在实现有界、可复现且不改变
方法语义的适配前，二者暂不实现。

### 3.4 `no_memory`

`no_memory` 给 Qwen 的内容只有：

```text
current request + response choices
```

不提供历史、不提供检索结果，也不调用 embedding model。它用于测量数据本身的
history-free shortcut 和 base-model prior。

### 3.5 `full_memory`：在线 compact-history 对照

`full_memory` 不训练参数、不构建向量库或 query-dependent 检索。它用相同的
Qwen3-8B 在看不到 probe query、choices、gold 或 hidden state 的条件下，在线压缩
旧历史，并保留最近完整 sessions 原文。

这里采用的是经典的 **rolling/recursive summarization + recent verbatim
buffer**：长输入先分块，再把“上一版摘要 + 下一段旧历史”逐级压缩。这是
[SUMM^N](https://aclanthology.org/2022.acl-long.112/) 的 multi-stage
split-then-summarize 思路在在线对话上的因果化版本，也对应
[MemGPT](https://arxiv.org/abs/2310.08560) 所强调的有限工作上下文与层次化长期记忆。
`full_memory` 是本仓库的主实验方法 ID；论文和结果表应写成
**Full-Memory (rolling compact + recent raw)**，不能把它解释为无损、无限窗口的
full history。原始最近历史截断另以 `full_history` 报告。

窗口不再固定为 36k。`eval/context_windows.py` 提供以下档位：

| 档位 | answer 最大输入 | 为 system/probe/choices 等预留 | history token budget |
| --- | ---: | ---: | ---: |
| `8k` | 8,000 | 2,000 | 6,000 |
| `16k` | 16,000 | 2,000 | 14,000 |
| `32k` | 32,000 | 2,000 | 30,000 |
| `40k` | 40,000 | 2,000 | 38,000 |
| `64k` | 64,000 | 4,000 | 60,000 |
| `128k` | 128,000 | 8,000 | 120,000 |
| `custom` | 用户指定 | 默认约为输入窗口的 1/16，至少 2,000 | 输入窗口减预留，或显式指定 |

默认 `HABITBENCH_CONTEXT_WINDOW_TIER=auto`。`auto` 会选择不超过
`HABITBENCH_MAX_MODEL_LEN` 的最大标准档位：

```text
model capacity  8,192  → 8k tier
model capacity 32,768  → 32k tier
model capacity 40,960  → 40k tier
model capacity 65,536  → 64k tier
model capacity 131,072 → 128k tier
```

因此当前 Qwen3-8B / vLLM 的 40,960 容量会自动使用：

```text
resolved tier:          40k
answer max input:       40,000
history token budget:   38,000
reserved prompt budget: 2,000
```

history 构建算法：

1. 仅收集 `session_index <= probe cutoff` 的 sessions。
2. 使用实际 base-model tokenizer 统计完整 history token 数。
3. 如果完整 history 不超过所选档位的 history budget，输入全部 history 原文。
4. 首次超窗后，把旧 sessions 与已有 compact state 递归合并；正常目标为最多
   2,048 tokens，当前 8,192 tokens 是 bounded state/completion 外层预算，长程
   数据产生的非空 length-limited 输出会保留并记录，而不是被 4,096 硬截断。
5. 只有空响应或不可用输出才按 session 边界递归二分并重试；单个 session 仍无法
   结束时，才启用最多六条事实、每条带合法 session ID 的严格 JSON-schema 兜底。
6. 每条摘要事实必须引用 `[SESSION_ID=...]`，并保留 scope、否定、例外、冲突和变化。
7. 40k profile 为最近完整 sessions 保留约 28,784 tokens 原文（38,000 history
   budget − 8,192 summary − 1,024 wrapper reserve）。
8. answerer 再执行一次最终 tokenizer hard-bound 检查。

原始 recency/session-boundary truncation 现在以 `full_history` 独立运行。

选择固定档位：

```bash
HABITBENCH_CONTEXT_WINDOW_TIER=32k \
bash scripts/submit_clusterx.sh \
  --methods full_memory \
  --datasets food,finance,software \
  --shards 8 --gpus 8 \
  --output-root results/full_memory_32k
```

只有在模型 checkpoint 和服务端确实支持更大窗口时才能选择更大档位：

```bash
HABITBENCH_CONTEXT_WINDOW_TIER=64k \
HABITBENCH_MAX_MODEL_LEN=65536 \
bash scripts/submit_clusterx.sh \
  --methods full_memory \
  --datasets food,finance,software \
  --shards 8 --gpus 8 \
  --output-root results/full_memory_64k
```

非标准窗口使用 `custom`：

```bash
HABITBENCH_CONTEXT_WINDOW_TIER=custom \
HABITBENCH_MAX_MODEL_LEN=49152 \
HABITBENCH_MAX_INPUT_TOKENS=48000 \
bash scripts/submit_clusterx.sh \
  --methods full_memory \
  --shards 8 --gpus 8 \
  --output-root results/full_memory_custom_48k
```

可选高级覆盖：

```text
HABITBENCH_FULL_MEMORY_RESERVED_TOKENS  # 覆盖预留空间
HABITBENCH_FULL_MEMORY_MAX_TOKENS       # 直接覆盖 history budget，主要用于 ablation
```

正式实验优先使用标准档位。实际解析出的档位、输入上限、预留和 history budget 会写入
`control_runtime.json`、每题 `memory_debug.context_window`、
`run_manifest.base_model.max_input_tokens` 和 `suite_runtime.json`。

## 4. 仓库文件分布

```text
HABIT-bench/
├── README.md
├── requirements.txt
├── configs/
│   ├── chat_templates/
│   │   └── qwen3_no_thinking.jinja
│   └── methods/
│       ├── full_memory.yaml
│       ├── recency_5.yaml
│       ├── recency_10.yaml
│       ├── bm25_rag.yaml
│       ├── dense_rag.yaml
│       ├── temporal_hybrid_rag.yaml
│       └── secom_bge_m3_qwen3.yaml
├── domain/
│   ├── food/
│   │   └── food_habit_lifelines_stress_v5/
│   ├── finance-software/
│   │   └── habit_bench_multidogo_finance_software_release_gated_v1_4/
│   └── travel/
│       └── release_candidate_v16_postrepair_repaired_r4/
├── eval/
│   ├── context_windows.py
│   ├── controls.py
│   ├── methods.json
│   ├── unsupported_methods.json
│   ├── run.py
│   ├── score.py
│   ├── validate.py
│   ├── merge_shards.py
│   ├── core/
│   │   ├── dataset.py
│   │   ├── answering.py
│   │   ├── scoring.py
│   │   ├── retrieval_scoring.py
│   │   └── io.py
│   ├── medmemorybench_adapters/
│   │   └── structured_memory.py
│   ├── official_adapters/
│   │   └── secom.py
│   └── supplementary/
│       ├── analyze.py
│       ├── compare.py
│       ├── oracle_controls.py
│       ├── merge_oracle.py
│       ├── human_audit.py
│       ├── human_audit_adjudication.py
│       └── human_audit_data_manager.py
├── schema/
│   ├── session.schema.json
│   ├── probe.schema.json
│   └── memory_context.schema.json
├── scripts/
│   ├── run_eval.sh
│   ├── create_shard_plan.py
│   ├── run_multigpu_plan.py
│   ├── merge_shard_plan.py
│   ├── run_supplementary_analysis.py
│   ├── submit_clusterx.sh
│   ├── submit_h_cluster.sh
│   └── cluster/
│       ├── create_h_envs.sh
│       ├── download_h_models.py
│       ├── env.example.sh
│       ├── env.h.example.sh
│       ├── submit_qwen3_8b_supplementary.sh
│       └── run_h_eval.sh
├── tests/
│   └── evaluation/
│       ├── test_supplementary.py
│       ├── test_supplementary_runner.py
│       └── test_retrieval_baselines.py
├── third_party/
│   ├── medmemorybench/
│   └── official-baselines/
├── docs/
│   ├── evaluation_protocol.md
│   ├── compact_context_baseline.md
│   ├── retrieval_baselines.md
│   ├── main_experiment_results_current.md
│   ├── supplementary_results_current.md
│   ├── multigpu_evaluation.md
│   ├── h_cluster_evaluation.md
│   ├── medmemorybench/
│   ├── supplementary_exp/
│   │   ├── ICLR27_supplementary_experiments.md
│   │   └── HABIT-Bench_ICLR27_experiment_analysis.md
│   └── human_audit/
│       ├── HUMAN_AUDIT_GUIDE.md
│       └── HUMAN_AUDIT_RESULTS_V3.md
├── research/
└── results/                         # 运行生成，Git ignored
```

关键文件职责：

| 文件 | 职责 |
| --- | --- |
| `eval/core/dataset.py` | 归一化两种数据格式、应用用户分片、构建无 gold 的 method payload |
| `eval/medmemorybench_adapters/structured_memory.py` | 七个 MedMemoryBench-source 方法的统一增量 ingestion/retrieval 入口 |
| `eval/official_adapters/secom.py` | SeCom 的官方源码薄适配层 |
| `eval/unsupported_methods.json` | Graphiti、O-Mem 等暂不实现方法的问题和恢复条件 |
| `eval/compact_history.py` | query-independent 在线 compact `full_memory` |
| `eval/controls.py` | `no_memory` 和原始 recency `full_history` |
| `eval/retrieval_baselines.py` | Recency-k、BM25-RAG、Dense-RAG 和 Temporal Hybrid-RAG 完整 session 检索 |
| `eval/api_gateway.py` | 外部 OpenAI-compatible API 的全局 RPM/TPM 限流、429/5xx/网络重试与脱敏运行统计 |
| `eval/context_windows.py` | 长上下文档位解析与验证 |
| `eval/core/answering.py` | 固定 Qwen answer prompt、token hard bound、choice JSON 解析 |
| `eval/run.py` | 单方法/单数据集/单分片端到端运行 |
| `eval/core/scoring.py` | Accuracy、Wilson CI、能力分组和统一结果输出 |
| `eval/core/retrieval_scoring.py` | 分域 evidence、chain、decoy、provenance 和 decision-unit 指标 |
| `eval/merge_shards.py` | 严格验证并合并某个方法/领域的全部分片 |
| `eval/supplementary/analyze.py` | 用户级 bootstrap、transfer gap、policy component、难度切片、效率和 answer–retrieval 分析 |
| `eval/supplementary/compare.py` | 方法间 paired user-cluster bootstrap、exact McNemar 和 Holm 校正 |
| `eval/supplementary/oracle_controls.py` | `oracle_evidence`、`oracle_habit_state` 诊断上下文和固定 answer-head 运行 |
| `eval/supplementary/merge_oracle.py` | 严格合并 Oracle 用户分片并保留 no-retrieval 语义 |
| `eval/supplementary/human_audit.py` | 分层盲审样本生成、A/B 评分、raw agreement 和 Cohen's κ |
| `eval/supplementary/human_audit_adjudication.py` | 不接触 gold 的分歧表生成、题目渲染、C 裁决合并和哈希冻结 |
| `eval/supplementary/human_audit_data_manager.py` | 冻结校验后的解盲、数据质量统计和 keep/modify/exclude 清单 |
| `scripts/create_shard_plan.py` | 生成确定性 method × dataset × shard 任务表及配置快照 |
| `scripts/run_multigpu_plan.py` | 每 GPU 一个持久 vLLM worker，调度分片任务并记录 wall-clock |
| `scripts/merge_shard_plan.py` | 合并所有 method/domain group，生成总览 |
| `scripts/run_supplementary_analysis.py` | 对完整 suite 批量生成四域单方法分析和全方法配对比较 |
| `scripts/submit_clusterx.sh` | 唯一 ClusterX 提交入口 |
| `scripts/submit_h_cluster.sh` | H 集群 4/8 卡每 Replica、支持多 Replica 的 RJob 提交入口 |
| `scripts/submit_h_api_suite.sh` | 三个外部 API 模型、全部 16 方法和当前四域的单节点 8-H200 提交入口 |
| `scripts/cluster/create_h_envs.sh` | 用当前 Miniconda 在 GPFS 创建固定方法/vLLM 环境 |
| `scripts/cluster/download_h_models.py` | 从官方 Hugging Face 断点下载 H 集群固定 revision 模型 |
| `scripts/cluster/run_h_eval.sh` | H worker 的 H200/卡数预检、分片执行和合并入口 |
| `docs/compact_context_baseline.md` | full-memory compact-history 的实现与对照规范 |
| `docs/retrieval_baselines.md` | 非 agentic session retrieval baselines 的公式、参数和运行规范 |
| `docs/supplementary_exp/` | supplementary 设计、可执行范围、不可从当前数据可靠得到的指标 |
| `docs/human_audit/` | 人工审计 rubric、盲法、裁决流程和 v3 最终审计结果 |

## 5. 环境

固定环境与模型：

```text
Method/evaluation Conda env (Python 3.11.15):
/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark

Dedicated vLLM Conda env (Python 3.10.20):
/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark-vllm

ClusterX:
/plm-shared/zhangjunming/miniconda3/envs/clusterx/bin/clusterx

Qwen3-8B:
/plm-shared/zhangjunming/Workspace/models/Qwen3-8B

BGE-M3:
/plm-shared/zhangjunming/Workspace/models/bge-m3

LLMLingua2:
/plm-shared/zhangjunming/Workspace/models/llmlingua-2-xlm-roberta-large-meetingbank
```

不要创建 `.venv`。两个环境严格分离，避免评测方法依赖与 vLLM 的
Torch/Triton 版本互相覆盖：

| 文件 | 环境 |
| --- | --- |
| `environment.yml` / `requirements.txt` | `habitbenchmark`；方法、adapter、scorer |
| `environment-vllm.yml` / `requirements-vllm.txt` | `habitbenchmark-vllm`；仅 Qwen3-8B serving |

在本项目根目录创建环境：

```bash
/plm-shared/zhangjunming/miniconda3/bin/conda env create \
  -p /plm-shared/zhangjunming/miniconda3/envs/habitbenchmark \
  -f environment.yml

/plm-shared/zhangjunming/miniconda3/bin/conda env create \
  -p /plm-shared/zhangjunming/miniconda3/envs/habitbenchmark-vllm \
  -f environment-vllm.yml
```

环境变量模板位于 `scripts/cluster/env.example.sh`。模型 checkpoint 不属于
Python requirements，也不会提交到 Git。

MemRL 经由其原生 MemOS reader 使用 `chonkie==1.2.1`、
`prometheus-client==0.23.1` 和 GPT-2 tokenizer；MemOS 使用相同的固定 Prometheus
依赖。正式 launcher 会在启动 vLLM 和执行长任务之前检查这些包版本，以及
`TIKTOKEN_CACHE_DIR` 中 GPT-2 的 vocab/encoder 缓存；缺失时 preflight 立即失败，
并用隔离子进程深层导入已选 adapter；不会等到运行数小时后才在 Memos/MemRL shard
初始化时暴露依赖或 Python namespace 问题。

Vendored Letta 和 MIRIX 的 Python import graph 会在创建本地数据库时加载一批即使
文本评测路径不直接调用、但仍属于必需的运行时依赖。`requirements.txt` 已固定当前
验证通过的 Letta/MIRIX 兼容版本；launcher 会在占用 GPU 前检查这些关键包。当前环境
已实际完成 Letta SQLite 与 MIRIX SQLite 的 `AgentManager` 初始化，而不只做静态
`import` 检查。

MIRIX 的 vLLM 启动方式参考官方 OpenAI-compatible 路径：保留
`--enable-auto-tool-choice --tool-call-parser hermes`。本评测已通过 chat template 禁用
thinking，因此不启用 `--reasoning-parser`；否则 schema-constrained JSON 可能被移出
`message.content`，与本地 JSON bridge 的传输约定冲突。正式 launcher 会在 MIRIX
preflight 中拒绝这类不兼容配置。

本地 serving 若仍提前结束 memory-tool JSON，bridge 会复用 MIRIX 官方的
`json.loads` / `demjson3` / `json-repair` 容错解析链，并在执行前再次按原工具
schema 严格校验；schema 通过后还会预先调用 MIRIX 官方 `validate_tool_args`，把
`steps=[]` 等 schema 未表达、但官方运行时拒绝的业务约束纳入同一组有界纠错生成。
纠错提示仅回传字段级校验原因，不包含 memory 正文。仅对已经由官方 parser
修复的截断对象，在所有声明必填字段均已
存在时，bridge 才会丢弃 schema 外的 repair artifact（例如
[MIRIX issue #103](https://github.com/Mirix-AI/MIRIX/issues/103) 中 semantic item
多出的 `tree_path`）。如果截断发生在对象数组的下一项中，则只在前缀至少包含一个
完整合法项、尾项缺少必填字段且丢弃后仍满足数组 schema 时，丢弃 parser 补出的
残缺尾项。完整 JSON、单独的残缺项、非法前缀或声明字段类型不符仍会拒绝。
后续纠错生成采用固定的逐次 seed 和轻量采样，避免温度为零时重复五次完全相同的
畸形输出。该适配复用而不改动 MIRIX 的 validator，也不改动 executor、storage
或多 memory-child 生命周期。

常用环境变量：

| 变量 | 默认值/用途 |
| --- | --- |
| `HABITBENCH_VLLM_PYTHON` | 独立 vLLM 0.17.1 环境；不复用 method Python |
| `HABITBENCH_LLM_MODEL` | Qwen checkpoint 路径 |
| `HABITBENCH_SERVED_MODEL` | OpenAI-compatible served name |
| `HABITBENCH_MAX_MODEL_LEN` | vLLM 服务端实际支持的总窗口 |
| `HABITBENCH_CONTEXT_WINDOW_TIER` | `full_memory` 的 `auto` 或显式档位 |
| `HABITBENCH_COMPACT_SUMMARY_MAX_TOKENS` | compact state 上限，当前正式 profile 为 8,192；非空 length-limited 输出会记录并保留 |
| `HABITBENCH_COMPACTOR_INPUT_TOKENS` | 单次 compactor 输入上限，默认 30,000 |
| `HABITBENCH_MAX_INPUT_TOKENS` | `custom` 档位的 answer 输入上限 |
| `HABITBENCH_EMBED_MODEL` | BGE-M3 本地路径 |
| `HABITBENCH_EMBED_DEVICE` | embedding 默认设备为 CPU；MIRIX/SeCom 按原生配置强制绑定其 worker GPU |
| `HABITBENCH_<METHOD>_USER_WORKERS` | 方法级用户并发：Mem0/A-MEM/MemOS/MemRL/Letta=7、LightMem/MIRIX=1 |
| `HABITBENCH_ADAPTER_CPU_THREADS` | 每个 adapter 进程的 BLAS/OpenMP 线程数，默认 2 |
| `HABITBENCH_LIGHTMEM_MODEL` | LightMem/LLMLingua2 模型路径 |
| `HABITBENCH_GPU_MEMORY_UTIL` | vLLM GPU memory utilization |
| `HABITBENCH_ENABLE_PREFIX_CACHING` | 默认 1；复用同用户长历史 prompt prefix |
| `HABITBENCH_VLLM_MIN_TOKENS_PER_SEC` | 启动后并发聚合 decode 吞吐门槛，默认 60 |
| `HABITBENCH_VLLM_BENCHMARK_CONCURRENCY` | 吞吐门禁的代表性并发数，默认 4 |
| `HABITBENCH_TASK_LOCK_POLL_SEC` | 跨 RJob shard 输出锁轮询间隔，默认 5 秒 |
| `HABITBENCH_TASK_LOCK_LOG_EVERY_SEC` | 输出锁等待日志间隔，默认 60 秒 |
| `HABITBENCH_FORCE_RERUN` | 设为 1 时重跑已有完整 shard |

## 6. 运行方式

### 6.1 数据和单元测试

```bash
cd /plm-shared/zhangjunming/Workspace/HABIT-bench

python \
  -m eval.validate \
  domain/food/food_habit_lifelines_stress_v5 \
  domain/finance-software/habit_bench_multidogo_finance_software_release_gated_v1_4 \
  domain/travel/release_candidate_v16_postrepair_repaired_r4

PYTHONDONTWRITEBYTECODE=1 \
python \
  -m unittest discover -s tests/evaluation -p 'test_*.py'
```

### 6.2 Prepare-only

Prepare-only 验证数据加载、配置 snapshot 和 manifest，不调用 memory 方法或模型：

```bash
bash scripts/run_eval.sh mem0 \
  domain/food/food_habit_lifelines_stress_v5 \
  ./results/habit_mem0_prepare \
  --max-users 1 --max-probes 4 --prepare-only
```

### 6.3 ClusterX 正式评测

默认运行八个已实现的 memory 方法，不自动包含 control：

```bash
bash scripts/submit_clusterx.sh \
  --datasets food,finance,software,travel \
  --shards 8 \
  --gpus 8 \
  --output-root results/habit_all_methods_v1
```

只运行主要 MedMemoryBench-source suite：

```bash
bash scripts/submit_clusterx.sh \
  --methods mem0,amem,memos,memrl,lightmem,letta,mirix \
  --datasets food,finance,software,travel \
  --shards 8 --gpus 8 \
  --output-root results/habit_medmemorybench_v1
```

运行 control：

```bash
bash scripts/submit_clusterx.sh \
  --methods no_memory,full_memory \
  --datasets food,finance,software,travel \
  --shards 8 --gpus 8 \
  --output-root results/habit_controls_v1
```

运行新增官方适配：

```bash
bash scripts/submit_clusterx.sh \
  --methods secom \
  --datasets food,finance,software,travel \
  --shards 8 --gpus 8 \
  --output-root results/habit_official_extra_v1
```

仅检查计划和 ClusterX 命令：

```bash
bash scripts/submit_clusterx.sh \
  --dry-run \
  --methods no_memory,full_memory \
  --output-root /plm-shared/zhangjunming/tmp/habit_clusterx_dry
```

真实端点 smoke 可在同一入口加 `--max-users N --max-probes N`。这两个参数会进入
plan manifest 和 dataset subset，不能把此类结果混入正式全量表格。

相同 `output-root` 可用于断点恢复；只有原子成功标记与完整最终产物同时存在的 shard
才会跳过，失败/中断半成品在重跑前删除。
每个真实 shard 输出目录还由相邻的 `.habitbench-shard-locks/` POSIX 锁保护；锁覆盖
断点检查、半成品清理、执行和最终标记发布。因此两个不同 `JOB_ID` 即使误用同一
`output-root`，也只能串行处理同一 shard，后取得锁的任务会重新检查并复用完整结果。
worker 或 RJob 被终止时内核自动释放锁，不会留下阻塞恢复的永久 claim。
只有明确希望覆盖完整 shard 时才设置 `HABITBENCH_FORCE_RERUN=1`。

### 6.4 H 集群 4/8 卡 H200 评测

H 集群使用独立 RJob 入口，不修改现有 ClusterX 命令。个人 User-AD 的 4 卡低优
评测示例：

```bash
bash scripts/submit_h_cluster.sh \
  --job-type managed-spot \
  --creator-type user \
  --creator-ad your-user-ad \
  --gpus 4 \
  --env-file scripts/cluster/env.h.local.sh \
  --output-root "$PWD/results/habit-h200-4g-v1"
```

launcher 会在任何 `rjob` 调用前自动设置 llmarchitecture 新调度入口
`KUBEBRAIN_CLUSTER_ENTRY=http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451`；
直接运行 `rjob list/logs/events/stop` 时需在当前 shell 手工导出同一变量。

H profile 默认复用
`/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/{envs,models}` 的只读环境和模型，但会把
结果及 HF/vLLM/Triton 等可写 cache 放到当前 clone 的 `results/`。其他用户在自己的
`plm-gpfs/<user>/...` 下 clone 后无需改共享模型路径；提交器会自动为共享资源根和该
用户的项目/输出根生成并校验两个 mount。

8 卡只需改为 `--gpus 8`；两个 8 卡节点使用 `--gpus 8 --replicas 2 --shards 16`。
多节点评测不使用 DDP：16 个常驻 GPU worker 从 GPFS 全局动态队列领取 shard，快卡可
跨 method/domain 边界继续工作，不再因静态奇偶分片而等待另一节点。
高优 reserved 必须把任务类型改成 `reserved`，并由真实
Group-AD 使用 `--creator-type group` 提交；idle 不带 priority、charged-group 或
private-machine。环境、模型持久化、dry-run 和中断恢复见
[`docs/h_cluster_evaluation.md`](docs/h_cluster_evaluation.md)。

#### 外部 API 主实验

外部 OpenAI-compatible 模型复用同一评测协议、当前四域数据和全部 16 个主方法；一个
本地网关对所有 worker 统一实施 RPM/TPM 限制，并在日志中隐藏 credential 和 prompt。
默认三模型为 `deepseek-v4-pro-0813`、`glm-5.2`、`kimi-k3`，单 Replica 的 8 张 H200
按 `3/3/2` 分配给三个模型轨道。H200 只承担 BGE-M3、MIRIX、SeCom 等本地组件，LLM
生成由远端 API 完成，因此不能用 GPU utilization 衡量 API suite 的有效吞吐。

```bash
bash scripts/submit_h_api_suite.sh \
  --job-type managed-spot \
  --creator-type user \
  --creator-ad your-user-ad \
  --credential-file /absolute/private/path/api.env \
  --output-root "$PWD/results/api-main-v1"
```

credential 文件必须是 `600` 权限且位于 Git 仓库外；格式、限流语义、断点恢复与输出
结构见 [`docs/external_api_evaluation.md`](docs/external_api_evaluation.md)。

### 6.5 当前四域 Qwen3-8B supplementary

当前入口覆盖 Food v5、Finance/Software v1.4 和 Travel v16，提交一个
`2 Replicas × 8 H200 = 16 H200` 的 reserved H 集群 RJob：

```bash
export HABITBENCH_CREATOR_AD=<实际 Group-AD>
bash scripts/cluster/submit_qwen3_8b_supplementary.sh
```

任务执行 `no_memory`、`oracle_evidence` 和 `oracle_habit_state`，补齐主实验已有方法之外
仍需 Qwen3-8B 推理的诊断对照。全部分片严格合并后，会自动运行用户级 bootstrap、配对
显著性、难度切片、policy-component、效率和 answer–retrieval sidecar；不准备、不执行
人审。结果默认写入当前 clone 的
`results/habit-h200-supplementary-qwen3-8b-v1`。每个完整用户分片都是持久化 checkpoint，
同一路径重提会校验并跳过成功分片。

旧 `submit_v3_three_nodes.sh` 和 `create_v3_experiment_plans.py` 只保留兼容转发，不再包含
三域、旧数据版本、ClusterX 或 A800 调度逻辑。

## 7. 当前四域实验结果

当前正式口径为 Food v5、Finance/Software v1.4、Travel v16，回答模型为同规模
Qwen3。完整的 4B/8B/14B/32B 主实验大表、Wilson 95% CI、证据指标和结果来源见
[docs/main_experiment_results_current.md](docs/main_experiment_results_current.md)。
以下只保留最常用的 Qwen3-8B Accuracy 摘要；README 不再汇报旧 v3/Food v4 结果。

### 7.1 Qwen3-8B memory-agent 主实验

| Method | Food | Finance | Software | Travel v16 | Macro | Pooled |
|---|---:|---:|---:|---:|---:|---:|
| Full-Memory | 53.20 | 20.76 | 23.24 | 27.23 | 31.11 | 33.61 |
| Mem0 | 37.21 | 23.10 | 26.76 | 26.62 | 28.42 | 29.22 |
| A-MEM | 50.48 | 22.59 | 26.47 | 32.77 | 33.08 | **34.64** |
| MemOS | 47.96 | 22.15 | 26.32 | 31.38 | 31.95 | 33.37 |
| MemRL | 51.09 | 21.86 | 25.00 | 32.46 | 32.60 | 34.33 |
| LightMem | 35.85 | 22.51 | 25.44 | 28.00 | 27.95 | 28.55 |
| Letta | 51.63 | 21.71 | 25.29 | **33.23** | 32.97 | **34.64** |
| MIRIX | 33.67 | 22.51 | **27.50** | 27.69 | 27.85 | 28.07 |
| SeCom | 48.23 | 23.03 | 24.85 | 32.92 | 32.26 | 33.76 |

4B、8B 和 14B 的 36/36 method×domain groups 均完整；32B 为 34/36，尚缺 Mem0
Finance/Software。各规模完整方法中的最高 pooled Accuracy 分别为 4B Letta 34.81%、
8B Letta/A-MEM 34.64%、14B Letta 36.13% 和 32B SeCom 37.72%。

### 7.2 Qwen3-8B 简单 session-retrieval 基线

这些方法不做 LLM 记忆抽取、摘要或 agentic memory lifecycle；实现与公平性边界见
[docs/retrieval_baselines.md](docs/retrieval_baselines.md)。

| Method | Food | Finance | Software | Travel v16 | Macro | Pooled |
|---|---:|---:|---:|---:|---:|---:|
| Recency-5 | 33.74 | 23.03 | 25.88 | 26.15 | 27.20 | 27.76 |
| Recency-10 | 35.99 | 22.81 | 26.32 | 29.08 | 28.55 | 29.01 |
| BM25-RAG | 49.25 | 22.95 | 24.41 | 32.46 | 32.27 | 33.95 |
| Dense-RAG | **51.16** | 22.73 | 25.44 | **32.46** | **32.95** | **34.72** |
| Temporal Hybrid-RAG | 49.46 | 21.05 | **26.32** | 32.31 | 32.28 | 33.69 |

五种基线已完成 320/320 shards 和 20/20 strict merges。Dense-RAG 的 pooled
Accuracy 为 34.72%，因此简单语义检索是复杂 memory lifecycle 必须面对的关键对照。

## 8. Supplementary 实验

当前 Qwen3-8B 四域 supplementary 结果以
[docs/supplementary_results_current.md](docs/supplementary_results_current.md) 为准；
完整设计、统计假设和指标边界见
[docs/supplementary_exp/ICLR27_supplementary_experiments.md](docs/supplementary_exp/ICLR27_supplementary_experiments.md)。

Supplementary 分析不修改主 scorer，也不覆盖 merged/metrics.json。它只读取 strict
merge 后的预测与运行 artifacts，把用户级统计、配对推断、难度切片、效率和诊断结果
写入独立的 supplementary/ 目录。

### 8.1 当前可用性

| 实验 | 当前四域状态 | 说明 |
|---|---:|---|
| 主方法离线 sidecar | 36/36 | 9 方法 × Food/Finance/Software/Travel |
| No Memory / Oracle diagnostics | 12/12 | 3 diagnostics × 4 域，192/192 shards |
| 用户级统计与 user-cluster CI | 36/36 | 10,000 次 bootstrap，seed 42 |
| 域内方法配对 | 4 × 36 pairs | exact McNemar、user-cluster bootstrap、Holm 校正 |
| Explicit-to-Habit transfer | Food 9/9 | 其余域没有同数据生成机制下的 explicit probes |
| Policy-component Accuracy | 27/27 | Finance、Software、Travel，Food taxonomy 不适用 |
| 难度切片、效率、错误耦合 | 36/36 | 缺少 lineage 的方法按不可归因解释 |
| 概率校准 / 严格 FPC | unavailable | 当前输出或标签不足，不用近似指标冒充 |
| 人工盲审 | 本轮排除 | 协议保留，但不计入当前自动 supplementary 结果 |

### 8.2 当前关键诊断

下表为 Qwen3-8B exact-choice micro Accuracy（%），Pooled 覆盖四域 4,168 probes。
Oracle 读取 private annotations，只是诊断上界，不属于可部署 memory system。

| Diagnostic | Food | Finance | Software | Travel v16 | Pooled | Correct / Total |
|---|---:|---:|---:|---:|---:|---:|
| No Memory | 26.26 | 23.32 | 27.21 | 26.46 | 25.48 | 1,062 / 4,168 |
| Oracle Evidence | **78.78** | 29.82 | 31.91 | 42.15 | 49.35 | 2,057 / 4,168 |
| Oracle Habit State | 59.86 | **93.71** | **94.56** | **98.31** | **82.63** | 3,444 / 4,168 |

Finance、Software 和 Travel 在给定最终 Oracle Habit State 后接近饱和，说明主要瓶颈
是从长期历史恢复正确状态；Food 的 Oracle Evidence 更强，符合显式历史证据可直接
支持部分题目的构造。用户等权 Accuracy、transfer gap、policy component、错误耦合、
效率、配对显著性和完整结果路径见 current supplementary 文档。

### 8.3 如何运行 supplementary 分析

前置条件是主实验已经 strict merge，且每个目标 method/domain 都存在完整
`merged/scored_predictions.jsonl`。Supplementary 是 CPU 离线统计，不需要重新启动
memory 方法或 answer model：

```bash
cd /path/to/your/habit-bench

PYTHON=/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/envs/habitbenchmark/bin/python
SUITE_ROOT="$PWD/results/<current-four-domain-suite>"

"$PYTHON" scripts/run_supplementary_analysis.py \
  --suite-root "$SUITE_ROOT" \
  --output-root "$SUITE_ROOT/supplementary" \
  --domains food,finance,software,travel \
  --bootstrap-samples 10000 \
  --seed 42
```

默认发现所有完整方法。若使用 `--methods mem0,memos,...`，任一指定方法缺少完整
merged 结果都会报错，不会静默跳过。对单个方法排障可直接运行：

```bash
"$PYTHON" -m eval.supplementary.analyze \
  --dataset-dir domain/food/food_habit_lifelines_stress_v5 \
  --scored-predictions \
    "$SUITE_ROOT/food/memos/merged/scored_predictions.jsonl" \
  --artifact-root "$SUITE_ROOT/food/memos/merged" \
  --output-dir "$SUITE_ROOT/supplementary/food/memos" \
  --bootstrap-samples 10000 \
  --seed 42
```

批处理器会自动运行 `compare.py`。手工比较时，应传入同一领域、probe coverage 完全
相同的多个 `--run METHOD=PATH`；默认拒绝 partial coverage。

Oracle 需要固定的 OpenAI-compatible answer endpoint。当前四域 H 启动器自动包含两个
Oracle；也可先只检查上下文和 private-label contract：

```bash
"$PYTHON" -m eval.supplementary.oracle_controls \
  --dataset-dir domain/food/food_habit_lifelines_stress_v5 \
  --output-dir ./results/oracle_food_smoke \
  --mode oracle_evidence \
  --max-users 1 \
  --max-probes 8 \
  --prepare-only
```

正式 Oracle 可移除 `--prepare-only` 并显式指定 `--base-url`、`--base-model` 和
`--base-model-path`。多用户分片必须使用 `eval.supplementary.merge_oracle` 严格
合并，尤其不能把 `oracle_habit_state` 的 no-retrieval 语义伪装成全零 Recall@5。

### 8.4 如何生成和完成人工审计

人工审计必须把盲态 annotation 与 private audit key 隔离。以下命令生成新的固定
seed、按 probe type 分层的四域样本；不要在已经开始标注的同一目录上重新运行：

```bash
cd /path/to/your/habit-bench

PYTHON=${PYTHON:-python}
REPO="$PWD"
AUDIT_ROOT="$REPO/results/<new_suite>/supplementary/human_audit"

"$PYTHON" -m eval.supplementary.human_audit prepare \
  --dataset-dir "$REPO/domain/food/food_habit_lifelines_stress_v5" \
  --output-dir "$AUDIT_ROOT/food" \
  --per-stratum 50 --seed 42

"$PYTHON" -m eval.supplementary.human_audit prepare \
  --dataset-dir "$REPO/domain/finance-software/habit_bench_multidogo_finance_software_release_gated_v1_4" \
  --domain-filter finance \
  --output-dir "$AUDIT_ROOT/finance" \
  --per-stratum 50 --seed 42

"$PYTHON" -m eval.supplementary.human_audit prepare \
  --dataset-dir "$REPO/domain/finance-software/habit_bench_multidogo_finance_software_release_gated_v1_4" \
  --domain-filter software \
  --output-dir "$AUDIT_ROOT/software" \
  --per-stratum 50 --seed 42

"$PYTHON" -m eval.supplementary.human_audit prepare \
  --dataset-dir "$REPO/domain/travel/release_candidate_v16_postrepair_repaired_r4" \
  --output-dir "$AUDIT_ROOT/travel" \
  --per-stratum 50 --seed 42
```

每个域会生成：

```text
annotation_template.csv      # 只含盲态题目、choices 和 evidence packet
audit_key.private.jsonl      # private gold；不得分发给标注员
audit_manifest.json          # dataset hash、seed、strata 和 sample
```

数据管理员为至少两名标注员复制独立文件；不能让标注员共享一份 CSV：

```bash
for domain in food finance software travel; do
  mkdir -p "$AUDIT_ROOT/$domain/annotations"
  cp -n "$AUDIT_ROOT/$domain/annotation_template.csv" \
    "$AUDIT_ROOT/$domain/annotations/annotator_a.csv"
  cp -n "$AUDIT_ROOT/$domain/annotation_template.csv" \
    "$AUDIT_ROOT/$domain/annotations/annotator_b.csv"
done
```

收回完整 A/B 文件后，由持有 private key 的数据管理员运行 pre-adjudication 评分：

```bash
for domain in food finance software travel; do
  DOMAIN_ROOT="$AUDIT_ROOT/$domain"
  "$PYTHON" -m eval.supplementary.human_audit score \
    --audit-key "$DOMAIN_ROOT/audit_key.private.jsonl" \
    --annotation "annotator_a=$DOMAIN_ROOT/annotations/annotator_a.csv" \
    --annotation "annotator_b=$DOMAIN_ROOT/annotations/annotator_b.csv" \
    --output-dir "$DOMAIN_ROOT/scored"
done
```

第三位审查者必须在不知道 gold 的情况下处理 A/B decision 分歧。辅助工具只做机械
对齐、渲染和冻结，不自动替 C 做语义判断：

```bash
DOMAIN_ROOT="$AUDIT_ROOT/food"

"$PYTHON" -m eval.supplementary.human_audit_adjudication prepare \
  --domain-root "$DOMAIN_ROOT"

"$PYTHON" -m eval.supplementary.human_audit_adjudication render \
  --domain-root "$DOMAIN_ROOT" \
  --item-id AUDIT-00001

# C 独立填写 adjudication/adjudicator_c_review.csv 后：
"$PYTHON" -m eval.supplementary.human_audit_adjudication merge \
  --domain-root "$DOMAIN_ROOT"
```

`merge` 会冻结 A/B/C、分歧表和全量 `adjudicated.csv` 的 SHA-256。只有冻结成功后，
数据管理员才能解盲并生成 keep/modify/exclude：

```bash
"$PYTHON" -m eval.supplementary.human_audit_data_manager \
  --domain-root "$AUDIT_ROOT/food" \
  --audit-key "$AUDIT_ROOT/food/audit_key.private.jsonl" \
  --source-probe-key \
    "$REPO/domain/food/food_habit_lifelines_stress_v5/private/probe_key.jsonl"

"$PYTHON" -m eval.supplementary.human_audit_data_manager \
  --domain-root "$AUDIT_ROOT/finance" \
  --audit-key "$AUDIT_ROOT/finance/audit_key.private.jsonl"
```

Software 和 Travel 与 Finance 相同，只需替换 domain。最终输出包括 pre-adjudication agreement、
C 后的质量指标、逐 probe-type 汇总、逐题 disposition，以及验证 private inputs 和
scored outputs 的 unblinding manifest。完整 rubric、字段定义、盲法和发布规则见
`docs/human_audit/HUMAN_AUDIT_GUIDE.md`。

## 9. 结果目录结构

一次完整 suite 的输出：

```text
results/<run_name>/
├── shard_plan.tsv
├── shard_plan.manifest.json
├── clusterx_submit.log
├── suite_runtime.json
├── evaluation_summary.json
├── vllm_logs/
│   ├── vllm_worker_00_gpu_0.log
│   └── ...
├── food/
│   └── <method>/
│       ├── shard_000_of_008/
│       ├── ...
│       ├── shard_007_of_008/
│       └── merged/
├── finance/
│   └── <method>/
│       ├── shard_000_of_008/
│       ├── ...
│       └── merged/
├── software/
│   └── <method>/
│       ├── shard_000_of_008/
│       ├── ...
│       └── merged/
└── travel/
    └── <method>/
        ├── shard_000_of_008/
        ├── ...
        └── merged/
```

顶层文件：

| 文件 | 内容 |
| --- | --- |
| `shard_plan.tsv` | 每行一个 method × dataset × user-shard 任务 |
| `shard_plan.manifest.json` | plan hash、dataset hash、分片数、方法 YAML、模型 identity、Git 状态和 ClusterX 参数 |
| `clusterx_submit.log` | ClusterX 返回的 job 信息 |
| `suite_runtime.json` | vLLM 版本/worker/GPU/端口、单流和并发吞吐门禁、各 method/domain group wall-clock、失败状态和安全环境快照 |
| `evaluation_summary.json` | 所有合并后 method/domain 的 score、配置、分片数和 timing 总览 |
| `vllm_logs/` | 每张 GPU 上持久 vLLM 服务的日志 |

### 9.1 每个 shard

```text
shard_000_of_008/
├── run_manifest.json
├── worker_runtime.json
├── method_input.json
├── memory_contexts.jsonl
├── predictions.jsonl
├── scored_predictions.jsonl
├── metrics.json
├── metrics_by_group.csv
├── retrieval_metrics.json
├── retrieval_metrics_by_group.csv
├── adapter.stdout.log
├── adapter.stderr.log
├── task.stdout.log
├── task.stderr.log
└── <method-specific runtime/config/state>
```

| 文件 | 内容 |
| --- | --- |
| `run_manifest.json` | implementation source/revision、dataset hash/subset、base-model 配置、方法 YAML snapshot/hash、adapter/answer runtime、GPU、shard index/count、最终状态 |
| `worker_runtime.json` | 外层 GPU worker、adapter CUDA/CPU/用户并发配置、vLLM endpoint、命令、开始/结束时间和 wall-clock |
| `method_input.json` | 发送给 adapter 的无 gold payload；可用于复现 adapter |
| `memory_contexts.jsonl` | 每个 probe 的 `memory_context`、evidence IDs、debug 和 cost |
| `predictions.jsonl` | Qwen 选择结果、模型 usage/latency、实际使用的 memory token 数 |
| `scored_predictions.jsonl` | scorer 加入 correctness、逐题 retrieval/chain 诊断和私有分组标签后的记录 |
| `metrics.json` | answer、retrieval、能力 panel、decision-unit 平衡聚合的完整 JSON |
| `metrics_by_group.csv` | `metrics.json` 中 grouped metrics 的扁平 CSV |
| `retrieval_metrics.json` | 只保留 retrieval/provenance 指标、定义和分域能力 panel 的视图 |
| `retrieval_metrics_by_group.csv` | retrieval 指标按 domain/probe/capability/stress 分组的 CSV |
| `adapter.*.log` | memory adapter 子进程日志 |
| `task.*.log` | ClusterX GPU worker 外层任务日志 |

方法特有文件：

| 方法 | 附加输出 |
| --- | --- |
| `no_memory/full_memory` | `control_runtime.json`；resolved window、history budget、截断 probe 数 |
| 七个 MedMemoryBench 方法 | `medmemorybench_adapter_runtime.json`（用户级耗时/CPU/RSS/并发吞吐）、`medmemorybench_state/` |
| `secom` | `secom_method_config.json`、`secom_config.json`、`secom_runtime.json` |

### 9.2 合并结果

```text
merged/
├── merge_manifest.json
├── memory_contexts.jsonl
├── predictions.jsonl
├── scored_predictions.jsonl
├── metrics.json
├── metrics_by_group.csv
├── retrieval_metrics.json
└── retrieval_metrics_by_group.csv
```

`merged/metrics.json` 才是完整领域结果。单个 shard 的 `metrics.json` 只覆盖部分用户，
不能直接作为完整方法分数，也不能与完整领域结果比较。

`merge_manifest.json` 会严格检查：

- shard 是否完整覆盖 `0 ... shard_count-1`；
- 每个 shard 的 dataset hash 是否一致；
- base model、implementation 和 method config 是否一致；
- 是否使用了 `max_users/max_probes` 子集；
- probe 是否完整且无重复；
- 合并后的完整 score 与 timing。

## 10. 指标和 timing

Answer 主指标仍是严格 choice accuracy：

```text
Accuracy = correct choice_id / all probes
```

同时报告 Wilson 95% confidence interval。Retrieval 不是用来取代 Accuracy，而是用来
回答三个 Accuracy 单独无法回答的问题：

1. 模型是否真正找到了能支持习惯判断的历史证据；
2. 多条弱证据是否形成了完整组件，而不是只命中一条表面相似记录；
3. 正确答案是否伴随可靠 provenance，且没有把局部例外或 assistant 建议当成用户习惯。

对有序 `evidence_session_ids` 去重后取前 5 个，公共指标定义为：

```text
Evidence Recall@5
  = |Top5 ∩ positive gold| / |positive gold|

Evidence Precision@5
  = |Top5 ∩ positive gold| / 5

Evidence Coverage Efficiency@5
  = |Top5 ∩ positive gold| / min(5, |positive gold|)

Joint Answer-Evidence Hit@5
  = 1[answer correct ∧ Top5 至少命中一条 positive gold]
```

其中 Food 的 positive gold 是 `gold_evidence_session_ids`，Finance/Software/Travel 的
positive gold 是 `decision_evidence_session_ids`。还统一报告：

| 指标 | 解决的问题 |
| --- | --- |
| `evidence_recall_at_5_macro/micro` | 每题和全局的决定性证据找回率 |
| `evidence_precision_at_5` | 有限 context slot 是否被真正证据占用 |
| `evidence_coverage_efficiency_at_5` | Finance 有 6 条 gold 时消除 Recall@5 的不可达上限影响 |
| `evidence_hit_rate_at_5` | 至少找到一条支持证据的 probe 比例 |
| `evidence_mrr_at_5` | 第一条决定性证据出现得是否足够靠前 |
| `evidence_ndcg_at_5` | 整个 Top5 排序质量 |
| `full_evidence_rate_at_5` | Top5 是否覆盖全部决定性证据 |
| `source_attribution_probe_coverage` | 有多少 probe 能追溯到至少一个合法可见 source session |
| `invalid/duplicate_attribution_rate` | provenance 是否包含未知、越界或重复 ID |
| `answer_accuracy_when_evidence_hit/miss` | 答对是否依赖检索，还是可能依靠 shortcut |
| `joint_answer_evidence_hit_rate_at_5` | 答案正确且有历史证据支持的端到端成功率 |

MedMemoryBench adapter 优先使用方法保留的 `[SESSION_ID=...]`，其次使用原生 metadata、
稳定 memory ID lineage 或 exact-memory-text lineage 恢复 source session。这些只记录
provenance，不改变方法的 memory 写入、检索排序或传给 answer model 的内容。若某方法
压缩后完全丢失来源，Recall@5 按 0 计入端到端可审计结果，同时另报
`attribution_conditional_evidence_recall_at_5`，避免把“无法归因”和“归因后检索错误”
混为一谈。

Food v5 的能力 panel：

| Panel | 主要指标 |
| --- | --- |
| `habit_induction` | `direct_use` Accuracy、5 条弱证据 Recall/nDCG、grounded joint success |
| `explicit_history_retrieval` | 显式检索 Accuracy 和 evidence Recall@5 |
| `boundary_calibration` | Accuracy、`false_personalization_cost = 1 - Accuracy`、boundary evidence Hit@5 |
| `exception_retention` | Accuracy、`exception_failure_rate`、exception evidence Hit@5 |
| `unseen_paraphrase_robustness` | 未见表达下的 Accuracy 与 evidence metrics |

Finance/Software v1.4 与 Travel v16 额外报告：

| 指标 | 含义 |
| --- | --- |
| `component_hit_coverage_at_5` | 每个目标习惯组件是否至少有一条决定性证据 |
| `component_complete_coverage_at_5` | 每个组件所需的弱证据是否全部找齐 |
| `complete_chain_rate_at_5` | 整个多习惯证据链是否完整 |
| `temporal_context_recall_at_5` | 独立的时序/as-of 上下文找回率 |
| `contextual_evidence_ndcg_at_5` | 决定性证据优先、时序上下文次之的分级排序 |
| `nonbinding_intrusion_rate_at_5` | Top5 被局部例外/未采纳建议占据的比例，越低越好 |
| `decisive_decoy_discrimination_at_5` | 决定性 Precision 减 nonbinding intrusion |
| `clean_grounded_answer_rate_at_5` | 答案正确、命中决定性证据且 Top5 无 nonbinding 干扰 |
| `decision_unit_macro_accuracy` | 潜在决策等权的答案准确率 |
| `decision_unit_macro_evidence_recall_at_5` | 潜在决策等权的组件证据 Recall@5 |
| `decision_bundle_macro_*` | 多习惯 decision bundle 等权结果 |

当单题决定性证据多于 5 条时，原始 Recall@5 存在结构性上限；
`Coverage Efficiency@5` 用于比较有限五个槽位是否全部有效，但不会替换原始 Recall@5。
两者必须同时报告。

Finance 能力 panel 聚合为 `temporal_and_drift_resolution`、
`provenance_and_decoy_rejection` 和 `reference_case_reconstruction`，并继续保留原始
`probe_type`、`capability_group`、`domain` 分组。所有数据还按以下字段输出：

```text
domain
probe_type
capability_group
habit_family
stress_variant
split
```

`no_memory` 没有 retrieval，Top5 指标标记为不适用。`full_memory` 不是有序检索器，
只报告所选长上下文窗口中的 `context_evidence_recall`、完整证据覆盖和 nonbinding
暴露，不把 history 的前五个 session 伪装成 Recall@5。

Timing 字段：

| 字段 | 含义 |
| --- | --- |
| `run_manifest.adapter_runtime.elapsed_sec` | 一个 shard 的 memory construction + retrieval 时间 |
| `run_manifest.answer_runtime.elapsed_sec` | 一个 shard 的固定 Qwen answer 时间 |
| `run_manifest.execution.wall_clock_sec` | shard evaluator 总 wall-clock |
| `worker_runtime.wall_clock_sec` | GPU worker 外层任务 wall-clock |
| `merge_manifest.timing.shard_wall_clock_sum_sec` | 所有 shard wall-clock 之和 |
| `merge_manifest.timing.shard_wall_clock_max_sec` | 理想完全并发下界 |
| `merge_manifest.timing.observed_window_sec` | 最早开始到最晚结束的实际并发窗口 |
| `suite_runtime.groups[].wall_clock_sec` | 一个 method/domain group 的端到端时间 |
| `suite_runtime.wall_clock_sec` | 整个 ClusterX node job，包括 vLLM 启停 |

## 11. 复现与结果命名

正式结果至少应同时保留：

- outer repository revision 和 dirty 状态；
- dataset public/private hash；
- method source/revision；
- method YAML 内容和 SHA-256；
- Qwen checkpoint、served name、thinking、temperature 和 context window；
- embedding model identity；
- shard count、GPU、wall-clock；
- 完整 `memory_contexts`、predictions、scores 和 strict merge manifest。

MedMemoryBench 方法应描述为 “MedMemoryBench-source HABIT adaptation”，不要写成所有原论文
配置的逐项精确复现。SeCom 应描述为 “official-source adapted”，并同时说明本地
compressor、embedding 和在线 ingestion 设置。Graphiti 与 O-Mem 不应出现在正式结果表中。

更严格的 protocol 和 ClusterX 说明见：

```text
docs/evaluation_protocol.md
docs/multigpu_evaluation.md
docs/medmemorybench/README.md
```
