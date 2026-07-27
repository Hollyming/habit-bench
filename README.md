# HABIT-Bench

HABIT-Bench 用于评估 memory agent 能否从超长用户—助手交互历史中的重复弱证据，
归纳并正确应用潜在的、概率性的、受情境约束的用户习惯。

核心假设是：在显式用户记忆基准上表现较好的方法，不一定能够处理需要跨多次弱证据
归纳的习惯，也可能在边界、异常、漂移或证据不足时产生错误个性化。

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
| `food` | food | 30 | 3,600 | 1,470 | `domain/food/food_habit_lifelines_stress_v2` |
| `finance_software` | finance、software | 54 | 29,160 | 2,048 | `domain/finance-software/habit_bench_multidogo_finance_software_scope_consistent_v1.3` |

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

`eval/core/dataset.py` 会把两个领域不同的 public lifeline 组织形式归一化为统一
session/probe contract。Gold answer、gold evidence、habit graph、persona profile 和
policy label 不会进入 memory 方法输入。

Food v2 是 content-constraint 习惯域，包含 210 个受控习惯：

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

Finance/Software v1.3 更强调多证据链、漂移、scope 和 provenance：

| Probe type | 数量 | 设计目标 |
| --- | ---: | --- |
| `dual_asof_reversal` | 384 | 两个习惯在不同 as-of 状态下的反转 |
| `triple_asof_interleaved` | 512 | 三习惯、跨时序的交错证据归纳 |
| `scope_temporal_pair` | 64 | scope 与时间边界联合校准 |
| `surface_decoy_pair` | 384 | 拒绝表面相似但不具约束力的记忆 |
| `suggestion_rejection_pair` | 256 | 区分用户采纳与 assistant 单方面建议 |
| `provenance_weighted_triple` | 128 | 三习惯证据 provenance 加权 |
| `reference_case_reconstruction` | 320 | 重建历史未完成状态和适用 policy |

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

第二组是额外的官方源码/API 适配：

```text
graphiti, secom, omem
```

第三组是 evaluator control，不是 learned memory 方法：

```text
no_memory, full_memory
```

`full_history` 是 `full_memory` 的向后兼容别名。它们不会被默认加入 memory-method
主实验，需要在 `--methods` 中显式选择。

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
| `mirix` | `mirix_qwen3-8b_adapted.yaml` | top-k 5；multi-store memory；4,096 retrieval context；严格工具 schema、单次 meta update 和本地 JSON tool bridge |

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

### 3.3 官方源码/API 适配

| 方法 | 构建接口 | 检索接口 | 主要配置 |
| --- | --- | --- | --- |
| `graphiti` | 官方 `Graphiti.add_episode`，每个 HABIT session 一个 episode | 官方 `Graphiti.search_` | Kuzu；edge cosine；RRF；top-k 5；BGE-M3；16,384-token 官方 extraction completion 上限；本地 schema 最多 64 items/1,000 chars |
| `secom` | 官方 segmentation + LLMLingua2 compression，按 session 增量写入 | 官方 FAISS retriever | segment granularity；compression rate 0.9；top-k 5；BGE-M3 |
| `omem` | 官方 `SimpleMemory.add_message`、working/episodic/persona lifecycle | `retrieve_from_memory_soft_segmentation` | top-n 12；drop threshold 0；BGE-M3；本地 JSON-output compatibility |

对应配置：

```text
configs/methods/
├── graphiti_bge_m3_qwen3.yaml
├── secom_bge_m3_qwen3.yaml
└── omem_bge_m3_qwen3.yaml
```

Graphiti 使用环境中固定的 `graphiti-core==0.29.2`。SeCom 和 O-Mem 是普通源码
快照，不是 Git submodule，也不含嵌套 `.git`：

```text
third_party/official-baselines/vendor/SeCom
third_party/official-baselines/vendor/O-Mem
```

固定 revision 和兼容边界见 `third_party/official-baselines/README.md`。

### 3.4 `no_memory`

`no_memory` 给 Qwen 的内容只有：

```text
current request + response choices
```

不提供历史、不提供检索结果，也不调用 embedding model。它用于测量数据本身的
history-free shortcut 和 base-model prior。

### 3.5 `full_memory`：容量感知的长上下文对照

`full_memory` 不训练参数、不构建向量库、不做检索，也不使用额外摘要模型。它直接把
probe cutoff 之前的 public lifelong history 交给相同的 answer model。

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

history 选择算法：

1. 仅收集 `session_index <= probe cutoff` 的 sessions。
2. 使用实际 base-model tokenizer 统计完整 history token 数。
3. 如果完整 history 不超过所选档位的 history budget，输入全部 history。
4. 如果超窗，从最新 session 向前选择尽可能多的完整 sessions。
5. 选中的 suffix 仍按时间正序交给模型。
6. 只有当最新一个 session 自身就超过全部 history budget 时，才在该 session
   内部保留 header 和最新 token tail。
7. answerer 仍执行一次最终 tokenizer hard-bound 检查。

这是一种明确、可审计的 recency/session-boundary truncation。它不会引入另一个
summary LLM，从而避免把“长上下文对照”变成新的 memory 方法。

选择固定档位：

```bash
HABITBENCH_CONTEXT_WINDOW_TIER=32k \
bash scripts/submit_clusterx.sh \
  --methods full_memory \
  --datasets food,finance_software \
  --shards 8 --gpus 8 \
  --output-root results/full_memory_32k
```

只有在模型 checkpoint 和服务端确实支持更大窗口时才能选择更大档位：

```bash
HABITBENCH_CONTEXT_WINDOW_TIER=64k \
HABITBENCH_MAX_MODEL_LEN=65536 \
bash scripts/submit_clusterx.sh \
  --methods full_memory \
  --datasets food,finance_software \
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
│       ├── graphiti_bge_m3_qwen3.yaml
│       ├── secom_bge_m3_qwen3.yaml
│       └── omem_bge_m3_qwen3.yaml
├── domain/
│   ├── food/
│   │   └── food_habit_lifelines_stress_v2/
│   └── finance-software/
│       └── habit_bench_multidogo_finance_software_scope_consistent_v1.3/
├── eval/
│   ├── context_windows.py
│   ├── controls.py
│   ├── methods.json
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
│   └── official_adapters/
│       ├── graphiti.py
│       ├── secom.py
│       └── omem.py
├── schema/
│   ├── session.schema.json
│   ├── probe.schema.json
│   └── memory_context.schema.json
├── scripts/
│   ├── run_eval.sh
│   ├── create_shard_plan.py
│   ├── run_multigpu_plan.py
│   ├── merge_shard_plan.py
│   ├── submit_clusterx.sh
│   └── cluster/
│       └── env.example.sh
├── tests/
│   └── evaluation/
├── third_party/
│   ├── medmemorybench/
│   └── official-baselines/
├── docs/
│   ├── evaluation_protocol.md
│   ├── multigpu_evaluation.md
│   └── medmemorybench/
├── research/
└── results/                         # 运行生成，Git ignored
```

关键文件职责：

| 文件 | 职责 |
| --- | --- |
| `eval/core/dataset.py` | 归一化两种数据格式、应用用户分片、构建无 gold 的 method payload |
| `eval/medmemorybench_adapters/structured_memory.py` | 七个 MedMemoryBench-source 方法的统一增量 ingestion/retrieval 入口 |
| `eval/official_adapters/*.py` | Graphiti、SeCom、O-Mem 的薄适配层 |
| `eval/controls.py` | `no_memory`、`full_memory` 和 `full_history` |
| `eval/context_windows.py` | 长上下文档位解析与验证 |
| `eval/core/answering.py` | 固定 Qwen answer prompt、token hard bound、choice JSON 解析 |
| `eval/run.py` | 单方法/单数据集/单分片端到端运行 |
| `eval/core/scoring.py` | Accuracy、Wilson CI、能力分组和统一结果输出 |
| `eval/core/retrieval_scoring.py` | 分域 evidence、chain、decoy、provenance 和 decision-unit 指标 |
| `eval/merge_shards.py` | 严格验证并合并某个方法/领域的全部分片 |
| `scripts/create_shard_plan.py` | 生成确定性 method × dataset × shard 任务表及配置快照 |
| `scripts/run_multigpu_plan.py` | 每 GPU 一个持久 vLLM worker，调度分片任务并记录 wall-clock |
| `scripts/merge_shard_plan.py` | 合并所有 method/domain group，生成总览 |
| `scripts/submit_clusterx.sh` | 唯一 ClusterX 提交入口 |

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

MemRL 经由其原生 MemOS reader 使用 `chonkie==1.2.1` 和 GPT-2 tokenizer。正式
launcher 会在启动 vLLM 和执行长任务之前检查该包版本，以及
`TIKTOKEN_CACHE_DIR` 中 GPT-2 的 vocab/encoder 缓存；缺失时 preflight 立即失败，
不会等到运行数小时后才在 MemRL shard 初始化时暴露。

Vendored Letta 和 MIRIX 的 Python import graph 会在创建本地数据库时加载一批即使
文本评测路径不直接调用、但仍属于必需的运行时依赖。`requirements.txt` 已固定当前
验证通过的 Letta/MIRIX 兼容版本；launcher 会在占用 GPU 前检查这些关键包。当前环境
已实际完成 Letta SQLite 与 MIRIX SQLite 的 `AgentManager` 初始化，而不只做静态
`import` 检查。

常用环境变量：

| 变量 | 默认值/用途 |
| --- | --- |
| `HABITBENCH_VLLM_PYTHON` | 独立 vLLM 0.17.1 环境；不复用 method Python |
| `HABITBENCH_LLM_MODEL` | Qwen checkpoint 路径 |
| `HABITBENCH_SERVED_MODEL` | OpenAI-compatible served name |
| `HABITBENCH_MAX_MODEL_LEN` | vLLM 服务端实际支持的总窗口 |
| `HABITBENCH_CONTEXT_WINDOW_TIER` | `full_memory` 的 `auto` 或显式档位 |
| `HABITBENCH_MAX_INPUT_TOKENS` | `custom` 档位的 answer 输入上限 |
| `HABITBENCH_EMBED_MODEL` | BGE-M3 本地路径 |
| `HABITBENCH_EMBED_DEVICE` | embedding 默认设备为 CPU；MIRIX/SeCom 按原生配置强制绑定其 worker GPU |
| `HABITBENCH_<METHOD>_USER_WORKERS` | 方法级用户并发：Mem0/A-MEM/MemOS/MemRL/Letta=7、LightMem/MIRIX=1 |
| `HABITBENCH_ADAPTER_CPU_THREADS` | 每个 adapter 进程的 BLAS/OpenMP 线程数，默认 2 |
| `HABITBENCH_LIGHTMEM_MODEL` | LightMem/LLMLingua2 模型路径 |
| `HABITBENCH_GRAPHITI_LLM_MAX_TOKENS` | Graphiti entity/edge extraction completion 上限，默认对齐官方 16,384 |
| `HABITBENCH_GRAPHITI_SCHEMA_MAX_ITEMS` | 本地 constrained decoding 的单数组上限，默认 64，防止重复枚举 |
| `HABITBENCH_GRAPHITI_SCHEMA_MAX_STRING_CHARS` | Graphiti schema 单字符串上限，默认对齐官方 summary 的 1,000 chars |
| `HABITBENCH_GPU_MEMORY_UTIL` | vLLM GPU memory utilization |
| `HABITBENCH_ENABLE_PREFIX_CACHING` | 默认 1；复用同用户长历史 prompt prefix |
| `HABITBENCH_VLLM_MIN_TOKENS_PER_SEC` | 启动后并发聚合 decode 吞吐门槛，默认 60 |
| `HABITBENCH_VLLM_BENCHMARK_CONCURRENCY` | 吞吐门禁的代表性并发数，默认 4 |
| `HABITBENCH_FORCE_RERUN` | 设为 1 时重跑已有完整 shard |

## 6. 运行方式

### 6.1 数据和单元测试

```bash
cd /plm-shared/zhangjunming/Workspace/HABIT-bench

/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark/bin/python \
  -m eval.validate \
  domain/food/food_habit_lifelines_stress_v2 \
  domain/finance-software/habit_bench_multidogo_finance_software_scope_consistent_v1.3

PYTHONDONTWRITEBYTECODE=1 \
/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark/bin/python \
  -m unittest discover -s tests/evaluation -p 'test_*.py'
```

### 6.2 Prepare-only

Prepare-only 验证数据加载、配置 snapshot 和 manifest，不调用 memory 方法或模型：

```bash
bash scripts/run_eval.sh mem0 \
  domain/food/food_habit_lifelines_stress_v2 \
  /plm-shared/zhangjunming/tmp/habit_mem0_prepare \
  --max-users 1 --max-probes 4 --prepare-only
```

### 6.3 ClusterX 正式评测

默认运行十个 memory 方法，不自动包含 control：

```bash
bash scripts/submit_clusterx.sh \
  --datasets food,finance_software \
  --shards 8 \
  --gpus 8 \
  --output-root results/habit_all_methods_v1
```

只运行主要 MedMemoryBench-source suite：

```bash
bash scripts/submit_clusterx.sh \
  --methods mem0,amem,memos,memrl,lightmem,letta,mirix \
  --datasets food,finance_software \
  --shards 8 --gpus 8 \
  --output-root results/habit_medmemorybench_v1
```

运行 control：

```bash
bash scripts/submit_clusterx.sh \
  --methods no_memory,full_memory \
  --datasets food,finance_software \
  --shards 8 --gpus 8 \
  --output-root results/habit_controls_v1
```

运行新增官方适配：

```bash
bash scripts/submit_clusterx.sh \
  --methods graphiti,secom,omem \
  --datasets food,finance_software \
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

相同 `output-root` 可用于断点恢复；已有 `metrics.json` 的 shard 会跳过。
只有明确希望覆盖完整 shard 时才设置 `HABITBENCH_FORCE_RERUN=1`。

## 7. 结果目录结构

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
└── finance_software/
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

### 7.1 每个 shard

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
| `graphiti` | `graphiti_config.json`、`graphiti_runtime.json`、`graphiti_kuzu_store/` |
| `secom` | `secom_method_config.json`、`secom_config.json`、`secom_runtime.json` |
| `omem` | `omem_config.json`、`omem_runtime.json`、`omem_store/` |

### 7.2 合并结果

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

## 8. 指标和 timing

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

其中 Food 的 positive gold 是 `gold_evidence_session_ids`，Finance/Software 的
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

Food v2 的能力 panel：

| Panel | 主要指标 |
| --- | --- |
| `habit_induction` | `direct_use` Accuracy、5 条弱证据 Recall/nDCG、grounded joint success |
| `explicit_history_retrieval` | 显式检索 Accuracy 和 evidence Recall@5 |
| `boundary_calibration` | Accuracy、`false_personalization_cost = 1 - Accuracy`、boundary evidence Hit@5 |
| `exception_retention` | Accuracy、`exception_failure_rate`、exception evidence Hit@5 |
| `unseen_paraphrase_robustness` | 未见表达下的 Accuracy 与 evidence metrics |

Finance/Software v1.3 额外报告：

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
| `decision_unit_macro_accuracy` | 449 个潜在决策等权的答案准确率 |
| `decision_unit_macro_evidence_recall_at_5` | 449 个潜在决策等权的组件证据 Recall@5 |
| `decision_bundle_macro_*` | 多习惯 decision bundle 等权结果 |

Finance 的 640 个 probe 有 6 条决定性证据，因此原始 Recall@5 理论上限为 5/6；
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

## 9. 复现与结果命名

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
配置的逐项精确复现。Graphiti、SeCom、O-Mem 应描述为 “official-source/API adapted”，
并同时说明本地 backend、embedding 和 JSON compatibility 设置。

更严格的 protocol 和 ClusterX 说明见：

```text
docs/evaluation_protocol.md
docs/multigpu_evaluation.md
docs/medmemorybench/README.md
```
