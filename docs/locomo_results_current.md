# LoCoMo 横向评测与当前结果（Qwen3-8B）

> 本文合并 LoCoMo 的评测协议、提交说明和当前结果；当前正式结果可用于报告和复现。

## 评测口径与统一模型

仓库通过 `third_party/medmemorybench` 保留的 LoCoMo evaluator 运行官方
`locomo10.json`。本仓库不把数据集提交进 Git；H 集群上的正式副本位于：

```text
/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/datasets/locomo/locomo10.json
```

提交器会把数据 SHA-256、Qwen3-8B 模型身份和 BGE-M3 身份写入
`locomo_plan.manifest.json`。

每个任务是一个独立的“方法 × LoCoMo conversation sample”，方法内部仍按
LoCoMo 原生时间顺序构建和检索记忆。当前正式计划包含：

```text
long_context, bm25_rag, embedding_rag,
mem0, amem, memos, memrl, lightmem, letta, mirix
```

前三项是上下文、词法 BM25 和语义 BGE-M3 对照；后七项是当前活动的
MedMemoryBench memory-agent 方法。所有方法的 reader/memory LLM 均使用同一
本地 Qwen3-8B（H200 上每张卡一个 vLLM server），embedding 方法统一使用
`BAAI/bge-m3`、1024 维固定 revision。方法侧不读取 LoCoMo gold evidence。

官方 LoCoMo 指标由 vendored `scripts/recompute_locomo_official.py` 重新计算，
主指标为官方 token-overlap F1 的均值；adversarial 类按官方
“not mentioned / no information available”规则处理。任务目录中同时保留
原始 evaluator report、query answers、memory-build 日志和 `locomo_official.json`。

## 断点、日志和提交

```bash
python scripts/submit_h_locomo.py \
  --creator-ad linzhouhan \
  --job-name zjm-locomo-q8b-v1
```

提交的是 2 个 Replica × 每个 Replica 8 张 H200（总计 16 卡）的 reserved/P5
RJob。`create_locomo_plan.py` 生成 100 个方法/样本任务；GPFS 原子目录队列
在两个 Replica 间动态分配任务。每个任务完成后写入
`<output_root>/<method>/<sample_id>/locomo_task_result.json`，重新提交同一
计划会跳过这些成功 marker，只重跑缺失或失败任务。重试会写入 `attempt-2/`，
不覆盖第一次运行的 artifacts。

实时日志位于：

```text
<output_root>/h_locomo_logs/<RJOB_ID>/
<output_root>/locomo_vllm_logs/<RJOB_ID>/
```

RJob worker 会先校验 8 张 GPU 都是 H200，再启动 vLLM 和吞吐 gate；任意任务
仍失败时全局 summary 标记为 failed，便于修复后沿用同一输出路径继续。

## 完成状态

LoCoMo 对比实验已完整完成，当前结果可用于报告和后续分析：

| 项目 | 值 |
| --- | --- |
| 状态 | **succeeded** |
| RJob | `zjm-locomo-q8b-retry2-32228054` |
| 资源 | 2 个 Replica × 8 张 H200（总计 16 卡） |
| 模型 | Qwen3-8B（所有方法统一 reader/memory LLM） |
| 数据集 | 官方 LoCoMo `locomo10.json`，10 个 conversation sample |
| QA 数量 | 1,986 |
| 任务数 | 100/100（10 方法 × 10 sample） |
| 成功/失败 | 100/0 |
| 完成时间 | 2026-08-28 00:19 HKT（RJob 状态观测） |

本次重提沿用原输出目录；90 个已有成功任务通过断点 marker 跳过，补跑此前
`embedding_rag` 失败的 10 个任务后全部成功。结果汇总文件为：

```text
results/habit-h200-locomo-qwen3-8b-v1/locomo_suite_summary.json
```

每个方法的单样本结果和官方后处理文件位于：

```text
results/habit-h200-locomo-qwen3-8b-v1/<method>/conv-<id>/
```

## 主指标

`Official mean F1` 是官方 LoCoMo token-overlap F1 在全部 1,986 个问题上的均值。
`Threshold accuracy` 是官方 F1 ≥ 0.5 的问题比例；两者均不使用 gold evidence
作为方法输入。

| 方法 | 类型 | 成功任务 | 查询数 | Official mean F1 | Threshold accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `letta` | memory agent | 10/10 | 1,986 | **0.5137** | 0.5242 |
| `amem` | memory agent | 10/10 | 1,986 | **0.5130** | 0.5227 |
| `long_context` | full-context baseline | 10/10 | 1,986 | 0.5110 | **0.5312** |
| `bm25_rag` | lexical retrieval baseline | 10/10 | 1,986 | 0.5011 | 0.5151 |
| `embedding_rag` | BGE-M3 semantic retrieval baseline | 10/10 | 1,986 | 0.4940 | 0.5086 |
| `memos` | memory agent | 10/10 | 1,986 | 0.3883 | 0.3872 |
| `mem0` | memory agent | 10/10 | 1,986 | 0.3614 | 0.3580 |
| `memrl` | memory agent | 10/10 | 1,986 | 0.3024 | 0.2875 |
| `lightmem` | memory agent | 10/10 | 1,986 | 0.2984 | 0.2910 |
| `mirix` | memory agent | 10/10 | 1,986 | 0.2662 | 0.2588 |

## 按问题类型的 Official F1

下表是在 10 个 sample 上按问题类型合并后的官方 F1 均值。类型问题数为：
adversarial 446、multi-hop 282、open-domain 96、single-hop 841、temporal 321。

| 方法 | Adversarial | Multi-hop | Open-domain | Single-hop | Temporal |
| --- | ---: | ---: | ---: | ---: | ---: |
| `letta` | 0.7309 | 0.4731 | 0.1927 | 0.5213 | 0.3234 |
| `amem` | 0.6906 | 0.4332 | 0.2088 | 0.5405 | 0.3552 |
| `long_context` | 0.6076 | 0.4709 | 0.2256 | 0.5609 | 0.3668 |
| `bm25_rag` | 0.6278 | 0.4393 | 0.2053 | 0.5499 | 0.3397 |
| `embedding_rag` | 0.6233 | 0.4569 | 0.1985 | 0.5302 | 0.3403 |
| `memos` | 0.9776 | 0.2428 | 0.1805 | 0.2071 | 0.2345 |
| `mem0` | 0.9776 | 0.1819 | 0.1619 | 0.1939 | 0.1617 |
| `memrl` | 0.9933 | 0.1463 | 0.1270 | 0.0736 | 0.1315 |
| `lightmem` | 0.9776 | 0.1443 | 0.1593 | 0.1119 | 0.0202 |
| `mirix` | 0.9910 | 0.0797 | 0.1299 | 0.0439 | 0.0463 |

## 复现与结果文件

- 数据为官方 LoCoMo 10-sample 版本；每个方法独立按对话时间顺序构建/检索记忆。
- 所有方法使用相同的 Qwen3-8B reader/memory LLM；`embedding_rag` 使用统一的
  BGE-M3（1024 维）。
- 官方指标由仓库中的 LoCoMo evaluator 后处理脚本重新计算，主指标为
  `mean_official_score`（即官方 token-overlap F1 均值）。
- 运行日志位于：

  ```text
  results/habit-h200-locomo-qwen3-8b-v1/h_locomo_logs/zjm-locomo-q8b-retry2-32228054/
  results/habit-h200-locomo-qwen3-8b-v1/locomo_vllm_logs/zjm-locomo-q8b-retry2-32228054/
  ```
