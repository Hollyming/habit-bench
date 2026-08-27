# Non-agentic session retrieval baselines

这些方法用于回答一个独立于复杂 memory lifecycle 的问题：如果不做 LLM 记忆抽取、
摘要、反思、状态更新或 agentic tool loop，只在 cutoff 前的原始完整 sessions 上做标准
检索，同一个 answer model 能达到什么水平？

所有方法共享以下边界：

- retrieval unit 始终是一个完整 session；不切 message、不切 chunk；
- 只读取 public probe 的 `query`，不把 choices 拼入 retrieval query；
- 候选 corpus 严格限制为该用户 `max_session_index` cutoff 前的 sessions；
- 返回原始 session，包含精确 `SESSION_ID`、`SESSION_INDEX` 和 `TIMESTAMP`；
- `evidence_session_ids` 按 retrieval rank 排列，供统一 Recall@5/MRR/NDCG scorer 使用；
- choice 仍由主实验共享的 Qwen answer head 产生。

实现入口是 `eval/retrieval_baselines.py`，固定参数保存在
`configs/methods/*.yaml`，协议 revision 为 `session_retrieval.v1`。

## Recency-k

正式注册两个设置：

- `recency_5`：最近 5 个完整 sessions；
- `recency_10`：最近 10 个完整 sessions。

排序为 newest-first；可见历史不足 k 个时返回全部可见 sessions。它与
`full_history` 不同：`full_history` 受 token window 控制，尽可能容纳最近完整历史，
必要时还可能保留一个超长 session 的 token tail；Recency-k 的选择只由固定 session
数量决定，绝不截断 session。

## BM25-RAG

`bm25_rag` 使用 `rank-bm25==0.2.2` 的 BM25Okapi：

- 每个完整 session 是一个 document；
- indexed text 只包含带 role 的消息正文，避免 session ID 自身影响词项匹配；
- tokenizer 是确定性的 Unicode word regex + Unicode casefold；
- IDF corpus 每次只由 cutoff-visible sessions 构成，未来 sessions 不会改变词权重；
- 返回 BM25 top-5。

## Dense-RAG

`dense_rag` 使用主实验统一的本地 BGE-M3：

- model：`BAAI/bge-m3`；
- revision：`5617a9f61b028005a4858fdac845db406aefb181`；
- 每个完整 session 的消息正文独立编码；
- public query 独立编码；
- document/query embeddings 做 L2 normalization 后，以 cosine 排序；
- 返回 top-5，不生成摘要或中间 memory object。

同一用户的 session embedding 只在首次进入某个 probe 的可见集合时编码，之后可按
session ID 做 query-independent 缓存；cutoff 后 session 不会提前进入 encoder 或候选集合。
H 集群默认把该 encoder 放在 CPU adapter 侧，避免与每卡常驻的 vLLM answer worker
争用 H200 显存；模型身份和挂载仍在提交前严格校验。

## Temporal Hybrid-RAG

`temporal_hybrid_rag` 首先分别计算 BM25 rank 和 Dense rank，再计算：

\[
s_i = \frac{1}{60+r_i^{BM25}} + \frac{1}{60+r_i^{dense}}
      + \lambda\,\alpha(q)\,s_i^{time},
\qquad \lambda=0.02.
\]

rank 从 1 开始。时间部分有两种语义：

1. query 含可解析的 `as-of` 历史时间时，target 取 query 中的历史时间。
   target 之前的 session 按 90 天 half-life 衰减；target 之后的 session 额外乘
   `0.05`，避免把后来状态当成历史时点状态。
2. query 没有显式历史时间时，target 取 public probe timestamp；若缺失则取最近可见
   session timestamp。此时使用较弱 recency prior，`α(q)=0.25`，时间 half-life 为
   180 天；时间戳不可解析时退化为按 session distance、20 sessions half-life。

因此 Hybrid 并非无条件“越新越好”：明确询问历史状态时，它围绕历史 target 排序；
只有 current/无时间 query 才加入较弱的 recency prior。每个返回结果的 BM25 score/rank、
dense cosine/rank、RRF、time score、time relation 和 final score 都写入 `debug`，便于审计。

## Qwen3-8B 四域实验结果

以下是 2026-08-26 完成的正式实验结果。五种方法共享同一个 Qwen3-8B answer model，
覆盖 Food v5、Finance/Software v1.4 和 Travel v16。每个“方法 × 域”使用 16 个 user
shards，共 320 个任务；所有任务均已完成，没有失败或缺失 shard。

- RJob：`hb-q8b-retrieval-v1-10959712`；
- 资源：2 replicas × 8 H200，共 16 卡；
- RJob 状态：`Succeeded`，2/2 replicas 成功；
- 运行时间：4376.365 秒（约 72 分 56 秒）；
- 原始汇总：`results/habit-h200-retrieval-baselines-qwen3-8b-v1/evaluation_summary.json`。

### Answer accuracy

`Overall` 是四域全部 4168 个 probes 的 micro accuracy，而不是四个域准确率的简单平均。

| Method | Food v5 (n=1470) | Finance v1.4 (n=1368) | Software v1.4 (n=680) | Travel v16 (n=650) | Overall (n=4168) | Correct |
|---|---:|---:|---:|---:|---:|---:|
| Recency-5 | 33.74% | **23.03%** | 25.88% | 26.15% | 27.76% | 1157/4168 |
| Recency-10 | 35.99% | 22.81% | 26.32% | 29.08% | 29.01% | 1209/4168 |
| BM25-RAG | 49.25% | 22.95% | 24.41% | **32.46%** | 33.95% | 1415/4168 |
| Dense-RAG | **51.16%** | 22.73% | 25.44% | **32.46%** | **34.72%** | **1447/4168** |
| Temporal Hybrid-RAG | 49.46% | 21.05% | **26.32%** | 32.31% | 33.69% | 1404/4168 |

### Retrieval and grounding metrics

下表按四域 retrieval-evaluable probes 加权汇总。所有指标均在 top-5 上计算；
`Joint answer + hit` 要求 answer 正确且至少命中一个 oracle evidence session。

| Method | Evidence hit | Evidence recall (macro) | Precision | MRR | NDCG | Full evidence | Joint answer + hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recency-5 | 24.42% | 6.37% | 7.80% | 10.45% | 7.42% | 0.02% | 6.96% |
| Recency-10 | 24.42% | 6.37% | 7.80% | 10.45% | 7.42% | 0.02% | 7.25% |
| BM25-RAG | 61.95% | 17.28% | 20.76% | 41.14% | 24.45% | 1.80% | 25.41% |
| Dense-RAG | 47.36% | 13.39% | 16.12% | 31.58% | 19.52% | **2.64%** | 22.77% |
| Temporal Hybrid-RAG | **64.32%** | **17.71%** | **21.43%** | **46.20%** | **26.00%** | 1.87% | **25.89%** |

### 结果解读

- Dense-RAG 的总体 answer accuracy 最高（34.72%），比 BM25-RAG 高 0.77 个百分点；
- Temporal Hybrid-RAG 在 Evidence Hit、Recall、Precision、MRR、NDCG 和 joint 指标上最好，
  但 answer accuracy 并未同步超过 Dense-RAG，说明更强的 oracle-evidence 排名未被 answer
  model 完全转化为最终答题收益；
- Recency-10 比 Recency-5 的总体 accuracy 高 1.25 个百分点，表明额外近期上下文有一定价值；
- Recency-5 与 Recency-10 的 top-5 retrieval 指标相同是预期行为：两者前五个 session
  完全相同，统一 scorer 会把 Recency-10 的 ranked attribution 截到前五，但 answer model
  实际分别读取 5 和 10 个完整 sessions。

## 运行

单个方法可复用统一入口：

```bash
bash scripts/run_eval.sh bm25_rag DATASET_DIR OUTPUT_DIR [eval args]
bash scripts/run_eval.sh dense_rag DATASET_DIR OUTPUT_DIR [eval args]
bash scripts/run_eval.sh temporal_hybrid_rag DATASET_DIR OUTPUT_DIR [eval args]
bash scripts/run_eval.sh recency_5 DATASET_DIR OUTPUT_DIR [eval args]
bash scripts/run_eval.sh recency_10 DATASET_DIR OUTPUT_DIR [eval args]
```

通用 shard planner 和 H launcher 已注册五个方法。新的 H 默认主计划包含它们；已有的
持久化计划不会被静默改写，resume 校验会拒绝 method list 不一致的旧计划。
