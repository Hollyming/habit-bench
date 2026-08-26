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
