# MedMemoryBench 方法接入

## 当前结构

本项目现在是单一 Git 仓库。MedMemoryBench 固定源码位于：

```text
third_party/medmemorybench
```

该目录不包含 `.git`，来源 commit、LightMem revision 和本地补丁记录在
`third_party/medmemorybench/VENDOR_INFO.md`。

活动方法为：

```text
mem0
amem
memos
memrl
lightmem
letta
mirix
```

这些名称直接表示 MedMemoryBench 源码版本，不再使用
`medmemorybench_mem0` 一类重复前缀，也不与另一套同名 adapter 并存。

## 统一执行链路

`eval.medmemorybench_adapters.structured_memory` 对每个 HABIT 用户创建独立
的 MedMemoryBench `AgentManager`，按 probe cutoff 增量写入 public
sessions。写入调用方法的 `memorize()`，检索调用方法的 `retrieve()`。

`retrieve()` 只返回原生 memory context，不调用方法自己的 reader。随后由
HABIT 的固定 Qwen3-8B answerer 根据：

```text
memory_context + current request + choices
```

生成一个 `choice_id`，最后由 private scorer 计算 Accuracy。

## 用户级并发

并发单位是完整用户，而不是 session。正式配置按 WJR 完整 lifeline
结果冻结：

| 方法 | 最大用户 workers |
| --- | ---: |
| Mem0 / MemOS / MemRL / Letta | 7 |
| A-MEM | 5 |
| LightMem / MIRIX | 1 |

MIRIX 的低并发不代表缩小 completion budget：memory-child 固定使用 WJR
q8a20 已验证的 8,192 tokens，并继续严格传播内部写入失败。

food 每分片最多 4 个用户，所以前五种方法的有效 worker 数会自动降到
3–4；finance-software 每分片最多 7 个用户。WJR 在 6-user shard 上验证
的 6 workers 在这里扩展为 7，避免第七个用户形成完整 540-session 的
串行尾部；A-MEM 仍保留其 finance 完整实验验证过的 5 workers。运行约束如下：

- 同一用户的 session 和 probe cutoff 始终按时间顺序执行；
- 不同用户由独立 spawn 进程并行；
- 每个用户拥有独立的 storage root、persistence root 和 runtime HOME；
- 输出最终按原始 probe 顺序重排；
- vLLM 使用 batch-invariant，避免请求批次组成改变结构化 memory action。

这只改变调度，不改变七种方法的 `memorize()`、`retrieve()`、memory
表示、top-k 或 prompt。`medmemorybench_adapter_runtime.json` 记录请求/
有效 worker 数、每个用户的 elapsed/CPU/max-RSS、聚合 user time 和
sessions/s。

## 方法与配置

七个方法共享以下实验口径：

- memory/answer backbone：本地 Qwen3-8B；
- embedding：本地 BGE-M3（1024 维，固定 revision
  `5617a9f61b028005a4858fdac845db406aefb181`）；
- nominal retrieval：top-k 5；
- 用户级状态隔离；
- memory build partial failure 直接失败；
- gold label、habit graph 和 gold evidence 不进入方法输入。

各方法保留不同的原生 memory 表示和检索长度，因此主表是端到端
method-native retrieval 比较，不是等 evidence-token ablation。

## 验收要求

正式结果必须同时包含：

- 仓库、vendor、模型和数据 revision/hash；
- 完整 user/probe coverage；
- 逐题 `memory_context`、prediction 和 score；
- strict shard merge；
- 失败扫描和资源记录；
- 对本地模型兼容修改的明确说明。

早期原型结果和预发布数据集摘要不再随当前仓库分发，也不能与正式运行直接合并；
当前四域结果请以 `docs/main_experiment_results_current.md` 和
`docs/supplementary_results_current.md` 为准。
