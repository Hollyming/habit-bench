# External API main evaluation

本入口用于在不改变 HABIT-Bench 数据、memory lifecycle 或 answer/scoring 协议的前提下，
把本地 Qwen serving 替换成 OpenAI-compatible 外部模型。默认正式 suite 同时评测：

- 模型：`deepseek-v4-pro-0813`、`glm-5.2`、`kimi-k3`；
- 数据：Food final、Finance v1.4、Software v1.4、Travel v16；
- 方法：`no_memory`、`full_memory`、`full_history`、`recency_5`、`recency_10`、
  `bm25_rag`、`dense_rag`、`temporal_hybrid_rag`、`mem0`、`amem`、`memos`、
  `memrl`、`lightmem`、`letta`、`mirix`、`secom`；
- 分片：每个 model × method × domain 使用 8 个持久化用户分片，共 1,536 个任务。

## 可比性边界

外部 API 模型是 answer head 以及需要 LLM 的 memory 写入/更新组件所实际调用的模型。
本地 Qwen3-8B checkpoint 不参与生成，只保留其 tokenizer，作为跨模型统一的 40k HABIT
输入预算计数器。这样方法之间看到的上下文容量不因厂商 tokenizer 差异而漂移；plan
manifest 会显式记录 provider、外部 served model 和本地路径的
`tokenizer_only_for_shared_context_budget` 角色。

每个模型单独拥有不可变 shard plan、输出目录、断点和最终合并结果。不同模型不共享
memory state。回答保持 temperature 0、固定 seed、JSON choice 输出，并统一关闭 thinking。
网关还会向 PJLab-compatible 请求注入 `chat_template_kwargs.enable_thinking=false`，避免
reasoning token 或 reasoning channel 改变结构化输出协议。

## 限流与重试

`eval/api_gateway.py` 是一个仅监听 `127.0.0.1` 的反向网关。整个 8 卡 RJob 只有一个网关
实例，因此所有模型、方法、进程和 GPU 共享同一滚动窗口限制：

- 60 requests/minute；
- 50,000,000 tokens/minute；
- 请求 token 使用“序列化字符数 + 最大 completion budget”的保守预留；
- 429 和 500/502/503/504 读取 provider `retry_after` 后全局冷却并重试；
- connect/read/write transport error 使用指数退避；
- adapter 只持有本地 dummy key，真实 key 只由网关进程读取。

网关不记录 prompt、response 或 Authorization header，只记录 model、状态码、延迟、重试、
request ID 和 provider usage 汇总。运行统计持续写入
`api_gateway/metrics.json`，实时日志写入 `api_gateway/gateway.log` 和 RJob stdout。

## 8-H200 分配

默认提交一个 Replica、8 张 H200，三个模型轨道同时运行：

| 模型 | 本地 worker/GPU | GPU ID |
| --- | ---: | --- |
| deepseek-v4-pro-0813 | 3 | 0,1,2 |
| glm-5.2 | 3 | 3,4,5 |
| kimi-k3 | 2 | 6,7 |

这些 GPU 不运行外部 LLM；它们用于 MIRIX、SeCom 等本地 CUDA 组件，并为各模型轨道提供
独立 shard worker。纯 Recency/BM25 主要占用 CPU，Dense/Hybrid 的默认公共 profile 仍按
冻结的 BGE-M3 设备策略执行。suite 的主要瓶颈是共享 60 RPM，所以 8 张卡不保证持续高
GPU utilization；这里按用户明确的 8 卡资源规格提交，3/3/2 用于避免某一模型独占队列。

## Credential

credential 必须放在仓库外、权限为 `600` 或更严格：

```text
OPENAI_API_KEY=<secret>
HABITBENCH_EXTERNAL_API_BASE_URL=https://example.invalid/v1
```

不要将 key 写入 shell launcher、Git、RJob command、plan metadata 或日志。提交器先通过
`/models` 验证三个精确 model ID，只输出 API origin 和 `credential=redacted`。

## 提交与恢复

H 集群个人低优任务示例：

```bash
bash scripts/submit_h_api_suite.sh \
  --job-type managed-spot \
  --creator-type user \
  --creator-ad your-user-ad \
  --credential-file /absolute/private/path/api.env \
  --output-root "$PWD/results/api-main-v1"
```

提交器固定核算 `8 GPUs/Replica × 1 Replica = 8 GPUs`，并按 H 集群最高规范设置 namespace、
priority、charged group、private machine、镜像、持久化挂载和新的 scheduler entry。重新使用
同一 `output-root` 时，完整 shard 会通过 manifest/config/hash 校验后跳过；失败或中断的
半成品 shard 会在持久锁内清理后重跑。限流网关本身无须 checkpoint，因为请求结果已经
落实在 shard 原子成功标记中。

## 输出结构

```text
results/api-main-v1/
├── api_suite_manifest.json
├── h_api_suite.log
├── h_rjob_submit.log
├── api_gateway/{gateway.log,metrics.json}
├── api_runner_logs/<model-slug>.log
├── api_runtime/suite_terminal.json
└── <model-slug>/
    ├── shard_plan.tsv
    ├── shard_plan.manifest.json
    ├── distributed_queue/
    ├── api_runtime/suite_runtime.json
    └── <method>/<domain>/
        ├── shard_000_of_008/ ... shard_007_of_008/
        └── merged metrics/results
```

每个 shard 的 `task.stdout.log` / `task.stderr.log` 提供实时且持久的进度；最终
`merge_shard_plan.py` 只合并通过完整性校验的成功 shard。
