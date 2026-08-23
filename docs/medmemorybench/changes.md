# 修改内容概览

## MedMemoryBench

`wjr` 分支增加了 retrieval-only 接口，同时保留各方法原有实现：

- `BaseAgent` 和 `AgentManager` 将结构化检索结果与方法原生回答生成解耦；
- Mem0 和 A-MEM 直接导入仓库中的 vendored 源码；
- MemOS、MemRL、LightMem、Letta 和 MIRIX 通过统一接口暴露原生检索结果；
- 持久状态按 persona、user、context 和 task 隔离；
- memory build 失败会直接向上抛出，不再伪装成成功的空结果；
- 增加 persona 分片、checkpoint、strict merge 和 coverage 校验；
- LoCoMo 支持统一 schema-constrained common reader 重评分。

各方法的重要修复：

- Mem0：固定本地 vendored import、按任务隔离 Qdrant、原子归一化重复 mutation，并严格处理结构化输出失败。
- A-MEM：保留调用方指定的本地 embedding 路径，并暴露原生 note retrieval。
- LightMem：在关闭 pre-compression 时初始化可选 compressor；裁正越界 source ID 后再查询 timestamp 和 speaker。LightMem 是外部子模块，因此两处修改记录在 `patches/lightmem-medmemorybench.patch`。
- MemOS、MemRL、Letta：隔离持久状态，并停止隐藏写入或检索错误。MemOS
  `general_text` 还会在构造 Pydantic memory item 前丢弃 LLM 返回的非列表
  `tags`，避免单条格式漂移终止整个用户 shard。
- MemRL：把方法 YAML 声明的 BGE-M3 `embedding.dim=1024` 显式传到原生
  MemOS/Qdrant reader，避免用模型路径字符串误猜为 768；本地 cube ID 增加微秒与
  随机后缀，避免多 shard 的局部 context ID 在 7-worker 并发初始化时撞 SQLite
  唯一键。这两项只修复本地 backend 配置与身份隔离，不改变 proceduralization、
  retrieval 或 Q-learning 策略。
- MIRIX：补充 SQLite cosine、embedding 维度处理、engine reset、two-stage bounded local JSON tool bridge、canonical tool conversion、stale replace-ID 归一化，以及 bounded memory update 所需的 completion budget。JSON bridge 先用极小 schema 选择一个工具，再只生成该工具的精确参数 schema，并保留官方通用 `finish_memory_update`。正式配置使用 `memory_agent_max_tokens=8192`，vLLM xgrammar 同时设置 `disable_any_whitespace=true` 与 `disable_fallback=true`；前者禁止字段间的无界空白，后者禁止约束静默退化。vLLM 保留官方建议的 Hermes tool-call parser，但在 no-thinking 协议下不启用 reasoning parser，避免 constrained JSON 被移出 `message.content`。若 serving 仍提前结束 JSON，bridge 会复用 MIRIX 官方的 `json.loads` / `demjson3` / `json-repair` 解析链，随后再次校验严格 schema；只有 parser 已修复的截断对象且声明必填字段齐全时，才把对象投影回 schema 并丢弃额外 repair artifact，以覆盖官方 issue #103 中的 `tree_path` 漂移。若截断发生在对象数组的下一项中，则仅在已有非空合法前缀且缩短后仍满足数组 schema 时，丢弃 parser 补出的残缺尾项。完整违规 JSON、单独残缺项、非法前缀和错类型仍然失败。纠错生成使用可复现但逐次变化的 seed 和轻量采样。日志仅输出 finish reason、长度、token 数、哈希、解析位置以及被丢弃字段/尾项数量，不泄露 memory 正文。每个 child 内的 selector/arguments 顺序执行，但官方 MIRIX 的多个 memory child 仍保持并发；这些适配不改变 tool validator、executor、storage 或 memory lifecycle。

2026-07-27 的 Food-v2 定点回归复现了旧 profile 的第 38 条
`apple butter spice cake`：旧 episodic child 运行 341 秒后
`finish_reason=length`；compact xgrammar profile 的同一 child 在 8 秒内闭合，
整个 lifecycle 14 秒完成。该点之前 38 个 lifecycle 均为 0 retry、0 error。
此 gate 是运行时正确性验收，不作为 benchmark 效果结果。

q8a18、q8a19 和 q8a20 是运行源码快照，不是长期源码分支。q8a20 是累计验证版本；最终只把有意义的文件级差异合并进 `wjr` 历史。运行日志、cache、数据库、build 目录和复制的 `.git` 元数据不会进入 Git。

## HABIT-Bench

本次接入增加：

- `eval/methods.json` 中的 7 个规范方法名：
  `mem0`、`amem`、`memos`、`memrl`、`lightmem`、`letta`、`mirix`；
- `eval.medmemorybench_adapters.structured_memory` 增量写入与 retrieval-only adapter；
- 对非法、partial 或 `success=false` memory write 的严格拒绝；
- `scripts/run_eval.sh` 中的七方法命令映射；
- session marker、dry-run contract 和失败传播测试。

早期 `medmemorybench_*` 重复前缀与另一套 HABIT adapter 已退出活动仓库，
历史文件保存在仓库外的只读式迁移归档中。

该结果属于跨 Benchmark 源码接入。如果 backbone、answer reader、judge 或 evidence budget 与论文不同，不应描述为论文 exact reproduction。

当前 HABIT 正式配置统一使用本地 BGE-M3（1024 维）替代早期 E5-base-v2
配置。改动位于七个 `*_qwen3-8b_adapted.yaml`，不改变方法的 memory
construction/retrieval 算法。BGE-M3 revision、完整 YAML 和 YAML SHA256 会写入
每个 shard 的 `run_manifest.json`。
