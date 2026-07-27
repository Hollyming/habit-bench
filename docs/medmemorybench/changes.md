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
- MemOS、MemRL、Letta：隔离持久状态，并停止隐藏写入或检索错误。
- MemRL：把方法 YAML 声明的 BGE-M3 `embedding.dim=1024` 显式传到原生
  MemOS/Qdrant reader，避免用模型路径字符串误猜为 768；本地 cube ID 增加微秒与
  随机后缀，避免多 shard 的局部 context ID 在 7-worker 并发初始化时撞 SQLite
  唯一键。这两项只修复本地 backend 配置与身份隔离，不改变 proceduralization、
  retrieval 或 Q-learning 策略。
- MIRIX：补充 SQLite cosine、embedding 维度处理、engine reset、bounded local JSON tool schema、canonical tool conversion、stale replace-ID 归一化，以及 bounded memory update 所需的 completion budget。

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
