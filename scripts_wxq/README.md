# `scripts_wxq`：当前唯一活动路线

本目录只保留 **Taskmaster-grounded planning defaults v0.4 数据生成**。
旧版 v0.1/v0.2/v0.3 生成脚本、旧评测适配器和失败修补脚本已经清除；
参考实现仍在项目的 `scripts/`、`runs/` 和 `eval/`，本次没有修改。

## 目录

```text
scripts_wxq/
├── README.md
├── 7.15.md                                      # 早期设计总结（保留不动）
├── HABIT-Bench 团队协作指南：构建长程 Agent-User Habit 数据集.pdf
├── V04_END_TO_END_GENERATION_CONTRACT.md        # v0.4 不可破坏的生成约束
└── pipeline_v04/
    ├── 01_prepare_taskmaster_sources.py
    ├── 02_discover_grounded_habits.py
    ├── 03_review_habit_pool.py
    ├── 04_generate_benchmark.py
    ├── 05_audit_release.py
    ├── 06_finalize_single_user.py
    ├── 07_reaudit_current_choice_layout.py
    ├── 08_generate_sessions_parallel.py
    ├── api_client.py
    ├── count_qwen_tokens.py
    └── maintenance_restore_user000_dossier.py
```

活动数据目录：

`runs_wxq/taskmaster_planning_defaults_v0_4`

已冻结、可先提交的单用户版本：

`runs_wxq/taskmaster_planning_defaults_v0_4/release_single_user_pilot`

## 整体流程

### 1. 准备真实来源

`01_prepare_taskmaster_sources.py` 从 v0.1 中下载保存的 Taskmaster-2
flights/hotels 原始对话构造完整 source cards。规则与 annotations 只帮助检索，
不能直接把 slot 值当成 habit。

### 2. GPT 端到端发现 grounded habits

`02_discover_grounded_habits.py` 让 GPT-5.5 xhigh 阅读完整真实对话，发现可复用、
有情境边界的具体 habit。每个候选必须引用至少 3 个独立 instruction 的原始证据。
family 只是检索和分类目录，不预先规定具体 habit 值。

### 3. 独立审核和去重

`03_review_habit_pool.py` 对候选做保留、收窄或拒绝，并做 family 内及跨 family
合并，产出 `private/curated_habit_pool.jsonl`。当前池为 13 条合格、无明显重叠的
grounded habit，不等于“每位用户固定 13 个 habit”。

### 4. 生成完整 benchmark

`04_generate_benchmark.py` 是主程序，分阶段运行：

1. `profiles`：从审核后的 pool 自由组合成多 habit 用户画像；每人 5–8 个具体
   habit，数量和 family 组合随 persona 变化。
2. `arcs`：GPT 根据完整画像生成跨月/跨年的自然旅行事件序列；此时不写对话，
   也不把 support/boundary/exception 标签告诉事件生成模型。
3. `sessions`：GPT 根据画像、事件、真实对话片段和连续性摘要直接写完整多轮
   session；没有固定话术模板，也没有“先模板后改写”。生成后由另一轮 xhigh
   调用进行 evidence 标注和质量验收。
4. `recovery`：另一个 GPT 只看最终历史、看不到隐藏 habit graph，重新归纳 habit；
   用它检查归纳、边界、例外、漂移和虚假个性化。
5. `probes`：只有 recovery 与长程证据门槛通过后，才生成困难多选题及私有标签。

`--only-users` 只用于 `arcs` 和 `sessions` 断点式扩展指定用户，并保留未选用户
已有的 arc、session 与版本历史。`recovery` 和 `probes` 必须等所有用户历史完成后
一次性全量运行，不能带 `--only-users`，否则会形成不完整的汇总或 probe 文件。

### 5. 审计与发布

`05_audit_release.py` 检查用户数、目标 session 数、public/private 对齐、Qwen3-8B
精确 token 长度、重复 session、probe/answer-key 一致性和选项位置分布。
完整多用户数据只有在 `--require-complete` 无 error 后才算 release candidate。

`06_finalize_single_user.py` 仅用于已经冻结的 user000 pilot：保持 lifeline 不变，
补齐并冻结 40 个 probes。它不是多用户主流程的必经步骤。

`07_reaudit_current_choice_layout.py` 在最终选项位置平衡之后运行。它先免费检查全量
语言、ID、证据引用和 manifest；只把明确引用 A/B/C/D、可能受重排影响的私有解释
送给 GPT-5.5 xhigh 改写。每题独立缓存，失败题单独重试，不改变公共题目或金标。

`08_generate_sessions_parallel.py` 只在不同用户之间并行；同一用户内部仍按时间顺序
生成。每位用户写入独立 shard，全部通过 evidence、长度和重复检查后才原子合并，
因此不会发生多个进程覆盖主 session 文件。

`maintenance_restore_user000_dossier.py` 是一次性恢复工具，正常生成时不要运行。

## 常用命令

所有 API 参数从环境变量读取，不要把 key 写入脚本或日志：

```bash
export HABITBENCH_BASE_URL='https://.../v1'
export HABITBENCH_API_KEY='...'
export HABITBENCH_GEN_MODEL='gpt-5.5'
export HABITBENCH_REASONING_EFFORT='xhigh'
```

按阶段生成：

```bash
python scripts_wxq/pipeline_v04/04_generate_benchmark.py profiles --users 6
python scripts_wxq/pipeline_v04/04_generate_benchmark.py arcs --users 6
python scripts_wxq/pipeline_v04/04_generate_benchmark.py sessions --users 6
python scripts_wxq/pipeline_v04/04_generate_benchmark.py recovery --users 6
python scripts_wxq/pipeline_v04/04_generate_benchmark.py probes --users 6
python scripts_wxq/pipeline_v04/05_audit_release.py --require-complete
```

不要在当前多用户扩展期间重新运行 source discovery/review，也不要重新运行不带
`--only-users` 的早期阶段覆盖已验收的 user000。

## 当前状态（2026-07-20）

- 真实来源准备：完成。
- grounded habit 发现与独立审核：完成，curated pool 为 13 条。
- 六位用户画像：完成；每位 6–7 个 habit，其中 5–6 个用于测试。
- user000：126 个 session、40 个 probes，已经冻结为单用户提交包。
- user001–user005：正在生成 chronological arcs；之后仍需依次生成 sessions、
  blind recovery、probes，并执行完整审计。
- 多用户 v0.4：尚未完成，因此当前不能把主目录称为完整多用户 benchmark release。

更严格的设计边界见 `V04_END_TO_END_GENERATION_CONTRACT.md`；早期设计动机见
`7.15.md`。
