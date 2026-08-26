# HABIT-Bench Supplementary 当前结果（Qwen3-8B）

> 快照时间：2026-08-26 15:30 HKT。本页汇总已完成的离线 supplementary sidecar 与 Qwen3-8B Oracle diagnostics；它不替换主实验指标。按实验要求不包含人审。

## 实验口径与完整性

- 模型：Qwen3-8B；方法与主实验一致，共 9 种。
- 数据：Food v5、Finance/Software v1.4、Travel v16（`release_candidate_v16_postrepair_repaired_r4`）。Travel 使用独立 v16 主实验目录覆盖旧四域目录中的 v13。
- 覆盖：主方法 sidecar 36/36 method×domain runs；No Memory、Oracle Evidence、Oracle Habit State diagnostics 12/12 runs；每个 run 覆盖同一组 4,168 probes。
- 统计：user-macro Accuracy 的 95% CI 使用 user-cluster percentile bootstrap，10,000 samples，seed 42；方法配对同时报告 user-cluster bootstrap 和 probe-level exact McNemar，并对域内 36 个 method pairs 做 Holm 校正。
- 输出：主方法 sidecar 位于 `./results/habit-h200-main-compact-v2/supplementary`；Oracle diagnostics 位于 `./results/habit-h200-supplementary-qwen3-8b-v1/{domain}/{method}/merged`，对应 sidecar 位于该根目录的 `supplementary/`。

| Domain | Dataset | Probes | Users | Completed methods |
|---|---|---:|---:|---:|
| Food | `food_habit_lifelines_stress_v5` | 1,470 | 30 | 9/9 |
| Finance | `habit_bench_multidogo_finance_software_release_gated_v1_4:finance` | 1,368 | 36 | 9/9 |
| Software | `habit_bench_multidogo_finance_software_release_gated_v1_4:software` | 680 | 18 | 9/9 |
| Travel | `release_candidate_v16_postrepair_repaired_r4` | 650 | 30 | 9/9 |

## 用户等权 Accuracy

域列为 `user-macro Accuracy [user-cluster bootstrap 95% CI]`，均为百分数。`Pooled micro` 按 4,168 个 probes 汇总，因此与主实验当前文档的 Qwen3-8B pooled Accuracy 一致；supplementary 的新增信息是用户等权估计与按用户聚类的区间。

| Method | Food | Finance | Software | Travel v16 | Pooled micro |
|---|---:|---:|---:|---:|---:|
| Full-Memory | 53.20 [47.96, 58.71] | 20.79 [18.08, 23.67] | 23.24 [20.06, 26.18] | 28.11 [24.08, 32.54] | 33.61 |
| Mem0 | 37.21 [34.22, 40.14] | 23.12 [20.34, 26.06] | 26.77 [23.42, 29.94] | 27.52 [24.50, 30.81] | 29.22 |
| A-MEM | 50.48 [46.12, 55.24] | 22.62 [20.17, 25.20] | 26.48 [23.77, 29.40] | 33.97 [30.19, 37.76] | 34.64 |
| MemOS | 47.96 [44.76, 51.29] | 22.17 [19.30, 25.22] | 26.34 [23.19, 29.35] | 31.73 [27.47, 35.72] | 33.37 |
| MemRL | 51.09 [47.14, 55.03] | 21.90 [19.27, 24.66] | 25.01 [21.83, 28.00] | 32.99 [30.00, 35.95] | 34.33 |
| LightMem | 35.85 [32.58, 39.12] | 22.54 [19.87, 25.37] | 25.45 [21.85, 28.85] | 28.62 [25.07, 32.39] | 28.55 |
| Letta | 51.63 [47.89, 55.65] | 21.75 [19.08, 24.50] | 25.31 [22.33, 28.14] | 33.98 [29.78, 38.06] | **34.64** |
| MIRIX | 33.67 [30.27, 37.01] | 22.56 [20.02, 25.16] | 27.51 [24.65, 30.06] | 28.23 [25.07, 31.42] | 28.07 |
| SeCom | 48.23 [43.81, 53.06] | 23.05 [20.49, 25.65] | 24.86 [21.79, 28.08] | 33.43 [29.81, 37.08] | 33.76 |

## No Memory 与 Oracle diagnostics（Qwen3-8B）

正式 RJob `hb-q8b-supp-oracle-v16-89329939` 已以 `2 Replicas × 8 H200 = 16 H200` 完成，2/2 Replicas succeeded，四域 192/192 shards strict merge 成功。Oracle 方法读取 private annotations，因此只作为诊断上界，不属于可部署 memory system，也不应与主方法混为同类排名。

下表为 exact-choice micro Accuracy（%）。`Pooled` 按四域全部 4,168 probes 汇总。

| Diagnostic | Food | Finance | Software | Travel v16 | Pooled | Correct / Total |
|---|---:|---:|---:|---:|---:|---:|
| No Memory | 26.26 | 23.32 | 27.21 | 26.46 | 25.48 | 1,062 / 4,168 |
| Oracle Evidence | **78.78** | 29.82 | 31.91 | 42.15 | 49.35 | 2,057 / 4,168 |
| Oracle Habit State | 59.86 | **93.71** | **94.56** | **98.31** | **82.63** | 3,444 / 4,168 |

域列的用户等权 Accuracy 与 user-cluster bootstrap 95% CI 如下（%）：

| Diagnostic | Food | Finance | Software | Travel v16 |
|---|---:|---:|---:|---:|
| No Memory | 26.26 [23.06, 29.86] | 23.35 [20.53, 26.27] | 27.20 [23.82, 30.44] | 26.87 [23.47, 30.38] |
| Oracle Evidence | **78.78 [74.83, 82.79]** | 29.85 [27.46, 32.30] | 31.90 [29.21, 34.74] | 42.19 [37.85, 46.28] |
| Oracle Habit State | 59.86 [57.14, 62.86] | **93.71 [92.45, 94.92]** | **94.57 [92.20, 96.63]** | **98.42 [97.55, 99.22]** |

Oracle Evidence 的 evidence hit@5 在四域均为 100%；evidence recall@5 macro 分别为 Food 49.78%、Finance 94.83%、Software 94.71%、Travel 100%。Food 的 gold evidence 链经常超过 top-5 容量，因此即使按 gold evidence 直接构造上下文，Recall@5 仍存在结构性上限。

主要现象：Finance、Software、Travel 在给定最终 Oracle Habit State 后接近饱和，说明主要瓶颈是从长期历史恢复正确状态，而不是 answer head；Oracle Evidence 在这三域只比 No Memory 提升 4.71–15.69 个百分点，说明原始证据即使完全命中，仍需要跨 session 解析、时序绑定和状态归纳。Food 的 Oracle Evidence 反而高于 Oracle Habit State，符合其显式检索 probes 与原文证据可直接作答的构造特点。

## Explicit-to-Habit transfer（Food v5）

该指标定义为 `explicit_retrieval Accuracy - latent-habit Accuracy`。只有 Food 同时包含两类 probes，因此其他三个域不跨数据生成机制强行计算。负 gap 只表示当前 Food 显式检索题准确率低于隐式习惯题，不能单独解释为迁移能力因果下降。

| Method | Explicit Acc | Latent-habit Acc | Gap | Paired user 95% CI |
|---|---:|---:|---:|---:|
| Full-Memory | 73.81 | 49.76 | +24.05 | [+16.03, +32.22] |
| Mem0 | 32.86 | 37.94 | -5.08 | [-11.35, +0.87] |
| A-MEM | 38.57 | 52.46 | -13.89 | [-21.98, -5.95] |
| MemOS | 31.43 | 50.71 | -19.29 | [-26.19, -12.38] |
| MemRL | 36.67 | 53.49 | -16.83 | [-23.97, -9.37] |
| LightMem | 29.05 | 36.98 | -7.94 | [-14.05, -1.75] |
| Letta | 32.86 | 54.76 | -21.90 | [-29.76, -13.89] |
| MIRIX | 32.86 | 33.81 | -0.95 | [-8.25, +6.59] |
| SeCom | 33.81 | 50.63 | -16.83 | [-24.21, -9.44] |

主要现象：Full-Memory 的显式检索优势显著；A-MEM、MemOS、MemRL、LightMem、Letta、SeCom 的 gap 区间完全低于 0；Mem0 与 MIRIX 的区间跨 0。

## Policy-component Accuracy

该指标检查多习惯组合 choice 中每个 target-habit variant 是否正确，比 exact-choice Accuracy 更宽松。Food 缺少精确的 per-choice component taxonomy，因此不可用；Travel v16 已提供相应签名。`Pooled` 按 Finance、Software、Travel 的 5,124 个 component evaluations 汇总。

| Method | Finance | Software | Travel v16 | Pooled | Correct / Total |
|---|---:|---:|---:|---:|---:|
| Full-Memory | 46.84 | 48.03 | 53.61 | 47.72 | 2,445 / 5,124 |
| Mem0 | 49.05 | 51.52 | 50.77 | 49.94 | 2,559 / 5,124 |
| A-MEM | 47.66 | 51.84 | **55.93** | 49.57 | 2,540 / 5,124 |
| MemOS | 48.07 | 50.95 | 55.15 | 49.49 | 2,536 / 5,124 |
| MemRL | 47.28 | 50.25 | 51.55 | 48.52 | 2,486 / 5,124 |
| LightMem | 49.05 | 50.76 | 52.06 | 49.80 | 2,552 / 5,124 |
| Letta | 47.63 | 50.44 | **55.93** | 49.12 | 2,517 / 5,124 |
| MIRIX | 48.54 | **51.90** | 54.38 | **50.02** | 2,563 / 5,124 |
| SeCom | 48.45 | 50.57 | 54.38 | 49.55 | 2,539 / 5,124 |

各方法 pooled component accuracy 仅为 47.72%–50.02%，显著高于部分域的 exact-choice Accuracy，但仍接近二元 component 的随机水平；这说明多组件组合不是唯一瓶颈，单个习惯状态恢复也仍然困难。

## Answer–retrieval 错误耦合

下表在四域合并的 4,168 probes 上给出描述性共现比例。`C/W` 表示答案正确/错误，`Hit/Miss` 表示证据命中/未命中。`W+NB` 的分母仅为带 nonbinding 标注的 Finance、Software、Travel probes。它不能被解释为模块错误的因果分解。

| Method | C+Hit | C+Miss | W+Hit | W+Miss | W+NB intrusion |
|---|---:|---:|---:|---:|---:|
| Full-Memory | — | — | — | — | — |
| Mem0 | 13.84 | 15.38 | 23.25 | 47.53 | 25.60 |
| A-MEM | 23.61 | 11.04 | 26.90 | 38.46 | 22.68 |
| MemOS | 20.01 | 13.36 | 24.02 | 42.61 | 29.87 |
| MemRL | 22.50 | 11.83 | 23.13 | 42.54 | 17.02 |
| LightMem | 0.00 | 28.55 | 0.00 | 71.45 | 0.00 |
| Letta | 23.44 | 11.20 | 23.92 | 41.43 | 17.81 |
| MIRIX | 0.00 | 28.07 | 0.00 | 71.93 | 0.00 |
| SeCom | 21.71 | 12.04 | 24.78 | 41.46 | 15.63 |

主要现象：A-MEM 的 grounded correct（C+Hit）最高；所有有 lineage 的方法仍有 23.13%–26.90% 的 W+Hit，说明“取回至少一条证据”远不足以保证正确恢复习惯状态。LightMem/MIRIX 的全零 Hit 来自缺少稳定 source-session lineage，不代表没有取回文本。Full-Memory 使用 context evidence 语义，不导出 Top-5 `evidence_hit`，因此本交叉表不可用。

## 效率与产物体积

均值按四域全部 probes 汇总。`Answer latency` 是逐题 answer 调用耗时；`Retrieval` 只统计方法记录的 `retrieval_elapsed_sec`，不含用户历史 ingestion 和整个 RJob wall-clock。`Artifacts` 是四个 merged 目录内普通文件的递归大小，缓存若位于目录外不会计入。

| Method | Answer latency (s) | Memory tokens | Prompt tokens | Total tokens | Retrieval (s) | Artifacts (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| Full-Memory | 0.364 | 34,351.2 | 34,833.9 | 34,841.9 | — | 0.556 |
| Mem0 | 0.178 | 107.0 | 590.6 | 598.6 | 0.425 | 0.029 |
| A-MEM | 0.252 | 2,998.4 | 3,481.1 | 3,489.1 | 0.400 | 0.201 |
| MemOS | 0.187 | 452.7 | 935.4 | 943.4 | 0.396 | 0.075 |
| MemRL | 0.218 | 1,541.0 | 2,024.5 | 2,032.5 | 0.418 | 0.113 |
| LightMem | 0.182 | 318.0 | 800.7 | 808.7 | 0.223 | 0.033 |
| Letta | 0.239 | 2,315.3 | 2,798.0 | 2,806.0 | 0.438 | 0.162 |
| MIRIX | 0.180 | 368.6 | 851.8 | 859.8 | 0.049 | 0.040 |
| SeCom | 0.225 | 2,070.5 | 2,553.2 | 2,561.2 | — | 0.055 |

Full-Memory 的 memory-token 开销约为最省 token 的 Mem0 的 321 倍；A-MEM 是当前外置 memory 方法中 token 与 artifact 开销最高的一组。延迟字段受共享推理服务、批处理和各方法计时边界影响，不能直接视为独立部署吞吐。

## 域内配对比较

`Significant wins` 是该域内对其余 8 个方法的 exact McNemar Holm-adjusted `p < 0.05` 胜场数。Top 与 runner-up 的 exact/cluster-bootstrap Holm-adjusted p 值在四域均为 1.0，因此首名与次名之间没有通过当前 family-wise 校正的显著差异。

| Domain | Top micro method | Acc | Runner-up | Acc | Top significant wins |
|---|---|---:|---|---:|---:|
| Food | Full-Memory | 53.20 | Letta | 51.63 | 3/8 |
| Finance | Mem0 | 23.10 | SeCom | 23.03 | 0/8 |
| Software | MIRIX | 27.50 | Mem0 | 26.76 | 0/8 |
| Travel v16 | Letta | 33.23 | SeCom | 32.92 | 3/8 |

## 可用性与未完成项

| Item | Current status | Reason / scope |
|---|---|---|
| User-macro + user-cluster CI | 36/36 available | 当前文档主统计 |
| Stress slices | 36/36 available | `domain`、`probe_type`、`support_count`、`history_length_bin`、`distractor_ratio_bin`、`evidence_position_bin`、`evidence_bands`；仅作观察性分层 |
| Policy components | 27/36 available | Finance、Software、Travel v16 可用；Food 缺 component taxonomy |
| Explicit-to-Habit gap | 9/36 available | 仅 Food 同时含 explicit 与 latent probes |
| Answer–retrieval coupling | 32/36 available | Full-Memory 四域使用 context 语义，不导出 Top-5 evidence hit |
| Calibration | 0/36 unavailable | 当前 exact-choice 输出不包含每个 choice 的完整概率，不能制造置信度 |
| False personalization | 0/36 unavailable | 当前数据缺少完整 `choice_action_taxonomy` 与 `personalization_applicable` 标注 |
| No Memory / Oracle Evidence / Oracle Habit State | 12/12 complete | 正式任务 `hb-q8b-supp-oracle-v16-89329939` 2/2 Replicas succeeded；四域 192/192 shards、strict merge 与非人审 sidecar 均完成，输出位于 `./results/habit-h200-supplementary-qwen3-8b-v1` |
| Human audit | intentionally excluded | 按本轮实验要求不执行 |

## 结果文件结构

主方法和 Oracle diagnostics 的每个 `domain/method` supplementary 目录包含：

- `supplementary_metrics.json`：用户统计、transfer、component、错误耦合、效率及可用性；
- `supplementary_metrics_by_slice.csv`：难度/历史/证据位置等观察性切片；
- `supplementary_metrics_by_user.csv`：逐用户正确数、probe 数与 Accuracy；
- `supplementary_probe_diagnostics.jsonl`：逐 probe 的切片和历史难度诊断。

每个域的 `comparison/` 包含全部 9 方法的 summary、36 个配对差值、user-cluster bootstrap CI、exact McNemar p 值及两类 Holm 校正结果。根目录另有 `supplementary_summary.csv` 与 `supplementary_manifest.json`，后者记录 Travel v16 override 的真实来源目录。
