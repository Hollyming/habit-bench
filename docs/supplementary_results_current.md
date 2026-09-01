# HABIT-Bench Supplementary 当前结果（Qwen3-8B）

> 快照时间：2026-08-31（HKT）。本文汇总当前四个正式域的离线 supplementary sidecar
> 与 Oracle diagnostics，不替换主实验指标，也不包含人审。Food 使用
> `food_habit_lifelines_final_check`，Finance/Software 使用 v1.4，Travel 使用 v16。

## 实验口径与完整性

- 模型：Qwen3-8B；lifecycle memory methods 共 9 种。
- Food final：30 users、3,900 sessions、630 probes。
- Finance/Software：v1.4，分别为 1,368 和 680 probes；Travel v16 为 650 probes。
  四域当前合计 3,328 probes。
- 用户等权 Accuracy 的 95% CI 使用 user-cluster percentile bootstrap（10,000 samples，
  seed 42）；方法配对同时报告 user-cluster bootstrap 和 probe-level exact McNemar，
  并对域内 36 个 method pairs 做 Holm 校正。
- Food final 与 Travel v16 的 sidecar 已覆盖到当前版本；主方法 sidecar 位于
  `./results/habit-h200-main-compact-v2/supplementary`，Food final 的 14-method
  扩展结果也写入 `./results/habit-h200-main-food-final-qwen3-8b-v1/supplementary`。

| Domain | Dataset | Probes | Users | Completed lifecycle methods |
|---|---|---:|---:|---:|
| Food | `food_habit_lifelines_final_check` | 630 | 30 | 9/9 |
| Finance | `habit_bench_multidogo_finance_software_release_gated_v1_4:finance` | 1,368 | 36 | 9/9 |
| Software | `habit_bench_multidogo_finance_software_release_gated_v1_4:software` | 680 | 18 | 9/9 |
| Travel | `release_candidate_v16_postrepair_repaired_r4` | 650 | 30 | 9/9 |

## 用户等权 Accuracy

域列为 `user-macro Accuracy [user-cluster bootstrap 95% CI]`，均为百分数；`Pooled micro`
按 3,328 个当前 probes 汇总。

| Method | Food final | Finance | Software | Travel v16 | Pooled micro |
|---|---:|---:|---:|---:|---:|
| Full-Memory | 41.11 [37.62, 44.76] | 20.76 [18.08, 23.67] | 23.24 [20.06, 26.18] | 27.23 [24.08, 32.54] | 26.38 |
| Mem0 | 29.84 [26.98, 32.70] | 23.10 [20.34, 26.06] | 26.76 [23.42, 29.94] | 26.62 [24.50, 30.81] | 25.81 |
| A-MEM | 39.52 [36.03, 43.17] | 22.59 [20.17, 25.20] | 26.47 [23.77, 29.40] | 32.77 [30.19, 37.76] | 28.58 |
| MemOS | 38.10 [34.76, 41.59] | 22.15 [19.30, 25.22] | 26.32 [23.19, 29.35] | 31.38 [27.47, 35.72] | 27.82 |
| MemRL | 37.62 [34.13, 41.11] | 21.86 [19.27, 24.66] | 25.00 [21.83, 28.00] | 32.46 [30.00, 35.95] | 27.55 |
| LightMem | 32.70 [29.21, 36.19] | 22.51 [19.87, 25.37] | 25.44 [21.85, 28.85] | 28.00 [25.07, 32.39] | 26.11 |
| Letta | 38.57 [35.08, 41.90] | 21.71 [19.08, 24.50] | 25.29 [22.33, 28.14] | 33.23 [29.78, 38.06] | 27.88 |
| MIRIX | 28.73 [25.87, 31.43] | 22.51 [20.02, 25.16] | 27.50 [24.65, 30.06] | 27.69 [25.07, 31.42] | 25.72 |
| SeCom | 38.57 [35.56, 41.59] | 23.03 [20.49, 25.65] | 24.85 [21.79, 28.08] | 32.92 [29.81, 37.08] | 28.28 |

## No Memory 与 Oracle diagnostics

Oracle 读取 private annotations，只作为诊断上界，不属于可部署 memory system，也不应与
主方法混为同类排名。Food final 的 No Memory、Oracle Evidence、Oracle Habit State 均为
630/630；其中 Oracle Habit State 已使用修复后的 graph-action/condition contract 重跑，
16/16 shards 成功并覆盖旧 merged 结果。

下表为 exact-choice micro Accuracy（%）：

| Diagnostic | Food final | Finance | Software | Travel v16 | Pooled | Correct / Total |
|---|---:|---:|---:|---:|---:|---:|
| No Memory | 20.79 | 23.32 | 27.21 | 26.46 | 24.25 | 807 / 3,328 |
| Oracle Evidence | **53.33** | 29.82 | 31.91 | 42.15 | 37.11 | 1,235 / 3,328 |
| Oracle Habit State | **92.54** | **93.71** | **94.56** | **98.31** | **94.56** | **3,147 / 3,328** |

域列的用户等权 Accuracy 与 user-cluster bootstrap 95% CI 如下（%）：

| Diagnostic | Food final | Finance | Software | Travel v16 |
|---|---:|---:|---:|---:|
| No Memory | 20.79 [17.14, 24.60] | 23.35 [20.53, 26.27] | 27.20 [23.82, 30.44] | 26.87 [23.47, 30.38] |
| Oracle Evidence | **53.33 [50.32, 56.51]** | 29.85 [27.46, 32.30] | 31.90 [29.21, 34.74] | 42.19 [37.85, 46.28] |
| Oracle Habit State | **92.54 [90.63, 94.44]** | **93.71 [92.45, 94.92]** | **94.57 [92.20, 96.63]** | **98.42 [97.55, 99.22]** |

主要现象：Finance、Software、Travel 在给定最终 Oracle Habit State 后接近饱和，说明
主要瓶颈是从长期历史恢复正确状态，而不是 answer head；Food final 的 Oracle Evidence
仍高于普通方法，符合 private evidence 可直接支持部分题目的构造。

## Explicit-to-Habit transfer

该指标定义为 `explicit_retrieval Accuracy - latent-habit Accuracy`。当前 Food final 只含
`direct_use`、`boundary`、`exception` 三类 probes，不含 `explicit_retrieval`；其他三个域
也没有同一数据生成机制下的 explicit probes，因此当前四域均按 contract 标记为
unavailable，不从退役的 Food v5 结果外推。

## Policy-component Accuracy

该指标检查多习惯组合 choice 中每个 target-habit variant 是否正确，比 exact-choice
Accuracy 更宽松。Food final 缺少精确的 choice-policy taxonomy，因此只对 Finance、
Software、Travel v16 计算；`Pooled` 按 5,124 个 component evaluations 汇总。

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

## Answer–retrieval 错误耦合

下表在当前四域合并的 3,328 probes 上给出描述性共现比例。`C/W` 表示答案正确/错误，
`Hit/Miss` 表示证据命中/未命中；`W+NB` 的分母仅为带 nonbinding 标注的 Finance、
Software、Travel probes，不能解释为模块错误的因果分解。Full-Memory 的 context evidence
不导出 Top-5 `evidence_hit`，因此不可用。

| Method | C+Hit | C+Miss | W+Hit | W+Miss | W+NB intrusion |
|---|---:|---:|---:|---:|---:|
| Full-Memory | — | — | — | — | — |
| Mem0 | 9.71 | 16.11 | 22.12 | 52.07 | 25.60 |
| A-MEM | 15.35 | 13.22 | 27.22 | 44.20 | 22.68 |
| MemOS | 13.31 | 14.51 | 24.37 | 47.81 | 29.87 |
| MemRL | 13.46 | 14.09 | 24.79 | 47.66 | 17.02 |
| LightMem | 0.00 | 26.11 | 0.00 | 73.89 | 0.00 |
| Letta | 14.39 | 13.49 | 25.45 | 46.66 | 17.81 |
| MIRIX | 0.00 | 25.72 | 0.00 | 74.28 | 0.00 |
| SeCom | 14.27 | 14.00 | 24.28 | 47.45 | 15.63 |

## 效率与产物体积

均值按当前四域全部 probes 汇总。`Answer latency` 是逐题 answer 调用耗时；`Retrieval`
只统计方法记录的 `retrieval_elapsed_sec`；`Artifacts` 是四个 merged 目录内普通文件的
递归大小。

| Method | Answer latency (s) | Memory tokens | Prompt tokens | Total tokens | Retrieval (s) | Artifacts (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| Full-Memory | 0.378 | 33,658.4 | 34,208.6 | 34,216.6 | — | 0.431 |
| Mem0 | 0.181 | 108.1 | 659.2 | 667.2 | 0.480 | 0.024 |
| A-MEM | 0.265 | 2,994.1 | 3,544.3 | 3,552.3 | 0.445 | 0.160 |
| MemOS | 0.189 | 459.0 | 1,009.2 | 1,017.2 | 0.446 | 0.060 |
| MemRL | 0.223 | 1,580.4 | 2,131.4 | 2,139.4 | 0.470 | 0.093 |
| LightMem | 0.186 | 328.5 | 878.7 | 886.7 | 0.248 | 0.027 |
| Letta | 0.243 | 2,272.6 | 2,822.8 | 2,830.8 | 0.486 | 0.126 |
| MIRIX | 0.184 | 369.0 | 919.8 | 927.8 | 0.053 | 0.032 |
| SeCom | 0.234 | 2,026.1 | 2,576.4 | 2,584.4 | — | 0.043 |

## 域内配对比较

`Significant wins` 是该域内对其余 8 个方法的 exact McNemar Holm-adjusted `p < 0.05`
胜场数。当前 Food final 和 Travel v16 仍使用同一配对检验协议。

| Domain | Top micro method | Acc | Runner-up | Acc | Top significant wins |
|---|---|---:|---|---:|---:|
| Food final | Full-Memory | 41.11 | A-MEM | 39.52 | 3/8 |
| Finance | Mem0 | 23.10 | SeCom | 23.03 | 0/8 |
| Software | MIRIX | 27.50 | Mem0 | 26.76 | 0/8 |
| Travel v16 | Letta | 33.23 | SeCom | 32.92 | 3/8 |

## 可用性与结果文件

| Item | Current status | Reason / scope |
|---|---|---|
| User-macro + user-cluster CI | 36/36 available | 当前四域 9 methods × 4 domains |
| Stress slices | 36/36 available | `domain`、`probe_type`、`support_count`、`history_length_bin`、`distractor_ratio_bin`、`evidence_position_bin`、`evidence_bands` |
| Policy components | 27/36 available | Finance、Software、Travel v16 可用；Food final 缺 taxonomy |
| Explicit-to-Habit gap | 0/36 available | 当前四域均无可配对的 explicit probes |
| Answer–retrieval coupling | 32/36 available | Full-Memory 使用 context 语义，不导出 Top-5 evidence hit |
| Calibration | 0/36 unavailable | exact-choice 输出不包含完整 choice probabilities |
| False personalization | 0/36 unavailable | 当前数据缺少完整 `choice_action_taxonomy` 与 `personalization_applicable` 标注 |
| No Memory / Oracle diagnostics | 12/12 complete | Food final 48/48 control shards；Finance/Software/Travel 144/144；strict merge 与非人审 sidecar 均完成 |
| Human audit | intentionally excluded | 按本轮实验要求不执行 |

主方法和 Oracle diagnostics 的每个 `domain/method` supplementary 目录包含：

- `supplementary_metrics.json`：用户统计、transfer、component、错误耦合、效率及可用性；
- `supplementary_metrics_by_slice.csv`：难度/历史/证据位置等观察性切片；
- `supplementary_metrics_by_user.csv`：逐用户正确数、probe 数与 Accuracy；
- `supplementary_probe_diagnostics.jsonl`：逐 probe 的切片和历史难度诊断。

主方法 sidecar 根目录为 `./results/habit-h200-main-compact-v2/supplementary`；Food final
controls 的 authoritative 根目录为 `./results/habit-h200-supplementary-food-final-qwen3-8b-v1`。
旧 Food v5 merged/shard artifacts 仅作为 legacy 复现材料，不进入当前表格。
