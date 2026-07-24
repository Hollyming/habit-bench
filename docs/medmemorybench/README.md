# MedMemoryBench 接入与复现

本目录记录 HABIT-Bench 对 MedMemoryBench 中 7 种结构化记忆方法的源码级接入方式。该实现直接使用仓库内的 Mem0、A-MEM 等方法源码，不调用托管的完整记忆工作流 API。

## 仓库布局

请将两个仓库克隆为同级目录：

```text
workspace/
├── habit-bench/
└── MedMemoryBench/
```

两个仓库均使用 `wjr` 分支。HABIT-Bench 的方法注册表固定到 MedMemoryBench commit：

```text
6591eb3251402f26535846ea4a95f5b4478ae35a
```

初始化 MedMemoryBench 子模块，并应用已经记录的 LightMem 兼容补丁：

```bash
cd MedMemoryBench
git submodule update --init --recursive
bash scripts/apply_lightmem_patch.sh
```

补丁脚本可重复执行。Mem0、A-MEM、MemOS、MemRL、Letta 和 MIRIX 的修改均已直接提交在 MedMemoryBench 的 `wjr` 分支中；LightMem 因为是外部子模块，使用固定 revision 加显式 patch 的方式复现。

## 环境

集群上已经验证过的环境如下：

```text
核心环境：/plm-shared/wangjiarui/anaconda3/envs/habit_medmemorybench
Letta：   /plm-shared/wangjiarui/anaconda3/envs/habit_medmemory_letta
MIRIX：   /plm-shared/wangjiarui/anaconda3/envs/habit_medmemory_mirix
```

这些路径仅是当前集群的已验证配置，不是跨机器的默认值。在其他机器上应创建等价环境，并通过 `PYTHON_BIN` 指定 Python。

## 轻量验证

运行 MedMemoryBench 测试：

```bash
cd MedMemoryBench
python -m pytest -q tests
```

运行 HABIT adapter 测试：

```bash
cd habit-bench
python -m pytest -q tests/evaluation/test_medmemorybench_adapter.py
```

查看可用配置：

```bash
cd MedMemoryBench
python main.py --list-methods
python main.py --list-datasets
```

## MedMemoryBench smoke

smoke 数据配置只使用一个 persona 和一个 evaluation unit：

```bash
cd MedMemoryBench
python main.py \
  --method mem0_qwen3-8b_smoke \
  --dataset medmemorybench_smoke_efficient \
  --output-dir outputs/mem0-smoke
```

其他方法可替换为 `amem`、`memos`、`memrl`、`lightmem`、`letta` 或 `mirix` 对应的 `*_qwen3-8b_smoke` 配置。

## HABIT-Bench smoke

两个仓库应保持同级；如果目录布局不同，则显式设置 `HABITBENCH_MEDMEMORYBENCH_ROOT`：

```bash
cd habit-bench
export HABITBENCH_MEDMEMORYBENCH_ROOT=../MedMemoryBench
export PYTHON_BIN=/path/to/the/method/environment/bin/python

bash scripts/run_method.sh \
  medmemorybench_mem0 \
  domain/food/food_habit_lifelines_stress \
  results/dev/food-medmemorybench-mem0 \
  --max-users 1 \
  --max-probes 4
```

支持的方法名：

```text
medmemorybench_mem0
medmemorybench_amem
medmemorybench_memos
medmemorybench_memrl
medmemorybench_lightmem
medmemorybench_letta
medmemorybench_mirix
```

adapter 按时间顺序增量写入 public sessions，只调用方法的原生 retrieval 路径，再将 `memory_context` 交给 HABIT 的统一 answerer。private label、gold habit graph 和 gold evidence 不会传给记忆方法。

## 正式结果的验收要求

正式结果至少应包含：

- 两个仓库的精确 commit 与子模块 revision；
- 数据、模型、prompt 和 metric hash；
- 完整的 user/persona 与 query/probe coverage；
- 逐题 prediction 和 retrieved context；
- strict shard merge 及失败扫描证据；
- method-native、common-reader 与 adapted 口径的明确标签。

实现修改见 [changes.md](changes.md)，精简实验记录见 [experiment_notes.md](experiment_notes.md)。
