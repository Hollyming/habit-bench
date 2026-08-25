# H 集群 4/8 卡及 2×8 卡 H200 评测

H 集群入口复用 HABIT-Bench 的用户分片执行器：每张 H200 启动一个独立的
Qwen3-8B vLLM worker，完整用户 lifeline 不跨 shard，因此不需要 DDP。多个
Replica 通过 `JOB_ID` 共享 GPFS 动态任务队列；`NODE_RANK/NODE_COUNT` 只标识领取者，
不再对 shard index 静态取模。任一 GPU 完成当前 shard 后立即通过原子 `mkdir` 领取下一项，
因此不会在 method/domain 边界等待另一节点的慢 shard：

| `--gpus` | `--replicas` | GPU/Replica | 总卡数 | 默认 CPU/Replica | 默认内存/Replica |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 4 | 4 | 32 cores | 262144 MiB |
| 8 | 1 | 8 | 8 | 64 cores | 524288 MiB |
| 8 | 2 | 8 | 16 | 64 cores | 524288 MiB |

提交入口为 `scripts/submit_h_cluster.sh`，容器入口为
`scripts/cluster/run_h_eval.sh`。后者会在占用模型和启动 vLLM 前确认可见 GPU 数量
准确，并确认每张卡的 `nvidia-smi` 型号包含 `H200`；型号不符时任务立即失败，不会
产生可误认为 H200 的评测结果。

默认主实验方法是 compact `full_memory` 加八个已实现 memory 方法；默认数据是
Food v5、Finance/Software v1.4 和 Travel v16。`full_history` 只作为原始 recency
消融对照，不隐式加入主计划。

## 1. 共享资源与每个 clone 的可写目录

H RJob 不继承开发机镜像或 PT 集群的 `/plm-shared`。先把两个固定 Python 环境和
模型 snapshot 放到 H 集群持久化 GPFS。它们是所有评测者复用的只读输入；默认布局是：

```text
/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/
├── envs/
│   ├── habitbenchmark/
│   └── habitbenchmark-vllm/
├── models/habitbench/
│   ├── Qwen3-8B/
│   ├── bge-m3/
│   └── llmlingua-2-xlm-roberta-large-meetingbank/
└── .cache/tiktoken/
```

`env.h.example.sh` 中的 `HABITBENCH_H_SHARED_ROOT` 因而有意默认指向上述
`jmzhang` 目录；这不代表结果也写入该用户目录。仓库路径由模板自身自动定位，运行结果
默认写到当前 clone 的 `results/`，可能写锁或编译产物的缓存也默认隔离在：

```text
<current-clone>/
└── results/
    ├── <experiment-name>/
    └── .cache/habitbench/
        ├── huggingface/
        ├── vllm/
        └── torch/
```

因此 Alice 将仓库 clone 到
`/mnt/shared-storage-gpfs2/plm-gpfs/alice/workspace/habit-bench` 后，不需要修改模型或
环境路径，输出和可写 cache 会自然归 Alice。旧私有 profile 中的
`HABITBENCH_H_ROOT` 仍作为共享根的兼容别名；新配置应优先使用
`HABITBENCH_H_SHARED_ROOT`。若共享资源位置不同才覆盖它，若只想另放可写缓存则覆盖
`HABITBENCH_H_CACHE_ROOT`。

`/root/miniconda3` 是当前开发容器里的 Conda 安装器，不是 RJob 会自动继承的持久化
环境。用仓库脚本将两个环境创建到 GPFS；脚本固定 Python/依赖版本、避开 Anaconda
默认源 TOS，并隔离机器或用户注入的 pip index。依赖只解析自官方 `conda-forge`、
`pypi.org` 和 PyTorch CUDA 12.8 index；网络只由外层 `proxy_on` / `proxy_off` 控制，
脚本不写入镜像配置：

```bash
source ~/.bashrc
proxy_on
trap 'proxy_off' EXIT

bash scripts/cluster/create_h_envs.sh \
  --conda-bin /root/miniconda3/bin/conda \
  --env-root /mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/envs

proxy_off
trap - EXIT
```

如只需继续某一个环境，可加 `--method-only` 或 `--vllm-only`。安装日志和 `pip check`
结果分别保存在对应环境根目录；脚本也预取离线评测需要的 `o200k_base` 和 GPT-2
tiktoken 缓存。日志完成且 `pip check` 通过前，不应把环境视为可提交。

同一个代理会话内，从官方 Hugging Face 下载固定 revision；脚本支持断点续传，并为
三个 snapshot 生成 `HABIT_MODEL_INFO.json`。其中 BGE-M3 标记还包含权重 SHA256：

```bash
source ~/.bashrc
proxy_on
trap 'proxy_off' EXIT

/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/envs/habitbenchmark/bin/python \
  scripts/cluster/download_h_models.py \
  --model-root /mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/models/habitbench

proxy_off
trap - EXIT
```

公共目录 `/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/model` 可以作为模型来源
参考，但提交器仍会验证主实验的精确模型身份和必要文件。截至 2026-08-20 的盘点，该目录只有
Qwen3.5-0.8B-Base 与 Qwen3.5-2B，没有固定的 Qwen3-8B、BGE-M3 和 LLMLingua2
snapshot，因此不能在正式可比实验中直接替换。

BGE-M3 snapshot 必须继续匹配固定身份
`BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`、1024 维，并包含
`config.json`、`HABIT_MODEL_INFO.json` 和 `pytorch_model.bin`。Qwen3-8B、
LLMLingua2 和 tiktoken cache 也会在提交前及 worker 启动前检查。若模型文件已经
完整下载，只需离线刷新或验证身份标记，可在下载命令后追加 `--local-files-only`。

Qwen3 scale ablation 固定使用以下官方 snapshot，只替换基础 LLM，
BGE-M3、LLMLingua2、40,960-token 上下文、非 thinking 模板、采样参数、方法集合和
数据版本均与 8B 主实验一致：

- `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`
- `Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18`
- `Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137`

```bash
source ~/.bashrc
proxy_on
trap 'proxy_off' EXIT

/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/envs/habitbenchmark/bin/python \
  scripts/cluster/download_h_models.py --models qwen4b,qwen14b,qwen32b

proxy_off
trap - EXIT
```

如目录不同，复制环境模板并只改物理路径；本地文件已被 Git ignore：

```bash
cp scripts/cluster/env.h.example.sh scripts/cluster/env.h.local.sh
```

评测方法、模型 revision、vLLM 版本和推理 profile 不因集群变化。计划 manifest 同时
记录原始方法配置哈希、H 集群实际模型路径和 snapshot 哈希。

## 2. 任务类型和身份

三类任务不能混用参数：

- `managed-spot`：个人 User-AD，`normal + priority=1`，使用
  `llmarchitecture_gpu`，可被 Killbot 停止；这是默认类型。
- `reserved`：只能由本组 Group-AD，`normal + priority=5`，使用团队 reserved
  quota。
- `idle`：平台 idle，固定 `restart-policy=never` 和 30 秒 termination grace；不设置
  priority、charged group 或 private machine。

提交器要求显式填写 `--creator-type` 和 `--creator-ad`。它会从 `ssh-init` 注入的
BrainPP 凭据环境解析真实 creator，并要求它与 `--creator-ad` 完全一致；不会用 Job
名称或开发机 `whoami` 代替身份校验。直接提交入口只接受 `user` 或 `group` 身份，
不实现 autoolchain 代提交例外。新调度规则同时要求提交端设置：

```bash
export KUBEBRAIN_CLUSTER_ENTRY=http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451
```

`submit_h_cluster.sh` 会在第一次使用 `rjob` 前自动导出该固定值，并在最终提交前
再次恢复和校验；调用者不需要额外手工设置。直接运行 `rjob list/logs/events/stop`
等管理命令时仍需在当前 shell 手工导出。

## 3. 提交示例

个人 User-AD 提交 4 卡低优、可恢复评测：

```bash
bash scripts/submit_h_cluster.sh \
  --job-type managed-spot \
  --creator-type user \
  --creator-ad your-user-ad \
  --gpus 4 \
  --env-file scripts/cluster/env.h.local.sh \
  --output-root "$PWD/results/habit-h200-4g-v1"
```

本组 Group-AD 提交 8 卡 reserved 评测：

```bash
bash scripts/submit_h_cluster.sh \
  --job-type reserved \
  --creator-type group \
  --creator-ad your-group-ad \
  --gpus 8 \
  --env-file scripts/cluster/env.h.local.sh \
  --output-root "$PWD/results/habit-h200-8g-v1"
```

`your-group-ad` 必须替换为真实 Group-AD；不要用个人 User-AD 提交这个命令。

两个独立 8 卡 Replica（总计 16 张 H200）继续同一份 16-shard 计划：

```bash
bash scripts/submit_h_cluster.sh \
  --job-type reserved \
  --creator-type group \
  --creator-ad your-group-ad \
  --gpus 8 \
  --replicas 2 \
  --shards 16 \
  --env-file scripts/cluster/env.h.local.sh \
  --output-root "$PWD/results/habit-h200-main"
```

多 Replica 提交自动添加 `--host-network=true` 和
`-e DISTRIBUTED_JOB=true`，但每个 Replica 的 vLLM 服务仍在本机独立运行，不启动
torchrun/NCCL。跨节点只通过 GPFS 上的 launch-scoped claim/result 元数据协作。

固定的 Qwen3-4B、14B 和 32B 四域主实验均使用两个 8 卡 Replica，总计 16 张
H200。三种 BF16 权重均可由单张 H200 容纳，因此保持一卡一个独立 vLLM worker，
不使用 tensor parallel；每个规模使用独立结果根，避免错误复用其他规模的 checkpoint：

```bash
bash scripts/cluster/submit_qwen3_4b_main.sh
bash scripts/cluster/submit_qwen3_14b_main.sh
bash scripts/cluster/submit_qwen3_32b_main.sh
```

这些规模化入口默认写到 `$PROJECT_ROOT/results/<experiment-name>`，即 clone 所有者
自己的持久化目录。可统一覆盖为另一个属于自己的 GPFS 绝对路径，例如
`HABITBENCH_RESULTS_ROOT=/mnt/shared-storage-gpfs2/plm-gpfs/<your-ad>/results`；也可用
`HABITBENCH_Q4_OUTPUT_ROOT`、`HABITBENCH_Q14_OUTPUT_ROOT` 或
`HABITBENCH_Q32_OUTPUT_ROOT` 单独覆盖某个规模。H 集群提交时，最终路径必须位于
RJob 挂载覆盖的持久存储中；不要使用容器内通常指向 `/root` 的 `$HOME`。

该入口覆盖 `full_memory` 与八个 memory 方法、Food v5、Finance/Software v1.4、
Travel v16，共 `9 × 4 × 16 = 576` 个 shard，并继承完整 shard 粒度的断点恢复。

平台 idle 示例：

```bash
bash scripts/submit_h_cluster.sh \
  --job-type idle \
  --creator-type user \
  --creator-ad your-user-ad \
  --gpus 4 \
  --env-file scripts/cluster/env.h.local.sh \
  --output-root "$PWD/results/habit-h200-idle-v1"
```

只验证计划并让 RJob 渲染 YAML，不创建任务：

```bash
bash scripts/submit_h_cluster.sh \
  --dry-run \
  --job-type managed-spot \
  --creator-type user \
  --creator-ad your-user-ad \
  --gpus 4 \
  --shards 1 \
  --methods no_memory,full_memory \
  --max-users 1 --max-probes 4 \
  --env-file scripts/cluster/env.h.local.sh \
  --output-root "$PWD/results/habit-h200-dry-run"
```

默认 `--shards` 等于 `GPU/Replica × Replica`；也可显式设置更多用户 shard，由所有
Replica 的常驻 worker 分批执行。提交器会从项目、共享环境/模型、可写 cache 和输出
路径推导所需的最窄 owner 级 mount。例如 Alice 的 clone 复用 jmzhang 模型时，会同时
生成 `jmzhang` 和 `alice` 两个 mount，并验证每条必需路径至少由其中一个覆盖。无需手写
mount；特殊目录布局可重复传入 `--mount CONFIG`，或用空格分隔的
`HABITBENCH_H_MOUNTS` 覆盖。默认镜像、namespace 和 quota group 都是具体值，可通过
`--image` 覆盖；提交前预检拒绝空值、未替换的尖括号占位符和漏挂载路径。
CPU 和内存只能向上覆盖，不能低于每卡 8 cores 和 65536 MiB。

## 4. 中断恢复和观察

所有 plan、单 shard 输出、worker log 和合并结果都写到 GPFS 的 `output-root`。低优或
idle 被停止后，以相同 `output-root` 重新提交即可复用原 plan。恢复点粒度为一个完整
user shard：仅当 `worker_runtime.json`、成功的 `run_manifest.json`、contexts、预测和
指标全部存在时才跳过；只出现 `metrics.json` 不足以证明完整。该原子完成标记每个
shard 只写一次，不按 session/probe 高频保存。失败或中断目录在重跑前删除，避免重复
ingestion 半成品 memory backend 状态。改变 methods/datasets/shards 时必须换 output
root，或显式使用 `--force-plan`。不要把唯一输出或恢复点写到容器临时盘。

launch 内的动态 task claim 以 `JOB_ID` 隔离；它不能阻止另一个 RJob 误用相同输出根。
因此 runner 会在每个 method/domain 的 `.habitbench-shard-locks/` 下持有 shard 级 POSIX
排他锁，并在取得锁后才检查 checkpoint 或删除半成品。锁一直持有到成功标记发布或失败
清理结束，等待者会定期向云端日志输出 owner 与等待时间。进程终止时内核自动释放锁，
所以中断恢复不依赖手工删除锁文件。

两个 Replica 分别在 `replica_runtime/$JOB_ID/`、`h_rjob_logs/$JOB_ID/` 和
`vllm_logs/$JOB_ID/` 写 launch-scoped runtime/log，避免续跑读取上一任务的元数据。每个 shard 的
adapter stdout/stderr 会同时落盘并带 shard 前缀实时转发到 RJob 云端日志。每个 Replica
结束时只写一次 terminal marker；最后成功结束者通过原子 `merge.claim` 目录取得全局
merge 所有权，因此不会
并发覆盖 `evaluation_summary.json`。本次 launch 的 task claim/result 位于
`distributed_queue/$JOB_ID/{claims,results}/`：每个 task 通过 GPFS 原子 `mkdir` 领取，
不依赖跨节点 `flock` 或共享 JSON 游标。进度从这些不可变 task 文件和 Replica runtime
汇总，不按 session/probe 高频写入。新的 RJob 使用新 `JOB_ID` 重新扫描同一 plan，完整
shard 会快速跳过，失败或中断 shard 会重新执行。

```bash
export KUBEBRAIN_CLUSTER_ENTRY=http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451
rjob list --namespace ailab-llmarchitecture
rjob events JOB_ID --namespace ailab-llmarchitecture
rjob logs job JOB_ID -n 500 --namespace ailab-llmarchitecture
rjob stop JOB_ID
```

优先用 `rjob stop`。输出目录中的 `h_rjob_submit.log` 保存提交结果，
`h_rjob_logs/$JOB_ID/`、`replica_runtime/$JOB_ID/` 和 `vllm_logs/$JOB_ID/`
保存容器侧实时运行证据；全部 Replica 成功后生成合并的 `suite_runtime.json`。
