# H 集群 8 卡长期调试节点

## 已申请任务

| 项目 | 值 |
| --- | --- |
| RJob 名称 | `zjm-debug-h200-8gpu-v1-78293916` |
| Show name | `zjm-debug-h200-8gpu-v1` |
| 当前状态 | `Stopped`（已释放 8 张 H200；正式 API sidecar 仍固定到同一节点） |
| 实际节点 | `gpu-lg-cmc-h-h200-1137.host.h.pjlab.org.cn` |
| Creator | `linzhouhan`（Group-AD） |
| 任务类型 | normal / priority 5 / reserved |
| 资源 | 1 Replica × 8 H200 GPU |
| CPU / 内存 | 每 Replica 64 CPU、524288 MiB（512 GiB） |
| 镜像 | `registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20251117` |
| SSH | 已启用 `--enable-sshd` |
| 持久化挂载 | `gpfs://gpfs2/plm-gpfs/jmzhang` → `/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang` |

该任务的主进程曾是 `sleep infinity`，现已按需停止并释放 8 张 reserved H200。
正式 API 续跑任务使用独立的 0-GPU sidecar 固定在同一节点，不依赖该调试进程。

## 查询状态和日志

在提交端或有 `rjob` CLI 的终端中，每次调用 RJob 命令前都执行：

```bash
source /etc/profile.d/ssh-init.sh
export KUBEBRAIN_CLUSTER_ENTRY=http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451
```

```bash
JOB=zjm-debug-h200-8gpu-v1-78293916
rjob get "$JOB" --namespace ailab-llmarchitecture
rjob events "$JOB" --namespace ailab-llmarchitecture
rjob logs job "$JOB" -n 200 --namespace ailab-llmarchitecture
```

任务曾绑定的节点为
`gpu-lg-cmc-h-h200-1137.host.h.pjlab.org.cn`。如果之后重新申请或任务被平台重启，
请以最新一次 `rjob get` 显示的节点和 Replica 名称为准。排队期间显示 `Inqueue` 是
正常的，表示 RJob 已创建但尚未绑定 Replica。

如重新申请调试节点，也可以在平台的“计算 → RJOB”页面打开该任务，待 Replica 进入 Running 后点击
Replica 的 Web terminal。由于提交时开启了 `--enable-sshd`，页面会显示平台生成的
SSH 命令和密钥使用方式；SSH 命令以平台页面显示的为准，不要自行改写端口或密钥。

## 进入节点后的初始化

节点内的项目、环境、模型和结果都使用持久化 GPFS2 路径：

```bash
export PROJECT=/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/workspace/habit-bench
cd "$PROJECT"

# H 集群环境：Python、vLLM、模型和缓存路径会统一设置。
source "$PROJECT/scripts/cluster/env.h.example.sh"

echo "host=$(hostname -f)"
nvidia-smi
"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
PY
```

当前环境默认使用：

```text
Python:       /mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/envs/habitbenchmark/bin/python
vLLM Python:  /mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/envs/habitbenchmark-vllm/bin/python
Qwen3-8B:     /mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/models/habitbench/Qwen3-8B
BGE-M3:       /mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/models/habitbench/bge-m3
```

如果需要使用其它模型或环境，在当前 shell 中显式覆盖对应的
`HABITBENCH_LLM_MODEL`、`HABITBENCH_VLLM_PYTHON` 等变量；不要把模型复制到容器
本地临时盘。

## 常用调试方式

### 运行单元测试或小样本测试

```bash
cd /mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/workspace/habit-bench
source scripts/cluster/env.h.example.sh

"$PYTHON_BIN" -m pytest -q tests/evaluation/test_protocol.py
"$PYTHON_BIN" scripts/run_locomo_task.py --help
```

长时间命令建议放入 `tmux`，避免 SSH 断开导致前台进程退出：

```bash
tmux new -s habit-debug
# 在 tmux 中运行实验
tmux attach -t habit-debug
```

### 外部 API smoke（已完成）

为排查 `zjm-api-main-v5` 的 API/并发稳定性，曾在本节点启动独立 smoke suite，排除
`kimi-k3`，只运行 DeepSeek 和 GLM 的较快方法（无 memory、上下文、Recency、BM25、
Dense、Temporal Hybrid）。每个 model × method × domain 只取 1 个用户、4 个 probe，
不写入正式 API 结果目录：

```text
/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/workspace/habit-bench/results/api-debug-smoke-20260828-1902/
```

该 smoke 已自然结束，DeepSeek 和 GLM 各 32/32 成功，网关返回码为 0；不再有
smoke 进程需要停止。结果仍保留在上述目录中。

运行中的正式 API sidecar 可通过 RJob replica 日志和正式结果目录查看。通过
`brainctl` 查看实时日志和网关指标（Replica 入口比直接 Pod 入口权限更合适）：

```bash
REPLICA=replica/zjm-api-formal-resume-deepseek-glm-v1-3032133-bx5c6
ROOT=/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/workspace/habit-bench/results/habit-h200-api-main-v1

brainctl -n ailab-llmarchitecture exec "$REPLICA" -- \
  tail -f "$ROOT/api_runtime/launcher.log"
brainctl -n ailab-llmarchitecture exec "$REPLICA" -- \
  cat "$ROOT/api_gateway/metrics.json"
```

判定重点是 `metrics.json` 中的 `status_counts`、`retries`、
`empty_response_retries`、`upstream_attempts`，以及 runner 日志中的
`task_failed`。正式 sidecar 使用 external-API CPU 模式（逻辑 worker 仍为 4+4），
不占用已释放的 H200；输出写入正式 suite root，并从持久化 queue 断点恢复。

### 单节点多卡检查

该任务是单 Replica 的 8 卡节点，适合检查单机 8 卡 CUDA、vLLM 或 NCCL。单机
调试不需要 `torchrun`；如确实需要测试多进程，可先确认 8 张卡均可见：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
"$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
```

不要覆盖平台注入的 `NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA` 或 `NCCL_IB_GID_INDEX`。

### 结果和缓存位置

将可复用结果、日志和 checkpoint 写到持久化路径，例如：

```text
/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/workspace/habit-bench/results/
/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/workspace/habit-bench/results/.cache/
```

不要把唯一 checkpoint、实验结果或日志放在 `/tmp`、容器根目录或本地 NVMe；节点
被回收后这些位置可能丢失。当前环境已经把 Hugging Face、vLLM、TorchInductor 和
Triton 缓存默认指向项目下的 `results/.cache/habitbench`。

## 结束任务

调试完成后停止 RJob，释放 8 张卡：

```bash
source /etc/profile.d/ssh-init.sh
export KUBEBRAIN_CLUSTER_ENTRY=http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451
rjob stop zjm-debug-h200-8gpu-v1-78293916 --namespace ailab-llmarchitecture
```

优先使用 `rjob stop`。只有在明确需要删除任务及其状态/日志元数据时才使用
`rjob delete`，因为删除不可逆。

## 调度和生命周期说明

- 这是 1 个 Replica × 8 GPU，总申请量为 8 张卡；`-P 1` 不代表额外 GPU。
- 第一次申请仍然需要经过调度，`Inqueue` 表示正在等待节点，不是提交失败。
- 绑定后，节点内的后续命令不会逐条重新排队；但任务仍受 H 集群配额和管理员策略
  管理，不是永久独占节点。
- 该任务是 P5 reserved，适用于 Group-AD。不要复制命令改用个人 User-AD；User-AD
  应按 H 规范使用 P1 managed spot 或平台 idle。
- 如需中断后恢复正式实验，checkpoint 必须写在 GPFS2，不能只放在节点本地目录。
