#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/plm-shared/zhangjunming/Workspace/HABIT-bench"
PYTHON_BIN="/plm-shared/zhangjunming/miniconda3/envs/habitbenchmark/bin/python"
CLUSTERX_BIN="/plm-shared/zhangjunming/miniconda3/envs/clusterx/bin/clusterx"
SUITE_ROOT="$PROJECT_ROOT/results/habit_3domain_v3"
ENV_FILE="$PROJECT_ROOT/scripts/cluster/env.example.sh"
FINANCE_SOFTWARE_DATASET="$PROJECT_ROOT/domain/finance-software/habit_bench_multidogo_finance_software_scope_consistent_v1.3"
FOOD_DATASET="$PROJECT_ROOT/domain/food/food_habit_lifelines_stress_v4"
QUEUE="queue-t-reserved-plm"
CLUSTER="cluster-t"
MACHINE_TYPE="n3ls.ii.i60a"
IMAGE="registry.pjlab.org.cn/ccr-lepton-official-images/ngc-pytorch:26.04-cu13.2-py3.12-ubuntu24.04"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/create_v3_experiment_plans.py" \
  --suite-root "$SUITE_ROOT"

# Human-audit scoring requires two real annotators. Prepare the blinded,
# stratified artifacts now and leave scoring explicitly pending.
"$PYTHON_BIN" -m eval.supplementary.human_audit prepare \
  --dataset-dir "$FOOD_DATASET" \
  --output-dir "$SUITE_ROOT/supplementary/human_audit/food" \
  --per-stratum 50 --seed 42
"$PYTHON_BIN" -m eval.supplementary.human_audit prepare \
  --dataset-dir "$FINANCE_SOFTWARE_DATASET" \
  --domain-filter finance \
  --output-dir "$SUITE_ROOT/supplementary/human_audit/finance" \
  --per-stratum 50 --seed 42
"$PYTHON_BIN" -m eval.supplementary.human_audit prepare \
  --dataset-dir "$FINANCE_SOFTWARE_DATASET" \
  --domain-filter software \
  --output-dir "$SUITE_ROOT/supplementary/human_audit/software" \
  --per-stratum 50 --seed 42

submit_node() {
  local node_name="$1"
  local port_base="$2"
  local job_name="hb3d-v3-$node_name"
  local plan="$SUITE_ROOT/plans/$node_name/shard_plan.tsv"
  local submit_log="$SUITE_ROOT/plans/$node_name/clusterx_submit.log"
  local node_command

  printf -v node_command \
    'set -euo pipefail; cd %q; %q %q --plan %q --gpus 0,1,2,3,4,5,6,7 --env-file %q --port-base %q --continue-on-group-error; %q %q --plan %q; %q %q --suite-root %q --node %q' \
    "$PROJECT_ROOT" \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_multigpu_plan.py" "$plan" \
    "$ENV_FILE" "$port_base" \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/merge_shard_plan.py" "$plan" \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/finalize_v3_experiment.py" \
    "$SUITE_ROOT" "$node_name"

  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy \
    "$CLUSTERX_BIN" run \
    -J "$job_name" \
    -q "$QUEUE" \
    -C "$CLUSTER" \
    --machine-type "$MACHINE_TYPE" \
    --gpus-per-task 8 \
    --cpus-per-task 64 \
    --memory-per-task 512 \
    --shm-size-gib 64 \
    --no-env \
    --image "$IMAGE" \
    bash -lc "$node_command" 2>&1 | tee "$submit_log"
}

submit_node node01 8100
submit_node node02 8200
submit_node node03 8300

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/create_v3_experiment_plans.py" \
  --suite-root "$SUITE_ROOT" --mark-submitted
