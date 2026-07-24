# v0.4 unified evaluation

This directory evaluates only the finalized Taskmaster travel release:

```text
runs_wxq/taskmaster_planning_defaults_v0_4/
  public/lifelines.jsonl
  public/probes.jsonl
  private/probe_key.jsonl
```

It deliberately reuses the repository-wide `eval/` contract described in the
root README.  A method receives public sessions and public probes only, returns
`memory_context`, and the shared Qwen3-8B answer stage selects a choice.  Gold
labels are read only by the scorer.

## Before a full run

Start the fixed Qwen3-8B server on a GPU node, as documented by the root
README.  The expected served name is `habitbench-qwen3-8b` and the default URL
is `http://127.0.0.1:8000/v1`.

Validate the release first:

```bash
python scripts_wxq/evaluation/validate_v04.py
```

## Run a method

```bash
# Small smoke test: one user, four probes.
bash scripts_wxq/evaluation/run_v04.sh no_memory smoke_no_memory \
  --max-users 1 --max-probes 4

# Full controls.  They should be reported alongside memory methods.
bash scripts_wxq/evaluation/run_v04.sh no_memory full_no_memory
bash scripts_wxq/evaluation/run_v04.sh full_history full_history

# Native memory methods, using the repository's standardized adapters.
bash scripts_wxq/evaluation/run_v04.sh mem0 mem0_topk5
bash scripts_wxq/evaluation/run_v04.sh graphiti graphiti_topk5
bash scripts_wxq/evaluation/run_v04.sh amem amem_topk5
bash scripts_wxq/evaluation/run_v04.sh secom secom_default
bash scripts_wxq/evaluation/run_v04.sh omem omem_default
```

Results are stored under
`runs_wxq/taskmaster_planning_defaults_v0_4/evaluation_results/<run-name>/`.
Each run contains the public method input, memory contexts, predictions, strict
scored predictions, overall metrics, group metrics, and a provenance manifest.

## Score predictions produced elsewhere

Predictions must be JSONL with exactly one row per probe:

```json
{"probe_id":"...", "choice_id":"A"}
```

Use:

```bash
bash scripts_wxq/evaluation/score_v04_predictions.sh METHOD_NAME \
  /absolute/path/to/predictions.jsonl RUN_NAME
```

The scorer enforces exact coverage and valid choice IDs.  It never exposes the
private key to the evaluated method.

## Comparison rule

Primary metric is exact-choice Accuracy with Wilson 95% confidence intervals.
Always report `no_memory` and `full_history` beside a memory method.  Compare
the same finalized v0.4 dataset hash and the same Qwen server configuration;
do not compare runs with different subsets as if they were full-run results.
