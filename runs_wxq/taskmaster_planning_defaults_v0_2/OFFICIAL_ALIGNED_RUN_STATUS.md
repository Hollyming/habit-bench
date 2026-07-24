# Reference-aligned Official Run Status

Status: **blocked before predictions**

The launcher was executed with the repository `habit-official` environment:

```bash
PATH="$HOME/conda_envs/habit-official/bin:$PATH" \
  bash scripts_wxq/run_taskmaster_planning_defaults_v02_official_adapters.sh
```

The launcher is identical to `scripts/run_official_subset_adapters.sh` except
for the default dataset path. The run stopped in the first Mem0 adapter while
resolving the reference default embedding model
`sentence-transformers/all-MiniLM-L6-v2`: `hf-mirror.com` returned repeated SSL
EOF errors.

`proxy_on` was then tested explicitly against both the configured
`hf-mirror.com` endpoint and `https://huggingface.co`; both returned the same
SSL EOF failure in the official Python environment.

No predictions were completed or scored. The partial `official_results/`
directory was removed so it cannot be mistaken for a valid result.

Do not enable the previous wxq offline-hash shim or add adapter-specific model
arguments to bypass this failure; either action would break reference
alignment. Cache/expose the reference embedding model in the official
environment, then rerun the same launcher unchanged.
