# Taskmaster Planning Defaults v0.2 Official Results Status

## Evaluation Contract

Formal official results for this dataset use exactly the repository reference
workflow in `scripts/run_official_subset_adapters.sh`. The wxq launcher changes
only the default dataset path:

```bash
bash scripts_wxq/run_taskmaster_planning_defaults_v02_official_adapters.sh
```

It directly invokes:

- `eval/run_external_baseline.py`
- `eval/official_adapters/official_mem0_adapter.py`
- `eval/official_adapters/official_amem_adapter.py`
- `eval/official_adapters/official_secom_adapter.py`
- `eval/official_adapters/official_graphiti_adapter.py`
- `eval/official_adapters/official_omem_adapter.py`
- `eval/collect_official_results.py`

No wxq-specific answer model, prompt, scorer, embedding shim, or offline-hash
fallback is part of the formal contract.

## Current Status

Previously generated Qwen full-history/memory-matrix and non-reference
offline-hash result directories were removed to prevent accidental comparison
with reference-aligned results.

The `official_results/` directory should be cited only after rerunning the
command above successfully in an environment satisfying the same reference
adapter dependencies. Until then, the valid completed formal results are the
unchanged lightweight reference baselines under `baseline_results/`.

The current default-Python reference preflight is stored under
`official_adapter_status_reference/`. Mem0 and Graphiti are not runnable in
that interpreter because their official packages are missing; A-MEM, SeCom,
and O-Mem repositories are present. The aligned launcher intentionally does
not work around this with wxq shims or offline embeddings. Activate/install the
same official environment intended for the reference run, then execute the
launcher.

An aligned attempt using the existing `habit-official` environment passed the
dependency preflight but stopped before predictions because the reference
MiniLM embedding download failed with an SSL EOF from `hf-mirror.com`. The
partial result directory was removed. See `OFFICIAL_ALIGNED_RUN_STATUS.md`.

Dataset readiness is recorded in:

- `reports/reference_alignment_audit.json`
- `reports/reference_alignment_audit.md`

The dataset passes structural and reference-loader compatibility checks. Its
content gate is judged only with the unchanged reference no-memory baseline;
see the alignment audit for the current result. Human review status and the
limited 30-user scale should still be reported when citing the slice.
