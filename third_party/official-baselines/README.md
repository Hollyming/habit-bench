# Official baseline source snapshots

This directory contains ordinary source snapshots, not nested Git repositories
or submodules. HABIT-Bench keeps the upstream method lifecycle intact and adds
only an evaluator-side chronological ingestion/retrieval adapter.

| snapshot | upstream | pinned revision | files retained |
| --- | --- | --- | --- |
| `vendor/SeCom` | <https://github.com/microsoft/SeCom> | `1738e563b5dc7c51df762247e3d0379f1132ad23` | runtime package, prompts/configs, setup, README and license; notebooks and experiment assets omitted |

Compatibility code is isolated in `eval/official_adapters`:

- SeCom receives one newly visible HABIT session at a time, while its native
  segmentation, LLMLingua compression and FAISS retrieval modules are used.
- The adapter returns retrieval context only. The fixed HABIT Qwen
  answerer performs choice selection.

Graphiti and O-Mem are not active snapshots or runnable methods. Their known
reliability and scaling blockers are recorded in
`eval/unsupported_methods.json`; their adapters and vendored O-Mem runtime were
removed until bounded full-domain implementations are available.
