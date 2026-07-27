# Official baseline source snapshots

This directory contains ordinary source snapshots, not nested Git repositories
or submodules. HABIT-Bench keeps the upstream method lifecycle intact and adds
only an evaluator-side chronological ingestion/retrieval adapter.

| snapshot | upstream | pinned revision | files retained |
| --- | --- | --- | --- |
| `vendor/SeCom` | <https://github.com/microsoft/SeCom> | `1738e563b5dc7c51df762247e3d0379f1132ad23` | runtime package, prompts/configs, setup, README and license; notebooks and experiment assets omitted |
| `vendor/O-Mem` | <https://github.com/OPPO-PersonalAI/O-Mem> | `46e131ac39af55d456304c61dfb881717044528e` | runtime source, config example, README, requirements and license; paper images and LoCoMo experiment data omitted |

Graphiti is consumed from the installed official `graphiti-core==0.29.2`
package. Its adapter uses the official `Graphiti.add_episode` and
`Graphiti.search_` APIs. The local Kuzu backend cannot provide Graphiti's BM25
path, so the adapter selects the documented edge-cosine search plus reciprocal
rank fusion configuration. For the local Qwen backend, one HABIT session is one
extraction unit and the adapter caps its completion at 4,096 tokens, adds
`maxItems=16` and `maxLength=512` to otherwise-unbounded response schemas, and
uses an explicit 300-second request timeout with two OpenAI-client retries.
This bounds repetitive constrained decoding; the resulting objects still pass
through graphiti-core's native validation and `add_episode` lifecycle unchanged.

Compatibility code is isolated in `eval/official_adapters`:

- SeCom receives one newly visible HABIT session at a time, while its native
  segmentation, LLMLingua compression and FAISS retrieval modules are used.
- O-Mem uses the official `SimpleMemory` lifecycle. The adapter supplies a
  bounded default for calls without an upstream budget, caps topic-merge JSON
  at 2,048 tokens, and repairs only invalid topic-group references. If a
  topic-merge request reaches the 180-second
  timeout, it retains every input topic as a separate group. This is a
  lossless no-merge fallback for an upstream loop that otherwise retries
  forever; every fallback is counted in `omem_runtime.json`.
- All three adapters return retrieval context only. The fixed HABIT Qwen
  answerer performs choice selection.
