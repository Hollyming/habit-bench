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
rank fusion configuration. Entity/edge extraction retains graphiti-core's
official 16,384-token completion default; applying the smaller generic
memory-method cap can truncate schema-constrained extraction JSON. For the
local Qwen backend, the adapter adds `maxItems=64` and `maxLength=1000` to
otherwise-unbounded response schemas. This prevents repetitive constrained
decoding from exhausting that budget; the resulting objects still pass through
graphiti-core's native validation and `add_episode` lifecycle unchanged.

Compatibility code is isolated in `eval/official_adapters`:

- SeCom receives one newly visible HABIT session at a time, while its native
  segmentation, LLMLingua compression and FAISS retrieval modules are used.
- O-Mem uses the official `SimpleMemory` lifecycle. The adapter specifies JSON
  response contracts at the OpenAI-compatible API boundary and repairs only
  invalid topic-group references emitted by the local backend.
- All three adapters return retrieval context only. The fixed HABIT Qwen
  answerer performs choice selection.
