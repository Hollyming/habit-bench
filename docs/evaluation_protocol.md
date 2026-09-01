# HABIT-Bench evaluation protocol

## Goal

Measure whether a memory method can turn a user's visible longitudinal history
into context that helps one fixed base model answer a current personalized
probe. The primary metric measures final task behavior, not embedding
similarity.

## Causal timeline

For a probe with cutoff `t`, an adapter may ingest only sessions whose
`session_index <= t`. Probes for one user are ordered by cutoff. The method
state is updated incrementally and is not rebuilt from hidden annotations.

If an active public dataset omits the cutoff, the loader copies only
`through_session_index` from the private key into evaluation control metadata.
No answer, evidence, habit, capability, or policy label is copied.

## Fixed answer model

- Model: `/plm-shared/zhangjunming/Workspace/models/Qwen3-8B`
- Served name: `Qwen3-8B` (shared by the evaluator and method-side OpenAI
  clients)
- Temperature: `0`
- Thinking: disabled. vLLM applies
  `configs/chat_templates/qwen3_no_thinking.jinja` as the server-wide default
  because the native methods use several different OpenAI clients. This keeps
  their prompts, schemas, temperatures, and memory lifecycles unchanged while
  preventing hidden reasoning from consuming the structured-output budget.
- Default maximum input tokens: `40,000` for the current Qwen3-8B profile.
  Capacity experiments may select another recorded full-memory tier; the same
  resolved maximum is passed to the answerer.
- Maximum answer tokens: `64`
- Memory-method LLM budget: configured separately and recorded in the run
  manifest
- Output: JSON object containing `choice_id`

The local tokenizer enforces the answer context budget. Ranked retrieval output
is truncated from the right, preserving its highest-ranked prefix.
`full_memory` resolves an `auto`, 8k, 16k, 32k, 40k, 64k, 128k or custom
input-window tier. It keeps all visible sessions verbatim while they fit. On
overflow, a query-independent Qwen3-8B compactor consolidates older sessions
into a 4k structured state with source session IDs and preserves a raw recent
suffix in the remaining history budget. The compactor runs before and without
the probe query, choices, gold evidence, or hidden annotations. The former raw
recency-truncation control remains available separately as `full_history`.

## Active methods

The primary MedMemoryBench-source methods are:

```text
mem0, amem, memos, memrl, lightmem, letta, mirix
```

Each uses the pinned MedMemoryBench source at
`third_party/medmemorybench` through one thin chronological adapter. The
adapter does not replace the method's memory construction or retrieval
algorithm. It isolates per-user state, enforces the HABIT input/output contract,
and passes retrieval-only output to the shared answer model.

These integrations are recorded as `benchmark_reproduction`. They use native
MedMemoryBench method code, but local model/back-end compatibility settings
mean they are not exact reproductions of every original paper configuration.
The additional official-source adaptation is:

```text
secom
```

SeCom uses official segmentation, LLMLingua compression and FAISS retrieval
with chronological online ingestion. Its pinned provenance is in
`third_party/official-baselines/README.md`.

Graphiti and O-Mem are explicitly `not_implemented`. They are absent from the
active registry and formal plans because the current Graphiti local
Kuzu/API adaptation has not passed full-lifeline reliability/efficiency
acceptance, while O-Mem exposes unbounded malformed-JSON retry loops and
message-level runtime that exceeds the formal shard timeout. The machine-readable
record is `eval/unsupported_methods.json`.

Finance and Software share the release-gated v1.4 source package but are separate evaluation
views. The plan records `domain_filter=finance` or `domain_filter=software`;
the loader applies that filter before user sharding and merge validates it
again. Results must report Food, Finance, Software, and Travel separately,
rather than one combined Finance–Software score.

`no_memory`, `full_memory`, and `full_history` are evaluator controls rather
than trained memory systems. `full_memory` is the primary compact-history
control; `full_history` is the deterministic raw recency control. Neither is an
oracle or lossless. Legacy `memory_context.v3` runs written under the old
`full_memory` ID must be reported as `40k Full-History (recency-truncated)`;
new compact runs are identified by `memory_context.v5` and reported as
`Full-Memory (online compact history)`. The exact design and comparison rules
are in `docs/compact_context_baseline.md`.

The main protocol also registers five non-agentic, complete-session retrieval
baselines: `recency_5`, `recency_10`, `bm25_rag`, `dense_rag`, and
`temporal_hybrid_rag`. They use the same public cutoff and shared answer head as
the memory systems. Dense retrieval uses the pinned BGE-M3 profile; the hybrid
combines BM25 and dense ranks by RRF and uses an explicit query `as-of` target
when present instead of always preferring the newest session. See
`docs/retrieval_baselines.md` for the fixed parameters.

## Fixed embedding profile

Every active `*_qwen3-8b_adapted.yaml` method profile uses the same local
embedding model so that retrieval-model choice is not confounded with the
memory method:

- model: `BAAI/bge-m3`;
- revision: `5617a9f61b028005a4858fdac845db406aefb181`;
- local path: `/plm-shared/zhangjunming/Workspace/models/bge-m3`;
- dense embedding dimension: `1024`.

The complete method YAML and its SHA-256 are stored in each run manifest and
in the shard-plan manifest. That manifest also records the local model identity
marker, Transformer-config hash, weight filename and byte size. The plan
builder fails before submission if an active profile no longer matches this
pinned BGE-M3 identity or if the local model snapshot is incomplete.

## LoCoMo cross-benchmark

The repository also exposes a Qwen3-8B LoCoMo comparison through the vendored
MedMemoryBench evaluator. `scripts/create_locomo_plan.py` decomposes the ten
official conversations into independent method/sample tasks, while
`scripts/run_locomo_plan.py` provides a GPFS-resumable queue for a 2 × 8-H200
RJob. The active seven memory-agent integrations and the long-context, BM25 and
BGE-M3 controls use the same Qwen3-8B model identity; official LoCoMo F1 is
recomputed from persisted query answers. See `docs/locomo_results_current.md` for
the submission command and artifact layout.

## Metrics

Primary:

```text
Accuracy = correct choice_id predictions / all probes
```

Every formal run requires exact probe coverage. Accuracy is accompanied by a
Wilson 95% confidence interval. Grouped Accuracy is reported for available
domain, probe type, capability group, habit family, stress variant, and split
annotations.

Retrieval token counts, storage estimates, latency, and evidence session IDs
are diagnostics. They do not select the answer and are not part of primary
Accuracy.

## Provenance

Each run records implementation kind, upstream source, pinned revision,
adapter note, public dataset hashes, Qwen configuration, command, runtime, and
output coverage. `eval/methods.json` is the canonical method registry, while
`third_party/medmemorybench/VENDOR_INFO.md` freezes the source revisions and
local compatibility patch. Official baseline revisions and compatibility
boundaries are recorded in `third_party/official-baselines/README.md`.

For sharded execution, `run_manifest.json` records per-shard wall-clock time,
shard index/count, host, and visible GPU. `merge_manifest.json` records summed
compute time, maximum shard time, and the observed parallel window.
`suite_runtime.json` adds vLLM startup and method/domain group wall-clock time;
`evaluation_summary.json` joins those timings to the final scores.

## Leakage rules

The memory method and Qwen answerer must not receive:

- `gold_choice_id`;
- hidden habit graphs or persona profiles;
- gold evidence session IDs;
- active/old policy variant labels;
- gold action text;
- probe type or capability labels from the private key.

Only the scorer loads these fields. Tests verify this boundary for both active
domain formats.
