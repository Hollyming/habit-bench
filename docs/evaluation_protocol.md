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
`full_memory` is different: it resolves an `auto`, 8k, 16k, 32k, 40k, 64k,
128k or custom input-window tier from the configured model capacity. The tier
reserves space for the system prompt, current probe and choices; the remainder
is the history budget. It includes all visible sessions when they fit. On
overflow it keeps the most recent complete sessions, then presents that suffix
chronologically. Only when one session alone exceeds the budget may its oldest
content be truncated. `full_history` is retained as a compatibility alias.

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
Additional official-source adaptations are:

```text
graphiti, secom, omem
```

Graphiti uses official `add_episode`/`search_` with local Kuzu
edge-cosine/RRF retrieval. SeCom uses official segmentation, LLMLingua
compression and FAISS retrieval with chronological online ingestion. O-Mem
uses the official `SimpleMemory` lifecycle plus documented JSON-output
compatibility handling at the local API boundary. Their pinned provenance is
in `third_party/official-baselines/README.md`.

`no_memory` and `full_memory` are evaluator controls, not memory methods.
`full_memory` performs no training, learned summarization, retrieval, or
embedding. The deterministic recent-session truncation makes the context-window
constraint explicit without adding another model whose errors would confound
the long-context baseline. This follows the full-history long-context control
used by [LongMemEval](https://github.com/xiaowu0162/longmemeval), with an
explicit recency fallback for HABIT lifelines that exceed the available
window.

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
