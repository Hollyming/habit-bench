# HABIT-Bench Evaluation Protocol

## Goal

Measure whether a memory method can turn a user's visible longitudinal history
into context that helps a fixed base model answer a current personalized probe.
The primary metric measures final task behavior, not embedding similarity.

## Causal Timeline

For a probe with cutoff `t`, an adapter may ingest only sessions whose
`session_index <= t`. Probes for one user are ordered by cutoff. The memory state
is updated incrementally and is not rebuilt from hidden annotations.

The public finance/software probe file omits the cutoff. Its private key stores
`through_session_index`, which the loader copies into the method-visible
protocol as control metadata. No other private field is copied.

## Base Model

- Model: `/data1/public/hf/Qwen/Qwen3-8B`
- Served name: `habitbench-qwen3-8b`
- Temperature: `0`
- Thinking: disabled
- Maximum input tokens: `40,000`
- Maximum completion tokens: `64`
- Output: JSON object containing `choice_id`
- Memory-LLM completion budget: `4,096` tokens

The evaluator uses the local tokenizer to enforce the context budget. Ranked
memory output is truncated from the right, preserving the highest-ranked prefix.
The full-history control first selects a recent bounded window and then presents
that window chronologically.

## Metrics

Primary:

```text
Accuracy = correct choice_id predictions / all probes
```

Every formal run requires exact probe coverage. Accuracy is accompanied by a
Wilson 95% confidence interval. Grouped Accuracy is reported for available
domain, probe type, capability group, habit family, stress variant, and split
annotations.

Retrieval token counts, storage estimates, latency, and evidence session ids are
diagnostics. They do not select the answer and are not part of primary Accuracy.

## Method Provenance

Each run records implementation kind, upstream source, pinned revision, adapter
note, public dataset hashes, Qwen configuration, command, runtime, and output
coverage. `eval/methods.json` is the human-readable method registry.

An `official` label means the adapter imports official upstream code and calls
its native write/retrieval interface. It does not imply an exact reproduction of
the paper's original model, backend, prompt, or benchmark settings. Material
adaptations are listed in the method registry and run manifest. The stricter
`official_adapted` label is used when backend, online ingestion, or compatibility
changes materially differ from the upstream default path.

## Leakage Rules

The memory method and Qwen answerer must not receive:

- `gold_choice_id`
- hidden habit graphs or persona profiles
- gold evidence session ids
- active/old policy variant labels
- gold action text
- probe type or capability labels from the private key

Only the scorer loads these fields. Tests verify this boundary for both active
domain formats.
