# v1.1 Probe Design, Evidence Topology and Shortcut Audit

## 1. Motivation

v1.0 的目标模型结果显示：`priority_triple_hard` 与 `temporal_composition_triple` 可在不读取历史时取得 50%–60% 以上准确率，另一些题型在约 34k history 下提升过大。v1.1 因此同时改变历史证据表示和问题设计，而不是只改题面。

## 2. Split-decision evidence topology

每个 durable policy 被拆成远距离连接：

```text
early ordered shortlist of two safe workflows
                +
distant ordinal resolution selecting first/second
                =
recoverable durable policy
```

任何单个 session 都不包含完整 policy realization。对 drift habit，replacement 同样由另一组 shortlist / resolution 组成。用户历史中还包含：

- nonresolving rehearsal；
- single-case exception；
- assistant-originated proposal later rejected as a standing rule；
- cross-session case records with one closed and one paused workstream；
- locally convenient but nonbinding workflows。

## 3. Evidence-window results

### Session-count proxy

- Window: any contiguous 205 sessions
- Complete probes: 0 / 2,048
- Minimum unresolved independent groups in the best window: 2

### Character-budget proxy

- Window: any contiguous 140,000 history characters
- Complete probes: 0 / 2048
- Median unresolved groups in best window: 2.0
- Minimum unresolved groups in best window: 2
- History characters / user: 311,624–345,552

These are evidence-availability audits, not model scores.

## 4. Probe distribution

```json
{
  "suggestion_rejection_pair": 256,
  "reference_case_reconstruction": 320,
  "triple_asof_interleaved": 512,
  "dual_asof_reversal": 384,
  "surface_decoy_pair": 384,
  "scope_temporal_pair": 64,
  "provenance_weighted_triple": 128
}
```

Removed as standalone categories:

```text
priority_triple_hard
temporal_composition_triple
single-habit direct use
boundary-only
exception-only
weak-signal meta question
```

## 5. Difficulty principles

1. Four choices are safe, domain-valid and task-complete.
2. The query does not state the durable policy or whether a historical signal was binding.
3. Surface cues may favor a plausible decoy workflow.
4. Gold recovery requires 2–3 independent policy joins.
5. Historical-time questions require reconstructing the policy state as of that time, not merely using the latest state.
6. User ratification outranks assistant proposals; a one-case exception does not automatically become a default.
7. Choice labels are perfectly balanced.

## 6. Construction-time no-history audits

- TF-IDF choice-only: 14.65%
- TF-IDF query+choices: 14.79%
- Longest: 25.20%
- Shortest: 24.51%
- Query overlap: 23.49%
- Safety lexicon: 24.12%
- Surface-decoy prior: 0.00%

These are construction proxies. They do not establish the requested Qwen3-8B target score.

## 7. Release gate

The package passes strict structural validation and scorer smoke testing. It remains a candidate until the same Qwen3-8B no-memory / 34k-history protocol used for v1.0 is rerun and human review is completed.
