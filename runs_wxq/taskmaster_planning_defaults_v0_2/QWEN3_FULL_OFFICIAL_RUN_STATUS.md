# Qwen3-8B Full-Official Evaluation Status

Updated: 2026-07-13 23:41 (Asia/Shanghai)

## Current status

- No active GPU job. Job `84320` was stopped after the 4096-token continuous-update smoke also reached the generation ceiling.
- GPU: one A100 40 GB
- Internal memory LLM: `/data1/public/hf/Qwen/Qwen3-8B`
- Embedding model: local `intfloat/e5-base-v2`, 768 dimensions, matching the archived reference result
- Dataset: `runs_wxq/taskmaster_planning_defaults_v0_2`
- Formal output directory is intentionally empty; failed capacity-smoke outputs are isolated under `failed_attempts/`.
- Mem0: official `Memory.add(infer=True)` then official search
- Graphiti: official `add_episode`/Kuzu in four user shards, merged before unchanged scoring, matching the archived reference execution shape
- Final answer head: unchanged HABIT-Bench lexical scoring over retrieved memory
- A full benchmark run is not currently scientifically valid on the coarse v0.2 session units.

## Why the output ceiling differs

The archived reference dataset averages about 30 Qwen tokens per session; the wxq dataset averages about 814. A strict 256-token run produced `finish_reason=length`, malformed JSON, and only 41 memory points after roughly 450 sessions.

A 24-session length-stratified pilot using Mem0's exact initial extraction prompt produced 24/24 valid JSON responses, P95 485 completion tokens and maximum 528. Continuous-update smokes then showed that prompts containing accumulated existing memories can exceed 1024, 2048, and finally 4096 tokens. The 4096 failure ended around 16155 characters and took close to a minute for one response.

The run was therefore stopped. The next valid step is to split/rebuild the dataset into reference-like memory-write sessions and retain the reference budget, rather than increasing the ceiling again.

Pilot evidence: `reports/qwen3_mem0_extraction_budget_pilot.json`.

## Alignment statement

The method packages, write/update paths, retrieval, top-k, Mem0 threshold, structured output, Kuzu backend, lexical answer head, and external evaluator/collector remain unchanged. Relative to the archived reference result, the intentional differences are:

1. dataset content and path;
2. Qwen3-8B replaces Qwen2.5-14B;
3. the 4096-token generation ceiling is calibrated for the much larger session unit.

Failed capacity-smoke outputs are isolated under `failed_attempts/` and are excluded from formal collection/audit. No file under `scripts/`, `runs/`, or `eval/` was modified.
