# Implementation changes

## MedMemoryBench

The `wjr` branch adds a retrieval-only contract without replacing the original
method implementations:

- `BaseAgent` and `AgentManager` expose structured retrieval results separately
  from native answer generation.
- Mem0 and A-MEM import the repository's vendored source directly.
- MemOS, MemRL, LightMem, Letta and MIRIX expose their native retrieval output
  through the same contract.
- state is isolated per persona, user, context and task;
- memory-build failures propagate instead of becoming empty successful output;
- persona sharding, checkpoints, strict merge and coverage validation are
  available for MedMemoryBench and LoCoMo;
- LoCoMo can be rescored through one schema-constrained common reader.

Important method fixes include:

- Mem0: local vendored imports, task-scoped Qdrant state, atomic normalization
  of repeated mutations and strict structured-output failure handling.
- A-MEM: preserve the exact local embedding model path and expose native note
  retrieval.
- LightMem: initialize the optional compressor when pre-compression is
  disabled and normalize clipped source IDs before timestamp/speaker lookup.
  These two changes are carried by `patches/lightmem-medmemorybench.patch`
  because LightMem is an external submodule.
- MemOS, MemRL and Letta: isolate persistent state and stop hiding write or
  retrieval failures.
- MIRIX: SQLite cosine support, embedding-dimension handling, deterministic
  engine reset, bounded local JSON tool schemas, canonical tool conversion,
  stale replace-ID normalization and sufficient completion budget for bounded
  memory updates.

The q8a18, q8a19 and q8a20 runtime directories are not source branches.
q8a20 was the cumulative validated snapshot; only its meaningful file-level
delta was merged into the `wjr` history. Runtime logs, caches, databases,
build directories and copied `.git` metadata are intentionally excluded.

## HABIT-Bench

The integration adds:

- seven `medmemorybench_*` entries in `eval/methods.json`;
- `eval.medmemorybench_adapters.structured_memory`, which incrementally ingests
  sessions and returns retrieval-only memory contexts;
- strict rejection of invalid, partial or `success=false` memory writes;
- command-line mappings in `scripts/run_method.sh`;
- contract tests for session markers, dry-run output and failure propagation.

This is a cross-benchmark source adaptation. Scores must not be described as
exact paper reproduction when the backbone, answer reader, judge or evidence
budget differs from the paper configuration.
