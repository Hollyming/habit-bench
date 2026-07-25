# MedMemoryBench vendored source

- Upstream: `https://github.com/AQ-MedAI/MedMemoryBench`
- Integration commit: `6591eb3251402f26535846ea4a95f5b4478ae35a`
- Baseline commit: `7227bc105b84a1a9f7a75861eb9e1be3ea502882`
- Imported as a source snapshot: 2026-07-24
- Nested Git metadata: intentionally omitted

The snapshot contains the MedMemoryBench evaluation framework and the native
source used by the seven active HABIT methods: Mem0, A-MEM, MemOS, MemRL,
LightMem, Letta and MIRIX.

LightMem is copied from upstream revision:

```text
a19ea88df47c73fd2f55d27f64616467ef576a81
```

The recorded compatibility patch at
`patches/lightmem-medmemorybench.patch` is already applied to the vendored
LightMem files. The original patch is retained for audit but must not be applied
a second time.

The seven `*_qwen3-8b_adapted.yaml` profiles use HABIT-local model paths:

```text
/plm-shared/zhangjunming/Workspace/models/bge-m3
/plm-shared/zhangjunming/Workspace/models/llmlingua-2-xlm-roberta-large-meetingbank
```

The shared embedding is `BAAI/bge-m3`, dimension 1024, pinned in the profile
snapshots to Hugging Face revision
`5617a9f61b028005a4858fdac845db406aefb181`.

Method source should remain close to this frozen snapshot. New HABIT behavior
belongs in the outer adapter unless an upstream compatibility defect makes a
source patch unavoidable; every such patch must be documented here.
