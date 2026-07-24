# Compact experiment record

This is the shareable summary of the longer cross-repository working note
`AGENTS_MedMem.md`. It intentionally omits task-by-task logs and private
credential paths.

## Frozen scope

- MedMemoryBench upstream base: `7227bc105b84a1a9f7a75861eb9e1be3ea502882`
- MedMemoryBench integration: `6591eb3251402f26535846ea4a95f5b4478ae35a`
- HABIT-Bench upstream base: `41740d69b5d6030a5d2f9c75f8a0dbff732ae811`
- Qwen3-8B revision: `b968826d9c46dd6066d109eabc6255188de91218`
- MedMemoryBench data: 20 personas and 1,986 queries
- HABIT food: 30 users, 1,410 sessions and 1,260 probes
- HABIT finance-software: 45 users, 14,400 sessions and 810 probes

## MedMemoryBench Qwen3-8B adapted

| method | Efficient | Mixed |
|---|---:|---:|
| Mem0 | 480/1986 = 24.1692% | 437/1986 = 22.0040% |
| A-MEM | 990/1986 = 49.8489% | 883/1986 = 44.4612% |
| MemOS | 727/1986 = 36.6062% | 635/1986 = 31.9738% |
| MemRL | 788/1986 = 39.6777% | 688/1986 = 34.6425% |
| LightMem | 642/1986 = 32.3263% | 620/1986 = 31.2185% |
| Letta | 1018/1986 = 51.2588% | 930/1986 = 46.8278% |
| MIRIX | 492/1986 = 24.7734% | pending final strict merge at documentation time |

The completed methods all decline from Efficient to Mixed, matching the
expected direction of memory saturation/noise. Absolute scores are not exact
paper reproduction because this run uses a Qwen3-8B adapted reader rather than
the paper's closed-model stack.

## HABIT-Bench method-native retrieval

| method | food | finance-software |
|---|---:|---:|
| Mem0 | 879/1260 = 69.7619% | 371/810 = 45.8025% |
| A-MEM | 1014/1260 = 80.4762% | 427/810 = 52.7160% |
| MemOS | 935/1260 = 74.2063% | 364/810 = 44.9383% |
| MemRL | 1016/1260 = 80.6349% | 456/810 = 56.2963% |
| LightMem | 936/1260 = 74.2857% | 364/810 = 44.9383% |
| Letta | 1023/1260 = 81.1905% | 442/810 = 54.5679% |
| MIRIX | 887/1260 = 70.3968% | 384/810 = 47.4074% |

All formal merges used unique probe coverage and method-native retrieval
followed by the shared HABIT answerer. Evidence lengths differ by method, so
this table is an end-to-end comparison rather than an equal-token retrieval
ablation.

## LoCoMo common reader

| method | official mean F1 | F1>=0.5 Accuracy |
|---|---:|---:|
| A-MEM | 0.353539 | 703/1986 = 35.3978% |
| Mem0 | 0.269810 | 540/1986 = 27.1903% |
| LightMem | 0.261031 | 524/1986 = 26.3847% |
| MemOS | 0.341059 | 689/1986 = 34.6928% |
| MemRL | 0.250684 | 497/1986 = 25.0252% |
| Letta | 0.352849 | 703/1986 = 35.3978% |
| MIRIX | 0.244402 | 487/1986 = 24.5217% |

All common-reader artifacts cover 1,986 unique queries, with no empty answers
or length-truncated completions. Method-native and common-reader scores must be
reported separately.

## Interpretation rules

- Do not infer implementation correctness from one close aggregate score.
- Do not compare scores across different readers, judges or evidence budgets
  without labeling the protocol difference.
- A formal result requires complete coverage, per-item artifacts, hashes and a
  zero-failure audit.
- Old failed or stopped q8 shards are never included in a formal merge.
