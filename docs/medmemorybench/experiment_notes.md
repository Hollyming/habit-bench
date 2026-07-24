# 实验记录

## 冻结范围

- MedMemoryBench 上游基线：`7227bc105b84a1a9f7a75861eb9e1be3ea502882`
- MedMemoryBench 接入版本：`6591eb3251402f26535846ea4a95f5b4478ae35a`
- HABIT-Bench 上游基线：`41740d69b5d6030a5d2f9c75f8a0dbff732ae811`
- Qwen3-8B revision：`b968826d9c46dd6066d109eabc6255188de91218`
- MedMemoryBench：20 personas、1,986 queries
- HABIT food：30 users、1,410 sessions、1,260 probes
- HABIT finance-software：45 users、14,400 sessions、810 probes

## MedMemoryBench Qwen3-8B adapted

| 方法 | Efficient | Mixed |
|---|---:|---:|
| Mem0 | 480/1986 = 24.1692% | 437/1986 = 22.0040% |
| A-MEM | 990/1986 = 49.8489% | 883/1986 = 44.4612% |
| MemOS | 727/1986 = 36.6062% | 635/1986 = 31.9738% |
| MemRL | 788/1986 = 39.6777% | 688/1986 = 34.6425% |
| LightMem | 642/1986 = 32.3263% | 620/1986 = 31.2185% |
| Letta | 1018/1986 = 51.2588% | 930/1986 = 46.8278% |
| MIRIX | 492/1986 = 24.7734% | 397/1986 = 19.9899% |

七种方法均表现为 Mixed 低于 Efficient，方向与 memory saturation/noise 预期一致。由于该实验使用 Qwen3-8B adapted reader，而不是论文中的闭源模型组合，绝对分数不能称为论文 exact reproduction。

## HABIT-Bench 方法原生检索

| 方法 | food | finance-software |
|---|---:|---:|
| Mem0 | 879/1260 = 69.7619% | 371/810 = 45.8025% |
| A-MEM | 1014/1260 = 80.4762% | 427/810 = 52.7160% |
| MemOS | 935/1260 = 74.2063% | 364/810 = 44.9383% |
| MemRL | 1016/1260 = 80.6349% | 456/810 = 56.2963% |
| LightMem | 936/1260 = 74.2857% | 364/810 = 44.9383% |
| Letta | 1023/1260 = 81.1905% | 442/810 = 54.5679% |
| MIRIX | 887/1260 = 70.3968% | 384/810 = 47.4074% |

正式 merge 均校验 unique probe coverage。各方法先执行原生 retrieval，再交给统一 HABIT answerer。不同方法返回的 evidence 长度并不相同，因此该表是端到端比较，不是等 token retrieval ablation。

## LoCoMo common reader

| 方法 | official mean F1 | F1>=0.5 Accuracy |
|---|---:|---:|
| A-MEM | 0.353539 | 703/1986 = 35.3978% |
| Mem0 | 0.269810 | 540/1986 = 27.1903% |
| LightMem | 0.261031 | 524/1986 = 26.3847% |
| MemOS | 0.341059 | 689/1986 = 34.6928% |
| MemRL | 0.250684 | 497/1986 = 25.0252% |
| Letta | 0.352849 | 703/1986 = 35.3978% |
| MIRIX | 0.244402 | 487/1986 = 24.5217% |

所有 common-reader 产物均覆盖 1,986 个 unique queries，没有空答案或因长度截断的 completion。method-native 与 common-reader 分数必须分列报告。

## 解释原则

- 不能因为单一总分接近论文就判定实现完全正确。
- reader、judge 或 evidence budget 不同时，必须明确标注协议差异。
- 正式结果必须同时具备完整 coverage、逐题产物、hash 和零失败审计。
- failed、stopped 或旧 q8 partial 不参与正式合并。
