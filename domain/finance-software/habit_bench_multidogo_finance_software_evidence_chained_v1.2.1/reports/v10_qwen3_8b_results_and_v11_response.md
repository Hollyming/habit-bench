# Qwen3-8B v1.0 Results Supplied by the Project Team and the v1.1 Design Response

The following numbers were supplied by the project team and were not rerun during packaging of v1.1.

| v1.0 probe type | Questions | No memory | 34k history | Change |
|---|---:|---:|---:|---:|
| `composition_pair_latent` | 704 | 25.99% | 34.66% | +8.66 pp |
| `dual_drift_asof` | 300 | 27.00% | 27.33% | +0.33 pp |
| `evidence_weighted_pair` | 108 | 24.07% | 36.11% | +12.04 pp |
| `priority_triple_hard` | 210 | 53.81% | 53.81% | 0.00 pp |
| `session_continuation_reconstruction` | 270 | 25.56% | 33.33% | +7.78 pp |
| `temporal_composition_triple` | 108 | 64.81% | 66.67% | +1.85 pp |
| `temporal_interference_pair` | 348 | 25.29% | 34.48% | +9.20 pp |
| **Overall** | **2,048** | **30.76%** | **37.11%** | **+6.35 pp** |

v1.1 responds by removing the two >50% categories, replacing all probe types, rewriting 4,816 history sessions, splitting each durable policy across distant sessions, and ensuring that neither a 205-session window nor a conservative 140,000-character contiguous history window contains a complete required evidence set.

The requested target—approximately 20% or lower for no-memory and below 30% for the project’s 34k-history Qwen3-8B run—remains an empirical release gate. Construction-time proxy audits are not substituted for that target-model measurement.
