# official_results Baseline Report

This is an external official-code adapter run, not a lightweight proxy baseline.
Adapter note: Official SeCom repository code is used for retrieval through SeCom.retrieve_external_memory with session-level BM25 and compression/segmentation disabled; vLLM is shimmed only because the unused LocalLLM class is imported at module load on Windows. The answer head is HABIT-Bench lexical choice scoring, so this is an official-code retrieval adapter, not full SeCom paper reproduction.

## Accuracy By Capability

| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_secom_bm25_session | 0.833 | 0.764 | 0.569 | 0.819 | 0.562 | 0.000 |

## Diagnostic Gap

| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |
| --- | ---: | ---: | ---: | ---: |
| official_secom_bm25_session | 0.833 | 0.612 | 0.221 | 0.456 |

## Variant Sensitivity

| baseline | original reviewed acc | unseen paraphrase acc |
| --- | ---: | ---: |
| official_secom_bm25_session | 0.702 | 0.648 |

## Cost Proxy

| baseline | avg retrieved tokens | avg stored items |
| --- | ---: | ---: |
| official_secom_bm25_session | 248.4 | 56.8 |

## Interpretation

The table reports HABIT-Bench metrics for `official_secom_bm25_session` under the adapter contract above. Interpret results according to the adapter note rather than as a full paper-reproduction claim.
Official conclusions should be made only for the exact method configuration represented by this adapter run.
