# v1.3 reviewer-feedback patch

## 1. Time-scope binding

All **64** `scope_temporal_pair` probes previously began with an `As of ...` phrase before the first task even though private `target_state_times` assigned the first task to the current policy and the second task to the historical policy. The wording was therefore structurally ambiguous.

v1.3 rewrites each query so that:

```text
first workstream  -> current standing policy
second workstream -> policy in force at the exact historical timestamp
```

The change does not disclose either workflow variant. It only makes the intended temporal operator well-formed.

## 2. Account-review scope and replacement

The source habit contract already covered balance, statement, transaction-history, and card-activity review. For `mdgo_v05_fin_user_0006`, however, the replacement shortlist and resolution used statement examples while many probes used card-activity wording. Four visible evidence sessions are revised so that:

- both statement and card-activity cases belong to the same account-review habit scope;
- `POLR-winter-docket-giisxq` is explicitly a family-wide replacement;
- it supersedes `POLB-winter-docket-giisxq` across that scope;
- the replacement is not a statement-only local exception.

The affected replacement decision unit appears in **29** probes, including all 10 reviewer-flagged probes. Their gold labels and choice signatures are unchanged because the revised visible evidence now unambiguously supports the existing replacement policy.

## 3. Repeated latent-decision weighting

v1.3 does not delete valid compositional probes, because the same latent decision can be tested under different companion habits and temporal interactions. Instead, every probe now carries private `decision_unit_ids`, where one unit corresponds to a unique `(user, habit, decisive evidence pair)`.

The scorer still reports ordinary exact probe accuracy, and additionally reports `decision_unit_macro_accuracy`. Each decision unit contributes equally to that macro metric regardless of how many probes reuse it. This directly prevents one ambiguous or difficult latent decision from being counted 10–30 times in the recommended chain-balanced aggregate.

## 4. Invariants retained

- 54 users;
- 29,160 sessions;
- 2,048 probes;
- A/B/C/D gold labels remain 512 each;
- all choices and gold answer texts remain unchanged;
- evidence IDs remain private;
- scoring remains strict choice-ID exact match.
