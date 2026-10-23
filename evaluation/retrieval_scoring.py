from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .dataset import DatasetBundle, DatasetContractError


RETRIEVAL_K = 5
RANKED_METHOD_EXCLUSIONS = {"no_memory", "full_memory", "full_history"}

RETRIEVAL_METRIC_DEFINITIONS = {
    "evidence_recall_at_5_macro": (
        "Macro-average over probes of the fraction of positive gold evidence sessions "
        "found in the first five unique attributed session IDs. Food uses "
        "gold_evidence_session_ids; Finance/Software uses "
        "decision_evidence_session_ids."
    ),
    "evidence_recall_at_5_micro": (
        "Total positive evidence hits in the first five ranks divided by the total "
        "number of positive gold evidence sessions."
    ),
    "evidence_precision_at_5": (
        "Positive evidence hits divided by five, macro-averaged over probes. Missing "
        "or unattributed ranks therefore consume retrieval capacity."
    ),
    "evidence_coverage_efficiency_at_5": (
        "Evidence hits divided by min(5, number of gold evidence sessions). This "
        "separates ranking quality from the unavoidable Recall@5 ceiling when a "
        "Finance probe has six decisive sessions."
    ),
    "evidence_hit_rate_at_5": "Fraction of probes with at least one positive evidence hit.",
    "evidence_mrr_at_5": "Mean reciprocal rank of the first positive evidence hit, truncated at rank 5.",
    "evidence_ndcg_at_5": "Binary-relevance normalized discounted cumulative gain at rank 5.",
    "full_evidence_rate_at_5": (
        "Fraction of probes for which every positive gold evidence session is present "
        "in the top five. It is necessarily zero when a probe has more than five "
        "positive sessions."
    ),
    "source_attribution_probe_coverage": (
        "Fraction of probes for which the method exposes at least one valid, visible "
        "source session ID. A method that retrieves summaries without provenance is "
        "measurably distinct from a method that retrieves no relevant evidence."
    ),
    "joint_answer_evidence_hit_rate_at_5": (
        "Fraction of probes whose answer is correct and whose top five contains at "
        "least one positive evidence session."
    ),
    "joint_answer_full_evidence_rate_at_5": (
        "Fraction of probes whose answer is correct and whose top five contains every "
        "positive evidence session."
    ),
    "attribution_conditional_evidence_recall_at_5": (
        "Evidence Recall@5 restricted to probes for which the method exposes at least "
        "one valid source session. Compare with unconditional recall to distinguish "
        "provenance loss from poor ranking after attribution succeeds."
    ),
    "evidence_utility_gap_at_5": (
        "Answer accuracy on probes with an evidence hit minus answer accuracy on "
        "probes without a hit. Positive values indicate that successful retrieval is "
        "associated with useful answer evidence."
    ),
    "context_evidence_recall_macro": (
        "For the full-memory control only: positive evidence coverage anywhere in the "
        "selected long-context window. It is not a ranked-retrieval metric."
    ),
    "context_evidence_recall_micro": (
        "For the full-memory control only: total positive evidence sessions retained "
        "anywhere in the selected window divided by all positive evidence sessions."
    ),
    "context_full_evidence_rate": (
        "For the full-memory control only: fraction of probes whose selected window "
        "contains every positive evidence session."
    ),
    "joint_answer_context_evidence_hit_rate": (
        "For the full-memory control only: fraction of probes with a correct answer "
        "and at least one positive evidence session in the selected window."
    ),
    "component_hit_coverage_at_5": (
        "Finance/Software only: fraction of required habit components represented by "
        "at least one of their decisive sessions in the top five."
    ),
    "component_complete_coverage_at_5": (
        "Finance/Software only: fraction of required habit components for which every "
        "decisive weak-evidence session is present in the top five."
    ),
    "complete_chain_rate_at_5": (
        "Finance/Software only: fraction of probes for which every required component "
        "is completely supported by its decisive weak-evidence sessions in the top five."
    ),
    "temporal_context_recall_at_5": (
        "Finance/Software only: recall of separately annotated temporal context. It is "
        "diagnostic and is not mixed into decisive-evidence Recall@5."
    ),
    "nonbinding_intrusion_rate_at_5": (
        "Finance/Software only: fraction of the five ranks occupied by local "
        "exceptions or unratified assistant suggestions that must not be treated as "
        "durable user memory."
    ),
    "contextual_evidence_ndcg_at_5": (
        "Finance/Software only: graded nDCG that values decisive evidence above "
        "temporal context and assigns no gain to nonbinding evidence."
    ),
    "decisive_decoy_discrimination_at_5": (
        "Finance/Software only: decisive evidence precision minus nonbinding evidence "
        "intrusion in the top five; higher is better."
    ),
    "clean_evidence_hit_rate_at_5": (
        "Finance/Software only: fraction of probes with a decisive evidence hit and "
        "no nonbinding evidence in the top five."
    ),
    "clean_grounded_answer_rate_at_5": (
        "Finance/Software only: correct answer, at least one decisive evidence hit, "
        "and no nonbinding evidence in the top five."
    ),
    "decision_unit_macro_accuracy": (
        "Finance/Software only: accuracy averaged within each unique user-habit "
        "decision unit and then macro-averaged across decision units."
    ),
    "decision_unit_macro_evidence_recall_at_5": (
        "Finance/Software only: component-specific Recall@5 averaged within each "
        "decision unit and then macro-averaged across decision units."
    ),
    "decision_bundle_macro_accuracy": (
        "Finance/Software only: answer accuracy averaged within each multi-habit "
        "decision bundle and then macro-averaged across bundles."
    ),
    "decision_bundle_macro_evidence_recall_at_5": (
        "Finance/Software only: decisive evidence Recall@5 averaged within each "
        "multi-habit decision bundle and then macro-averaged across bundles."
    ),
}


def _mean(values: list[float | int | bool]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _dcg(relevances: list[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances)
    )


def _ndcg(binary_relevance: list[int], gold_count: int, k: int) -> float:
    actual = _dcg(binary_relevance[:k])
    ideal = _dcg([1] * min(gold_count, k))
    return _ratio(actual, ideal)


def _graded_ndcg(
    ranked_ids: list[str],
    decisive: set[str],
    temporal: set[str],
    k: int,
) -> float:
    relevance = [
        2 if session_id in decisive else 1 if session_id in temporal else 0
        for session_id in ranked_ids[:k]
    ]
    ideal_relevance = sorted(
        [2] * len(decisive) + [1] * len(temporal),
        reverse=True,
    )[:k]
    return _ratio(_dcg(relevance), _dcg(ideal_relevance))


def _retrieval_mode(method_name: str) -> str:
    if method_name == "no_memory":
        return "none"
    if method_name in {"full_memory", "full_history"}:
        return "context"
    return "ranked"


def _reported_retrieved_items(prediction: dict[str, Any], attributed: int) -> int:
    debug = prediction.get("memory_debug") or prediction.get("debug") or {}
    for field in ("retrieved_count", "retrieved_sessions", "context_sessions"):
        value = debug.get(field)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                pass
    return attributed


def _evidence_spec(key: dict[str, Any]) -> dict[str, Any] | None:
    if "decision_evidence_session_ids" in key:
        positive = [str(value) for value in key["decision_evidence_session_ids"]]
        semantics = "decisive_habit_evidence"
    elif "gold_evidence_session_ids" in key:
        positive = [str(value) for value in key["gold_evidence_session_ids"]]
        semantics = "gold_evidence"
    else:
        return None
    if not positive:
        raise DatasetContractError(
            f"Probe {key.get('probe_id')} has an empty positive evidence set"
        )
    return {
        "semantics": semantics,
        "positive": positive,
        "temporal": [
            str(value) for value in key.get("temporal_context_session_ids", [])
        ],
        "nonbinding": [
            str(value) for value in key.get("nonbinding_evidence_session_ids", [])
        ],
        "components": [
            [str(value) for value in group]
            for group in key.get("required_component_groups", [])
        ],
    }


def _validate_gold_spec(
    probe: dict[str, Any],
    spec: dict[str, Any],
    session_lookup: dict[str, tuple[str, int]],
) -> None:
    user_id = probe["user_id"]
    cutoff = int(probe["visible_history_scope"]["max_session_index"])
    for field in ("positive", "temporal", "nonbinding"):
        values = spec[field]
        if len(values) != len(set(values)):
            raise DatasetContractError(
                f"Duplicate {field} evidence IDs for probe {probe['probe_id']}"
            )
        for session_id in values:
            owner_and_index = session_lookup.get(session_id)
            if owner_and_index is None:
                raise DatasetContractError(
                    f"Unknown {field} evidence ID {session_id} for probe {probe['probe_id']}"
                )
            owner, index = owner_and_index
            if owner != user_id or index > cutoff:
                raise DatasetContractError(
                    f"Out-of-scope {field} evidence ID {session_id} for probe "
                    f"{probe['probe_id']}"
                )
    positive = set(spec["positive"])
    for group in spec["components"]:
        if not group or not set(group).issubset(positive):
            raise DatasetContractError(
                f"Invalid required component group for probe {probe['probe_id']}"
            )


def _score_one(
    prediction: dict[str, Any],
    probe: dict[str, Any],
    key: dict[str, Any],
    *,
    correct: bool,
    mode: str,
    session_lookup: dict[str, tuple[str, int]],
) -> dict[str, Any]:
    spec = _evidence_spec(key)
    if spec is None:
        return {
            "evaluable": False,
            "mode": mode,
            "reason": "dataset_has_no_evidence_annotations",
        }
    _validate_gold_spec(probe, spec, session_lookup)

    raw = prediction.get("evidence_session_ids", [])
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise DatasetContractError(
            f"Invalid evidence_session_ids for prediction {probe['probe_id']}"
        )
    ranked = _unique(raw)
    duplicate_count = len(raw) - len(ranked)
    user_id = probe["user_id"]
    cutoff = int(probe["visible_history_scope"]["max_session_index"])
    unknown: list[str] = []
    wrong_user: list[str] = []
    after_cutoff: list[str] = []
    valid_visible: list[str] = []
    for session_id in ranked:
        owner_and_index = session_lookup.get(session_id)
        if owner_and_index is None:
            unknown.append(session_id)
        elif owner_and_index[0] != user_id:
            wrong_user.append(session_id)
        elif owner_and_index[1] > cutoff:
            after_cutoff.append(session_id)
        else:
            valid_visible.append(session_id)

    positive = set(spec["positive"])
    temporal = set(spec["temporal"])
    nonbinding = set(spec["nonbinding"])
    topk = ranked[:RETRIEVAL_K]
    positive_hits = len(positive.intersection(topk))
    first_hit_rank = next(
        (rank for rank, session_id in enumerate(topk, start=1) if session_id in positive),
        None,
    )
    evidence_relevance = [int(session_id in positive) for session_id in topk]
    evidence_relevance.extend([0] * (RETRIEVAL_K - len(evidence_relevance)))
    gold_count = len(positive)
    hit = positive_hits > 0
    full = positive.issubset(set(topk))

    labels = []
    for session_id in topk:
        if session_id in positive:
            labels.append("positive")
        elif session_id in temporal:
            labels.append("temporal")
        elif session_id in nonbinding:
            labels.append("nonbinding")
        elif session_id in unknown:
            labels.append("unknown")
        elif session_id in wrong_user:
            labels.append("wrong_user")
        elif session_id in after_cutoff:
            labels.append("after_cutoff")
        else:
            labels.append("other")

    result: dict[str, Any] = {
        "evaluable": True,
        "mode": mode,
        "gold_semantics": spec["semantics"],
        "gold_evidence_count": gold_count,
        "reported_retrieved_items": _reported_retrieved_items(prediction, len(ranked)),
        "attributed_session_ids": len(raw),
        "unique_attributed_session_ids": len(ranked),
        "duplicate_attribution_count": duplicate_count,
        "valid_visible_attribution_count": len(valid_visible),
        "unknown_attribution_count": len(unknown),
        "wrong_user_attribution_count": len(wrong_user),
        "after_cutoff_attribution_count": len(after_cutoff),
        "source_attribution_available": bool(valid_visible),
        "top_5_relevance_labels": labels,
    }

    if mode == "ranked":
        result.update(
            {
                "positive_hits_at_5": positive_hits,
                "evidence_recall_at_5": _ratio(positive_hits, gold_count),
                "evidence_coverage_efficiency_at_5": _ratio(
                    positive_hits, min(RETRIEVAL_K, gold_count)
                ),
                "evidence_precision_at_5": _ratio(positive_hits, RETRIEVAL_K),
                "returned_evidence_precision_at_5": _ratio(
                    positive_hits, len(topk)
                ),
                "evidence_hit_at_5": hit,
                "full_evidence_at_5": full,
                "evidence_mrr_at_5": _ratio(1, first_hit_rank or 0),
                "evidence_ndcg_at_5": _ndcg(
                    evidence_relevance, gold_count, RETRIEVAL_K
                ),
                "joint_answer_evidence_hit_at_5": bool(correct and hit),
                "joint_answer_full_evidence_at_5": bool(correct and full),
            }
        )
    elif mode == "context":
        context_hits = len(positive.intersection(ranked))
        context_full = positive.issubset(set(ranked))
        result.update(
            {
                "context_positive_hits": context_hits,
                "context_evidence_recall": _ratio(context_hits, gold_count),
                "context_full_evidence": context_full,
                "joint_answer_context_evidence_hit": bool(
                    correct and context_hits > 0
                ),
            }
        )

    if spec["components"]:
        retrieval_ids = set(topk if mode == "ranked" else ranked)
        component_hits = [
            bool(retrieval_ids.intersection(group)) for group in spec["components"]
        ]
        component_complete = [
            set(group).issubset(retrieval_ids) for group in spec["components"]
        ]
        prefix = "at_5" if mode == "ranked" else "in_context"
        result.update(
            {
                f"component_hit_coverage_{prefix}": _ratio(
                    sum(component_hits), len(component_hits)
                ),
                f"component_complete_coverage_{prefix}": _ratio(
                    sum(component_complete), len(component_complete)
                ),
                f"complete_chain_{prefix}": all(component_complete),
            }
        )

    if spec["temporal"]:
        retrieval_ids = topk if mode == "ranked" else ranked
        temporal_hits = len(temporal.intersection(retrieval_ids))
        suffix = "at_5" if mode == "ranked" else "in_context"
        result[f"temporal_context_recall_{suffix}"] = _ratio(
            temporal_hits, len(temporal)
        )
        if mode == "ranked":
            result["contextual_evidence_ndcg_at_5"] = _graded_ndcg(
                topk, positive, temporal, RETRIEVAL_K
            )

    if spec["nonbinding"]:
        retrieval_ids = topk if mode == "ranked" else ranked
        nonbinding_hits = len(nonbinding.intersection(retrieval_ids))
        if mode == "ranked":
            intrusion = _ratio(nonbinding_hits, RETRIEVAL_K)
            clean_hit = hit and nonbinding_hits == 0
            result.update(
                {
                    "nonbinding_hits_at_5": nonbinding_hits,
                    "nonbinding_intrusion_rate_at_5": intrusion,
                    "nonbinding_probe_intrusion_at_5": nonbinding_hits > 0,
                    "decisive_decoy_discrimination_at_5": (
                        result["evidence_precision_at_5"] - intrusion
                    ),
                    "clean_evidence_hit_at_5": clean_hit,
                    "clean_grounded_answer_at_5": bool(correct and clean_hit),
                }
            )
        else:
            result.update(
                {
                    "nonbinding_context_hits": nonbinding_hits,
                    "nonbinding_context_exposure": _ratio(
                        nonbinding_hits, len(nonbinding)
                    ),
                }
            )
    return result


def score_retrieval_predictions(
    predictions_by_id: dict[str, dict[str, Any]],
    bundle: DatasetBundle,
    method_name: str,
) -> dict[str, dict[str, Any]]:
    mode = _retrieval_mode(method_name)
    session_lookup = {
        session["session_id"]: (user_id, int(session["session_index"]))
        for user_id, sessions in bundle.sessions_by_user.items()
        for session in sessions
    }
    scores: dict[str, dict[str, Any]] = {}
    for probe in bundle.probes:
        probe_id = probe["probe_id"]
        prediction = predictions_by_id[probe_id]
        scores[probe_id] = _score_one(
            prediction,
            probe,
            bundle.keys[probe_id],
            correct=prediction["choice_id"]
            == bundle.keys[probe_id]["gold_choice_id"],
            mode=mode,
            session_lookup=session_lookup,
        )
    return scores


def aggregate_retrieval_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval_rows = [
        row["retrieval"] for row in rows if row.get("retrieval", {}).get("evaluable")
    ]
    if not retrieval_rows:
        return {
            "retrieval_mode": "unavailable",
            "retrieval_evaluable_probes": 0,
        }
    modes = {row["mode"] for row in retrieval_rows}
    if len(modes) != 1:
        raise DatasetContractError(f"Mixed retrieval modes in one score bucket: {modes}")
    mode = next(iter(modes))
    result: dict[str, Any] = {
        "retrieval_mode": mode,
        "retrieval_evaluable_probes": len(retrieval_rows),
        "avg_reported_retrieved_items": _rounded(
            _mean([row["reported_retrieved_items"] for row in retrieval_rows]), 3
        ),
        "avg_attributed_session_ids": _rounded(
            _mean([row["attributed_session_ids"] for row in retrieval_rows]), 3
        ),
        "source_attribution_probe_coverage": _rounded(
            _mean([row["source_attribution_available"] for row in retrieval_rows])
        ),
    }
    attributed = sum(row["attributed_session_ids"] for row in retrieval_rows)
    invalid = sum(
        row["unknown_attribution_count"]
        + row["wrong_user_attribution_count"]
        + row["after_cutoff_attribution_count"]
        for row in retrieval_rows
    )
    duplicates = sum(row["duplicate_attribution_count"] for row in retrieval_rows)
    result.update(
        {
            "invalid_attribution_rate": round(_ratio(invalid, attributed), 6),
            "duplicate_attribution_rate": round(_ratio(duplicates, attributed), 6),
            "wrong_user_attribution_count": sum(
                row["wrong_user_attribution_count"] for row in retrieval_rows
            ),
            "after_cutoff_attribution_count": sum(
                row["after_cutoff_attribution_count"] for row in retrieval_rows
            ),
        }
    )

    if mode == "ranked":
        hits = sum(row["positive_hits_at_5"] for row in retrieval_rows)
        gold = sum(row["gold_evidence_count"] for row in retrieval_rows)
        result.update(
            {
                "evidence_recall_at_5_macro": _rounded(
                    _mean([row["evidence_recall_at_5"] for row in retrieval_rows])
                ),
                "evidence_recall_at_5_micro": round(_ratio(hits, gold), 6),
                "evidence_coverage_efficiency_at_5": _rounded(
                    _mean(
                        [
                            row["evidence_coverage_efficiency_at_5"]
                            for row in retrieval_rows
                        ]
                    )
                ),
                "evidence_precision_at_5": _rounded(
                    _mean([row["evidence_precision_at_5"] for row in retrieval_rows])
                ),
                "evidence_hit_rate_at_5": _rounded(
                    _mean([row["evidence_hit_at_5"] for row in retrieval_rows])
                ),
                "full_evidence_rate_at_5": _rounded(
                    _mean([row["full_evidence_at_5"] for row in retrieval_rows])
                ),
                "evidence_mrr_at_5": _rounded(
                    _mean([row["evidence_mrr_at_5"] for row in retrieval_rows])
                ),
                "evidence_ndcg_at_5": _rounded(
                    _mean([row["evidence_ndcg_at_5"] for row in retrieval_rows])
                ),
                "joint_answer_evidence_hit_rate_at_5": _rounded(
                    _mean(
                        [
                            row["joint_answer_evidence_hit_at_5"]
                            for row in retrieval_rows
                        ]
                    )
                ),
                "joint_answer_full_evidence_rate_at_5": _rounded(
                    _mean(
                        [
                            row["joint_answer_full_evidence_at_5"]
                            for row in retrieval_rows
                        ]
                    )
                ),
            }
        )
        attributable_rows = [
            row for row in retrieval_rows if row["source_attribution_available"]
        ]
        result["attribution_conditional_evidence_recall_at_5"] = _rounded(
            _mean([row["evidence_recall_at_5"] for row in attributable_rows])
        )
        hit_answer_rows = [
            original
            for original in rows
            if original.get("retrieval", {}).get("evidence_hit_at_5")
        ]
        miss_answer_rows = [
            original
            for original in rows
            if original.get("retrieval", {}).get("evaluable")
            and not original.get("retrieval", {}).get("evidence_hit_at_5")
        ]
        hit_accuracy = _mean([row["correct"] for row in hit_answer_rows])
        miss_accuracy = _mean([row["correct"] for row in miss_answer_rows])
        result.update(
            {
                "answer_accuracy_when_evidence_hit_at_5": _rounded(hit_accuracy),
                "answer_accuracy_when_evidence_miss_at_5": _rounded(miss_accuracy),
                "evidence_utility_gap_at_5": _rounded(
                    (
                        hit_accuracy - miss_accuracy
                        if hit_accuracy is not None and miss_accuracy is not None
                        else None
                    )
                ),
            }
        )
    elif mode == "context":
        hits = sum(row["context_positive_hits"] for row in retrieval_rows)
        gold = sum(row["gold_evidence_count"] for row in retrieval_rows)
        result.update(
            {
                "context_evidence_recall_macro": _rounded(
                    _mean([row["context_evidence_recall"] for row in retrieval_rows])
                ),
                "context_evidence_recall_micro": round(_ratio(hits, gold), 6),
                "context_full_evidence_rate": _rounded(
                    _mean([row["context_full_evidence"] for row in retrieval_rows])
                ),
                "joint_answer_context_evidence_hit_rate": _rounded(
                    _mean(
                        [
                            row["joint_answer_context_evidence_hit"]
                            for row in retrieval_rows
                        ]
                    )
                ),
            }
        )

    optional_average_fields = (
        "component_hit_coverage_at_5",
        "component_complete_coverage_at_5",
        "complete_chain_at_5",
        "temporal_context_recall_at_5",
        "contextual_evidence_ndcg_at_5",
        "nonbinding_intrusion_rate_at_5",
        "nonbinding_probe_intrusion_at_5",
        "decisive_decoy_discrimination_at_5",
        "clean_evidence_hit_at_5",
        "clean_grounded_answer_at_5",
        "component_hit_coverage_in_context",
        "component_complete_coverage_in_context",
        "complete_chain_in_context",
        "temporal_context_recall_in_context",
        "nonbinding_context_exposure",
    )
    aggregate_names = {
        "complete_chain_at_5": "complete_chain_rate_at_5",
        "nonbinding_probe_intrusion_at_5": "nonbinding_probe_intrusion_rate_at_5",
        "clean_evidence_hit_at_5": "clean_evidence_hit_rate_at_5",
        "clean_grounded_answer_at_5": "clean_grounded_answer_rate_at_5",
        "complete_chain_in_context": "complete_chain_rate_in_context",
    }
    for field in optional_average_fields:
        values = [row[field] for row in retrieval_rows if field in row]
        if values:
            result[aggregate_names.get(field, field)] = _rounded(_mean(values))
    return result


def balanced_decision_metrics(
    detailed: list[dict[str, Any]],
    bundle: DatasetBundle,
) -> dict[str, Any]:
    ranked_mode = any(
        row.get("retrieval", {}).get("mode") == "ranked" for row in detailed
    )
    rows_by_id = {row["probe_id"]: row for row in detailed}
    unit_events: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    bundle_events: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    for probe in bundle.probes:
        probe_id = probe["probe_id"]
        key = bundle.keys[probe_id]
        unit_ids = key.get("decision_unit_ids") or []
        groups = key.get("required_component_groups") or []
        if not unit_ids:
            continue
        if len(unit_ids) > len(groups):
            raise DatasetContractError(
                f"More decision units than component groups for probe {probe_id}"
            )
        row = rows_by_id[probe_id]
        ranked = set(_unique(row.get("evidence_session_ids", []))[:RETRIEVAL_K])
        nonbinding = set(key.get("nonbinding_evidence_session_ids", []))
        no_decoy = not ranked.intersection(nonbinding)
        for unit_id, group in zip(unit_ids, groups):
            group_set = set(group)
            hits = len(group_set.intersection(ranked))
            unit_events[str(unit_id)].append(
                {
                    "accuracy": bool(row["correct"]),
                    "recall": _ratio(hits, len(group_set)),
                    "hit": hits > 0,
                    "complete": group_set.issubset(ranked),
                    "clean_grounded": bool(
                        row["correct"] and hits > 0 and no_decoy
                    ),
                }
            )
        bundle_id = key.get("decision_bundle_id")
        if bundle_id:
            retrieval = row["retrieval"]
            bundle_events[str(bundle_id)].append(
                {
                    "accuracy": bool(row["correct"]),
                    "recall": float(retrieval.get("evidence_recall_at_5", 0.0)),
                    "clean_grounded": bool(
                        retrieval.get("clean_grounded_answer_at_5", False)
                    ),
                }
            )
    if not unit_events:
        return {}

    unit_means = {
        unit_id: {
            field: _mean([event[field] for event in events])
            for field in ("accuracy", "recall", "hit", "complete", "clean_grounded")
        }
        for unit_id, events in unit_events.items()
    }
    bundle_means = {
        bundle_id: {
            field: _mean([event[field] for event in events])
            for field in ("accuracy", "recall", "clean_grounded")
        }
        for bundle_id, events in bundle_events.items()
    }
    result = {
        "decision_unit_count": len(unit_means),
        "decision_unit_macro_accuracy": _rounded(
            _mean([values["accuracy"] for values in unit_means.values()])
        ),
        "decision_bundle_count": len(bundle_means),
        "decision_bundle_macro_accuracy": _rounded(
            _mean([values["accuracy"] for values in bundle_means.values()])
        ),
    }
    if not ranked_mode:
        return result
    result.update(
        {
        "decision_unit_macro_evidence_recall_at_5": _rounded(
            _mean([values["recall"] for values in unit_means.values()])
        ),
        "decision_unit_macro_evidence_hit_rate_at_5": _rounded(
            _mean([values["hit"] for values in unit_means.values()])
        ),
        "decision_unit_macro_component_complete_rate_at_5": _rounded(
            _mean([values["complete"] for values in unit_means.values()])
        ),
        "decision_unit_macro_clean_grounded_answer_rate_at_5": _rounded(
            _mean([values["clean_grounded"] for values in unit_means.values()])
        ),
        "decision_bundle_macro_evidence_recall_at_5": _rounded(
            _mean([values["recall"] for values in bundle_means.values()])
        ),
        "decision_bundle_macro_clean_grounded_answer_rate_at_5": _rounded(
            _mean([values["clean_grounded"] for values in bundle_means.values()])
        ),
        }
    )
    return result
