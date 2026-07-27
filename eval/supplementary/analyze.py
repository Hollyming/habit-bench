#!/usr/bin/env python
"""Compute additive paper analyses from an existing scored-prediction file.

The primary HABIT-Bench scorer is deliberately not called or modified here.
This sidecar joins scored rows to the immutable private key and emits separate
user-level, stress-slice, component, calibration, and efficiency diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from eval.core.dataset import DatasetBundle, DatasetContractError, load_dataset
from eval.core.io import read_jsonl, write_csv, write_json, write_jsonl


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metric_distribution(values: Iterable[float]) -> dict[str, Any]:
    materialized = list(values)
    return {
        "count": len(materialized),
        "mean": _rounded(_mean(materialized)),
        "p50": _rounded(percentile(materialized, 0.50)),
        "p95": _rounded(percentile(materialized, 0.95)),
        "sum": _rounded(sum(materialized)) if materialized else None,
    }


def _probe_maps(
    bundle: DatasetBundle,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    probes = {str(probe["probe_id"]): probe for probe in bundle.probes}
    return probes, bundle.keys


def validate_scored_rows(
    rows: list[dict[str, Any]],
    bundle: DatasetBundle,
    *,
    allow_partial: bool = False,
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    probes, _ = _probe_maps(bundle)
    for row in rows:
        probe_id = str(row.get("probe_id", ""))
        if not probe_id:
            raise DatasetContractError("Scored row is missing probe_id")
        if probe_id in by_id:
            raise DatasetContractError(f"Duplicate scored row for {probe_id}")
        if probe_id not in probes:
            raise DatasetContractError(f"Unknown scored probe {probe_id}")
        valid_choices = {
            str(choice["choice_id"]) for choice in probes[probe_id]["choices"]
        }
        if str(row.get("choice_id", "")) not in valid_choices:
            raise DatasetContractError(f"Invalid predicted choice for {probe_id}")
        expected_correct = (
            str(row.get("choice_id")) == str(bundle.keys[probe_id]["gold_choice_id"])
        )
        if not isinstance(row.get("correct"), bool):
            raise DatasetContractError(
                f"Scored row for {probe_id} is missing boolean correct"
            )
        if row["correct"] != expected_correct:
            raise DatasetContractError(
                f"Scored row has inconsistent correctness for {probe_id}"
            )
        by_id[probe_id] = row
    if not allow_partial:
        missing = set(probes) - set(by_id)
        if missing:
            raise DatasetContractError(
                f"Scored prediction coverage is incomplete: missing={len(missing)}"
            )
    if not by_id:
        raise DatasetContractError("Scored prediction file is empty")
    return by_id


def _user_rows(
    rows: list[dict[str, Any]], bundle: DatasetBundle
) -> dict[str, list[dict[str, Any]]]:
    probes, _ = _probe_maps(bundle)
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(probes[row["probe_id"]]["user_id"])].append(row)
    return dict(result)


def user_accuracy_rows(
    rows: list[dict[str, Any]], bundle: DatasetBundle
) -> list[dict[str, Any]]:
    probes, _ = _probe_maps(bundle)
    output: list[dict[str, Any]] = []
    for user_id, selected in sorted(_user_rows(rows, bundle).items()):
        domains = sorted({str(probes[row["probe_id"]]["domain"]) for row in selected})
        correct = sum(bool(row.get("correct")) for row in selected)
        output.append(
            {
                "user_id": user_id,
                "domains": "+".join(domains),
                "correct": correct,
                "total": len(selected),
                "accuracy": round(correct / len(selected), 6),
            }
        )
    return output


def cluster_bootstrap_ci(
    user_values: dict[str, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    user_ids = sorted(user_values)
    if not user_ids:
        return {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "samples": samples,
            "clusters": 0,
        }
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = [generator.choice(user_ids) for _ in user_ids]
        estimates.append(sum(user_values[user_id] for user_id in selected) / len(selected))
    return {
        "estimate": _rounded(_mean(user_values.values())),
        "ci95_low": _rounded(percentile(estimates, 0.025)),
        "ci95_high": _rounded(percentile(estimates, 0.975)),
        "samples": samples,
        "clusters": len(user_ids),
        "seed": seed,
        "resampling_unit": "user",
    }


def _accuracy_summary(
    rows: list[dict[str, Any]],
    bundle: DatasetBundle,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    per_user = user_accuracy_rows(rows, bundle)
    micro = _mean(float(bool(row.get("correct"))) for row in rows)
    user_values = {row["user_id"]: float(row["accuracy"]) for row in per_user}
    return {
        "probes": len(rows),
        "users": len(per_user),
        "micro_accuracy": _rounded(micro),
        "user_macro_accuracy": _rounded(_mean(user_values.values())),
        "user_macro_cluster_bootstrap": cluster_bootstrap_ci(
            user_values, samples=bootstrap_samples, seed=seed
        ),
        "per_user_accuracy": _metric_distribution(user_values.values()),
    }


def _explicit_to_habit_gap(
    rows: list[dict[str, Any]],
    bundle: DatasetBundle,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    probes, keys = _probe_maps(bundle)
    explicit: list[dict[str, Any]] = []
    latent: list[dict[str, Any]] = []
    by_user: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"explicit": [], "latent": []}
    )
    for row in rows:
        probe_id = str(row["probe_id"])
        probe_type = str(keys[probe_id].get("probe_type", row.get("probe_type", "")))
        family = "explicit" if probe_type == "explicit_retrieval" else "latent"
        (explicit if family == "explicit" else latent).append(row)
        by_user[str(probes[probe_id]["user_id"])][family].append(
            float(bool(row.get("correct")))
        )
    if not explicit or not latent:
        return {
            "status": "unavailable",
            "reason": (
                "Both explicit_retrieval and latent-habit probes are required "
                "within the selected dataset."
            ),
            "explicit_probes": len(explicit),
            "latent_probes": len(latent),
        }
    paired_user_gap = {
        user_id: float(_mean(parts["explicit"]) - _mean(parts["latent"]))
        for user_id, parts in by_user.items()
        if parts["explicit"] and parts["latent"]
    }
    explicit_accuracy = _mean(float(bool(row.get("correct"))) for row in explicit)
    latent_accuracy = _mean(float(bool(row.get("correct"))) for row in latent)
    return {
        "status": "available",
        "definition": "explicit_retrieval accuracy minus all latent-habit probe accuracy",
        "explicit_probes": len(explicit),
        "latent_probes": len(latent),
        "explicit_accuracy": _rounded(explicit_accuracy),
        "latent_habit_accuracy": _rounded(latent_accuracy),
        "explicit_minus_latent_gap": _rounded(explicit_accuracy - latent_accuracy),
        "paired_user_gap": cluster_bootstrap_ci(
            paired_user_gap, samples=bootstrap_samples, seed=seed + 17
        ),
    }


def _session_indices(bundle: DatasetBundle) -> dict[str, int]:
    return {
        str(session["session_id"]): int(session["session_index"])
        for sessions in bundle.sessions_by_user.values()
        for session in sessions
    }


def _stress_diagnostic(
    row: dict[str, Any],
    bundle: DatasetBundle,
    session_indices: dict[str, int],
) -> dict[str, Any]:
    probes, keys = _probe_maps(bundle)
    probe_id = str(row["probe_id"])
    probe = probes[probe_id]
    key = keys[probe_id]
    decisive = list(
        key.get("decision_evidence_session_ids")
        if key.get("decision_evidence_session_ids") is not None
        else key.get("gold_evidence_session_ids") or []
    )
    temporal = list(key.get("temporal_context_session_ids") or [])
    nonbinding = list(key.get("nonbinding_evidence_session_ids") or [])
    relevant = set(map(str, decisive + temporal + nonbinding))
    positive_indices = [
        session_indices[str(session_id)]
        for session_id in decisive
        if str(session_id) in session_indices
    ]
    cutoff = int(probe["visible_history_scope"]["max_session_index"])
    visible = [
        session
        for session in bundle.sessions_by_user[probe["user_id"]]
        if int(session["session_index"]) <= cutoff
    ]
    history_count = len(visible)
    relevant_visible = sum(
        str(session["session_id"]) in relevant for session in visible
    )
    support_count = len(set(map(str, decisive)))
    distractor_count = max(0, history_count - relevant_visible)
    distractor_ratio = distractor_count / max(1, support_count)
    if positive_indices and cutoff > 0:
        position_fraction = sum(index / cutoff for index in positive_indices) / len(
            positive_indices
        )
    elif positive_indices:
        position_fraction = 1.0
    else:
        position_fraction = None
    if position_fraction is None:
        position_bin = "unknown"
    elif position_fraction <= 0.25:
        position_bin = "early"
    elif position_fraction <= 0.75:
        position_bin = "middle"
    else:
        position_bin = "late"
    if history_count < 128:
        history_bin = "000-127"
    elif history_count < 256:
        history_bin = "128-255"
    elif history_count < 512:
        history_bin = "256-511"
    else:
        history_bin = "512+"
    if distractor_ratio < 10:
        distractor_bin = "<10x"
    elif distractor_ratio < 50:
        distractor_bin = "10-49x"
    else:
        distractor_bin = "50x+"
    evidence_span = key.get("evidence_span_sessions")
    if evidence_span is None and positive_indices:
        evidence_span = max(positive_indices) - min(positive_indices)
    newest_age = cutoff - max(positive_indices) if positive_indices else None
    bands = key.get("evidence_bands") or []
    return {
        "probe_id": probe_id,
        "user_id": probe["user_id"],
        "domain": probe.get("domain", "unknown"),
        "probe_type": key.get("probe_type", row.get("probe_type", "unknown")),
        "correct": bool(row.get("correct")),
        "support_count": support_count,
        "history_session_count": history_count,
        "history_length_bin": history_bin,
        "distractor_count": distractor_count,
        "distractor_to_decisive_ratio": round(distractor_ratio, 6),
        "distractor_ratio_bin": distractor_bin,
        "mean_decisive_position_fraction": _rounded(position_fraction),
        "evidence_position_bin": position_bin,
        "evidence_span_sessions": evidence_span,
        "newest_decisive_evidence_age_sessions": newest_age,
        "evidence_bands": "+".join(sorted(map(str, bands))) if bands else "unknown",
    }


def _retrieval_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        retrieval = row.get("retrieval") or {}
        if not retrieval.get("evaluable"):
            continue
        value = retrieval.get(field)
        if isinstance(value, bool):
            values.append(float(value))
        else:
            number = _number(value)
            if number is not None:
                values.append(number)
    return values


def _slice_row(
    dimension: str,
    value: str,
    rows: list[dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
    bundle: DatasetBundle,
) -> dict[str, Any]:
    probes, _ = _probe_maps(bundle)
    user_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        user_id = str(probes[row["probe_id"]]["user_id"])
        user_values[user_id].append(float(bool(row.get("correct"))))
    return {
        "dimension": dimension,
        "value": value,
        "probes": len(rows),
        "users": len(user_values),
        "accuracy": _rounded(
            _mean(float(bool(row.get("correct"))) for row in rows)
        ),
        "user_macro_accuracy": _rounded(
            _mean(float(_mean(values)) for values in user_values.values())
        ),
        "evidence_recall_at_5": _rounded(
            _mean(_retrieval_values(rows, "evidence_recall_at_5"))
        ),
        "full_evidence_at_5": _rounded(
            _mean(_retrieval_values(rows, "full_evidence_at_5"))
        ),
        "complete_chain_at_5": _rounded(
            _mean(_retrieval_values(rows, "complete_chain_at_5"))
        ),
        "component_complete_coverage_at_5": _rounded(
            _mean(_retrieval_values(rows, "component_complete_coverage_at_5"))
        ),
        "joint_answer_evidence_hit_at_5": _rounded(
            _mean(_retrieval_values(rows, "joint_answer_evidence_hit_at_5"))
        ),
        "clean_grounded_answer_at_5": _rounded(
            _mean(_retrieval_values(rows, "clean_grounded_answer_at_5"))
        ),
        "nonbinding_intrusion_rate_at_5": _rounded(
            _mean(_retrieval_values(rows, "nonbinding_intrusion_rate_at_5"))
        ),
        "mean_history_sessions": _rounded(
            _mean(
                float(diagnostics[row["probe_id"]]["history_session_count"])
                for row in rows
            )
        ),
    }


def stress_slice_rows(
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    bundle: DatasetBundle,
) -> list[dict[str, Any]]:
    by_probe = {row["probe_id"]: row for row in diagnostics}
    dimensions: dict[str, Callable[[dict[str, Any]], str]] = {
        "domain": lambda row: str(row["domain"]),
        "probe_type": lambda row: str(row["probe_type"]),
        "support_count": lambda row: str(row["support_count"]),
        "history_length_bin": lambda row: str(row["history_length_bin"]),
        "distractor_ratio_bin": lambda row: str(row["distractor_ratio_bin"]),
        "evidence_position_bin": lambda row: str(row["evidence_position_bin"]),
        "evidence_bands": lambda row: str(row["evidence_bands"]),
    }
    output: list[dict[str, Any]] = []
    for dimension, selector in dimensions.items():
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[selector(by_probe[row["probe_id"]])].append(row)
        for value, selected in sorted(buckets.items()):
            output.append(
                _slice_row(dimension, value, selected, by_probe, bundle)
            )
    return output


def _variants(signature: Any) -> dict[str, Any]:
    if not isinstance(signature, dict):
        return {}
    value = signature.get("variants", signature)
    return value if isinstance(value, dict) else {}


def policy_component_metrics(
    rows: list[dict[str, Any]], bundle: DatasetBundle
) -> dict[str, Any]:
    _, keys = _probe_maps(bundle)
    evaluable = 0
    component_total = 0
    component_correct = 0
    wrong_histogram: dict[int, int] = defaultdict(int)
    decoy_selected = 0
    decoy_evaluable = 0
    per_habit: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "surface_decoy": 0}
    )
    for row in rows:
        key = keys[row["probe_id"]]
        signatures = key.get("choice_policy_signatures")
        if not isinstance(signatures, dict):
            continue
        gold = _variants(signatures.get(str(key.get("gold_choice_id"))))
        predicted = _variants(signatures.get(str(row.get("choice_id"))))
        if not gold or not predicted:
            continue
        evaluable += 1
        wrong = 0
        decoys = key.get("surface_decoy_variants") or {}
        for habit_id, gold_variant in gold.items():
            predicted_variant = predicted.get(habit_id)
            is_correct = predicted_variant == gold_variant
            component_total += 1
            component_correct += int(is_correct)
            wrong += int(not is_correct)
            per_habit[str(habit_id)]["total"] += 1
            per_habit[str(habit_id)]["correct"] += int(is_correct)
            if habit_id in decoys:
                decoy_evaluable += 1
                is_decoy = predicted_variant == decoys[habit_id] and not is_correct
                decoy_selected += int(is_decoy)
                per_habit[str(habit_id)]["surface_decoy"] += int(is_decoy)
        wrong_histogram[wrong] += 1
    if not evaluable:
        return {
            "status": "unavailable",
            "reason": "choice_policy_signatures are absent in the selected dataset.",
        }
    per_habit_rows = []
    for habit_id, values in sorted(per_habit.items()):
        per_habit_rows.append(
            {
                "habit_id": habit_id,
                **values,
                "component_accuracy": round(
                    values["correct"] / values["total"], 6
                ),
                "surface_decoy_rate": round(
                    values["surface_decoy"] / values["total"], 6
                ),
            }
        )
    return {
        "status": "available",
        "evaluable_probes": evaluable,
        "component_total": component_total,
        "component_correct": component_correct,
        "policy_component_accuracy": round(
            component_correct / component_total, 6
        ),
        "mean_wrong_components": round(
            sum(wrong * count for wrong, count in wrong_histogram.items()) / evaluable,
            6,
        ),
        "zero_wrong_component_rate": round(
            wrong_histogram.get(0, 0) / evaluable, 6
        ),
        "one_wrong_component_rate": round(
            wrong_histogram.get(1, 0) / evaluable, 6
        ),
        "multiple_wrong_component_rate": round(
            sum(count for wrong, count in wrong_histogram.items() if wrong >= 2)
            / evaluable,
            6,
        ),
        "surface_decoy_component_selection_rate": (
            round(decoy_selected / decoy_evaluable, 6)
            if decoy_evaluable
            else None
        ),
        "surface_decoy_component_evaluations": decoy_evaluable,
        "per_habit": per_habit_rows,
    }


def calibration_metrics(
    rows: list[dict[str, Any]], bundle: DatasetBundle, bins: int = 10
) -> dict[str, Any]:
    probes, keys = _probe_maps(bundle)
    observations: list[tuple[float, bool, float, float]] = []
    for row in rows:
        probabilities = row.get("choice_probabilities")
        if probabilities is None:
            probabilities = (row.get("answer") or {}).get("choice_probabilities")
        if not isinstance(probabilities, dict):
            continue
        choice_ids = [
            str(choice["choice_id"]) for choice in probes[row["probe_id"]]["choices"]
        ]
        values = [_number(probabilities.get(choice_id)) for choice_id in choice_ids]
        if any(value is None or value < 0 for value in values):
            continue
        total = sum(value for value in values if value is not None)
        if total <= 0:
            continue
        normalized = [float(value) / total for value in values]
        gold = str(keys[row["probe_id"]]["gold_choice_id"])
        gold_index = choice_ids.index(gold)
        predicted_index = max(range(len(normalized)), key=normalized.__getitem__)
        confidence = normalized[predicted_index]
        correct = predicted_index == gold_index
        brier = sum(
            (probability - float(index == gold_index)) ** 2
            for index, probability in enumerate(normalized)
        )
        nll = -math.log(max(normalized[gold_index], 1e-15))
        observations.append((confidence, correct, brier, nll))
    if len(observations) != len(rows):
        return {
            "status": "unavailable",
            "reason": (
                "Every prediction must provide a probability for every choice "
                "in answer.choice_probabilities. Current exact-choice runs do not."
            ),
            "rows_with_probabilities": len(observations),
            "total_rows": len(rows),
        }
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            observation
            for observation in observations
            if lower <= observation[0] <= upper
            and (bin_index == bins - 1 or observation[0] < upper)
        ]
        if selected:
            bin_confidence = _mean(item[0] for item in selected)
            bin_accuracy = _mean(float(item[1]) for item in selected)
            ece += len(selected) / len(observations) * abs(
                bin_confidence - bin_accuracy
            )
    ordered = sorted(observations, key=lambda item: item[0], reverse=True)
    cumulative_errors = 0
    risks: list[float] = []
    for rank, observation in enumerate(ordered, start=1):
        cumulative_errors += int(not observation[1])
        risks.append(cumulative_errors / rank)
    return {
        "status": "available",
        "evaluable_probes": len(observations),
        "multiclass_brier_score": _rounded(_mean(item[2] for item in observations)),
        "negative_log_likelihood": _rounded(
            _mean(item[3] for item in observations)
        ),
        "expected_calibration_error": round(ece, 6),
        "ece_bins": bins,
        "area_under_risk_coverage_curve": _rounded(_mean(risks)),
    }


def personalization_metrics(
    rows: list[dict[str, Any]],
    bundle: DatasetBundle,
    lambdas: list[float],
) -> dict[str, Any]:
    """Compute true false-personalization costs only with explicit taxonomy.

    No current dataset is silently coerced into this contract.  This prevents
    ordinary answer errors from being mislabeled as false personalization.
    """

    _, keys = _probe_maps(bundle)
    selected: list[tuple[bool, str, bool]] = []
    for row in rows:
        key = keys[row["probe_id"]]
        taxonomy = key.get("choice_action_taxonomy")
        applicable = key.get("personalization_applicable")
        if not isinstance(taxonomy, dict) or not isinstance(applicable, bool):
            continue
        label = str(taxonomy.get(str(row.get("choice_id")), ""))
        if not label:
            continue
        selected.append((applicable, label, bool(row.get("correct"))))
    if len(selected) != len(rows):
        return {
            "status": "unavailable",
            "reason": (
                "Requires private choice_action_taxonomy and boolean "
                "personalization_applicable for every probe. They are not "
                "present in current Food v4 or Finance/Software v1.3."
            ),
            "annotated_probes": len(selected),
            "total_probes": len(rows),
        }
    no_habit = [item for item in selected if not item[0]]
    applicable = [item for item in selected if item[0]]
    personalized_labels = {
        "applicable_habit",
        "stale_habit",
        "boundary_violation",
        "exception_violation",
        "unsupported_personalization",
    }
    false_positive = sum(item[1] in personalized_labels for item in no_habit)
    missed = sum(item[1] == "generic" for item in applicable)
    accuracy = _mean(float(item[2]) for item in selected)
    false_rate = false_positive / len(no_habit) if no_habit else None
    return {
        "status": "available",
        "false_personalization_rate": _rounded(false_rate),
        "missed_personalization_rate": (
            round(missed / len(applicable), 6) if applicable else None
        ),
        "stale_habit_selection_rate": round(
            sum(item[1] == "stale_habit" for item in selected) / len(selected), 6
        ),
        "utility_sensitivity": [
            {
                "lambda": penalty,
                "utility": (
                    round(accuracy - penalty * false_rate, 6)
                    if false_rate is not None
                    else None
                ),
            }
            for penalty in lambdas
        ],
    }


def answer_retrieval_error_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross answer correctness with decisive-evidence retrieval.

    The quadrants are descriptive rather than a causal error attribution:
    a correct answer without evidence can be a shortcut or model prior, while
    a wrong answer with evidence can arise from state induction or answer use.
    """

    quadrants = {
        "correct_with_evidence_hit": 0,
        "correct_without_evidence_hit": 0,
        "wrong_with_evidence_hit": 0,
        "wrong_without_evidence_hit": 0,
    }
    full_quadrants = {
        "correct_with_complete_evidence": 0,
        "correct_without_complete_evidence": 0,
        "wrong_with_complete_evidence": 0,
        "wrong_without_complete_evidence": 0,
    }
    evaluable = 0
    full_evaluable = 0
    wrong_with_nonbinding_intrusion = 0
    nonbinding_evaluable = 0
    invalid_attribution_rows = 0
    for row in rows:
        retrieval = row.get("retrieval") or {}
        if not retrieval.get("evaluable"):
            continue
        evidence_hit = retrieval.get("evidence_hit_at_5")
        if isinstance(evidence_hit, bool):
            evaluable += 1
            answer_label = "correct" if row.get("correct") else "wrong"
            evidence_label = (
                "with_evidence_hit" if evidence_hit else "without_evidence_hit"
            )
            quadrants[f"{answer_label}_{evidence_label}"] += 1
        complete = retrieval.get("complete_chain_at_5")
        if not isinstance(complete, bool):
            complete = retrieval.get("full_evidence_at_5")
        if isinstance(complete, bool):
            full_evaluable += 1
            answer_label = "correct" if row.get("correct") else "wrong"
            complete_label = (
                "with_complete_evidence"
                if complete
                else "without_complete_evidence"
            )
            full_quadrants[f"{answer_label}_{complete_label}"] += 1
        intrusion = retrieval.get("nonbinding_probe_intrusion_at_5")
        if isinstance(intrusion, bool):
            nonbinding_evaluable += 1
            wrong_with_nonbinding_intrusion += int(
                intrusion and not row.get("correct")
            )
        invalid_attribution_rows += int(
            any(
                int(retrieval.get(field) or 0) > 0
                for field in (
                    "wrong_user_attribution_count",
                    "after_cutoff_attribution_count",
                    "unknown_attribution_count",
                )
            )
        )
    if not evaluable:
        return {
            "status": "unavailable",
            "reason": "No ranked/context retrieval rows expose evidence_hit.",
        }
    return {
        "status": "available",
        "evaluable_probes": evaluable,
        "answer_x_evidence_hit": {
            key: {
                "count": count,
                "rate": round(count / evaluable, 6),
            }
            for key, count in quadrants.items()
        },
        "complete_evidence_evaluable_probes": full_evaluable,
        "answer_x_complete_evidence": {
            key: {
                "count": count,
                "rate": (
                    round(count / full_evaluable, 6)
                    if full_evaluable
                    else None
                ),
            }
            for key, count in full_quadrants.items()
        },
        "wrong_with_nonbinding_intrusion_rate": (
            round(wrong_with_nonbinding_intrusion / nonbinding_evaluable, 6)
            if nonbinding_evaluable
            else None
        ),
        "nonbinding_evaluable_probes": nonbinding_evaluable,
        "invalid_attribution_probe_rate": round(
            invalid_attribution_rows / evaluable, 6
        ),
        "interpretation_warning": (
            "Descriptive coupling only; it does not prove which module caused an error."
        ),
    }


def efficiency_metrics(
    rows: list[dict[str, Any]],
    *,
    artifact_root: Path | None,
    users: int,
) -> dict[str, Any]:
    def answer_values(field: str) -> list[float]:
        return [
            number
            for row in rows
            if (number := _number((row.get("answer") or {}).get(field))) is not None
        ]

    def usage_values(field: str) -> list[float]:
        return [
            number
            for row in rows
            if (
                number := _number(
                    ((row.get("answer") or {}).get("usage") or {}).get(field)
                )
            )
            is not None
        ]

    def cost_values(field: str) -> list[float]:
        return [
            number
            for row in rows
            if (number := _number((row.get("memory_cost") or {}).get(field)))
            is not None
        ]

    result: dict[str, Any] = {
        "answer_latency_sec": _metric_distribution(answer_values("latency_sec")),
        "memory_tokens_used": _metric_distribution(
            answer_values("memory_tokens_used")
        ),
        "prompt_tokens": _metric_distribution(usage_values("prompt_tokens")),
        "completion_tokens": _metric_distribution(
            usage_values("completion_tokens")
        ),
        "total_tokens": _metric_distribution(usage_values("total_tokens")),
        "retrieval_elapsed_sec": _metric_distribution(
            cost_values("retrieval_elapsed_sec")
        ),
        "sessions_added_for_user": _metric_distribution(
            cost_values("sessions_added_for_user")
        ),
    }
    if artifact_root is None:
        result["artifact_storage"] = {
            "status": "not_requested",
            "definition": "recursive regular-file bytes under --artifact-root",
        }
    else:
        resolved = artifact_root.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Artifact root not found: {resolved}")
        total_bytes = sum(
            path.stat().st_size
            for path in resolved.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        result["artifact_storage"] = {
            "status": "available",
            "root": str(resolved),
            "total_bytes": total_bytes,
            "bytes_per_user": round(total_bytes / users, 3) if users else None,
            "definition": "recursive regular-file bytes, excluding symlinks",
        }
    return result


def build_supplementary_analysis(
    rows: list[dict[str, Any]],
    bundle: DatasetBundle,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    artifact_root: Path | None = None,
    utility_lambdas: list[float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validate_scored_rows(rows, bundle, allow_partial=True)
    ordered = sorted(rows, key=lambda row: str(row["probe_id"]))
    session_indices = _session_indices(bundle)
    diagnostics = [
        _stress_diagnostic(row, bundle, session_indices) for row in ordered
    ]
    slices = stress_slice_rows(ordered, diagnostics, bundle)
    users = len(_user_rows(ordered, bundle))
    method_names = sorted(
        {str(row.get("method_name", "unknown")) for row in ordered}
    )
    payload = {
        "contract_version": "habitbench.supplementary_analysis.v1",
        "analysis_role": (
            "additive sidecar; does not replace or modify primary HABIT-Bench metrics"
        ),
        "method_names": method_names,
        "dataset": bundle.manifest,
        "accuracy": _accuracy_summary(
            ordered,
            bundle,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "explicit_to_habit_transfer": _explicit_to_habit_gap(
            ordered,
            bundle,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "policy_components": policy_component_metrics(ordered, bundle),
        "calibration": calibration_metrics(ordered, bundle),
        "false_personalization": personalization_metrics(
            ordered, bundle, utility_lambdas or [0.0, 0.5, 1.0, 2.0, 5.0]
        ),
        "answer_retrieval_error_analysis": answer_retrieval_error_analysis(
            ordered
        ),
        "efficiency": efficiency_metrics(
            ordered, artifact_root=artifact_root, users=users
        ),
        "stress_slice_dimensions": [
            "domain",
            "probe_type",
            "support_count",
            "history_length_bin",
            "distractor_ratio_bin",
            "evidence_position_bin",
            "evidence_bands",
        ],
        "metric_definitions": {
            "user_macro_accuracy": (
                "Mean of per-user exact-choice accuracies; each user has equal weight."
            ),
            "user_cluster_bootstrap": (
                "Percentile CI obtained by resampling users, never individual probes."
            ),
            "policy_component_accuracy": (
                "Finance/Software accuracy of each target habit variant encoded by "
                "choice_policy_signatures; stricter exact-choice accuracy is unchanged."
            ),
            "distractor_to_decisive_ratio": (
                "(visible sessions minus annotated relevant sessions) divided by "
                "the number of decisive evidence sessions."
            ),
            "mean_decisive_position_fraction": (
                "Mean decisive session index divided by the visible cutoff index; "
                "0 is early history and 1 is recent history."
            ),
            "calibration_availability": (
                "Calibration is reported only when the predictor exports full "
                "choice probabilities; hard-choice outputs are never assigned "
                "invented confidence."
            ),
            "false_personalization_availability": (
                "False-personalization cost is reported only with an explicit "
                "no-habit/applicability and option-action taxonomy."
            ),
        },
    }
    per_user = user_accuracy_rows(ordered, bundle)
    return payload, slices, diagnostics, per_user


def run(args: argparse.Namespace) -> None:
    bundle = load_dataset(args.dataset_dir, domain_filter=args.domain_filter)
    rows = read_jsonl(args.scored_predictions)
    validate_scored_rows(rows, bundle, allow_partial=args.allow_partial)
    payload, slices, diagnostics, per_user = build_supplementary_analysis(
        rows,
        bundle,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        artifact_root=args.artifact_root,
        utility_lambdas=args.utility_lambda,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "supplementary_metrics.json", payload)
    write_csv(args.output_dir / "supplementary_metrics_by_slice.csv", slices)
    write_jsonl(
        args.output_dir / "supplementary_probe_diagnostics.jsonl", diagnostics
    )
    write_csv(args.output_dir / "supplementary_metrics_by_user.csv", per_user)
    print(json.dumps(payload["accuracy"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--scored-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain-filter")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Optional run directory for recursive on-disk artifact size.",
    )
    parser.add_argument(
        "--utility-lambda",
        type=float,
        action="append",
        default=None,
        help="False-personalization penalty; repeat for a sensitivity grid.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
