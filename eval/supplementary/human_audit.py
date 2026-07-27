#!/usr/bin/env python
"""Prepare and score a blinded, stratified HABIT-Bench human audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from eval.controls import render_session
from eval.core.dataset import DatasetContractError, load_dataset
from eval.core.io import write_csv, write_json, write_jsonl
from eval.supplementary.oracle_controls import oracle_evidence_ids


BINARY_FIELDS = (
    "answerable_from_evidence",
    "evidence_sufficient",
    "scope_condition_correct",
    "boundary_exception_correct",
    "choices_balanced",
    "language_natural",
    "source_grounded",
    "privacy_safe",
    "needs_modification",
)


def _parse_binary(value: Any) -> int | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "yes", "y", "true", "pass"}:
        return 1
    if normalized in {"0", "no", "n", "false", "fail"}:
        return 0
    return None


def _strata(bundle: Any) -> dict[tuple[str, str], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for probe in bundle.probes:
        key = bundle.keys[probe["probe_id"]]
        buckets[
            (
                str(probe.get("domain", "unknown")),
                str(key.get("probe_type", "unknown")),
            )
        ].append(probe)
    return dict(buckets)


def prepare_audit(args: argparse.Namespace) -> None:
    if args.per_stratum < 1:
        raise DatasetContractError("--per-stratum must be positive")
    bundle = load_dataset(args.dataset_dir, domain_filter=args.domain_filter)
    generator = random.Random(args.seed)
    sampled: list[dict[str, Any]] = []
    stratum_counts: dict[str, dict[str, int]] = {}
    for stratum, probes in sorted(_strata(bundle).items()):
        ordered = sorted(probes, key=lambda row: str(row["probe_id"]))
        count = min(args.per_stratum, len(ordered))
        selected = generator.sample(ordered, count)
        sampled.extend(selected)
        stratum_counts["/".join(stratum)] = {
            "available": len(ordered),
            "sampled": count,
        }
    generator.shuffle(sampled)

    session_lookup = {
        str(session["session_id"]): session
        for sessions in bundle.sessions_by_user.values()
        for session in sessions
    }
    templates: list[dict[str, Any]] = []
    audit_keys: list[dict[str, Any]] = []
    for index, probe in enumerate(sampled, start=1):
        probe_id = str(probe["probe_id"])
        key = bundle.keys[probe_id]
        evidence_ids = list(
            dict.fromkeys(
                oracle_evidence_ids(key)
                + list(key.get("nonbinding_evidence_session_ids") or [])
            )
        )
        evidence_ids.sort(
            key=lambda session_id: (
                int(session_lookup[session_id]["session_index"])
                if session_id in session_lookup
                else float("inf")
            )
        )
        evidence = [
            render_session(session_lookup[session_id])
            for session_id in evidence_ids
            if session_id in session_lookup
        ]
        item_id = f"AUDIT-{index:05d}"
        row = {
            "item_id": item_id,
            "probe_id": probe_id,
            "domain": probe.get("domain", "unknown"),
            "probe_type": key.get("probe_type", "unknown"),
            "query": probe["query"],
            "choices_json": json.dumps(
                probe["choices"], ensure_ascii=False, sort_keys=True
            ),
            "annotated_evidence_packet_json": json.dumps(
                evidence, ensure_ascii=False
            ),
            "selected_best_choice_id": "",
            **{field: "" for field in BINARY_FIELDS},
            "exclusion_reason": "",
            "notes": "",
        }
        templates.append(row)
        audit_keys.append(
            {
                "item_id": item_id,
                "probe_id": probe_id,
                "domain": probe.get("domain", "unknown"),
                "probe_type": key.get("probe_type", "unknown"),
                "valid_choice_ids": [
                    choice["choice_id"] for choice in probe["choices"]
                ],
                "gold_choice_id": key["gold_choice_id"],
                "evidence_session_ids": evidence_ids,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "annotation_template.csv", templates)
    write_jsonl(args.output_dir / "audit_key.private.jsonl", audit_keys)
    write_json(
        args.output_dir / "audit_manifest.json",
        {
            "contract_version": "habitbench.human_audit.v1",
            "dataset": bundle.manifest,
            "blinding": (
                "Give annotators annotation_template.csv only. "
                "audit_key.private.jsonl contains gold labels."
            ),
            "sampling": {
                "strategy": "uniform within domain x probe_type strata",
                "per_stratum": args.per_stratum,
                "seed": args.seed,
                "items": len(templates),
                "strata": stratum_counts,
            },
            "binary_encoding": "1=yes/pass, 0=no/fail",
            "binary_fields": list(BINARY_FIELDS),
        },
    )
    print(json.dumps({"items": len(templates), "strata": len(stratum_counts)}, indent=2))


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            raise DatasetContractError(f"Missing item_id in {path}")
        if item_id in result:
            raise DatasetContractError(f"Duplicate item_id {item_id} in {path}")
        result[item_id] = row
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cohen_kappa(left: list[str], right: list[str]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("Kappa inputs must have equal length")
    if not left:
        return {"n": 0, "raw_agreement": None, "cohen_kappa": None}
    categories = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[category] / len(left) * right_counts[category] / len(right)
        for category in categories
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {
        "n": len(left),
        "raw_agreement": round(observed, 6),
        "cohen_kappa": round(kappa, 6) if kappa is not None else None,
    }


def _common_values(
    left: dict[str, dict[str, str]],
    right: dict[str, dict[str, str]],
    item_ids: Iterable[str],
    field: str,
    *,
    binary: bool,
) -> tuple[list[str], list[str]]:
    left_values: list[str] = []
    right_values: list[str] = []
    for item_id in item_ids:
        left_raw = left[item_id].get(field)
        right_raw = right[item_id].get(field)
        if binary:
            left_value = _parse_binary(left_raw)
            right_value = _parse_binary(right_raw)
            if left_value is None or right_value is None:
                continue
            left_values.append(str(left_value))
            right_values.append(str(right_value))
        else:
            left_value = str(left_raw or "").strip()
            right_value = str(right_raw or "").strip()
            if not left_value or not right_value:
                continue
            left_values.append(left_value)
            right_values.append(right_value)
    return left_values, right_values


def score_audit(args: argparse.Namespace) -> None:
    key_rows = _read_jsonl(args.audit_key)
    keys = {str(row["item_id"]): row for row in key_rows}
    annotations: dict[str, dict[str, dict[str, str]]] = {}
    for value in args.annotation:
        if "=" not in value:
            raise DatasetContractError("--annotation must be ANNOTATOR=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in annotations:
            raise DatasetContractError("Annotator names must be non-empty and unique")
        annotations[name] = _read_csv(Path(raw_path))
    if len(annotations) < 2:
        raise DatasetContractError("At least two --annotation inputs are required")
    expected = set(keys)
    for name, rows in annotations.items():
        if set(rows) != expected:
            raise DatasetContractError(
                f"Annotation coverage mismatch for {name}: "
                f"missing={len(expected - set(rows))}, extra={len(set(rows) - expected)}"
            )

    annotator_rows: list[dict[str, Any]] = []
    for name, rows in sorted(annotations.items()):
        selected = [
            (item_id, str(row.get("selected_best_choice_id", "")).strip())
            for item_id, row in rows.items()
        ]
        completed = [
            (item_id, choice_id)
            for item_id, choice_id in selected
            if choice_id in set(map(str, keys[item_id]["valid_choice_ids"]))
        ]
        gold_matches = sum(
            choice_id == str(keys[item_id]["gold_choice_id"])
            for item_id, choice_id in completed
        )
        summary: dict[str, Any] = {
            "annotator": name,
            "items": len(rows),
            "valid_best_choice_annotations": len(completed),
            "gold_choice_agreement": (
                round(gold_matches / len(completed), 6) if completed else None
            ),
        }
        for field in BINARY_FIELDS:
            values = [
                value
                for row in rows.values()
                if (value := _parse_binary(row.get(field))) is not None
            ]
            summary[f"{field}_n"] = len(values)
            summary[f"{field}_rate"] = (
                round(sum(values) / len(values), 6) if values else None
            )
        annotator_rows.append(summary)

    agreement_rows: list[dict[str, Any]] = []
    for left_name, right_name in combinations(sorted(annotations), 2):
        fields = ("selected_best_choice_id",) + BINARY_FIELDS
        for field in fields:
            left_values, right_values = _common_values(
                annotations[left_name],
                annotations[right_name],
                sorted(expected),
                field,
                binary=field in BINARY_FIELDS,
            )
            agreement_rows.append(
                {
                    "left_annotator": left_name,
                    "right_annotator": right_name,
                    "field": field,
                    **cohen_kappa(left_values, right_values),
                }
            )

    payload = {
        "contract_version": "habitbench.human_audit_scores.v1",
        "items": len(keys),
        "annotators": sorted(annotations),
        "annotator_metrics": annotator_rows,
        "inter_annotator_agreement": agreement_rows,
        "interpretation": (
            "Gold-choice agreement measures dataset validity, not model accuracy. "
            "Report raw agreement beside Cohen's kappa and adjudicate disagreements."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "human_audit_metrics.json", payload)
    write_csv(args.output_dir / "human_audit_by_annotator.csv", annotator_rows)
    write_csv(args.output_dir / "human_audit_agreement.csv", agreement_rows)
    print(json.dumps({"items": len(keys), "annotators": sorted(annotations)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create a blinded audit sample.")
    prepare.add_argument("--dataset-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--domain-filter")
    prepare.add_argument("--per-stratum", type=int, default=20)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(function=prepare_audit)

    score = subparsers.add_parser("score", help="Score two or more annotation files.")
    score.add_argument("--audit-key", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument(
        "--annotation",
        action="append",
        required=True,
        help="ANNOTATOR=/path/to/completed.csv; repeat at least twice.",
    )
    score.set_defaults(function=score_audit)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    parsed_args.function(parsed_args)
