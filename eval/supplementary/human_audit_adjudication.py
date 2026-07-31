#!/usr/bin/env python3
"""Mechanical helpers for blind third-reviewer adjudication.

This module deliberately has no access to the private audit key and makes no
annotation decisions. It only aligns the frozen A/B files, writes the
disagreement worksheet, and renders a complete disputed item for Reviewer C.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIXED_COLUMNS = [
    "item_id",
    "probe_id",
    "domain",
    "probe_type",
    "query",
    "choices_json",
    "annotated_evidence_packet_json",
]

ANNOTATION_COLUMNS = [
    "selected_best_choice_id",
    "answerable_from_evidence",
    "evidence_sufficient",
    "scope_condition_correct",
    "boundary_exception_correct",
    "choices_balanced",
    "language_natural",
    "source_grounded",
    "privacy_safe",
    "needs_modification",
    "exclusion_reason",
    "notes",
]

DECISION_COLUMNS = ANNOTATION_COLUMNS[:10]
BINARY_COLUMNS = DECISION_COLUMNS[1:]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indexed(rows: list[dict[str, str]], source: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = row["item_id"]
        if item_id in result:
            raise ValueError(f"{source}: duplicate item_id {item_id}")
        result[item_id] = row
    return result


def prepare(domain_root: Path) -> None:
    a_path = domain_root / "annotations/annotator_a.csv"
    b_path = domain_root / "annotations/annotator_b.csv"
    a_fields, a_rows = read_csv(a_path)
    b_fields, b_rows = read_csv(b_path)
    if a_fields != b_fields:
        raise ValueError("Annotator A/B column order differs")
    a_by_id = indexed(a_rows, a_path)
    b_by_id = indexed(b_rows, b_path)
    if set(a_by_id) != set(b_by_id):
        raise ValueError("Annotator A/B item_id sets differ")

    output_rows: list[dict[str, str]] = []
    for a_row in a_rows:
        item_id = a_row["item_id"]
        b_row = b_by_id[item_id]
        for column in FIXED_COLUMNS:
            if a_row[column] != b_row[column]:
                raise ValueError(f"{item_id}: fixed column {column} differs")
        changed = [
            column
            for column in DECISION_COLUMNS
            if a_row[column] != b_row[column]
        ]
        if not changed:
            continue
        output: dict[str, str] = {
            column: a_row[column] for column in FIXED_COLUMNS
        }
        output["disagreement_fields_json"] = json.dumps(changed)
        for column in ANNOTATION_COLUMNS:
            output[f"annotator_a_{column}"] = a_row[column]
            output[f"annotator_b_{column}"] = b_row[column]
            output[f"adjudicated_{column}"] = ""
        output["adjudication_notes"] = ""
        output_rows.append(output)

    output_fields = [
        *FIXED_COLUMNS,
        "disagreement_fields_json",
        *[f"annotator_a_{column}" for column in ANNOTATION_COLUMNS],
        *[f"annotator_b_{column}" for column in ANNOTATION_COLUMNS],
        *[f"adjudicated_{column}" for column in ANNOTATION_COLUMNS],
        "adjudication_notes",
    ]
    output_path = domain_root / "adjudication/disagreements.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"{output_path}: {len(output_rows)} disputed items")


def display_annotation(row: dict[str, str], prefix: str) -> dict[str, str]:
    return {
        column: row[f"{prefix}_{column}"] for column in ANNOTATION_COLUMNS
    }


def render(
    domain_root: Path,
    item_id: str,
    part: str,
    evidence_start: int,
    evidence_count: int | None,
) -> None:
    path = domain_root / "adjudication/disagreements.csv"
    _, rows = read_csv(path)
    matches = [row for row in rows if row["item_id"] == item_id]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one row for {item_id}, got {len(matches)}")
    row = matches[0]

    evidence_packet: list[Any] = json.loads(
        row["annotated_evidence_packet_json"]
    )
    print("=" * 100)
    print(f"ITEM_ID: {row['item_id']}")
    if part in {"all", "overview"}:
        print(f"PROBE_ID: {row['probe_id']}")
        print(f"DOMAIN: {row['domain']}")
        print(f"PROBE_TYPE: {row['probe_type']}")
        print(f"DISAGREEMENT_FIELDS: {row['disagreement_fields_json']}")
        print(f"EVIDENCE_COUNT: {len(evidence_packet)}")
        print("\nQUERY\n")
        print(row["query"])
        print("\nCHOICES\n")
        for choice in json.loads(row["choices_json"]):
            print(f"[{choice['choice_id']}] {choice['text']}\n")
        print("ANNOTATOR A\n")
        print(json.dumps(display_annotation(row, "annotator_a"), indent=2))
        print("\nANNOTATOR B\n")
        print(json.dumps(display_annotation(row, "annotator_b"), indent=2))
    if part in {"all", "evidence"}:
        if evidence_start < 1:
            raise ValueError("--evidence-start must be at least 1")
        start = evidence_start - 1
        stop = (
            len(evidence_packet)
            if evidence_count is None
            else min(len(evidence_packet), start + evidence_count)
        )
        print(
            f"\nEVIDENCE PACKET ITEMS {start + 1}-{stop} "
            f"OF {len(evidence_packet)}\n"
        )
        for index in range(start, stop):
            print(f"--- EVIDENCE {index + 1} ---")
            print(evidence_packet[index])
            print()
    print(f"\nEND_ITEM: {row['item_id']}")


def validate_annotation(
    row: dict[str, str],
    *,
    source: Path,
    valid_choice_ids: set[str],
) -> None:
    item_id = row["item_id"]
    choice_id = row["selected_best_choice_id"].strip()
    if choice_id and choice_id not in valid_choice_ids:
        raise ValueError(
            f"{source}: {item_id}: invalid selected_best_choice_id {choice_id!r}"
        )
    for column in BINARY_COLUMNS:
        value = row[column].strip()
        if value not in {"", "0", "1"}:
            raise ValueError(
                f"{source}: {item_id}: invalid {column} value {value!r}"
            )
    if row["answerable_from_evidence"].strip() == "0":
        if choice_id:
            raise ValueError(
                f"{source}: {item_id}: unanswerable item has a selected choice"
            )
        if row["needs_modification"].strip() != "1":
            raise ValueError(
                f"{source}: {item_id}: unanswerable item must need modification"
            )


def merge(domain_root: Path) -> None:
    """Freeze the blind C decisions into a complete adjudicated annotation.

    This operation deliberately does not accept or open the private audit key.
    Consensus A/B rows retain A's annotation; disputed rows use Reviewer C.
    """

    a_path = domain_root / "annotations/annotator_a.csv"
    b_path = domain_root / "annotations/annotator_b.csv"
    disagreement_path = domain_root / "adjudication/disagreements.csv"
    c_path = domain_root / "adjudication/adjudicator_c_review.csv"
    output_path = domain_root / "adjudication/adjudicated.csv"
    manifest_path = domain_root / "adjudication/blind_adjudication_manifest.json"

    a_fields, a_rows = read_csv(a_path)
    b_fields, b_rows = read_csv(b_path)
    disagreement_fields, disagreement_rows = read_csv(disagreement_path)
    c_fields, c_rows = read_csv(c_path)
    expected_c_fields = ["item_id", *ANNOTATION_COLUMNS]
    if a_fields != b_fields:
        raise ValueError("Annotator A/B column order differs")
    if c_fields != expected_c_fields:
        raise ValueError(
            f"{c_path}: expected columns {expected_c_fields}, got {c_fields}"
        )

    a_by_id = indexed(a_rows, a_path)
    b_by_id = indexed(b_rows, b_path)
    disagreement_by_id = indexed(disagreement_rows, disagreement_path)
    c_by_id = indexed(c_rows, c_path)
    if set(a_by_id) != set(b_by_id):
        raise ValueError("Annotator A/B item_id sets differ")

    computed_disagreement_ids = {
        item_id
        for item_id, a_row in a_by_id.items()
        if any(
            a_row[column] != b_by_id[item_id][column]
            for column in DECISION_COLUMNS
        )
    }
    if set(disagreement_by_id) != computed_disagreement_ids:
        raise ValueError(
            "disagreements.csv item set does not match A/B decision differences"
        )
    if set(c_by_id) != computed_disagreement_ids:
        missing = sorted(computed_disagreement_ids - set(c_by_id))
        extra = sorted(set(c_by_id) - computed_disagreement_ids)
        raise ValueError(f"Reviewer C coverage mismatch: missing={missing}, extra={extra}")

    output_rows: list[dict[str, str]] = []
    for a_row in a_rows:
        item_id = a_row["item_id"]
        b_row = b_by_id[item_id]
        for column in FIXED_COLUMNS:
            if a_row[column] != b_row[column]:
                raise ValueError(f"{item_id}: fixed column {column} differs")
        valid_choice_ids = {
            str(choice["choice_id"]) for choice in json.loads(a_row["choices_json"])
        }
        source = c_path if item_id in computed_disagreement_ids else a_path
        annotation = c_by_id[item_id] if item_id in computed_disagreement_ids else a_row
        validate_annotation(
            annotation,
            source=source,
            valid_choice_ids=valid_choice_ids,
        )
        output_rows.append(
            {
                **{column: a_row[column] for column in FIXED_COLUMNS},
                **{column: annotation[column] for column in ANNOTATION_COLUMNS},
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*FIXED_COLUMNS, *ANNOTATION_COLUMNS]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    manifest = {
        "contract_version": "human_audit_blind_adjudication.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain": output_rows[0]["domain"] if output_rows else domain_root.name,
        "items": len(output_rows),
        "disputed_items": len(computed_disagreement_ids),
        "consensus_items": len(output_rows) - len(computed_disagreement_ids),
        "merge_rule": (
            "A/B decision-consensus rows retain annotator A's complete annotation; "
            "A/B decision-disagreement rows use blind Reviewer C's complete annotation."
        ),
        "gold_access": "none; this command does not accept or read an audit key",
        "files": {},
    }
    for label, path in [
        ("annotator_a", a_path),
        ("annotator_b", b_path),
        ("disagreements", disagreement_path),
        ("adjudicator_c_review", c_path),
        ("adjudicated", output_path),
    ]:
        manifest["files"][label] = {
            "path": str(path.relative_to(domain_root)),
            "sha256": sha256_file(path),
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{output_path}: {len(output_rows)} items "
        f"({len(computed_disagreement_ids)} adjudicated by C)"
    )
    print(f"{manifest_path}: blind adjudication frozen")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--domain-root", type=Path, required=True)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--domain-root", type=Path, required=True)
    render_parser.add_argument("--item-id", required=True)
    render_parser.add_argument(
        "--part", choices=("all", "overview", "evidence"), default="all"
    )
    render_parser.add_argument("--evidence-start", type=int, default=1)
    render_parser.add_argument("--evidence-count", type=int)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--domain-root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.domain_root)
    elif args.command == "render":
        render(
            args.domain_root,
            args.item_id,
            args.part,
            args.evidence_start,
            args.evidence_count,
        )
    else:
        merge(args.domain_root)


if __name__ == "__main__":
    main()
