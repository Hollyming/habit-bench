#!/usr/bin/env python3
"""Unblind a frozen HABIT-Bench human audit and assign release dispositions.

This command is for the data-manager stage only.  It first verifies the hashes
in ``blind_adjudication_manifest.json`` and only then opens the private audit
key.  It never changes the frozen A/B/C or adjudicated annotation files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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

CORE_VALIDITY_FIELDS = (
    "answerable_from_evidence",
    "evidence_sufficient",
    "scope_condition_correct",
    "boundary_exception_correct",
    "source_grounded",
    "privacy_safe",
)

SOFTWARE_LOCALLY_REPAIRABLE_ITEMS = {
    "AUDIT-00202",
    "AUDIT-00221",
    "AUDIT-00269",
    "AUDIT-00289",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def indexed(
    rows: Iterable[dict[str, Any]], source: Path, key: str = "item_id"
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row[key])
        if item_id in result:
            raise ValueError(f"{source}: duplicate {key} {item_id}")
        result[item_id] = row
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verify_blind_freeze(domain_root: Path) -> dict[str, Any]:
    manifest_path = (
        domain_root / "adjudication/blind_adjudication_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("gold_access") != (
        "none; this command does not accept or read an audit key"
    ):
        raise ValueError(f"{manifest_path}: blind-stage gold-access claim is missing")
    for label, metadata in manifest["files"].items():
        path = domain_root / metadata["path"]
        actual = sha256_file(path)
        if actual != metadata["sha256"]:
            raise ValueError(
                f"{manifest_path}: frozen {label} hash mismatch: "
                f"expected {metadata['sha256']}, got {actual}"
            )
    return manifest


def decide_disposition(
    row: dict[str, str],
    gold_choice_id: str,
) -> tuple[str, str, str]:
    """Apply the guide's frozen keep/modify/exclude rules.

    The four named Software rows are manually reviewed exceptions: their
    policy pair and gold are recoverable, while an extraneous placement clause
    is absent from every choice. Removing that clause (or adding it uniformly
    to all choices) is a local wording repair that does not change the target
    habit, decision unit, evidence topology, or target capability.
    """

    item_id = row["item_id"]
    domain = row["domain"]
    choice_id = row["selected_best_choice_id"].strip()

    if not choice_id:
        if domain == "software" and item_id in SOFTWARE_LOCALLY_REPAIRABLE_ITEMS:
            return (
                "modify",
                "all_choices_omit_extraneous_placement_clause",
                (
                    "Remove the unsupported placement sub-task from the query, "
                    "or add the same placement requirement to every choice; "
                    "then re-audit and regenerate the dataset version."
                ),
            )
        return (
            "exclude",
            "no_unique_grounded_choice",
            (
                "Exclude from the current release: repairing the missing scope "
                "link or target condition would change the evidence topology or "
                "target decision."
            ),
        )

    reasons: list[str] = []
    actions: list[str] = []
    if choice_id != gold_choice_id:
        reasons.append("gold_choice_id_mismatch")
        actions.append(
            f"Change gold_choice_id from {gold_choice_id} to {choice_id} in a "
            "new dataset version and rerun all affected evaluations"
        )
    if row["needs_modification"].strip() == "1":
        reasons.append(row["exclusion_reason"].strip() or "adjudicated_modification")
        if "annotation_packet_error" in row["exclusion_reason"]:
            actions.append(
                "Repair the contradictory evidence rendering and re-audit the item"
            )
        if "unnatural_language" in row["exclusion_reason"]:
            actions.append(
                "Remove duplicated or materially unnatural wording and re-audit"
            )
        if not actions:
            actions.append(
                "Apply the adjudicated local repair in a new dataset version "
                "and re-audit"
            )
    if reasons:
        return ("modify", ";".join(reasons), "; ".join(actions))

    core_failures = [
        field for field in CORE_VALIDITY_FIELDS if row[field].strip() == "0"
    ]
    if core_failures:
        raise ValueError(
            f"{item_id}: core validity failure without needs_modification=1: "
            f"{core_failures}"
        )
    return ("keep", "passes_release_rule", "No content change required")


def food_hidden_graph_validation(
    rows: list[dict[str, str]],
    keys: dict[str, dict[str, Any]],
    source_probe_key: Path,
) -> dict[str, Any]:
    source_rows = read_jsonl(source_probe_key)
    source_by_public_id = indexed(
        source_rows, source_probe_key, key="public_probe_id"
    )
    latent_rows = [
        row
        for row in rows
        if row["probe_type"] in {"direct_use", "boundary", "exception"}
    ]
    adjudicated_matches = 0
    gold_matches = 0
    valid_adjudicated = 0
    mismatch_items: list[str] = []
    for row in latent_rows:
        source = source_by_public_id[row["probe_id"]]
        graph = source["hidden_habit_graph"]
        if row["probe_type"] == "direct_use":
            expected_text = (
                graph.get("selected_variant_action") or graph["default_action"]
            )
        elif row["probe_type"] == "boundary":
            expected_text = graph["boundary_action"]
        else:
            expected_text = graph["exception_action"]
        choices = {
            str(choice["choice_id"]): choice["text"]
            for choice in json.loads(row["choices_json"])
        }
        choice_id = row["selected_best_choice_id"].strip()
        gold_choice_id = str(keys[row["item_id"]]["gold_choice_id"])
        if choices[gold_choice_id] == expected_text:
            gold_matches += 1
        else:
            mismatch_items.append(row["item_id"])
        if choice_id:
            valid_adjudicated += 1
            if choices[choice_id] == expected_text:
                adjudicated_matches += 1
    return {
        "scope": "Food latent probes: direct_use, boundary, exception",
        "items": len(latent_rows),
        "valid_adjudicated_choices": valid_adjudicated,
        "adjudicated_choice_matches_hidden_graph_action": adjudicated_matches,
        "dataset_gold_matches_hidden_graph_action": gold_matches,
        "dataset_gold_mismatches_hidden_graph_action": (
            len(latent_rows) - gold_matches
        ),
        "dataset_gold_mismatch_item_ids": mismatch_items,
        "interpretation": (
            "The source private key's hidden_habit_graph is used only after "
            "blind adjudication. A mismatch here identifies an internal "
            "gold-label consistency defect, not memory-model performance."
        ),
    }


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def summarize_group(
    rows: list[dict[str, str]],
    keys: dict[str, dict[str, Any]],
    dispositions: dict[str, str],
) -> dict[str, Any]:
    valid = [
        row for row in rows if row["selected_best_choice_id"].strip()
    ]
    gold_matches = sum(
        row["selected_best_choice_id"].strip()
        == str(keys[row["item_id"]]["gold_choice_id"])
        for row in valid
    )
    result: dict[str, Any] = {
        "items": len(rows),
        "valid_best_choice_annotations": len(valid),
        "gold_choice_matches": gold_matches,
        "gold_choice_agreement": rate(gold_matches, len(valid)),
    }
    for field in BINARY_FIELDS:
        values = [
            int(row[field]) for row in rows if row[field].strip() in {"0", "1"}
        ]
        result[f"{field}_n"] = len(values)
        result[f"{field}_rate"] = rate(sum(values), len(values))
    counts = Counter(dispositions[row["item_id"]] for row in rows)
    for disposition in ("keep", "modify", "exclude"):
        result[f"{disposition}_n"] = counts[disposition]
        result[f"{disposition}_rate"] = rate(counts[disposition], len(rows))
    return result


def finalize(
    domain_root: Path,
    audit_key: Path,
    source_probe_key: Path | None,
) -> None:
    blind_manifest = verify_blind_freeze(domain_root)

    # Private material is intentionally opened only after the blind hashes pass.
    adjudicated_path = domain_root / "adjudication/adjudicated.csv"
    rows = read_csv(adjudicated_path)
    key_rows = read_jsonl(audit_key)
    keys = indexed(key_rows, audit_key)
    if {row["item_id"] for row in rows} != set(keys):
        raise ValueError("Adjudicated annotation and private key coverage differ")

    disposition_rows: list[dict[str, Any]] = []
    disposition_by_id: dict[str, str] = {}
    for row in rows:
        item_id = row["item_id"]
        gold_choice_id = str(keys[item_id]["gold_choice_id"])
        disposition, reason, action = decide_disposition(row, gold_choice_id)
        disposition_by_id[item_id] = disposition
        selected = row["selected_best_choice_id"].strip()
        disposition_rows.append(
            {
                "item_id": item_id,
                "probe_id": row["probe_id"],
                "domain": row["domain"],
                "probe_type": row["probe_type"],
                "adjudicated_choice_id": selected,
                "gold_choice_id": gold_choice_id,
                "gold_choice_agreement": (
                    "1" if selected and selected == gold_choice_id else "0"
                ),
                "disposition": disposition,
                "disposition_reason": reason,
                "required_action": action,
                "adjudicated_exclusion_reason": row["exclusion_reason"],
                "adjudicated_notes": row["notes"],
            }
        )

    overall = summarize_group(rows, keys, disposition_by_id)
    by_probe_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["probe_type"]].append(row)
    for probe_type, probe_rows in sorted(grouped.items()):
        by_probe_rows.append(
            {
                "domain": probe_rows[0]["domain"],
                "probe_type": probe_type,
                **summarize_group(probe_rows, keys, disposition_by_id),
            }
        )

    hidden_validation = None
    if source_probe_key is not None:
        hidden_validation = food_hidden_graph_validation(
            rows, keys, source_probe_key
        )

    metrics = {
        "contract_version": "habitbench.human_audit_adjudicated.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain": rows[0]["domain"] if rows else domain_root.name,
        "blind_adjudication_manifest_sha256": sha256_file(
            domain_root / "adjudication/blind_adjudication_manifest.json"
        ),
        "blind_items": blind_manifest["items"],
        "blind_disputed_items": blind_manifest["disputed_items"],
        "adjudicated_metrics": overall,
        "food_hidden_graph_consistency": hidden_validation,
        "disposition_policy": {
            "keep": (
                "Valid selected choice agrees with gold, all core validity "
                "fields pass, and needs_modification=0."
            ),
            "modify": (
                "A local repair preserves target habit, decision unit, evidence "
                "topology, and capability (gold-label repair, evidence wording, "
                "duplicate language, or the four reviewed Software clauses)."
            ),
            "exclude": (
                "No unique grounded choice and repair would require a new scope "
                "link, target condition, or evidence topology."
            ),
            "software_local_repair_item_ids": sorted(
                SOFTWARE_LOCALLY_REPAIRABLE_ITEMS
            ),
        },
        "interpretation": (
            "Gold-choice agreement is a dataset-validity statistic, not model "
            "accuracy. A/B raw agreement and Cohen's kappa remain the frozen "
            "pre-adjudication statistics."
        ),
    }

    scored_root = domain_root / "scored"
    disposition_path = scored_root / "human_audit_dispositions.csv"
    metrics_path = scored_root / "human_audit_adjudicated_metrics.json"
    by_probe_path = scored_root / "human_audit_adjudicated_by_probe_type.csv"
    write_csv(disposition_path, disposition_rows)
    write_json(metrics_path, metrics)
    write_csv(by_probe_path, by_probe_rows)

    output_manifest = {
        "contract_version": "habitbench.human_audit_unblinding_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "blind_manifest": {
                "path": "adjudication/blind_adjudication_manifest.json",
                "sha256": sha256_file(
                    domain_root
                    / "adjudication/blind_adjudication_manifest.json"
                ),
            },
            "audit_key": {
                "path": str(audit_key.relative_to(domain_root)),
                "sha256": sha256_file(audit_key),
            },
        },
        "outputs": {},
    }
    if source_probe_key is not None:
        output_manifest["inputs"]["source_probe_key"] = {
            "path": str(source_probe_key),
            "sha256": sha256_file(source_probe_key),
        }
    for label, path in [
        ("adjudicated_metrics", metrics_path),
        ("dispositions", disposition_path),
        ("by_probe_type", by_probe_path),
    ]:
        output_manifest["outputs"][label] = {
            "path": str(path.relative_to(domain_root)),
            "sha256": sha256_file(path),
        }
    write_json(scored_root / "unblinding_manifest.json", output_manifest)
    print(
        json.dumps(
            {
                "domain": metrics["domain"],
                "items": len(rows),
                "dispositions": Counter(disposition_by_id.values()),
            },
            indent=2,
            default=dict,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--audit-key", type=Path, required=True)
    parser.add_argument(
        "--source-probe-key",
        type=Path,
        help=(
            "Optional source private/probe_key.jsonl for post-unblinding "
            "Food hidden-graph consistency validation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finalize(args.domain_root, args.audit_key, args.source_probe_key)


if __name__ == "__main__":
    main()
