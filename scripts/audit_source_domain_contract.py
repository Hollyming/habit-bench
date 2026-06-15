#!/usr/bin/env python
"""Audit HABIT-Bench source and family-domain contracts from raw rows.

This is stricter than the provenance summary: it is meant to be a completion
gate for the dataset construction claim that HABIT-Bench is single-source in
external prompt provenance while using one representative task domain per habit
family.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


EXPECTED_FAMILY_DOMAIN = {
    "format_style": "work",
    "coding_review": "code",
    "planning_defaults": "travel",
    "content_constraints": "food",
    "tool_action": "news",
    "risk_threshold": "commitment",
    "meeting_prep": "meeting",
    "drift_seasonality": "equipment",
    "privacy_consent": "privacy",
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sorted_counter(counter: Counter) -> Dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def format_counter(counter: Dict[str, int]) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}: {counter[key]}" for key in sorted(counter))


def load_optional_contract(dataset_dir: Path) -> Dict[str, Any]:
    manifest_path = first_existing(
        [
            dataset_dir / "reports" / "official_subset_manifest.json",
            dataset_dir / "reports" / "balanced_v03_manifest.json",
            dataset_dir / "reports" / "v02_manifest.json",
            dataset_dir / "reports" / "build_manifest.json",
        ]
    )
    provenance_path = dataset_dir / "reports" / "domain_provenance_summary.json"
    manifest = read_json(manifest_path) if manifest_path else {}
    provenance = read_json(provenance_path) if provenance_path.exists() else {}
    return {
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest": manifest,
        "provenance_path": str(provenance_path) if provenance_path.exists() else None,
        "provenance": provenance,
    }


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_dir = args.dataset_dir
    lifelines_path = dataset_dir / "public" / "lifelines.jsonl"
    probes_path = dataset_dir / "public" / "probes.jsonl"
    key_path = dataset_dir / "private" / "probe_key.jsonl"
    errors: List[str] = []
    warnings: List[str] = []

    for label, path in {
        "lifelines": lifelines_path,
        "probes": probes_path,
        "probe_key": key_path,
    }.items():
        if not path.exists():
            errors.append(f"missing_required_file:{label}:{path}")

    sessions = read_jsonl(lifelines_path) if lifelines_path.exists() else []
    probes = read_jsonl(probes_path) if probes_path.exists() else []
    keys = read_jsonl(key_path) if key_path.exists() else []
    session_by_id = {session.get("session_id"): session for session in sessions}
    public_probe_ids = {probe.get("probe_id") for probe in probes}
    key_public_probe_ids = {key.get("public_probe_id") for key in keys}
    users = {session.get("user_id") for session in sessions}

    missing_key_for_public_probe = sorted(public_probe_ids - key_public_probe_ids)
    key_without_public_probe = sorted(key_public_probe_ids - public_probe_ids)
    if missing_key_for_public_probe:
        errors.append(f"public_probe_missing_private_key:count={len(missing_key_for_public_probe)}")
    if key_without_public_probe:
        errors.append(f"private_key_missing_public_probe:count={len(key_without_public_probe)}")

    all_session_sources = Counter()
    all_seed_domains = Counter()
    all_session_domains = Counter()
    missing_source_seed = 0
    for session in sessions:
        all_session_domains[str(session.get("domain", "missing"))] += 1
        source_seed = session.get("source_seed") or {}
        if not source_seed:
            missing_source_seed += 1
        all_session_sources[str(source_seed.get("source_dataset", "missing"))] += 1
        all_seed_domains[str(source_seed.get("domain", "missing"))] += 1
    if missing_source_seed:
        errors.append(f"missing_source_seed:count={missing_source_seed}")
    unexpected_sources = {
        source: count
        for source, count in all_session_sources.items()
        if source != args.expected_source
    }
    if unexpected_sources:
        errors.append(f"unexpected_source_dataset:{sorted_counter(unexpected_sources)}")

    expected_domains = Counter(EXPECTED_FAMILY_DOMAIN.values())
    duplicate_domains = {domain: count for domain, count in expected_domains.items() if count > 1}
    if args.require_unique_domains and duplicate_domains:
        errors.append(f"non_unique_representative_domains:{sorted_counter(duplicate_domains)}")

    family_probe_counts = Counter()
    family_habits: Dict[str, set] = defaultdict(set)
    family_probe_types: Dict[str, Counter] = defaultdict(Counter)
    family_evidence_sessions: Dict[str, set] = defaultdict(set)
    family_evidence_domains: Dict[str, Counter] = defaultdict(Counter)
    family_evidence_seed_domains: Dict[str, Counter] = defaultdict(Counter)
    family_evidence_sources: Dict[str, Counter] = defaultdict(Counter)
    family_missing_evidence_ids: Dict[str, List[str]] = defaultdict(list)
    family_bad_evidence_rows: Dict[str, List[str]] = defaultdict(list)

    for key in keys:
        graph = key.get("hidden_habit_graph") or {}
        family = key.get("habit_family") or graph.get("family") or "missing"
        expected_domain = EXPECTED_FAMILY_DOMAIN.get(family)
        family_probe_counts[family] += 1
        family_probe_types[family][str(key.get("probe_type", "missing"))] += 1
        if key.get("habit_id"):
            family_habits[family].add(key["habit_id"])
        if family not in EXPECTED_FAMILY_DOMAIN:
            family_bad_evidence_rows[family].append("unexpected_family")
            continue

        for session_id in key.get("gold_evidence_session_ids", []):
            session = session_by_id.get(session_id)
            if session is None:
                family_missing_evidence_ids[family].append(session_id)
                continue
            source_seed = session.get("source_seed") or {}
            session_domain = str(session.get("domain", "missing"))
            seed_domain = str(source_seed.get("domain", "missing"))
            source_dataset = str(source_seed.get("source_dataset", "missing"))
            family_evidence_sessions[family].add(session_id)
            family_evidence_domains[family][session_domain] += 1
            family_evidence_seed_domains[family][seed_domain] += 1
            family_evidence_sources[family][source_dataset] += 1
            if session_domain != expected_domain:
                family_bad_evidence_rows[family].append(
                    f"session_domain:{session_id}:expected={expected_domain}:actual={session_domain}"
                )
            if seed_domain != expected_domain:
                family_bad_evidence_rows[family].append(
                    f"seed_domain:{session_id}:expected={expected_domain}:actual={seed_domain}"
                )
            if source_dataset != args.expected_source:
                family_bad_evidence_rows[family].append(
                    f"source_dataset:{session_id}:expected={args.expected_source}:actual={source_dataset}"
                )

    observed_families = set(family_probe_counts)
    expected_families = set(EXPECTED_FAMILY_DOMAIN)
    missing_families = sorted(expected_families - observed_families)
    unexpected_families = sorted(observed_families - expected_families)
    if missing_families:
        errors.append(f"missing_expected_families:{missing_families}")
    if unexpected_families:
        errors.append(f"unexpected_families:{unexpected_families}")

    if args.expected_probes is not None and len(probes) != args.expected_probes:
        errors.append(f"probe_count_mismatch:expected={args.expected_probes}:actual={len(probes)}")
    if args.expected_keys is not None and len(keys) != args.expected_keys:
        errors.append(f"key_count_mismatch:expected={args.expected_keys}:actual={len(keys)}")
    if args.expected_families is not None and len(observed_families) != args.expected_families:
        errors.append(
            f"family_count_mismatch:expected={args.expected_families}:actual={len(observed_families)}"
        )
    if args.expected_probes_per_family is not None:
        for family in sorted(expected_families):
            actual = family_probe_counts.get(family, 0)
            if actual != args.expected_probes_per_family:
                errors.append(
                    "family_probe_count_mismatch:"
                    f"{family}:expected={args.expected_probes_per_family}:actual={actual}"
                )

    family_rows = []
    for family in sorted(expected_families | observed_families):
        expected_domain = EXPECTED_FAMILY_DOMAIN.get(family)
        evidence_domains = family_evidence_domains[family]
        evidence_seed_domains = family_evidence_seed_domains[family]
        evidence_sources = family_evidence_sources[family]
        evidence_total = sum(evidence_domains.values())
        aligned_session_domains = evidence_domains.get(expected_domain, 0)
        aligned_seed_domains = evidence_seed_domains.get(expected_domain, 0)
        source_aligned = evidence_sources.get(args.expected_source, 0)
        if family_missing_evidence_ids[family]:
            errors.append(
                f"missing_gold_evidence_session_ids:{family}:count={len(set(family_missing_evidence_ids[family]))}"
            )
        if family_bad_evidence_rows[family]:
            errors.append(
                f"family_evidence_contract_violations:{family}:count={len(family_bad_evidence_rows[family])}"
            )

        family_rows.append(
            {
                "family": family,
                "representative_domain": expected_domain,
                "probes": family_probe_counts.get(family, 0),
                "selected_habits": len(family_habits[family]),
                "probe_types": sorted_counter(family_probe_types[family]),
                "unique_gold_evidence_sessions": len(family_evidence_sessions[family]),
                "gold_evidence_domain_counts": sorted_counter(evidence_domains),
                "gold_evidence_seed_domain_counts": sorted_counter(evidence_seed_domains),
                "gold_evidence_source_counts": sorted_counter(evidence_sources),
                "session_domain_alignment_rate": pct(aligned_session_domains, evidence_total),
                "seed_domain_alignment_rate": pct(aligned_seed_domains, evidence_total),
                "source_alignment_rate": pct(source_aligned, evidence_total),
                "missing_evidence_ids": sorted(set(family_missing_evidence_ids[family])),
                "contract_violation_examples": family_bad_evidence_rows[family][:10],
            }
        )

    optional_contract = load_optional_contract(dataset_dir)
    manifest = optional_contract["manifest"]
    provenance = optional_contract["provenance"]
    manifest_source_contract = manifest.get("source_contract", {})
    provenance_source_contract = provenance.get("source_contract", {})
    if manifest_source_contract:
        if manifest_source_contract.get("seed_prompts") != args.expected_source:
            errors.append(
                "manifest_source_contract_unexpected:"
                f"{manifest_source_contract.get('seed_prompts')}"
            )
        if manifest_source_contract.get("family_domain_contract") != "nine_unique_representative_domains":
            errors.append(
                "manifest_family_domain_contract_unexpected:"
                f"{manifest_source_contract.get('family_domain_contract')}"
            )
    else:
        warnings.append("manifest_source_contract_missing")
    if provenance:
        if provenance.get("status") != "pass":
            errors.append(f"domain_provenance_status_not_pass:{provenance.get('status')}")
        if provenance_source_contract.get("family_domain_contract") != "nine_unique_representative_domains":
            errors.append(
                "provenance_family_domain_contract_unexpected:"
                f"{provenance_source_contract.get('family_domain_contract')}"
            )
    else:
        warnings.append("domain_provenance_summary_missing")

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "dataset_dir": str(dataset_dir),
        "expected_source": args.expected_source,
        "expected_family_domain": dict(sorted(EXPECTED_FAMILY_DOMAIN.items())),
        "expected_family_domains_unique": not duplicate_domains,
        "counts": {
            "users": len(users),
            "sessions": len(sessions),
            "public_probes": len(probes),
            "private_keys": len(keys),
            "families": len(observed_families),
            "missing_key_for_public_probe": len(missing_key_for_public_probe),
            "key_without_public_probe": len(key_without_public_probe),
        },
        "session_source_dataset_counts": sorted_counter(all_session_sources),
        "session_seed_domain_counts": sorted_counter(all_seed_domains),
        "session_domain_counts": sorted_counter(all_session_domains),
        "manifest_path": optional_contract["manifest_path"],
        "provenance_path": optional_contract["provenance_path"],
        "family_rows": family_rows,
        "errors": errors,
        "warnings": warnings,
    }


def write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Source/Domain Contract Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Expected source: `{summary['expected_source']}`",
        f"- Users: {summary['counts']['users']}",
        f"- Sessions: {summary['counts']['sessions']}",
        f"- Public probes: {summary['counts']['public_probes']}",
        f"- Private keys: {summary['counts']['private_keys']}",
        f"- Families: {summary['counts']['families']}",
        f"- Session source datasets: {format_counter(summary['session_source_dataset_counts'])}",
        f"- Session seed domains: {format_counter(summary['session_seed_domain_counts'])}",
        "",
        "## Family Contract",
        "",
        "| family | representative domain | probes | habits | evidence sessions | session-domain alignment | seed-domain alignment | source alignment |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "| {family} | {domain} | {probes} | {habits} | {evidence} | {session:.3f} | {seed:.3f} | {source:.3f} |".format(
                family=row["family"],
                domain=row["representative_domain"],
                probes=row["probes"],
                habits=row["selected_habits"],
                evidence=row["unique_gold_evidence_sessions"],
                session=row["session_domain_alignment_rate"],
                seed=row["seed_domain_alignment_rate"],
                source=row["source_alignment_rate"],
            )
        )
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in summary["errors"]) if summary["errors"] else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary["warnings"]) if summary["warnings"] else lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--expected-source", default="allenai/WildChat")
    parser.add_argument("--expected-probes", type=int, default=None)
    parser.add_argument("--expected-keys", type=int, default=None)
    parser.add_argument("--expected-families", type=int, default=9)
    parser.add_argument("--expected-probes-per-family", type=int, default=None)
    parser.add_argument("--require-unique-domains", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit(args)
    out_json = args.out_json or args.dataset_dir / "reports" / "source_domain_contract_audit.json"
    out_md = args.out_md or args.dataset_dir / "reports" / "source_domain_contract_audit.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(out_md, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "errors": len(summary["errors"]),
                "warnings": len(summary["warnings"]),
                "out_json": str(out_json),
                "out_md": str(out_md),
            },
            indent=2,
        )
    )
    if summary["errors"]:
        raise SystemExit("Source/domain contract audit failed")


if __name__ == "__main__":
    main()
