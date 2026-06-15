#!/usr/bin/env python
"""Summarize HABIT-Bench source and family-domain provenance.

This report is intentionally separate from scoring. It answers a reviewer
question that is easy to confuse: HABIT-Bench currently uses one real prompt
source, while each habit family is grounded in a unique representative task
domain.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


FAMILY_DOMAIN = {
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

FAMILY_CAPABILITY = {
    "format_style": "Infer recurring response format under a scoped work context.",
    "coding_review": "Infer review ordering and patch minimality for code tasks.",
    "planning_defaults": "Infer planning defaults for business travel.",
    "content_constraints": "Infer routine content constraints for weekday family meals.",
    "tool_action": "Infer when freshness checks are part of the user's workflow.",
    "risk_threshold": "Infer confirmation thresholds before costly or committing actions.",
    "meeting_prep": "Infer recurring meeting-prep document structure.",
    "drift_seasonality": "Update a habit after sustained recent counterevidence.",
    "privacy_consent": "Avoid durable use of sensitive one-off facts without consent.",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def sorted_dict(counter: Counter) -> Dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def counter_from_rows(rows: Iterable[Dict[str, Any]], key: str) -> Counter:
    return Counter(str(row.get(key, "missing")) for row in rows)


def load_dataset(dataset_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sessions_path = dataset_dir / "public" / "lifelines.jsonl"
    keys_path = dataset_dir / "private" / "probe_key.jsonl"
    if not sessions_path.exists():
        raise FileNotFoundError(f"Missing public lifelines: {sessions_path}")
    if not keys_path.exists():
        raise FileNotFoundError(f"Missing private probe key: {keys_path}")
    return read_jsonl(sessions_path), read_jsonl(keys_path)


def summarize(dataset_dir: Path, min_alignment: float) -> Dict[str, Any]:
    sessions, probe_keys = load_dataset(dataset_dir)
    session_by_id = {session["session_id"]: session for session in sessions}
    users = {session["user_id"] for session in sessions}

    family_habits: Dict[str, set] = defaultdict(set)
    family_probes = Counter()
    family_probe_types: Dict[str, Counter] = defaultdict(Counter)
    family_weighted_evidence_domains: Dict[str, Counter] = defaultdict(Counter)
    family_unique_evidence_domains: Dict[str, Counter] = defaultdict(Counter)
    family_unique_evidence_sources: Dict[str, Counter] = defaultdict(Counter)
    family_unique_evidence_seed_domains: Dict[str, Counter] = defaultdict(Counter)
    family_unique_evidence_sessions: Dict[str, set] = defaultdict(set)
    missing_evidence_ids: Dict[str, List[str]] = defaultdict(list)

    for key in probe_keys:
        graph = key.get("hidden_habit_graph") or {}
        family = key.get("habit_family") or graph.get("family") or "unknown"
        habit_id = key.get("habit_id")
        if habit_id:
            family_habits[family].add(habit_id)
        family_probes[family] += 1
        family_probe_types[family][key.get("probe_type", "missing")] += 1

        seen_for_probe = set()
        for session_id in key.get("gold_evidence_session_ids", []):
            session = session_by_id.get(session_id)
            if session is None:
                missing_evidence_ids[family].append(session_id)
                continue
            domain = session.get("domain", "missing")
            family_weighted_evidence_domains[family][domain] += 1
            if session_id in seen_for_probe:
                continue
            seen_for_probe.add(session_id)
            family_unique_evidence_sessions[family].add(session_id)

    for family, session_ids in family_unique_evidence_sessions.items():
        for session_id in session_ids:
            session = session_by_id[session_id]
            source_seed = session.get("source_seed") or {}
            family_unique_evidence_domains[family][session.get("domain", "missing")] += 1
            family_unique_evidence_sources[family][source_seed.get("source_dataset", "missing")] += 1
            family_unique_evidence_seed_domains[family][source_seed.get("domain", "missing")] += 1

    family_rows = []
    errors = []
    expected_domain_counts = Counter(FAMILY_DOMAIN.values())
    duplicate_expected_domains = {
        domain: count for domain, count in expected_domain_counts.items() if count > 1
    }
    if duplicate_expected_domains:
        errors.append(
            {
                "type": "non_unique_representative_domains",
                "duplicate_expected_domains": duplicate_expected_domains,
            }
        )
    all_families = sorted(set(FAMILY_DOMAIN) | set(family_probes))
    for family in all_families:
        expected_domain = FAMILY_DOMAIN.get(family, "unknown")
        unique_domain_counts = family_unique_evidence_domains[family]
        unique_seed_domain_counts = family_unique_evidence_seed_domains[family]
        unique_evidence_total = sum(unique_domain_counts.values())
        aligned = unique_domain_counts.get(expected_domain, 0)
        alignment_rate = pct(aligned, unique_evidence_total)
        seed_domain_total = sum(unique_seed_domain_counts.values())
        seed_aligned = unique_seed_domain_counts.get(expected_domain, 0)
        seed_alignment_rate = pct(seed_aligned, seed_domain_total)
        if unique_evidence_total and alignment_rate < min_alignment:
            errors.append(
                {
                    "type": "session_domain_alignment_below_threshold",
                    "family": family,
                    "expected_domain": expected_domain,
                    "alignment_rate": alignment_rate,
                    "unique_evidence_domain_counts": sorted_dict(unique_domain_counts),
                }
            )
        if seed_domain_total and seed_alignment_rate < min_alignment:
            errors.append(
                {
                    "type": "seed_domain_alignment_below_threshold",
                    "family": family,
                    "expected_domain": expected_domain,
                    "seed_alignment_rate": seed_alignment_rate,
                    "unique_evidence_seed_domain_counts": sorted_dict(unique_seed_domain_counts),
                }
            )
        family_rows.append(
            {
                "family": family,
                "representative_domain": expected_domain,
                "capability": FAMILY_CAPABILITY.get(family, ""),
                "selected_habits": len(family_habits[family]),
                "probes": family_probes[family],
                "probe_types": sorted_dict(family_probe_types[family]),
                "unique_gold_evidence_sessions": unique_evidence_total,
                "gold_evidence_domain_counts_unique": sorted_dict(unique_domain_counts),
                "gold_evidence_domain_counts_probe_weighted": sorted_dict(
                    family_weighted_evidence_domains[family]
                ),
                "gold_evidence_source_datasets_unique": sorted_dict(
                    family_unique_evidence_sources[family]
                ),
                "gold_evidence_seed_domains_unique": sorted_dict(
                    family_unique_evidence_seed_domains[family]
                ),
                "domain_alignment_rate": alignment_rate,
                "seed_domain_alignment_rate": seed_alignment_rate,
                "missing_evidence_ids": sorted(set(missing_evidence_ids[family])),
            }
        )

    session_source_datasets = Counter()
    session_seed_domains = Counter()
    for session in sessions:
        source_seed = session.get("source_seed") or {}
        session_source_datasets[source_seed.get("source_dataset", "missing")] += 1
        session_seed_domains[source_seed.get("domain", "missing")] += 1

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "status": "pass" if not errors else "fail",
        "min_alignment": min_alignment,
        "counts": {
            "users": len(users),
            "sessions": len(sessions),
            "probes": len(probe_keys),
            "families": len(all_families),
        },
        "source_contract": {
            "real_prompt_seed_source": sorted_dict(session_source_datasets),
            "seed_domains": sorted_dict(session_seed_domains),
            "controlled_synthetic_components": [
                "hidden_habit_graphs",
                "assistant_feedback",
                "counterfactual_probe_contexts",
                "answer_choices",
                "gold_labels",
                "evidence_links",
            ],
            "interpretation": (
                "The current release is single-source in external provenance "
                "and uses one unique representative task domain per habit family."
            ),
            "family_domain_contract": "nine_unique_representative_domains",
        },
        "expected_family_domains_unique": not duplicate_expected_domains,
        "expected_family_domains": dict(sorted(FAMILY_DOMAIN.items())),
        "session_domain_counts": sorted_dict(counter_from_rows(sessions, "domain")),
        "family_rows": family_rows,
        "errors": errors,
    }


def format_counter(value: Dict[str, int]) -> str:
    if not value:
        return "-"
    return ", ".join(f"{key}: {count}" for key, count in value.items())


def write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# HABIT-Bench Domain Provenance Summary",
        "",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Status: `{summary['status']}`",
        f"- Users: {summary['counts']['users']}",
        f"- Sessions: {summary['counts']['sessions']}",
        f"- Probes: {summary['counts']['probes']}",
        "",
        "## Source Contract",
        "",
        "HABIT-Bench currently uses a single real external prompt source and "
        "nine unique representative task domains. The real prompts provide "
        "task surface form; hidden habits, feedback, probes, answer choices, "
        "labels, and evidence links are controlled synthetic components.",
        "",
        f"- Real prompt seed source: {format_counter(summary['source_contract']['real_prompt_seed_source'])}",
        f"- Family-domain contract: `{summary['source_contract']['family_domain_contract']}`",
        f"- Seed domain buckets: {format_counter(summary['source_contract']['seed_domains'])}",
        f"- Session domains: {format_counter(summary['session_domain_counts'])}",
        "",
        "## 9-Family Table",
        "",
        "| family | representative domain | selected habits | probes | unique gold evidence sessions | session alignment | seed alignment | evidence source datasets | evidence domains | seed domains | capability |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "| {family} | {domain} | {habits} | {probes} | {evidence} | {alignment:.1%} | {seed_alignment:.1%} | {sources} | {domains} | {seed_domains} | {capability} |".format(
                family=f"`{row['family']}`",
                domain=row["representative_domain"],
                habits=row["selected_habits"],
                probes=row["probes"],
                evidence=row["unique_gold_evidence_sessions"],
                alignment=row["domain_alignment_rate"],
                seed_alignment=row["seed_domain_alignment_rate"],
                sources=format_counter(row["gold_evidence_source_datasets_unique"]),
                domains=format_counter(row["gold_evidence_domain_counts_unique"]),
                seed_domains=format_counter(row["gold_evidence_seed_domains_unique"]),
                capability=row["capability"],
            )
        )

    lines.extend(
        [
            "",
            "## Reviewer-Facing Interpretation",
            "",
            "Do not describe the current split as drawing each habit family from a "
            "different external dataset. The accurate claim is that all real task "
            "seeds come from WildChat, then are filtered into domain buckets that "
            "ground the nine controlled habit families with unique representative "
            "domains.",
        ]
    )
    if summary["errors"]:
        lines.extend(["", "## Alignment Errors", ""])
        for error in summary["errors"]:
            lines.append(f"- `{error['family']}`: {error}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix without extension. Defaults to reports/domain_provenance_summary.",
    )
    parser.add_argument("--min-alignment", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    output_prefix = args.output_prefix or dataset_dir / "reports" / "domain_provenance_summary"
    summary = summarize(dataset_dir, args.min_alignment)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    (output_prefix.with_suffix(".json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(output_prefix.with_suffix(".md"), summary)
    print(json.dumps({"status": summary["status"], **summary["counts"]}, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit("Domain provenance alignment check failed")


if __name__ == "__main__":
    main()
