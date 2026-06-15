#!/usr/bin/env python
"""Create a stratified HABIT-Bench subset for expensive official-method runs."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs" / "habit_bench_balanced_v0_3"
DEFAULT_OUTPUT = ROOT / "runs" / "habit_bench_balanced_v0_3_official_subset_90"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_output_dir(output_dir: Path) -> None:
    runs_root = (ROOT / "runs").resolve()
    target = output_dir.resolve()
    if not str(target).lower().startswith(str(runs_root).lower()):
        raise ValueError(f"Refusing to delete output outside runs directory: {target}")
    if target.exists():
        shutil.rmtree(target)
    output_dir.mkdir(parents=True, exist_ok=True)


def select_keys(
    keys: List[Dict[str, Any]],
    total: int,
    seed: int,
    include_variants: str,
    min_per_capability: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    filtered = [
        key
        for key in keys
        if include_variants == "all" or key.get("stress_variant") == include_variants
    ]
    families = sorted({key["habit_family"] for key in filtered})
    if not families:
        raise ValueError("No probes available after variant filtering.")
    if total < len(families):
        raise ValueError(f"total_probes={total} is smaller than family count={len(families)}")

    family_quota = {family: total // len(families) for family in families}
    for family in families[: total % len(families)]:
        family_quota[family] += 1

    family_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for key in filtered:
        family_buckets[key["habit_family"]].append(key)
    short_families = {
        family: len(rows)
        for family, rows in family_buckets.items()
        if len(rows) < family_quota[family]
    }
    if short_families:
        raise ValueError(f"Not enough probes for family-balanced subset: {short_families}")

    selected = []
    selected_ids = set()

    # First reserve rare/special capabilities under the per-family quotas. This
    # keeps privacy and drift probes visible without letting them dominate the
    # official subset.
    capability_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for key in filtered:
        capability_buckets[key["capability_group"]].append(key)
    for capability, rows in sorted(
        capability_buckets.items(),
        key=lambda item: (len(item[1]), item[0]),
    ):
        target = min(min_per_capability, len(rows))
        if sum(1 for row in selected if row["capability_group"] == capability) >= target:
            continue
        rows = sorted(rows, key=lambda row: row["probe_id"])
        rng.shuffle(rows)
        for row in rows:
            if sum(1 for item in selected if item["capability_group"] == capability) >= target:
                break
            family = row["habit_family"]
            if row["probe_id"] in selected_ids:
                continue
            if sum(1 for item in selected if item["habit_family"] == family) >= family_quota[family]:
                continue
            selected.append(row)
            selected_ids.add(row["probe_id"])

    # Then fill each family quota, round-robin over capability buckets so a
    # family does not collapse to only paraphrases or only one probe type.
    for family in families:
        family_rows = [row for row in family_buckets[family] if row["probe_id"] not in selected_ids]
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in family_rows:
            grouped[(row["capability_group"], row.get("stress_variant", "unknown"))].append(row)
        for bucket in grouped.values():
            bucket.sort(key=lambda row: row["probe_id"])
            rng.shuffle(bucket)
        bucket_keys = sorted(grouped)
        while sum(1 for item in selected if item["habit_family"] == family) < family_quota[family]:
            progressed = False
            for bucket_key in bucket_keys:
                bucket = grouped[bucket_key]
                if not bucket:
                    continue
                row = bucket.pop()
                if row["probe_id"] in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(row["probe_id"])
                progressed = True
                if sum(1 for item in selected if item["habit_family"] == family) >= family_quota[family]:
                    break
            if not progressed:
                raise ValueError(f"Could not fill family quota for {family}")

    if len(selected) < total:
        remaining = [row for row in filtered if row["probe_id"] not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: total - len(selected)])
    elif len(selected) > total:
        raise ValueError(f"Family quotas selected {len(selected)} probes for total={total}")

    capability_counts = Counter(row["capability_group"] for row in selected)
    missing_capabilities = {
        capability: min(min_per_capability, len(rows))
        for capability, rows in capability_buckets.items()
        if capability_counts[capability] < min(min_per_capability, len(rows))
    }
    if missing_capabilities:
        raise ValueError(
            "Family-balanced selection could not satisfy min_per_capability: "
            f"counts={dict(capability_counts)}, targets={missing_capabilities}"
        )

    return sorted(selected, key=lambda row: row["public_probe_id"])


def subset(args: argparse.Namespace) -> None:
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    prepare_output_dir(output_dir)

    public_probes = read_jsonl(input_dir / "public" / "probes.jsonl")
    private_keys = read_jsonl(input_dir / "private" / "probe_key.jsonl")
    public_sessions = read_jsonl(input_dir / "public" / "lifelines.jsonl")
    private_sessions = read_jsonl(input_dir / "private" / "sessions_with_annotations.jsonl")

    selected_keys = select_keys(
        private_keys,
        args.total_probes,
        args.seed,
        args.include_variants,
        args.min_per_capability,
    )
    selected_public_ids = {key["public_probe_id"] for key in selected_keys}
    selected_probes = [probe for probe in public_probes if probe["probe_id"] in selected_public_ids]
    selected_user_ids = {probe["user_id"] for probe in selected_probes}

    max_scope = defaultdict(lambda: -1)
    for probe in selected_probes:
        max_scope[probe["user_id"]] = max(
            max_scope[probe["user_id"]],
            probe["visible_history_scope"]["max_session_index"],
        )
    selected_public_sessions = [
        session
        for session in public_sessions
        if session["user_id"] in selected_user_ids
        and session["session_index"] <= max_scope[session["user_id"]]
    ]
    selected_private_sessions = [
        session
        for session in private_sessions
        if session["user_id"] in selected_user_ids
        and session["session_index"] <= max_scope[session["user_id"]]
    ]

    session_ids = {session["session_id"] for session in selected_private_sessions}
    missing_evidence = {
        key["probe_id"]: [
            sid for sid in key.get("gold_evidence_session_ids", []) if sid not in session_ids
        ]
        for key in selected_keys
    }
    missing_evidence = {key: value for key, value in missing_evidence.items() if value}
    if missing_evidence:
        raise SystemExit(f"Missing evidence sessions in subset: {missing_evidence}")

    write_jsonl(output_dir / "public" / "probes.jsonl", selected_probes)
    write_jsonl(output_dir / "public" / "lifelines.jsonl", selected_public_sessions)
    write_jsonl(output_dir / "private" / "probe_key.jsonl", selected_keys)
    write_jsonl(output_dir / "private" / "sessions_with_annotations.jsonl", selected_private_sessions)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "official_full_config_subset",
        "input_dir": str(input_dir),
        "source_contract": {
            "seed_prompts": "allenai/WildChat",
            "domain_assignment": "keyword-filtered WildChat task seed buckets",
            "family_domain_contract": "nine_unique_representative_domains",
            "release_claim": "real-prompt-seeded, domain-grounded, synthetic longitudinal habit benchmark",
            "claim_to_avoid": "each habit family is drawn from a different external dataset",
            "controlled_components": [
                "hidden_habit_graphs",
                "assistant feedback",
                "counterfactual probes",
                "answer choices",
                "gold labels",
                "evidence links",
            ],
        },
        "seed": args.seed,
        "include_variants": args.include_variants,
        "counts": {
            "users": len(selected_user_ids),
            "sessions": len(selected_public_sessions),
            "probes": len(selected_probes),
            "keys": len(selected_keys),
        },
        "by_capability_group": dict(Counter(key["capability_group"] for key in selected_keys)),
        "by_habit_family": dict(Counter(key["habit_family"] for key in selected_keys)),
        "by_probe_type": dict(Counter(key["probe_type"] for key in selected_keys)),
        "by_stress_variant": dict(Counter(key.get("stress_variant", "unknown") for key in selected_keys)),
    }
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "official_subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "DATASET_CARD.md").write_text(
        "# HABIT-Bench Official Full-Config Subset\n\n"
        "This stratified subset is intended for expensive full official-method "
        "runs on Lumia or another GPU/server environment.\n\n"
        "## Source And Domain Contract\n\n"
        "- Real prompt seed source: `allenai/WildChat`.\n"
        "- Domain assignment: keyword-filtered WildChat task seed buckets, with "
        "one representative domain per habit family.\n"
        "- Controlled synthetic components: hidden habit graphs, assistant "
        "feedback, counterfactual probes, answer choices, gold labels, and "
        "evidence links.\n"
        "- Accurate release claim: real-prompt-seeded, domain-grounded, "
        "synthetic longitudinal habit benchmark.\n"
        "- Claim to avoid: each habit family is drawn from a different external "
        "dataset.\n\n"
        "## Contents\n\n"
        f"- Source split: `{input_dir}`\n"
        f"- Probes: {manifest['counts']['probes']}\n"
        f"- Users: {manifest['counts']['users']}\n"
        f"- Sessions: {manifest['counts']['sessions']}\n"
        f"- Include variants: `{args.include_variants}`\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-probes", type=int, default=90)
    parser.add_argument("--min-per-capability", type=int, default=8)
    parser.add_argument(
        "--include-variants",
        choices=["all", "original_balanced", "unseen_paraphrase", "original_reviewed"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


if __name__ == "__main__":
    subset(parse_args())
