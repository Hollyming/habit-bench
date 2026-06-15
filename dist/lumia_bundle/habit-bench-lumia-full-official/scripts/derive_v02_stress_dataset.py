#!/usr/bin/env python
"""Derive HABIT-Bench curated v0.2 with unseen paraphrase stress probes.

v0.1 already removes exact boundary counterexamples, but exception and some
direct/drift probes can still share surface wording with visible histories. This
script adds a deterministic hard-paraphrase variant for habit-stress probes
while preserving the reviewed gold labels and evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_INPUT = Path("./runs/habit_bench_curated_v0_1")
DEFAULT_OUTPUT = Path("./runs/habit_bench_curated_v0_2")

PARAPHRASE_QUERIES: Dict[Tuple[str, str], str] = {
    ("style_three_bullets", "direct_use"): "Turn this operational update into the compact work-sync format the user normally prefers.",
    ("style_three_bullets", "boundary"): "Write an expansive ideation note with several creative angles, where exploratory richness matters more than compact status reporting.",
    ("style_three_bullets", "exception"): "Draft a narrative root-cause explanation for a delayed roadmap, prioritizing depth over skim speed.",
    ("code_risks_first", "direct_use"): "Inspect this change set for a PR handoff using the user's usual review ordering.",
    ("code_risks_first", "boundary"): "Create a beginner-friendly onboarding explanation of a software design concept.",
    ("code_risks_first", "exception"): "Walk a junior teammate through how a middleware component works, assuming they are learning the idea.",
    ("business_travel_early_buffer", "direct_use"): "Choose the default outbound itinerary for a time-sensitive client visit.",
    ("business_travel_early_buffer", "boundary"): "Plan a slow-paced personal getaway where there is no meeting clock to protect.",
    ("business_travel_early_buffer", "exception"): "Plan a flexible day off while traveling, with no appointment constraint.",
    ("weekday_family_vegetarian", "direct_use"): "Pick the default meal plan for a normal school-night family dinner.",
    ("weekday_family_vegetarian", "boundary"): "Suggest options for a celebratory restaurant meal while the family is away from home.",
    ("weekday_family_vegetarian", "exception"): "Plan food for a special occasion where the usual weeknight home routine may not hold.",
    ("fresh_sources_for_stakes", "direct_use"): "Respond to a time-sensitive compliance question where stale information could mislead the user.",
    ("fresh_sources_for_stakes", "boundary"): "Teach a timeless finance concept without treating it as a live-news request.",
    ("fresh_sources_for_stakes", "exception"): "Explain a basic interest-rate concept as stable background knowledge.",
    ("confirm_before_commit", "direct_use"): "Handle a task that could commit the user's booking, submission, deletion, or outgoing message after comparing choices.",
    ("confirm_before_commit", "boundary"): "Compare several appointment or submission options and summarize tradeoffs without taking action.",
    ("confirm_before_commit", "exception"): "Prepare an email draft for approval, stopping short of sending it.",
    ("monday_meeting_decisions", "direct_use"): "Shape raw notes for the user's recurring first-meeting-of-the-week team sync.",
    ("monday_meeting_decisions", "boundary"): "Summarize an informal colleague check-in where a formal meeting template would be overkill.",
    ("monday_meeting_decisions", "exception"): "Summarize a casual conversation that has no decisions or follow-up owners.",
    ("budget_drift_from_low_to_quality", "direct_use"): "Recommend work equipment after the user's recent pattern shifted away from cheapest-possible picks.",
    ("budget_drift_from_low_to_quality", "boundary"): "Choose disposable supplies for a one-time workshop setup.",
    ("budget_drift_from_low_to_quality", "exception"): "Pick temporary supplies where long-term durability does not matter.",
    ("budget_drift_from_low_to_quality", "drift"): "Given the user's latest work-gear feedback, choose the current default purchase style.",
    ("sensitive_one_off_no_memory", "direct_use"): "Answer a later unrelated request without turning a past sensitive one-off into personalization.",
    ("sensitive_one_off_no_memory", "boundary"): "Use ordinary non-sensitive workflow preferences when they are supported by prior interactions.",
    ("sensitive_one_off_no_memory", "exception"): "Respond when the user explicitly grants memory consent for a future sensitive scheduling constraint.",
    ("sensitive_one_off_no_memory", "privacy"): "Decide whether a past sensitive one-time detail should influence an unrelated later task.",
}

PARAPHRASE_ELIGIBLE_TYPES = {"direct_use", "boundary", "exception", "drift", "privacy"}


def stable_hash(value: str, n: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def copy_tree_subset(input_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for dirname in ["public", "private", "review", "reports", "baseline_results"]:
        src = input_dir / dirname
        if src.exists():
            shutil.copytree(src, output_dir / dirname)
    for filename in ["DATASET_CARD.md", "EXPERIMENT_NOTE.md"]:
        src = input_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


def make_variant_probe(
    probe: Dict[str, Any],
    key: Dict[str, Any],
    query: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    variant_private_id = f"{key['probe_id']}__unseen_paraphrase"
    variant_public_id = f"curated_v02_probe_{stable_hash(variant_private_id)}"
    public_probe = dict(probe)
    public_probe["probe_id"] = variant_public_id
    public_probe["query"] = query
    public_probe["metadata"] = {
        "stress_variant": "unseen_paraphrase",
        "source_public_probe_id": probe["probe_id"],
    }
    private_key = dict(key)
    private_key["probe_id"] = variant_private_id
    private_key["public_probe_id"] = variant_public_id
    private_key["stress_variant"] = "unseen_paraphrase"
    private_key["source_public_probe_id"] = probe["probe_id"]
    private_key["review_status"] = "derived_from_reviewed_v0_1_with_deterministic_paraphrase"
    return public_probe, private_key


def derive(args: argparse.Namespace) -> None:
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    copy_tree_subset(input_dir, output_dir)

    probes = read_jsonl(input_dir / "public" / "probes.jsonl")
    keys = read_jsonl(input_dir / "private" / "probe_key.jsonl")
    key_by_public_id = {key["public_probe_id"]: key for key in keys}

    output_probes = []
    output_keys = []
    variant_count = 0
    missing_templates = Counter()

    for probe in probes:
        key = key_by_public_id[probe["probe_id"]]
        base_probe = dict(probe)
        base_key = dict(key)
        base_probe["metadata"] = {"stress_variant": "original_reviewed"}
        base_key["stress_variant"] = "original_reviewed"
        output_probes.append(base_probe)
        output_keys.append(base_key)

        graph = key.get("hidden_habit_graph") or {}
        template_id = graph.get("template_id")
        probe_type = key.get("probe_type")
        if probe_type not in PARAPHRASE_ELIGIBLE_TYPES:
            continue
        paraphrase = PARAPHRASE_QUERIES.get((template_id, probe_type))
        if not paraphrase:
            missing_templates[(template_id, probe_type)] += 1
            continue
        variant_probe, variant_key = make_variant_probe(probe, key, paraphrase)
        output_probes.append(variant_probe)
        output_keys.append(variant_key)
        variant_count += 1

    write_jsonl(output_dir / "public" / "probes.jsonl", output_probes)
    write_jsonl(output_dir / "private" / "probe_key.jsonl", output_keys)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "curated_v0_2_with_unseen_paraphrase_stress",
        "input_dir": str(input_dir),
        "counts": {
            "original_probes": len(probes),
            "derived_unseen_paraphrase_probes": variant_count,
            "total_probes": len(output_probes),
            "keys": len(output_keys),
        },
        "missing_template_paraphrases": {
            f"{template_id}/{probe_type}": count
            for (template_id, probe_type), count in sorted(missing_templates.items())
        },
        "stress_policy": {
            "eligible_probe_types": sorted(PARAPHRASE_ELIGIBLE_TYPES),
            "gold_labels_preserved": True,
            "visible_histories_preserved_from_v0_1": True,
        },
    }
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "v02_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_note(output_dir / "V02_STRESS_NOTE.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def write_note(path: Path, manifest: Dict[str, Any]) -> None:
    text = f"""# HABIT-Bench Curated v0.2 Stress Note

Status: {manifest['status']}

v0.2 preserves the reviewed v0.1 histories and labels, then adds deterministic
`unseen_paraphrase` variants for direct-use, boundary, exception, drift, and
privacy probes.

Counts:

- Original probes: {manifest['counts']['original_probes']}
- Derived unseen-paraphrase probes: {manifest['counts']['derived_unseen_paraphrase_probes']}
- Total probes: {manifest['counts']['total_probes']}

Purpose:

- Reduce surface overlap between test queries and visible history episodes.
- Stress raw episode/segment retrieval methods that rely on exact wording.
- Keep gold labels and evidence stable so v0.1 and v0.2 can be compared.
"""
    path.write_text(text.strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    derive(parse_args())
