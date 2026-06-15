#!/usr/bin/env python
"""Build HABIT-Bench v0.3 as a larger balanced candidate split.

v0.3 is intentionally not a replacement for the reviewed v0.2 set. It expands
from the auto-validated pilot pool into a balanced 9-family candidate for
scaling experiments, while preserving an explicit "requires human audit" status
in the manifest and review queue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from curate_reviewed_dataset import (
    DISTRACTOR_REWRITES,
    clean_session,
    explicit_probe_from_key,
    infer_probe_type_from_id,
    public_session,
    rewrite_choices,
)
from derive_v02_stress_dataset import PARAPHRASE_QUERIES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs" / "habit_bench_pilot_v0"
DEFAULT_OUTPUT = ROOT / "runs" / "habit_bench_balanced_v0_3"

CORE_PROBE_TYPES = ["direct_use", "boundary", "exception"]
SPECIAL_PROBE_TYPES = {
    "drift_seasonality": ["drift"],
    "privacy_consent": ["privacy"],
}
CAPABILITY_GROUP = {
    "explicit_retrieval": "explicit_fact_preference_retrieval",
    "direct_use": "habit_direct_use",
    "boundary": "habit_boundary_false_personalization",
    "exception": "counterevidence_exception",
    "drift": "habit_drift",
    "privacy": "false_personalization_privacy",
}


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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(output_dir: Path) -> None:
    runs_root = (ROOT / "runs").resolve()
    target = output_dir.resolve()
    if not str(target).lower().startswith(str(runs_root).lower()):
        raise ValueError(f"Refusing to delete output outside runs directory: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def make_balanced_public_id(private_probe_id: str) -> str:
    return f"balanced_v03_probe_{stable_hash(private_probe_id)}"


def make_public_probe(
    pilot_probe: Dict[str, Any],
    private_probe_id: str,
    probe_type: str,
    stress_variant: str,
    source_public_probe_id: str | None = None,
) -> Dict[str, Any]:
    query = pilot_probe["query"]
    if probe_type == "privacy":
        query = "Answer the current request using only appropriate long-term user information."
    probe = {
        "probe_id": make_balanced_public_id(private_probe_id),
        "user_id": pilot_probe["user_id"],
        "split": pilot_probe["split"],
        "query": query,
        "choices": rewrite_choices(pilot_probe["choices"]),
        "visible_history_scope": pilot_probe["visible_history_scope"],
        "evaluation_contract": {
            "answer_format": "return one choice_id and optional evidence_session_ids",
            "validator_type": "choice_equals",
        },
        "metadata": {
            "stress_variant": stress_variant,
            "source_public_probe_id": source_public_probe_id,
        }
        if source_public_probe_id
        else {"stress_variant": stress_variant},
    }
    return probe


def make_private_key(
    pilot_key: Dict[str, Any],
    private_probe_id: str,
    public_probe_id: str,
    probe_type: str,
    evidence_ids: Sequence[str],
    stress_variant: str,
    review_status: str,
    source_public_probe_id: str | None = None,
) -> Dict[str, Any]:
    graph = pilot_key.get("hidden_habit_graph") or {}
    return {
        **pilot_key,
        "probe_id": private_probe_id,
        "public_probe_id": public_probe_id,
        "probe_type": probe_type,
        "habit_family": graph.get("family", pilot_key.get("habit_family", "unknown")),
        "capability_group": CAPABILITY_GROUP[probe_type],
        "gold_evidence_session_ids": list(evidence_ids),
        "stress_variant": stress_variant,
        "source_public_probe_id": source_public_probe_id,
        "review_status": review_status,
    }


def variant_query_for(key: Dict[str, Any], probe_type: str) -> str | None:
    graph = key.get("hidden_habit_graph") or {}
    return PARAPHRASE_QUERIES.get((graph.get("template_id"), probe_type))


def make_variant_probe(
    public_probe: Dict[str, Any],
    private_key: Dict[str, Any],
    paraphrase_query: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    variant_private_id = f"{private_key['probe_id']}__unseen_paraphrase"
    variant_public_id = make_balanced_public_id(variant_private_id)
    variant_probe = dict(public_probe)
    variant_probe["probe_id"] = variant_public_id
    variant_probe["query"] = paraphrase_query
    variant_probe["metadata"] = {
        "stress_variant": "unseen_paraphrase",
        "source_public_probe_id": public_probe["probe_id"],
    }
    variant_key = dict(private_key)
    variant_key["probe_id"] = variant_private_id
    variant_key["public_probe_id"] = variant_public_id
    variant_key["stress_variant"] = "unseen_paraphrase"
    variant_key["source_public_probe_id"] = public_probe["probe_id"]
    variant_key["review_status"] = "derived_from_balanced_candidate_with_deterministic_paraphrase"
    return variant_probe, variant_key


def group_pilot_by_habit(keys: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        habit_id = key.get("habit_id")
        graph = key.get("hidden_habit_graph")
        if not habit_id or not graph:
            continue
        row = grouped.setdefault(
            habit_id,
            {
                "habit_id": habit_id,
                "family": graph["family"],
                "graph": graph,
                "probes": {},
            },
        )
        probe_type = infer_probe_type_from_id(key["probe_id"])
        row["probes"][probe_type] = key
    return grouped


def select_habits(
    habits_by_id: Dict[str, Dict[str, Any]],
    target_per_family: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for habit in habits_by_id.values():
        probes = habit["probes"]
        if not all(probe_type in probes for probe_type in CORE_PROBE_TYPES):
            continue
        by_family[habit["family"]].append(habit)

    selected = []
    missing = {}
    for family, rows in sorted(by_family.items()):
        rows = sorted(rows, key=lambda row: row["habit_id"])
        if len(rows) < target_per_family:
            missing[family] = len(rows)
        selected.extend(rng.sample(rows, min(target_per_family, len(rows))))
    if missing:
        raise ValueError(f"Not enough habits for target_per_family={target_per_family}: {missing}")
    return sorted(selected, key=lambda row: (row["family"], row["habit_id"]))


def support_evidence_by_habit(sessions: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    support: Dict[str, List[str]] = defaultdict(list)
    for session in sessions:
        signal_type = session["memory_annotations"].get("signal_type")
        if signal_type not in {"support", "post_drift_support"}:
            continue
        for habit_id in session["memory_annotations"].get("linked_habit_ids", []):
            support[habit_id].append(session["session_id"])
    return support


def clean_public_probe_choices(probe: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(probe)
    row["choices"] = [
        {
            "choice_id": choice["choice_id"],
            "text": DISTRACTOR_REWRITES.get(choice["text"], choice["text"]),
        }
        for choice in probe["choices"]
    ]
    return row


def add_selected_probe(
    output_probes: List[Dict[str, Any]],
    output_keys: List[Dict[str, Any]],
    pilot_probe_by_public_id: Dict[str, Dict[str, Any]],
    pilot_key: Dict[str, Any],
    probe_type: str,
    evidence_ids: Sequence[str],
) -> None:
    private_probe_id = f"{pilot_key['probe_id']}__balanced_v03"
    pilot_probe = pilot_probe_by_public_id[pilot_key["public_probe_id"]]
    public_probe = make_public_probe(
        pilot_probe,
        private_probe_id,
        probe_type,
        stress_variant="original_balanced",
    )
    private_key = make_private_key(
        pilot_key,
        private_probe_id,
        public_probe["probe_id"],
        probe_type,
        evidence_ids,
        stress_variant="original_balanced",
        review_status="balanced_candidate_needs_human_review",
    )
    output_probes.append(public_probe)
    output_keys.append(private_key)

    paraphrase = variant_query_for(private_key, probe_type)
    if paraphrase:
        variant_probe, variant_key = make_variant_probe(public_probe, private_key, paraphrase)
        output_probes.append(variant_probe)
        output_keys.append(variant_key)


def add_explicit_probe(
    output_probes: List[Dict[str, Any]],
    output_keys: List[Dict[str, Any]],
    pilot_probe_by_public_id: Dict[str, Dict[str, Any]],
    direct_key: Dict[str, Any],
    rng: random.Random,
) -> None:
    original_probe = pilot_probe_by_public_id[direct_key["public_probe_id"]]
    public_probe, private_key = explicit_probe_from_key(direct_key, original_probe, rng)
    private_id = f"{private_key['probe_id']}__balanced_v03"
    public_probe["probe_id"] = make_balanced_public_id(private_id)
    public_probe["choices"] = rewrite_choices(public_probe["choices"])
    public_probe["metadata"] = {"stress_variant": "original_balanced"}
    private_key["probe_id"] = private_id
    private_key["public_probe_id"] = public_probe["probe_id"]
    private_key["stress_variant"] = "original_balanced"
    private_key["review_status"] = "balanced_candidate_explicit_control_needs_human_review"
    private_key["source_public_probe_id"] = direct_key["public_probe_id"]
    output_probes.append(public_probe)
    output_keys.append(private_key)


def evidence_preview(
    evidence_ids: Sequence[str],
    session_by_id: Dict[str, Dict[str, Any]],
    max_items: int = 4,
) -> List[Dict[str, Any]]:
    rows = []
    for sid in list(evidence_ids)[:max_items]:
        session = session_by_id.get(sid)
        if not session:
            continue
        user_msg = next((m["content"] for m in session["messages"] if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in session["messages"] if m["role"] == "assistant"), "")
        rows.append(
            {
                "session_id": sid,
                "session_index": session["session_index"],
                "signal_type": session["memory_annotations"].get("signal_type"),
                "domain": session.get("domain"),
                "user": re.sub(r"\s+", " ", user_msg).strip()[:220],
                "assistant": re.sub(r"\s+", " ", assistant_msg).strip()[:180],
            }
        )
    return rows


def write_review_queue(
    path_csv: Path,
    path_jsonl: Path,
    public_probes: Sequence[Dict[str, Any]],
    private_keys: Sequence[Dict[str, Any]],
    session_by_id: Dict[str, Dict[str, Any]],
    sample_rate: float,
    rng: random.Random,
) -> None:
    public_by_id = {probe["probe_id"]: probe for probe in public_probes}
    rows = []
    for key in private_keys:
        probe = public_by_id[key["public_probe_id"]]
        rows.append(
            {
                "review_id": f"review_{key['probe_id']}",
                "public_probe_id": key["public_probe_id"],
                "probe_id": key["probe_id"],
                "user_id": key["user_id"],
                "split": probe["split"],
                "probe_type": key["probe_type"],
                "habit_family": key["habit_family"],
                "stress_variant": key.get("stress_variant", "unknown"),
                "query": probe["query"],
                "choices_json": json.dumps(probe["choices"], ensure_ascii=False),
                "proposed_gold_choice_id": key["gold_choice_id"],
                "proposed_gold_action": key["gold_action"],
                "evidence_preview_json": json.dumps(
                    evidence_preview(key.get("gold_evidence_session_ids", []), session_by_id),
                    ensure_ascii=False,
                ),
                "auto_validation_status": "pass",
                "reviewer_decision": "",
                "reviewer_notes": "",
            }
        )

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["probe_type"], row["habit_family"], row["stress_variant"])].append(row)
    sample = []
    for bucket in grouped.values():
        k = max(1, round(len(bucket) * sample_rate))
        sample.extend(rng.sample(bucket, min(k, len(bucket))))

    rows.sort(key=lambda row: row["review_id"])
    sample.sort(key=lambda row: row["review_id"])
    write_csv(path_csv.parent / "balanced_review_queue_all.csv", rows)
    write_csv(path_csv, sample)
    write_jsonl(path_jsonl.parent / "balanced_review_queue_all.jsonl", rows)
    write_jsonl(path_jsonl, sample)


def validate_dataset(
    public_probes: Sequence[Dict[str, Any]],
    private_keys: Sequence[Dict[str, Any]],
    public_sessions: Sequence[Dict[str, Any]],
    private_sessions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    public_ids = [probe["probe_id"] for probe in public_probes]
    key_public_ids = [key["public_probe_id"] for key in private_keys]
    session_ids = {session["session_id"] for session in private_sessions}
    errors = []
    if len(public_ids) != len(set(public_ids)):
        errors.append("duplicate_public_probe_id")
    if set(public_ids) != set(key_public_ids):
        errors.append("public_probe_key_mismatch")
    for probe in public_probes:
        choice_ids = {choice["choice_id"] for choice in probe["choices"]}
        key = next(k for k in private_keys if k["public_probe_id"] == probe["probe_id"])
        if key["gold_choice_id"] not in choice_ids:
            errors.append(f"gold_choice_missing:{probe['probe_id']}")
    for key in private_keys:
        missing = [sid for sid in key.get("gold_evidence_session_ids", []) if sid not in session_ids]
        if missing:
            errors.append(f"missing_evidence:{key['probe_id']}:{missing[:3]}")
    if len(public_sessions) != len(private_sessions):
        errors.append("public_private_session_count_mismatch")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors[:20],
        "public_probes": len(public_probes),
        "private_keys": len(private_keys),
        "public_sessions": len(public_sessions),
        "private_sessions": len(private_sessions),
    }


def write_dataset_card(path: Path, manifest: Dict[str, Any]) -> None:
    text = f"""# HABIT-Bench Balanced v0.3 Candidate

Status: larger balanced candidate split; automatic validation passed; human
audit still required before paper-scale claims.

## Source

- Real prompt seed source: `allenai/WildChat`.
- Domain assignment: keyword-filtered WildChat task seed buckets, with one
  representative domain per habit family.
- Controlled synthetic components: hidden habit graphs, assistant feedback,
  counterfactual probes, answer choices, gold labels, and evidence links.
- Accurate release claim: real-prompt-seeded, domain-grounded, synthetic
  longitudinal habit benchmark.
- Claim to avoid: each habit family is drawn from a different external dataset.
- Input pool: `{manifest['input_dir']}`.

## Contents

- Users: {manifest['counts']['users']}
- Sessions: {manifest['counts']['sessions']}
- Probes: {manifest['counts']['probes']}
- Selected habits per family: {manifest['selection_policy']['target_habits_per_family']}
- Stress variants: original balanced + deterministic unseen paraphrase for
  habit-stress probes.

## Important Boundary

This split is larger and more balanced than v0.2, but most rows are selected
from the auto-validated pilot pool rather than the senior-reviewed v0.2 set.
Use it for scaling experiments and review planning. Treat v0.2 as the stronger
reviewed evidence set until v0.3 receives human audit.

## Files

- `public/lifelines.jsonl`: histories for evaluated memory systems.
- `public/probes.jsonl`: public queries and answer choices.
- `private/probe_key.jsonl`: gold labels, evidence ids, and hidden habit graphs.
- `reports/balanced_v03_manifest.json`: counts, balance, validation status.
- `review/balanced_review_queue_sample.csv`: stratified human audit sample.
"""
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_summary(path: Path, manifest: Dict[str, Any]) -> None:
    lines = [
        "# HABIT-Bench Balanced v0.3 Candidate Summary",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Status: {manifest['status']}",
        f"- Users: {manifest['counts']['users']}",
        f"- Sessions: {manifest['counts']['sessions']}",
        f"- Probes: {manifest['counts']['probes']}",
        f"- Target habits per family: {manifest['selection_policy']['target_habits_per_family']}",
        "",
        "## Probe Counts",
        "",
    ]
    for key, value in sorted(manifest["by_probe_type"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Habit Family Counts", ""])
    for key, value in sorted(manifest["by_habit_family"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Stress Variants", ""])
    for key, value in sorted(manifest["by_stress_variant"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Human Audit Status",
            "",
            "Human audit has not been completed for this larger split. The review",
            "queue in `review/` is the next handoff point.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    prepare_output_dir(output_dir)

    pilot_probes = read_jsonl(input_dir / "public" / "probes.jsonl")
    pilot_probe_by_public_id = {probe["probe_id"]: probe for probe in pilot_probes}
    pilot_keys = read_jsonl(input_dir / "private" / "probe_key.jsonl")
    pilot_sessions = read_jsonl(input_dir / "private" / "sessions_with_annotations.jsonl")
    cleaned_sessions_all = [clean_session(session) for session in pilot_sessions]
    support_by_habit = support_evidence_by_habit(cleaned_sessions_all)

    habits = group_pilot_by_habit(pilot_keys)
    selected_habits = select_habits(habits, args.target_habits_per_family, rng)

    output_probes: List[Dict[str, Any]] = []
    output_keys: List[Dict[str, Any]] = []
    for habit in selected_habits:
        probes = habit["probes"]
        for probe_type in CORE_PROBE_TYPES:
            key = probes[probe_type]
            evidence = (
                support_by_habit.get(habit["habit_id"], [])[:5]
                if probe_type == "boundary"
                else key["gold_evidence_session_ids"]
            )
            add_selected_probe(
                output_probes,
                output_keys,
                pilot_probe_by_public_id,
                key,
                probe_type,
                evidence,
            )
        for probe_type in SPECIAL_PROBE_TYPES.get(habit["family"], []):
            if probe_type in probes:
                key = probes[probe_type]
                add_selected_probe(
                    output_probes,
                    output_keys,
                    pilot_probe_by_public_id,
                    key,
                    probe_type,
                    key["gold_evidence_session_ids"],
                )
        add_explicit_probe(
            output_probes,
            output_keys,
            pilot_probe_by_public_id,
            probes["direct_use"],
            rng,
        )

    selected_user_ids = {probe["user_id"] for probe in output_probes}
    max_scope = defaultdict(lambda: -1)
    for probe in output_probes:
        scope = probe["visible_history_scope"]["max_session_index"]
        max_scope[probe["user_id"]] = max(max_scope[probe["user_id"]], scope)

    private_sessions = [
        session
        for session in cleaned_sessions_all
        if session["user_id"] in selected_user_ids
        and session["session_index"] <= max_scope[session["user_id"]]
        and session["memory_annotations"].get("signal_type") != "boundary_counterexample"
    ]
    public_sessions = [public_session(session) for session in private_sessions]
    session_by_id = {session["session_id"]: session for session in private_sessions}

    validation = validate_dataset(output_probes, output_keys, public_sessions, private_sessions)
    if validation["status"] != "pass":
        raise SystemExit(json.dumps(validation, ensure_ascii=False, indent=2))

    output_public = output_dir / "public"
    output_private = output_dir / "private"
    output_reports = output_dir / "reports"
    output_review = output_dir / "review"
    output_public.mkdir(parents=True, exist_ok=True)
    output_private.mkdir(parents=True, exist_ok=True)
    output_reports.mkdir(parents=True, exist_ok=True)
    output_review.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_public / "lifelines.jsonl", public_sessions)
    write_jsonl(output_public / "probes.jsonl", output_probes)
    write_jsonl(output_private / "sessions_with_annotations.jsonl", private_sessions)
    write_jsonl(output_private / "probe_key.jsonl", output_keys)
    write_review_queue(
        output_review / "balanced_review_queue_sample.csv",
        output_review / "balanced_review_queue_sample.jsonl",
        output_probes,
        output_keys,
        session_by_id,
        args.review_sample_rate,
        rng,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "balanced_candidate_auto_validated_pending_human_audit",
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
        "selection_policy": {
            "target_habits_per_family": args.target_habits_per_family,
            "core_probe_types": CORE_PROBE_TYPES,
            "special_probe_types": SPECIAL_PROBE_TYPES,
            "explicit_retrieval_control_per_selected_habit": True,
            "drop_boundary_counterexample_sessions_from_visible_history": True,
            "unseen_paraphrase_for_stress_probes": True,
            "human_audit_required": True,
        },
        "counts": {
            "users": len(selected_user_ids),
            "sessions": len(public_sessions),
            "selected_habits": len(selected_habits),
            "probes": len(output_probes),
            "review_queue_all": len(output_keys),
            "review_queue_sample": sum(1 for _ in (output_review / "balanced_review_queue_sample.csv").open(encoding="utf-8-sig")) - 1,
        },
        "by_probe_type": dict(Counter(key["probe_type"] for key in output_keys)),
        "by_habit_family": dict(Counter(key["habit_family"] for key in output_keys)),
        "by_capability_group": dict(Counter(key["capability_group"] for key in output_keys)),
        "by_stress_variant": dict(Counter(key.get("stress_variant", "unknown") for key in output_keys)),
        "validation": validation,
    }
    (output_reports / "balanced_v03_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(output_reports / "balanced_v03_summary.md", manifest)
    write_dataset_card(output_dir / "DATASET_CARD.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-habits-per-family", type=int, default=30)
    parser.add_argument("--review-sample-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
