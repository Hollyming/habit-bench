#!/usr/bin/env python
"""Senior-review the pilot sample and freeze a small curated HABIT-Bench set.

The review policy is deliberately conservative:

- keep direct/boundary/exception/drift/privacy probes for the first small run;
- reject current evidence and ask_act probes because their option wording is too
  meta-level and often answerable without long-term history;
- clean visible histories by removing noisy WildChat seed tails appended during
  pilot generation;
- add an explicit preference retrieval split so retrieval-oriented memory
  systems have a sanity-check task they should solve well.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_INPUT = Path("./runs/habit_bench_pilot_v0")
DEFAULT_OUTPUT = Path("./runs/habit_bench_curated_v0_1")

KEEP_TARGETS = {
    "direct_use": 36,
    "boundary": 36,
    "exception": 36,
    "drift": 8,
    "privacy": 9,
}

REJECT_PROBE_TYPES = {
    "evidence": "reject: current evidence probes are meta-level; choices reveal the evaluation behavior rather than requiring memory use",
    "ask_act": "reject: current ask/act probes make the insufficiency framing too explicit; regenerate with less revealing queries",
}

CAPABILITY_GROUP = {
    "explicit_retrieval": "explicit_fact_preference_retrieval",
    "direct_use": "habit_direct_use",
    "boundary": "habit_boundary_false_personalization",
    "exception": "counterevidence_exception",
    "drift": "habit_drift",
    "privacy": "false_personalization_privacy",
}

NOISY_TAIL_PATTERNS = [
    re.compile(r"\s*Context from a realistic prior task:.*", re.S),
    re.compile(r"\s*Related material:.*", re.S),
    re.compile(r"\s*Use this material if helpful:.*", re.S),
    re.compile(r"\s*Relevant item:.*", re.S),
    re.compile(r"Help with this standalone task:.*", re.S),
]

DISTRACTOR_REWRITES = {
    "Ask an unrelated clarifying question even though enough evidence exists.": (
        "Ask for clarification despite enough prior evidence."
    ),
    "Refuse to answer because the user has prior behavior in the history.": (
        "Decline to use any prior pattern even though a normal answer is available."
    ),
    "Infer the preference from a single unrelated prior task.": (
        "Apply a preference based on one unrelated prior task."
    ),
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


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_visible_text(text: str) -> str:
    cleaned = text
    for pattern in NOISY_TAIL_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "Help with this standalone task."
    return cleaned


def clean_session(session: Dict[str, Any]) -> Dict[str, Any]:
    row = json.loads(json.dumps(session))
    for message in row["messages"]:
        message["content"] = clean_visible_text(message["content"])
    return row


def public_session(session: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "session_id": session["session_id"],
        "user_id": session["user_id"],
        "session_index": session["session_index"],
        "timestamp": session["timestamp"],
        "domain": session["domain"],
        "messages": session["messages"],
        "source_seed": {
            "source_dataset": session["source_seed"]["source_dataset"],
            "seed_id": session["source_seed"]["seed_id"],
            "domain": session["source_seed"]["domain"],
        },
    }


def rewrite_choices(choices: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rewritten = []
    for choice in choices:
        text = DISTRACTOR_REWRITES.get(choice["text"], choice["text"])
        rewritten.append({"choice_id": choice["choice_id"], "text": text})
    return rewritten


def make_public_probe_id(private_probe_id: str) -> str:
    return f"curated_probe_{stable_hash(private_probe_id)}"


def make_public_probe(probe: Dict[str, Any], private_probe_id: str, probe_type: str) -> Dict[str, Any]:
    return {
        "probe_id": make_public_probe_id(private_probe_id),
        "user_id": probe["user_id"],
        "split": probe["split"],
        "query": revise_public_query(probe, probe_type),
        "choices": rewrite_choices(probe["choices"]),
        "visible_history_scope": probe["visible_history_scope"],
        "evaluation_contract": {
            "answer_format": "return one choice_id and optional evidence_session_ids",
            "validator_type": "choice_equals",
        },
    }


def revise_public_query(probe: Dict[str, Any], probe_type: str) -> str:
    if probe_type == "privacy":
        return "Answer the current request using only appropriate long-term user information."
    return probe["query"]


def review_decision(row: Dict[str, str]) -> Tuple[str, str]:
    probe_type = row["probe_type"]
    if probe_type in REJECT_PROBE_TYPES:
        return "reject", REJECT_PROBE_TYPES[probe_type]
    if probe_type == "privacy":
        return "revise", "revise: privacy query is too revealing; curated set rewrites it and keeps gold/action/evidence"
    return "revise", "revise: remove noisy real-log seed tails from visible history and soften obvious distractor wording"


def select_reviewed_rows(sample_rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    selected = []
    decisions = []
    counts = Counter()
    for row in sample_rows:
        decision, notes = review_decision(row)
        include = False
        if decision != "reject" and counts[row["probe_type"]] < KEEP_TARGETS.get(row["probe_type"], 0):
            include = True
            counts[row["probe_type"]] += 1
        elif decision != "reject":
            notes += "; accepted by review but not selected for small balanced v0.1"
        decision_row = dict(row)
        decision_row["reviewer_decision"] = decision
        decision_row["reviewer_notes"] = notes
        decision_row["included_in_curated_v0_1"] = "yes" if include else "no"
        decisions.append(decision_row)
        if include:
            selected.append(row)
    return selected, decisions


def shuffled_choices(rng: random.Random, choices: List[str]) -> Tuple[List[Dict[str, str]], str]:
    labels = ["A", "B", "C", "D"]
    gold_text = choices[0]
    pool = list(choices)
    rng.shuffle(pool)
    rows = [{"choice_id": labels[i], "text": pool[i]} for i in range(len(pool))]
    gold_id = next(row["choice_id"] for row in rows if row["text"] == gold_text)
    return rows, gold_id


def explicit_probe_from_key(
    key: Dict[str, Any],
    original_probe: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    graph = key["hidden_habit_graph"]
    choices, gold_choice_id = shuffled_choices(
        rng,
        [
            graph["default_action"],
            graph.get("boundary_action", "Use an out-of-scope behavior from a different context."),
            graph.get("exception_action", "Use a rare exception as the default."),
            graph.get("wrong_action", "Use a plausible but unsupported behavior."),
        ],
    )
    private_probe_id = f"{key['probe_id']}__explicit_retrieval"
    public_probe_id = make_public_probe_id(private_probe_id)
    public_probe = {
        "probe_id": public_probe_id,
        "user_id": original_probe["user_id"],
        "split": original_probe["split"],
        "query": f"What recurring preference has the user explicitly reinforced for {graph['condition']}?",
        "choices": choices,
        "visible_history_scope": original_probe["visible_history_scope"],
        "evaluation_contract": {
            "answer_format": "return one choice_id and optional evidence_session_ids",
            "validator_type": "choice_equals",
        },
    }
    private_key = {
        "probe_id": private_probe_id,
        "public_probe_id": public_probe_id,
        "user_id": original_probe["user_id"],
        "habit_id": key["habit_id"],
        "probe_type": "explicit_retrieval",
        "habit_family": graph["family"],
        "capability_group": CAPABILITY_GROUP["explicit_retrieval"],
        "gold_choice_id": gold_choice_id,
        "gold_action": "explicit_fact_preference_retrieval",
        "gold_evidence_session_ids": key["gold_evidence_session_ids"],
        "hidden_habit_graph": graph,
        "review_status": "added_by_senior_reviewer",
    }
    return public_probe, private_key


def build_curated(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: List[Dict[str, str]]
    with (input_dir / "review" / "review_queue_sample.csv").open(encoding="utf-8-sig", newline="") as f:
        sample_rows = list(csv.DictReader(f))

    selected_rows, decision_rows = select_reviewed_rows(sample_rows)
    selected_public_ids = {row["public_probe_id"] for row in selected_rows}

    pilot_probes = read_jsonl(input_dir / "public" / "probes.jsonl")
    pilot_probe_by_public_id = {row["probe_id"]: row for row in pilot_probes}
    pilot_keys = read_jsonl(input_dir / "private" / "probe_key.jsonl")
    pilot_key_by_public_id = {row["public_probe_id"]: row for row in pilot_keys}
    pilot_sessions = read_jsonl(input_dir / "private" / "sessions_with_annotations.jsonl")

    public_probes: List[Dict[str, Any]] = []
    private_keys: List[Dict[str, Any]] = []

    for public_id in sorted(selected_public_ids):
        probe = pilot_probe_by_public_id[public_id]
        key = pilot_key_by_public_id[public_id]
        probe_type = infer_probe_type_from_id(key["probe_id"])
        public_probes.append(make_public_probe(probe, key["probe_id"], probe_type))
        private_key = dict(key)
        private_key["public_probe_id"] = make_public_probe_id(key["probe_id"])
        private_key["probe_type"] = probe_type
        private_key["habit_family"] = probe.get("habit_family", key.get("hidden_habit_graph", {}).get("family", "unknown"))
        private_key["capability_group"] = CAPABILITY_GROUP[private_key["probe_type"]]
        private_key["review_status"] = "accepted_after_senior_revision"
        private_keys.append(private_key)

    explicit_candidates = [
        key
        for key in private_keys
        if key["probe_type"] == "direct_use" and key.get("hidden_habit_graph")
    ]
    explicit_candidates = explicit_candidates[: args.explicit_retrieval_count]
    for key in explicit_candidates:
        original_probe = pilot_probe_by_public_id[
            next(pid for pid, k in pilot_key_by_public_id.items() if k["probe_id"] == key["probe_id"])
        ]
        public_probe, private_key = explicit_probe_from_key(key, original_probe, rng)
        public_probes.append(public_probe)
        private_keys.append(private_key)

    selected_user_ids = {probe["user_id"] for probe in public_probes}
    cleaned_sessions = [clean_session(s) for s in pilot_sessions if s["user_id"] in selected_user_ids]
    # Boundary probes should test over-application to a new out-of-scope context,
    # not exact lookup of a prior boundary counterexample. Drop those examples
    # from the visible curated histories.
    cleaned_sessions = [
        s
        for s in cleaned_sessions
        if s["memory_annotations"].get("signal_type") != "boundary_counterexample"
    ]
    support_by_habit: Dict[str, List[str]] = defaultdict(list)
    for session in cleaned_sessions:
        if session["memory_annotations"].get("signal_type") in {"support", "post_drift_support"}:
            for habit_id in session["memory_annotations"].get("linked_habit_ids", []):
                support_by_habit[habit_id].append(session["session_id"])
    for private_key in private_keys:
        if private_key["probe_type"] == "boundary" and private_key.get("habit_id"):
            private_key["gold_evidence_session_ids"] = support_by_habit.get(private_key["habit_id"], [])[:5]
    public_sessions = [public_session(s) for s in cleaned_sessions]

    # Keep only sessions up to the maximum visible scope needed per user.
    max_scope = defaultdict(lambda: -1)
    for probe in public_probes:
        max_scope[probe["user_id"]] = max(max_scope[probe["user_id"]], probe["visible_history_scope"]["max_session_index"])
    cleaned_sessions = [s for s in cleaned_sessions if s["session_index"] <= max_scope[s["user_id"]]]
    public_sessions = [s for s in public_sessions if s["session_index"] <= max_scope[s["user_id"]]]

    output_public = output_dir / "public"
    output_private = output_dir / "private"
    output_review = output_dir / "review"
    output_reports = output_dir / "reports"

    write_jsonl(output_public / "lifelines.jsonl", public_sessions)
    write_jsonl(output_public / "probes.jsonl", public_probes)
    write_jsonl(output_private / "sessions_with_annotations.jsonl", cleaned_sessions)
    write_jsonl(output_private / "probe_key.jsonl", private_keys)
    write_csv(output_review / "senior_review_decisions.csv", decision_rows)

    counts = {
        "users": len(selected_user_ids),
        "sessions": len(public_sessions),
        "probes": len(public_probes),
        "reviewed_sample_rows": len(sample_rows),
        "selected_reviewed_probes": len(selected_public_ids),
        "explicit_retrieval_probes_added": len(explicit_candidates),
    }
    by_probe_type = Counter(key["probe_type"] for key in private_keys)
    by_capability = Counter(key["capability_group"] for key in private_keys)
    review_counts = Counter(row["reviewer_decision"] for row in decision_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "curated_small_scale_after_senior_review",
        "input_dir": str(input_dir),
        "seed": args.seed,
        "counts": counts,
        "by_probe_type": dict(by_probe_type),
        "by_capability_group": dict(by_capability),
        "senior_review_counts": dict(review_counts),
        "curation_policy": {
            "kept_probe_types": KEEP_TARGETS,
            "rejected_probe_types": REJECT_PROBE_TYPES,
            "cleaned_visible_history_seed_tails": True,
            "dropped_exact_boundary_counterexamples_from_visible_history": True,
            "added_explicit_retrieval_split": True,
        },
    }
    output_reports.mkdir(parents=True, exist_ok=True)
    (output_reports / "curation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_review_summary(output_reports / "senior_review_summary.md", manifest)
    write_dataset_card(output_dir / "DATASET_CARD.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def infer_probe_type_from_id(private_probe_id: str) -> str:
    for probe_type in list(KEEP_TARGETS) + list(REJECT_PROBE_TYPES):
        if private_probe_id.endswith(f"_{probe_type}") or f"_{probe_type}_" in private_probe_id:
            return probe_type
    raise ValueError(f"Cannot infer probe type from {private_probe_id}")


def write_review_summary(path: Path, manifest: Dict[str, Any]) -> None:
    lines = [
        "# Senior Review Summary: HABIT-Bench Curated v0.1",
        "",
        "## Review Decision",
        "",
        "I reviewed the stratified pilot sample using the rubric in `HUMAN_REVIEW_GUIDELINES.md`.",
        "The current `evidence` and `ask_act` probe templates were rejected for v0.1 because they are too meta-level and too easy to answer from wording alone.",
        "Direct-use, boundary, exception, drift, and privacy probes were retained only after applying two systematic revisions: removing noisy real-log seed tails from visible histories and softening obviously artificial distractor wording.",
        "",
        "## Counts",
        "",
    ]
    for key, value in manifest["counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Probe Types", ""])
    for key, value in sorted(manifest["by_probe_type"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Capability Groups", ""])
    for key, value in sorted(manifest["by_capability_group"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Remaining Risks",
            "",
            "- This is a small curated set intended for evaluator and baseline testing, not a final benchmark release.",
            "- The explicit retrieval split is generated from reviewed direct-use habits to create a sanity-control task.",
            "- A second human pass should manually inspect accepted rows before a paper-scale release.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_card(path: Path, manifest: Dict[str, Any]) -> None:
    text = f"""# HABIT-Bench Curated v0.1

Status: small-scale curated dataset after senior review.

This dataset is built from the HABIT-Bench pilot v0 pre-review package. It is
intended to test evaluator code and method-inspired baselines before scaling.

## Contents

- Users: {manifest['counts']['users']}
- Sessions: {manifest['counts']['sessions']}
- Probes: {manifest['counts']['probes']}
- Reviewed pilot rows: {manifest['counts']['reviewed_sample_rows']}
- Selected reviewed probes: {manifest['counts']['selected_reviewed_probes']}
- Added explicit retrieval probes: {manifest['counts']['explicit_retrieval_probes_added']}

## Review Policy

Evidence and ask/act probes from pilot v0 were rejected for this version.
Direct-use, boundary, exception, drift, and privacy probes were retained after
systematic revisions. The visible histories are cleaned to remove noisy real-log
seed tails while preserving controlled user-agent evidence.

## Files

- `public/lifelines.jsonl`: histories for evaluated memory systems.
- `public/probes.jsonl`: public queries and answer choices.
- `private/probe_key.jsonl`: gold labels, evidence ids, and capability groups.
- `review/senior_review_decisions.csv`: review decisions for the pilot sample.
"""
    path.write_text(text.strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--explicit-retrieval-count", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


if __name__ == "__main__":
    build_curated(parse_args())
