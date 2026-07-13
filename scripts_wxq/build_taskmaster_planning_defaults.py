#!/usr/bin/env python
"""Build a HABIT-Bench planning_defaults slice from Taskmaster-2 travel data.

The slice is intentionally scoped to one HABIT-Bench family:

    planning_defaults -> travel -> business travel prefers early arrivals
    and a buffer before meetings.

Only Taskmaster-2 `flights` and `hotels` dialogs are used as real task seeds.
The long-range habit signals, feedback, probes, labels, and evidence links are
controlled synthetic components, matching the HABIT-Bench construction pattern.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TASKMASTER_DATASET = "google-research-datasets/Taskmaster-2"
SOURCE_URLS = {
    "flights": [
        "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-2-2020/data/flights.json",
        "https://cdn.jsdelivr.net/gh/google-research-datasets/Taskmaster@master/TM-2-2020/data/flights.json",
    ],
    "hotels": [
        "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-2-2020/data/hotels.json",
        "https://cdn.jsdelivr.net/gh/google-research-datasets/Taskmaster@master/TM-2-2020/data/hotels.json",
    ],
}

PII_PATTERNS = [
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"\b(?:\+?\d[\d .()\-]{7,}\d)\b"),
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"https?://\S+", re.I),
]

TRAVEL_KEYWORDS = {
    "flight",
    "flights",
    "airline",
    "airport",
    "depart",
    "departure",
    "arrival",
    "arrive",
    "round trip",
    "one way",
    "hotel",
    "hotels",
    "room",
    "rooms",
    "check in",
    "check-in",
    "checkout",
    "check out",
    "stay",
    "nights",
    "reservation",
    "travel",
    "trip",
}

BUSINESS_SUPPORT_USER = [
    "This is for a client meeting, so choose timing that keeps me from rushing.",
    "For work travel, I want the option with enough breathing room before the meeting.",
    "When this is a business trip, please default to the safer arrival window.",
    "I am heading to a meeting; build in a real buffer instead of cutting it close.",
    "For client visits, I prefer predictable timing over the absolute cheapest itinerary.",
]

SUPPORT_ASSISTANT = [
    "I would choose the earlier arrival and leave about a 90-minute buffer before the meeting.",
    "For a business meeting, the safer default is the earlier option with a meeting buffer.",
    "I would avoid the tight arrival and preserve a 90-minute cushion before work starts.",
]

SUPPORT_FEEDBACK = [
    "Yes, that buffer is exactly what I need for business travel.",
    "Good call. I do not want to land and rush straight into a meeting.",
    "That is the right default for client trips: early arrival and breathing room.",
    "Exactly. For work travel, reliability matters more than shaving off a little cost.",
]

BOUNDARY_USER = [
    "Plan a relaxed weekend getaway where I can sleep in and wander without a fixed schedule.",
    "This is a leisure trip with no meetings; optimize for comfort and a relaxed pace.",
    "Help me compare vacation options where arrival time is flexible and there is no appointment.",
]

EXCEPTION_USER = [
    "This trip is explicitly flexible; I do not need the usual work-travel buffer.",
    "For this vacation day, there is no meeting clock to protect, so do not force an early arrival.",
    "Plan this as a personal trip where the timing can be looser than my business travel default.",
]

DISTRACTOR_USER = [
    "Help summarize this travel note into a simple checklist.",
    "Compare these restaurants near the hotel without making a booking.",
    "Draft a short message asking whether late checkout is available.",
    "List what I should pack for a two-night trip.",
]


@dataclass
class TravelSeed:
    seed_id: str
    source_dataset: str
    source_domain: str
    prompt: str
    assistant_reply: str
    original_id: str
    turns: int
    user_turns: int
    assistant_turns: int
    quality_score: float


def compact_text(text: Any, max_chars: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = text.replace("\u0000", "")
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def stable_hash(value: str, n: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def has_pii(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in PII_PATTERNS)


def looks_like_english(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 20:
        return False
    non_ascii = sum(1 for ch in compact if ord(ch) > 127)
    if non_ascii / max(len(compact), 1) > 0.08:
        return False
    alpha = [ch for ch in compact if ch.isalpha()]
    if not alpha:
        return False
    ascii_alpha = sum(1 for ch in alpha if "a" <= ch.lower() <= "z")
    return ascii_alpha / max(len(alpha), 1) >= 0.92


def keyword_hits(text: str) -> int:
    lower = text.lower()
    return sum(1 for kw in TRAVEL_KEYWORDS if kw in lower)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def download_file(url: str, dst: Path, timeout: int) -> Tuple[bool, str]:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "habit-bench-taskmaster-builder/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(payload)
        return True, f"downloaded:{url}:bytes={len(payload)}"
    except Exception as exc:
        return False, f"failed:{url}:{type(exc).__name__}:{str(exc)[:240]}"


def ensure_raw_files(raw_dir: Path, timeout: int, refresh: bool) -> Dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"raw_dir": str(raw_dir), "domains": {}, "errors": []}
    for domain in ["flights", "hotels"]:
        dst = raw_dir / f"{domain}.json"
        domain_report = {"path": str(dst), "attempts": []}
        if dst.exists() and not refresh:
            domain_report["status"] = "cache"
            domain_report["bytes"] = dst.stat().st_size
            report["domains"][domain] = domain_report
            continue
        ok = False
        for url in SOURCE_URLS[domain]:
            ok, message = download_file(url, dst, timeout)
            domain_report["attempts"].append(message)
            if ok:
                break
        domain_report["status"] = "ok" if ok else "missing"
        if ok:
            domain_report["bytes"] = dst.stat().st_size
        else:
            report["errors"].append(f"missing_raw_file:{domain}:{dst}")
        report["domains"][domain] = domain_report
    return report


def normalize_turns(row: Dict[str, Any]) -> List[Dict[str, str]]:
    raw_turns = row.get("utterances")
    if raw_turns is None:
        raw_turns = row.get("turns")
    turns: List[Dict[str, str]] = []
    for turn in raw_turns or []:
        speaker = str(turn.get("speaker", turn.get("role", ""))).lower()
        if speaker in {"assistant", "system", "agent"}:
            role = "assistant"
        elif speaker in {"user", "customer"}:
            role = "user"
        else:
            continue
        text = compact_text(turn.get("text", turn.get("utterance", "")), 500)
        if text:
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] = compact_text(turns[-1]["content"] + " " + text, 800)
            else:
                turns.append({"role": role, "content": text})
    return turns


def row_id(row: Dict[str, Any], source_domain: str, index: int) -> str:
    return str(
        row.get("conversation_id")
        or row.get("dialogue_id")
        or row.get("original_id")
        or row.get("id")
        or f"{source_domain}_{index:05d}"
    )


def seed_from_row(row: Dict[str, Any], source_domain: str, index: int) -> Tuple[Optional[TravelSeed], Optional[str]]:
    turns = normalize_turns(row)
    user_turns = [turn["content"] for turn in turns if turn["role"] == "user"]
    assistant_turns = [turn["content"] for turn in turns if turn["role"] == "assistant"]
    if len(user_turns) < 2 or len(assistant_turns) < 2:
        return None, "too_few_turns"

    first_user = next((u for u in user_turns if len(u.split()) >= 5), user_turns[0])
    first_assistant = next((a for a in assistant_turns if len(a.split()) >= 4), assistant_turns[0])
    context = " ".join(turn["content"] for turn in turns[:8])
    seed_text = compact_text(first_user, 450)
    combined = f"{seed_text} {context}"

    if not looks_like_english(seed_text):
        return None, "not_english"
    if has_pii(combined):
        return None, "pii"
    hits = keyword_hits(combined)
    if hits < 2:
        return None, "weak_travel_signal"
    if len(seed_text) < 30 or len(seed_text) > 450:
        return None, "length"

    original_id = row_id(row, source_domain, index)
    quality = hits + min(len(turns), 18) * 0.15 + min(len(user_turns), 8) * 0.25
    return (
        TravelSeed(
            seed_id=f"taskmaster2_{source_domain}_{stable_hash(original_id, 12)}",
            source_dataset=TASKMASTER_DATASET,
            source_domain=source_domain,
            prompt=seed_text,
            assistant_reply=compact_text(first_assistant, 450),
            original_id=original_id,
            turns=len(turns),
            user_turns=len(user_turns),
            assistant_turns=len(assistant_turns),
            quality_score=round(quality, 3),
        ),
        None,
    )


def load_and_filter_seeds(raw_dir: Path, max_seeds_per_domain: int) -> Tuple[List[TravelSeed], Dict[str, Any]]:
    seeds: List[TravelSeed] = []
    report: Dict[str, Any] = {"domains": {}, "total_selected": 0}
    for domain in ["flights", "hotels"]:
        path = raw_dir / f"{domain}.json"
        rows = read_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON list")
        rejected = Counter()
        domain_seeds: List[TravelSeed] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                rejected["non_object_row"] += 1
                continue
            seed, reason = seed_from_row(row, domain, idx)
            if seed is None:
                rejected[reason or "unknown"] += 1
                continue
            domain_seeds.append(seed)
        domain_seeds.sort(key=lambda seed: (-seed.quality_score, seed.seed_id))
        selected = domain_seeds[:max_seeds_per_domain]
        seeds.extend(selected)
        report["domains"][domain] = {
            "raw_rows": len(rows),
            "accepted_before_cap": len(domain_seeds),
            "selected": len(selected),
            "rejected": dict(sorted(rejected.items())),
            "quality_score_min_selected": selected[-1].quality_score if selected else None,
            "quality_score_max_selected": selected[0].quality_score if selected else None,
        }
    report["total_selected"] = len(seeds)
    return seeds, report


def split_for_user(user_index: int) -> str:
    bucket = user_index % 10
    if bucket < 2:
        return "dev"
    if bucket == 9:
        return "stress"
    return "test"


def make_session(
    user_id: str,
    temp_index: int,
    seed: TravelSeed,
    user_text: str,
    assistant_text: str,
    feedback: Optional[str],
    habit_id: Optional[str],
    signal_type: str,
    phase: float,
) -> Dict[str, Any]:
    messages = [
        {"role": "user", "content": compact_text(user_text, 900)},
        {"role": "assistant", "content": compact_text(assistant_text, 900)},
    ]
    if feedback:
        messages.append({"role": "user", "content": compact_text(feedback, 500)})
    return {
        "_phase": phase,
        "_temp_index": temp_index,
        "session_id": "",
        "user_id": user_id,
        "session_index": -1,
        "timestamp": "",
        "domain": "travel",
        "messages": messages,
        "source_seed": {
            "source_dataset": seed.source_dataset,
            "seed_id": seed.seed_id,
            "domain": "travel",
            "source_domain": seed.source_domain,
            "original_id": seed.original_id,
            "prompt_snippet": seed.prompt,
        },
        "memory_annotations": {
            "linked_habit_ids": [habit_id] if habit_id else [],
            "signal_type": signal_type,
        },
    }


def make_choice_set(rng: random.Random, correct_text: str, distractors: Sequence[str]) -> Tuple[List[Dict[str, str]], str]:
    labels = ["A", "B", "C", "D"]
    pool = [correct_text] + list(distractors[:3])
    if len({compact_text(item, 1000) for item in pool}) != 4:
        raise ValueError(f"choice texts must be unique: {pool}")
    rng.shuffle(pool)
    choices = [{"choice_id": labels[i], "text": pool[i]} for i in range(4)]
    gold = next(choice["choice_id"] for choice in choices if choice["text"] == correct_text)
    return choices, gold


def public_probe_id(private_probe_id: str) -> str:
    return f"taskmaster_planning_v01_probe_{stable_hash(private_probe_id, 16)}"


def public_probe(probe: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "probe_id": public_probe_id(probe["probe_id"]),
        "user_id": probe["user_id"],
        "split": probe["split"],
        "query": probe["query"],
        "choices": probe["choices"],
        "visible_history_scope": probe["visible_history_scope"],
        "evaluation_contract": {
            "answer_format": "return one choice_id and optional evidence_session_ids",
            "validator_type": "choice_equals",
        },
        "metadata": {
            "source_slice": "taskmaster2_flights_hotels_planning_defaults_v0_1",
            "stress_variant": probe["metadata"]["stress_variant"],
        },
    }


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
            "source_domain": session["source_seed"]["source_domain"],
        },
    }


def generate_dataset(args: argparse.Namespace, seeds: Sequence[TravelSeed]) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    if len(seeds) < 8:
        raise ValueError(f"Need at least 8 filtered Taskmaster travel seeds, got {len(seeds)}")

    rng = random.Random(args.seed)
    by_domain: Dict[str, List[TravelSeed]] = defaultdict(list)
    for seed in seeds:
        by_domain[seed.source_domain].append(seed)
    all_seeds = list(seeds)

    sessions: List[Dict[str, Any]] = []
    habit_graphs: List[Dict[str, Any]] = []
    probes: List[Dict[str, Any]] = []
    probe_keys: List[Dict[str, Any]] = []

    for user_index in range(args.n_users):
        user_id = f"tm2_planning_user_{user_index:04d}"
        split = split_for_user(user_index)
        habit_id = f"{user_id}_habit_business_travel_early_buffer"
        temp_index = 0
        user_sessions: List[Dict[str, Any]] = []
        support_ids: List[str] = []
        boundary_ids: List[str] = []
        exception_ids: List[str] = []

        habit_graph = {
            "habit_id": habit_id,
            "user_id": user_id,
            "split": split,
            "template_id": "business_travel_early_buffer",
            "family": "planning_defaults",
            "name": "business travel prefers early flights and buffer time",
            "condition": "business travel planning and meeting travel",
            "default_action": "Prefer early arrivals and leave about a 90-minute buffer before meetings.",
            "wrong_action": "Optimize only for cheapest arrival or tight transfers.",
            "boundary_condition": "leisure trips or flexible vacations",
            "boundary_action": "Optimize for comfort and preference discovery instead of early business timing.",
            "exception_condition": "trips explicitly marked flexible or personal",
            "exception_action": "Do not assume the business-travel buffer; ask about leisure priorities or use relaxed timing.",
            "strength": "usually",
            "support_episode_target": args.support_episodes_per_user,
            "source": "taskmaster2_flights_hotels_seeded_synthetic_hidden_habit_graph",
            "sensitivity": "ordinary",
        }
        habit_graphs.append(habit_graph)

        source_cycle = ["flights", "hotels"] * max(args.support_episodes_per_user, 1)
        rng.shuffle(source_cycle)
        for support_idx in range(args.support_episodes_per_user):
            source_domain = source_cycle[support_idx % len(source_cycle)]
            candidates = by_domain.get(source_domain) or all_seeds
            seed = rng.choice(candidates)
            user_text = (
                f"{rng.choice(BUSINESS_SUPPORT_USER)}\n"
                f"Prior travel-search context from Taskmaster-2 {seed.source_domain}: {seed.prompt}"
            )
            session = make_session(
                user_id=user_id,
                temp_index=temp_index,
                seed=seed,
                user_text=user_text,
                assistant_text=rng.choice(SUPPORT_ASSISTANT),
                feedback=rng.choice(SUPPORT_FEEDBACK),
                habit_id=habit_id,
                signal_type="support",
                phase=0.12 + rng.random() * 0.48,
            )
            user_sessions.append(session)
            temp_index += 1

        for _ in range(args.boundary_episodes_per_user):
            seed = rng.choice(by_domain.get("hotels") or all_seeds)
            session = make_session(
                user_id=user_id,
                temp_index=temp_index,
                seed=seed,
                user_text=f"{rng.choice(BOUNDARY_USER)}\nRelated travel context: {seed.prompt}",
                assistant_text="Because this is leisure travel, I would optimize for comfort and timing preference rather than defaulting to an early business arrival.",
                feedback="Right, this is not a work trip, so the business buffer should not drive the plan.",
                habit_id=habit_id,
                signal_type="boundary_counterexample",
                phase=0.50 + rng.random() * 0.25,
            )
            user_sessions.append(session)
            temp_index += 1

        for _ in range(args.exception_episodes_per_user):
            seed = rng.choice(all_seeds)
            session = make_session(
                user_id=user_id,
                temp_index=temp_index,
                seed=seed,
                user_text=f"{rng.choice(EXCEPTION_USER)}\nUse this travel context if useful: {seed.prompt}",
                assistant_text="Since you explicitly marked this trip as flexible, I will not force the early-arrival business default.",
                feedback="Yes, this is an exception to my usual business travel timing.",
                habit_id=habit_id,
                signal_type="exception",
                phase=0.62 + rng.random() * 0.25,
            )
            user_sessions.append(session)
            temp_index += 1

        while len(user_sessions) < args.sessions_per_user:
            seed = rng.choice(all_seeds)
            user_sessions.append(
                make_session(
                    user_id=user_id,
                    temp_index=temp_index,
                    seed=seed,
                    user_text=f"{rng.choice(DISTRACTOR_USER)}\nTaskmaster-2 travel context: {seed.prompt}",
                    assistant_text="I can help with this immediate travel task without adding a durable preference.",
                    feedback=None,
                    habit_id=None,
                    signal_type="distractor",
                    phase=rng.random(),
                )
            )
            temp_index += 1

        user_sessions.sort(key=lambda row: (row["_phase"], row["_temp_index"]))
        start = datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc) + timedelta(days=user_index % 17)
        for idx, session in enumerate(user_sessions):
            session["session_index"] = idx
            session["session_id"] = f"{user_id}_s{idx:04d}"
            session["timestamp"] = (start + timedelta(days=idx * 3 + rng.randint(0, 1))).isoformat()
            session.pop("_phase", None)
            session.pop("_temp_index", None)
            signal = session["memory_annotations"]["signal_type"]
            if signal == "support":
                support_ids.append(session["session_id"])
            elif signal == "boundary_counterexample":
                boundary_ids.append(session["session_id"])
            elif signal == "exception":
                exception_ids.append(session["session_id"])
        sessions.extend(user_sessions)

        max_session_index = max(session["session_index"] for session in user_sessions)
        probe_specs = [
            {
                "probe_type": "direct_use",
                "capability_group": "habit_direct_use",
                "query": "Pick the default travel option for a client meeting next Thursday using the user's established planning preference.",
                "correct": "Choose the option that arrives early and leaves about a 90-minute buffer before the meeting.",
                "distractors": [
                    "Choose the cheapest option even if it arrives right before the meeting.",
                    "Choose the latest departure because shorter travel days are always best.",
                    "Ask the user to restate their preference despite repeated prior evidence.",
                ],
                "gold_action": "apply_scoped_habit",
                "evidence_ids": support_ids,
                "stress_variant": "taskmaster_seeded_direct",
            },
            {
                "probe_type": "boundary",
                "capability_group": "habit_boundary_false_personalization",
                "query": "Plan a relaxed weekend vacation where there is no meeting or appointment to protect.",
                "correct": "Do not apply the business-travel timing rule; optimize for comfort and ask about leisure priorities.",
                "distractors": [
                    "Force an early arrival and a 90-minute meeting buffer anyway.",
                    "Optimize only for the cheapest arrival and ignore comfort.",
                    "Refuse to help because the user has a business travel habit.",
                ],
                "gold_action": "do_not_apply_out_of_scope",
                "evidence_ids": boundary_ids,
                "stress_variant": "taskmaster_seeded_boundary",
            },
            {
                "probe_type": "exception",
                "capability_group": "counterevidence_exception",
                "query": "Plan a personal trip that the user explicitly says is flexible and not tied to a meeting.",
                "correct": "Treat this as an exception and avoid assuming the early-arrival business buffer.",
                "distractors": [
                    "Apply the business-travel buffer because it is the most frequent prior pattern.",
                    "Pick the tightest itinerary solely because it is cheap.",
                    "Say there is no way to plan without deleting the user's travel preference.",
                ],
                "gold_action": "apply_exception",
                "evidence_ids": exception_ids,
                "stress_variant": "taskmaster_seeded_exception",
            },
            {
                "probe_type": "explicit_retrieval",
                "capability_group": "explicit_fact_preference_retrieval",
                "query": "Which travel planning preference has the user repeatedly reinforced for business or client-meeting trips?",
                "correct": "For business travel, prefer early arrivals and leave about a 90-minute buffer before meetings.",
                "distractors": [
                    "For every trip, always choose the cheapest possible arrival.",
                    "For vacations, always force the earliest possible arrival.",
                    "The user has not shown any recurring business-travel timing preference.",
                ],
                "gold_action": "retrieve_explicit_preference_from_support",
                "evidence_ids": support_ids,
                "stress_variant": "taskmaster_seeded_explicit",
            },
        ]

        for pidx, spec in enumerate(probe_specs):
            choices, gold_choice_id = make_choice_set(rng, spec["correct"], spec["distractors"])
            private_probe_id = f"{habit_id}_p{pidx:02d}_{spec['probe_type']}"
            public_id = public_probe_id(private_probe_id)
            probe = {
                "probe_id": private_probe_id,
                "user_id": user_id,
                "split": split,
                "probe_type": spec["probe_type"],
                "habit_id": habit_id,
                "habit_family": "planning_defaults",
                "query": spec["query"],
                "choices": choices,
                "validator": {
                    "type": "choice_equals",
                    "gold_choice_id": gold_choice_id,
                    "gold_action": spec["gold_action"],
                },
                "visible_history_scope": {
                    "user_id": user_id,
                    "max_session_index": max_session_index,
                },
                "metadata": {
                    "template_id": "business_travel_early_buffer",
                    "source_dataset": TASKMASTER_DATASET,
                    "source_domains": ["flights", "hotels"],
                    "stress_variant": spec["stress_variant"],
                    "support_count": len(support_ids),
                    "boundary_count": len(boundary_ids),
                    "exception_count": len(exception_ids),
                    "horizon_sessions": max_session_index + 1,
                },
            }
            probes.append(probe)
            probe_keys.append(
                {
                    "public_probe_id": public_id,
                    "probe_id": private_probe_id,
                    "user_id": user_id,
                    "habit_id": habit_id,
                    "habit_family": "planning_defaults",
                    "probe_type": spec["probe_type"],
                    "capability_group": spec["capability_group"],
                    "gold_choice_id": gold_choice_id,
                    "gold_action": spec["gold_action"],
                    "gold_evidence_session_ids": spec["evidence_ids"],
                    "hidden_habit_graph": habit_graph,
                    "review_status": "taskmaster_planning_defaults_needs_human_review",
                    "source_public_probe_id": None,
                    "stress_variant": spec["stress_variant"],
                }
            )

    return sessions, habit_graphs, probes, probe_keys


def validate_outputs(
    sessions: Sequence[Dict[str, Any]],
    probes: Sequence[Dict[str, Any]],
    probe_keys: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sessions_by_id = {session["session_id"]: session for session in sessions}
    key_by_probe = {key["probe_id"]: key for key in probe_keys}
    public_ids = [key["public_probe_id"] for key in probe_keys]
    summary = Counter()
    rows: List[Dict[str, Any]] = []

    if len(public_ids) != len(set(public_ids)):
        summary["duplicate_public_probe_id"] += 1

    for session in sessions:
        if session.get("domain") != "travel":
            summary["bad_session_domain"] += 1
        seed = session.get("source_seed") or {}
        if seed.get("source_dataset") != TASKMASTER_DATASET:
            summary["bad_source_dataset"] += 1
        if seed.get("source_domain") not in {"flights", "hotels"}:
            summary["bad_source_domain"] += 1
        text = " ".join(msg.get("content", "") for msg in session.get("messages", []))
        if has_pii(text):
            summary["session_pii"] += 1

    for probe in probes:
        errors: List[str] = []
        warnings: List[str] = []
        key = key_by_probe.get(probe["probe_id"])
        if key is None:
            errors.append("missing_private_key")
        if len({choice["text"] for choice in probe["choices"]}) != len(probe["choices"]):
            errors.append("duplicate_choice_text")
        if key:
            if key["gold_choice_id"] not in {choice["choice_id"] for choice in probe["choices"]}:
                errors.append("gold_choice_not_in_choices")
            if key["habit_family"] != "planning_defaults":
                errors.append("bad_habit_family")
            for sid in key.get("gold_evidence_session_ids", []):
                if sid not in sessions_by_id:
                    errors.append(f"missing_evidence_session:{sid}")
            if probe["probe_type"] in {"direct_use", "explicit_retrieval"} and len(key.get("gold_evidence_session_ids", [])) < 3:
                errors.append("insufficient_support_evidence")
            if probe["probe_type"] in {"boundary", "exception"} and not key.get("gold_evidence_session_ids"):
                errors.append("missing_stress_evidence")
        if "habit" in probe["query"].lower():
            warnings.append("query_mentions_habit")
        status = "pass" if not errors else "fail"
        summary[f"probe_status_{status}"] += 1
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "public_probe_id": key["public_probe_id"] if key else "",
                "probe_type": probe["probe_type"],
                "status": status,
                "errors": errors,
                "warnings": warnings,
            }
        )

    summary.update(
        {
            "sessions_total": len(sessions),
            "users_total": len({session["user_id"] for session in sessions}),
            "probes_total": len(probes),
            "private_keys_total": len(probe_keys),
            "support_sessions": sum(1 for s in sessions if s["memory_annotations"]["signal_type"] == "support"),
            "boundary_sessions": sum(1 for s in sessions if s["memory_annotations"]["signal_type"] == "boundary_counterexample"),
            "exception_sessions": sum(1 for s in sessions if s["memory_annotations"]["signal_type"] == "exception"),
            "distractor_sessions": sum(1 for s in sessions if s["memory_annotations"]["signal_type"] == "distractor"),
        }
    )
    return rows, dict(summary)


def make_review_rows(
    probes: Sequence[Dict[str, Any]],
    probe_keys: Sequence[Dict[str, Any]],
    sessions: Sequence[Dict[str, Any]],
    validation_rows: Sequence[Dict[str, Any]],
    sample_rate: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    key_by_probe = {key["probe_id"]: key for key in probe_keys}
    session_by_id = {session["session_id"]: session for session in sessions}
    validation_by_probe = {row["probe_id"]: row for row in validation_rows}
    rows: List[Dict[str, Any]] = []

    for probe in probes:
        key = key_by_probe[probe["probe_id"]]
        evidence_preview = []
        for sid in key.get("gold_evidence_session_ids", [])[:4]:
            session = session_by_id.get(sid)
            if not session:
                continue
            evidence_preview.append(
                {
                    "session_id": sid,
                    "session_index": session["session_index"],
                    "signal_type": session["memory_annotations"]["signal_type"],
                    "domain": session["domain"],
                    "source_domain": session["source_seed"].get("source_domain"),
                    "user": compact_text(session["messages"][0]["content"], 240),
                    "assistant": compact_text(session["messages"][1]["content"], 180),
                    "feedback": compact_text(session["messages"][-1]["content"], 180)
                    if len(session["messages"]) > 2
                    else "",
                }
            )
        validation = validation_by_probe.get(probe["probe_id"], {})
        rows.append(
            {
                "review_id": f"review_{probe['probe_id']}",
                "public_probe_id": key["public_probe_id"],
                "probe_id": probe["probe_id"],
                "user_id": probe["user_id"],
                "split": probe["split"],
                "probe_type": probe["probe_type"],
                "habit_family": probe["habit_family"],
                "stress_variant": key["stress_variant"],
                "query": probe["query"],
                "choices_json": json.dumps(probe["choices"], ensure_ascii=False),
                "proposed_gold_choice_id": key["gold_choice_id"],
                "proposed_gold_action": key["gold_action"],
                "evidence_preview_json": json.dumps(evidence_preview, ensure_ascii=False),
                "auto_validation_status": validation.get("status", "missing"),
                "reviewer_decision": "",
                "reviewer_notes": "",
            }
        )

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["probe_type"]].append(row)
    sample: List[Dict[str, Any]] = []
    for group_rows in grouped.values():
        k = max(1, int(round(len(group_rows) * sample_rate)))
        sample.extend(rng.sample(group_rows, min(k, len(group_rows))))
    sample.sort(key=lambda row: row["review_id"])
    return rows, sample


def write_reports(
    out_dir: Path,
    args: argparse.Namespace,
    download_report: Dict[str, Any],
    filter_report: Dict[str, Any],
    validation_summary: Dict[str, Any],
) -> None:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "taskmaster_planning_defaults_auto_validated_pending_human_audit"
        if validation_summary.get("probe_status_fail", 0) == 0
        else "taskmaster_planning_defaults_validation_failed",
        "source_contract": {
            "seed_prompts": TASKMASTER_DATASET,
            "source_domains": ["flights", "hotels"],
            "habit_family": "planning_defaults",
            "representative_domain": "travel",
            "release_claim": "Taskmaster-2 travel-dialog-seeded, domain-grounded, synthetic longitudinal habit slice",
            "license": "CC BY 4.0",
        },
        "args": serializable_args,
        "download": download_report,
        "filter": filter_report,
        "validation": validation_summary,
        "artifacts": {
            "public_lifelines": "public/lifelines.jsonl",
            "public_probes": "public/probes.jsonl",
            "private_sessions": "private/sessions_with_annotations.jsonl",
            "private_probe_key": "private/probe_key.jsonl",
            "review_queue_all": "review/planning_defaults_review_queue_all.csv",
            "review_queue_sample": "review/planning_defaults_review_queue_sample.csv",
        },
    }
    (reports_dir / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Taskmaster-2 Planning Defaults v0.1",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Status: {manifest['status']}",
        f"- Seed source: `{TASKMASTER_DATASET}`",
        "- Source domains: `flights`, `hotels`",
        "- HABIT-Bench family: `planning_defaults`",
        "- Representative domain: `travel`",
        "- Hidden habit: business travel prefers early arrivals and about a 90-minute meeting buffer",
        "",
        "## Counts",
        "",
        f"- Users: {validation_summary.get('users_total', 0)}",
        f"- Sessions: {validation_summary.get('sessions_total', 0)}",
        f"- Probes: {validation_summary.get('probes_total', 0)}",
        f"- Support sessions: {validation_summary.get('support_sessions', 0)}",
        f"- Boundary sessions: {validation_summary.get('boundary_sessions', 0)}",
        f"- Exception sessions: {validation_summary.get('exception_sessions', 0)}",
        f"- Distractor sessions: {validation_summary.get('distractor_sessions', 0)}",
        "",
        "## Filter Summary",
        "",
    ]
    for domain, row in filter_report.get("domains", {}).items():
        lines.append(
            f"- `{domain}`: raw={row.get('raw_rows')}, accepted_before_cap={row.get('accepted_before_cap')}, selected={row.get('selected')}"
        )
    lines.extend(
        [
            "",
            "## Human Review Handoff",
            "",
            "Review `review/planning_defaults_review_queue_sample.csv` first, then audit the full queue.",
            "Use `accept`, `revise`, or `reject` in `reviewer_decision` and keep notes short and concrete.",
            "",
            "Primary accept criteria: Taskmaster seed is travel-domain coherent, repeated support establishes the business-travel buffer default, boundary/exception probes avoid false personalization, and the gold choice is unique.",
        ]
    )
    (reports_dir / "planning_defaults_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    raw_dir = args.raw_data_dir or out_dir / "data" / "raw_taskmaster"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    download_report = ensure_raw_files(raw_dir, timeout=args.download_timeout_sec, refresh=args.refresh_raw)
    if download_report.get("errors"):
        reports_dir = out_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "download_failure.json").write_text(
            json.dumps(download_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise SystemExit(
            "Could not locate/download Taskmaster-2 flights.json and hotels.json. "
            f"Place them under {raw_dir} or pass --raw-data-dir. "
            f"Wrote {reports_dir / 'download_failure.json'}"
        )
    stale_failure = out_dir / "reports" / "download_failure.json"
    if stale_failure.exists():
        stale_failure.unlink()

    seeds, filter_report = load_and_filter_seeds(raw_dir, args.max_seeds_per_domain)
    rng = random.Random(args.seed)
    rng.shuffle(seeds)
    seed_rows = [seed.__dict__ for seed in seeds]
    write_jsonl(out_dir / "data" / "filtered_taskmaster_travel_seeds.jsonl", seed_rows)

    sessions, habit_graphs, probes, probe_keys = generate_dataset(args, seeds)
    public_probes = [public_probe(probe) for probe in probes]
    for key in probe_keys:
        key["public_probe_id"] = public_probe_id(key["probe_id"])

    validation_rows, validation_summary = validate_outputs(sessions, probes, probe_keys)
    validation_summary["elapsed_sec"] = round(time.time() - started, 3)

    public_sessions = [public_session(session) for session in sessions]
    write_jsonl(out_dir / "public" / "lifelines.jsonl", public_sessions)
    write_jsonl(out_dir / "public" / "probes.jsonl", public_probes)
    write_jsonl(out_dir / "private" / "sessions_with_annotations.jsonl", sessions)
    write_jsonl(out_dir / "private" / "habit_graphs.jsonl", habit_graphs)
    write_jsonl(out_dir / "private" / "probe_key.jsonl", probe_keys)
    write_jsonl(out_dir / "reports" / "auto_validation_rows.jsonl", validation_rows)
    (out_dir / "reports" / "auto_validation_summary.json").write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    review_rows, review_sample = make_review_rows(
        probes=probes,
        probe_keys=probe_keys,
        sessions=sessions,
        validation_rows=validation_rows,
        sample_rate=args.review_sample_rate,
        rng=rng,
    )
    write_csv(out_dir / "review" / "planning_defaults_review_queue_all.csv", review_rows)
    write_csv(out_dir / "review" / "planning_defaults_review_queue_sample.csv", review_sample)
    write_jsonl(out_dir / "review" / "planning_defaults_review_queue_all.jsonl", review_rows)
    write_jsonl(out_dir / "review" / "planning_defaults_review_queue_sample.jsonl", review_sample)

    write_reports(out_dir, args, download_report, filter_report, validation_summary)

    print(
        json.dumps(
            {
                "status": "ok" if validation_summary.get("probe_status_fail", 0) == 0 else "validation_failed",
                "out_dir": str(out_dir),
                "users": validation_summary.get("users_total"),
                "sessions": validation_summary.get("sessions_total"),
                "probes": validation_summary.get("probes_total"),
                "filtered_seeds": filter_report.get("total_selected"),
                "elapsed_sec": validation_summary["elapsed_sec"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/petrelfs/linzhouhan/xqwang/project/habit-bench/runs_wxq/taskmaster_planning_defaults_v0_1"),
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Optional directory containing flights.json and hotels.json. Defaults to OUT/data/raw_taskmaster.",
    )
    parser.add_argument("--n-users", type=int, default=30)
    parser.add_argument("--sessions-per-user", type=int, default=36)
    parser.add_argument("--support-episodes-per-user", type=int, default=5)
    parser.add_argument("--boundary-episodes-per-user", type=int, default=2)
    parser.add_argument("--exception-episodes-per-user", type=int, default=1)
    parser.add_argument("--max-seeds-per-domain", type=int, default=800)
    parser.add_argument("--review-sample-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--download-timeout-sec", type=int, default=45)
    parser.add_argument("--refresh-raw", action="store_true")
    args = parser.parse_args()

    if args.sessions_per_user < args.support_episodes_per_user + args.boundary_episodes_per_user + args.exception_episodes_per_user:
        parser.error("--sessions-per-user is too small for requested evidence episodes")
    if not (0 < args.review_sample_rate <= 1):
        parser.error("--review-sample-rate must be in (0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
