#!/usr/bin/env python
"""Build v0.2 of the Taskmaster-2 planning_defaults HABIT-Bench slice.

Compared with v0.1, this version focuses on:

- longer multi-turn sessions;
- less revealing probe wording;
- stronger plausible distractor choices;
- optional gpt-5.5/xhigh generation of a probe wording bank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


TASKMASTER_DATASET = "google-research-datasets/Taskmaster-2"
FAMILY = "planning_defaults"
DOMAIN = "travel"
TEMPLATE_ID = "business_travel_early_buffer"
VALID_PROBE_TYPES = {"direct_use", "boundary", "exception", "explicit_retrieval"}
VALID_LLM_SIGNALS = {"support", "boundary_counterexample", "exception", "distractor"}
CAPABILITY_GROUP_BY_TYPE = {
    "direct_use": "habit_direct_use",
    "boundary": "habit_boundary_false_personalization",
    "exception": "counterevidence_exception",
    "explicit_retrieval": "explicit_fact_preference_retrieval",
}


LLM_SYSTEM_PROMPT = """You generate high-quality HABIT-Bench benchmark data.

Return strict JSON only. Do not include markdown.

The benchmark family is planning_defaults: an agent must infer a scoped user
planning default from long user history and apply it only when appropriate.

Each user has one hidden planning default selected from a small template bank.
Use the provided template exactly as the scoped hidden pattern for that user.

Quality requirements:
- Sessions must be natural assistant-user conversations, not copied seed text
  with a few appended turns.
- Use Taskmaster-2 flight/hotel seeds as scenario inspiration, but do not copy
  them verbatim.
- Avoid public meta words: habit, benchmark, gold, label, evidence, probe,
  dataset, annotation.
- Avoid making choices trivial. Wrong choices must be plausible travel tradeoffs.
- Correct answers should require reading user history, except explicit retrieval
  questions which ask for the repeated preference.
"""


HABIT_TEMPLATES = [
    {
        "template_id": "business_travel_arrival_buffer",
        "name": "business travel prefers protected arrival buffer",
        "condition": "business/client/onsite meeting travel planning",
        "default_action": "Prefer an arrival window with meaningful cushion before meeting-dependent commitments over tighter cheaper schedules.",
        "boundary_condition": "relaxed leisure or personal travel without a same-day commitment",
        "exception_condition": "current itinerary explicitly relaxes arrival timing or moves commitments to a later day",
    },
    {
        "template_id": "tight_schedule_nonstop_priority",
        "name": "tight schedules prefer nonstop flights",
        "condition": "travel with same-day meetings, short connection windows, or high delay sensitivity",
        "default_action": "Prefer nonstop or lowest-transfer itineraries over slightly cheaper connecting options.",
        "boundary_condition": "relaxed travel where transfers are acceptable and cost matters more",
        "exception_condition": "current trip explicitly prioritizes budget or airline miles over schedule reliability",
    },
    {
        "template_id": "uncertain_trip_refundable_fare",
        "name": "uncertain trips prefer refundable fares",
        "condition": "work trips with tentative meetings, pending approvals, or plans that may shift",
        "default_action": "Prefer refundable or flexible fares even when a nonrefundable fare is cheaper.",
        "boundary_condition": "fixed personal trips with confirmed dates",
        "exception_condition": "current request explicitly says the dates are locked and lowest fare is the priority",
    },
    {
        "template_id": "work_hotel_near_venue",
        "name": "work hotels prefer venue proximity",
        "condition": "work, conference, client, or onsite meeting hotel selection",
        "default_action": "Prefer hotels close to the meeting venue over cheaper hotels with a longer commute.",
        "boundary_condition": "leisure trips where neighborhood character or sightseeing access matters more",
        "exception_condition": "current request explicitly prioritizes budget or a specific neighborhood over commute time",
    },
    {
        "template_id": "quiet_hotel_for_work",
        "name": "work hotels prefer quiet properties",
        "condition": "business travel or trips requiring preparation, calls, or focused work",
        "default_action": "Prefer quieter hotels with reliable workspace over nightlife-heavy or amenity-focused properties.",
        "boundary_condition": "social or leisure trips where nightlife and amenities are the goal",
        "exception_condition": "current request explicitly asks for lively atmosphere or group entertainment",
    },
    {
        "template_id": "leisure_relaxed_pacing",
        "name": "leisure trips prefer relaxed pacing",
        "condition": "vacation, weekend, or low-pressure personal travel planning",
        "default_action": "Prefer relaxed arrival times and fewer tightly packed logistics over maximizing every hour.",
        "boundary_condition": "work trips or event travel with fixed commitments",
        "exception_condition": "current request explicitly asks to maximize sightseeing time or fit a hard schedule",
    },
    {
        "template_id": "family_trip_flexible_cancellation",
        "name": "family trips prefer flexible cancellation",
        "condition": "family travel, trips with children, or plans involving multiple relatives",
        "default_action": "Prefer flexible cancellation and change policies over the absolute cheapest prepaid option.",
        "boundary_condition": "solo fixed-date trips with low uncertainty",
        "exception_condition": "current request explicitly says plans are locked and budget is the only priority",
    },
    {
        "template_id": "short_trip_no_checked_bag",
        "name": "short trips avoid checked baggage",
        "condition": "one- or two-night trips, quick work visits, or short weekend travel",
        "default_action": "Prefer carry-on-friendly itineraries and avoid options that require checked baggage.",
        "boundary_condition": "longer trips or trips requiring bulky equipment",
        "exception_condition": "current request explicitly includes items that require checked luggage",
    },
    {
        "template_id": "international_long_layover_buffer",
        "name": "international trips prefer longer layover buffers",
        "condition": "international flights, customs, immigration, or airport changes",
        "default_action": "Prefer longer connection buffers over tight layovers, even if total travel time is longer.",
        "boundary_condition": "domestic point-to-point travel with low connection risk",
        "exception_condition": "current request explicitly prioritizes the shortest possible total itinerary",
    },
    {
        "template_id": "red_eye_avoidance",
        "name": "avoid red-eye unless savings are large",
        "condition": "flight planning where an overnight flight is optional",
        "default_action": "Avoid red-eye flights unless the price or schedule advantage is substantial.",
        "boundary_condition": "trips where overnight travel is necessary or the user asks to preserve daytime hours",
        "exception_condition": "current request explicitly asks for red-eye timing or maximum daytime availability",
    },
    {
        "template_id": "airport_transfer_reliability",
        "name": "client trips prefer reliable airport transfers",
        "condition": "client, meeting, or work trips requiring ground transfer from the airport",
        "default_action": "Prefer predictable ground transfer options over the cheapest or least certain option.",
        "boundary_condition": "casual personal trips with flexible arrival plans",
        "exception_condition": "current request explicitly prioritizes lowest transport cost or public transit exploration",
    },
    {
        "template_id": "early_morning_avoidance",
        "name": "avoid very early departures for nonurgent trips",
        "condition": "nonurgent personal or flexible travel where departure time is optional",
        "default_action": "Prefer mid-morning or later departures over very early flights when tradeoffs are modest.",
        "boundary_condition": "business or event travel where arriving early is critical",
        "exception_condition": "current request explicitly needs the earliest possible arrival",
    },
]


def habit_for_user(user_index: int) -> Dict[str, str]:
    return HABIT_TEMPLATES[user_index % len(HABIT_TEMPLATES)]


DEFAULT_PROBE_BANK = {
    "direct_use": [
        {
            "query": "The current request has two outbound options for a client visit: one is cheaper but leaves little time after arrival, and one arrives earlier at a modest extra cost. With the user's history available, what should the assistant recommend?",
            "correct": "Prefer the earlier arrival and explain that the cheaper tight option is only a fallback if the user wants to trade away schedule cushion.",
            "distractors": [
                "Prefer the cheaper option because it still reaches the city before the visit begins.",
                "Prefer the shortest total travel time and treat waiting time after arrival as wasted time.",
                "Present both options without a recommendation because the user has not named a preferred airline.",
            ],
        },
        {
            "query": "For an upcoming onsite work trip, the assistant must choose between a tight but convenient itinerary and a slightly less convenient itinerary that gives more room before arrival-dependent plans. What is the best default?",
            "correct": "Rank the itinerary with more room first, while still reporting the convenience and fare tradeoff.",
            "distractors": [
                "Rank the tight itinerary first because convenience should dominate unless the user says otherwise.",
                "Rank by fare alone because both options technically satisfy the calendar.",
                "Defer all ranking until the user repeats their full travel profile in the current chat.",
            ],
        },
    ],
    "boundary": [
        {
            "query": "The current request is a relaxed weekend visit with no fixed appointment after arrival. How should the assistant use the user's earlier travel history?",
            "correct": "Use the history cautiously and optimize this case around comfort, pace, and the user's stated leisure preferences.",
            "distractors": [
                "Carry over the earliest-arrival preference as the main ranking factor because it appeared in earlier trips.",
                "Ignore comfort and neighborhood details because timing patterns are more stable than trip purpose.",
                "Decline to recommend because any mismatch between past and current trip purpose makes history unusable.",
            ],
        },
        {
            "query": "The user asks for a low-pressure hotel-and-flight plan for a personal getaway, and says the first day is flexible. What should the assistant avoid assuming?",
            "correct": "Avoid treating the itinerary as if it must protect a same-day professional commitment.",
            "distractors": [
                "Avoid considering later arrivals because the user's prior trips make early arrival the only safe recommendation.",
                "Avoid asking about comfort or neighborhood because the task is still travel-related.",
                "Avoid using the user's current statement because long-term history should override it.",
            ],
        },
    ],
    "exception": [
        {
            "query": "The user says this particular trip can start at any comfortable time because the first commitment is not until the next day. Which recommendation is most appropriate?",
            "correct": "Use the looser timing for this case and compare comfort, cost, and arrival fatigue before choosing.",
            "distractors": [
                "Still prioritize the earlier arrival as the deciding factor because it matched several previous work trips.",
                "Choose the latest arrival automatically because flexibility means timing no longer matters.",
                "Refuse to use any past travel history because one constraint differs from prior trips.",
            ],
        },
        {
            "query": "The user notes that this itinerary is for an informal visit and arrival timing is not protecting a same-day obligation. What should the assistant do?",
            "correct": "Treat the relaxed current constraint as controlling and compare normal travel tradeoffs.",
            "distractors": [
                "Make the earlier-arrival cushion the deciding factor because it is useful in many travel settings.",
                "Pick the cheapest option without discussing comfort or timing tradeoffs.",
                "Ask the user to create a new travel profile before answering this itinerary.",
            ],
        },
    ],
    "explicit_retrieval": [
        {
            "query": "Across the user's prior work-travel planning, what recurring timing preference is best supported by the evidence?",
            "correct": "They tend to prefer arriving with meaningful time to spare before client or onsite meetings.",
            "distractors": [
                "They tend to prefer the cheapest itinerary as long as it arrives before the listed commitment.",
                "They tend to prefer the shortest total travel time even when arrival timing is tight.",
                "They have expressed only hotel-location preferences, not timing preferences.",
            ],
        },
        {
            "query": "Which statement best summarizes the user's repeated pattern for meeting-related travel?",
            "correct": "For meeting-related travel, they value a safer arrival window over a tightly optimized schedule.",
            "distractors": [
                "For meeting-related travel, they value minimal total travel time above arrival cushion.",
                "For meeting-related travel, they value lower fare above the risk of arriving rushed.",
                "For meeting-related travel, they have shown only airline and seat preferences.",
            ],
        },
    ],
}


def compact_text(text: Any, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def stable_hash(value: str, n: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


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


def choice_set(rng: random.Random, correct: str, distractors: Sequence[str]) -> Tuple[List[Dict[str, str]], str]:
    labels = ["A", "B", "C", "D"]
    texts = [correct] + list(distractors[:3])
    if len(set(texts)) != 4:
        raise ValueError("choice texts are not unique")
    rng.shuffle(texts)
    choices = [{"choice_id": labels[i], "text": texts[i]} for i in range(4)]
    gold = next(c["choice_id"] for c in choices if c["text"] == correct)
    return choices, gold


def public_probe_id(private_probe_id: str) -> str:
    return f"taskmaster_planning_v02_probe_{stable_hash(private_probe_id, 16)}"


def split_for_user(user_index: int) -> str:
    bucket = user_index % 10
    if bucket < 2:
        return "dev"
    if bucket == 9:
        return "stress"
    return "test"


def call_curl_json(base_url: str, api_key: str, payload: Dict[str, Any], timeout: int, retries: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload, ensure_ascii=False)
    last = None
    for attempt in range(retries + 1):
        proc = subprocess.run(
            [
                "curl",
                "-sS",
                "--http1.1",
                "--connect-timeout",
                "30",
                "--max-time",
                str(timeout),
                "-H",
                f"Authorization: Bearer {api_key}",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
                url,
            ],
            input=body,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode == 0:
            try:
                data = json.loads(proc.stdout)
                if "error" not in data:
                    return data
                last = json.dumps(data["error"], ensure_ascii=False)
            except Exception as exc:
                last = f"bad json: {type(exc).__name__}: {proc.stdout[:300]}"
        else:
            last = f"curl exit {proc.returncode}: {proc.stderr[:300]}"
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(last or "unknown curl error")


def generate_probe_bank_with_llm(args: argparse.Namespace, out_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    path = out_dir / "data" / "llm_probe_bank_gpt55_xhigh.json"
    if path.exists() and not args.refresh_llm_probe_bank:
        return json.loads(path.read_text(encoding="utf-8"))
    api_key = os.environ.get("HABITBENCH_API_KEY")
    base_url = os.environ.get("HABITBENCH_BASE_URL", args.base_url)
    if not api_key:
        return DEFAULT_PROBE_BANK
    prompt = {
        "task": "Generate robust multiple-choice probe wording for HABIT-Bench planning_defaults.",
        "hidden_habit": "For business/client-meeting travel, the user tends to prefer arrival with a meaningful buffer before meetings, even when cheaper or shorter options are available.",
        "requirements": [
            "Return strict JSON with keys direct_use, boundary, exception, explicit_retrieval.",
            "Each key must contain exactly 3 objects.",
            "Each object has query, correct, distractors.",
            "distractors is a list of exactly 3 plausible but wrong choices.",
            "Do not use the words habit, benchmark, gold, business-travel rule, exception, or established preference.",
            "Avoid straw-man choices; make distractors plausible tradeoffs.",
            "Queries should require reading history, not be answerable purely from wording.",
        ],
    }
    payload = {
        "model": args.llm_model,
        "messages": [
            {"role": "system", "content": "You write benchmark probe text as strict JSON."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "max_tokens": 1800,
        "reasoning_effort": "xhigh",
        "response_format": {"type": "json_object"},
    }
    response = call_curl_json(base_url, api_key, payload, timeout=args.llm_timeout_sec, retries=args.llm_retries)
    content = response["choices"][0]["message"]["content"]
    bank = json.loads(content)
    if not validate_probe_bank(bank):
        bank = DEFAULT_PROBE_BANK
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    return bank


def validate_probe_bank(bank: Dict[str, Any]) -> bool:
    required = {"direct_use", "boundary", "exception", "explicit_retrieval"}
    if set(bank) != required:
        return False
    banned = ["habit", "benchmark", "gold", "business-travel rule", "exception", "established preference"]
    for key in required:
        if not isinstance(bank[key], list) or len(bank[key]) < 2:
            return False
        for item in bank[key]:
            if not all(k in item for k in ["query", "correct", "distractors"]):
                return False
            if len(item["distractors"]) != 3:
                return False
            text = " ".join([item["query"], item["correct"], *item["distractors"]]).lower()
            if any(b in text for b in banned):
                return False
    return True


def seed_text(seed: Dict[str, Any]) -> str:
    return compact_text(seed.get("prompt", ""), 260)


def session_chars(messages: Sequence[Dict[str, str]]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def session_words(messages: Sequence[Dict[str, str]]) -> int:
    return sum(len(str(m.get("content", "")).split()) for m in messages)


def percentile(values: Sequence[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def normalize_raw_utterances(utterances: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    role_map = {"USER": "user", "ASSISTANT": "assistant"}
    messages = []
    for utterance in utterances:
        text = re.sub(r"\s+", " ", str(utterance.get("text", ""))).strip()
        if not text:
            continue
        role = role_map.get(str(utterance.get("speaker", "")).upper(), "user")
        messages.append({"role": role, "content": text})
    return messages


def hydrate_seeds_with_raw_dialogs(seeds: List[Dict[str, Any]], raw_dir: Path) -> List[Dict[str, Any]]:
    """Attach original Taskmaster utterances so v0.2 can keep long-session evidence."""
    raw_by_id: Dict[str, List[Dict[str, str]]] = {}
    for source_domain in ["flights", "hotels"]:
        path = raw_dir / f"{source_domain}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            conversation_id = row.get("conversation_id")
            messages = normalize_raw_utterances(row.get("utterances", []))
            if conversation_id and len(messages) >= 4:
                raw_by_id[conversation_id] = messages
    hydrated = []
    for seed in seeds:
        messages = raw_by_id.get(seed.get("original_id"))
        if not messages:
            continue
        item = dict(seed)
        item["raw_messages"] = messages
        item["source_chars"] = session_chars(messages)
        item["source_words"] = session_words(messages)
        item["source_messages"] = len(messages)
        hydrated.append(item)
    return hydrated


def seed_export_row(seed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "seed_id": seed.get("seed_id"),
        "original_id": seed.get("original_id"),
        "source_dataset": seed.get("source_dataset", TASKMASTER_DATASET),
        "source_domain": seed.get("source_domain"),
        "turns": seed.get("turns"),
        "user_turns": seed.get("user_turns"),
        "assistant_turns": seed.get("assistant_turns"),
        "quality_score": seed.get("quality_score"),
        "source_messages": seed.get("source_messages"),
        "source_chars": seed.get("source_chars"),
        "source_words": seed.get("source_words"),
        "prompt": seed.get("prompt"),
    }


def append_until_long_enough(
    messages: List[Dict[str, str]],
    min_chars: int,
    min_messages: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    neutral_pairs = [
        (
            "Could you also summarize the timing tradeoffs before making the final recommendation?",
            "Sure. I will separate schedule risk, total price, convenience, and any assumptions that could change the recommendation.",
        ),
        (
            "Please keep the comparison practical rather than just listing every option.",
            "Understood. I will focus on what changes the decision and keep minor differences in a short note.",
        ),
        (
            "If two options are close, call out what would make one safer or easier in practice.",
            "I will do that and make the tradeoff explicit instead of treating close options as identical.",
        ),
    ]
    i = 0
    while session_chars(messages) < min_chars or len(messages) < min_messages:
        user_text, assistant_text = neutral_pairs[i % len(neutral_pairs)]
        messages.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}])
        i += 1
        if i > 8:
            break
    return messages


def source_messages_for_session(seed: Dict[str, Any], max_source_messages: int, rng: random.Random) -> List[Dict[str, str]]:
    messages = list(seed.get("raw_messages", []))
    if len(messages) <= max_source_messages:
        return [dict(m) for m in messages]
    # Keep the opening task setup and a contiguous later block, so the dialogue still reads naturally.
    head = messages[:4]
    remaining_slots = max_source_messages - len(head)
    start_max = max(4, len(messages) - remaining_slots)
    start = rng.randint(4, start_max) if start_max > 4 else 4
    return [dict(m) for m in head + messages[start : start + remaining_slots]]


def make_long_session(
    user_id: str,
    temp_index: int,
    seed: Dict[str, Any],
    habit_id: str | None,
    signal_type: str,
    phase: float,
    rng: random.Random,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    source_domain = seed.get("source_domain", "travel")
    context = seed_text(seed)
    messages = source_messages_for_session(seed, args.max_source_messages, rng)
    if signal_type == "support":
        meeting_time = rng.choice(["1:30 p.m.", "2 p.m.", "3 p.m.", "late morning"])
        messages.extend(
            [
                {"role": "user", "content": f"One more constraint: this is for an onsite client meeting around {meeting_time}, so the arrival day matters."},
                {"role": "assistant", "content": "Should I treat that as a same-day commitment where arrival reliability matters, or is there room to arrive shortly before it starts?"},
                {"role": "user", "content": "It is a same-day commitment. I care about not walking in rushed, but I still want a reasonable fare."},
                {"role": "assistant", "content": "Then I would shortlist the option that gets you in with real breathing room before the meeting, and keep a cheaper tight option as a backup only if the schedule is very reliable."},
                {"role": "user", "content": rng.choice([
                "That is the right tradeoff for my client trips. I would rather have time to settle than save a little money.",
                "Yes, for meeting travel I want the safer arrival window, not the option that cuts it close.",
                "Good. When there is a client meeting, give me a cushion before I need to be onsite.",
                ])},
                {"role": "assistant", "content": "Got it. For meeting-related travel I will treat schedule cushion as a priority, while still noting cost and convenience tradeoffs."},
            ]
        )
    elif signal_type == "boundary_counterexample":
        messages.extend(
            [
                {"role": "user", "content": "For this case, treat it as a relaxed personal trip rather than work travel."},
                {"role": "assistant", "content": "Do you have anything scheduled right after arrival, or is the first day flexible?"},
                {"role": "user", "content": "It is flexible. I want the first day to feel easy, not maximized."},
                {"role": "assistant", "content": "In that case I would compare comfort, neighborhood, arrival convenience, and price rather than optimizing around a protected meeting window."},
                {"role": "user", "content": rng.choice([
                "Exactly. This is not one of those work trips where the arrival buffer decides everything.",
                "Right, for a weekend trip I care more about pace and comfort.",
                ])},
                {"role": "assistant", "content": "I will frame this one as leisure planning and avoid carrying over work-trip timing assumptions."},
            ]
        )
    elif signal_type == "exception":
        messages.extend(
            [
                {"role": "assistant", "content": "Before I rank the final timing, should I treat arrival as protecting a same-day commitment?"},
                {"role": "user", "content": "This one is looser. The important thing starts the next day, so I do not need to optimize around a same-day meeting."},
                {"role": "assistant", "content": "Then I would compare normal travel tradeoffs: comfort, fare, hotel check-in, and how tired you want to be on arrival."},
                {"role": "user", "content": "Yes. For this particular trip, do not let my usual meeting-day caution drive the whole plan."},
                {"role": "assistant", "content": "Understood. I will handle this itinerary by its current constraints rather than defaulting to the meeting-day pattern."},
            ]
        )
    else:
        messages.extend(
            [
                {"role": "user", "content": rng.choice([
                    "Mostly summarize the options so I can decide later.",
                    "Just make a checklist for this specific trip.",
                    "Compare the practical details without turning this into a future rule.",
                ])},
                {"role": "assistant", "content": "Here is a neutral comparison of the immediate options without adding a durable preference."},
            ]
        )
    messages = append_until_long_enough(messages, args.min_session_chars, args.min_messages_per_session, rng)
    return {
        "_phase": phase,
        "_temp_index": temp_index,
        "session_id": "",
        "user_id": user_id,
        "session_index": -1,
        "timestamp": "",
        "domain": DOMAIN,
        "messages": messages,
        "source_seed": {
            "source_dataset": TASKMASTER_DATASET,
            "seed_id": seed["seed_id"],
            "domain": DOMAIN,
            "source_domain": source_domain,
            "original_id": seed.get("original_id"),
            "prompt_snippet": context,
            "source_chars": seed.get("source_chars"),
            "source_messages": seed.get("source_messages"),
        },
        "memory_annotations": {
            "linked_habit_ids": [habit_id] if habit_id else [],
            "signal_type": signal_type,
        },
    }


def make_public_session(session: Dict[str, Any]) -> Dict[str, Any]:
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


def generate(args: argparse.Namespace, seeds: List[Dict[str, Any]], probe_bank: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(args.seed)
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        by_domain[seed.get("source_domain", "travel")].append(seed)
    long_by_domain: Dict[str, List[Dict[str, Any]]] = {
        domain: [s for s in rows if int(s.get("source_chars") or 0) >= args.min_raw_session_chars]
        for domain, rows in by_domain.items()
    }

    def pick_seed(source_domain: str | None = None) -> Dict[str, Any]:
        if source_domain:
            pool = long_by_domain.get(source_domain) or by_domain.get(source_domain) or seeds
        else:
            pool = [s for rows in long_by_domain.values() for s in rows] or seeds
        return rng.choice(pool)

    sessions: List[Dict[str, Any]] = []
    graphs: List[Dict[str, Any]] = []
    probes: List[Dict[str, Any]] = []
    keys: List[Dict[str, Any]] = []
    for user_index in range(args.n_users):
        user_id = f"tm2_planning_v02_user_{user_index:04d}"
        split = split_for_user(user_index)
        habit_id = f"{user_id}_habit_business_travel_buffer"
        graph = {
            "habit_id": habit_id,
            "user_id": user_id,
            "split": split,
            "template_id": TEMPLATE_ID,
            "family": FAMILY,
            "name": "business travel prefers early arrival buffer",
            "condition": "business/client-meeting travel planning",
            "default_action": "Favor an arrival window with meaningful cushion before meetings over tight cheaper schedules.",
            "boundary_condition": "leisure or relaxed personal travel",
            "exception_condition": "current itinerary explicitly does not protect a same-day meeting",
            "source": "taskmaster2_seeded_longer_multiturn_synthetic_habit_graph_v02",
            "support_episode_target": args.support_episodes_per_user,
        }
        graphs.append(graph)
        temp = 0
        user_sessions: List[Dict[str, Any]] = []
        for j in range(args.support_episodes_per_user):
            seed = pick_seed("flights" if j % 2 == 0 else "hotels")
            user_sessions.append(make_long_session(user_id, temp, seed, habit_id, "support", 0.12 + rng.random() * 0.45, rng, args))
            temp += 1
        for _ in range(args.boundary_episodes_per_user):
            user_sessions.append(make_long_session(user_id, temp, pick_seed(), habit_id, "boundary_counterexample", 0.50 + rng.random() * 0.25, rng, args))
            temp += 1
        for _ in range(args.exception_episodes_per_user):
            user_sessions.append(make_long_session(user_id, temp, pick_seed(), habit_id, "exception", 0.62 + rng.random() * 0.25, rng, args))
            temp += 1
        while len(user_sessions) < args.sessions_per_user:
            user_sessions.append(make_long_session(user_id, temp, pick_seed(), None, "distractor", rng.random(), rng, args))
            temp += 1
        user_sessions.sort(key=lambda row: (row["_phase"], row["_temp_index"]))
        start = datetime(2025, 2, 3, 9, 0, tzinfo=timezone.utc) + timedelta(days=user_index % 19)
        evidence = defaultdict(list)
        for idx, s in enumerate(user_sessions):
            s["session_index"] = idx
            s["session_id"] = f"{user_id}_s{idx:04d}"
            s["timestamp"] = (start + timedelta(days=idx * 3 + rng.randint(0, 1))).isoformat()
            s.pop("_phase", None)
            s.pop("_temp_index", None)
            if s["memory_annotations"]["linked_habit_ids"]:
                evidence[s["memory_annotations"]["signal_type"]].append(s["session_id"])
        sessions.extend(user_sessions)
        max_idx = len(user_sessions) - 1
        specs = [
            ("direct_use", "habit_direct_use", "apply_scoped_habit", evidence["support"]),
            ("boundary", "habit_boundary_false_personalization", "do_not_apply_out_of_scope", evidence["boundary_counterexample"]),
            ("exception", "counterevidence_exception", "apply_current_trip_constraint", evidence["exception"]),
            ("explicit_retrieval", "explicit_fact_preference_retrieval", "retrieve_supported_preference", evidence["support"]),
        ]
        for pidx, (ptype, capability, action, evidence_ids) in enumerate(specs):
            item = rng.choice(probe_bank[ptype])
            choices, gold = choice_set(rng, item["correct"], item["distractors"])
            private_id = f"{habit_id}_p{pidx:02d}_{ptype}"
            public_id = public_probe_id(private_id)
            probe = {
                "probe_id": private_id,
                "user_id": user_id,
                "split": split,
                "probe_type": ptype,
                "habit_id": habit_id,
                "habit_family": FAMILY,
                "query": item["query"],
                "choices": choices,
                "validator": {"type": "choice_equals", "gold_choice_id": gold, "gold_action": action},
                "visible_history_scope": {"user_id": user_id, "max_session_index": max_idx},
                "metadata": {
                    "template_id": TEMPLATE_ID,
                    "source_dataset": TASKMASTER_DATASET,
                    "source_domains": ["flights", "hotels"],
                    "stress_variant": "taskmaster_seeded_v02_longer_sessions",
                    "support_count": len(evidence["support"]),
                    "boundary_count": len(evidence["boundary_counterexample"]),
                    "exception_count": len(evidence["exception"]),
                    "horizon_sessions": max_idx + 1,
                },
            }
            probes.append(probe)
            keys.append(
                {
                    "public_probe_id": public_id,
                    "probe_id": private_id,
                    "user_id": user_id,
                    "habit_id": habit_id,
                    "habit_family": FAMILY,
                    "probe_type": ptype,
                    "capability_group": capability,
                    "gold_choice_id": gold,
                    "gold_action": action,
                    "gold_evidence_session_ids": evidence_ids,
                    "hidden_habit_graph": graph,
                    "review_status": "taskmaster_planning_defaults_v02_needs_human_review",
                    "source_public_probe_id": None,
                    "stress_variant": "taskmaster_seeded_v02_longer_sessions",
                }
            )
    return sessions, graphs, probes, keys


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
            "source_slice": "taskmaster2_flights_hotels_planning_defaults_v0_2",
            "stress_variant": probe["metadata"]["stress_variant"],
        },
    }


def normalize_llm_messages(raw_messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for raw in raw_messages:
        role = str(raw.get("role", "")).strip().lower()
        content = compact_text(raw.get("content", ""), 2600)
        if role not in {"user", "assistant"} or not content:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = compact_text(messages[-1]["content"] + " " + content, 3000)
        else:
            messages.append({"role": role, "content": content})
    return messages


def build_llm_user_prompt(user_id: str, split: str, seeds: Sequence[Dict[str, Any]], args: argparse.Namespace) -> str:
    seed_payload = [
        {
            "seed_id": seed["seed_id"],
            "source_domain": seed.get("source_domain"),
            "prompt": seed.get("prompt"),
            "assistant_reply": seed.get("assistant_reply"),
        }
        for seed in seeds
    ]
    distractor_count = args.sessions_per_user - args.support_episodes_per_user - args.boundary_episodes_per_user - args.exception_episodes_per_user
    payload = {
        "task": "Generate one complete synthetic HABIT-Bench user for planning_defaults.",
        "user_id": user_id,
        "split": split,
        "counts": {
            "sessions_total": args.sessions_per_user,
            "support_sessions": args.support_episodes_per_user,
            "boundary_sessions": args.boundary_episodes_per_user,
            "exception_sessions": args.exception_episodes_per_user,
            "distractor_sessions": distractor_count,
            "probes_total": 4,
        },
        "length_requirements": {
            "min_chars_per_session": args.min_session_chars,
            "target_chars_per_session": "1100-1800",
            "min_messages_per_session": args.min_messages_per_session,
        },
        "source_seeds": seed_payload,
        "output_schema": {
            "sessions": [
                {
                    "source_seed_id": "one seed_id from source_seeds",
                    "source_domain": "flights|hotels",
                    "signal_type": "support|boundary_counterexample|exception|distractor",
                    "messages": [{"role": "user|assistant", "content": "natural turn"}],
                    "private_evidence_summary": "one sentence",
                }
            ],
            "probes": [
                {
                    "probe_type": "direct_use|boundary|exception|explicit_retrieval",
                    "query": "public multiple-choice question",
                    "choices": [
                        {"choice_id": "A", "text": "choice text"},
                        {"choice_id": "B", "text": "choice text"},
                        {"choice_id": "C", "text": "choice text"},
                        {"choice_id": "D", "text": "choice text"},
                    ],
                    "correct_choice_id": "A|B|C|D",
                }
            ],
        },
        "probe_requirements": [
            "Exactly one probe of each type.",
            "direct_use asks for an in-scope recommendation, but does not reveal the preference in wording.",
            "boundary asks about a case outside business/client-meeting scope.",
            "exception asks about a current trip with an explicit override.",
            "explicit_retrieval asks what repeated preference is supported by history.",
            "All distractors must be plausible travel-planning choices.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_llm_user_generation(user_index: int, seeds: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    api_key = os.environ.get("HABITBENCH_API_KEY")
    if not api_key:
        raise SystemExit("Missing HABITBENCH_API_KEY")
    base_url = os.environ.get("HABITBENCH_BASE_URL", args.base_url)
    user_id = f"tm2_planning_v02_user_{user_index:04d}"
    split = split_for_user(user_index)
    payload = {
        "model": args.llm_model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": build_llm_user_prompt(user_id, split, seeds, args)},
        ],
        "temperature": args.llm_temperature,
        "max_tokens": args.llm_max_tokens,
        "reasoning_effort": "xhigh",
        "response_format": {"type": "json_object"},
    }
    response = call_curl_json(base_url, api_key, payload, timeout=args.llm_timeout_sec, retries=args.llm_retries)
    content = response["choices"][0]["message"]["content"]
    return {
        "user_id": user_id,
        "split": split,
        "model": args.llm_model,
        "reasoning_effort": "xhigh",
        "seed_ids": [seed["seed_id"] for seed in seeds],
        "raw_response": json.loads(content),
    }


def build_llm_session_prompt(user_id: str, split: str, session_index: int, signal_type: str, seed: Dict[str, Any], habit: Dict[str, str], args: argparse.Namespace) -> str:
    payload = {
        "task": "Generate one long synthetic HABIT-Bench travel-planning session.",
        "user_id": user_id,
        "split": split,
        "session_index": session_index,
        "signal_type": signal_type,
        "source_seed": {
            "seed_id": seed["seed_id"],
            "source_domain": seed.get("source_domain"),
            "prompt": seed.get("prompt"),
            "assistant_reply": seed.get("assistant_reply"),
        },
        "hidden_planning_default": habit,
        "signal_instructions": {
            "support": "Make this an in-scope travel-planning conversation where the user naturally reinforces the hidden default_action under the hidden condition.",
            "boundary_counterexample": "Make this a travel-planning conversation matching the boundary_condition, where the hidden default should not be applied.",
            "exception": "Make this a travel-planning conversation with an explicit current-trip override matching the exception_condition.",
            "distractor": "Make this a realistic travel-related conversation with no durable preference signal.",
        }[signal_type],
        "length_requirements": {
            "min_chars": args.min_session_chars,
            "target_chars": "1800-2600",
            "min_messages": args.min_messages_per_session,
        },
        "output_schema": {
            "source_seed_id": seed["seed_id"],
            "source_domain": seed.get("source_domain"),
            "signal_type": signal_type,
            "messages": [{"role": "user|assistant", "content": "natural turn"}],
            "private_evidence_summary": "one sentence explaining the signal for private review",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_llm_session_generation(user_id: str, split: str, session_index: int, signal_type: str, seed: Dict[str, Any], habit: Dict[str, str], args: argparse.Namespace) -> Dict[str, Any]:
    api_key = os.environ.get("HABITBENCH_API_KEY")
    if not api_key:
        raise SystemExit("Missing HABITBENCH_API_KEY")
    base_url = os.environ.get("HABITBENCH_BASE_URL", args.base_url)
    payload = {
        "model": args.llm_model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": build_llm_session_prompt(user_id, split, session_index, signal_type, seed, habit, args)},
        ],
        "temperature": args.llm_temperature,
        "max_tokens": args.llm_session_max_tokens,
        "reasoning_effort": "xhigh",
        "response_format": {"type": "json_object"},
    }
    response = call_curl_json(base_url, api_key, payload, timeout=args.llm_timeout_sec, retries=args.llm_retries)
    raw = json.loads(response["choices"][0]["message"]["content"])
    raw["signal_type"] = raw.get("signal_type") or signal_type
    raw["source_seed_id"] = raw.get("source_seed_id") or seed["seed_id"]
    raw["source_domain"] = raw.get("source_domain") or seed.get("source_domain")
    return raw


def build_llm_probe_prompt(user_id: str, split: str, session_summaries: Sequence[Dict[str, Any]], habit: Dict[str, str]) -> str:
    payload = {
        "task": "Generate four multiple-choice probes for one HABIT-Bench planning_defaults user.",
        "user_id": user_id,
        "split": split,
        "private_session_summaries": session_summaries,
        "hidden_planning_default": habit,
        "requirements": [
            "Return exactly one probe for each type: direct_use, boundary, exception, explicit_retrieval.",
            "Each probe has query, choices, correct_choice_id.",
            "Each choices list has exactly A/B/C/D.",
            "Distractors must be plausible travel-planning tradeoffs.",
            "Do not use public meta words: habit, benchmark, gold, label, evidence, probe, dataset, annotation.",
        ],
        "output_schema": {
            "probes": [
                {
                    "probe_type": "direct_use|boundary|exception|explicit_retrieval",
                    "query": "question",
                    "choices": [
                        {"choice_id": "A", "text": "choice text"},
                        {"choice_id": "B", "text": "choice text"},
                        {"choice_id": "C", "text": "choice text"},
                        {"choice_id": "D", "text": "choice text"},
                    ],
                    "correct_choice_id": "A|B|C|D",
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_llm_probe_generation(user_id: str, split: str, session_summaries: Sequence[Dict[str, Any]], habit: Dict[str, str], args: argparse.Namespace) -> Dict[str, Any]:
    api_key = os.environ.get("HABITBENCH_API_KEY")
    if not api_key:
        raise SystemExit("Missing HABITBENCH_API_KEY")
    base_url = os.environ.get("HABITBENCH_BASE_URL", args.base_url)
    payload = {
        "model": args.llm_model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": build_llm_probe_prompt(user_id, split, session_summaries, habit)},
        ],
        "temperature": 0.25,
        "max_tokens": args.llm_probe_max_tokens,
        "reasoning_effort": "xhigh",
        "response_format": {"type": "json_object"},
    }
    response = call_curl_json(base_url, api_key, payload, timeout=args.llm_timeout_sec, retries=args.llm_retries)
    return json.loads(response["choices"][0]["message"]["content"])


def build_llm_single_probe_prompt(user_id: str, split: str, probe_type: str, session_summaries: Sequence[Dict[str, Any]], habit: Dict[str, str]) -> str:
    instructions = {
        "direct_use": "Create an in-scope current travel recommendation question. The answer should apply the hidden default_action, but the wording should not reveal it without history.",
        "boundary": "Create a question matching the hidden boundary_condition. The answer should avoid carrying over the hidden default_action.",
        "exception": "Create a current-trip override question matching the hidden exception_condition. The answer should follow the explicit current constraint rather than the usual default.",
        "explicit_retrieval": "Ask what repeated planning preference is supported by the user's history.",
    }
    compact_summaries = [
        {
            "signal_type": row.get("signal_type"),
            "summary": compact_text(row.get("summary", ""), 140),
        }
        for row in session_summaries
        if row.get("signal_type") != "distractor"
    ]
    payload = {
        "task": "Generate one multiple-choice planning_defaults evaluation question.",
        "user_id": user_id,
        "split": split,
        "probe_type": probe_type,
        "private_summaries": compact_summaries,
        "hidden_planning_default": habit,
        "instruction": instructions[probe_type],
        "requirements": [
            "Return exactly one probe.",
            "The choices list must have exactly A/B/C/D.",
            "Distractors must be plausible travel-planning tradeoffs.",
            "Do not use public meta words: habit, benchmark, gold, label, evidence, probe, dataset, annotation.",
        ],
        "output_schema": {
            "probe_type": probe_type,
            "query": "question",
            "choices": [
                {"choice_id": "A", "text": "choice text"},
                {"choice_id": "B", "text": "choice text"},
                {"choice_id": "C", "text": "choice text"},
                {"choice_id": "D", "text": "choice text"},
            ],
            "correct_choice_id": "A|B|C|D",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_llm_single_probe_generation(user_id: str, split: str, probe_type: str, session_summaries: Sequence[Dict[str, Any]], habit: Dict[str, str], args: argparse.Namespace) -> Dict[str, Any]:
    api_key = os.environ.get("HABITBENCH_API_KEY")
    if not api_key:
        raise SystemExit("Missing HABITBENCH_API_KEY")
    base_url = os.environ.get("HABITBENCH_BASE_URL", args.base_url)
    payload = {
        "model": args.llm_model,
        "messages": [
            {"role": "system", "content": "Return strict JSON only. Write one robust multiple-choice benchmark question with plausible distractors."},
            {"role": "user", "content": build_llm_single_probe_prompt(user_id, split, probe_type, session_summaries, habit)},
        ],
        "temperature": 0.25,
        "max_tokens": args.llm_single_probe_max_tokens,
        "reasoning_effort": "xhigh",
        "response_format": {"type": "json_object"},
    }
    response = call_curl_json(base_url, api_key, payload, timeout=args.llm_timeout_sec, retries=args.llm_retries)
    raw = json.loads(response["choices"][0]["message"]["content"])
    raw["probe_type"] = raw.get("probe_type") or probe_type
    return raw


def llm_choice_set(rng: random.Random, choices: Sequence[Dict[str, Any]], correct_choice_id: str) -> Tuple[List[Dict[str, str]], str]:
    correct_text = None
    texts = []
    for choice in choices:
        text = compact_text(choice.get("text", ""), 700)
        if not text:
            continue
        texts.append(text)
        if str(choice.get("choice_id", "")).strip() == str(correct_choice_id).strip():
            correct_text = text
    if len(texts) != 4 or len(set(texts)) != 4 or not correct_text:
        raise ValueError("invalid LLM choices")
    rng.shuffle(texts)
    labels = ["A", "B", "C", "D"]
    out = [{"choice_id": labels[i], "text": texts[i]} for i in range(4)]
    gold = next(row["choice_id"] for row in out if row["text"] == correct_text)
    return out, gold


def normalize_llm_generation(generation: Dict[str, Any], seed_by_id: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    user_id = generation["user_id"]
    split = generation["split"]
    raw = generation["raw_response"]
    habit_id = f"{user_id}_habit_business_travel_buffer"
    graph = {
        "habit_id": habit_id,
        "user_id": user_id,
        "split": split,
        "template_id": TEMPLATE_ID,
        "family": FAMILY,
        "name": "business travel prefers protected arrival buffer",
        "condition": "business/client/onsite meeting travel planning",
        "default_action": "Prefer an arrival window with meaningful cushion before meeting-dependent commitments over tighter cheaper schedules.",
        "boundary_condition": "leisure or relaxed personal travel",
        "exception_condition": "current itinerary explicitly relaxes arrival timing or moves commitments to a later day",
        "source": "gpt-5.5_xhigh_taskmaster_seeded_synthetic_habit_graph_v02",
    }
    sessions = []
    evidence = defaultdict(list)
    start = datetime(2025, 2, 3, 9, 0, tzinfo=timezone.utc) + timedelta(days=int(user_id[-4:]) % 19)
    for idx, raw_session in enumerate(raw.get("sessions", [])):
        signal = str(raw_session.get("signal_type", "")).strip()
        if signal not in VALID_LLM_SIGNALS:
            raise ValueError(f"{user_id}: invalid signal_type {signal!r}")
        messages = normalize_llm_messages(raw_session.get("messages", []))
        if len(messages) < args.min_messages_per_session:
            raise ValueError(f"{user_id}: session {idx} has {len(messages)} messages")
        if session_chars(messages) < args.min_session_chars:
            raise ValueError(f"{user_id}: session {idx} has {session_chars(messages)} chars")
        seed_id = raw_session.get("source_seed_id")
        seed = seed_by_id.get(seed_id) or seed_by_id[generation["seed_ids"][idx % len(generation["seed_ids"])]]
        session_id = f"{user_id}_s{idx:04d}"
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "session_index": idx,
            "timestamp": (start + timedelta(days=idx * 3)).isoformat(),
            "domain": DOMAIN,
            "messages": messages,
            "source_seed": {
                "source_dataset": TASKMASTER_DATASET,
                "seed_id": seed["seed_id"],
                "domain": DOMAIN,
                "source_domain": seed.get("source_domain"),
                "original_id": seed.get("original_id"),
                "prompt_snippet": compact_text(seed.get("prompt", ""), 260),
            },
            "memory_annotations": {
                "linked_habit_ids": [habit_id] if signal != "distractor" else [],
                "signal_type": signal,
                "evidence_summary": compact_text(raw_session.get("private_evidence_summary", ""), 500),
                "generated_by": {"model": generation["model"], "reasoning_effort": generation["reasoning_effort"]},
            },
        }
        sessions.append(session)
        if signal != "distractor":
            evidence[signal].append(session_id)
    if len(sessions) != args.sessions_per_user:
        raise ValueError(f"{user_id}: got {len(sessions)} sessions, expected {args.sessions_per_user}")
    expected_counts = {
        "support": args.support_episodes_per_user,
        "boundary_counterexample": args.boundary_episodes_per_user,
        "exception": args.exception_episodes_per_user,
        "distractor": args.sessions_per_user - args.support_episodes_per_user - args.boundary_episodes_per_user - args.exception_episodes_per_user,
    }
    got_counts = dict(Counter(s["memory_annotations"]["signal_type"] for s in sessions))
    if got_counts != expected_counts:
        raise ValueError(f"{user_id}: signal counts {got_counts}, expected {expected_counts}")
    action_by_type = {
        "direct_use": "apply_scoped_habit",
        "boundary": "do_not_apply_out_of_scope",
        "exception": "apply_current_trip_constraint",
        "explicit_retrieval": "retrieve_supported_preference",
    }
    evidence_by_type = {
        "direct_use": evidence["support"],
        "boundary": evidence["boundary_counterexample"],
        "exception": evidence["exception"],
        "explicit_retrieval": evidence["support"],
    }
    probes = []
    keys = []
    seen_types = set()
    rng = random.Random(stable_hash(user_id, 12))
    for pidx, raw_probe in enumerate(raw.get("probes", [])):
        ptype = str(raw_probe.get("probe_type", "")).strip()
        if ptype not in VALID_PROBE_TYPES:
            raise ValueError(f"{user_id}: invalid probe_type {ptype!r}")
        seen_types.add(ptype)
        choices, gold = llm_choice_set(rng, raw_probe.get("choices", []), raw_probe.get("correct_choice_id", ""))
        private_id = f"{habit_id}_p{pidx:02d}_{ptype}"
        probe = {
            "probe_id": private_id,
            "user_id": user_id,
            "split": split,
            "probe_type": ptype,
            "habit_id": habit_id,
            "habit_family": FAMILY,
            "query": compact_text(raw_probe.get("query", ""), 900),
            "choices": choices,
            "validator": {"type": "choice_equals", "gold_choice_id": gold, "gold_action": action_by_type[ptype]},
            "visible_history_scope": {"user_id": user_id, "max_session_index": len(sessions) - 1},
            "metadata": {
                "template_id": TEMPLATE_ID,
                "source_dataset": TASKMASTER_DATASET,
                "source_domains": ["flights", "hotels"],
                "stress_variant": "taskmaster_seeded_gpt55_xhigh_v02",
                "horizon_sessions": len(sessions),
            },
        }
        probes.append(probe)
        keys.append(
            {
                "public_probe_id": public_probe_id(private_id),
                "probe_id": private_id,
                "user_id": user_id,
                "habit_id": habit_id,
                "habit_family": FAMILY,
                "probe_type": ptype,
                "capability_group": CAPABILITY_GROUP_BY_TYPE[ptype],
                "gold_choice_id": gold,
                "gold_action": action_by_type[ptype],
                "gold_evidence_session_ids": evidence_by_type[ptype],
                "hidden_habit_graph": graph,
                "review_status": "taskmaster_planning_defaults_v02_gpt55_xhigh_needs_human_review",
                "source_public_probe_id": None,
                "stress_variant": "taskmaster_seeded_gpt55_xhigh_v02",
            }
        )
    if seen_types != VALID_PROBE_TYPES:
        raise ValueError(f"{user_id}: probe types {seen_types}, expected {VALID_PROBE_TYPES}")
    return sessions, graph, probes, keys


def generate_llm_dataset(args: argparse.Namespace, seeds: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(args.seed)
    seed_by_id = {seed["seed_id"]: seed for seed in seeds}
    by_domain = defaultdict(list)
    for seed in seeds:
        by_domain[seed.get("source_domain", "travel")].append(seed)
    for rows in by_domain.values():
        rng.shuffle(rows)
    sessions: List[Dict[str, Any]] = []
    graphs: List[Dict[str, Any]] = []
    probes: List[Dict[str, Any]] = []
    keys: List[Dict[str, Any]] = []
    raw_generations: List[Dict[str, Any]] = []
    raw_path = args.out_dir / "data" / "llm_raw_generations_gpt55_xhigh.jsonl"
    existing = {}
    if args.resume_llm_generation and raw_path.exists():
        for row in read_jsonl(raw_path):
            existing[row["raw_id"]] = row
    for user_index in range(args.n_users):
        user_id = f"tm2_planning_v02_user_{user_index:04d}"
        split = split_for_user(user_index)
        habit = habit_for_user(user_index)
        habit_id = f"{user_id}_habit_{habit['template_id']}"
        graph = {
            "habit_id": habit_id,
            "user_id": user_id,
            "split": split,
            "template_id": habit["template_id"],
            "family": FAMILY,
            "name": habit["name"],
            "condition": habit["condition"],
            "default_action": habit["default_action"],
            "boundary_condition": habit["boundary_condition"],
            "exception_condition": habit["exception_condition"],
            "source": "gpt-5.5_xhigh_taskmaster_seeded_sessionwise_multihabit_graph_v02",
        }
        graphs.append(graph)
        signal_plan = (
            ["support"] * args.support_episodes_per_user
            + ["boundary_counterexample"] * args.boundary_episodes_per_user
            + ["exception"] * args.exception_episodes_per_user
            + ["distractor"] * (args.sessions_per_user - args.support_episodes_per_user - args.boundary_episodes_per_user - args.exception_episodes_per_user)
        )
        rng.shuffle(signal_plan)
        user_sessions: List[Dict[str, Any]] = []
        evidence = defaultdict(list)
        start = datetime(2025, 2, 3, 9, 0, tzinfo=timezone.utc) + timedelta(days=user_index % 19)
        for session_index, signal_type in enumerate(signal_plan):
            domain = "flights" if session_index % 2 == 0 else "hotels"
            pool = by_domain.get(domain) or seeds
            seed = pool[(user_index * args.sessions_per_user + session_index) % len(pool)]
            raw_id = f"{user_id}_session_{session_index:04d}"
            raw_row = existing.get(raw_id)
            if raw_row is None:
                raw_session = call_llm_session_generation(user_id, split, session_index, signal_type, seed, habit, args)
                raw_row = {
                    "raw_id": raw_id,
                    "kind": "session",
                    "user_id": user_id,
                    "session_index": session_index,
                    "habit_template_id": habit["template_id"],
                    "model": args.llm_model,
                    "reasoning_effort": "xhigh",
                    "seed_id": seed["seed_id"],
                    "raw_response": raw_session,
                }
                raw_generations.append(raw_row)
                write_jsonl(raw_path, raw_generations)
            else:
                raw_session = raw_row["raw_response"]
                raw_generations.append(raw_row)
            signal = str(raw_session.get("signal_type") or signal_type).strip()
            if signal != signal_type and signal_type != "distractor":
                signal = signal_type
            if signal not in VALID_LLM_SIGNALS:
                raise ValueError(f"{raw_id}: invalid signal_type {signal!r}")
            messages = normalize_llm_messages(raw_session.get("messages", []))
            if len(messages) < args.min_messages_per_session:
                raise ValueError(f"{raw_id}: only {len(messages)} messages")
            if session_chars(messages) < args.min_session_chars:
                raise ValueError(f"{raw_id}: only {session_chars(messages)} chars")
            session_id = f"{user_id}_s{session_index:04d}"
            session = {
                "session_id": session_id,
                "user_id": user_id,
                "session_index": session_index,
                "timestamp": (start + timedelta(days=session_index * 3)).isoformat(),
                "domain": DOMAIN,
                "messages": messages,
                "source_seed": {
                    "source_dataset": TASKMASTER_DATASET,
                    "seed_id": seed["seed_id"],
                    "domain": DOMAIN,
                    "source_domain": seed.get("source_domain"),
                    "original_id": seed.get("original_id"),
                    "prompt_snippet": compact_text(seed.get("prompt", ""), 260),
                },
                "memory_annotations": {
                    "linked_habit_ids": [habit_id] if signal != "distractor" else [],
                    "signal_type": signal,
                    "evidence_summary": compact_text(raw_session.get("private_evidence_summary", ""), 500),
                    "generated_by": {"model": args.llm_model, "reasoning_effort": "xhigh"},
                },
            }
            user_sessions.append(session)
            if signal != "distractor":
                evidence[signal].append(session_id)
        sessions.extend(user_sessions)
        summaries = [
            {
                "session_id": s["session_id"],
                "signal_type": s["memory_annotations"]["signal_type"],
                "summary": s["memory_annotations"].get("evidence_summary", ""),
                "first_user_turn": next((m["content"] for m in s["messages"] if m["role"] == "user"), "")[:220],
            }
            for s in user_sessions
        ]
        raw_probe_items = []
        for probe_type in ["direct_use", "boundary", "exception", "explicit_retrieval"]:
            probe_raw_id = f"{user_id}_probe_{probe_type}"
            probe_raw_row = existing.get(probe_raw_id)
            if probe_raw_row is None:
                raw_probe = call_llm_single_probe_generation(user_id, split, probe_type, summaries, habit, args)
                probe_raw_row = {
                    "raw_id": probe_raw_id,
                    "kind": "probe",
                    "probe_type": probe_type,
                    "user_id": user_id,
                    "habit_template_id": habit["template_id"],
                    "model": args.llm_model,
                    "reasoning_effort": "xhigh",
                    "raw_response": raw_probe,
                }
                raw_generations.append(probe_raw_row)
                write_jsonl(raw_path, raw_generations)
            else:
                raw_probe = probe_raw_row["raw_response"]
                raw_generations.append(probe_raw_row)
            raw_probe_items.append(raw_probe)
        action_by_type = {
            "direct_use": "apply_scoped_habit",
            "boundary": "do_not_apply_out_of_scope",
            "exception": "apply_current_trip_constraint",
            "explicit_retrieval": "retrieve_supported_preference",
        }
        evidence_by_type = {
            "direct_use": evidence["support"],
            "boundary": evidence["boundary_counterexample"],
            "exception": evidence["exception"],
            "explicit_retrieval": evidence["support"],
        }
        seen_types = set()
        for pidx, raw_probe in enumerate(raw_probe_items):
            ptype = str(raw_probe.get("probe_type", "")).strip()
            if ptype not in VALID_PROBE_TYPES:
                raise ValueError(f"{user_id}: invalid probe_type {ptype!r}")
            seen_types.add(ptype)
            private_id = f"{habit_id}_p{pidx:02d}_{ptype}"
            choices, gold = llm_choice_set(rng, raw_probe.get("choices", []), raw_probe.get("correct_choice_id", ""))
            probe = {
                "probe_id": private_id,
                "user_id": user_id,
                "split": split,
                "probe_type": ptype,
                "habit_id": habit_id,
                "habit_family": FAMILY,
                "query": compact_text(raw_probe.get("query", ""), 900),
                "choices": choices,
                "validator": {"type": "choice_equals", "gold_choice_id": gold, "gold_action": action_by_type[ptype]},
                "visible_history_scope": {"user_id": user_id, "max_session_index": len(user_sessions) - 1},
                "metadata": {
                    "template_id": habit["template_id"],
                    "source_dataset": TASKMASTER_DATASET,
                    "source_domains": ["flights", "hotels"],
                    "stress_variant": "taskmaster_seeded_gpt55_xhigh_sessionwise_multihabit_v02",
                    "horizon_sessions": len(user_sessions),
                },
            }
            probes.append(probe)
            keys.append(
                {
                    "public_probe_id": public_probe_id(private_id),
                    "probe_id": private_id,
                    "user_id": user_id,
                    "habit_id": habit_id,
                    "habit_family": FAMILY,
                    "probe_type": ptype,
                    "capability_group": CAPABILITY_GROUP_BY_TYPE[ptype],
                    "gold_choice_id": gold,
                    "gold_action": action_by_type[ptype],
                    "gold_evidence_session_ids": evidence_by_type[ptype],
                    "hidden_habit_graph": graph,
                    "review_status": "taskmaster_planning_defaults_v02_gpt55_xhigh_needs_human_review",
                    "source_public_probe_id": None,
                    "stress_variant": "taskmaster_seeded_gpt55_xhigh_sessionwise_multihabit_v02",
                }
            )
        if seen_types != VALID_PROBE_TYPES:
            raise ValueError(f"{user_id}: probe types {seen_types}, expected {VALID_PROBE_TYPES}")
        write_jsonl(raw_path, raw_generations)
        print(json.dumps({"llm_generated_users": user_index + 1, "user_id": user_id}, ensure_ascii=False), flush=True)
    return sessions, graphs, probes, keys, raw_generations


def generate_hybrid_llm_dataset(args: argparse.Namespace, seeds: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(args.seed)
    seed_by_id = {seed["seed_id"]: seed for seed in seeds}
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        by_domain[seed.get("source_domain", "travel")].append(seed)
    for rows in by_domain.values():
        rng.shuffle(rows)

    def pick_seed(source_domain: str | None = None, offset: int = 0) -> Dict[str, Any]:
        if source_domain:
            pool = by_domain.get(source_domain) or seeds
        else:
            pool = seeds
        return pool[offset % len(pool)]

    sessions: List[Dict[str, Any]] = []
    graphs: List[Dict[str, Any]] = []
    probes: List[Dict[str, Any]] = []
    keys: List[Dict[str, Any]] = []
    raw_generations: List[Dict[str, Any]] = []
    raw_path = args.out_dir / "data" / "llm_raw_generations_gpt55_xhigh.jsonl"
    existing = {}
    if args.resume_llm_generation and raw_path.exists():
        for row in read_jsonl(raw_path):
            existing[row["raw_id"]] = row

    for user_index in range(args.n_users):
        user_id = f"tm2_planning_v02_user_{user_index:04d}"
        split = split_for_user(user_index)
        habit = habit_for_user(user_index)
        habit_id = f"{user_id}_habit_{habit['template_id']}"
        graph = {
            "habit_id": habit_id,
            "user_id": user_id,
            "split": split,
            "template_id": habit["template_id"],
            "family": FAMILY,
            "name": habit["name"],
            "condition": habit["condition"],
            "default_action": habit["default_action"],
            "boundary_condition": habit["boundary_condition"],
            "exception_condition": habit["exception_condition"],
            "source": "gpt-5.5_xhigh_taskmaster_seeded_signal_generation_with_taskmaster_background_multihabit_v02",
        }
        graphs.append(graph)
        user_sessions: List[Dict[str, Any]] = []
        llm_signal_plan = (
            ["support"] * args.support_episodes_per_user
            + ["boundary_counterexample"] * args.boundary_episodes_per_user
            + ["exception"] * args.exception_episodes_per_user
            + ["distractor"] * args.llm_distractor_episodes_per_user
        )
        rng.shuffle(llm_signal_plan)
        temp_index = 0
        for signal_type in llm_signal_plan:
            domain = "flights" if temp_index % 2 == 0 else "hotels"
            seed = pick_seed(domain, user_index * 37 + temp_index)
            raw_id = f"{user_id}_llm_session_{temp_index:04d}_{signal_type}"
            raw_row = existing.get(raw_id)
            if raw_row is None:
                raw_session = call_llm_session_generation(user_id, split, temp_index, signal_type, seed, habit, args)
                raw_row = {
                    "raw_id": raw_id,
                    "kind": "session",
                    "user_id": user_id,
                    "temp_index": temp_index,
                    "signal_type": signal_type,
                    "habit_template_id": habit["template_id"],
                    "model": args.llm_model,
                    "reasoning_effort": "xhigh",
                    "seed_id": seed["seed_id"],
                    "raw_response": raw_session,
                }
                raw_generations.append(raw_row)
                write_jsonl(raw_path, raw_generations)
            else:
                raw_session = raw_row["raw_response"]
                raw_generations.append(raw_row)
            messages = normalize_llm_messages(raw_session.get("messages", []))
            if len(messages) < args.min_messages_per_session:
                raise ValueError(f"{raw_id}: only {len(messages)} messages")
            if session_chars(messages) < args.min_session_chars:
                raise ValueError(f"{raw_id}: only {session_chars(messages)} chars")
            phase_base = {"support": 0.12, "boundary_counterexample": 0.50, "exception": 0.62, "distractor": 0.0}[signal_type]
            user_sessions.append(
                {
                    "_phase": phase_base + rng.random() * 0.35,
                    "_temp_index": temp_index,
                    "session_id": "",
                    "user_id": user_id,
                    "session_index": -1,
                    "timestamp": "",
                    "domain": DOMAIN,
                    "messages": messages,
                    "source_seed": {
                        "source_dataset": TASKMASTER_DATASET,
                        "seed_id": seed["seed_id"],
                        "domain": DOMAIN,
                        "source_domain": seed.get("source_domain"),
                        "original_id": seed.get("original_id"),
                        "prompt_snippet": compact_text(seed.get("prompt", ""), 260),
                    },
                    "memory_annotations": {
                        "linked_habit_ids": [habit_id] if signal_type != "distractor" else [],
                        "signal_type": signal_type,
                        "evidence_summary": compact_text(raw_session.get("private_evidence_summary", ""), 500),
                        "generated_by": {"model": args.llm_model, "reasoning_effort": "xhigh"},
                    },
                }
            )
            temp_index += 1

        while len(user_sessions) < args.sessions_per_user:
            seed = pick_seed(None, user_index * 101 + temp_index)
            user_sessions.append(make_long_session(user_id, temp_index, seed, None, "distractor", rng.random(), rng, args))
            temp_index += 1

        user_sessions.sort(key=lambda row: (row["_phase"], row["_temp_index"]))
        start = datetime(2025, 2, 3, 9, 0, tzinfo=timezone.utc) + timedelta(days=user_index % 19)
        evidence = defaultdict(list)
        for idx, session in enumerate(user_sessions):
            session["session_index"] = idx
            session["session_id"] = f"{user_id}_s{idx:04d}"
            session["timestamp"] = (start + timedelta(days=idx * 3 + rng.randint(0, 1))).isoformat()
            session.pop("_phase", None)
            session.pop("_temp_index", None)
            signal = session["memory_annotations"]["signal_type"]
            if signal != "distractor":
                evidence[signal].append(session["session_id"])
        sessions.extend(user_sessions)

        summaries = [
            {
                "session_id": s["session_id"],
                "signal_type": s["memory_annotations"]["signal_type"],
                "summary": s["memory_annotations"].get("evidence_summary", ""),
                "first_user_turn": next((m["content"] for m in s["messages"] if m["role"] == "user"), "")[:180],
            }
            for s in user_sessions
            if s["memory_annotations"]["signal_type"] != "distractor"
        ]
        action_by_type = {
            "direct_use": "apply_scoped_habit",
            "boundary": "do_not_apply_out_of_scope",
            "exception": "apply_current_trip_constraint",
            "explicit_retrieval": "retrieve_supported_preference",
        }
        evidence_by_type = {
            "direct_use": evidence["support"],
            "boundary": evidence["boundary_counterexample"],
            "exception": evidence["exception"],
            "explicit_retrieval": evidence["support"],
        }
        seen_types = set()
        for pidx, probe_type in enumerate(["direct_use", "boundary", "exception", "explicit_retrieval"]):
            probe_raw_id = f"{user_id}_probe_{probe_type}"
            probe_raw_row = existing.get(probe_raw_id)
            if probe_raw_row is None:
                raw_probe = call_llm_single_probe_generation(user_id, split, probe_type, summaries, habit, args)
                probe_raw_row = {
                    "raw_id": probe_raw_id,
                    "kind": "probe",
                    "probe_type": probe_type,
                    "user_id": user_id,
                    "habit_template_id": habit["template_id"],
                    "model": args.llm_model,
                    "reasoning_effort": "xhigh",
                    "raw_response": raw_probe,
                }
                raw_generations.append(probe_raw_row)
                write_jsonl(raw_path, raw_generations)
            else:
                raw_probe = probe_raw_row["raw_response"]
                raw_generations.append(probe_raw_row)
            ptype = str(raw_probe.get("probe_type", "")).strip()
            if ptype != probe_type:
                ptype = probe_type
            seen_types.add(ptype)
            private_id = f"{habit_id}_p{pidx:02d}_{ptype}"
            choices, gold = llm_choice_set(rng, raw_probe.get("choices", []), raw_probe.get("correct_choice_id", ""))
            probe = {
                "probe_id": private_id,
                "user_id": user_id,
                "split": split,
                "probe_type": ptype,
                "habit_id": habit_id,
                "habit_family": FAMILY,
                "query": compact_text(raw_probe.get("query", ""), 900),
                "choices": choices,
                "validator": {"type": "choice_equals", "gold_choice_id": gold, "gold_action": action_by_type[ptype]},
                "visible_history_scope": {"user_id": user_id, "max_session_index": len(user_sessions) - 1},
                "metadata": {
                    "template_id": habit["template_id"],
                    "source_dataset": TASKMASTER_DATASET,
                    "source_domains": ["flights", "hotels"],
                    "stress_variant": "taskmaster_seeded_gpt55_xhigh_hybrid_multihabit_v02",
                    "horizon_sessions": len(user_sessions),
                },
            }
            probes.append(probe)
            keys.append(
                {
                    "public_probe_id": public_probe_id(private_id),
                    "probe_id": private_id,
                    "user_id": user_id,
                    "habit_id": habit_id,
                    "habit_family": FAMILY,
                    "probe_type": ptype,
                    "capability_group": CAPABILITY_GROUP_BY_TYPE[ptype],
                    "gold_choice_id": gold,
                    "gold_action": action_by_type[ptype],
                    "gold_evidence_session_ids": evidence_by_type[ptype],
                    "hidden_habit_graph": graph,
                    "review_status": "taskmaster_planning_defaults_v02_gpt55_xhigh_needs_human_review",
                    "source_public_probe_id": None,
                    "stress_variant": "taskmaster_seeded_gpt55_xhigh_hybrid_multihabit_v02",
                }
            )
        if seen_types != VALID_PROBE_TYPES:
            raise ValueError(f"{user_id}: probe types {seen_types}, expected {VALID_PROBE_TYPES}")
        write_jsonl(raw_path, raw_generations)
        print(json.dumps({"llm_generated_users": user_index + 1, "user_id": user_id, "sessions_per_user": len(user_sessions)}, ensure_ascii=False), flush=True)
    return sessions, graphs, probes, keys, raw_generations


def validate(sessions: List[Dict[str, Any]], probes: List[Dict[str, Any]], keys: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    session_ids = {s["session_id"] for s in sessions}
    key_by_probe = {k["probe_id"]: k for k in keys}
    rows = []
    counts = Counter()
    for probe in probes:
        errors = []
        key = key_by_probe.get(probe["probe_id"])
        if not key:
            errors.append("missing_key")
        else:
            if key["gold_choice_id"] not in {c["choice_id"] for c in probe["choices"]}:
                errors.append("gold_missing")
            for sid in key["gold_evidence_session_ids"]:
                if sid not in session_ids:
                    errors.append(f"missing_evidence:{sid}")
        texts = [probe["query"], *[c["text"] for c in probe["choices"]]]
        leakage = any(word in " ".join(texts).lower() for word in ["habit", "benchmark", "gold", "business-travel rule", "established preference"])
        if leakage:
            errors.append("leakage_wording")
        status = "pass" if not errors else "fail"
        counts[f"probe_status_{status}"] += 1
        rows.append({"probe_id": probe["probe_id"], "public_probe_id": public_probe_id(probe["probe_id"]), "probe_type": probe["probe_type"], "status": status, "errors": errors})
    msg_counts = [len(s["messages"]) for s in sessions]
    word_counts = [session_words(s["messages"]) for s in sessions]
    char_counts = [session_chars(s["messages"]) for s in sessions]
    below_min_chars = sum(1 for n in char_counts if n < args.min_session_chars)
    below_min_messages = sum(1 for n in msg_counts if n < args.min_messages_per_session)
    summary_fields = {
        "users_total": len({s["user_id"] for s in sessions}),
        "sessions_total": len(sessions),
        "probes_total": len(probes),
        "private_keys_total": len(keys),
        "avg_messages_per_session": round(sum(msg_counts) / len(msg_counts), 3),
        "median_messages_per_session": percentile(msg_counts, 0.5),
        "p10_messages_per_session": percentile(msg_counts, 0.1),
        "avg_words_per_session": round(sum(word_counts) / len(word_counts), 3),
        "median_words_per_session": percentile(word_counts, 0.5),
        "avg_chars_per_session": round(sum(char_counts) / len(char_counts), 3),
        "median_chars_per_session": percentile(char_counts, 0.5),
        "p10_chars_per_session": percentile(char_counts, 0.1),
        "min_chars_per_session": min(char_counts),
        "max_chars_per_session": max(char_counts),
        "sessions_below_min_chars": below_min_chars,
        "sessions_below_min_messages": below_min_messages,
        "support_sessions": sum(1 for s in sessions if s["memory_annotations"]["signal_type"] == "support"),
        "boundary_sessions": sum(1 for s in sessions if s["memory_annotations"]["signal_type"] == "boundary_counterexample"),
        "exception_sessions": sum(1 for s in sessions if s["memory_annotations"]["signal_type"] == "exception"),
        "distractor_sessions": sum(1 for s in sessions if s["memory_annotations"]["signal_type"] == "distractor"),
        "length_contract": {
            "min_session_chars": args.min_session_chars,
            "min_messages_per_session": args.min_messages_per_session,
            "min_raw_session_chars": args.min_raw_session_chars,
            "max_source_messages": args.max_source_messages,
            "rationale": "Target one Taskmaster-like long dialogue per session; public datasets commonly average about 14-23 turns/dialogue.",
        },
    }
    for key, value in summary_fields.items():
        counts[key] = value
    return rows, dict(counts)


def review_rows(probes: List[Dict[str, Any]], keys: List[Dict[str, Any]], sessions: List[Dict[str, Any]], validation_rows: List[Dict[str, Any]], sample_rate: float, rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    key_by_probe = {k["probe_id"]: k for k in keys}
    session_by_id = {s["session_id"]: s for s in sessions}
    val_by_probe = {v["probe_id"]: v for v in validation_rows}
    rows = []
    for probe in probes:
        key = key_by_probe[probe["probe_id"]]
        preview = []
        for sid in key["gold_evidence_session_ids"][:3]:
            s = session_by_id[sid]
            preview.append(
                {
                    "session_id": sid,
                    "session_index": s["session_index"],
                    "signal_type": s["memory_annotations"]["signal_type"],
                    "source_domain": s["source_seed"]["source_domain"],
                    "messages_preview": " | ".join(f"{m['role']}: {compact_text(m['content'], 160)}" for m in s["messages"][:5]),
                }
            )
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
                "evidence_preview_json": json.dumps(preview, ensure_ascii=False),
                "auto_validation_status": val_by_probe.get(probe["probe_id"], {}).get("status", "missing"),
                "reviewer_decision": "",
                "reviewer_notes": "",
            }
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["probe_type"]].append(row)
    sample = []
    for group_rows in grouped.values():
        k = max(1, round(len(group_rows) * sample_rate))
        sample.extend(rng.sample(group_rows, min(k, len(group_rows))))
    sample.sort(key=lambda r: r["review_id"])
    return rows, sample


def write_reports(out_dir: Path, args: argparse.Namespace, validation_summary: Dict[str, Any], probe_bank: Dict[str, Any]) -> None:
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    release_claim = (
        "Taskmaster-2 travel-seeded gpt-5.5/xhigh synthetic longitudinal planning_defaults slice"
        if args.generation_mode == "llm"
        else "Taskmaster-2 travel-dialog-seeded, longer multi-turn synthetic longitudinal planning_defaults slice"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "taskmaster_planning_defaults_v02_auto_validated_pending_human_audit",
        "source_contract": {
            "seed_prompts": TASKMASTER_DATASET,
            "source_domains": ["flights", "hotels"],
            "habit_family": FAMILY,
            "representative_domain": DOMAIN,
            "license": "CC BY 4.0",
            "release_claim": release_claim,
        },
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "validation": validation_summary,
        "artifacts": {
            "public_lifelines": "public/lifelines.jsonl",
            "public_probes": "public/probes.jsonl",
            "private_probe_key": "private/probe_key.jsonl",
            "review_queue_all": "review/planning_defaults_review_queue_all.csv",
            "review_queue_sample": "review/planning_defaults_review_queue_sample.csv",
        },
    }
    (reports / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / "probe_bank_used.json").write_text(json.dumps(probe_bank, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Taskmaster Planning Defaults v0.2",
        "",
        "- Goal: build a longer multi-habit planning_defaults benchmark slice with stronger distractors.",
        f"- Generation mode: {args.generation_mode}.",
        f"- LLM generation: {args.llm_model} with reasoning_effort=xhigh." if args.generation_mode == "llm" else "- LLM generation: not used for sessions.",
        f"- Habit templates: {validation_summary.get('habit_templates_total', 'n/a')} global planning-default types.",
        f"- Users: {validation_summary['users_total']}",
        f"- Sessions: {validation_summary['sessions_total']}",
        f"- Probes: {validation_summary['probes_total']}",
        f"- Avg messages/session: {validation_summary['avg_messages_per_session']}",
        f"- Median messages/session: {validation_summary['median_messages_per_session']}",
        f"- Avg chars/session: {validation_summary['avg_chars_per_session']}",
        f"- Median chars/session: {validation_summary['median_chars_per_session']}",
        f"- P10 chars/session: {validation_summary['p10_chars_per_session']}",
        f"- Avg words/session: {validation_summary['avg_words_per_session']}",
        f"- Length contract: at least {validation_summary['length_contract']['min_session_chars']} chars and {validation_summary['length_contract']['min_messages_per_session']} messages per session.",
        "",
        "Primary human-review file: `review/planning_defaults_review_queue_all.csv`.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dataset_card = [
        "# Dataset Card: Taskmaster Planning Defaults v0.2",
        "",
        "## Scope",
        "",
        "- Habit family: `planning_defaults`.",
        "- Source dataset: Taskmaster-2 flights and hotels.",
        "- Purpose: evaluate whether an agent can infer and apply a user's scoped default planning preference over long travel-planning histories.",
        f"- Habit template bank: {validation_summary.get('habit_templates_total', 'n/a')} global planning-default types, assigned across users.",
        f"- Generation mode: `{args.generation_mode}`.",
        f"- Session/probe generator: `{args.llm_model}` with `reasoning_effort=xhigh`." if args.generation_mode == "llm" else "- Session/probe generator: deterministic template code.",
        "",
        "## Size",
        "",
        f"- Users: {validation_summary['users_total']}.",
        f"- Sessions: {validation_summary['sessions_total']}.",
        f"- Probes: {validation_summary['probes_total']} total, with one `direct_use`, one `boundary`, one `exception`, and one `explicit_retrieval` probe per user.",
        f"- Support sessions: {validation_summary['support_sessions']}.",
        f"- Boundary sessions: {validation_summary['boundary_sessions']}.",
        f"- Exception sessions: {validation_summary['exception_sessions']}.",
        f"- Distractor sessions: {validation_summary['distractor_sessions']}.",
        "",
        "## Length Contract",
        "",
        f"- Minimum chars/session: {validation_summary['length_contract']['min_session_chars']}.",
        f"- Minimum messages/session: {validation_summary['length_contract']['min_messages_per_session']}.",
        f"- Average chars/session: {validation_summary['avg_chars_per_session']}.",
        f"- Median chars/session: {validation_summary['median_chars_per_session']}.",
        f"- P10 chars/session: {validation_summary['p10_chars_per_session']}.",
        f"- Average messages/session: {validation_summary['avg_messages_per_session']}.",
        f"- Median messages/session: {validation_summary['median_messages_per_session']}.",
        "",
        "## Files",
        "",
        "- `public/lifelines.jsonl`: public session histories without hidden habit annotations.",
        "- `public/probes.jsonl`: public multiple-choice probes.",
        "- `private/sessions_with_annotations.jsonl`: private sessions with signal annotations.",
        "- `private/habit_graphs.jsonl`: hidden habit graphs.",
        "- `private/probe_key.jsonl`: gold answers and evidence session IDs.",
        "- `review/planning_defaults_review_queue_all.csv`: full human-review queue.",
        "- `reports/auto_validation_summary.json`: validation and length statistics.",
        "",
        "## Evaluation Note",
        "",
        "The public probe format remains `choice_equals`, but reporting should not rely only on aggregate accuracy. Because this slice tests scoped preference use, metrics should also be broken down by probe type and should track boundary/exception failures separately from direct-use success.",
    ]
    (out_dir / "DATASET_CARD.md").write_text("\n".join(dataset_card) + "\n", encoding="utf-8")
    construction_note = [
        "# Construction Note",
        "",
        "v0.2 revises v0.1 after review found many short-session artifacts and answer leakage.",
        "",
        "## Main Changes From v0.1",
        "",
        "- Sessions and probes are generated with `gpt-5.5` `reasoning_effort=xhigh` from Taskmaster-2 flights/hotels seed scenarios." if args.generation_mode == "llm" else "- Sessions are raw-backed by original Taskmaster-2 flights/hotels conversations instead of short prompt snippets.",
        f"- The final LLM-generated slice uses a {validation_summary.get('habit_templates_total', 'n/a')}-template planning-default bank instead of a single repeated hidden preference.",
        f"- Each generated session must pass a length floor of {validation_summary['length_contract']['min_session_chars']} characters and {validation_summary['length_contract']['min_messages_per_session']} messages.",
        "- Probe wording avoids obvious phrases such as `habit`, `gold`, and `established preference`.",
        "- Distractor choices are written as plausible travel-planning tradeoffs rather than obvious nonanswers.",
        "- The directory keeps the same public/private/review/reports split used by the reference runs.",
        "",
        "## Generation Recipe",
        "",
        "1. Start from the v0.1 filtered Taskmaster travel seeds.",
        "2. Provide balanced flights/hotels seed scenarios to the generator.",
        "3. Generate each synthetic user's support, boundary, exception, and distractor sessions.",
        "4. Generate one probe of each type per user.",
        "5. Normalize, validate length/choice/evidence contracts, then write public/private/review artifacts.",
        "",
        "## Known Limitation",
        "",
        "The habit evidence is synthetic and model-generated from Taskmaster seed scenarios. Human review should still check whether each probe is answerable only after reading the user's history and whether the distractors are sufficiently competitive.",
    ]
    (out_dir / "CONSTRUCTION_NOTE.md").write_text("\n".join(construction_note) + "\n", encoding="utf-8")
    review_note = [
        "# Human Review Guidelines",
        "",
        "Use `review/planning_defaults_review_queue_all.csv` for full review.",
        "",
        "## Decision Labels",
        "",
        "- `accept`: the probe is answerable, the gold choice is correct, evidence is sufficient, and distractors are plausible.",
        "- `revise`: the idea is usable but wording, evidence strength, or distractors need edits.",
        "- `reject`: the probe is not usable for this habit family or cannot be repaired locally.",
        "",
        "## What To Check",
        "",
        "- The query should not reveal the answer without reading history, except `explicit_retrieval`, which intentionally asks for the remembered preference.",
        "- `boundary` probes should reward not applying the work-trip default to leisure or relaxed travel.",
        "- `exception` probes should reward following the current trip's explicit constraint.",
        "- Distractors should be plausible travel recommendations, not absurd or format-only answers.",
        "- Evidence sessions listed in `evidence_preview_json` should support the proposed gold action.",
    ]
    (out_dir / "HUMAN_REVIEW_GUIDELINES.md").write_text("\n".join(review_note) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = read_jsonl(args.seed_jsonl)
    if len(seeds) < 50:
        raise SystemExit(f"not enough seeds: {len(seeds)}")
    raw_generations: List[Dict[str, Any]] = []
    if args.generation_mode == "llm":
        sessions, graphs, probes, keys, raw_generations = generate_llm_dataset(args, seeds)
        probe_bank = {
            "generation_mode": "gpt-5.5_xhigh_sessionwise_multihabit_full_generation",
            "habit_templates": HABIT_TEMPLATES,
        }
    else:
        seeds = hydrate_seeds_with_raw_dialogs(seeds, args.raw_taskmaster_dir)
        if len(seeds) < 50:
            raise SystemExit(f"not enough raw-backed seeds after hydration: {len(seeds)}")
        probe_bank = generate_probe_bank_with_llm(args, out_dir) if args.use_llm_probe_bank else DEFAULT_PROBE_BANK
        sessions, graphs, probes, keys = generate(args, seeds, probe_bank)
    validation_rows, validation_summary = validate(sessions, probes, keys, args)
    validation_summary["generation_mode"] = args.generation_mode
    habit_template_ids = sorted({g.get("template_id") for g in graphs if g.get("template_id")})
    validation_summary["habit_templates_total"] = len(habit_template_ids)
    validation_summary["habit_template_ids"] = habit_template_ids
    validation_summary["habit_template_user_counts"] = dict(Counter(g.get("template_id") for g in graphs))
    if raw_generations:
        validation_summary["llm_generation"] = {
            "model": args.llm_model,
            "reasoning_effort": "xhigh",
            "raw_generations": len(raw_generations),
        }
    if validation_summary["sessions_below_min_chars"] or validation_summary["sessions_below_min_messages"]:
        raise SystemExit(
            "length contract failed: "
            f"{validation_summary['sessions_below_min_chars']} sessions below min chars; "
            f"{validation_summary['sessions_below_min_messages']} sessions below min messages"
        )
    public_sessions = [make_public_session(s) for s in sessions]
    public_probes = [public_probe(p) for p in probes]
    review_all, review_sample = review_rows(probes, keys, sessions, validation_rows, args.review_sample_rate, rng)
    if args.generation_mode == "template":
        write_jsonl(out_dir / "data" / "filtered_taskmaster_travel_seeds_raw_backed.jsonl", [seed_export_row(s) for s in seeds])
    write_jsonl(out_dir / "public" / "lifelines.jsonl", public_sessions)
    write_jsonl(out_dir / "public" / "probes.jsonl", public_probes)
    write_jsonl(out_dir / "private" / "sessions_with_annotations.jsonl", sessions)
    write_jsonl(out_dir / "private" / "habit_graphs.jsonl", graphs)
    write_jsonl(out_dir / "private" / "probe_key.jsonl", keys)
    write_csv(out_dir / "review" / "planning_defaults_review_queue_all.csv", review_all)
    write_jsonl(out_dir / "review" / "planning_defaults_review_queue_all.jsonl", review_all)
    write_csv(out_dir / "review" / "planning_defaults_review_queue_sample.csv", review_sample)
    write_jsonl(out_dir / "review" / "planning_defaults_review_queue_sample.jsonl", review_sample)
    write_jsonl(out_dir / "reports" / "auto_validation_rows.jsonl", validation_rows)
    (out_dir / "reports" / "auto_validation_summary.json").write_text(json.dumps(validation_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(out_dir, args, validation_summary, probe_bank)
    print(json.dumps({"out_dir": str(out_dir), **validation_summary}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    root = Path("/mnt/petrelfs/linzhouhan/xqwang/project/habit-bench")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-jsonl", type=Path, default=root / "runs_wxq/taskmaster_planning_defaults_v0_1/data/filtered_taskmaster_travel_seeds.jsonl")
    parser.add_argument("--raw-taskmaster-dir", type=Path, default=root / "runs_wxq/taskmaster_planning_defaults_v0_1/data/raw_taskmaster")
    parser.add_argument("--out-dir", type=Path, default=root / "runs_wxq/taskmaster_planning_defaults_v0_2")
    parser.add_argument("--generation-mode", choices=["llm", "template"], default="llm")
    parser.add_argument("--n-users", type=int, default=30)
    parser.add_argument("--sessions-per-user", type=int, default=36)
    parser.add_argument("--support-episodes-per-user", type=int, default=5)
    parser.add_argument("--boundary-episodes-per-user", type=int, default=2)
    parser.add_argument("--exception-episodes-per-user", type=int, default=1)
    parser.add_argument("--llm-distractor-episodes-per-user", type=int, default=2)
    parser.add_argument("--min-raw-session-chars", type=int, default=850)
    parser.add_argument("--min-session-chars", type=int, default=1500)
    parser.add_argument("--min-messages-per-session", type=int, default=12)
    parser.add_argument("--max-source-messages", type=int, default=32)
    parser.add_argument("--review-sample-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--use-llm-probe-bank", action="store_true")
    parser.add_argument("--refresh-llm-probe-bank", action="store_true")
    parser.add_argument("--base-url", default="https://queqiao.online/v1")
    parser.add_argument("--llm-model", default="gpt-5.5")
    parser.add_argument("--llm-temperature", type=float, default=0.45)
    parser.add_argument("--llm-max-tokens", type=int, default=14000)
    parser.add_argument("--llm-session-max-tokens", type=int, default=4500)
    parser.add_argument("--llm-probe-max-tokens", type=int, default=3200)
    parser.add_argument("--llm-single-probe-max-tokens", type=int, default=650)
    parser.add_argument("--llm-timeout-sec", type=int, default=900)
    parser.add_argument("--llm-retries", type=int, default=1)
    parser.add_argument("--llm-seeds-per-user", type=int, default=8)
    parser.add_argument("--resume-llm-generation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
