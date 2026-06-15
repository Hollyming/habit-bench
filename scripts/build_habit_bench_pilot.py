#!/usr/bin/env python
"""Build a HABIT-Bench pilot package up to the pre-human-review stage.

The generated benchmark is intentionally split into public, private, review,
and report artifacts. Public files are safe for baseline systems. Private files
contain hidden habit graphs, gold labels, and evidence ids. Review files are
the handoff point for human audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import jsonschema
import pandas as pd
import requests


HF_WILDCHAT_PARQUET = (
    "https://huggingface.co/datasets/allenai/WildChat/resolve/main/"
    "data/train-00000-of-00006.parquet"
)

ROOT = Path(__file__).resolve().parents[1]
PROBE_SCHEMA_PATH = ROOT / "schema" / "probe.schema.json"
SESSION_SCHEMA_PATH = ROOT / "schema" / "session.schema.json"

PII_PATTERNS = [
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"\b(?:\+?\d[\d .()\-]{7,}\d)\b"),
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"https?://\S+", re.I),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
]

CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")


HABIT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "template_id": "style_three_bullets",
        "family": "format_style",
        "name": "work updates in three crisp bullets",
        "domain": "work",
        "condition": "work status updates, planning notes, and operational summaries",
        "default_action": "Answer in exactly three concise bullets with no preamble.",
        "wrong_action": "Give a long-form explanation with background paragraphs.",
        "boundary_condition": "creative brainstorming or exploratory writing",
        "boundary_action": "Use a richer exploratory response; do not force three bullets.",
        "exception_condition": "the user asks for a deep dive or a narrative draft",
        "exception_action": "Provide a longer structured answer because this is a deep-dive request.",
        "support_user": [
            "Can you turn this project update into something I can scan before standup?",
            "I need a quick work status note from this. Keep it easy to skim.",
            "Can you condense this operational update for my manager?",
        ],
        "support_assistant": "Here are three concise bullets: impact, blocker, next step.",
        "support_feedback": [
            "Perfect, this is the kind of scan-friendly format I use.",
            "This is much easier for standup, thanks.",
            "Great, three bullets is exactly manageable here.",
        ],
        "boundary_query": "I am exploring campaign ideas and want a playful brainstorm with lots of angles.",
        "direct_query": "Prepare a status update for a work sync using the user's established style.",
        "exception_query": "Write a deep-dive narrative explaining why the roadmap slipped this quarter.",
    },
    {
        "template_id": "code_risks_first",
        "family": "coding_review",
        "name": "code review starts with risks and minimal patch",
        "domain": "code",
        "condition": "code review, pull request review, debugging handoff",
        "default_action": "Start with concrete risks, then propose the smallest patch.",
        "wrong_action": "Start with general praise and broad refactoring suggestions.",
        "boundary_condition": "onboarding tutorials or conceptual explanations",
        "boundary_action": "Teach the concept patiently; do not use a terse risk-first review.",
        "exception_condition": "the user asks for tutorial-style onboarding",
        "exception_action": "Explain the concept step by step before any patch.",
        "support_user": [
            "Review this change and tell me what could break first.",
            "Can you look at this patch? I mainly need the risks and the smallest fix.",
            "Please inspect this bug report like a code review.",
        ],
        "support_assistant": "Risk first: the edge case fails when the input is empty. Minimal patch: guard before parsing.",
        "support_feedback": [
            "Good, risks first saves me time.",
            "That ordering works for my reviews.",
            "Exactly: risk first, then patch.",
        ],
        "boundary_query": "I am onboarding a teammate. Explain what dependency injection means from first principles.",
        "direct_query": "Review a pull request for the user and adapt to their established review style.",
        "exception_query": "Teach a junior teammate how this authentication middleware works.",
    },
    {
        "template_id": "business_travel_early_buffer",
        "family": "planning_defaults",
        "name": "business travel prefers early flights and buffer time",
        "domain": "travel",
        "condition": "business travel planning and meeting travel",
        "default_action": "Prefer early flights and leave a 90-minute buffer before meetings.",
        "wrong_action": "Optimize only for cheapest arrival and tight transfers.",
        "boundary_condition": "leisure trips or flexible vacations",
        "boundary_action": "Optimize for comfort and preference discovery instead of early business timing.",
        "exception_condition": "leisure travel or a trip explicitly marked flexible",
        "exception_action": "Ask about leisure priorities; do not assume early flights.",
        "support_user": [
            "Find a flight option for a client meeting; I need enough breathing room before it starts.",
            "Can you plan the outbound leg so I am not rushing into the meeting?",
            "For this work trip, choose timing that keeps the morning predictable.",
        ],
        "support_assistant": "I would choose the early arrival and block a 90-minute buffer before the meeting.",
        "support_feedback": [
            "Yes, I hate rushing straight into work meetings.",
            "The buffer is what I needed.",
            "Good call on the early arrival for business travel.",
        ],
        "boundary_query": "Plan a relaxed weekend trip where the user can sleep in and wander.",
        "direct_query": "Pick the default travel option for a client meeting next Thursday.",
        "exception_query": "Plan a flexible vacation day with no fixed appointment.",
    },
    {
        "template_id": "weekday_family_vegetarian",
        "family": "content_constraints",
        "name": "weekday family meals are vegetarian",
        "domain": "food",
        "condition": "weekday family meal planning",
        "default_action": "Suggest vegetarian weekday family meals.",
        "wrong_action": "Suggest meat-centered weekday dinners.",
        "boundary_condition": "birthday dinners, travel meals, or restaurant exploration",
        "boundary_action": "Do not assume vegetarian-only unless the user says so.",
        "exception_condition": "birthday, travel, or special-occasion meal",
        "exception_action": "Offer flexible options and ask whether the usual weekday constraint applies.",
        "support_user": [
            "Plan a quick Tuesday dinner for the family.",
            "Need weekday meals the kids will actually eat.",
            "Can you make a grocery list for Wednesday dinner?",
        ],
        "support_assistant": "A vegetarian sheet-pan dinner with chickpeas, potatoes, and yogurt sauce should fit.",
        "support_feedback": [
            "Good, weekday vegetarian keeps everyone happy.",
            "This fits our weeknight routine.",
            "Nice, vegetarian is the right default for school nights.",
        ],
        "boundary_query": "Pick restaurants for a birthday dinner while the family is traveling.",
        "direct_query": "Suggest a default Wednesday family dinner.",
        "exception_query": "Plan a birthday dinner during travel where the usual home routine may not apply.",
    },
    {
        "template_id": "fresh_sources_for_stakes",
        "family": "tool_action",
        "name": "freshness check for high-stakes or current topics",
        "domain": "news",
        "condition": "finance, legal, medical, regulations, or current-news tasks",
        "default_action": "Check or cite fresh sources before giving a concrete recommendation.",
        "wrong_action": "Answer from memory without noting freshness.",
        "boundary_condition": "evergreen concepts or historical explanations",
        "boundary_action": "Explain normally; a freshness check is optional.",
        "exception_condition": "the user asks for a timeless conceptual explanation",
        "exception_action": "Give the explanation without pretending a live lookup is required.",
        "support_user": [
            "Before I act on this finance question, make sure the info is current.",
            "Can you check whether this regulation changed recently before summarizing?",
            "I need the current answer, not a stale one.",
        ],
        "support_assistant": "I will verify recency first, then separate current facts from general guidance.",
        "support_feedback": [
            "Exactly, freshness matters for this kind of thing.",
            "Thanks for checking before advising.",
            "Good, I do not want stale answers here.",
        ],
        "boundary_query": "Explain the difference between simple and compound interest for a student.",
        "direct_query": "Answer a current regulatory question for the user.",
        "exception_query": "Give a timeless conceptual explanation of compound interest.",
    },
    {
        "template_id": "confirm_before_commit",
        "family": "risk_threshold",
        "name": "confirm before bookings, sends, or irreversible actions",
        "domain": "commitment",
        "condition": "actions that book, send, delete, submit, or otherwise commit user resources",
        "default_action": "Draft or recommend, but ask for confirmation before committing.",
        "wrong_action": "Proceed as if the agent may execute the booking, submission, deletion, or send action.",
        "boundary_condition": "low-risk drafting, comparison, planning, or brainstorming",
        "boundary_action": "Provide the draft/recommendation without unnecessary confirmation friction.",
        "exception_condition": "the user only asks for a comparison or draft",
        "exception_action": "Return the comparison or draft; do not ask for confirmation yet.",
        "support_user": [
            "Compare these appointment slots, but do not book anything without checking with me.",
            "Draft the reply first. I want to approve before it is sent.",
            "Prepare the submission, then pause before committing it.",
        ],
        "support_assistant": "I will recommend the option and wait for explicit confirmation before taking action.",
        "support_feedback": [
            "Good, I always want the final say before committing.",
            "Thanks for pausing before action.",
            "That confirmation step is important to me.",
        ],
        "boundary_query": "Compare three possible appointment windows and summarize the tradeoffs.",
        "direct_query": "The user asks the assistant to handle a booking-like or submission-like task after comparing options.",
        "exception_query": "Draft an email but do not send it.",
    },
    {
        "template_id": "monday_meeting_decisions",
        "family": "meeting_prep",
        "name": "Monday meeting prep emphasizes decisions, blockers, next actions",
        "domain": "meeting",
        "condition": "recurring Monday team meeting prep",
        "default_action": "Organize notes into decisions, blockers, and next actions.",
        "wrong_action": "Write a chronological narrative summary.",
        "boundary_condition": "casual check-ins and non-recurring chats",
        "boundary_action": "Keep it conversational; do not impose the Monday meeting template.",
        "exception_condition": "a casual check-in or social catch-up",
        "exception_action": "Use a light conversational summary.",
        "support_user": [
            "Turn this into Monday prep for the team.",
            "I need the usual Monday meeting shape from these notes.",
            "Can you make this useful for the Monday sync?",
        ],
        "support_assistant": "Decisions: ... Blockers: ... Next actions: ...",
        "support_feedback": [
            "This is the structure I need every Monday.",
            "Good, decisions/blockers/actions is the useful split.",
            "That meeting shape works.",
        ],
        "boundary_query": "Summarize a casual one-on-one check-in with a colleague.",
        "direct_query": "Prepare notes for the user's recurring Monday team meeting.",
        "exception_query": "Summarize an informal coffee chat with no action items.",
    },
    {
        "template_id": "budget_drift_from_low_to_quality",
        "family": "drift_seasonality",
        "name": "equipment preference drifts from cheapest to durable quality",
        "domain": "equipment",
        "condition": "work equipment and durable gear recommendations",
        "default_action": "Before the drift, prefer budget options; after the drift, prefer durable midrange quality.",
        "wrong_action": "Continue recommending the cheapest equipment after repeated quality-oriented feedback.",
        "boundary_condition": "one-time disposable supplies",
        "boundary_action": "Do not over-apply the durable-equipment preference to disposable supplies.",
        "exception_condition": "one-time disposable supplies",
        "exception_action": "Optimize for sufficient and low-cost because durability is irrelevant.",
        "drift_action": "Prefer durable midrange equipment quality over the absolute cheapest option.",
        "support_user": [
            "Help me pick the cheapest acceptable work accessory.",
            "Find a budget equipment option that gets the job done.",
            "I care most about low cost for this work gear.",
        ],
        "post_drift_user": [
            "The cheap equipment broke again; pick something durable this time.",
            "I am done replacing flimsy gear. Recommend the reliable midrange option.",
            "For work equipment now, durability matters more than the lowest price.",
        ],
        "support_assistant": "The budget option is the best fit if price is the priority.",
        "post_drift_assistant": "The durable midrange option is the better default now.",
        "support_feedback": [
            "Good, cheapest acceptable is right for now.",
            "Thanks, budget matters here.",
            "That low-cost pick works.",
        ],
        "post_drift_feedback": [
            "Yes, durability is the new priority.",
            "That is the direction I want now.",
            "Good, I would rather pay a bit more than replace it again.",
        ],
        "boundary_query": "Choose a disposable item for a one-off workshop.",
        "direct_query": "Recommend default work equipment after the user's recent purchasing pattern changed.",
        "exception_query": "Pick disposable supplies for a one-time event.",
    },
    {
        "template_id": "sensitive_one_off_no_memory",
        "family": "privacy_consent",
        "name": "sensitive one-off facts require consent before memory use",
        "domain": "privacy",
        "condition": "sensitive health, finance, identity, or family facts mentioned once",
        "default_action": "Do not store or use the sensitive inferred trait unless explicitly asked to remember it.",
        "wrong_action": "Personalize future answers using the sensitive one-off trait.",
        "boundary_condition": "non-sensitive stable workflow preferences",
        "boundary_action": "Use ordinary workflow habits when they are supported.",
        "exception_condition": "the user explicitly asks the agent to remember a sensitive preference",
        "exception_action": "Acknowledge consent and scope before using it later.",
        "support_user": [
            "For this one message, help me phrase a note about a medical appointment.",
            "I need to ask a financial hardship question once; do not make it part of my profile.",
            "Help summarize this family-health email, but I am not asking you to remember it.",
        ],
        "support_assistant": "I will help with this message and will not treat it as a durable preference.",
        "support_feedback": [
            "Thanks. This is just for today.",
            "Good, please do not turn that into a profile detail.",
            "Exactly, no need to remember it.",
        ],
        "boundary_query": "Use the user's stable code-review workflow preference on a code review task.",
        "direct_query": "A later meal-planning task mentions nothing medical; decide whether to use a past one-off health detail.",
        "exception_query": "The user explicitly says to remember a medical scheduling constraint for future appointment planning.",
    },
]


FALLBACK_SEEDS = [
    ("code", "Review this patch and point out any edge cases around empty inputs."),
    ("code", "Explain why this async job sometimes retries twice."),
    ("work", "Turn these rough project notes into a short update for the team."),
    ("work", "Summarize the blockers from this roadmap discussion."),
    ("travel", "Compare two flight options for a meeting in Chicago."),
    ("travel", "Plan a low-stress itinerary around a conference talk."),
    ("food", "Suggest a weekday dinner that can be cooked in thirty minutes."),
    ("food", "Make a grocery list for family meals this week."),
    ("news", "Summarize what changed in the new policy announcement."),
    ("news", "Explain this market headline in plain language."),
    ("commitment", "Compare appointment slots, but leave the actual booking for approval."),
    ("commitment", "Draft a message for approval before it is sent."),
    ("meeting", "Turn these notes into a meeting prep document."),
    ("meeting", "Summarize the action items from this sync."),
    ("equipment", "Compare three laptop bags for commuting durability."),
    ("equipment", "Find a replacement keyboard option for daily work."),
    ("privacy", "Help rewrite a sensitive personal note without saving details."),
    ("general", "Help me improve this message so it is clearer."),
]


@dataclass
class SeedPrompt:
    seed_id: str
    source_dataset: str
    domain: str
    prompt: str
    timestamp: Optional[str] = None


def compact_text(text: str, max_chars: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.replace("\u0000", "")
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def has_pii(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in PII_PATTERNS)


def looks_like_usable_english(text: str) -> bool:
    """Reject mislabeled English rows that are mostly symbols or non-Latin text."""
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False
    non_ascii = sum(1 for ch in compact if ord(ch) > 127)
    if non_ascii / max(len(compact), 1) > 0.12:
        return False
    if len(CJK_PATTERN.findall(text or "")) > 2:
        return False
    alpha = [ch for ch in compact if ch.isalpha()]
    if alpha:
        ascii_alpha = sum(1 for ch in alpha if "a" <= ch.lower() <= "z")
        if ascii_alpha / max(len(alpha), 1) < 0.85:
            return False
    return True


def categorize_prompt(text: str) -> str:
    lower = text.lower()
    keyword_map = [
        ("code", ["code", "python", "javascript", "bug", "pull request", "api", "function", "debug"]),
        ("privacy", ["medical", "health", "family", "sensitive", "private", "hardship", "identity"]),
        ("work", ["project", "roadmap", "manager", "standup", "status", "operations"]),
        ("travel", ["flight", "hotel", "travel", "itinerary", "trip", "conference"]),
        ("food", ["meal", "dinner", "recipe", "grocery", "restaurant", "cook"]),
        ("news", ["current", "policy", "regulation", "market", "legal", "finance", "recent", "up-to-date"]),
        ("meeting", ["meeting", "agenda", "sync", "notes", "action items", "blockers"]),
        ("equipment", ["equipment", "gear", "keyboard", "monitor", "laptop bag", "headset", "durable", "replacement"]),
        ("commitment", ["book", "booking", "send", "submit", "approve", "approval", "delete", "commit", "buy", "purchase"]),
    ]
    for domain, keywords in keyword_map:
        if any(keyword in lower for keyword in keywords):
            return domain
    return "general"


def stable_hash(value: str, n: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_first_user_prompt(conversation: Any) -> Optional[str]:
    if conversation is None:
        return None
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") == "user":
            if turn.get("toxic") is True or turn.get("redacted") is True:
                return None
            return turn.get("content")
    return None


def fallback_seed_prompts(limit: int) -> List[SeedPrompt]:
    prompts: List[SeedPrompt] = []
    i = 0
    while len(prompts) < limit:
        domain, prompt = FALLBACK_SEEDS[i % len(FALLBACK_SEEDS)]
        prompts.append(
            SeedPrompt(
                seed_id=f"fallback_{i:05d}",
                source_dataset="fallback_templates",
                domain=domain,
                prompt=prompt,
                timestamp=None,
            )
        )
        i += 1
    return prompts


def load_wildchat_seed_prompts(
    cache_path: Path,
    limit: int,
    rng: random.Random,
    refresh: bool = False,
) -> Tuple[List[SeedPrompt], Dict[str, Any]]:
    if cache_path.exists() and not refresh:
        rows = read_jsonl(cache_path)
        seeds = []
        recategorized = 0
        for row in rows[:limit]:
            original_domain = row.get("domain")
            row = dict(row)
            row["domain"] = categorize_prompt(row.get("prompt", ""))
            if row["domain"] != original_domain:
                recategorized += 1
            seeds.append(SeedPrompt(**row))
        if recategorized:
            write_jsonl(cache_path, [seed.__dict__ for seed in seeds])
        return seeds, {
            "source": "cache",
            "cache_path": str(cache_path),
            "loaded_seed_prompts": len(seeds),
            "recategorized_seed_prompts": recategorized,
        }

    started = time.time()
    df = pd.read_parquet(
        HF_WILDCHAT_PARQUET,
        columns=["conversation_id", "timestamp", "conversation", "language", "toxic", "redacted"],
    )
    accepted: List[SeedPrompt] = []
    rejected = Counter()

    for row in df.sample(frac=1.0, random_state=rng.randint(0, 10**9)).itertuples(index=False):
        if len(accepted) >= limit:
            break
        if getattr(row, "language", None) != "English":
            rejected["non_english"] += 1
            continue
        if bool(getattr(row, "toxic", False)) or bool(getattr(row, "redacted", False)):
            rejected["moderated_or_redacted"] += 1
            continue
        prompt = extract_first_user_prompt(getattr(row, "conversation", None))
        if not prompt:
            rejected["missing_prompt"] += 1
            continue
        prompt = compact_text(prompt)
        if not (30 <= len(prompt) <= 360):
            rejected["length"] += 1
            continue
        if has_pii(prompt):
            rejected["pii"] += 1
            continue
        if not looks_like_usable_english(prompt):
            rejected["not_usable_english"] += 1
            continue
        domain = categorize_prompt(prompt)
        accepted.append(
            SeedPrompt(
                seed_id=f"wildchat_{stable_hash(str(getattr(row, 'conversation_id', '')))}",
                source_dataset="allenai/WildChat",
                domain=domain,
                prompt=prompt,
                timestamp=str(getattr(row, "timestamp", "")) or None,
            )
        )

    write_jsonl(cache_path, [seed.__dict__ for seed in accepted])
    return accepted, {
        "source": "allenai/WildChat",
        "parquet_url": HF_WILDCHAT_PARQUET,
        "loaded_seed_prompts": len(accepted),
        "rejected": dict(rejected),
        "elapsed_sec": round(time.time() - started, 2),
    }


def load_seed_prompts(args: argparse.Namespace, out_dir: Path, rng: random.Random) -> Tuple[List[SeedPrompt], Dict[str, Any]]:
    raw_path = out_dir / "data" / "raw_seed_prompts.jsonl"
    if args.source in ("auto", "wildchat"):
        try:
            seeds, meta = load_wildchat_seed_prompts(
                raw_path,
                args.seed_prompts,
                rng,
                refresh=args.refresh_seeds,
            )
            if seeds:
                return seeds, meta
        except Exception as exc:  # Keep benchmark construction moving.
            if args.source == "wildchat":
                raise
            meta = {
                "source": "fallback_templates",
                "fallback_reason": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
            seeds = fallback_seed_prompts(args.seed_prompts)
            write_jsonl(raw_path, [seed.__dict__ for seed in seeds])
            meta["loaded_seed_prompts"] = len(seeds)
            return seeds, meta

    seeds = fallback_seed_prompts(args.seed_prompts)
    write_jsonl(raw_path, [seed.__dict__ for seed in seeds])
    return seeds, {"source": "fallback_templates", "loaded_seed_prompts": len(seeds)}


def maybe_naturalize_templates(
    args: argparse.Namespace,
    out_dir: Path,
    templates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not args.use_llm_template_naturalization:
        return {"enabled": False}

    api_key = args.api_key or os.environ.get("HABITBENCH_API_KEY")
    base_url = (args.base_url or os.environ.get("HABITBENCH_BASE_URL") or "").rstrip("/")
    if not api_key or not base_url:
        return {"enabled": True, "status": "skipped_missing_api_env"}

    prompt = {
        "task": "Create natural but concise user request variants for benchmark generation.",
        "constraints": [
            "Do not include personally identifying information.",
            "Do not reveal the word habit or the hidden rule.",
            "Return valid JSON only.",
        ],
        "templates": [
            {
                "template_id": t["template_id"],
                "family": t["family"],
                "condition": t["condition"],
                "direct_query": t["direct_query"],
                "boundary_query": t["boundary_query"],
                "exception_query": t["exception_query"],
            }
            for t in templates
        ],
    }
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You generate benchmark text variants as strict JSON."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.4,
        "reasoning_effort": args.reasoning_effort,
        "response_format": {"type": "json_object"},
    }

    started = time.time()
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=args.llm_timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        variants = json.loads(content)
        path = out_dir / "data" / "llm_template_variants.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(variants, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "enabled": True,
            "status": "ok",
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "elapsed_sec": round(time.time() - started, 2),
            "output_path": str(path),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "status": "failed_but_continued",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "elapsed_sec": round(time.time() - started, 2),
        }


def seed_by_domain(seeds: Sequence[SeedPrompt]) -> Dict[str, List[SeedPrompt]]:
    grouped: Dict[str, List[SeedPrompt]] = defaultdict(list)
    for seed in seeds:
        grouped[seed.domain].append(seed)
        grouped["any"].append(seed)
    return grouped


def pick_seed(grouped: Dict[str, List[SeedPrompt]], domain: str, rng: random.Random) -> SeedPrompt:
    candidates = grouped.get(domain) or grouped.get("any") or fallback_seed_prompts(1)
    return rng.choice(candidates)


def make_choice_set(
    rng: random.Random,
    correct_text: str,
    distractors: Sequence[str],
    gold_action: str,
) -> Tuple[List[Dict[str, str]], str]:
    pool = [correct_text] + list(distractors)
    labels = ["A", "B", "C", "D"]
    rng.shuffle(pool)
    choices = [{"choice_id": labels[i], "text": pool[i]} for i in range(len(pool))]
    gold_choice_id = next(choice["choice_id"] for choice in choices if choice["text"] == correct_text)
    return choices, gold_choice_id


def split_for_user(user_index: int) -> str:
    bucket = user_index % 10
    if bucket < 2:
        return "dev"
    if bucket == 9:
        return "stress"
    return "test"


def make_session(
    user_id: str,
    temp_session_index: int,
    domain: str,
    seed: SeedPrompt,
    user_request: str,
    assistant_response: str,
    feedback: Optional[str],
    linked_habit_ids: Sequence[str],
    signal_type: str,
    phase: float,
) -> Dict[str, Any]:
    messages = [
        {"role": "user", "content": compact_text(user_request, 900)},
        {"role": "assistant", "content": compact_text(assistant_response, 900)},
    ]
    if feedback:
        messages.append({"role": "user", "content": compact_text(feedback, 500)})
    return {
        "_phase": phase,
        "_temp_session_index": temp_session_index,
        "session_id": "",
        "user_id": user_id,
        "session_index": -1,
        "timestamp": "",
        "domain": domain,
        "messages": messages,
        "source_seed": {
            "source_dataset": seed.source_dataset,
            "seed_id": seed.seed_id,
            "domain": seed.domain,
            "prompt_snippet": seed.prompt,
        },
        "memory_annotations": {
            "linked_habit_ids": list(linked_habit_ids),
            "signal_type": signal_type,
        },
    }


def generate_user_package(
    user_index: int,
    args: argparse.Namespace,
    grouped_seeds: Dict[str, List[SeedPrompt]],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    user_id = f"user_{user_index:04d}"
    split = split_for_user(user_index)
    habit_count = rng.randint(args.min_habits_per_user, args.max_habits_per_user)
    templates = rng.sample(HABIT_TEMPLATES, k=habit_count)
    if user_index % 5 == 0 and not any(t["family"] == "privacy_consent" for t in templates):
        templates[-1] = next(t for t in HABIT_TEMPLATES if t["family"] == "privacy_consent")
    if user_index % 7 == 0 and not any(t["family"] == "drift_seasonality" for t in templates):
        templates[0] = next(t for t in HABIT_TEMPLATES if t["family"] == "drift_seasonality")

    sessions: List[Dict[str, Any]] = []
    habit_graphs: List[Dict[str, Any]] = []
    user_habit_ids: List[str] = []
    temp_counter = 0

    for hidx, template in enumerate(templates):
        habit_id = f"{user_id}_habit_{hidx:02d}_{template['template_id']}"
        user_habit_ids.append(habit_id)
        support_target = args.support_episodes_per_habit
        if template.get("template_id") == "budget_drift_from_low_to_quality":
            support_target = max(3, support_target - 2)
        habit_graphs.append(
            {
                "habit_id": habit_id,
                "user_id": user_id,
                "split": split,
                "template_id": template["template_id"],
                "family": template["family"],
                "name": template["name"],
                "condition": template["condition"],
                "default_action": template["default_action"],
                "boundary_condition": template["boundary_condition"],
                "exception_condition": template["exception_condition"],
                "strength": "usually",
                "support_episode_target": support_target,
                "counterevidence_policy": "boundary and exception episodes are not negative evidence against the scoped habit",
                "temporal_policy": "latest sustained evidence wins for drift templates",
                "sensitivity": "sensitive" if template["family"] == "privacy_consent" else "ordinary",
                "source": "synthetic_hidden_habit_graph_from_real_prompt_seeds",
            }
        )

        support_prompts = template["support_user"]
        for j in range(support_target):
            seed = pick_seed(grouped_seeds, template["domain"], rng)
            user_request = (
                f"{rng.choice(support_prompts)}\n"
                f"Context from a realistic prior task: {seed.prompt}"
            )
            assistant = template.get("support_assistant", template["default_action"])
            feedback = rng.choice(template.get("support_feedback", ["This works for me."]))
            sessions.append(
                make_session(
                    user_id,
                    temp_counter,
                    template["domain"],
                    seed,
                    user_request,
                    assistant,
                    feedback,
                    [habit_id],
                    "support",
                    phase=0.18 + rng.random() * 0.42,
                )
            )
            temp_counter += 1

        if template.get("post_drift_user"):
            for j in range(args.post_drift_episodes):
                seed = pick_seed(grouped_seeds, template["domain"], rng)
                sessions.append(
                    make_session(
                        user_id,
                        temp_counter,
                        template["domain"],
                        seed,
                        f"{rng.choice(template['post_drift_user'])}\nRelevant item: {seed.prompt}",
                        template["post_drift_assistant"],
                        rng.choice(template["post_drift_feedback"]),
                        [habit_id],
                        "post_drift_support",
                        phase=0.72 + rng.random() * 0.22,
                    )
                )
                temp_counter += 1

        seed = pick_seed(grouped_seeds, template["domain"], rng)
        sessions.append(
            make_session(
                user_id,
                temp_counter,
                template["domain"],
                seed,
                f"{template['boundary_query']}\nRelated material: {seed.prompt}",
                template["boundary_action"],
                "This different context should not use my usual pattern.",
                [habit_id],
                "boundary_counterexample",
                phase=0.35 + rng.random() * 0.40,
            )
        )
        temp_counter += 1

        seed = pick_seed(grouped_seeds, template["domain"], rng)
        sessions.append(
            make_session(
                user_id,
                temp_counter,
                template["domain"],
                seed,
                f"{template['exception_query']}\nUse this material if helpful: {seed.prompt}",
                template["exception_action"],
                "Yes, this is one of the exceptions.",
                [habit_id],
                "exception",
                phase=0.45 + rng.random() * 0.40,
            )
        )
        temp_counter += 1

    while len(sessions) < args.sessions_per_user:
        seed = pick_seed(
            grouped_seeds,
            rng.choice(["any", "work", "code", "commitment", "meeting", "food", "equipment"]),
            rng,
        )
        sessions.append(
            make_session(
                user_id,
                temp_counter,
                seed.domain,
                seed,
                f"Help with this standalone task: {seed.prompt}",
                "I can help with the immediate task without adding a durable user preference.",
                None,
                [],
                "distractor",
                phase=rng.random(),
            )
        )
        temp_counter += 1

    rng.shuffle(sessions)
    sessions.sort(key=lambda row: (row["_phase"], row["_temp_session_index"]))
    start = datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc) + timedelta(days=user_index % 31)
    for idx, session in enumerate(sessions):
        session["session_index"] = idx
        session["session_id"] = f"{user_id}_s{idx:04d}"
        session["timestamp"] = (start + timedelta(days=idx * 2 + rng.randint(0, 1))).isoformat()
        session.pop("_phase", None)
        session.pop("_temp_session_index", None)

    habit_graph_by_id = {h["habit_id"]: h for h in habit_graphs}
    evidence_by_habit: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for session in sessions:
        signal_type = session["memory_annotations"]["signal_type"]
        for habit_id in session["memory_annotations"]["linked_habit_ids"]:
            evidence_by_habit[habit_id][signal_type].append(session["session_id"])

    probes: List[Dict[str, Any]] = []
    probe_keys: List[Dict[str, Any]] = []

    for habit_id, template in zip(user_habit_ids, templates):
        support_ids = evidence_by_habit[habit_id].get("support", [])
        post_drift_ids = evidence_by_habit[habit_id].get("post_drift_support", [])
        exception_ids = evidence_by_habit[habit_id].get("exception", [])
        boundary_ids = evidence_by_habit[habit_id].get("boundary_counterexample", [])
        last_session_index = max(s["session_index"] for s in sessions)

        probe_specs = [
            (
                "direct_use",
                template["direct_query"],
                template.get("drift_action", template["default_action"]) if post_drift_ids else template["default_action"],
                [
                    template["wrong_action"],
                    "Ask an unrelated clarifying question even though enough evidence exists.",
                    template["boundary_action"],
                ],
                "apply_scoped_habit",
                support_ids + post_drift_ids,
            ),
            (
                "boundary",
                template["boundary_query"],
                template["boundary_action"],
                [
                    template["default_action"],
                    template["wrong_action"],
                    "Refuse to answer because the user has prior behavior in the history.",
                ],
                "do_not_apply_out_of_scope",
                boundary_ids,
            ),
            (
                "exception",
                template["exception_query"],
                template["exception_action"],
                [
                    template["default_action"],
                    template["wrong_action"],
                    "Ignore the exception and follow the most frequent old pattern.",
                ],
                "apply_exception",
                exception_ids,
            ),
            (
                "evidence",
                f"Which evidence best supports the user's recurring behavior for: {template['condition']}?",
                "Cite multiple support episodes and distinguish boundary or exception cases.",
                [
                    "Cite only the most recent semantically similar episode.",
                    "Cite the boundary case as if it were support.",
                    "Say there is no evidence in the history.",
                ],
                "return_evidence_with_scope",
                support_ids + post_drift_ids,
            ),
        ]
        if post_drift_ids:
            probe_specs.append(
                (
                    "drift",
                    template["direct_query"],
                    template["drift_action"],
                    [
                        template["wrong_action"],
                        template["support_assistant"],
                        "Ask the user to restate all old preferences from scratch.",
                    ],
                    "use_latest_sustained_evidence",
                    post_drift_ids,
                )
            )
        if template["family"] == "privacy_consent":
            probe_specs.append(
                (
                    "privacy",
                    template["direct_query"],
                    template["default_action"],
                    [
                        template["wrong_action"],
                        "Use the one-off sensitive detail to personalize a later unrelated task.",
                        "Add the sensitive fact to the durable profile silently.",
                    ],
                    "do_not_store_or_use_sensitive_one_off",
                    support_ids,
                )
            )

        for pidx, (probe_type, query, correct, distractors, gold_action, evidence_ids) in enumerate(probe_specs):
            choices, gold_choice_id = make_choice_set(rng, correct, distractors[:3], gold_action)
            probe_id = f"{habit_id}_p{pidx:02d}_{probe_type}"
            probe = {
                "probe_id": probe_id,
                "user_id": user_id,
                "split": split,
                "probe_type": probe_type,
                "habit_id": habit_id,
                "habit_family": template["family"],
                "query": query,
                "choices": choices,
                "validator": {
                    "type": "choice_equals",
                    "gold_choice_id": gold_choice_id,
                    "gold_action": gold_action,
                },
                "visible_history_scope": {
                    "user_id": user_id,
                    "max_session_index": last_session_index,
                },
                "metadata": {
                    "template_id": template["template_id"],
                    "condition": template["condition"],
                    "boundary_condition": template["boundary_condition"],
                    "exception_condition": template["exception_condition"],
                    "horizon_sessions": last_session_index + 1,
                    "support_count": len(support_ids),
                    "post_drift_support_count": len(post_drift_ids),
                    "boundary_count": len(boundary_ids),
                    "exception_count": len(exception_ids),
                },
            }
            probes.append(probe)
            probe_keys.append(
                {
                    "probe_id": probe_id,
                    "user_id": user_id,
                    "habit_id": habit_id,
                    "gold_choice_id": gold_choice_id,
                    "gold_action": gold_action,
                    "gold_evidence_session_ids": evidence_ids,
                    "hidden_habit_graph": habit_graph_by_id[habit_id],
                }
            )

    unassigned = [t for t in HABIT_TEMPLATES if t["template_id"] not in {t["template_id"] for t in templates}]
    if unassigned:
        template = rng.choice(unassigned)
        correct = "Ask a brief clarifying question because the history does not establish this preference."
        choices, gold_choice_id = make_choice_set(
            rng,
            correct,
            [
                template["default_action"],
                template["wrong_action"],
                "Infer the preference from a single unrelated prior task.",
            ],
            "ask_when_evidence_insufficient",
        )
        probe_id = f"{user_id}_unassigned_{template['template_id']}_ask_act"
        last_session_index = max(s["session_index"] for s in sessions)
        probes.append(
            {
                "probe_id": probe_id,
                "user_id": user_id,
                "split": split,
                "probe_type": "ask_act",
                "habit_id": None,
                "habit_family": template["family"],
                "query": f"Should the assistant apply this possible preference now? {template['direct_query']}",
                "choices": choices,
                "validator": {
                    "type": "choice_equals",
                    "gold_choice_id": gold_choice_id,
                    "gold_action": "ask_when_evidence_insufficient",
                },
                "visible_history_scope": {
                    "user_id": user_id,
                    "max_session_index": last_session_index,
                },
                "metadata": {
                    "template_id": template["template_id"],
                    "condition": template["condition"],
                    "horizon_sessions": last_session_index + 1,
                    "support_count": 0,
                    "negative_control": True,
                },
            }
        )
        probe_keys.append(
            {
                "probe_id": probe_id,
                "user_id": user_id,
                "habit_id": None,
                "gold_choice_id": gold_choice_id,
                "gold_action": "ask_when_evidence_insufficient",
                "gold_evidence_session_ids": [],
                "hidden_habit_graph": None,
            }
        )

    return sessions, habit_graphs, probes, probe_keys


def validate_outputs(
    sessions: List[Dict[str, Any]],
    probes: List[Dict[str, Any]],
    probe_keys: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    session_schema = json.loads(SESSION_SCHEMA_PATH.read_text(encoding="utf-8"))
    probe_schema = json.loads(PROBE_SCHEMA_PATH.read_text(encoding="utf-8"))
    session_ids = {session["session_id"] for session in sessions}
    key_by_probe = {key["probe_id"]: key for key in probe_keys}
    probe_ids = set()
    validation_rows: List[Dict[str, Any]] = []
    summary = Counter()

    for session in sessions:
        errors = []
        try:
            jsonschema.validate(session, session_schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"schema:{exc.message}")
        for msg in session.get("messages", []):
            if has_pii(msg.get("content", "")):
                errors.append("pii_in_message")
        status = "pass" if not errors else "fail"
        summary[f"session_{status}"] += 1

    for probe in probes:
        errors = []
        warnings = []
        try:
            jsonschema.validate(probe, probe_schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"schema:{exc.message}")
        if probe["probe_id"] in probe_ids:
            errors.append("duplicate_probe_id")
        probe_ids.add(probe["probe_id"])
        if has_pii(probe["query"]):
            errors.append("pii_in_query")
        choice_ids = [choice["choice_id"] for choice in probe["choices"]]
        if len(choice_ids) != len(set(choice_ids)):
            errors.append("duplicate_choice_id")
        gold_choice_id = probe["validator"].get("gold_choice_id")
        if gold_choice_id not in choice_ids:
            errors.append("gold_choice_missing")
        if "habit" in probe["query"].lower():
            warnings.append("query_mentions_habit_word")
        if len(probe["query"]) < 20:
            warnings.append("short_query")
        key = key_by_probe.get(probe["probe_id"])
        if not key:
            errors.append("missing_private_key")
        else:
            missing_evidence = [sid for sid in key.get("gold_evidence_session_ids", []) if sid not in session_ids]
            if missing_evidence:
                errors.append("missing_evidence_session")
            if probe["probe_type"] in {"direct_use", "evidence"} and not key.get("gold_evidence_session_ids"):
                errors.append("missing_gold_evidence")
        status = "pass" if not errors else "fail"
        summary[f"probe_{status}"] += 1
        if warnings:
            summary["probe_warning"] += 1
        validation_rows.append(
            {
                "probe_id": probe["probe_id"],
                "status": status,
                "errors": errors,
                "warnings": warnings,
            }
        )

    summary.update(
        {
            "sessions_total": len(sessions),
            "probes_total": len(probes),
            "private_keys_total": len(probe_keys),
            "unique_probe_ids": len(probe_ids),
        }
    )
    return validation_rows, dict(summary)


def public_probe(probe: Dict[str, Any]) -> Dict[str, Any]:
    validator = dict(probe["validator"])
    return {
        "probe_id": public_probe_id(probe["probe_id"]),
        "user_id": probe["user_id"],
        "split": probe["split"],
        "query": probe["query"],
        "choices": probe["choices"],
        "visible_history_scope": probe["visible_history_scope"],
        "evaluation_contract": {
            "answer_format": "return one choice_id and optional evidence_session_ids",
            "validator_type": validator["type"],
        },
    }


def public_probe_id(private_probe_id: str) -> str:
    return f"probe_{stable_hash(private_probe_id, 16)}"


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


def make_review_rows(
    probes: List[Dict[str, Any]],
    probe_keys: List[Dict[str, Any]],
    sessions: List[Dict[str, Any]],
    validation_rows: List[Dict[str, Any]],
    rng: random.Random,
    sample_rate: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    session_by_id = {session["session_id"]: session for session in sessions}
    key_by_probe = {key["probe_id"]: key for key in probe_keys}
    validation_by_probe = {row["probe_id"]: row for row in validation_rows}
    all_rows: List[Dict[str, Any]] = []

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
                    "user": compact_text(session["messages"][0]["content"], 220),
                    "feedback": compact_text(session["messages"][-1]["content"], 180)
                    if len(session["messages"]) > 2
                    else "",
                }
            )
        validation = validation_by_probe.get(probe["probe_id"], {})
        all_rows.append(
            {
                "review_id": f"review_{probe['probe_id']}",
                "public_probe_id": key["public_probe_id"],
                "probe_id": probe["probe_id"],
                "user_id": probe["user_id"],
                "split": probe["split"],
                "probe_type": probe["probe_type"],
                "habit_family": probe["habit_family"],
                "query": probe["query"],
                "choices_json": json.dumps(probe["choices"], ensure_ascii=False),
                "proposed_gold_choice_id": key["gold_choice_id"],
                "proposed_gold_action": key["gold_action"],
                "evidence_preview_json": json.dumps(evidence_preview, ensure_ascii=False),
                "auto_validation_status": validation.get("status", "missing"),
                "auto_validation_errors": ";".join(validation.get("errors", [])),
                "auto_validation_warnings": ";".join(validation.get("warnings", [])),
                "reviewer_decision": "",
                "reviewer_notes": "",
            }
        )

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["probe_type"], row["habit_family"])].append(row)
    sample_rows: List[Dict[str, Any]] = []
    for rows in grouped.values():
        k = max(1, int(round(len(rows) * sample_rate)))
        sample_rows.extend(rng.sample(rows, min(k, len(rows))))
    sample_rows.sort(key=lambda row: row["review_id"])
    return all_rows, sample_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    sessions: List[Dict[str, Any]],
    habit_graphs: List[Dict[str, Any]],
    probes: List[Dict[str, Any]],
    review_rows: List[Dict[str, Any]],
    review_sample: List[Dict[str, Any]],
    validation_summary: Dict[str, Any],
    source_meta: Dict[str, Any],
) -> None:
    by_probe = Counter(p["probe_type"] for p in probes)
    by_family = Counter(p["habit_family"] for p in probes)
    by_split = Counter(p["split"] for p in probes)
    sessions_per_user = Counter(s["user_id"] for s in sessions)
    support_counts = [p["metadata"].get("support_count", 0) for p in probes if p["probe_type"] == "direct_use"]
    avg_support = round(statistics.mean(support_counts), 2) if support_counts else 0
    lines = [
        "# HABIT-Bench Pilot Auto-Validation Summary",
        "",
        f"- Created: {datetime.now(timezone.utc).isoformat()}",
        f"- Users: {args.n_users}",
        f"- Sessions: {len(sessions)}",
        f"- Habit graphs: {len(habit_graphs)}",
        f"- Probes: {len(probes)}",
        f"- Review rows: {len(review_rows)}",
        f"- Review sample rows: {len(review_sample)}",
        f"- Seed source: {source_meta.get('source')}",
        f"- Validation probe pass: {validation_summary.get('probe_pass', 0)}",
        f"- Validation probe fail: {validation_summary.get('probe_fail', 0)}",
        f"- Validation warnings: {validation_summary.get('probe_warning', 0)}",
        f"- Avg support count for direct probes: {avg_support}",
        f"- Sessions per user min/max: {min(sessions_per_user.values())}/{max(sessions_per_user.values())}",
        "",
        "## Probe Counts",
        "",
    ]
    for key, value in sorted(by_probe.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Habit Family Counts", ""])
    for key, value in sorted(by_family.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Split Counts", ""])
    for key, value in sorted(by_split.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Human Review Boundary",
            "",
            "No human review has been performed. The CSV files in `review/` are the next handoff point.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_card(path: Path, manifest: Dict[str, Any]) -> None:
    text = f"""# HABIT-Bench Pilot Dataset Card

This pilot benchmark package was generated automatically and has not yet
undergone human review.

## Intended Use

Evaluate whether long-term memory agents can infer and use implicit,
context-scoped user habits while avoiding false personalization.

## Construction

- Seed source: {manifest['source_meta'].get('source')}
- Users: {manifest['counts']['users']}
- Sessions: {manifest['counts']['sessions']}
- Probes: {manifest['counts']['probes']}
- Habit graphs: {manifest['counts']['habit_graphs']}

Real user-assistant prompts are used only as sanitized task seeds. Hidden habit
graphs, feedback, and probes are synthetic and controlled.

## Human Review Status

Human review has not been performed. Use `review/review_queue_sample.csv` for a
first audit and `review/review_queue_all.csv` for full audit.

## Leakage Controls

Public files omit hidden habit graphs, gold labels, and explicit signal types.
Private files contain labels and should not be exposed to benchmarked systems.

## Known Limitations

- Lifelines are pseudo-users stitched from real prompt seeds and synthetic
  controlled feedback.
- The current pilot is English-only unless the seed loader is extended.
- Automatic validators check structure, labels, evidence links, and obvious PII;
  they do not replace human audit.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "habit_bench_pilot_v0")
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--sessions-per-user", type=int, default=60)
    parser.add_argument("--min-habits-per-user", type=int, default=2)
    parser.add_argument("--max-habits-per-user", type=int, default=4)
    parser.add_argument("--support-episodes-per-habit", type=int, default=5)
    parser.add_argument("--post-drift-episodes", type=int, default=4)
    parser.add_argument("--seed-prompts", type=int, default=5000)
    parser.add_argument("--source", choices=["auto", "wildchat", "fallback"], default="auto")
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--refresh-seeds", action="store_true")
    parser.add_argument("--review-sample-rate", type=float, default=0.10)
    parser.add_argument("--use-llm-template-naturalization", action="store_true")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--llm-timeout-sec", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    source_started = time.time()
    seeds, source_meta = load_seed_prompts(args, out_dir, rng)
    source_meta["total_seed_loader_elapsed_sec"] = round(time.time() - source_started, 2)
    naturalization_meta = maybe_naturalize_templates(args, out_dir, HABIT_TEMPLATES)

    grouped = seed_by_domain(seeds)
    sessions: List[Dict[str, Any]] = []
    habit_graphs: List[Dict[str, Any]] = []
    probes: List[Dict[str, Any]] = []
    probe_keys: List[Dict[str, Any]] = []

    for user_index in range(args.n_users):
        user_sessions, user_habits, user_probes, user_keys = generate_user_package(
            user_index,
            args,
            grouped,
            rng,
        )
        sessions.extend(user_sessions)
        habit_graphs.extend(user_habits)
        probes.extend(user_probes)
        probe_keys.extend(user_keys)

    for key in probe_keys:
        key["public_probe_id"] = public_probe_id(key["probe_id"])

    validation_rows, validation_summary = validate_outputs(sessions, probes, probe_keys)
    review_rows, review_sample = make_review_rows(
        probes,
        probe_keys,
        sessions,
        validation_rows,
        rng,
        args.review_sample_rate,
    )

    public_dir = out_dir / "public"
    private_dir = out_dir / "private"
    reports_dir = out_dir / "reports"
    review_dir = out_dir / "review"

    write_jsonl(public_dir / "lifelines.jsonl", [public_session(s) for s in sessions])
    write_jsonl(public_dir / "probes.jsonl", [public_probe(p) for p in probes])
    write_jsonl(private_dir / "sessions_with_annotations.jsonl", sessions)
    write_jsonl(private_dir / "habit_graphs.jsonl", habit_graphs)
    write_jsonl(private_dir / "probe_key.jsonl", probe_keys)
    write_jsonl(reports_dir / "auto_validation_rows.jsonl", validation_rows)
    write_csv(review_dir / "review_queue_all.csv", review_rows)
    write_csv(review_dir / "review_queue_sample.csv", review_sample)
    write_jsonl(review_dir / "review_queue_all.jsonl", review_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder": "build_habit_bench_pilot.py",
        "status": "pre_human_review",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "api_key"
        },
        "counts": {
            "users": args.n_users,
            "sessions": len(sessions),
            "habit_graphs": len(habit_graphs),
            "probes": len(probes),
            "review_rows_all": len(review_rows),
            "review_rows_sample": len(review_sample),
            "seed_prompts": len(seeds),
        },
        "source_meta": source_meta,
        "llm_naturalization": naturalization_meta,
        "validation_summary": validation_summary,
        "artifact_paths": {
            "public_lifelines": str(public_dir / "lifelines.jsonl"),
            "public_probes": str(public_dir / "probes.jsonl"),
            "private_habit_graphs": str(private_dir / "habit_graphs.jsonl"),
            "private_probe_key": str(private_dir / "probe_key.jsonl"),
            "review_queue_all": str(review_dir / "review_queue_all.csv"),
            "review_queue_sample": str(review_dir / "review_queue_sample.csv"),
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(
        reports_dir / "auto_validation_summary.md",
        args,
        sessions,
        habit_graphs,
        probes,
        review_rows,
        review_sample,
        validation_summary,
        source_meta,
    )
    write_dataset_card(out_dir / "DATASET_CARD.md", manifest)

    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    if validation_summary.get("probe_fail", 0) or validation_summary.get("session_fail", 0):
        raise SystemExit("Automatic validation failed; inspect reports/auto_validation_rows.jsonl")


if __name__ == "__main__":
    main()
