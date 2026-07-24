#!/usr/bin/env python3
"""GPT-first habit discovery from complete Taskmaster conversations.

Rules and annotations are used only to retrieve potentially relevant source
dialogues.  They never create a habit, choose its concrete value, or assign a
label.  GPT reads complete conversations and must cite exact user turns from
three independent conversations and instruction templates for every habit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api_client import post_chat, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
REVISION = "v04_gpt_complete_dialogue_habit_discovery_r1"


FAMILY_DIRECTORIES: dict[str, dict[str, Any]] = {
    "flight_route_connection_policy": {
        "domain": "flights", "annotations": ["flight_search.stops"],
        "terms": r"non[- ]?stop|direct flight|layover|connection|stopover|tight connection|overnight connection",
        "description": "nonstop defaults, connection limits, and connection-quality tradeoffs",
    },
    "flight_price_quality_tradeoff": {
        "domain": "flights", "annotations": ["flight_search.price_range", "flight_search.total_fare", "flight_search.other_description"],
        "terms": r"cheap|cheapest|price|fare|cost|reliab|on[- ]time|duration|shortest|convenient|worth",
        "description": "price versus reliability, duration, convenience, or itinerary quality",
        "context_turn_limit": 10,
    },
    "flight_departure_arrival_timing": {
        "domain": "flights", "annotations": ["flight_search.time_of_day"],
        "terms": r"morning|afternoon|evening|night|red[- ]?eye|early|late|arriv|depart",
        "description": "recurring departure/arrival timing defaults and arrival buffers",
    },
    "flight_seat_preference": {
        "domain": "flights", "annotations": ["flight_search.seat_location"],
        "terms": r"aisle|window|middle seat|seat location|sit together",
        "description": "seat-location and seating-arrangement preferences",
    },
    "flight_cabin_preference": {
        "domain": "flights", "annotations": ["flight_search.seating_class"],
        "terms": r"coach|economy|premium economy|business class|first class|cabin",
        "description": "cabin defaults and context-dependent cabin upgrades",
    },
    "flight_airline_preference": {
        "domain": "flights", "annotations": ["flight_search.airline"],
        "terms": r"airline|delta|united|american|jetblue|southwest|alaska",
        "description": "soft or strong carrier preferences and their exceptions",
    },
    "flight_schedule_flexibility": {
        "domain": "flights", "annotations": ["flight_search.time_of_day", "flight_search.other_description"],
        "terms": r"flexib|any time|fixed|must arrive|need to be there|before|after|change|refundable",
        "description": "fixed versus flexible schedules, arrival buffers, and change tolerance",
    },
    "trip_context_flight_policy": {
        "domain": "flights", "annotations": ["flight_search.stops", "flight_search.price_range", "flight_search.seating_class"],
        "terms": r"business|work|meeting|family|kids|children|vacation|leisure|urgent|emergency",
        "description": "flight defaults that genuinely differ by business, family, leisure, or urgent context",
        "context_turn_limit": 10,
    },
    "hotel_location_price_tradeoff": {
        "domain": "hotels", "annotations": ["hotel_search.sub_location.hotel", "hotel_search.price_range"],
        "terms": r"downtown|airport|walking distance|walkable|near|close to|location|price|cheap|budget",
        "description": "hotel location priorities and explicit location-versus-price tradeoffs",
    },
    "hotel_quality_threshold": {
        "domain": "hotels", "annotations": ["hotel_search.star_rating", "hotel_search.customer_rating"],
        "terms": r"star|rating|review|quality|well[- ]rated",
        "description": "stable hotel star/review floors and quality tradeoffs",
    },
    "hotel_included_services": {
        "domain": "hotels", "annotations": ["hotel_search.amenity", "hotel_search.other_request"],
        "terms": r"breakfast|parking|wi[- ]?fi|internet|shuttle|included|complimentary|free ",
        "description": "included breakfast, parking, Wi-Fi, shuttle, and similar service priorities",
    },
    "hotel_room_bed_policy": {
        "domain": "hotels", "annotations": ["hotel_search.type.bed", "hotel_search. num.beds", "hotel_search.type.room", "hotel_search.num.rooms"],
        "terms": r"king|queen|bed|room|suite|adjoining|connecting room|separate beds",
        "description": "room and bed defaults not mechanically determined by current party size",
    },
    "hotel_property_atmosphere": {
        "domain": "hotels", "annotations": ["hotel_search.other_request", "hotel_search.amenity"],
        "terms": r"quiet|lively|nightlife|boutique|resort|family[- ]friendly|romantic|business hotel|peaceful",
        "description": "quiet/lively/property-style preferences with a recurring context",
    },
    "trip_context_hotel_policy": {
        "domain": "hotels", "annotations": ["hotel_search.sub_location.hotel", "hotel_search.type.room", "hotel_search.amenity"],
        "terms": r"business|work|meeting|conference|family|kids|children|vacation|leisure|group",
        "description": "hotel defaults that genuinely differ for work, family, group, or leisure trips",
    },
    "hotel_accessibility_amenity_priority": {
        "domain": "hotels", "annotations": ["hotel_search.amenity", "hotel_search.other_request"],
        "terms": r"accessible|accessibility|wheelchair|elevator|fitness|pool|pet[- ]friendly|non[- ]smoking|amenit",
        "description": "accessibility requirements and durable amenity priorities",
    },
    "hotel_view_balcony_flexibility": {
        "domain": "hotels", "annotations": ["hotel_search.amenity", "hotel_search.other_request"],
        "terms": r"balcony|view|ocean view|city view|refundable|cancel|flexib|change",
        "description": "balcony/view defaults and hotel cancellation/flexibility preferences",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def compact_card(card: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    wanted = set(spec["annotations"])
    pattern = re.compile(spec["terms"], re.I)
    relevant_turn_ids = {
        item["turn_index"] for item in card.get("preference_evidence", [])
        if item.get("annotation") in wanted
    }
    positions = [
        index for index, turn in enumerate(card["turns"])
        if turn["turn_index"] in relevant_turn_ids or pattern.search(turn["text"])
    ]
    keep = set()
    for position in positions:
        keep.update(range(max(0, position - 3), min(len(card["turns"]), position + 4)))
    if not keep:
        keep.update(range(min(12, len(card["turns"]))))
    ordered = sorted(keep)
    turn_limit = int(spec.get("context_turn_limit", 16))
    if len(ordered) > turn_limit:
        ranked = sorted(ordered, key=lambda pos: (min(abs(pos - hit) for hit in positions), pos))
        ordered = sorted(ranked[:turn_limit])
    ordered_set = set(ordered)
    return {
        "conversation_id": card["conversation_id"],
        "instruction_id": card["instruction_id"],
        "domain": card["source_domain"],
        "contextual_dialogue_window": [
            {"turn_index": turn["turn_index"], "role": turn["role"], "text": turn["text"]}
            for index, turn in enumerate(card["turns"]) if index in ordered_set
        ],
        "annotations_for_retrieval_only": [
            {"annotation": item["annotation"], "turn_index": item["turn_index"], "segment": item["segment"]}
            for item in card.get("preference_evidence", [])
        ],
    }


def retrieve_cards(cards: list[dict[str, Any]], spec: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    pattern = re.compile(spec["terms"], re.I)
    wanted_annotations = set(spec["annotations"])
    scored = []
    for card in cards:
        if card["source_domain"] != spec["domain"]:
            continue
        user_text = " ".join(turn["text"] for turn in card["turns"] if turn["role"] == "user")
        matches = len(pattern.findall(user_text))
        evidence = card.get("preference_evidence", [])
        annotation_hits = sum(item.get("annotation") in wanted_annotations for item in evidence)
        preference_hits = sum(bool(item.get("preference_language")) for item in evidence)
        habitual_hits = sum(bool(item.get("habitual_language")) for item in evidence)
        score = 4 * min(annotation_hits, 3) + 2 * min(matches, 4) + preference_hits + 4 * habitual_hits
        if score:
            scored.append((score, card["quality_score"], card))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen, instruction_counts = [], Counter()
    # First pass maximizes independent Taskmaster instruction templates.
    for _, _, card in scored:
        instruction = card.get("instruction_id")
        if instruction_counts[instruction] == 0:
            chosen.append(card); instruction_counts[instruction] += 1
        if len(chosen) >= limit:
            break
    # Second pass adds strong within-family variation while capping paraphrases.
    if len(chosen) < limit:
        chosen_ids = {row["conversation_id"] for row in chosen}
        for _, _, card in scored:
            instruction = card.get("instruction_id")
            if card["conversation_id"] in chosen_ids or instruction_counts[instruction] >= 2:
                continue
            chosen.append(card); chosen_ids.add(card["conversation_id"]); instruction_counts[instruction] += 1
            if len(chosen) >= limit:
                break
    return chosen


def normalized(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def validate_discovery(raw: Any, family: str, cards: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = raw.get("habits") if isinstance(raw, dict) else None
    if not isinstance(candidates, list) or len(candidates) > 1:
        raise ValueError("response must contain habits list with at most one entry")
    by_id = {card["conversation_id"]: card for card in cards}
    output = []
    for index, item in enumerate(candidates):
        required = ["name", "preference_value", "condition", "default_action", "boundary_condition", "exception_condition", "rationale"]
        if any(not normalized(item.get(key)) for key in required):
            raise ValueError(f"incomplete discovered habit {index}")
        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError("habit evidence must be a list")
        clean_evidence, seen_conversations, seen_instructions = [], set(), set()
        for citation in evidence:
            conversation_id = citation.get("conversation_id")
            card = by_id.get(conversation_id)
            if not card or conversation_id in seen_conversations:
                raise ValueError("unknown or repeated evidence conversation")
            turn_index = citation.get("turn_index")
            turn = next((row for row in card["turns"] if row["turn_index"] == turn_index and row["role"] == "user"), None)
            quote = normalized(citation.get("exact_user_quote"))
            if not turn or len(quote) < 8 or quote not in normalized(turn["text"]):
                raise ValueError(f"untraceable exact quote {conversation_id}:{turn_index}")
            seen_conversations.add(conversation_id); seen_instructions.add(card["instruction_id"])
            clean_evidence.append({
                "conversation_id": conversation_id, "instruction_id": card["instruction_id"],
                "turn_index": turn_index, "exact_user_quote": str(citation["exact_user_quote"]).strip(),
                "contribution": str(citation.get("contribution", "")).strip(),
            })
        if len(seen_conversations) < 3 or len(seen_instructions) < 3:
            raise ValueError(f"habit lacks 3 independent conversations/instructions: {index}")
        digest = hashlib.sha256(json.dumps({
            "family": family, "preference": item["preference_value"],
            "conversation_ids": sorted(seen_conversations),
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]
        habit_id = f"tm_v04_e2e_habit_{digest}"
        bundle_id = f"tm_v04_e2e_bundle_{digest}"
        source_examples = []
        for citation in clean_evidence:
            source_examples.append({
                "conversation_id": citation["conversation_id"], "instruction_id": citation["instruction_id"],
                "turn_index": citation["turn_index"], "user_utterance": citation["exact_user_quote"],
                "annotated_segment": citation["exact_user_quote"], "annotation": "gpt_complete_dialogue_discovery",
                "source_domain": FAMILY_DIRECTORIES[family]["domain"],
                "preference_language": True, "habitual_language": False,
            })
        raw_confidence = item.get("confidence", 0.7)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = {"low": 0.55, "medium": 0.70, "high": 0.85}.get(normalized(raw_confidence), 0.70)
        output.append({
            "habit_instance_id": habit_id, "bundle_id": bundle_id, "family": family,
            "name": str(item["name"]).strip(), "preference_value": str(item["preference_value"]).strip(),
            "condition": str(item["condition"]).strip(), "default_action": str(item["default_action"]).strip(),
            "boundary_condition": str(item["boundary_condition"]).strip(),
            "exception_condition": str(item["exception_condition"]).strip(),
            "rationale": str(item["rationale"]).strip(), "confidence": confidence,
            "source_evidence": clean_evidence, "source_examples": source_examples,
            "source_dataset": "google-research-datasets/Taskmaster-2",
            "grounding_contract": {
                "synthetic_user_habit": True, "not_claimed_as_original_speaker_longitudinal_habit": True,
                "minimum_independent_source_conversations": 3, "minimum_distinct_source_instructions": 3,
                "multi_turn_context_windows_seen_by_discovery_model": True,
            },
            "decision": "accept", "induced_by": args.model, "reasoning_effort": args.reasoning_effort,
            "induction_method": "gpt_semantic_discovery_from_complete_taskmaster_dialogues",
            "induced_at": datetime.now(timezone.utc).isoformat(),
        })
    return output


def discovery_batches(selected: list[dict[str, Any]], spec: dict[str, Any], batch_size: int, max_batches: int) -> list[list[dict[str, Any]]]:
    """Group semantically promising sources for retrieval only.

    Annotation values decide which complete dialogues GPT sees together, not
    whether a habit exists or what it means.  Abstract directories naturally
    fall back to ranked complete-dialogue batches.
    """
    wanted = set(spec["annotations"])
    groups: dict[str, list[dict[str, Any]]] = {}
    for card in selected:
        values = [
            item.get("normalized_value") for item in card.get("preference_evidence", [])
            if item.get("annotation") in wanted and item.get("normalized_value")
        ]
        for value in dict.fromkeys(values):
            groups.setdefault(str(value), []).append(card)
    candidates = []
    for value, rows in groups.items():
        unique, instructions = [], set()
        for card in rows:
            if card["instruction_id"] in instructions:
                continue
            unique.append(card); instructions.add(card["instruction_id"])
            if len(unique) == batch_size:
                break
        if len(unique) >= 3:
            candidates.append((len(rows), value, unique))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    batches, used_signatures = [], set()
    for _, value, rows in candidates:
        signature = tuple(sorted(card["conversation_id"] for card in rows))
        if signature not in used_signatures:
            batches.append(rows); used_signatures.add(signature)
        if len(batches) >= max_batches:
            return batches
    # Fill missing slots with ranked, non-overlapping complete-dialogue groups.
    used_ids = {card["conversation_id"] for batch in batches for card in batch}
    remainder = [card for card in selected if card["conversation_id"] not in used_ids]
    for start in range(0, len(remainder), batch_size):
        batch = remainder[start:start + batch_size]
        if len(batch) >= 3 and len({card["instruction_id"] for card in batch}) >= 3:
            batches.append(batch)
        if len(batches) >= max_batches:
            break
    return batches


def discover_family(family: str, spec: dict[str, Any], cards: list[dict[str, Any]], args: argparse.Namespace) -> tuple[str, list[dict[str, Any]], int]:
    selected = retrieve_cards(cards, spec, args.dialogues_per_family)
    batches = discovery_batches(selected, spec, args.dialogues_per_batch, args.max_batches_per_family)
    discovered = []
    for batch_index, batch in enumerate(batches):
        compact = [compact_card(card, spec) for card in batch]
        fingerprint = hashlib.sha256(json.dumps({
            "revision": REVISION + "_small_complete_dialogue_batch_r1", "model": args.model,
            "reasoning_effort": args.reasoning_effort, "family": family,
            "description": spec["description"], "dialogues": compact,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path = args.dataset / "work" / "gpt_habit_discovery_cache" / f"{family}_{batch_index}_{fingerprint}.json"
        if path.exists():
            discovered.extend(json.loads(path.read_text(encoding="utf-8"))["habits"])
            continue
        prompt = {
            "task": "Discover reusable synthetic longitudinal travel habits by semantically reading grounded multi-turn context windows from independent Taskmaster dialogues.",
            "family_directory": {"name": family, "description": spec["description"]},
            "critical_rules": [
                "The family is only a broad retrieval directory. You decide the concrete habit values and conditions from dialogue semantics.",
                "Return zero habits if the dialogues do not support a coherent reusable default. Never force family coverage.",
                "Return at most one concrete habit from this small evidence group. Return zero if the sources do not independently support the same policy.",
                "Each habit must be supported by at least three distinct conversation_id values and three distinct instruction_id values.",
                "Each cited conversation must independently support the same concrete policy, not merely mention the same broad family or slot.",
                "Read cross-turn negotiations and tradeoffs. Do not reduce a habit to an annotation segment when the surrounding dialogue changes its meaning.",
                "Reject dates, destinations, routes, current party size, one-off prices, and constraints mechanically implied by the current transaction.",
                "A default and its fallback belong in one conditional habit graph. Separate genuinely different concrete defaults instead of averaging them.",
                "Use soft wording unless the evidence repeatedly supports a hard requirement.",
                "For every citation copy an exact user substring and its exact turn_index. Do not paraphrase quotes.",
                "Return {habits:[{name,preference_value,condition,default_action,boundary_condition,exception_condition,rationale,confidence,evidence:[{conversation_id,turn_index,exact_user_quote,contribution}]}]}",
            ],
            "grounded_multi_turn_context_windows": compact,
        }
        last_error = None
        for attempt in range(3):
            request = dict(prompt)
            if last_error:
                request["previous_output_rejection"] = last_error
            try:
                raw = post_chat(
                    base_url=args.base_url, api_key=args.api_key, model=args.model,
                    messages=[
                        {"role": "system", "content": "You are a conservative travel-memory dataset scientist. Return strict JSON only."},
                        {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                    ],
                    max_tokens=1800, timeout=args.timeout, retries=args.retries,
                    transport=args.transport, reasoning_effort=args.reasoning_effort,
                )["json"]
                habits = validate_discovery(raw, family, batch, args)
                write_json(path, {"cache_fingerprint": fingerprint, "habits": habits, "selected_conversation_ids": [x["conversation_id"] for x in batch]})
                discovered.extend(habits)
                break
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                last_error = str(exc)
        else:
            validation_markers = [
                "unknown or repeated evidence conversation", "untraceable exact quote",
                "lacks 3 independent conversations/instructions", "incomplete discovered habit",
                "habit evidence must be a list", "response must contain habits list",
            ]
            if any(marker in str(last_error) for marker in validation_markers):
                write_json(path, {
                    "cache_fingerprint": fingerprint, "habits": [],
                    "selected_conversation_ids": [x["conversation_id"] for x in batch],
                    "xhigh_rejected_after_three_invalid_outputs": str(last_error),
                    "reasoning_effort": args.reasoning_effort,
                })
                continue
            raise RuntimeError(f"{family} batch {batch_index} failed after three fresh GPT discoveries: {last_error}")
    return family, discovered, len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default=os.getenv("HABITBENCH_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("HABITBENCH_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.getenv("HABITBENCH_GEN_MODEL", "gpt-5.5"))
    parser.add_argument("--reasoning-effort", default=os.getenv("HABITBENCH_REASONING_EFFORT", "xhigh"))
    parser.add_argument("--dialogues-per-family", type=int, default=20)
    parser.add_argument("--dialogues-per-batch", type=int, default=4)
    parser.add_argument("--max-batches-per-family", type=int, default=2)
    parser.add_argument("--families", help="Optional comma-separated family names; omitted means all")
    parser.add_argument("--test-only", action="store_true", help="Run/cache selected families without replacing release files")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--transport", choices=["curl", "urllib"], default="curl")
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("Set HABITBENCH_BASE_URL and HABITBENCH_API_KEY")
    cards = read_jsonl(args.dataset / "sources" / "taskmaster_source_cards.jsonl")
    selected_families = FAMILY_DIRECTORIES
    if args.families:
        names = [item.strip() for item in args.families.split(",") if item.strip()]
        unknown = [name for name in names if name not in FAMILY_DIRECTORIES]
        if unknown:
            raise SystemExit(f"unknown families: {unknown}")
        selected_families = {name: FAMILY_DIRECTORIES[name] for name in names}
    all_habits, retrieval_counts = [], {}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(discover_family, family, spec, cards, args): family
            for family, spec in selected_families.items()
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            family = futures[future]
            try:
                _, habits, retrieved = future.result()
                all_habits.extend(habits); retrieval_counts[family] = retrieved
                print(f"gpt_habit_discovery_progress {index}/{len(futures)} family={family} habits={len(habits)}", flush=True)
            except Exception as exc:
                failures.append({"family": family, "error": str(exc)})
                print(f"gpt_habit_discovery_failed family={family} error={exc}", flush=True)
    if failures:
        write_json(args.dataset / "reports" / "gpt_habit_discovery_failures.json", {"failures": failures})
        raise SystemExit(f"GPT habit discovery incomplete: {len(failures)} family calls failed")
    (args.dataset / "reports" / "gpt_habit_discovery_failures.json").unlink(missing_ok=True)
    all_habits.sort(key=lambda row: (row["family"], row["habit_instance_id"]))
    if args.test_only:
        print(json.dumps({"test_only": True, "habits": all_habits}, ensure_ascii=False, indent=2))
        return
    bundles = [{
        "bundle_id": row["bundle_id"], "family": row["family"],
        "source_examples": row["source_examples"], "gpt_contextual_dialogue_discovery": True,
    } for row in all_habits]
    write_jsonl(args.dataset / "sources" / "habit_evidence_bundles.jsonl", bundles)
    write_jsonl(args.dataset / "private" / "grounded_habit_instances.jsonl", all_habits)
    write_jsonl(args.dataset / "review" / "habit_induction_all.jsonl", all_habits)
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(), "pipeline_revision": REVISION,
        "model": args.model, "reasoning_effort": args.reasoning_effort,
        "processed": len(bundles), "bundles": len(bundles), "accepted": len(all_habits),
        "decision_counts": {"accept": len(all_habits)},
        "accepted_family_counts": dict(Counter(row["family"] for row in all_habits)),
        "retrieval_counts": retrieval_counts, "multi_turn_context_windows_seen_by_gpt": True,
        "annotation_used_only_for_retrieval": True,
    }
    write_json(args.dataset / "reports" / "habit_induction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
