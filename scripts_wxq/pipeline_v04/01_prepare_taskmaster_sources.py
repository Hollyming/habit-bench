#!/usr/bin/env python3
"""Prepare Taskmaster-2 grounded source cards and habit-evidence bundles for v0.4."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_1" / "data" / "raw_taskmaster"
DEFAULT_OUT = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
SOURCE_DATASET = "google-research-datasets/Taskmaster-2"

ANNOTATION_FAMILY = {
    "flight_search.time_of_day": "departure_time",
    "flight_search.stops": "stop_tolerance",
    "flight_search.airline": "airline_choice",
    "flight_search.seat_location": "seat_location",
    "flight_search.seating_class": "cabin_class",
    "flight_search.price_range": "flight_price_tradeoff",
    "flight_search.total_fare": "flight_price_tradeoff",
    "flight_search.other_description": "flight_feature",
    "hotel_search.sub_location.hotel": "hotel_location_priority",
    "hotel_search.amenity": "hotel_amenity_priority",
    "hotel_search.star_rating": "hotel_quality_threshold",
    "hotel_search.customer_rating": "hotel_quality_threshold",
    "hotel_search.type.bed": "bed_configuration",
    "hotel_search. num.beds": "bed_configuration",
    "hotel_search.type.room": "room_configuration",
    "hotel_search.num.rooms": "room_configuration",
    "hotel_search.price_range": "hotel_price_tradeoff",
    "hotel_search.name.hotel": "hotel_brand_choice",
    "hotel_search.other_request": "hotel_feature",
}

PREFERENCE_RE = re.compile(
    r"\b(prefer|preference|would like|i(?:'d| would) rather|want|need|must|avoid|"
    r"non[- ]?stop|aisle|window|cheapest|budget|no more than|at most|at least|"
    r"morning|afternoon|evening|balcony|breakfast|parking|wi-?fi|pet[- ]friendly)\b",
    re.I,
)
HABITUAL_RE = re.compile(r"\b(always|usually|normally|generally|whenever|every time|tend to|my usual)\b", re.I)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def compact(value: Any, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value[:64] or "unspecified"


def normalize_value(family: str, segment: str, utterance: str) -> str:
    text = f"{segment} {utterance}".lower()
    if family == "departure_time":
        for name, pattern in [
            ("early_morning", r"early(?: in the)? morning|earliest"),
            ("morning", r"morning|a\.m\.|\bam\b"),
            ("afternoon", r"afternoon|midday|noon"),
            ("evening", r"evening|p\.m\.|\bpm\b"),
            ("overnight", r"overnight|red[- ]?eye|late night"),
        ]:
            if re.search(pattern, text):
                return name
    if family == "stop_tolerance":
        if re.search(r"non[- ]?stop|direct|straight through|zero stops", text):
            return "nonstop"
        if re.search(r"at most one|one (?:layover|stop)|1 (?:layover|stop)", text):
            return "at_most_one_stop"
        if re.search(r"layover|connection|stop", text):
            return "connections_allowed"
    if family == "seat_location":
        if "aisle" in text:
            return "aisle"
        if "window" in text:
            return "window"
    if family == "cabin_class":
        for value in ["first", "business", "premium economy", "economy", "coach"]:
            if value in text:
                return slug(value)
    if family in {"flight_price_tradeoff", "hotel_price_tradeoff"}:
        if re.search(r"price (?:doesn'?t|does not) matter|any price|no budget", text):
            return "price_flexible"
        if re.search(r"cheapest|tight budget|budget|under |less than|no more than|at most|max(?:imum)?", text):
            return "budget_sensitive"
        return "stated_price_band"
    if family == "hotel_quality_threshold":
        number = re.search(r"([1-5](?:\.\d)?)\s*(?:star|rating)", text)
        return f"at_least_{number.group(1).replace('.', '_')}" if number else "quality_threshold"
    if family == "hotel_location_priority":
        # A destination name is a transaction constraint, not a reusable
        # planning default.  Keep only transferable location *types* here.
        if re.search(r"beachfront|oceanfront|on the beach|near (?:the )?beach", text):
            return "beachfront_or_oceanfront"
        if re.search(r"downtown|city cent(?:er|re)|centrally located|central location", text):
            return "central_or_downtown"
        if re.search(r"near (?:the )?airport|airport hotel", text):
            return "near_airport"
        if re.search(r"walking distance|walkable|(?:can|could) walk", text):
            return "walkable_to_target"
        if re.search(r"near|close to|nearby|by the", text):
            return "near_target_venue"
        return "specific_destination_only"
    # Free-form values are grounded in the annotated segment; later LLM
    # induction may consolidate synonyms without losing provenance.
    return slug(segment)


def card_from_row(row: dict[str, Any], domain: str) -> dict[str, Any] | None:
    turns = []
    evidence = []
    user_chars = 0
    for utterance in row.get("utterances", []):
        role = "user" if utterance.get("speaker") == "USER" else "assistant"
        text = compact(utterance.get("text"), 900)
        if not text:
            continue
        turn_index = int(utterance.get("index", len(turns)))
        turns.append({"turn_index": turn_index, "role": role, "text": text})
        if role != "user":
            continue
        user_chars += len(text)
        for segment in utterance.get("segments", []):
            segment_text = compact(segment.get("text"), 180)
            for annotation in segment.get("annotations", []):
                annotation_name = annotation.get("name")
                family = ANNOTATION_FAMILY.get(annotation_name)
                if not family:
                    continue
                evidence.append(
                    {
                        "family": family,
                        "normalized_value": normalize_value(family, segment_text, text),
                        "annotation": annotation_name,
                        "segment": segment_text,
                        "turn_index": turn_index,
                        "user_utterance": text,
                        "preference_language": bool(PREFERENCE_RE.search(text)),
                        "habitual_language": bool(HABITUAL_RE.search(text)),
                    }
                )
    if len(turns) < 6 or user_chars < 100 or not evidence:
        return None
    unique = {}
    for item in evidence:
        key = (item["family"], item["normalized_value"], item["turn_index"], item["segment"])
        unique[key] = item
    evidence = list(unique.values())
    score = min(len(turns), 40) / 10 + len({x["family"] for x in evidence})
    score += 1.5 * sum(x["preference_language"] for x in evidence)
    score += 3.0 * sum(x["habitual_language"] for x in evidence)
    return {
        "source_dataset": SOURCE_DATASET,
        "source_domain": domain,
        "conversation_id": row["conversation_id"],
        "instruction_id": row.get("instruction_id"),
        "turns": turns,
        "preference_evidence": evidence,
        "quality_score": round(score, 3),
    }


def build_bundles(cards: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for card in cards:
        for evidence in card["preference_evidence"]:
            if evidence["normalized_value"] == "specific_destination_only":
                continue
            key = (evidence["family"], evidence["normalized_value"])
            current = grouped[key].get(card["conversation_id"])
            candidate = {"card": card, "evidence": evidence}
            if current is None or evidence["habitual_language"] or (
                evidence["preference_language"] and not current["evidence"]["preference_language"]
            ):
                grouped[key][card["conversation_id"]] = candidate

    candidates_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (family, value), by_conversation in grouped.items():
        rows = list(by_conversation.values())
        if len(rows) < 3:
            continue
        rows.sort(
            key=lambda item: (
                item["evidence"]["habitual_language"],
                item["evidence"]["preference_language"],
                item["card"]["quality_score"],
            ),
            reverse=True,
        )
        # Taskmaster can contain many conversations paraphrasing the same
        # instruction. Select at most one conversation per instruction so
        # those paraphrases never masquerade as independent evidence.
        best_by_instruction: dict[str, dict[str, Any]] = {}
        for item in rows:
            instruction_id = item["card"].get("instruction_id")
            if instruction_id and instruction_id not in best_by_instruction:
                best_by_instruction[instruction_id] = item
        independent_rows = list(best_by_instruction.values())
        if len(independent_rows) < 3:
            continue
        # Non-overlapping chunks prevent one strong instruction from
        # supporting several nominally different habit instances.
        for chunk_index, start in enumerate(range(0, len(independent_rows), 5)):
            chunk = independent_rows[start : start + 5]
            if len(chunk) < 3 or chunk_index >= 4:
                break
            digest = hashlib.sha256(
                (family + value + "|".join(x["card"]["conversation_id"] for x in chunk)).encode()
            ).hexdigest()[:12]
            candidates_by_family[family].append(
                {
                    "bundle_id": f"tm_v04_bundle_{digest}",
                    "family": family,
                    "seed_value": value,
                    "source_examples": [
                        {
                            "conversation_id": item["card"]["conversation_id"],
                            "instruction_id": item["card"]["instruction_id"],
                            "source_domain": item["card"]["source_domain"],
                            "turn_index": item["evidence"]["turn_index"],
                            "user_utterance": item["evidence"]["user_utterance"],
                            "annotated_segment": item["evidence"]["segment"],
                            "annotation": item["evidence"]["annotation"],
                            "preference_language": item["evidence"]["preference_language"],
                            "habitual_language": item["evidence"]["habitual_language"],
                        }
                        for item in chunk
                    ],
                    "independence": {
                        "distinct_conversation_ids": len({x["card"]["conversation_id"] for x in chunk}),
                        "distinct_instruction_ids": len({x["card"]["instruction_id"] for x in chunk}),
                    },
                }
            )
    for rows in candidates_by_family.values():
        rng.shuffle(rows)
    families = sorted(candidates_by_family)
    bundles = []
    while len(bundles) < target:
        progressed = False
        for family in families:
            if candidates_by_family[family]:
                bundles.append(candidates_by_family[family].pop())
                progressed = True
                if len(bundles) >= target:
                    break
        if not progressed:
            break
    return bundles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-bundles", type=int, default=72)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    raw_counts = {}
    for domain in ["flights", "hotels"]:
        path = args.raw_dir / f"{domain}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        raw_counts[domain] = len(rows)
        cards.extend(card for row in rows if (card := card_from_row(row, domain)) is not None)
    cards.sort(key=lambda row: (row["source_domain"], row["conversation_id"]))
    bundles = build_bundles(cards, args.target_bundles, args.seed)

    write_jsonl(args.output_dir / "sources" / "taskmaster_source_cards.jsonl", cards)
    write_jsonl(args.output_dir / "sources" / "habit_evidence_bundles.jsonl", bundles)
    manifest = {
        "dataset": "taskmaster_planning_defaults_v0_4",
        "status": "source_preparation_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": SOURCE_DATASET,
        "raw_data_dir": str(args.raw_dir.resolve()),
        "raw_conversation_counts": raw_counts,
        "source_cards": len(cards),
        "habit_evidence_bundles": len(bundles),
        "bundle_family_counts": dict(Counter(row["family"] for row in bundles)),
        "contract": {
            "minimum_source_conversations_per_bundle": 3,
            "minimum_distinct_instruction_ids_per_bundle": 3,
            "same_instruction_paraphrases_count_as_one_source": True,
            "exact_destinations_are_not_habit_values": True,
            "habit_family_is_taxonomy_only": True,
            "specific_habit_value_is_instance_level": True,
            "generation_only_no_formal_evaluation": True,
        },
    }
    reports = args.output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
