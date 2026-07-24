#!/usr/bin/env python3
"""Restore the frozen user000 dossier after a same-ID profile regeneration.

The authoritative habit graph is the version history used by the already
frozen sessions, annotations, and probes. Public content and labels are never
rewritten by this utility.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
RELEASE = DATASET / "release_single_user_pilot"
USER_ID = "tm_pd_v04_user_000"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    release_dossier_path = RELEASE / "private" / "user_dossiers.jsonl"
    main_dossier_path = DATASET / "private" / "user_dossiers.jsonl"
    version_path = RELEASE / "private" / "habit_version_history.jsonl"
    release_rows = read_jsonl(release_dossier_path)
    if len(release_rows) != 1 or release_rows[0].get("user_id") != USER_ID:
        raise SystemExit("single-user release dossier is not uniquely user000")
    base = copy.deepcopy(release_rows[0])
    versions = [
        row for row in read_jsonl(version_path)
        if row.get("user_id") == USER_ID and int(row.get("version", 1)) == 1
    ]
    if len(versions) != 7:
        raise SystemExit(f"expected seven authoritative v1 habits, got {len(versions)}")
    habits = []
    for row in sorted(versions, key=lambda item: item["habit_id"]):
        habit = copy.deepcopy(row)
        habit.pop("version", None)
        habit.pop("effective_from_session", None)
        habits.append(habit)
    base["habits"] = habits
    base["longitudinal_plan"] = {
        "target_history_characters": 332000,
        "target_sessions": 126,
        "travel_tempo": (
            "Roughly 8 to 10 planning episodes per year over a little more than three years, "
            "with most work trips generating several short sessions and family trips generating "
            "longer comparison sessions."
        ),
        "typical_episode_shape": [
            "Initial trip framing with dates, destination, travelers, and constraints.",
            "Flight search or comparison, often followed by a refinement around carrier, connection quality, return timing, or seat selection.",
            "Hotel shortlist with ratings and practical amenities noted.",
            "Decision or booking confirmation, sometimes after checking a missing hotel detail.",
            "Occasional follow-up for schedule changes, check-in, seat availability, or a last-mile planning question.",
        ],
    }
    base["travel_contexts"] = [
        {
            "context_key": "solo_client_travel",
            "description": "Recurring 2- to 4-night domestic trips to healthcare client sites, usually combining flights, a practical hotel, and early or tightly scheduled workdays.",
        },
        {
            "context_key": "conference_and_internal_meetings",
            "description": "Several annual conferences, implementation workshops, and company meetings where flight reliability and hotel workability matter.",
        },
        {
            "context_key": "family_and_personal_trips",
            "description": "Occasional weekends and school-break trips with her spouse and child, where some travel defaults carry over but family logistics can override them.",
        },
    ]
    base["habit_interactions"] = [
        "For flights, connection quality is a primary screen; Delta receives only a soft ranking boost when it remains otherwise comparable.",
        "Afternoon return timing is a recurring soft default, especially after work or family activities, but schedule feasibility can override it.",
        "A window seat is a background comfort preference and should not determine the itinerary when stronger constraints conflict.",
        "For hotels, the customer-rating floor is the reliability screen; included breakfast and convenient dining access are softer practical layers.",
        "When a strong hotel option has an unclear amenity, verify it rather than inventing the detail or automatically rejecting the property.",
    ]
    base["consistency_restoration"] = {
        "reason": "same user_id was regenerated during multiuser profile expansion after the lifeline had been frozen",
        "authoritative_habit_source": "private/habit_version_history.jsonl version=1",
        "public_content_changed": False,
        "probe_content_or_gold_changed": False,
    }

    archive = DATASET / "work" / "archive" / "same_id_regenerated_user000_dossier"
    archive.mkdir(parents=True, exist_ok=True)
    if not (archive / "release_user_dossiers_before_restore.jsonl").exists():
        shutil.copy2(release_dossier_path, archive / "release_user_dossiers_before_restore.jsonl")
    if not (archive / "multiuser_dossiers_before_restore.jsonl").exists():
        shutil.copy2(main_dossier_path, archive / "multiuser_dossiers_before_restore.jsonl")

    write_jsonl(release_dossier_path, [base])
    main_rows = read_jsonl(main_dossier_path)
    replaced = False
    for index, row in enumerate(main_rows):
        if row.get("user_id") == USER_ID:
            main_rows[index] = copy.deepcopy(base)
            replaced = True
    if not replaced:
        raise SystemExit("multiuser dossier file has no user000 row")
    write_jsonl(main_dossier_path, main_rows)
    print(json.dumps({
        "status": "restored",
        "release_dossier_count": 1,
        "multiuser_dossier_count": len(main_rows),
        "user000_sessions": 126,
        "user000_habits": [
            [row["habit_id"], row["family"], row["testable"]] for row in habits
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
