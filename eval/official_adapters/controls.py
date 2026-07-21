#!/usr/bin/env python
"""No-memory and bounded full-history context controls for HABIT-Bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def render_session(session: dict[str, Any]) -> str:
    messages = "\n".join(
        f"{message['role']}: {message['content']}" for message in session["messages"]
    )
    return (
        f"[SESSION_ID={session['session_id']}]\n"
        f"[TIMESTAMP={session.get('timestamp')}]\n{messages}"
    )


def visible_sessions(
    probe: dict[str, Any], sessions_by_user: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    cutoff = probe["visible_history_scope"]["max_session_index"]
    return [
        session
        for session in sessions_by_user[probe["user_id"]]
        if session["session_index"] <= cutoff
    ]


def bounded_recent_history(
    sessions: list[dict[str, Any]], max_context_chars: int
) -> tuple[str, list[str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    used = 0
    for session in reversed(sessions):
        text = render_session(session)
        if selected and used + len(text) > max_context_chars:
            break
        selected.append((session, text))
        used += len(text)
    selected.reverse()
    return "\n\n".join(text for _, text in selected), [
        session["session_id"] for session, _ in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["no_memory", "full_history"], required=True)
    parser.add_argument("--max-context-chars", type=int, default=120_000)
    args = parser.parse_args()

    payload = read_json(args.input)
    rows: list[dict[str, Any]] = []
    for probe in payload["probes"]:
        sessions = visible_sessions(probe, payload["sessions_by_user"])
        if args.mode == "no_memory":
            context, evidence = "", []
        else:
            context, evidence = bounded_recent_history(sessions, args.max_context_chars)
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "memory_context": context,
                "evidence_session_ids": evidence,
                "debug": {
                    "adapter": args.mode,
                    "visible_sessions": len(sessions),
                    "context_sessions": len(evidence),
                },
                "cost": {"retrieved_sessions": len(evidence)},
            }
        )
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
