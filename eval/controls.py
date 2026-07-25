#!/usr/bin/env python
"""No-memory and token-bounded long-context controls for HABIT-Bench."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.context_windows import WINDOW_TIER_CHOICES, resolve_context_window


DEFAULT_MODEL_PATH = "/plm-shared/zhangjunming/Workspace/models/Qwen3-8B"
DEFAULT_MODEL_CONTEXT_TOKENS = 40_960
SESSION_SEPARATOR = "\n\n"


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
        f"[SESSION_INDEX={session['session_index']}]\n"
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


@dataclass(frozen=True)
class HistorySelection:
    context: str
    evidence_session_ids: list[str]
    full_history_tokens: int
    context_tokens: int
    dropped_sessions: int
    truncated: bool
    partial_oldest_session: bool


def _encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _token_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    """Tokenize each session once; fast tokenizers can batch this operation."""

    if callable(tokenizer):
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
        return [len(ids) for ids in input_ids]
    return [len(_encode(tokenizer, text)) for text in texts]


def _partial_session_tail(
    session: dict[str, Any], tokenizer: Any, max_history_tokens: int
) -> str:
    """Retain a marker plus the newest token tail if one session exceeds the budget."""

    header = (
        f"[SESSION_ID={session['session_id']}]\n"
        f"[SESSION_INDEX={session['session_index']}]\n"
        f"[TIMESTAMP={session.get('timestamp')}]\n"
        "[EARLIER_CONTENT_TRUNCATED]\n"
    )
    header_ids = _encode(tokenizer, header)
    if len(header_ids) >= max_history_tokens:
        return tokenizer.decode(
            header_ids[:max_history_tokens], skip_special_tokens=True
        )
    body = "\n".join(
        f"{message['role']}: {message['content']}" for message in session["messages"]
    )
    body_budget = max_history_tokens - len(header_ids)
    body_ids = _encode(tokenizer, body)
    return header + tokenizer.decode(body_ids[-body_budget:], skip_special_tokens=True)


def select_recent_history(
    sessions: list[dict[str, Any]], tokenizer: Any, max_history_tokens: int
) -> HistorySelection:
    """Keep all visible history when possible, otherwise whole recent sessions.

    The retained sessions are returned in chronological order. Only the oldest
    retained session may be partially truncated, and only when a single session
    itself is larger than the complete history budget.
    """

    if max_history_tokens < 1:
        raise ValueError("max_history_tokens must be positive")
    if not sessions:
        return HistorySelection("", [], 0, 0, 0, False, False)

    rendered = [render_session(session) for session in sessions]
    full_context = SESSION_SEPARATOR.join(rendered)
    full_tokens = len(_encode(tokenizer, full_context))
    if full_tokens <= max_history_tokens:
        return HistorySelection(
            context=full_context,
            evidence_session_ids=[session["session_id"] for session in sessions],
            full_history_tokens=full_tokens,
            context_tokens=full_tokens,
            dropped_sessions=0,
            truncated=False,
            partial_oldest_session=False,
        )

    token_lengths = _token_lengths(tokenizer, rendered)
    separator_tokens = len(_encode(tokenizer, SESSION_SEPARATOR))
    selected_start = len(sessions)
    selected_tokens_estimate = 0
    for index in range(len(sessions) - 1, -1, -1):
        additional = token_lengths[index]
        if selected_start < len(sessions):
            additional += separator_tokens
        if selected_tokens_estimate + additional > max_history_tokens:
            break
        selected_start = index
        selected_tokens_estimate += additional

    partial = False
    if selected_start == len(sessions):
        selected_start = len(sessions) - 1
        selected_context = _partial_session_tail(
            sessions[selected_start], tokenizer, max_history_tokens
        )
        partial = True
    else:
        selected_context = SESSION_SEPARATOR.join(rendered[selected_start:])

        # Independent per-session tokenization differs slightly from encoding
        # the joined text at BPE boundaries. Enforce the exact hard budget.
        context_tokens = len(_encode(tokenizer, selected_context))
        while context_tokens > max_history_tokens and selected_start < len(sessions) - 1:
            selected_start += 1
            selected_context = SESSION_SEPARATOR.join(rendered[selected_start:])
            context_tokens = len(_encode(tokenizer, selected_context))

        # Recover an older complete session when boundary effects left room.
        while selected_start > 0:
            candidate = SESSION_SEPARATOR.join(rendered[selected_start - 1 :])
            candidate_tokens = len(_encode(tokenizer, candidate))
            if candidate_tokens > max_history_tokens:
                break
            selected_start -= 1
            selected_context = candidate
            context_tokens = candidate_tokens

    context_tokens = len(_encode(tokenizer, selected_context))
    if context_tokens > max_history_tokens:
        # Tokenizer decode/re-encode is not guaranteed to be perfectly
        # idempotent. Enforce the hard bound after constructing the partial tail.
        ids = _encode(tokenizer, selected_context)
        selected_context = tokenizer.decode(
            ids[-max_history_tokens:], skip_special_tokens=True
        )
        context_tokens = len(_encode(tokenizer, selected_context))

    evidence = [
        session["session_id"] for session in sessions[selected_start:]
    ]
    return HistorySelection(
        context=selected_context,
        evidence_session_ids=evidence,
        full_history_tokens=full_tokens,
        context_tokens=context_tokens,
        dropped_sessions=selected_start,
        truncated=True,
        partial_oldest_session=partial,
    )


def load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["no_memory", "full_memory", "full_history"],
        required=True,
    )
    parser.add_argument(
        "--tokenizer-path",
        default=os.getenv("HABITBENCH_LLM_MODEL", DEFAULT_MODEL_PATH),
    )
    parser.add_argument(
        "--context-window-tier",
        choices=WINDOW_TIER_CHOICES,
        default=os.getenv("HABITBENCH_CONTEXT_WINDOW_TIER", "auto"),
    )
    parser.add_argument(
        "--model-context-tokens",
        type=int,
        default=int(
            os.getenv(
                "HABITBENCH_MAX_MODEL_LEN",
                str(DEFAULT_MODEL_CONTEXT_TOKENS),
            )
        ),
    )
    parser.add_argument(
        "--custom-max-input-tokens",
        type=int,
        default=(
            int(os.environ["HABITBENCH_MAX_INPUT_TOKENS"])
            if os.environ.get("HABITBENCH_MAX_INPUT_TOKENS")
            else None
        ),
    )
    parser.add_argument(
        "--reserved-prompt-tokens",
        type=int,
        default=(
            int(os.environ["HABITBENCH_FULL_MEMORY_RESERVED_TOKENS"])
            if os.environ.get("HABITBENCH_FULL_MEMORY_RESERVED_TOKENS")
            else None
        ),
    )
    parser.add_argument(
        "--max-history-tokens",
        type=int,
        default=(
            int(os.environ["HABITBENCH_FULL_MEMORY_MAX_TOKENS"])
            if os.environ.get("HABITBENCH_FULL_MEMORY_MAX_TOKENS")
            else None
        ),
        help="Explicit history-budget override; normally the selected tier sets it.",
    )
    args = parser.parse_args()

    payload = read_json(args.input)
    tokenizer = None if args.mode == "no_memory" else load_tokenizer(args.tokenizer_path)
    window = (
        None
        if args.mode == "no_memory"
        else resolve_context_window(
            args.context_window_tier,
            args.model_context_tokens,
            custom_max_input_tokens=args.custom_max_input_tokens,
            reserved_prompt_tokens=args.reserved_prompt_tokens,
            max_history_tokens=args.max_history_tokens,
        )
    )
    selection_cache: dict[tuple[str, int], HistorySelection] = {}
    rows: list[dict[str, Any]] = []
    for probe in payload["probes"]:
        sessions = visible_sessions(probe, payload["sessions_by_user"])
        if args.mode == "no_memory":
            selection = HistorySelection("", [], 0, 0, 0, False, False)
        else:
            cache_key = (
                probe["user_id"],
                int(probe["visible_history_scope"]["max_session_index"]),
            )
            if cache_key not in selection_cache:
                selection_cache[cache_key] = select_recent_history(
                    sessions, tokenizer, window.history_token_budget
                )
            selection = selection_cache[cache_key]

        rows.append(
            {
                "probe_id": probe["probe_id"],
                "memory_context": selection.context,
                "evidence_session_ids": selection.evidence_session_ids,
                "debug": {
                    "adapter": (
                        "no_memory"
                        if args.mode == "no_memory"
                        else "full_memory_recent_session_tail"
                    ),
                    "requested_mode": args.mode,
                    "strategy": (
                        "none"
                        if args.mode == "no_memory"
                        else "all_if_fit_else_recent_complete_sessions"
                    ),
                    "tokenizer_path": (
                        None if args.mode == "no_memory" else args.tokenizer_path
                    ),
                    "max_history_tokens": (
                        0
                        if args.mode == "no_memory"
                        else window.history_token_budget
                    ),
                    "context_window": (
                        None if window is None else window.public_dict()
                    ),
                    "visible_sessions": len(sessions),
                    "context_sessions": len(selection.evidence_session_ids),
                    "dropped_sessions": selection.dropped_sessions,
                    "full_history_tokens": selection.full_history_tokens,
                    "context_tokens": selection.context_tokens,
                    "truncated": selection.truncated,
                    "partial_oldest_session": selection.partial_oldest_session,
                },
                "cost": {
                    "visible_history_sessions": len(sessions),
                    "retrieved_sessions": len(selection.evidence_session_ids),
                    "full_history_tokens": selection.full_history_tokens,
                    "retrieved_tokens": selection.context_tokens,
                },
            }
        )
    write_jsonl(args.output, rows)

    runtime = {
        "mode": args.mode,
        "strategy": (
            "none"
            if args.mode == "no_memory"
            else "all_if_fit_else_recent_complete_sessions"
        ),
        "tokenizer_path": None if args.mode == "no_memory" else args.tokenizer_path,
        "context_window": None if window is None else window.public_dict(),
        "probes": len(rows),
        "unique_history_cutoffs": len(selection_cache),
        "truncated_probes": sum(
            bool(row["debug"]["truncated"]) for row in rows
        ),
    }
    (args.output.parent / "control_runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
