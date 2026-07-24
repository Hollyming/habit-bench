#!/usr/bin/env python3
"""Run a MedMemoryBench source adapter under the HABIT memory-context contract."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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
        f"[TIMESTAMP={session.get('timestamp')}]\n"
        f"[DOMAIN={session.get('domain', 'unknown')}]\n"
        f"{messages}"
    )


def extract_session_ids(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            re.findall(r"\[SESSION_ID=([^\]]+)\]", text)
        )
    )


def require_successful_memory_build(result: Any, *, session_id: str) -> None:
    """Reject silent or partial memory-build failures from source adapters."""
    if result is None or not hasattr(result, "success"):
        raise TypeError(
            f"Memory build for session {session_id} returned an invalid result: "
            f"{type(result).__name__}"
        )
    if not result.success:
        extra = getattr(result, "extra", {}) or {}
        error = extra.get("error", "unknown memory build error")
        method = getattr(result, "method", "unknown") or "unknown"
        raise RuntimeError(
            f"Memory build failed for session {session_id} with method {method}: {error}"
        )


def default_med_repo() -> Path:
    return Path(__file__).resolve().parents[3] / "MedMemoryBench"


def load_med_components(med_repo: Path, config_name: str):
    if not (med_repo / "src" / "agent.py").is_file():
        raise FileNotFoundError(f"MedMemoryBench source tree not found: {med_repo}")
    sys.path.insert(0, str(med_repo))
    from src.agent import AgentManager
    from src.config import ConfigLoader, DatasetConfig

    method_config = ConfigLoader(project_root=med_repo).load_method_config(config_name)
    dataset_config = DatasetConfig(dataset_name="habitbench", language="en")
    return AgentManager, method_config, dataset_config


def empty_contexts(payload: dict[str, Any], config_name: str) -> list[dict[str, Any]]:
    return [
        {
            "probe_id": probe["probe_id"],
            "memory_context": "",
            "evidence_session_ids": [],
            "debug": {
                "adapter": "medmemorybench_structured_memory",
                "method_config": config_name,
                "dry_run_config": True,
            },
            "cost": {},
        }
        for probe in payload["probes"]
    ]


def run(args: argparse.Namespace) -> None:
    payload = read_json(args.input)
    if args.dry_run_config:
        write_jsonl(args.output, empty_contexts(payload, args.method_config))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_root = args.state_root or (args.output.parent / "medmemorybench_state")
    runtime_home = state_root / "runtime_home"
    runtime_home.mkdir(parents=True, exist_ok=True)
    # MIRIX and some vendored libraries derive their SQLite/config paths from
    # HOME at import time.  Isolate them per adapter task before importing any
    # MedMemoryBench method module.
    os.environ["HOME"] = str(runtime_home.resolve())

    AgentManager, base_method_config, dataset_config = load_med_components(
        args.med_repo.resolve(),
        args.method_config,
    )
    probes_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for probe in payload["probes"]:
        probes_by_user[probe["user_id"]].append(probe)
    for probes in probes_by_user.values():
        probes.sort(
            key=lambda row: (
                row["visible_history_scope"]["max_session_index"],
                row["probe_id"],
            )
        )

    predictions: dict[str, dict[str, Any]] = {}
    started = time.time()
    sessions_added = 0

    for context_id, (user_id, probes) in enumerate(sorted(probes_by_user.items()), start=1):
        method_config = copy.deepcopy(base_method_config)
        method_config.agent_params["storage_root"] = str(state_root / user_id)
        method_config.agent_params["persistence_root"] = str(
            state_root / user_id / "letta"
        )
        manager = AgentManager(method_config=method_config, dataset_config=dataset_config)
        manager.set_context_id(context_id)
        sessions = payload["sessions_by_user"][user_id]
        next_session = 0

        try:
            for probe in probes:
                cutoff = probe["visible_history_scope"]["max_session_index"]
                while (
                    next_session < len(sessions)
                    and sessions[next_session]["session_index"] <= cutoff
                ):
                    session = sessions[next_session]
                    build_result = manager.send_message(
                        render_session(session),
                        memorizing=True,
                        context_id=context_id,
                    )
                    require_successful_memory_build(
                        build_result,
                        session_id=session["session_id"],
                    )
                    next_session += 1
                    sessions_added += 1
                    if args.progress_every and sessions_added % args.progress_every == 0:
                        print(
                            f"memory_add_progress method={method_config.method_name} "
                            f"sessions={sessions_added} elapsed_sec={time.time() - started:.1f}",
                            file=sys.stderr,
                            flush=True,
                        )

                retrieval_started = time.time()
                retrieval = manager.retrieve(probe["query"], context_id=context_id)
                retrieval_elapsed = time.time() - retrieval_started
                memory_context = retrieval["memory_context"]
                evidence = extract_session_ids(memory_context)
                predictions[probe["probe_id"]] = {
                    "probe_id": probe["probe_id"],
                    "memory_context": memory_context,
                    "evidence_session_ids": evidence,
                    "debug": {
                        "adapter": "medmemorybench_structured_memory",
                        "method_config": args.method_config,
                        "method_name": method_config.method_name,
                        "retrieved_count": retrieval["retrieved_count"],
                        "retrieved_memories": retrieval["retrieved_memories"],
                        "added_until_session_index": cutoff,
                    },
                    "cost": {
                        "sessions_added_for_user": next_session,
                        "retrieval_elapsed_sec": round(retrieval_elapsed, 6),
                    },
                }
        finally:
            manager.reset()

    ordered = [predictions[probe["probe_id"]] for probe in payload["probes"]]
    write_jsonl(args.output, ordered)
    runtime = {
        "adapter": "medmemorybench_structured_memory",
        "method_config": args.method_config,
        "med_repo": str(args.med_repo.resolve()),
        "sessions_added": sessions_added,
        "predictions": len(ordered),
        "elapsed_sec": round(time.time() - started, 3),
    }
    (args.output.parent / "medmemorybench_adapter_runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--med-repo", type=Path, default=default_med_repo())
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run-config", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
