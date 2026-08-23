#!/usr/bin/env python3
"""Run a MedMemoryBench source adapter under the HABIT memory-context contract."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import os
import re
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _record_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def _append_lineage(mapping: dict[str, list[str]], key: str, session_id: str) -> None:
    if not key:
        return
    values = mapping.setdefault(key, [])
    if session_id not in values:
        values.append(session_id)


def record_memory_lineage(
    build_result: Any,
    session_id: str,
    id_lineage: dict[str, list[str]],
    text_lineage: dict[str, list[str]],
) -> None:
    """Attribute adapter-visible memory records to the session that created them.

    This does not change a memory method's storage or retrieval behavior. It records
    provenance only when the method's native build result exposes a stable memory ID
    or exact memory text.
    """
    collections = []
    for attribute in ("memory_entries", "all_passages"):
        value = getattr(build_result, attribute, None)
        if isinstance(value, list):
            collections.extend(value)
    for value in collections:
        if isinstance(value, str):
            _append_lineage(text_lineage, value.strip(), session_id)
            continue
        record = _record_dict(value)
        if not record:
            continue
        metadata = _record_dict(record.get("metadata"))
        event = str(record.get("event", "")).upper()
        identifiers = [
            str(candidate)
            for candidate in (
                record.get("id"),
                record.get("memory_id"),
                record.get("note_id"),
                metadata.get("id"),
                metadata.get("memory_id"),
                metadata.get("session_id"),
            )
            if candidate
        ]
        if event == "DELETE":
            for identifier in identifiers:
                id_lineage.pop(identifier, None)
            continue
        for identifier in identifiers:
            _append_lineage(id_lineage, identifier, session_id)
        for field in ("memory", "text", "content"):
            text = record.get(field)
            if isinstance(text, str) and text.strip():
                _append_lineage(text_lineage, text.strip(), session_id)


def attribute_retrieved_sessions(
    memory_context: str,
    retrieved_memories: Any,
    id_lineage: dict[str, list[str]],
    text_lineage: dict[str, list[str]],
) -> tuple[list[str], dict[str, int]]:
    """Resolve ranked source-session provenance without altering method outputs."""
    evidence: list[str] = []
    methods = {
        "direct_marker": 0,
        "native_metadata": 0,
        "memory_id_lineage": 0,
        "exact_text_lineage": 0,
    }

    def add(values: Iterable[str], source: str) -> None:
        before = len(evidence)
        for value in values:
            session_id = str(value)
            if session_id and session_id not in evidence:
                evidence.append(session_id)
        methods[source] += len(evidence) - before

    records = retrieved_memories if isinstance(retrieved_memories, list) else []
    for value in records:
        if isinstance(value, str):
            markers = extract_session_ids(value)
            if markers:
                add(markers, "direct_marker")
            elif value.strip() in text_lineage:
                add(text_lineage[value.strip()], "exact_text_lineage")
            continue
        record = _record_dict(value)
        if not record:
            continue
        text = next(
            (
                record[field].strip()
                for field in ("memory", "text", "content")
                if isinstance(record.get(field), str) and record[field].strip()
            ),
            "",
        )
        markers = extract_session_ids(text)
        if markers:
            add(markers, "direct_marker")
        metadata = _record_dict(record.get("metadata"))
        native_session_ids = []
        for candidate in (
            record.get("session_id"),
            metadata.get("session_id"),
            metadata.get("source_session_id"),
        ):
            if isinstance(candidate, str) and candidate:
                native_session_ids.append(candidate)
            elif isinstance(candidate, list):
                native_session_ids.extend(str(item) for item in candidate if item)
        add(native_session_ids, "native_metadata")
        for identifier in (
            record.get("id"),
            record.get("memory_id"),
            record.get("note_id"),
            metadata.get("id"),
            metadata.get("memory_id"),
        ):
            if identifier and str(identifier) in id_lineage:
                add(id_lineage[str(identifier)], "memory_id_lineage")
        if text and text in text_lineage:
            add(text_lineage[text], "exact_text_lineage")

    # Some methods expose only the rendered context rather than structured retrieval
    # records. Preserve its native rank order as a final direct-provenance source.
    add(extract_session_ids(memory_context), "direct_marker")
    return evidence, methods


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
    return Path(__file__).resolve().parents[2] / "third_party" / "medmemorybench"


def apply_runtime_model_paths(method_config: Any) -> dict[str, str]:
    """Apply the selected scale model and cluster-specific physical paths."""
    overrides: dict[str, str] = {}
    served_model = os.environ.get("HABITBENCH_SERVED_MODEL")
    if served_model:
        model = getattr(method_config, "model", None)
        if model is not None and getattr(model, "name", None) != served_model:
            model.name = served_model
            overrides["model.name"] = served_model
        agent_params = getattr(method_config, "agent_params", {})
        for key in ("amem_model", "memos_model"):
            if key in agent_params and agent_params[key] != served_model:
                agent_params[key] = served_model
                overrides[f"agent_params.{key}"] = served_model

    embedding_model = os.environ.get("HABITBENCH_EMBED_MODEL")
    if embedding_model:
        resolved = str(Path(embedding_model).expanduser().resolve())
        embedding = getattr(method_config, "embedding", None)
        if embedding is not None:
            embedding.model_path = resolved
            overrides["embedding.model_path"] = resolved
        agent_params = getattr(method_config, "agent_params", {})
        for key in ("amem_embedding_model", "embedding_model_path"):
            if key in agent_params:
                agent_params[key] = resolved
                overrides[f"agent_params.{key}"] = resolved

    lightmem_model = os.environ.get("HABITBENCH_LIGHTMEM_MODEL")
    if lightmem_model:
        resolved = str(Path(lightmem_model).expanduser().resolve())
        agent_params = getattr(method_config, "agent_params", {})
        if "topic_segmenter_model_path" in agent_params:
            agent_params["topic_segmenter_model_path"] = resolved
            overrides["agent_params.topic_segmenter_model_path"] = resolved
    return overrides


def load_med_components(med_repo: Path, config_name: str):
    if not (med_repo / "src" / "agent.py").is_file():
        raise FileNotFoundError(f"MedMemoryBench source tree not found: {med_repo}")
    med_repo_text = str(med_repo)
    if med_repo_text in sys.path:
        sys.path.remove(med_repo_text)
    sys.path.insert(0, med_repo_text)
    from src.agent import AgentManager
    from src.config import ConfigLoader, DatasetConfig

    method_config = ConfigLoader(project_root=med_repo).load_method_config(config_name)
    apply_runtime_model_paths(method_config)
    dataset_config = DatasetConfig(dataset_name="habitbench", language="en")
    return AgentManager, method_config, dataset_config


def group_user_jobs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Create deterministic, independent user jobs from one HABIT shard."""
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

    jobs = []
    for context_id, (user_id, probes) in enumerate(
        sorted(probes_by_user.items()),
        start=1,
    ):
        if user_id not in payload["sessions_by_user"]:
            raise KeyError(f"Missing sessions for probe user: {user_id}")
        jobs.append(
            {
                "context_id": context_id,
                "user_id": user_id,
                "probes": probes,
                "sessions": payload["sessions_by_user"][user_id],
            }
        )
    return jobs


def resolve_user_state_root(state_root: Path, user_id: str) -> Path:
    """Keep the existing per-user layout while rejecting path traversal."""
    root = state_root.resolve()
    candidate = (root / user_id).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"Unsafe user_id for state directory: {user_id!r}")
    return candidate


def process_user_job(
    job: dict[str, Any],
    *,
    AgentManager: Any,
    base_method_config: Any,
    dataset_config: Any,
    state_root: Path,
    config_name: str,
    progress_every: int,
) -> dict[str, Any]:
    """Process one user's chronological history in one isolated process."""
    user_id = job["user_id"]
    context_id = int(job["context_id"])
    probes = job["probes"]
    sessions = job["sessions"]
    user_state_root = resolve_user_state_root(state_root, user_id)

    method_config = copy.deepcopy(base_method_config)
    method_config.agent_params["storage_root"] = str(user_state_root)
    method_config.agent_params["persistence_root"] = str(user_state_root / "letta")

    started = time.time()
    cpu_started = time.process_time()
    init_started = time.time()
    manager = AgentManager(
        method_config=method_config,
        dataset_config=dataset_config,
    )
    manager.set_context_id(context_id)
    init_elapsed = time.time() - init_started

    predictions: dict[str, dict[str, Any]] = {}
    next_session = 0
    retrieval_elapsed_total = 0.0
    reset_elapsed = 0.0
    id_lineage: dict[str, list[str]] = {}
    text_lineage: dict[str, list[str]] = {}
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
                record_memory_lineage(
                    build_result,
                    session["session_id"],
                    id_lineage,
                    text_lineage,
                )
                next_session += 1
                if progress_every and next_session % progress_every == 0:
                    print(
                        f"memory_add_progress method={method_config.method_name} "
                        f"user={user_id} context={context_id} "
                        f"sessions={next_session} "
                        f"elapsed_sec={time.time() - started:.1f}",
                        file=sys.stderr,
                        flush=True,
                    )

            retrieval_started = time.time()
            retrieval = manager.retrieve(probe["query"], context_id=context_id)
            retrieval_elapsed = time.time() - retrieval_started
            retrieval_elapsed_total += retrieval_elapsed
            memory_context = retrieval["memory_context"]
            evidence, attribution = attribute_retrieved_sessions(
                memory_context,
                retrieval["retrieved_memories"],
                id_lineage,
                text_lineage,
            )
            predictions[probe["probe_id"]] = {
                "probe_id": probe["probe_id"],
                "memory_context": memory_context,
                "evidence_session_ids": evidence,
                "debug": {
                    "adapter": "medmemorybench_structured_memory",
                    "method_config": config_name,
                    "method_name": method_config.method_name,
                    "retrieved_count": retrieval["retrieved_count"],
                    "retrieved_memories": retrieval["retrieved_memories"],
                    "evidence_attribution": attribution,
                    "added_until_session_index": cutoff,
                },
                "cost": {
                    "sessions_added_for_user": next_session,
                    "retrieval_elapsed_sec": round(retrieval_elapsed, 6),
                },
            }
    finally:
        reset_started = time.time()
        manager.reset()
        reset_elapsed = time.time() - reset_started

    elapsed = time.time() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_id": user_id,
        "context_id": context_id,
        "predictions": predictions,
        "runtime": {
            "user_id": user_id,
            "context_id": context_id,
            "pid": os.getpid(),
            "sessions_added": next_session,
            "predictions": len(predictions),
            "init_elapsed_sec": round(init_elapsed, 3),
            "retrieval_elapsed_sec": round(retrieval_elapsed_total, 3),
            "reset_elapsed_sec": round(reset_elapsed, 3),
            "elapsed_sec": round(elapsed, 3),
            "process_cpu_sec": round(time.process_time() - cpu_started, 3),
            "process_max_rss_kib": int(usage.ru_maxrss),
        },
    }


def process_user_job_in_worker(
    job: dict[str, Any],
    *,
    med_repo: str,
    config_name: str,
    state_root: str,
    progress_every: int,
) -> dict[str, Any]:
    """Spawn-safe worker entrypoint; set HOME before importing any method."""
    resolved_state_root = Path(state_root).resolve()
    runtime_home = (
        resolved_state_root
        / "runtime_home"
        / f"worker-{os.getpid()}"
    )
    runtime_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(runtime_home)
    AgentManager, base_method_config, dataset_config = load_med_components(
        Path(med_repo),
        config_name,
    )
    return process_user_job(
        job,
        AgentManager=AgentManager,
        base_method_config=base_method_config,
        dataset_config=dataset_config,
        state_root=resolved_state_root,
        config_name=config_name,
        progress_every=progress_every,
    )


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
    if args.user_workers <= 0:
        raise ValueError(f"user_workers must be positive: {args.user_workers}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_root = (
        args.state_root or (args.output.parent / "medmemorybench_state")
    ).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    med_repo = args.med_repo.resolve()
    jobs = group_user_jobs(payload)
    effective_workers = min(args.user_workers, len(jobs)) if jobs else 1
    predictions: dict[str, dict[str, Any]] = {}
    started = time.time()
    user_runs: list[dict[str, Any]] = []

    if effective_workers == 1:
        runtime_home = state_root / "runtime_home"
        runtime_home.mkdir(parents=True, exist_ok=True)
        # Preserve the legacy single-process import boundary exactly. MIRIX
        # and some vendored libraries derive SQLite/config paths from HOME at
        # module import time.
        os.environ["HOME"] = str(runtime_home)
        AgentManager, base_method_config, dataset_config = load_med_components(
            med_repo,
            args.method_config,
        )
        for job in jobs:
            result = process_user_job(
                job,
                AgentManager=AgentManager,
                base_method_config=base_method_config,
                dataset_config=dataset_config,
                state_root=state_root,
                config_name=args.method_config,
                progress_every=args.progress_every,
            )
            predictions.update(result["predictions"])
            user_runs.append(result["runtime"])
    else:
        process_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=process_context,
        ) as executor:
            future_to_job = {
                executor.submit(
                    process_user_job_in_worker,
                    job,
                    med_repo=str(med_repo),
                    config_name=args.method_config,
                    state_root=str(state_root),
                    progress_every=args.progress_every,
                ): job
                for job in jobs
            }
            try:
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        for pending in future_to_job:
                            pending.cancel()
                        raise RuntimeError(
                            "User worker failed: "
                            f"user={job['user_id']} "
                            f"context={job['context_id']}"
                        ) from exc
                    predictions.update(result["predictions"])
                    user_runs.append(result["runtime"])
            finally:
                for future in future_to_job:
                    future.cancel()

    ordered = [predictions[probe["probe_id"]] for probe in payload["probes"]]
    write_jsonl(args.output, ordered)
    user_runs.sort(key=lambda row: row["context_id"])
    sessions_added = sum(int(row["sessions_added"]) for row in user_runs)
    elapsed = time.time() - started
    runtime = {
        "adapter": "medmemorybench_structured_memory",
        "method_config": args.method_config,
        "med_repo": str(med_repo),
        "user_workers_requested": args.user_workers,
        "user_workers_effective": effective_workers,
        "users": len(jobs),
        "sessions_added": sessions_added,
        "predictions": len(ordered),
        "elapsed_sec": round(elapsed, 3),
        "aggregate_user_elapsed_sec": round(
            sum(float(row["elapsed_sec"]) for row in user_runs),
            3,
        ),
        "sessions_per_sec": (
            round(sessions_added / elapsed, 6) if sessions_added else 0.0
        ),
        "user_runs": user_runs,
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
    parser.add_argument(
        "--user-workers",
        type=int,
        default=int(os.environ.get("HABITBENCH_MED_USER_WORKERS", "1")),
        help="Concurrent isolated user processes sharing the configured LLM endpoint",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run-config", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
