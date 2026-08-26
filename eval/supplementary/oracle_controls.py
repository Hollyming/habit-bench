#!/usr/bin/env python
"""Run diagnostic Oracle Evidence and Oracle Habit State controls.

These controls intentionally read private evaluation annotations.  They are
upper-bound diagnostics, not deployable memory methods, and are kept outside
the normal method registry and evaluation entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from eval.controls import render_session
from eval.core.answering import (
    QwenChoiceAnswerer,
    add_answer_args,
    answer_config_from_args,
)
from eval.core.dataset import DatasetBundle, DatasetContractError, load_dataset
from eval.core.io import write_json, write_jsonl
from eval.core.scoring import score_predictions, write_score_outputs


ORACLE_MODES = ("oracle_evidence", "oracle_habit_state")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _session_lookup(bundle: DatasetBundle) -> dict[str, dict[str, Any]]:
    return {
        str(session["session_id"]): session
        for sessions in bundle.sessions_by_user.values()
        for session in sessions
    }


def oracle_evidence_ids(key: dict[str, Any]) -> list[str]:
    """Return decisive evidence plus required temporal context.

    Finance/Software explicitly annotates non-binding evidence.  It is
    deliberately excluded from this oracle so that the control represents
    perfect evidence selection rather than a private-data history dump.
    """

    decisive = key.get("decision_evidence_session_ids")
    if decisive is not None:
        return _stable_unique(
            list(decisive or []) + list(key.get("temporal_context_session_ids") or [])
        )
    return _stable_unique(key.get("gold_evidence_session_ids") or [])


def _validated_oracle_sessions(
    probe: dict[str, Any],
    evidence_ids: list[str],
    lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cutoff = int(probe["visible_history_scope"]["max_session_index"])
    sessions: list[dict[str, Any]] = []
    for session_id in evidence_ids:
        session = lookup.get(session_id)
        if session is None:
            raise DatasetContractError(
                f"Oracle evidence for {probe['probe_id']} references missing session {session_id}"
            )
        if str(session["user_id"]) != str(probe["user_id"]):
            raise DatasetContractError(
                f"Oracle evidence {session_id} belongs to another user"
            )
        if int(session["session_index"]) > cutoff:
            raise DatasetContractError(
                f"Oracle evidence {session_id} occurs after the probe cutoff"
            )
        sessions.append(session)
    sessions.sort(key=lambda row: (int(row["session_index"]), str(row["session_id"])))
    return sessions


def build_oracle_evidence_context(
    probe: dict[str, Any],
    key: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence_ids = oracle_evidence_ids(key)
    sessions = _validated_oracle_sessions(probe, evidence_ids, lookup)
    context = "\n\n".join(render_session(session) for session in sessions)
    return {
        "probe_id": probe["probe_id"],
        "memory_context": (
            "[DIAGNOSTIC ORACLE EVIDENCE: private gold evidence only]\n\n"
            + (context or "[No decisive evidence sessions are annotated.]")
        ),
        # Preserve the private annotation's ranking: decisive sessions precede
        # temporal context.  The rendered text remains chronological.
        "evidence_session_ids": evidence_ids,
        "debug": {
            "diagnostic_only": True,
            "private_annotations_used": True,
            "oracle_kind": "evidence",
            "session_count": len(sessions),
        },
        "cost": {"oracle_selected_sessions": len(sessions)},
    }


def _controlled_habit_graph_state(key: dict[str, Any]) -> dict[str, Any]:
    """Render one controlled latent habit graph used by Food and Travel.

    Travel v16 reuses the single-habit graph contract for part of its probes,
    but has many more probe-type names than the original Food-only control.
    Select boundary/exception rules by semantic probe-type markers and expose
    the private required action consistently with the multi-habit signature
    branch.  No choice id or choice position is included.
    """

    graph = key.get("hidden_habit_graph") or {}
    probe_type = str(key.get("probe_type", "unknown"))
    if "exception" in probe_type:
        target_field = "exception_action"
    elif "boundary" in probe_type:
        target_field = "boundary_action"
    else:
        target_field = "default_action"
    required_action = (
        key.get("gold_action_text")
        or key.get("gold_action")
        or graph.get(target_field)
    )
    return {
        "habit_id": key.get("habit_id") or graph.get("habit_id"),
        "habit_name": graph.get("name"),
        "habit_family": key.get("habit_family") or graph.get("family"),
        "default_policy": {
            "condition": graph.get("default_condition"),
            "action": graph.get("default_action"),
        },
        "boundary_policy": {
            "condition": graph.get("boundary_condition"),
            "action": graph.get("boundary_action"),
        },
        "local_exception": {
            "condition": graph.get("exception_condition"),
            "action": graph.get("exception_action"),
        },
        "current_state": {
            "probe_type": probe_type,
            "applicable_rule": target_field,
            "required_action": required_action,
        },
    }


def _policy_signature(key: dict[str, Any]) -> dict[str, Any]:
    signatures = key.get("choice_policy_signatures") or {}
    gold_choice_id = str(key.get("gold_choice_id", ""))
    raw_signature = signatures.get(gold_choice_id) or {}
    if not isinstance(raw_signature, dict):
        return {}
    signature = raw_signature.get("variants", raw_signature)
    if not isinstance(signature, dict):
        return {}
    return {str(habit_id): value for habit_id, value in signature.items()}


def _finance_software_habit_state(key: dict[str, Any]) -> dict[str, Any]:
    signature = _policy_signature(key)
    target_habits = _stable_unique(
        list(key.get("target_habit_ids") or []) + list(signature)
    )
    return {
        "target_habits": [
            {
                "habit_id": habit_id,
                "active_policy_variant": signature.get(habit_id),
            }
            for habit_id in target_habits
        ],
        "target_state_times": key.get("target_state_times"),
        "required_action_description": key.get("gold_action_text")
        or key.get("gold_action"),
        "probe_type": key.get("probe_type"),
    }


def build_oracle_habit_state_context(
    probe: dict[str, Any], key: dict[str, Any]
) -> dict[str, Any]:
    if key.get("hidden_habit_graph"):
        state = _controlled_habit_graph_state(key)
        schema = "controlled_habit_graph"
    else:
        state = _finance_software_habit_state(key)
        schema = "finance_software_policy_signature"
    context = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)
    return {
        "probe_id": probe["probe_id"],
        "memory_context": (
            "[DIAGNOSTIC ORACLE HABIT STATE: private latent-state annotation; "
            "no choice identifier]\n\n" + context
        ),
        "evidence_session_ids": [],
        "debug": {
            "diagnostic_only": True,
            "private_annotations_used": True,
            "oracle_kind": "habit_state",
            "state_schema": schema,
        },
        "cost": {"oracle_selected_sessions": 0},
    }


def build_oracle_contexts(
    bundle: DatasetBundle, mode: str
) -> list[dict[str, Any]]:
    if mode not in ORACLE_MODES:
        raise ValueError(f"Unknown oracle mode: {mode}")
    lookup = _session_lookup(bundle)
    rows: list[dict[str, Any]] = []
    for probe in bundle.probes:
        key = bundle.keys[probe["probe_id"]]
        if mode == "oracle_evidence":
            row = build_oracle_evidence_context(probe, key, lookup)
        else:
            row = build_oracle_habit_state_context(probe, key)
        rows.append(row)
    return rows


def _relabel_method(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                new
                if key == "method_name" and child == old
                else _relabel_method(child, old, new)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_relabel_method(child, old, new) for child in value]
    return value


def score_oracle_predictions(
    predictions: list[dict[str, Any]],
    bundle: DatasetBundle,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Reuse the unchanged primary scorer with correct Oracle semantics.

    Oracle Habit State does not perform retrieval.  Scoring it as an ordinary
    method would manufacture zero Recall@5 values.  We therefore invoke the
    existing no-retrieval control path and relabel only the method name; answer
    accuracy and every capability grouping still come from the same scorer.
    """

    scoring_method = "no_memory" if mode == "oracle_habit_state" else mode
    detailed, metrics, metric_rows = score_predictions(
        predictions, bundle, scoring_method
    )
    if scoring_method != mode:
        detailed = _relabel_method(detailed, scoring_method, mode)
        metrics = _relabel_method(metrics, scoring_method, mode)
        metric_rows = _relabel_method(metric_rows, scoring_method, mode)
    return detailed, metrics, metric_rows


def run(args: argparse.Namespace) -> None:
    started_at = _utc_now()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset(
        args.dataset_dir,
        domain_filter=args.domain_filter,
        max_users=args.max_users,
        max_probes=args.max_probes,
        user_shard_index=args.user_shard_index,
        user_shard_count=args.user_shard_count,
    )
    answer_config = answer_config_from_args(args)
    contexts = build_oracle_contexts(bundle, args.mode)
    context_path = args.output_dir / "memory_contexts.jsonl"
    predictions_path = args.output_dir / "predictions.jsonl"
    manifest_path = args.output_dir / "supplementary_manifest.json"
    write_jsonl(context_path, contexts)
    manifest: dict[str, Any] = {
        "contract_version": "habitbench.supplementary_oracle.v1",
        "experiment_role": "diagnostic_upper_bound",
        "method_name": args.mode,
        "implementation": {
            "kind": "control",
            "source": "eval.supplementary.oracle_controls",
            "revision": "v1",
            "note": "Private-label diagnostic upper bound; not a deployable memory method.",
        },
        "method_config": None,
        "warning": (
            "Uses private evaluation annotations. Do not compare as a deployable "
            "memory method and do not include in the normal method registry."
        ),
        "dataset": bundle.manifest,
        "base_model": answer_config.public_dict(),
        "execution": {
            "status": "contexts_prepared",
            "started_at": started_at,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "shard_index": args.user_shard_index,
            "shard_count": args.user_shard_count or 1,
        },
        "files": {
            "memory_contexts": str(context_path),
            "predictions": str(predictions_path),
        },
    }
    write_json(manifest_path, manifest)
    # Also expose the standard filename so the existing, unchanged shard
    # merger can combine Oracle shards after validating dataset hashes.
    write_json(args.output_dir / "run_manifest.json", manifest)
    if args.prepare_only:
        manifest["execution"].update(
            {
                "status": "prepared",
                "finished_at": _utc_now(),
                "wall_clock_sec": round(time.perf_counter() - started, 3),
            }
        )
        write_json(manifest_path, manifest)
        write_json(args.output_dir / "run_manifest.json", manifest)
        print(json.dumps(manifest["execution"], indent=2, sort_keys=True))
        return

    try:
        answerer = QwenChoiceAnswerer(answer_config)
        by_probe = {row["probe_id"]: row for row in contexts}
        predictions: list[dict[str, Any]] = []
        answer_started = time.perf_counter()
        for index, probe in enumerate(bundle.probes, start=1):
            context_row = by_probe[probe["probe_id"]]
            answer = answerer.answer(probe, context_row["memory_context"])
            predictions.append(
                {
                    "probe_id": probe["probe_id"],
                    "choice_id": answer["choice_id"],
                    "evidence_session_ids": context_row["evidence_session_ids"],
                    "answer": answer,
                    "memory_debug": context_row["debug"],
                    "memory_cost": context_row["cost"],
                }
            )
            if args.progress_every and (
                index == 1 or index % args.progress_every == 0
            ):
                print(
                    f"oracle_progress mode={args.mode} completed={index} "
                    f"total={len(bundle.probes)} "
                    f"elapsed_sec={time.perf_counter() - answer_started:.1f}",
                    flush=True,
                )
        write_jsonl(predictions_path, predictions)
        detailed, metrics, metric_rows = score_oracle_predictions(
            predictions, bundle, args.mode
        )
        write_score_outputs(args.output_dir, detailed, metrics, metric_rows)
        manifest["answer_runtime"] = {
            "elapsed_sec": round(time.perf_counter() - answer_started, 3),
            "predictions": len(predictions),
        }
        manifest["execution"].update(
            {
                "status": "succeeded",
                "finished_at": _utc_now(),
                "wall_clock_sec": round(time.perf_counter() - started, 3),
            }
        )
        manifest["result"] = metrics["overall"]
    except BaseException as exc:
        manifest["execution"].update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "wall_clock_sec": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(manifest_path, manifest)
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise
    write_json(manifest_path, manifest)
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest["result"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=ORACLE_MODES, required=True)
    parser.add_argument("--domain-filter")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-probes", type=int)
    parser.add_argument("--user-shard-index", type=int)
    parser.add_argument("--user-shard-count", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    add_answer_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
