#!/usr/bin/env python
"""Run one memory adapter, answer with Qwen3-8B, and score exact-choice accuracy."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from eval.core.answering import QwenChoiceAnswerer, add_answer_args, answer_config_from_args
from eval.core.dataset import DatasetContractError, load_dataset
from eval.core.io import read_jsonl, write_json, write_jsonl
from eval.core.scoring import score_predictions, write_score_outputs


FORBIDDEN_CONTEXT_FIELDS = {"choice_id", "gold_choice_id", "scores"}


def validate_memory_contexts(
    rows: list[dict[str, Any]], probe_ids: list[str]
) -> dict[str, dict[str, Any]]:
    expected = set(probe_ids)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        forbidden = FORBIDDEN_CONTEXT_FIELDS.intersection(row)
        if forbidden:
            raise DatasetContractError(
                f"Memory adapter output contains answer/scoring fields {sorted(forbidden)}"
            )
        probe_id = str(row.get("probe_id", ""))
        if not probe_id:
            raise DatasetContractError("Memory adapter output is missing probe_id")
        if probe_id in by_id:
            raise DatasetContractError(f"Duplicate memory context for {probe_id}")
        if not isinstance(row.get("memory_context"), str):
            raise DatasetContractError(f"memory_context must be a string for {probe_id}")
        evidence = row.get("evidence_session_ids", [])
        if not isinstance(evidence, list) or not all(isinstance(value, str) for value in evidence):
            raise DatasetContractError(f"Invalid evidence_session_ids for {probe_id}")
        by_id[probe_id] = row
    missing = expected - set(by_id)
    extra = set(by_id) - expected
    if missing or extra:
        raise DatasetContractError(
            f"Memory-context coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    return by_id


def _run_adapter(args: argparse.Namespace, input_path: Path, output_path: Path) -> dict[str, Any]:
    command_text = args.adapter_command.format(input=str(input_path), output=str(output_path))
    command = shlex.split(command_text, posix=os.name != "nt")
    stdout_path = args.output_dir / "adapter.stdout.log"
    stderr_path = args.output_dir / "adapter.stderr.log"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=args.adapter_cwd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=args.timeout_sec,
            check=False,
        )
    runtime = {
        "command": command,
        "elapsed_sec": round(time.time() - started, 3),
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"Memory adapter failed with code {completed.returncode}; see {stderr_path}"
        )
    if not output_path.is_file():
        raise RuntimeError(f"Memory adapter did not write {output_path}")
    return runtime


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset(
        args.dataset_dir,
        max_users=args.max_users,
        max_probes=args.max_probes,
        user_shard_index=args.user_shard_index,
        user_shard_count=args.user_shard_count,
    )
    method_input = bundle.method_payload(args.method_name)
    input_path = args.output_dir / "method_input.json"
    contexts_path = args.output_dir / "memory_contexts.jsonl"
    predictions_path = args.output_dir / "predictions.jsonl"
    write_json(input_path, method_input)

    answer_config = answer_config_from_args(args)
    manifest: dict[str, Any] = {
        "contract_version": "habitbench.e2e_run.v1",
        "method_name": args.method_name,
        "implementation": {
            "kind": args.implementation_kind,
            "source": args.implementation_source,
            "revision": args.implementation_revision,
            "note": args.adapter_note,
        },
        "dataset": bundle.manifest,
        "base_model": answer_config.public_dict(),
        "files": {
            "method_input": str(input_path),
            "memory_contexts": str(contexts_path),
            "predictions": str(predictions_path),
        },
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", **bundle.manifest}, indent=2))
        return
    if not args.adapter_command:
        raise DatasetContractError("--adapter-command is required unless --prepare-only is used")

    manifest["adapter_runtime"] = _run_adapter(args, input_path, contexts_path)
    context_rows = read_jsonl(contexts_path)
    probe_order = [probe["probe_id"] for probe in bundle.probes]
    contexts = validate_memory_contexts(context_rows, probe_order)

    answerer = QwenChoiceAnswerer(answer_config)
    predictions: list[dict[str, Any]] = []
    answer_started = time.time()
    for index, probe in enumerate(bundle.probes, start=1):
        context_row = contexts[probe["probe_id"]]
        answer = answerer.answer(probe, context_row["memory_context"])
        predictions.append(
            {
                "probe_id": probe["probe_id"],
                "choice_id": answer["choice_id"],
                "evidence_session_ids": context_row.get("evidence_session_ids", []),
                "answer": answer,
                "memory_debug": context_row.get("debug", {}),
                "memory_cost": context_row.get("cost", {}),
            }
        )
        if args.progress_every and (index == 1 or index % args.progress_every == 0):
            print(
                f"answer_progress completed={index} total={len(bundle.probes)} "
                f"elapsed_sec={time.time() - answer_started:.1f}",
                flush=True,
            )
    write_jsonl(predictions_path, predictions)

    detailed, metrics, metric_rows = score_predictions(
        predictions, bundle, args.method_name
    )
    write_score_outputs(args.output_dir, detailed, metrics, metric_rows)
    manifest["answer_runtime"] = {
        "elapsed_sec": round(time.time() - answer_started, 3),
        "predictions": len(predictions),
    }
    manifest["result"] = metrics["overall"]
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps(metrics["overall"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--adapter-command")
    parser.add_argument("--adapter-cwd", type=Path)
    parser.add_argument("--timeout-sec", type=int, default=172_800)
    parser.add_argument(
        "--implementation-kind",
        choices=["official", "official_adapted", "benchmark_reproduction", "control"],
        required=True,
    )
    parser.add_argument("--implementation-source", required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--adapter-note", default="")
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
