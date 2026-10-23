#!/usr/bin/env python
"""Run a main-experiment memory method and score its answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from evaluation.answering import AnswerGenerationError, QwenChoiceAnswerer, add_answer_args, answer_config_from_args
from evaluation.checkpoints import AnswerCheckpoint, atomic_json, context_identity, digest, validate_context_cache
from evaluation.answer_pool import answer_results, configured_answer_workers
from evaluation.dataset import DatasetContractError, load_dataset
from evaluation.io import read_jsonl, sha256_file, write_json, write_jsonl
from evaluation.scoring import score_predictions, write_score_outputs

FORBIDDEN_CONTEXT_FIELDS = {"choice_id", "gold_choice_id", "scores"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _method_config_snapshot(
    config_name: str | None, config_path: Path | None
) -> dict[str, Any] | None:
    if config_path is None:
        return None
    resolved = config_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Method config was not found: {resolved}")
    raw = resolved.read_bytes()
    parsed = yaml.safe_load(raw) or {}
    if not isinstance(parsed, dict):
        raise DatasetContractError(f"Method config must contain a YAML object: {resolved}")
    return {
        "name": config_name or resolved.stem,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "config": parsed,
    }


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
    command_text = args.adapter_command.format(input=shlex.quote(str(input_path)), output=shlex.quote(str(output_path)))
    command = shlex.split(command_text, posix=os.name != "nt")
    stdout_path = args.output_dir / "adapter.stdout.log"
    stderr_path = args.output_dir / "adapter.stderr.log"
    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=args.adapter_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def relay(source: Any, path: Path, console: Any) -> None:
        with path.open("w", encoding="utf-8") as sink:
            for line in iter(source.readline, ""):
                sink.write(line)
                sink.flush()
                print(line, end="", file=console, flush=True)
        source.close()

    relay_threads = [
        threading.Thread(
            target=relay,
            args=(process.stdout, stdout_path, sys.stdout),
            daemon=True,
        ),
        threading.Thread(
            target=relay,
            args=(process.stderr, stderr_path, sys.stderr),
            daemon=True,
        ),
    ]
    for thread in relay_threads:
        thread.start()
    try:
        returncode = process.wait(timeout=args.timeout_sec)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        raise
    finally:
        for thread in relay_threads:
            thread.join()
    runtime = {
        "entrypoint": command[2] if len(command) > 2 else "adapter",
        "elapsed_sec": round(time.time() - started, 3),
        "returncode": returncode,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
    }
    if returncode != 0:
        raise RuntimeError(
            f"Memory adapter failed with code {returncode}; see {stderr_path}"
        )
    if not output_path.is_file():
        raise RuntimeError(f"Memory adapter did not write {output_path}")
    return runtime


def run(args: argparse.Namespace) -> None:
    configure_method(args)
    # Fail on invalid execution settings before any expensive adapter work.
    answer_workers = configured_answer_workers(os.environ)
    run_started = time.perf_counter()
    started_at = _utc_now()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset(
        args.dataset_dir,
        domain_filter=args.domain_filter,
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
        "contract_version": "habitbench.e2e_run.v2",
        "method_name": args.method_name,
        "implementation": {
            "kind": args.implementation_kind,
            "source": args.implementation_source,
            "revision": args.implementation_revision,
            "note": args.adapter_note,
        },
        "dataset": bundle.manifest,
        "base_model": {**answer_config.public_dict(), "selected_protocol": args.answer_protocol},
        "method_config": _method_config_snapshot(
            args.method_config_name, args.method_config_path
        ),
        "memory_budget_env": {
            "context_window_tier": os.getenv("HABITBENCH_CONTEXT_WINDOW_TIER", "auto"),
            "model_context_tokens": int(os.getenv("HABITBENCH_MAX_MODEL_LEN", "40960")),
            "max_input_tokens": answer_config.max_input_tokens,
            "summary_max_tokens": int(os.getenv("HABITBENCH_COMPACT_SUMMARY_MAX_TOKENS", "8192")),
            "compactor_input_tokens": int(os.getenv("HABITBENCH_COMPACTOR_INPUT_TOKENS", "30000")),
        },
        "execution": {
            "status": "running",
            "started_at": started_at,
            "wall_clock_sec": None,
            "shard_index": args.user_shard_index,
            "shard_count": args.user_shard_count or 1,
            "timeout_sec": args.timeout_sec,
        },
        "files": {
            "method_input": input_path.name,
            "memory_contexts": contexts_path.name,
            "predictions": predictions_path.name,
        },
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    if args.prepare_only:
        manifest["execution"].update(
            {
                "status": "prepared",
                "finished_at": _utc_now(),
                "wall_clock_sec": round(time.perf_counter() - run_started, 3),
            }
        )
        write_json(args.output_dir / "run_manifest.json", manifest)
        print(json.dumps({"status": "prepared", **bundle.manifest}, indent=2))
        return
    if not args.adapter_command:
        raise DatasetContractError("--adapter-command is required unless --prepare-only is used")

    try:
        context_key = digest(context_identity(manifest, method_input))
        context_checkpoint = args.output_dir / "context_checkpoint.json"
        resume_partial = os.getenv("HABITBENCH_RESUME_PARTIAL", "0") == "1"
        cached = False
        if resume_partial and context_checkpoint.is_file():
            saved = json.loads(context_checkpoint.read_text(encoding="utf-8"))
            if saved.get("identity_sha256") != context_key or saved.get("contexts_sha256") != sha256_file(contexts_path):
                raise ValueError("Context checkpoint config/content mismatch; use a new output directory")
            manifest["adapter_runtime"] = saved["adapter_runtime"]
            cached = True
        cache_dir = os.getenv("HABITBENCH_REUSE_CONTEXT_DIR")
        if not cached and cache_dir:
            source = Path(cache_dir)
            provenance = validate_context_cache(source, manifest, method_input)
            if provenance is not None:
                shutil.copyfile(source / "memory_contexts.jsonl", contexts_path)
                if sha256_file(contexts_path) != provenance["contexts_sha256"]:
                    raise ValueError("Source context cache changed during copy")
                manifest["adapter_runtime"] = {"cache_reuse": provenance, "elapsed_sec": 0.0, "returncode": 0}
                cached = True
                print(f"memory_context_cache_reused source={source}; regenerating all answers", flush=True)
        if not cached:
            manifest["adapter_runtime"] = _run_adapter(args, input_path, contexts_path)
        context_rows = read_jsonl(contexts_path)
        probe_order = [probe["probe_id"] for probe in bundle.probes]
        contexts = validate_memory_contexts(context_rows, probe_order)
        if resume_partial:
            atomic_json(context_checkpoint, {
                "identity_sha256": context_key,
                "contexts_sha256": sha256_file(contexts_path),
                "adapter_runtime": manifest["adapter_runtime"],
            })

        answerer = QwenChoiceAnswerer(answer_config)
        checkpoint = AnswerCheckpoint(args.output_dir / "answers.checkpoint.json", {
            "contexts_sha256": sha256_file(contexts_path),
            "context_identity_sha256": context_key,
            "answer_config": {k: v for k, v in answer_config.public_dict().items() if k not in {"base_url", "api_key"}},
        }) if resume_partial else None
        predictions: list[dict[str, Any]] = []
        answer_failures: list[dict[str, Any]] = []
        continue_answer_errors = resume_partial and os.getenv("HABITBENCH_CONTINUE_ANSWER_ERRORS") == "1"
        answer_started = time.time()
        pending_probes = []
        for probe in bundle.probes:
            if checkpoint is not None and probe["probe_id"] in checkpoint.predictions:
                predictions.append(checkpoint.predictions[probe["probe_id"]])
            else:
                pending_probes.append(probe)
        def publish_answer_progress():
            if resume_partial:
                atomic_json(args.output_dir / "answer_progress.json", {
                    "observed_at": _utc_now(), "workers": answer_workers,
                    "completed": len(predictions), "failed": len(answer_failures),
                    "remaining": len(bundle.probes)-len(predictions)-len(answer_failures),
                    "total": len(bundle.probes), "elapsed_sec": round(time.time()-answer_started, 3),
                })
        publish_answer_progress()
        for probe, answer, error in answer_results(pending_probes, contexts, answerer, answer_workers):
            context_row = contexts[probe["probe_id"]]
            if error is not None:
                exc = error if isinstance(error, AnswerGenerationError) else AnswerGenerationError(
                    probe["probe_id"], [{"error_type":type(error).__name__, "error":str(error)[:1000]}])
                answer_failures.append({"probe_id": exc.probe_id, "attempts": exc.attempts})
                atomic_json(args.output_dir / "answer_failure.json", {
                    "probe_id": exc.probe_id, "attempts": exc.attempts,
                    "completed_predictions": len(predictions),
                    "failures": answer_failures,
                })
                publish_answer_progress()
                if continue_answer_errors:
                    print(f"answer_failed probe_id={exc.probe_id} continuing=true failures={len(answer_failures)}", flush=True)
                    continue
                raise exc
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
            if checkpoint is not None:
                checkpoint.put(predictions[-1])
            publish_answer_progress()
            index = len(predictions)+len(answer_failures)
            if args.progress_every and (index == 1 or index % args.progress_every == 0):
                print(
                    f"answer_progress completed={index} total={len(bundle.probes)} "
                    f"elapsed_sec={time.time() - answer_started:.1f}",
                    flush=True,
                )
        positions = {probe["probe_id"]:index for index,probe in enumerate(bundle.probes)}
        predictions.sort(key=lambda row:positions[row["probe_id"]])
        write_jsonl(predictions_path, predictions)
        if answer_failures:
            atomic_json(args.output_dir / "answer_failure.json", {
                "failures": answer_failures, "completed_predictions": len(predictions),
                "total_probes": len(bundle.probes), "all_probes_attempted": True,
            })
            manifest["answer_runtime"] = {"predictions": len(predictions), "failed_probes": len(answer_failures),
                                          "all_probes_attempted": True, "elapsed_sec": round(time.time() - answer_started, 3)}
            # Never mark a partial shard complete or score an incomplete subset.
            raise AnswerGenerationError(answer_failures[0]["probe_id"], answer_failures[0]["attempts"])
        if resume_partial:
            atomic_json(args.output_dir / "answer_failure.json", {"failures": [], "completed_predictions": len(predictions), "all_probes_attempted": True})

        detailed, metrics, metric_rows = score_predictions(
            predictions, bundle, args.method_name
        )
        write_score_outputs(args.output_dir, detailed, metrics, metric_rows)
        manifest["answer_runtime"] = {
            "elapsed_sec": round(time.time() - answer_started, 3),
            "predictions": len(predictions),
            "workers": answer_workers,
        }
        manifest["result"] = metrics["overall"]
    except BaseException as exc:
        manifest["execution"].update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "wall_clock_sec": round(time.perf_counter() - run_started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise
    else:
        manifest["execution"].update(
            {
                "status": "succeeded",
                "finished_at": _utc_now(),
                "wall_clock_sec": round(time.perf_counter() - run_started, 3),
            }
        )
        write_json(args.output_dir / "run_manifest.json", manifest)
        print(json.dumps(metrics["overall"], indent=2, sort_keys=True))



MAIN_METHODS = ("full_memory", "mem0", "amem", "memos", "memrl", "lightmem", "letta", "mirix", "secom", "no_memory")


def configure_method(args):
    """Select the included implementation and share model settings with its subprocess."""
    root = Path(__file__).resolve().parent
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.method_name not in MAIN_METHODS and not args.adapter_command:
        raise ValueError("Unknown main-experiment method")
    if args.answer_protocol == "main":
        from evaluation.answering_main import QwenChoiceAnswerer as MainAnswerer, answer_config_from_args as main_config
        global QwenChoiceAnswerer, answer_config_from_args
        QwenChoiceAnswerer, answer_config_from_args = MainAnswerer, main_config
        if configured_answer_workers(os.environ) != 1:
            raise ValueError("The main reader uses serial answer generation")
    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["HABITBENCH_SERVED_MODEL"] = args.base_model
    os.environ["HABITBENCH_LLM_MODEL"] = args.base_model_path
    if not args.prepare_only and args.method_name not in {"full_memory", "no_memory"} and args.embedding_model == "BAAI/bge-m3":
        from huggingface_hub import snapshot_download
        args.embedding_model = snapshot_download("BAAI/bge-m3", revision="5617a9f61b028005a4858fdac845db406aefb181")
    os.environ["HABITBENCH_EMBED_MODEL"] = args.embedding_model
    os.environ["HABITBENCH_EMBED_DEVICE"] = args.embedding_device
    os.environ["HABITBENCH_USER_WORKERS"] = str(args.user_workers)
    args.adapter_cwd = root
    args.implementation_source = "included implementation"
    args.implementation_revision = "submission"
    args.method_config_name = args.method_name
    prefix = shlex.quote(sys.executable)
    if args.adapter_command:
        return
    if args.method_name == "no_memory":
        command = ["-m", "evaluation.controls", "--mode", "no_memory"]
        args.implementation_kind = "control"
    elif args.method_name == "full_memory":
        tier = {8000: "8k", 16000: "16k", 32000: "32k", 40000: "40k", 64000: "64k", 128000: "128k"}.get(args.max_input_tokens, "custom")
        command = ["-m", "evaluation.compact_history", "--tokenizer-path", args.base_model_path,
                   "--context-window-tier", tier, "--custom-max-input-tokens", str(args.max_input_tokens),
                   "--model-context-tokens", str(max(args.max_input_tokens + args.answer_max_tokens, 40960))]
        args.implementation_kind = "control"
    elif args.method_name == "secom":
        command = ["-m", "secom_adapter", "--embedding-model", args.embedding_model,
                   "--embedding-device", args.embedding_device]
        args.implementation_kind = "official_adapted"
    else:
        args.method_config_path = root / "configs" / "method_config" / (args.method_name + ".yaml")
        command = ["-m", "memory_adapter", "--method-config", str(args.method_config_path),
                   "--user-workers", str(args.user_workers)]
        args.implementation_kind = "benchmark_reproduction"
    args.adapter_command = prefix + " " + " ".join(shlex.quote(str(part)) for part in command) + " --input {input} --output {output}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", "--method-name", dest="method_name", required=True, help="Included method name, or a name for --adapter-command.")
    parser.add_argument("--answer-protocol", choices=["current", "main"], default="main")
    parser.add_argument("--embedding-model", default=os.getenv("HABITBENCH_EMBED_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--embedding-device", default=os.getenv("HABITBENCH_EMBED_DEVICE", "cuda"))
    parser.add_argument("--user-workers", type=int, default=1)
    parser.add_argument("--adapter-command")
    parser.add_argument("--adapter-cwd", type=Path)
    parser.add_argument("--timeout-sec", type=int, default=172_800)
    parser.add_argument(
        "--implementation-kind",
        choices=[
            "benchmark_reproduction",
            "official_adapted",
            "retrieval_baseline",
            "control",
        ],
        default="benchmark_reproduction",
    )
    parser.add_argument("--implementation-source", default="included implementation")
    parser.add_argument("--implementation-revision", default="submission")
    parser.add_argument("--adapter-note", default="")
    parser.add_argument("--method-config-name")
    parser.add_argument("--method-config-path", type=Path)
    parser.add_argument(
        "--domain-filter",
        help="Evaluate only this public domain while retaining the source dataset files.",
    )
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
