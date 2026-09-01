#!/usr/bin/env python3
"""Run one LoCoMo method/sample unit with the vendored evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MED_ROOT_DEFAULT = PROJECT_ROOT / "third_party" / "medmemorybench"
METHOD_CONFIGS = {
    "long_context": "long_context_qwen3-8b_locomo",
    "bm25_rag": "bm25_rag_qwen3-8b_locomo",
    "embedding_rag": "embedding_rag_qwen3-8b_locomo",
    "mem0": "mem0_qwen3-8b_adapted",
    "amem": "amem_qwen3-8b_adapted",
    "memos": "memos_qwen3-8b_adapted",
    "memrl": "memrl_qwen3-8b_adapted",
    "lightmem": "lightmem_qwen3-8b_adapted",
    "letta": "letta_qwen3-8b_adapted",
    "mirix": "mirix_qwen3-8b_adapted",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _latest_query_answer(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("*_query_answer.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _load_components(args: argparse.Namespace):
    med_root = args.med_repo.expanduser().resolve()
    if str(med_root) not in sys.path:
        sys.path.insert(0, str(med_root))
    from src.config import ConfigLoader, DatasetConfig
    from benchmarks.locomo.evaluator import LoCoMoEvaluator

    config_name = METHOD_CONFIGS[args.method]
    config = ConfigLoader(project_root=med_root).load_method_config(config_name)
    config.model.name = args.model_name
    config.model.api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    config.model.base_url = args.base_url.rstrip("/")
    if config.embedding is not None:
        config.embedding.model = "BAAI/bge-m3"
        config.embedding.model_path = str(args.embedding_model_path.expanduser().resolve())
        config.embedding.dim = 1024
        # The Qwen vLLM process owns the visible GPU.  Memory-method embedding
        # encoders stay on CPU so one sample never competes with the reader
        # server for H200 memory.  This does not alter retrieval semantics.
        config.agent_params["embedding_device"] = "cpu"
        for key in ("amem_embedding_model", "embedding_model_path"):
            if key in config.agent_params:
                config.agent_params[key] = str(args.embedding_model_path.expanduser().resolve())
    if args.method == "lightmem":
        # The adapted config predates the H-cluster mount and may still carry
        # the old ClusterX absolute path. Pass the resolved H model explicitly;
        # LightMemAgent also performs the same fallback for other callers.
        h_lightmem_path = os.environ.get("HABITBENCH_LIGHTMEM_MODEL")
        if h_lightmem_path:
            config.agent_params["topic_segmenter_model_path"] = h_lightmem_path
    state_root = args.output_dir / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    config.agent_params["storage_root"] = str(state_root)
    config.agent_params["persistence_root"] = str(state_root / "letta")
    # Keep native method imports/configuration isolated from the user's home.
    os.environ["HOME"] = str(args.output_dir / "home")
    Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)

    data_file = args.dataset_file.expanduser().resolve()
    dataset_config = DatasetConfig.from_dict(
        {
            "dataset_name": "locomo",
            "description": "Official LoCoMo 10-sample benchmark",
            "language": "en",
            "data": {"root_dir": str(data_file.parent), "data_file": data_file.name},
            "evaluation": {
                "mode": "independent",
                "sample_ids": [args.sample_id],
                "max_samples": 1,
                "include_images": True,
                "memory_chunk_size": 32000,
            },
            "output": {"save_intermediate": True, "save_retrieved_context": True},
        }
    )
    return LoCoMoEvaluator(
        method_config=config,
        dataset_config=dataset_config,
        output_dir=args.output_dir,
        dry_run=False,
        verbose=True,
        resume=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_CONFIGS), required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", default=os.environ.get("HABITBENCH_SERVED_MODEL", "Qwen3-8B"))
    parser.add_argument(
        "--llm-model-path",
        type=Path,
        default=Path(os.environ.get("HABITBENCH_LLM_MODEL", "/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/models/habitbench/Qwen3-8B")),
    )
    parser.add_argument(
        "--embedding-model-path",
        type=Path,
        default=Path(os.environ.get("HABITBENCH_EMBED_MODEL", "/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/models/habitbench/bge-m3")),
    )
    parser.add_argument("--med-repo", type=Path, default=MED_ROOT_DEFAULT)
    parser.add_argument("--attempt", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = args.output_dir / "locomo_task_result.json"
    # A retry gets a clean evaluator/state directory. The root marker remains
    # the single resumable completion record consumed by the distributed queue.
    work_dir = args.output_dir if args.attempt == 1 else args.output_dir / f"attempt-{args.attempt}"
    work_dir.mkdir(parents=True, exist_ok=True)
    running = {
        "contract_version": "habitbench.locomo_task.v1",
        "status": "running",
        "method": args.method,
        "sample_id": args.sample_id,
        "attempt": args.attempt,
        "started_at": _utc_now(),
        "output_dir": str(args.output_dir),
    }
    _atomic_json(args.output_dir / "locomo_task_runtime.json", running)
    try:
        if not args.dataset_file.is_file():
            raise FileNotFoundError(f"LoCoMo dataset file not found: {args.dataset_file}")
        if not args.llm_model_path.is_dir():
            raise FileNotFoundError(f"Qwen3-8B model path not found: {args.llm_model_path}")
        if not args.embedding_model_path.is_dir():
            raise FileNotFoundError(f"BGE-M3 model path not found: {args.embedding_model_path}")

        task_args = argparse.Namespace(**vars(args))
        task_args.output_dir = work_dir
        evaluator = _load_components(task_args)
        report = evaluator.evaluate()
        query_answer = _latest_query_answer(work_dir)
        if query_answer is None:
            raise RuntimeError("LoCoMo evaluator completed without a query-answer artifact")

        # Import the frozen official postprocessor after the evaluator has
        # finished.  The source report remains untouched and receives a second
        # artifact with official per-question scores.
        med_root = args.med_repo.expanduser().resolve()
        if str(med_root) not in sys.path:
            sys.path.insert(0, str(med_root))
        from scripts.recompute_locomo_official import recompute

        official_path = work_dir / "locomo_official.json"
        official_summary = recompute(query_answer, official_path)
        payload = {
            **running,
            "status": "succeeded",
            "finished_at": _utc_now(),
            "report": report.to_dict(),
            "official_locomo": official_summary,
            "query_answer": str(query_answer),
            "official_artifact": str(official_path),
            "artifacts_root": str(work_dir),
        }
        _atomic_json(marker, payload)
        # Keep the lightweight runtime marker truthful as well.  The marker
        # consumed by the queue contains the full evaluator report, whereas
        # this file is used by operators for progress inspection.  Previously
        # it stayed at ``status=running`` forever, which made completed LoCoMo
        # samples look active after a retry or failed suite.
        _atomic_json(
            args.output_dir / "locomo_task_runtime.json",
            {
                **running,
                "status": "succeeded",
                "finished_at": payload["finished_at"],
                "attempt": args.attempt,
                "artifacts_root": str(work_dir),
                "query_answer": str(query_answer),
                "official_artifact": str(official_path),
                "official_mean_f1": official_summary["summary"]["mean_official_score"],
            },
        )
        print(json.dumps({"status": "succeeded", "method": args.method, "sample_id": args.sample_id, "official_mean_f1": official_summary["summary"]["mean_official_score"]}, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        error_path = args.output_dir / "locomo_task_error.log"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        payload = {
            **running,
            "status": "failed",
            "finished_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": str(error_path),
        }
        _atomic_json(marker, payload)
        _atomic_json(args.output_dir / "locomo_task_runtime.json", payload)
        print(f"LoCoMo task failed method={args.method} sample={args.sample_id}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
