#!/usr/bin/env python3
"""Create a resumable method/sample plan for the official LoCoMo benchmark.

The vendored MedMemoryBench evaluator already implements the LoCoMo data parser,
method lifecycles, and official metric mapping.  This planner only decomposes
the run into independent sample units so multiple H-cluster workers can share
the work without changing a method's chronological memory behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = (
    "long_context,bm25_rag,embedding_rag,mem0,amem,memos,memrl,"
    "lightmem,letta,mirix"
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, str | bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": dirty}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path(
            os.environ.get(
                "HABITBENCH_LOCOMO_DATA_FILE",
                "/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/datasets/locomo/locomo10.json",
            )
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--samples", help="Optional comma-separated sample IDs")
    parser.add_argument("--model-name", default=os.environ.get("HABITBENCH_SERVED_MODEL", "Qwen3-8B"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            os.environ.get(
                "HABITBENCH_LLM_MODEL",
                "/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/models/habitbench/Qwen3-8B",
            )
        ),
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path(
            os.environ.get(
                "HABITBENCH_EMBED_MODEL",
                "/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/models/habitbench/bge-m3",
            )
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_file = args.dataset_file.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    plan_path = (args.plan or (output_root / "locomo_plan.tsv")).expanduser().resolve()
    manifest_path = plan_path.with_suffix(".manifest.json")

    if not dataset_file.is_file():
        raise FileNotFoundError(f"LoCoMo dataset file not found: {dataset_file}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Qwen3-8B model path not found: {args.model_path}")
    if not args.embedding_path.is_dir():
        raise FileNotFoundError(f"BGE-M3 model path not found: {args.embedding_path}")

    samples = json.loads(dataset_file.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("LoCoMo dataset must be a non-empty JSON list")
    available = [str(item.get("sample_id", f"sample_{i}")) for i, item in enumerate(samples)]
    requested_samples = (
        [value.strip() for value in args.samples.split(",") if value.strip()]
        if args.samples
        else available
    )
    unknown_samples = sorted(set(requested_samples) - set(available))
    if unknown_samples:
        raise ValueError(f"Unknown LoCoMo sample IDs: {unknown_samples}")

    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown_methods = sorted(set(methods) - set(METHOD_CONFIGS))
    if not methods or unknown_methods:
        raise ValueError(f"Unknown or empty LoCoMo methods: {unknown_methods}")
    for method in methods:
        config = PROJECT_ROOT / "third_party/medmemorybench/configs/method_config" / f"{METHOD_CONFIGS[method]}.yaml"
        if not config.is_file():
            raise FileNotFoundError(f"Method config not found for {method}: {config}")

    rows: list[dict[str, str | int]] = []
    task_count = len(methods) * len(requested_samples)
    ordinal = 0
    for method in methods:
        for sample_id in requested_samples:
            rows.append(
                {
                    "task_id": ordinal,
                    "method": method,
                    "dataset_name": "locomo",
                    "sample_id": sample_id,
                    "dataset_file": str(dataset_file),
                    "method_config": METHOD_CONFIGS[method],
                    "output_dir": str(output_root / method / sample_id),
                    # The shared H queue coordinator uses the same generic
                    # claim schema as HABIT shards.  Here each sample is one
                    # independent unit, so the values are only queue metadata.
                    "shard_index": ordinal,
                    "shard_count": task_count,
                }
            )
            ordinal += 1

    output_root.mkdir(parents=True, exist_ok=True)
    if plan_path.exists() and not args.force:
        existing = plan_path.read_text(encoding="utf-8")
        expected_header = "task_id\tmethod\tdataset_name\tsample_id\tdataset_file\tmethod_config\toutput_dir\tshard_index\tshard_count\n"
        if not existing.startswith(expected_header):
            raise FileExistsError(
                f"Existing LoCoMo plan has an incompatible schema: {plan_path}; use --force to replace"
            )
        print(json.dumps({"status": "reused", "plan": str(plan_path)}, indent=2))
        return 0

    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "method",
                "dataset_name",
                "sample_id",
                "dataset_file",
                "method_config",
                "output_dir",
                "shard_index",
                "shard_count",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "contract_version": "habitbench.locomo_plan.v1",
        "created_at": _utc_now(),
        "dataset": {
            "name": "locomo",
            "file": str(dataset_file),
            "sha256": _sha256(dataset_file),
            "samples": requested_samples,
            "sample_count": len(requested_samples),
            "source": "https://github.com/snap-research/locomo",
        },
        "methods": [
            {
                "name": method,
                "config": METHOD_CONFIGS[method],
                "config_path": str(
                    (PROJECT_ROOT / "third_party/medmemorybench/configs/method_config" / f"{METHOD_CONFIGS[method]}.yaml").resolve()
                ),
            }
            for method in methods
        ],
        "model": {
            "name": args.model_name,
            "path": str(args.model_path.expanduser().resolve()),
            "embedding_path": str(args.embedding_path.expanduser().resolve()),
            "backbone_alignment": "Qwen3-8B for all LoCoMo methods and reader calls; BGE-M3 for embedding methods",
        },
        "task_count": task_count,
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "git": _git_state(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "created", "plan": str(plan_path), "manifest": str(manifest_path), "tasks": task_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
