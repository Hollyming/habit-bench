#!/usr/bin/env python
"""Create a portable task plan for user-sharded HABIT-Bench evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.core.dataset import load_dataset
from eval.core.io import sha256_file, write_json


DEFAULT_DATASETS = {
    "food": PROJECT_ROOT / "domain/food/food_habit_lifelines_stress_v2",
    "finance_software": PROJECT_ROOT
    / "domain/finance-software/habit_bench_multidogo_finance_software_scope_consistent_v1.3",
}
METHOD_CONFIGS = {
    method: f"{method}_qwen3-8b_adapted"
    for method in ("mem0", "amem", "memos", "memrl", "lightmem", "letta", "mirix")
}
LOCAL_METHOD_CONFIGS = {
    "full_memory": PROJECT_ROOT / "configs/methods/full_memory.yaml",
    "full_history": PROJECT_ROOT / "configs/methods/full_memory.yaml",
    "graphiti": PROJECT_ROOT / "configs/methods/graphiti_bge_m3_qwen3.yaml",
    "secom": PROJECT_ROOT / "configs/methods/secom_bge_m3_qwen3.yaml",
    "omem": PROJECT_ROOT / "configs/methods/omem_bge_m3_qwen3.yaml",
}
BGE_M3_METHODS = set(METHOD_CONFIGS) | {"graphiti", "secom", "omem"}
BGE_M3_ID = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_PATH = Path("/plm-shared/zhangjunming/Workspace/models/bge-m3")
BGE_M3_DIM = 1024
PLAN_FIELDS = (
    "task_id",
    "method",
    "dataset_name",
    "dataset_dir",
    "method_output_root",
    "shard_index",
    "shard_count",
)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dataset_overrides(values: list[str]) -> dict[str, Path]:
    datasets = dict(DEFAULT_DATASETS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--dataset must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        datasets[name.strip()] = Path(raw_path).expanduser().resolve()
    return datasets


def _metadata(values: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--metadata must be KEY=VALUE, got {value!r}")
        key, item = value.split("=", 1)
        if not key.strip():
            raise ValueError("--metadata key cannot be empty")
        metadata[key.strip()] = item
    return metadata


def create_plan(args: argparse.Namespace) -> list[dict[str, str | int]]:
    registry = json.loads((PROJECT_ROOT / "eval/methods.json").read_text(encoding="utf-8"))
    methods = _split_csv(args.methods)
    unknown_methods = set(methods) - set(registry)
    if not methods or unknown_methods:
        raise ValueError(f"Unknown or empty methods: {sorted(unknown_methods)}")

    datasets = _dataset_overrides(args.dataset)
    selected_datasets = _split_csv(args.datasets)
    unknown_datasets = set(selected_datasets) - set(datasets)
    if not selected_datasets or unknown_datasets:
        raise ValueError(f"Unknown or empty datasets: {sorted(unknown_datasets)}")
    if args.shards < 1:
        raise ValueError("--shards must be positive")

    output_root = args.output_root.expanduser().resolve()
    rows: list[dict[str, str | int]] = []
    dataset_user_counts = {
        dataset_name: int(load_dataset(datasets[dataset_name]).manifest["users"])
        for dataset_name in selected_datasets
    }
    for method in methods:
        for dataset_name in selected_datasets:
            dataset_dir = datasets[dataset_name]
            user_count = dataset_user_counts[dataset_name]
            if args.shards > user_count:
                raise ValueError(
                    f"Dataset {dataset_name} has {user_count} users; "
                    f"cannot create {args.shards} nonempty shards"
                )
            method_output_root = output_root / dataset_name / method
            for shard_index in range(args.shards):
                rows.append(
                    {
                        "task_id": len(rows),
                        "method": method,
                        "dataset_name": dataset_name,
                        "dataset_dir": str(dataset_dir),
                        "method_output_root": str(method_output_root),
                        "shard_index": shard_index,
                        "shard_count": args.shards,
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        required=True,
        help="Comma-separated methods. Controls run only when explicitly listed.",
    )
    parser.add_argument(
        "--datasets",
        default="food,finance_software",
        help="Comma-separated dataset aliases.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add or override a dataset alias; may be repeated.",
    )
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Plan metadata JSON; defaults to PLAN with .manifest.json suffix.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Record launcher or cluster metadata in the plan manifest.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def _validate_bge_m3(config: dict, *, config_path: Path) -> None:
    embedding = config.get("embedding")
    expected = {
        "provider": "local",
        "model": BGE_M3_ID,
        "revision": BGE_M3_REVISION,
        "model_path": str(BGE_M3_PATH),
        "dim": BGE_M3_DIM,
    }
    if not isinstance(embedding, dict):
        raise ValueError(f"Missing embedding object in {config_path}")
    mismatches = {
        key: {"expected": value, "actual": embedding.get(key)}
        for key, value in expected.items()
        if embedding.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Active method config must use the pinned BGE-M3 profile: "
            f"{config_path}: {mismatches}"
        )

    model_config_path = BGE_M3_PATH / "config.json"
    identity_path = BGE_M3_PATH / "HABIT_MODEL_INFO.json"
    weight_path = BGE_M3_PATH / "pytorch_model.bin"
    if (
        not model_config_path.is_file()
        or not identity_path.is_file()
        or not weight_path.is_file()
    ):
        raise FileNotFoundError(
            f"Incomplete BGE-M3 snapshot at {BGE_M3_PATH}; "
            "config.json, HABIT_MODEL_INFO.json and pytorch_model.bin are required"
        )
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    if model_config.get("hidden_size") != BGE_M3_DIM:
        raise ValueError(
            f"BGE-M3 hidden_size mismatch: expected {BGE_M3_DIM}, "
            f"got {model_config.get('hidden_size')!r}"
        )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity.get("model_id") != BGE_M3_ID
        or identity.get("revision") != BGE_M3_REVISION
        or identity.get("dense_embedding_dimension") != BGE_M3_DIM
    ):
        raise ValueError(f"BGE-M3 identity marker does not match the pinned profile: {identity}")


def _method_configs(methods: list[str]) -> dict[str, dict | None]:
    config_root = (
        PROJECT_ROOT / "third_party/medmemorybench/configs/method_config"
    )
    records: dict[str, dict | None] = {}
    for method in methods:
        config_name = METHOD_CONFIGS.get(method)
        if config_name is not None:
            path = config_root / f"{config_name}.yaml"
        elif method in LOCAL_METHOD_CONFIGS:
            path = LOCAL_METHOD_CONFIGS[method]
            config_name = path.stem
        else:
            records[method] = None
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Method config was not found: {path}")
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"Method config must contain a YAML object: {path}")
        if method in BGE_M3_METHODS:
            _validate_bge_m3(parsed, config_path=path)
        records[method] = {
            "name": config_name,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "config": parsed,
        }
    return records


def _bge_m3_snapshot() -> dict:
    identity_path = BGE_M3_PATH / "HABIT_MODEL_INFO.json"
    config_path = BGE_M3_PATH / "config.json"
    weight_path = BGE_M3_PATH / "pytorch_model.bin"
    return {
        "identity": json.loads(identity_path.read_text(encoding="utf-8")),
        "identity_path": str(identity_path),
        "identity_sha256": sha256_file(identity_path),
        "transformers_config_sha256": sha256_file(config_path),
        "weight_path": str(weight_path),
        "weight_size_bytes": weight_path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    if args.plan.exists() and not args.force:
        raise FileExistsError(f"Plan already exists; pass --force to replace it: {args.plan}")
    rows = create_plan(args)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    with args.plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PLAN_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    dataset_dirs = list(dict.fromkeys(str(row["dataset_dir"]) for row in rows))
    method_configs = _method_configs(methods)
    manifest_path = args.manifest or args.plan.with_suffix(".manifest.json")
    manifest = {
        "contract_version": "habitbench.shard_plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "output_root": str(args.output_root.expanduser().resolve()),
        "task_count": len(rows),
        "shard_count": args.shards,
        "methods": method_configs,
        "models": {
            "embedding": (
                _bge_m3_snapshot()
                if any(method in BGE_M3_METHODS for method in methods)
                else None
            )
        },
        "datasets": {
            str(Path(path).resolve()): load_dataset(Path(path)).manifest
            for path in dataset_dirs
        },
        "project": {
            "root": str(PROJECT_ROOT),
            **_git_state(),
        },
        "launcher": _metadata(args.metadata),
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "plan": str(args.plan),
                "manifest": str(manifest_path),
                "tasks": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
