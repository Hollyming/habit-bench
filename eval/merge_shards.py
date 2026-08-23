#!/usr/bin/env python
"""Validate, merge, and rescore completed user-sharded HABIT-Bench runs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.core.dataset import DatasetBundle, DatasetContractError, load_dataset
from eval.core.io import read_jsonl, write_json, write_jsonl
from eval.core.scoring import score_predictions, write_score_outputs
from eval.run import validate_memory_contexts


SHARD_DIR_PATTERN = re.compile(r"^shard_(\d+)_of_(\d+)$")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetContractError(f"Required shard file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _comparable_base_model(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "base_url"}


def _sum_present(values: list[float | int | None]) -> float:
    return round(sum(float(value) for value in values if value is not None), 3)


def _observed_window_sec(shards: list[dict[str, Any]]) -> float | None:
    starts = [row.get("started_at") for row in shards if row.get("started_at")]
    finishes = [row.get("finished_at") for row in shards if row.get("finished_at")]
    if not starts or not finishes:
        return None
    started = min(datetime.fromisoformat(value) for value in starts)
    finished = max(datetime.fromisoformat(value) for value in finishes)
    return round((finished - started).total_seconds(), 3)


def _discover_shards(shard_root: Path, expected_shards: int) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    for path in shard_root.iterdir() if shard_root.is_dir() else []:
        if not path.is_dir():
            continue
        match = SHARD_DIR_PATTERN.match(path.name)
        if not match:
            continue
        index, count = (int(value) for value in match.groups())
        if count != expected_shards:
            raise DatasetContractError(
                f"Shard directory {path} declares {count} shards; expected {expected_shards}"
            )
        if index in found:
            raise DatasetContractError(f"Duplicate shard index {index}: {found[index]} and {path}")
        found[index] = path

    expected = set(range(expected_shards))
    missing = expected - set(found)
    extra = set(found) - expected
    if missing or extra:
        raise DatasetContractError(
            f"Shard coverage mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return [(index, found[index]) for index in range(expected_shards)]


def _load_merged_dataset_view(
    dataset_dir: Path,
    *,
    domain_filter: str | None,
    max_users: int | None,
    max_probes: int | None,
) -> DatasetBundle:
    """Load the global subset whose disjoint user shards are being merged."""

    return load_dataset(
        dataset_dir,
        domain_filter=domain_filter,
        max_users=max_users,
        max_probes=max_probes,
    )


def merge_shards(
    dataset_dir: Path,
    shard_root: Path,
    output_dir: Path,
    method_name: str,
    expected_shards: int,
    domain_filter: str | None = None,
    max_users: int | None = None,
    max_probes: int | None = None,
) -> dict[str, Any]:
    if expected_shards < 1:
        raise DatasetContractError("expected_shards must be positive")

    bundle = _load_merged_dataset_view(
        dataset_dir,
        domain_filter=domain_filter,
        max_users=max_users,
        max_probes=max_probes,
    )
    shards = _discover_shards(shard_root, expected_shards)
    expected_hashes = {
        field: bundle.manifest[field]
        for field in (
            "public_lifelines_sha256",
            "public_probes_sha256",
            "private_probe_key_sha256",
        )
    }

    predictions: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    reference_implementation: dict[str, Any] | None = None
    reference_base_model: dict[str, Any] | None = None
    reference_method_config: dict[str, Any] | None = None

    for index, shard_dir in shards:
        manifest = _read_json(shard_dir / "run_manifest.json")
        if manifest.get("method_name") != method_name:
            raise DatasetContractError(
                f"Method mismatch in {shard_dir}: {manifest.get('method_name')!r}"
            )
        if "result" not in manifest:
            raise DatasetContractError(f"Shard run is incomplete: {shard_dir}")

        dataset_manifest = manifest.get("dataset") or {}
        if dataset_manifest.get("domain_filter") != bundle.manifest.get(
            "domain_filter"
        ):
            raise DatasetContractError(
                f"Dataset domain filter mismatch in {shard_dir}: "
                f"{dataset_manifest.get('domain_filter')!r}"
            )
        for field, expected_value in expected_hashes.items():
            if dataset_manifest.get(field) != expected_value:
                raise DatasetContractError(f"Dataset hash mismatch for {field} in {shard_dir}")
        subset = dataset_manifest.get("subset") or {}
        if subset.get("user_shard_index") != index:
            raise DatasetContractError(f"Shard index mismatch in {shard_dir}")
        if subset.get("user_shard_count") != expected_shards:
            raise DatasetContractError(f"Shard count mismatch in {shard_dir}")
        if subset.get("max_users") != max_users:
            raise DatasetContractError(
                f"max-users mismatch in {shard_dir}: "
                f"{subset.get('max_users')!r} != {max_users!r}"
            )
        if subset.get("max_probes") != max_probes:
            raise DatasetContractError(
                f"max-probes mismatch in {shard_dir}: "
                f"{subset.get('max_probes')!r} != {max_probes!r}"
            )

        implementation = manifest.get("implementation") or {}
        base_model = manifest.get("base_model") or {}
        method_config = manifest.get("method_config")
        if reference_implementation is None:
            reference_implementation = implementation
            reference_base_model = base_model
            reference_method_config = method_config
        elif implementation != reference_implementation or _comparable_base_model(
            base_model
        ) != _comparable_base_model(reference_base_model or {}):
            raise DatasetContractError(f"Implementation/base-model mismatch in {shard_dir}")
        elif method_config != reference_method_config:
            raise DatasetContractError(f"Method-config mismatch in {shard_dir}")

        shard_predictions = read_jsonl(shard_dir / "predictions.jsonl")
        shard_contexts = read_jsonl(shard_dir / "memory_contexts.jsonl")
        predictions.extend(shard_predictions)
        contexts.extend(shard_contexts)
        execution = manifest.get("execution") or {}
        shard_records.append(
            {
                "index": index,
                "directory": str(shard_dir),
                "users": dataset_manifest.get("users"),
                "sessions": dataset_manifest.get("sessions"),
                "probes": dataset_manifest.get("probes"),
                "adapter_elapsed_sec": (manifest.get("adapter_runtime") or {}).get(
                    "elapsed_sec"
                ),
                "answer_elapsed_sec": (manifest.get("answer_runtime") or {}).get(
                    "elapsed_sec"
                ),
                "wall_clock_sec": execution.get("wall_clock_sec"),
                "started_at": execution.get("started_at"),
                "finished_at": execution.get("finished_at"),
                "host": execution.get("host"),
                "cuda_visible_devices": execution.get("cuda_visible_devices"),
            }
        )

    probe_order = [probe["probe_id"] for probe in bundle.probes]
    contexts_by_id = validate_memory_contexts(contexts, probe_order)
    detailed, metrics, metric_rows = score_predictions(predictions, bundle, method_name)

    predictions_by_id = {row["probe_id"]: row for row in predictions}
    ordered_predictions = [predictions_by_id[probe_id] for probe_id in probe_order]
    ordered_contexts = [contexts_by_id[probe_id] for probe_id in probe_order]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "memory_contexts.jsonl", ordered_contexts)
    write_jsonl(output_dir / "predictions.jsonl", ordered_predictions)
    write_score_outputs(output_dir, detailed, metrics, metric_rows)

    wall_times = [row.get("wall_clock_sec") for row in shard_records]
    timing = {
        "shard_count": expected_shards,
        "shard_wall_clock_sum_sec": _sum_present(wall_times),
        "shard_wall_clock_max_sec": (
            round(max(float(value) for value in wall_times if value is not None), 3)
            if any(value is not None for value in wall_times)
            else None
        ),
        "observed_window_sec": _observed_window_sec(shard_records),
        "adapter_compute_sum_sec": _sum_present(
            [row.get("adapter_elapsed_sec") for row in shard_records]
        ),
        "answer_compute_sum_sec": _sum_present(
            [row.get("answer_elapsed_sec") for row in shard_records]
        ),
    }
    merge_manifest = {
        "contract_version": "habitbench.shard_merge.v2",
        "method_name": method_name,
        "implementation": reference_implementation,
        "method_config": reference_method_config,
        "base_model": {
            **(reference_base_model or {}),
            "base_url": "<per-shard-local-endpoint>",
        },
        "dataset": bundle.manifest,
        "expected_shards": expected_shards,
        "shard_root": str(shard_root),
        "shards": shard_records,
        "timing": timing,
        "result": metrics["overall"],
    }
    write_json(output_dir / "merge_manifest.json", merge_manifest)
    return merge_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--domain-filter")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-probes", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = merge_shards(
        args.dataset_dir,
        args.shard_root,
        args.output_dir,
        args.method_name,
        args.expected_shards,
        args.domain_filter,
        args.max_users,
        args.max_probes,
    )
    print(json.dumps(result["result"], indent=2, sort_keys=True))
