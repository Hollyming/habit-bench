#!/usr/bin/env python
"""Validate, merge, and score user-sharded supplementary Oracle runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from eval.core.dataset import DatasetContractError, load_dataset
from eval.core.io import read_jsonl, write_json, write_jsonl
from eval.core.scoring import write_score_outputs
from eval.run import validate_memory_contexts
from eval.supplementary.oracle_controls import (
    ORACLE_MODES,
    score_oracle_predictions,
)


SHARD_PATTERN = re.compile(r"^shard_(\d+)_of_(\d+)$")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetContractError(f"Required Oracle shard file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _discover(root: Path, expected: int) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    for path in root.iterdir() if root.is_dir() else []:
        if not path.is_dir():
            continue
        match = SHARD_PATTERN.match(path.name)
        if match is None:
            continue
        index, count = map(int, match.groups())
        if count != expected:
            raise DatasetContractError(
                f"{path} declares {count} shards; expected {expected}"
            )
        if index in found:
            raise DatasetContractError(f"Duplicate Oracle shard index {index}")
        found[index] = path
    missing = set(range(expected)) - set(found)
    if missing:
        raise DatasetContractError(f"Missing Oracle shards: {sorted(missing)}")
    return [(index, found[index]) for index in range(expected)]


def merge_oracle_shards(
    *,
    dataset_dir: Path,
    shard_root: Path,
    output_dir: Path,
    mode: str,
    expected_shards: int,
    domain_filter: str | None = None,
) -> dict[str, Any]:
    if mode not in ORACLE_MODES:
        raise DatasetContractError(f"Unknown Oracle mode: {mode}")
    if expected_shards < 1:
        raise DatasetContractError("expected_shards must be positive")
    bundle = load_dataset(dataset_dir, domain_filter=domain_filter)
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
    records: list[dict[str, Any]] = []
    base_model_reference: dict[str, Any] | None = None
    for index, shard_dir in _discover(shard_root, expected_shards):
        manifest = _read_json(shard_dir / "supplementary_manifest.json")
        if manifest.get("method_name") != mode:
            raise DatasetContractError(
                f"Oracle mode mismatch in {shard_dir}: {manifest.get('method_name')}"
            )
        if (manifest.get("execution") or {}).get("status") != "succeeded":
            raise DatasetContractError(f"Oracle shard is incomplete: {shard_dir}")
        dataset = manifest.get("dataset") or {}
        if dataset.get("domain_filter") != bundle.manifest.get("domain_filter"):
            raise DatasetContractError(
                f"Domain filter mismatch in Oracle shard {shard_dir}"
            )
        for field, expected_hash in expected_hashes.items():
            if dataset.get(field) != expected_hash:
                raise DatasetContractError(
                    f"Dataset hash mismatch for {field} in {shard_dir}"
                )
        subset = dataset.get("subset") or {}
        if subset.get("user_shard_index") != index:
            raise DatasetContractError(f"Shard index mismatch in {shard_dir}")
        if subset.get("user_shard_count") != expected_shards:
            raise DatasetContractError(f"Shard count mismatch in {shard_dir}")
        if subset.get("max_users") is not None or subset.get("max_probes") is not None:
            raise DatasetContractError(
                f"Cannot fully merge an Oracle smoke-test subset: {shard_dir}"
            )
        base_model = manifest.get("base_model") or {}
        comparable = {key: value for key, value in base_model.items() if key != "base_url"}
        if base_model_reference is None:
            base_model_reference = comparable
        elif comparable != base_model_reference:
            raise DatasetContractError(
                f"Base-model configuration mismatch in {shard_dir}"
            )
        shard_predictions = read_jsonl(shard_dir / "predictions.jsonl")
        shard_contexts = read_jsonl(shard_dir / "memory_contexts.jsonl")
        predictions.extend(shard_predictions)
        contexts.extend(shard_contexts)
        execution = manifest["execution"]
        records.append(
            {
                "index": index,
                "directory": str(shard_dir),
                "users": dataset.get("users"),
                "sessions": dataset.get("sessions"),
                "probes": dataset.get("probes"),
                "wall_clock_sec": execution.get("wall_clock_sec"),
                "host": execution.get("host"),
                "cuda_visible_devices": execution.get("cuda_visible_devices"),
            }
        )

    probe_order = [probe["probe_id"] for probe in bundle.probes]
    contexts_by_id = validate_memory_contexts(contexts, probe_order)
    detailed, metrics, metric_rows = score_oracle_predictions(
        predictions, bundle, mode
    )
    predictions_by_id = {row["probe_id"]: row for row in predictions}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output_dir / "memory_contexts.jsonl",
        [contexts_by_id[probe_id] for probe_id in probe_order],
    )
    write_jsonl(
        output_dir / "predictions.jsonl",
        [predictions_by_id[probe_id] for probe_id in probe_order],
    )
    write_score_outputs(output_dir, detailed, metrics, metric_rows)
    wall_times = [
        float(record["wall_clock_sec"])
        for record in records
        if record.get("wall_clock_sec") is not None
    ]
    manifest = {
        "contract_version": "habitbench.supplementary_oracle_merge.v1",
        "experiment_role": "diagnostic_upper_bound",
        "method_name": mode,
        "dataset": bundle.manifest,
        "base_model": {
            **(base_model_reference or {}),
            "base_url": "<per-shard-local-endpoint>",
        },
        "expected_shards": expected_shards,
        "shard_root": str(shard_root),
        "shards": records,
        "timing": {
            "shard_wall_clock_sum_sec": (
                round(sum(wall_times), 3) if wall_times else None
            ),
            "shard_wall_clock_max_sec": (
                round(max(wall_times), 3) if wall_times else None
            ),
        },
        "result": metrics["overall"],
    }
    write_json(output_dir / "merge_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=ORACLE_MODES, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--domain-filter")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merged = merge_oracle_shards(
        dataset_dir=args.dataset_dir,
        shard_root=args.shard_root,
        output_dir=args.output_dir,
        mode=args.mode,
        expected_shards=args.expected_shards,
        domain_filter=args.domain_filter,
    )
    print(json.dumps(merged["result"], indent=2, sort_keys=True))
