#!/usr/bin/env python
"""Create a portable task plan for user-sharded HABIT-Bench evaluation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
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
    "food": PROJECT_ROOT / "domain/food/food_habit_lifelines_stress_v5",
    "finance": PROJECT_ROOT
    / "domain/finance-software/habit_bench_multidogo_finance_software_release_gated_v1_4",
    "software": PROJECT_ROOT
    / "domain/finance-software/habit_bench_multidogo_finance_software_release_gated_v1_4",
    "travel": PROJECT_ROOT
    / "domain/travel/release_candidate_v16_postrepair_repaired_r4",
    # Backward-compatible combined view; no longer part of the default suite.
    "finance_software": PROJECT_ROOT
    / "domain/finance-software/habit_bench_multidogo_finance_software_release_gated_v1_4",
}
DEFAULT_DATASET_DOMAINS = {
    "food": None,
    "finance": "finance",
    "software": "software",
    "travel": None,
    "finance_software": None,
}
METHOD_CONFIGS = {
    method: f"{method}_qwen3-8b_adapted"
    for method in ("mem0", "amem", "memos", "memrl", "lightmem", "letta", "mirix")
}
LOCAL_METHOD_CONFIGS = {
    "full_memory": PROJECT_ROOT / "configs/methods/full_memory.yaml",
    "full_history": PROJECT_ROOT / "configs/methods/full_history.yaml",
    "secom": PROJECT_ROOT / "configs/methods/secom_bge_m3_qwen3.yaml",
}
BGE_M3_METHODS = set(METHOD_CONFIGS) | {"secom"}
BGE_M3_ID = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DEFAULT_BGE_M3_PATH = Path("/plm-shared/zhangjunming/Workspace/models/bge-m3")
DEFAULT_LLM_PATH = Path("/plm-shared/zhangjunming/Workspace/models/Qwen3-8B")
DEFAULT_SERVED_MODEL = "Qwen3-8B"
DEFAULT_COMPRESSOR_PATH = Path(
    "/plm-shared/zhangjunming/Workspace/models/"
    "llmlingua-2-xlm-roberta-large-meetingbank"
)
BGE_M3_DIM = 1024
PLAN_FIELDS = (
    "task_id",
    "method",
    "dataset_name",
    "dataset_dir",
    "domain_filter",
    "max_users",
    "max_probes",
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


def _dataset_domain_overrides(values: list[str]) -> dict[str, str | None]:
    domains = dict(DEFAULT_DATASET_DOMAINS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--dataset-domain must be NAME=DOMAIN, got {value!r}")
        name, domain = value.split("=", 1)
        if not name.strip() or not domain.strip():
            raise ValueError("--dataset-domain requires nonempty NAME and DOMAIN")
        domains[name.strip()] = domain.strip().lower()
    return domains


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
    dataset_domains = _dataset_domain_overrides(args.dataset_domain)
    selected_datasets = _split_csv(args.datasets)
    unknown_datasets = set(selected_datasets) - set(datasets)
    if not selected_datasets or unknown_datasets:
        raise ValueError(f"Unknown or empty datasets: {sorted(unknown_datasets)}")
    if args.shards < 1:
        raise ValueError("--shards must be positive")

    output_root = args.output_root.expanduser().resolve()
    rows: list[dict[str, str | int]] = []
    dataset_user_counts = {
        dataset_name: int(
            load_dataset(
                datasets[dataset_name],
                domain_filter=dataset_domains.get(dataset_name),
                max_users=args.max_users,
                max_probes=args.max_probes,
            ).manifest["users"]
        )
        for dataset_name in selected_datasets
    }
    for method in methods:
        for dataset_name in selected_datasets:
            dataset_dir = datasets[dataset_name]
            domain_filter = dataset_domains.get(dataset_name)
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
                        "domain_filter": domain_filter or "",
                        "max_users": args.max_users or "",
                        "max_probes": args.max_probes or "",
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
        default="food,finance,software,travel",
        help="Comma-separated dataset aliases.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add or override a dataset alias; may be repeated.",
    )
    parser.add_argument(
        "--dataset-domain",
        action="append",
        default=[],
        metavar="NAME=DOMAIN",
        help="Attach a public domain filter to a dataset alias; may be repeated.",
    )
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument(
        "--embedding-model-path",
        type=Path,
        default=Path(
            os.environ.get("HABITBENCH_EMBED_MODEL", str(DEFAULT_BGE_M3_PATH))
        ),
        help=(
            "Physical BGE-M3 snapshot used by this run. The model identity and "
            "revision remain fixed; defaults to HABITBENCH_EMBED_MODEL or the "
            "legacy ClusterX path."
        ),
    )
    parser.add_argument(
        "--llm-model-path",
        type=Path,
        default=Path(
            os.environ.get("HABITBENCH_LLM_MODEL", str(DEFAULT_LLM_PATH))
        ),
        help="Physical Qwen snapshot recorded in the effective method config.",
    )
    parser.add_argument(
        "--served-model-name",
        default=os.environ.get("HABITBENCH_SERVED_MODEL", DEFAULT_SERVED_MODEL),
        help="OpenAI-compatible served model identity recorded in method configs.",
    )
    parser.add_argument(
        "--lightmem-model-path",
        type=Path,
        default=Path(
            os.environ.get(
                "HABITBENCH_LIGHTMEM_MODEL",
                str(DEFAULT_COMPRESSOR_PATH),
            )
        ),
        help="Physical LLMLingua2 snapshot used by LightMem.",
    )
    parser.add_argument(
        "--secom-compressor-path",
        type=Path,
        default=Path(
            os.environ.get(
                "HABITBENCH_SECOM_COMPRESSOR",
                str(DEFAULT_COMPRESSOR_PATH),
            )
        ),
        help="Physical LLMLingua2 snapshot used by SeCom.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        help="Optional smoke-test subset; omit for formal full-dataset plans.",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        help="Optional smoke-test subset; omit for formal full-dataset plans.",
    )
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


def _validate_bge_m3(
    config: dict,
    *,
    config_path: Path,
    model_path: Path,
) -> None:
    embedding = config.get("embedding")
    expected = {
        "provider": "local",
        "model": BGE_M3_ID,
        "revision": BGE_M3_REVISION,
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

    model_path = model_path.expanduser().resolve()
    model_config_path = model_path / "config.json"
    identity_path = model_path / "HABIT_MODEL_INFO.json"
    weight_path = model_path / "pytorch_model.bin"
    if (
        not model_config_path.is_file()
        or not identity_path.is_file()
        or not weight_path.is_file()
    ):
        raise FileNotFoundError(
            f"Incomplete BGE-M3 snapshot at {model_path}; "
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


def _effective_method_config(
    config: dict,
    *,
    embedding_model_path: Path,
    llm_model_path: Path | None = None,
    served_model_name: str | None = None,
    compressor_model_path: Path | None = None,
) -> tuple[dict, dict[str, str]]:
    """Apply the selected LLM identity and physical-path overrides."""
    effective = copy.deepcopy(config)
    overrides: dict[str, str] = {}
    embedding = effective.get("embedding")
    if isinstance(embedding, dict):
        configured_path = embedding.get("model_path")
        resolved_path = str(embedding_model_path.expanduser().resolve())
        if configured_path != resolved_path:
            overrides["embedding.model_path"] = resolved_path
        embedding["model_path"] = resolved_path
        agent_params = effective.get("agent_params")
        if isinstance(agent_params, dict):
            for key in ("amem_embedding_model", "embedding_model_path"):
                if key in agent_params and agent_params[key] != resolved_path:
                    agent_params[key] = resolved_path
                    overrides[f"agent_params.{key}"] = resolved_path
    if llm_model_path is not None:
        history = effective.get("history")
        resolved_llm_path = str(llm_model_path.expanduser().resolve())
        if isinstance(history, dict) and "tokenizer_path" in history:
            if history["tokenizer_path"] != resolved_llm_path:
                overrides["history.tokenizer_path"] = resolved_llm_path
            history["tokenizer_path"] = resolved_llm_path
    if served_model_name is not None:
        served_model_name = served_model_name.strip()
        if not served_model_name:
            raise ValueError("served_model_name cannot be empty")
        description = effective.get("description")
        if isinstance(description, str) and "Qwen3-8B" in description:
            effective["description"] = description.replace(
                "Qwen3-8B", served_model_name
            )
            overrides["description"] = effective["description"]
        for section_name in ("model", "answer_model"):
            section = effective.get(section_name)
            if isinstance(section, dict) and section.get("name") != served_model_name:
                section["name"] = served_model_name
                overrides[f"{section_name}.name"] = served_model_name
        history = effective.get("history")
        compactor = history.get("compactor") if isinstance(history, dict) else None
        if isinstance(compactor, dict) and compactor.get("name") != served_model_name:
            compactor["name"] = served_model_name
            overrides["history.compactor.name"] = served_model_name
        agent_params = effective.get("agent_params")
        if isinstance(agent_params, dict):
            for key in ("amem_model", "memos_model"):
                if key in agent_params and agent_params[key] != served_model_name:
                    agent_params[key] = served_model_name
                    overrides[f"agent_params.{key}"] = served_model_name
    if compressor_model_path is not None:
        resolved_compressor_path = str(
            compressor_model_path.expanduser().resolve()
        )
        memory = effective.get("memory")
        if isinstance(memory, dict) and "compressor_model_path" in memory:
            if memory["compressor_model_path"] != resolved_compressor_path:
                overrides["memory.compressor_model_path"] = resolved_compressor_path
            memory["compressor_model_path"] = resolved_compressor_path
        agent_params = effective.get("agent_params")
        if (
            isinstance(agent_params, dict)
            and "topic_segmenter_model_path" in agent_params
        ):
            if agent_params["topic_segmenter_model_path"] != resolved_compressor_path:
                overrides["agent_params.topic_segmenter_model_path"] = (
                    resolved_compressor_path
                )
            agent_params["topic_segmenter_model_path"] = resolved_compressor_path
    return effective, overrides


def _method_configs(
    methods: list[str],
    *,
    embedding_model_path: Path,
    llm_model_path: Path,
    lightmem_model_path: Path,
    secom_compressor_path: Path,
    served_model_name: str = DEFAULT_SERVED_MODEL,
) -> dict[str, dict | None]:
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
            _validate_bge_m3(
                parsed,
                config_path=path,
                model_path=embedding_model_path,
            )
        compressor_model_path = None
        if method == "lightmem":
            compressor_model_path = lightmem_model_path
        elif method == "secom":
            compressor_model_path = secom_compressor_path
        effective_config, path_overrides = _effective_method_config(
            parsed,
            embedding_model_path=embedding_model_path,
            llm_model_path=llm_model_path,
            served_model_name=served_model_name,
            compressor_model_path=compressor_model_path,
        )
        records[method] = {
            "name": config_name,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "config": effective_config,
            "path_overrides": path_overrides,
        }
    return records


def _bge_m3_snapshot(model_path: Path) -> dict:
    model_path = model_path.expanduser().resolve()
    identity_path = model_path / "HABIT_MODEL_INFO.json"
    config_path = model_path / "config.json"
    weight_path = model_path / "pytorch_model.bin"
    return {
        "identity": json.loads(identity_path.read_text(encoding="utf-8")),
        "identity_path": str(identity_path),
        "identity_sha256": sha256_file(identity_path),
        "transformers_config_sha256": sha256_file(config_path),
        "weight_path": str(weight_path),
        "weight_size_bytes": weight_path.stat().st_size,
    }


def _portable_model_snapshot(model_path: Path) -> dict:
    """Record H identity markers when present without breaking legacy paths."""
    model_path = model_path.expanduser().resolve()
    record: dict[str, object] = {"path": str(model_path)}
    config_path = model_path / "config.json"
    if config_path.is_file():
        record["transformers_config_sha256"] = sha256_file(config_path)
    identity_path = model_path / "HABIT_MODEL_INFO.json"
    if identity_path.is_file():
        record.update(
            {
                "identity": json.loads(identity_path.read_text(encoding="utf-8")),
                "identity_path": str(identity_path),
                "identity_sha256": sha256_file(identity_path),
            }
        )
    return record


def main() -> None:
    args = parse_args()
    embedding_model_path = args.embedding_model_path.expanduser().resolve()
    llm_model_path = args.llm_model_path.expanduser().resolve()
    lightmem_model_path = args.lightmem_model_path.expanduser().resolve()
    secom_compressor_path = args.secom_compressor_path.expanduser().resolve()
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
    dataset_records = {
        str(row["dataset_name"]): {
            "dataset_dir": str(row["dataset_dir"]),
            "domain_filter": str(row.get("domain_filter") or "") or None,
            "max_users": int(row["max_users"]) if row.get("max_users") else None,
            "max_probes": int(row["max_probes"]) if row.get("max_probes") else None,
        }
        for row in rows
    }
    method_configs = _method_configs(
        methods,
        embedding_model_path=embedding_model_path,
        llm_model_path=llm_model_path,
        lightmem_model_path=lightmem_model_path,
        secom_compressor_path=secom_compressor_path,
        served_model_name=args.served_model_name,
    )
    manifest_path = args.manifest or args.plan.with_suffix(".manifest.json")
    manifest = {
        "contract_version": "habitbench.shard_plan.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "output_root": str(args.output_root.expanduser().resolve()),
        "task_count": len(rows),
        "shard_count": args.shards,
        "methods": method_configs,
        "models": {
            "llm": _portable_model_snapshot(llm_model_path),
            "embedding": (
                _bge_m3_snapshot(embedding_model_path)
                if any(method in BGE_M3_METHODS for method in methods)
                else None
            ),
            "lightmem_model": (
                _portable_model_snapshot(lightmem_model_path)
                if "lightmem" in methods
                else None
            ),
            "secom_compressor": (
                _portable_model_snapshot(secom_compressor_path)
                if "secom" in methods
                else None
            ),
        },
        "datasets": {
            dataset_name: {
                **record,
                "manifest": load_dataset(
                    Path(record["dataset_dir"]),
                    domain_filter=record["domain_filter"],
                    max_users=record["max_users"],
                    max_probes=record["max_probes"],
                ).manifest,
            }
            for dataset_name, record in dataset_records.items()
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
