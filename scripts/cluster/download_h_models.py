#!/usr/bin/env python3
"""Download and validate the pinned HABIT-Bench model snapshots for H."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


SNAPSHOTS = {
    "qwen": {
        "repo_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "directory": "Qwen3-8B",
        "required": ("config.json", "tokenizer.json"),
        "weight_globs": ("*.safetensors", "pytorch_model*.bin"),
    },
    "qwen14b": {
        "repo_id": "Qwen/Qwen3-14B",
        "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "directory": "Qwen3-14B",
        "required": ("config.json", "tokenizer.json"),
        "weight_globs": ("*.safetensors", "pytorch_model*.bin"),
    },
    "qwen4b": {
        "repo_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "directory": "Qwen3-4B",
        "required": ("config.json", "tokenizer.json"),
        "weight_globs": ("*.safetensors", "pytorch_model*.bin"),
    },
    "qwen32b": {
        "repo_id": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "directory": "Qwen3-32B",
        "required": ("config.json", "tokenizer.json"),
        "weight_globs": ("*.safetensors", "pytorch_model*.bin"),
    },
    "bge": {
        "repo_id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "directory": "bge-m3",
        "required": ("config.json", "pytorch_model.bin"),
        "weight_globs": (),
    },
    "llmlingua": {
        "repo_id": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        "revision": "ebaba9b0e874dadd3003ffcff828e4397e568089",
        "directory": "llmlingua-2-xlm-roberta-large-meetingbank",
        "required": ("config.json", "model.safetensors"),
        "weight_globs": (),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(
            "/mnt/shared-storage-gpfs2/plm-gpfs/jmzhang/models/habitbench"
        ),
    )
    parser.add_argument(
        "--models",
        default="qwen,bge,llmlingua",
        help=(
            "Comma-separated subset of "
            "qwen,qwen4b,qwen14b,qwen32b,bge,llmlingua"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Validate existing snapshots and refresh identity markers without network access",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_snapshot(target: Path, spec: dict[str, object]) -> None:
    missing = [name for name in spec["required"] if not (target / name).is_file()]
    weight_globs = spec["weight_globs"]
    if weight_globs and not any(
        candidate.is_file()
        for pattern in weight_globs
        for candidate in target.glob(pattern)
    ):
        missing.append(" or ".join(weight_globs))
    if missing:
        raise RuntimeError(f"incomplete snapshot at {target}: missing {missing}")


def write_snapshot_identity(
    target: Path,
    spec: dict[str, object],
    *,
    name: str,
) -> None:
    weight_files = sorted(
        {
            candidate
            for pattern in spec["weight_globs"]
            for candidate in target.glob(pattern)
            if candidate.is_file()
        }
    )
    for required_name in spec["required"]:
        required_path = target / required_name
        if required_path.suffix in {".bin", ".safetensors"}:
            weight_files.append(required_path)
    weight_files = sorted(set(weight_files))
    marker = {
        "model_id": spec["repo_id"],
        "revision": spec["revision"],
        "weights": [
            {"file": path.name, "size_bytes": path.stat().st_size}
            for path in weight_files
        ],
    }
    if name == "bge":
        marker.update(
            {
                "dense_embedding_dimension": 1024,
                "max_sequence_length": 8192,
                "weight_file": "pytorch_model.bin",
                "weight_sha256": sha256(target / "pytorch_model.bin"),
            }
        )
    marker_path = target / "HABIT_MODEL_INFO.json"
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")
    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(SNAPSHOTS))
    if unknown or not requested:
        raise SystemExit(f"unknown or empty --models selection: {unknown}")
    if not str(args.model_root).startswith("/mnt/shared-storage-"):
        raise SystemExit("--model-root must be on persistent /mnt/shared-storage-* storage")

    args.model_root.mkdir(parents=True, exist_ok=True)
    for name in requested:
        spec = SNAPSHOTS[name]
        target = args.model_root / str(spec["directory"])
        if args.local_files_only:
            print(f"validating local snapshot {target}", flush=True)
        else:
            print(
                f"downloading {spec['repo_id']}@{spec['revision']} -> {target}",
                flush=True,
            )
            resolved = snapshot_download(
                repo_id=str(spec["repo_id"]),
                revision=str(spec["revision"]),
                local_dir=target,
                max_workers=args.max_workers,
            )
            if Path(resolved).resolve() != target.resolve():
                raise RuntimeError(f"unexpected snapshot path: {resolved}")
        validate_snapshot(target, spec)
        write_snapshot_identity(target, spec, name=name)
        print(f"ready model={name} path={target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
