#!/usr/bin/env python
"""Download or validate open models and write an auditable manifest."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


WEIGHT_PATTERNS = (
    "*.safetensors",
    "pytorch_model*.bin",
    "*.gguf",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def looks_like_local_path(model_ref: str) -> bool:
    return (
        model_ref.startswith("/")
        or model_ref.startswith(".")
        or "\\" in model_ref
        or Path(model_ref).expanduser().exists()
    )


def validate_local_model(model_ref: str) -> Dict[str, Any]:
    path = Path(model_ref).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"local model path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"local model path is not a directory: {path}")
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"local model path is missing config.json: {path}")

    weight_files: List[Path] = []
    for pattern in WEIGHT_PATTERNS:
        weight_files.extend(sorted(path.glob(pattern)))
    unique_weights = sorted({item.resolve() for item in weight_files})
    if not unique_weights:
        raise FileNotFoundError(f"local model path has no recognized weight files: {path}")
    if not any(item.is_file() and item.stat().st_size > 0 for item in unique_weights):
        raise FileNotFoundError(f"local model path has no non-empty recognized weight files: {path}")

    return {
        "cache_path": str(path),
        "source": "local_path",
        "config_path": str(config_path),
        "weight_files": len(unique_weights),
        "size_bytes_hint": sum(item.stat().st_size for item in unique_weights if item.is_file()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-model", default=os.getenv("HABITBENCH_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--embed-model", default=os.getenv("HABITBENCH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.getenv("HABITBENCH_MODEL_DOWNLOAD_MANIFEST", "./runs/lumia_manifests/model_download_manifest.json")),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write a dry-run manifest; do not call snapshot_download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_ids = [args.llm_model, args.embed_model]
    manifest: Dict[str, Any] = {
        "created_at": now(),
        "updated_at": now(),
        "status": "running",
        "dry_run": args.dry_run,
        "models": [],
        "errors": [],
    }
    write_manifest(args.out, manifest)

    snapshot_download = None
    if any(not looks_like_local_path(model_id) for model_id in model_ids) and not args.dry_run:
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
            snapshot_download = hf_snapshot_download
        except Exception as exc:
            error = f"huggingface_hub import failed: {type(exc).__name__}: {exc}"
            manifest["status"] = "fail"
            manifest["updated_at"] = now()
            manifest["errors"].append(error)
            write_manifest(args.out, manifest)
            raise SystemExit(error)

    for repo_id in model_ids:
        row: Dict[str, Any] = {"repo_id": repo_id, "started_at": now(), "status": "running"}
        manifest["models"].append(row)
        manifest["updated_at"] = now()
        write_manifest(args.out, manifest)
        try:
            if args.dry_run:
                row.update({"status": "dry_run", "cache_path": None, "finished_at": now()})
            elif looks_like_local_path(repo_id):
                row.update(validate_local_model(repo_id))
                row.update({"status": "pass", "finished_at": now()})
            else:
                assert snapshot_download is not None
                path = snapshot_download(repo_id=repo_id)
                row.update({"status": "pass", "cache_path": path, "source": "huggingface_hub", "finished_at": now()})
        except Exception as exc:
            row.update(
                {
                    "status": "fail",
                    "finished_at": now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc()[-4000:],
                }
            )
            manifest["errors"].append(f"{repo_id}: {row['error']}")
            manifest["updated_at"] = now()
            manifest["status"] = "fail"
            write_manifest(args.out, manifest)
            raise SystemExit(f"Model download failed for {repo_id}: {row['error']}")
        finally:
            manifest["updated_at"] = now()
            if manifest["status"] != "fail":
                manifest["status"] = "dry_run" if args.dry_run else "running"
            write_manifest(args.out, manifest)

    manifest["status"] = "dry_run" if args.dry_run else "pass"
    manifest["updated_at"] = now()
    write_manifest(args.out, manifest)
    print(json.dumps({"status": manifest["status"], "out": str(args.out), "models": model_ids}, indent=2))


if __name__ == "__main__":
    main()
