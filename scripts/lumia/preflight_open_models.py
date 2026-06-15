#!/usr/bin/env python
"""Preflight HuggingFace or local model access before a Lumia download/run."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


WEIGHT_PATTERNS = (
    "*.safetensors",
    "pytorch_model*.bin",
    "*.gguf",
)


def looks_like_local_path(model_ref: str) -> bool:
    return (
        model_ref.startswith("/")
        or model_ref.startswith(".")
        or "\\" in model_ref
        or Path(model_ref).expanduser().exists()
    )


def local_model_status(model_ref: str) -> Dict[str, Any]:
    path = Path(model_ref).expanduser()
    row: Dict[str, Any] = {
        "repo_id": model_ref,
        "status": "fail",
        "source": "local_path",
        "path": str(path),
    }
    if not path.exists():
        row["error"] = "local_path_missing"
        return row
    if not path.is_dir():
        row["error"] = "local_path_not_directory"
        return row

    config_path = path / "config.json"
    weight_files: List[Path] = []
    for pattern in WEIGHT_PATTERNS:
        weight_files.extend(sorted(path.glob(pattern)))
    unique_weights = sorted({item.resolve() for item in weight_files})

    if not config_path.exists():
        row["error"] = "missing_config_json"
        row["weight_files"] = len(unique_weights)
        return row
    if not unique_weights:
        row["error"] = "missing_weight_files"
        row["config_path"] = str(config_path)
        return row
    if not any(item.is_file() and item.stat().st_size > 0 for item in unique_weights):
        row["error"] = "empty_weight_files"
        row["config_path"] = str(config_path)
        row["weight_files"] = len(unique_weights)
        return row

    row.update(
        {
            "status": "pass",
            "config_path": str(config_path),
            "weight_files": len(unique_weights),
            "size_bytes_hint": sum(item.stat().st_size for item in unique_weights if item.is_file()),
        }
    )
    return row


def model_status(repo_id: str, token: str | None) -> Dict[str, Any]:
    if looks_like_local_path(repo_id):
        return local_model_status(repo_id)

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "status": "fail",
            "error": f"huggingface_hub import failed: {type(exc).__name__}: {exc}",
        }

    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id)
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }

    siblings = getattr(info, "siblings", []) or []
    size_hint = 0
    for sibling in siblings:
        size_hint += int(getattr(sibling, "size", 0) or 0)
    return {
        "repo_id": repo_id,
        "status": "pass",
        "private": bool(getattr(info, "private", False)),
        "gated": bool(getattr(info, "gated", False)),
        "sha": getattr(info, "sha", None),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "tags": list(getattr(info, "tags", []) or []),
        "siblings": len(siblings),
        "size_bytes_hint": size_hint or None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-model", default=os.getenv("HABITBENCH_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--embed-model", default=os.getenv("HABITBENCH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--out", type=Path, default=Path("./runs/lumia_manifests/model_preflight_manifest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [model_status(args.llm_model, args.token), model_status(args.embed_model, args.token)]
    status = "pass" if all(row["status"] == "pass" for row in rows) else "fail"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "models": rows,
        "token_present": bool(args.token),
        "note": "Gated models may still require accepted license terms even when model_info succeeds.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "out": str(args.out), "models": [row["repo_id"] for row in rows]}, indent=2))
    if status != "pass":
        raise SystemExit("Model preflight failed")


if __name__ == "__main__":
    main()
