"""Auditable, atomic API recovery without reusing old answers under a new policy."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from evaluation.io import sha256_file


def digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def context_identity(manifest: dict[str, Any], method_input: dict[str, Any]) -> dict[str, Any]:
    model = manifest["base_model"]
    method = manifest.get("method_config") or {}
    return {
        "method_input_sha256": digest(method_input),
        "method_name": manifest["method_name"],
        "implementation": manifest["implementation"],
        "method_config": {k: method.get(k) for k in ("name", "config")},
        "memory_model": {k: model.get(k) for k in ("model", "model_path", "temperature", "seed")},
        "memory_budget_env": manifest.get("memory_budget_env", {}),
    }


def validate_context_cache(source: Path, manifest: dict[str, Any], method_input: dict[str, Any]) -> dict[str, Any] | None:
    """Only reuse a complete same-model/method/dataset cache; never copy predictions."""
    source_manifest = source / "run_manifest.json"
    source_input = source / "method_input.json"
    source_contexts = source / "memory_contexts.jsonl"
    if not all(p.is_file() for p in (source_manifest, source_input, source_contexts)):
        return None
    old = json.loads(source_manifest.read_text(encoding="utf-8"))
    if old.get("execution", {}).get("status") != "succeeded":
        return None
    old_input = json.loads(source_input.read_text(encoding="utf-8"))
    expected = context_identity(manifest, method_input)
    observed = context_identity(old, old_input)
    # Legacy v1 manifests predate explicit budget env snapshots. Admit only
    # the known 40k/8192/30k legacy profile, never an arbitrary missing value.
    if "memory_budget_env" not in old:
        legacy = {
            "context_window_tier": "auto", "model_context_tokens": 40960,
            "max_input_tokens": 40000, "summary_max_tokens": 8192,
            "compactor_input_tokens": 30000,
        }
        if expected["memory_budget_env"] != legacy or old["base_model"].get("max_input_tokens") != 40000:
            return None
        observed["memory_budget_env"] = legacy
    if observed != expected:
        return None
    return {
        "source_dir": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "contexts_sha256": sha256_file(source_contexts),
        "source_answer_protocol": old["base_model"].get("answer_protocol", "legacy_v1"),
        "source_thinking_request_policy": old["base_model"].get("thinking_request_policy", "legacy_chat_template_flag_unverified"),
        "reuse_scope": "memory_contexts_only_all_answers_regenerated",
    }


class AnswerCheckpoint:
    def __init__(self, path: Path, identity: dict[str, Any]):
        self.path = path
        self.identity_sha256 = digest(identity)
        self.predictions: dict[str, dict[str, Any]] = {}
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("identity_sha256") != self.identity_sha256:
                raise ValueError("Answer checkpoint protocol/input mismatch; use a new output directory")
            self.predictions = payload["predictions"]

    def put(self, prediction: dict[str, Any]) -> None:
        self.predictions[prediction["probe_id"]] = prediction
        atomic_json(self.path, {
            "contract_version": "habitbench.answer_checkpoint.v1",
            "identity_sha256": self.identity_sha256,
            "predictions": self.predictions,
        })
