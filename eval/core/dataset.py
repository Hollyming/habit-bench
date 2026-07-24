from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, sha256_file


class DatasetContractError(ValueError):
    """Raised when a dataset cannot satisfy the public evaluation contract."""


PRIVATE_ONLY_FIELDS = {
    "active_policy_variants",
    "gold_action",
    "gold_action_text",
    "gold_choice_id",
    "gold_evidence_session_ids",
    "hidden_habit_graph",
    "old_policy_variants",
    "persona_profiles",
    "target_habit_ids",
}


@dataclass(frozen=True)
class DatasetBundle:
    dataset_dir: Path
    sessions_by_user: dict[str, list[dict[str, Any]]]
    probes: list[dict[str, Any]]
    keys: dict[str, dict[str, Any]]
    manifest: dict[str, Any]

    def method_payload(self, method_name: str) -> dict[str, Any]:
        public_manifest = {
            key: value
            for key, value in self.manifest.items()
            if not key.startswith("private_")
        }
        payload = {
            "contract_version": "habitbench.memory_context.v1",
            "method_name": method_name,
            "dataset": public_manifest,
            "sessions_by_user": self.sessions_by_user,
            "probes": self.probes,
            "output_contract": {
                "required_fields": ["probe_id", "memory_context"],
                "optional_fields": ["evidence_session_ids", "debug", "cost"],
                "forbidden_fields": ["choice_id", "scores"],
            },
        }
        assert_no_private_fields(payload)
        return payload


def _unique_rows(rows: Iterable[dict[str, Any]], field: str, source: Path) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        value = str(row.get(field, ""))
        if not value:
            raise DatasetContractError(f"Missing {field} in {source}")
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise DatasetContractError(f"Duplicate {field} values in {source}: {duplicates[:5]}")


def _normalize_sessions(path: Path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    rows = read_jsonl(path)
    if not rows:
        raise DatasetContractError(f"No lifelines found in {path}")

    nested = all(isinstance(row.get("sessions"), list) for row in rows)
    flat = all("session_id" in row for row in rows)
    if not (nested or flat):
        raise DatasetContractError(
            f"Lifelines must be consistently user-nested or one-session-per-row: {path}"
        )

    sessions: list[dict[str, Any]] = []
    if nested:
        _unique_rows(rows, "user_id", path)
        for lifeline in rows:
            user_id = str(lifeline["user_id"])
            expected = lifeline.get("session_count")
            if expected is not None and int(expected) != len(lifeline["sessions"]):
                raise DatasetContractError(
                    f"session_count mismatch for {user_id}: {expected} != {len(lifeline['sessions'])}"
                )
            for session in lifeline["sessions"]:
                normalized = dict(session)
                normalized.setdefault("user_id", user_id)
                normalized.setdefault("domain", lifeline.get("domain", "unknown"))
                if normalized["user_id"] != user_id:
                    raise DatasetContractError(
                        f"Nested session {normalized.get('session_id')} has a different user_id"
                    )
                sessions.append(normalized)
        source_format = "user_nested"
    else:
        sessions = [dict(row) for row in rows]
        source_format = "session_rows"

    _unique_rows(sessions, "session_id", path)
    by_user: dict[str, list[dict[str, Any]]] = {}
    for raw in sessions:
        for field in ("session_id", "user_id", "session_index", "messages"):
            if field not in raw:
                raise DatasetContractError(f"Session missing {field}: {raw.get('session_id')}")
        messages = raw["messages"]
        if not isinstance(messages, list) or not messages:
            raise DatasetContractError(f"Session has no messages: {raw['session_id']}")
        clean_messages = []
        for message in messages:
            if message.get("role") not in {"user", "assistant", "system", "tool"}:
                raise DatasetContractError(
                    f"Invalid role in session {raw['session_id']}: {message.get('role')}"
                )
            if not isinstance(message.get("content"), str):
                raise DatasetContractError(f"Invalid content in session {raw['session_id']}")
            clean_messages.append({"role": message["role"], "content": message["content"]})
        session = {
            "session_id": str(raw["session_id"]),
            "user_id": str(raw["user_id"]),
            "session_index": int(raw["session_index"]),
            "timestamp": raw.get("timestamp"),
            "domain": raw.get("domain", "unknown"),
            "messages": clean_messages,
        }
        by_user.setdefault(session["user_id"], []).append(session)

    for user_id, user_sessions in by_user.items():
        user_sessions.sort(key=lambda row: (row["session_index"], row["session_id"]))
        indices = [row["session_index"] for row in user_sessions]
        if len(indices) != len(set(indices)):
            raise DatasetContractError(f"Duplicate session_index for user {user_id}")
    return by_user, source_format


def _public_key_id(row: dict[str, Any]) -> str:
    return str(row.get("public_probe_id") or row.get("probe_id") or "")


def _scope_from_key(key: dict[str, Any]) -> int | None:
    scope = key.get("visible_history_scope") or {}
    for field in ("max_session_index", "through_session_index"):
        if scope.get(field) is not None:
            return int(scope[field])
    return None


def _normalize_probes(
    public_path: Path,
    key_path: Path,
    sessions_by_user: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    public_rows = read_jsonl(public_path)
    key_rows = read_jsonl(key_path)
    _unique_rows(public_rows, "probe_id", public_path)

    keys: dict[str, dict[str, Any]] = {}
    for raw_key in key_rows:
        public_id = _public_key_id(raw_key)
        if not public_id:
            raise DatasetContractError(f"Private key has no probe identifier in {key_path}")
        if public_id in keys:
            raise DatasetContractError(f"Duplicate key for public probe {public_id}")
        key = dict(raw_key)
        key["private_probe_id"] = raw_key.get("probe_id")
        key["probe_id"] = public_id
        keys[public_id] = key

    probes: list[dict[str, Any]] = []
    for raw in public_rows:
        probe_id = str(raw["probe_id"])
        user_id = str(raw.get("user_id", ""))
        if probe_id not in keys:
            raise DatasetContractError(f"Missing private key for public probe {probe_id}")
        if user_id not in sessions_by_user:
            raise DatasetContractError(f"Probe {probe_id} references unknown user {user_id}")
        choices = raw.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            raise DatasetContractError(f"Probe {probe_id} must have at least two choices")
        choice_ids = [str(choice.get("choice_id", "")) for choice in choices]
        if any(not choice_id for choice_id in choice_ids) or len(choice_ids) != len(set(choice_ids)):
            raise DatasetContractError(f"Probe {probe_id} has invalid choice ids")

        public_scope = raw.get("visible_history_scope") or {}
        max_index = public_scope.get("max_session_index")
        if max_index is None:
            max_index = _scope_from_key(keys[probe_id])
        if max_index is None:
            max_index = sessions_by_user[user_id][-1]["session_index"]
        max_index = int(max_index)
        available = {session["session_index"] for session in sessions_by_user[user_id]}
        if max_index < min(available) or max_index > max(available):
            raise DatasetContractError(
                f"Probe {probe_id} cutoff {max_index} is outside user history "
                f"[{min(available)}, {max(available)}]"
            )

        clean_choices = [
            {"choice_id": str(choice["choice_id"]), "text": str(choice["text"])}
            for choice in choices
        ]
        gold_choice = str(keys[probe_id].get("gold_choice_id", ""))
        if gold_choice not in choice_ids:
            raise DatasetContractError(
                f"Gold choice {gold_choice!r} is invalid for public probe {probe_id}"
            )
        probe = {
            "probe_id": probe_id,
            "user_id": user_id,
            "domain": raw.get("domain") or sessions_by_user[user_id][0].get("domain", "unknown"),
            "timestamp": raw.get("timestamp"),
            "query": str(raw.get("query", "")),
            "choices": clean_choices,
            "visible_history_scope": {"user_id": user_id, "max_session_index": max_index},
        }
        for field in ("split", "metadata"):
            if field in raw:
                probe[field] = raw[field]
        if not probe["query"].strip():
            raise DatasetContractError(f"Probe {probe_id} has an empty query")
        probes.append(probe)

    extra_keys = set(keys) - {probe["probe_id"] for probe in probes}
    if extra_keys:
        raise DatasetContractError(f"Private keys without public probes: {sorted(extra_keys)[:5]}")
    return probes, keys


def _select_subset(
    sessions_by_user: dict[str, list[dict[str, Any]]],
    probes: list[dict[str, Any]],
    keys: dict[str, dict[str, Any]],
    max_users: int | None,
    max_probes: int | None,
    user_shard_index: int | None,
    user_shard_count: int | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if max_users is not None and max_users < 1:
        raise DatasetContractError("max_users must be positive")
    if max_probes is not None and max_probes < 1:
        raise DatasetContractError("max_probes must be positive")
    if (user_shard_index is None) != (user_shard_count is None):
        raise DatasetContractError(
            "user_shard_index and user_shard_count must be provided together"
        )
    if user_shard_count is not None:
        if user_shard_count < 1:
            raise DatasetContractError("user_shard_count must be positive")
        if user_shard_index is None or not 0 <= user_shard_index < user_shard_count:
            raise DatasetContractError(
                "user_shard_index must be in [0, user_shard_count)"
            )

    selected_users = sorted({probe["user_id"] for probe in probes})
    if max_users is not None:
        selected_users = selected_users[:max_users]
    if user_shard_count is not None:
        selected_users = selected_users[user_shard_index::user_shard_count]
    selected_user_set = set(selected_users)
    selected_probes = [probe for probe in probes if probe["user_id"] in selected_user_set]
    if max_probes is not None:
        selected_probes = selected_probes[:max_probes]
    selected_probe_ids = {probe["probe_id"] for probe in selected_probes}
    selected_user_set = {probe["user_id"] for probe in selected_probes}
    return (
        {user_id: sessions_by_user[user_id] for user_id in sorted(selected_user_set)},
        selected_probes,
        {probe_id: keys[probe_id] for probe_id in selected_probe_ids},
    )


def _find_private_fields(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in PRIVATE_ONLY_FIELDS:
                found.append(child_path)
            found.extend(_find_private_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_private_fields(child, f"{path}[{index}]"))
    return found


def assert_no_private_fields(payload: dict[str, Any]) -> None:
    leaked = _find_private_fields(payload)
    if leaked:
        raise DatasetContractError(f"Private evaluation fields leaked into method input: {leaked[:5]}")


def load_dataset(
    dataset_dir: Path,
    *,
    max_users: int | None = None,
    max_probes: int | None = None,
    user_shard_index: int | None = None,
    user_shard_count: int | None = None,
) -> DatasetBundle:
    dataset_dir = dataset_dir.resolve()
    public_lifelines = dataset_dir / "public" / "lifelines.jsonl"
    public_probes = dataset_dir / "public" / "probes.jsonl"
    private_keys = dataset_dir / "private" / "probe_key.jsonl"
    for path in (public_lifelines, public_probes, private_keys):
        if not path.is_file():
            raise DatasetContractError(f"Required dataset file not found: {path}")

    sessions_by_user, source_format = _normalize_sessions(public_lifelines)
    probes, keys = _normalize_probes(public_probes, private_keys, sessions_by_user)
    sessions_by_user, probes, keys = _select_subset(
        sessions_by_user,
        probes,
        keys,
        max_users,
        max_probes,
        user_shard_index,
        user_shard_count,
    )
    if not probes:
        raise DatasetContractError("Dataset selection contains no probes")

    manifest = {
        "name": dataset_dir.name,
        "source_format": source_format,
        "users": len(sessions_by_user),
        "sessions": sum(len(rows) for rows in sessions_by_user.values()),
        "probes": len(probes),
        "public_lifelines_sha256": sha256_file(public_lifelines),
        "public_probes_sha256": sha256_file(public_probes),
        "private_probe_key_sha256": sha256_file(private_keys),
        "subset": {
            "max_users": max_users,
            "max_probes": max_probes,
            "user_shard_index": user_shard_index,
            "user_shard_count": user_shard_count,
        },
    }
    return DatasetBundle(dataset_dir, sessions_by_user, probes, keys, manifest)
