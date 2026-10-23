"""Pure referential-integrity helpers for adapted local memory updates."""

from __future__ import annotations

from collections.abc import Awaitable, Callable


def normalize_existing_update_ids(
    requested_ids: list[str], existing_ids: list[str], *, allow_missing: bool
) -> tuple[list[str], list[str]]:
    """Return existing delete targets and missing targets in stable order."""
    existing = set(existing_ids)
    kept = [item_id for item_id in requested_ids if item_id in existing]
    missing = [item_id for item_id in requested_ids if item_id not in existing]
    if missing and not allow_missing:
        raise ValueError(f"Memory item with id {missing[0]} not found")
    return kept, missing


async def normalize_existing_update_ids_with_fetch(
    requested_ids: list[str],
    fetch_item: Callable[[str], Awaitable[object | None]],
    *,
    missing_exception: type[BaseException]
    | tuple[type[BaseException], ...],
    allow_missing: bool,
) -> tuple[list[str], list[str]]:
    """Resolve update targets before mutation and normalize stale references."""
    unique_ids = list(dict.fromkeys(requested_ids))
    existing_ids = []
    for item_id in unique_ids:
        try:
            item = await fetch_item(item_id)
        except missing_exception:
            item = None
        if item is not None:
            existing_ids.append(item_id)
    return normalize_existing_update_ids(
        unique_ids,
        existing_ids,
        allow_missing=allow_missing,
    )
