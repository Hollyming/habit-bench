"""
Tool argument validation registry.

Usage:
    @register_validator("episodic_memory_insert", "episodic_memory_replace")
    def validate_episodic_memory(function_name: str, args: dict) -> Optional[str]:
        '''Returns error message if invalid, None if valid.'''
        ...

    # In agent.py:
    error = validate_tool_args(function_name, function_args)
    if error:
        # handle validation failure
"""

import os
import re
from typing import Any, Callable, Dict, Optional, Tuple

# Registry: tool_name -> validator_function
_VALIDATORS: Dict[str, Callable[[str, dict], Optional[str]]] = {}

_MEMORY_WRITE_TOOLS = {
    "core_memory_append",
    "core_memory_rewrite",
    "episodic_memory_insert",
    "episodic_memory_merge",
    "episodic_memory_replace",
    "resource_memory_insert",
    "resource_memory_update",
    "procedural_memory_insert",
    "procedural_memory_update",
    "semantic_memory_insert",
    "semantic_memory_update",
    "knowledge_vault_insert",
    "knowledge_vault_update",
}


def bound_adapted_memory_input(message_copy: Any, memory_type: str) -> Any:
    """Bound a local-model memory child's view of a long input message.

    This helper lives outside ``functions/function_sets`` because MIRIX treats
    every function defined in those modules as an LLM tool candidate.
    """
    if os.environ.get("MIRIX_BOUNDED_MEMORY_TOOL_SCHEMA") != "1":
        return message_copy
    if memory_type == "core":
        env_name = "MIRIX_CORE_INPUT_MAX_CHARS"
    else:
        env_name = "MIRIX_MEMORY_AGENT_INPUT_MAX_CHARS"
    limit = int(os.environ.get(env_name, "0"))
    if limit <= 0:
        return message_copy

    def bound_text(text: str, char_limit: int = limit) -> str:
        original_length = len(text)
        if memory_type == "core":
            # A literal newline/control byte inside a generated JSON string is
            # rejected by vLLM before MIRIX can apply runtime validation.
            # Give the fragile local core child a compact single-line source.
            text = re.sub(r"[\x00-\x20]+", " ", text).strip()
        if len(text) <= char_limit:
            return text
        marker = (
            f" ...[adapted {memory_type} middle omitted]... "
            if memory_type == "core"
            else f"\n...[adapted {memory_type} middle omitted]...\n"
        )
        available = max(char_limit - len(marker), 2)
        head = available // 2
        tail = available - head
        bounded = text[:head] + marker + text[-tail:]
        import logging

        logging.getLogger(__name__).warning(
            "Bounded adapted %s input: chars %d->%d",
            memory_type,
            original_length,
            len(bounded),
        )
        return bounded

    if isinstance(message_copy.content, str):
        message_copy.content = bound_text(message_copy.content)
    elif isinstance(message_copy.content, list):
        remaining = limit
        for content_item in message_copy.content:
            text = getattr(content_item, "text", None)
            if not isinstance(text, str):
                continue
            item_limit = min(remaining, limit)
            if item_limit <= 0:
                content_item.text = ""
                continue
            content_item.text = bound_text(text, item_limit)
            remaining -= len(content_item.text)
    return message_copy


def bound_adapted_core_input(message_copy: Any) -> Any:
    """Backward-compatible wrapper for focused regressions and old callers."""
    return bound_adapted_memory_input(message_copy, "core")


def bound_memory_tool_args(function_name: str, args: dict) -> Tuple[dict, list[str]]:
    """Apply the adapted local-model limits to parsed tool arguments.

    ``maxItems``/``maxLength`` in a tool schema are advisory under vLLM's
    automatic tool parser.  Enforce the same limits again after JSON parsing
    and before validation or mutation so an oversized but valid response
    cannot create an unbounded write batch.
    """
    if (
        os.environ.get("MIRIX_BOUNDED_MEMORY_TOOL_SCHEMA") != "1"
        or function_name not in _MEMORY_WRITE_TOOLS
    ):
        return args, []

    max_items = int(os.environ.get("MIRIX_MEMORY_TOOL_MAX_ITEMS", "4"))
    max_chars = int(os.environ.get("MIRIX_MEMORY_TOOL_MAX_STRING_CHARS", "1024"))
    core_max_chars = int(
        os.environ.get("MIRIX_CORE_MEMORY_TOOL_MAX_STRING_CHARS", "256")
    )
    changes: list[str] = []

    def visit(value: Any, path: str = "") -> Any:
        if isinstance(value, dict):
            return {
                key: visit(item, f"{path}.{key}" if path else key)
                for key, item in value.items()
            }
        if isinstance(value, list):
            if len(value) > max_items:
                changes.append(f"{path}:items:{len(value)}->{max_items}")
            return [visit(item, f"{path}[{index}]") for index, item in enumerate(value[:max_items])]
        if isinstance(value, str):
            field_name = path.rsplit(".", 1)[-1].split("[", 1)[0]
            if (
                function_name in {"core_memory_append", "core_memory_rewrite"}
                and field_name == "content"
            ):
                # MIRIX renders core blocks with visual ``Line n:`` prefixes,
                # which local models sometimes copy back despite the prompt.
                # They are presentation metadata, not user memory content.
                sanitized = re.sub(r"(?im)^\s*Line\s+\d+\s*:\s*", "", value).strip()
                if sanitized != value:
                    changes.append(f"{path}:removed-line-prefix")
                    value = sanitized
            if (
                function_name in {"core_memory_append", "core_memory_rewrite"}
                and field_name == "content"
            ):
                single_line = re.sub(r"[\x00-\x20]+", " ", value).strip()
                if single_line != value:
                    changes.append(f"{path}:removed-control-whitespace")
                    value = single_line
                limit = min(max_chars, core_max_chars)
            else:
                limit = max_chars
            limit = min(limit, 512) if field_name in {
                "name", "summary", "source", "actor", "event_type"
            } else limit
            if len(value) > limit:
                changes.append(f"{path}:chars:{len(value)}->{limit}")
                return value[:limit]
        return value

    return visit(args), changes


def register_validator(*tool_names: str):
    """
    Decorator to register a validation function for one or more tools.

    The validator function signature: (function_name: str, args: dict) -> Optional[str]
    Returns error message if validation fails, None if valid.
    """

    def decorator(func: Callable[[str, dict], Optional[str]]):
        for name in tool_names:
            _VALIDATORS[name] = func
        return func

    return decorator


def validate_tool_args(function_name: str, function_args: dict) -> Optional[str]:
    """
    Validate tool arguments using registered validator.
    Returns error message if validation fails, None if valid.
    """
    validator = _VALIDATORS.get(function_name)
    if validator:
        return validator(function_name, function_args)
    return None


# ============================================================
# Validators - Add new validators below using @register_validator
# ============================================================


@register_validator("episodic_memory_insert")
def validate_episodic_memory_insert(function_name: str, args: dict) -> Optional[str]:
    """Validate episodic_memory_insert arguments."""
    items = args.get("items", [])
    for i, item in enumerate(items):
        if not item.get("details", "").strip():
            return (
                f"Validation error: 'details' field in item {i} cannot be empty. "
                "Please provide a detailed description of the event."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in item {i} cannot be empty. "
                "Please provide a concise summary of the event."
            )
    return None


@register_validator("episodic_memory_replace")
def validate_episodic_memory_replace(function_name: str, args: dict) -> Optional[str]:
    """Validate episodic_memory_replace arguments."""
    items = args.get("new_items", [])
    for i, item in enumerate(items):
        if not item.get("details", "").strip():
            return (
                f"Validation error: 'details' field in new_items[{i}] cannot be empty. "
                "Please provide a detailed description of the event."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in new_items[{i}] cannot be empty. "
                "Please provide a concise summary of the event."
            )
    return None


@register_validator("episodic_memory_merge")
def validate_episodic_memory_merge(function_name: str, args: dict) -> Optional[str]:
    """Validate episodic_memory_merge arguments."""
    if not args.get("event_id", "").strip():
        return "Validation error: 'event_id' cannot be empty. Please provide the ID of the event to merge into."
    return None


# ============================================================
# Semantic Memory Validators
# ============================================================


@register_validator("semantic_memory_insert")
def validate_semantic_memory_insert(function_name: str, args: dict) -> Optional[str]:
    """Validate semantic_memory_insert arguments."""
    items = args.get("items", [])
    for i, item in enumerate(items):
        if not item.get("name", "").strip():
            return (
                f"Validation error: 'name' field in item {i} cannot be empty. "
                "Please provide the name or main concept for this knowledge entry."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in item {i} cannot be empty. "
                "Please provide a concise summary of the concept."
            )
        if not item.get("details", "").strip():
            return (
                f"Validation error: 'details' field in item {i} cannot be empty. "
                "Please provide detailed explanation or context for the concept."
            )
    return None


@register_validator("semantic_memory_update")
def validate_semantic_memory_update(function_name: str, args: dict) -> Optional[str]:
    """Validate semantic_memory_update arguments."""
    items = args.get("new_items", [])
    for i, item in enumerate(items):
        if not item.get("name", "").strip():
            return (
                f"Validation error: 'name' field in new_items[{i}] cannot be empty. "
                "Please provide the name or main concept for this knowledge entry."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in new_items[{i}] cannot be empty. "
                "Please provide a concise summary of the concept."
            )
        if not item.get("details", "").strip():
            return (
                f"Validation error: 'details' field in new_items[{i}] cannot be empty. "
                "Please provide detailed explanation or context for the concept."
            )
    return None


# ============================================================
# Resource Memory Validators
# ============================================================


@register_validator("resource_memory_insert")
def validate_resource_memory_insert(function_name: str, args: dict) -> Optional[str]:
    """Validate resource_memory_insert arguments."""
    items = args.get("items", [])
    for i, item in enumerate(items):
        if not item.get("title", "").strip():
            return (
                f"Validation error: 'title' field in item {i} cannot be empty. "
                "Please provide a title for this resource."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in item {i} cannot be empty. "
                "Please provide a summary of this resource."
            )
    return None


@register_validator("resource_memory_update")
def validate_resource_memory_update(function_name: str, args: dict) -> Optional[str]:
    """Validate resource_memory_update arguments."""
    items = args.get("new_items", [])
    for i, item in enumerate(items):
        if not item.get("title", "").strip():
            return (
                f"Validation error: 'title' field in new_items[{i}] cannot be empty. "
                "Please provide a title for this resource."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in new_items[{i}] cannot be empty. "
                "Please provide a summary of this resource."
            )
    return None


# ============================================================
# Procedural Memory Validators
# ============================================================


@register_validator("procedural_memory_insert")
def validate_procedural_memory_insert(function_name: str, args: dict) -> Optional[str]:
    """Validate procedural_memory_insert arguments."""
    items = args.get("items", [])
    for i, item in enumerate(items):
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in item {i} cannot be empty. "
                "Please provide a descriptive summary of this procedure."
            )
        steps = item.get("steps", [])
        if not steps or all(not s.strip() for s in steps):
            return (
                f"Validation error: 'steps' field in item {i} cannot be empty. "
                "Please provide at least one non-empty step."
            )
    return None


@register_validator("procedural_memory_update")
def validate_procedural_memory_update(function_name: str, args: dict) -> Optional[str]:
    """Validate procedural_memory_update arguments."""
    items = args.get("new_items", [])
    for i, item in enumerate(items):
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in new_items[{i}] cannot be empty. "
                "Please provide a descriptive summary of this procedure."
            )
        steps = item.get("steps", [])
        if not steps or all(not s.strip() for s in steps):
            return (
                f"Validation error: 'steps' field in new_items[{i}] cannot be empty. "
                "Please provide at least one non-empty step."
            )
    return None


# ============================================================
# Knowledge Vault Validators
# ============================================================


@register_validator("knowledge_vault_insert")
def validate_knowledge_vault_insert(function_name: str, args: dict) -> Optional[str]:
    """Validate knowledge_vault_insert arguments."""
    items = args.get("items", [])
    for i, item in enumerate(items):
        if not item.get("caption", "").strip():
            return (
                f"Validation error: 'caption' field in item {i} cannot be empty. "
                "Please provide a description for this knowledge vault entry."
            )
        if not item.get("secret_value", "").strip():
            return (
                f"Validation error: 'secret_value' field in item {i} cannot be empty. "
                "Please provide the credential or data value."
            )
    return None


@register_validator("knowledge_vault_update")
def validate_knowledge_vault_update(function_name: str, args: dict) -> Optional[str]:
    """Validate knowledge_vault_update arguments."""
    items = args.get("new_items", [])
    for i, item in enumerate(items):
        if not item.get("caption", "").strip():
            return (
                f"Validation error: 'caption' field in new_items[{i}] cannot be empty. "
                "Please provide a description for this knowledge vault entry."
            )
        if not item.get("secret_value", "").strip():
            return (
                f"Validation error: 'secret_value' field in new_items[{i}] cannot be empty. "
                "Please provide the credential or data value."
            )
    return None
