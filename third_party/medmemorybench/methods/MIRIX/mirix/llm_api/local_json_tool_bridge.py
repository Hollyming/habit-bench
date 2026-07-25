"""Deterministic JSON-schema bridge for fragile local memory tool calls.

Some OpenAI-compatible servers parse native tool-call text before returning a
response.  A malformed or truncated arguments string therefore becomes HTTP
400 before MIRIX can validate it.  For adapted local memory children we instead
request one schema-constrained JSON object and convert that object back to
MIRIX's normal OpenAI tool-call shape.  Tool execution remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


MEMORY_TOOL_FAMILIES = {
    "core": frozenset({"core_memory_append", "core_memory_rewrite"}),
    "episodic": frozenset(
        {
            "episodic_memory_insert",
            "episodic_memory_merge",
            "episodic_memory_replace",
            "check_episodic_memory",
        }
    ),
    "procedural": frozenset(
        {"procedural_memory_insert", "procedural_memory_update"}
    ),
    "resource": frozenset({"resource_memory_insert", "resource_memory_update"}),
    "knowledge": frozenset(
        {"knowledge_vault_insert", "knowledge_vault_update"}
    ),
    "semantic": frozenset(
        {
            "semantic_memory_insert",
            "semantic_memory_update",
            "check_semantic_memory",
        }
    ),
}
CORE_MEMORY_TOOL_NAMES = MEMORY_TOOL_FAMILIES["core"]
NON_CORE_MEMORY_TOOL_NAMES = frozenset(
    {
        "episodic_memory_insert",
        "episodic_memory_merge",
        "episodic_memory_replace",
        "check_episodic_memory",
        "procedural_memory_insert",
        "procedural_memory_update",
        "resource_memory_insert",
        "resource_memory_update",
        "knowledge_vault_insert",
        "knowledge_vault_update",
        "semantic_memory_insert",
        "semantic_memory_update",
        "check_semantic_memory",
        "trigger_memory_update",
    }
)
ALL_MEMORY_CHILD_TOOL_NAMES = frozenset().union(*MEMORY_TOOL_FAMILIES.values())


def identify_memory_tool_family(
    tools: list[dict[str, Any]] | None,
) -> str | None:
    """Return the unique memory-child family represented by a request."""
    if not tools:
        return None
    names = {
        str(tool.get("function", {}).get("name", ""))
        for tool in tools
    }
    matched = [
        family
        for family, family_names in MEMORY_TOOL_FAMILIES.items()
        if names & family_names
    ]
    # A real child request contains one family plus arbitrary universal
    # read/message helpers.  Reject mixed-family requests rather than silently
    # constraining a meta agent or the wrong child.
    return matched[0] if len(matched) == 1 else None


def is_memory_tool_request(tools: list[dict[str, Any]] | None) -> bool:
    return identify_memory_tool_family(tools) is not None


def is_core_memory_tool_request(tools: list[dict[str, Any]] | None) -> bool:
    """Backward-compatible core-only predicate used by older tests/callers."""
    return identify_memory_tool_family(tools) == "core"


def build_memory_json_tool_bridge(
    tools: list[dict[str, Any]], force_tool_call: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an OpenAI response_format plus private conversion metadata."""
    family = identify_memory_tool_family(tools)
    if family is None:
        raise ValueError("JSON tool bridge requires exactly one memory-tool family")
    family_names = MEMORY_TOOL_FAMILIES[family]

    functions = [
        deepcopy(tool["function"])
        for tool in tools
        if str(tool.get("function", {}).get("name", "")) in family_names
    ]
    if force_tool_call is not None:
        functions = [item for item in functions if item.get("name") == force_tool_call]
        if not functions:
            raise ValueError(
                f"Forced {family}-memory tool is unavailable: {force_tool_call}"
            )

    names = sorted(str(item["name"]) for item in functions)
    parameters_by_name = {
        str(function["name"]): deepcopy(
            function.get("parameters")
            or {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
        )
        for function in functions
    }
    encoded_parameters = {
        json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        for parameters in parameters_by_name.values()
    }
    if len(encoded_parameters) == 1:
        # Core append/rewrite share one argument shape.  Keep the compact schema
        # that has already been validated against the local vLLM grammar.
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": names},
                "arguments": next(iter(parameters_by_name.values())),
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
    else:
        # Different tools in one child often have incompatible signatures
        # (e.g. insert(items) versus update(old_ids,new_items)).  A free-standing
        # name enum plus arguments anyOf permits invalid cross-pairs.  Bind each
        # name to its exact argument schema in a discriminated union instead.
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": [name]},
                        "arguments": parameters_by_name[name],
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                }
                for name in names
            ]
        }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": f"mirix_{family}_memory_tool_call",
            "strict": True,
            "schema": schema,
        },
    }
    metadata = {"allowed_names": names, "family": family}
    return response_format, metadata


def build_core_json_tool_bridge(
    tools: list[dict[str, Any]], force_tool_call: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Backward-compatible core-only bridge builder."""
    if not is_core_memory_tool_request(tools):
        raise ValueError("JSON tool bridge only accepts core-memory tools")
    return build_memory_json_tool_bridge(tools, force_tool_call=force_tool_call)


def convert_memory_json_tool_response(
    response_data: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Convert a schema-constrained content object to one native tool call."""
    choices = response_data.get("choices") or []
    if not choices:
        raise ValueError("Memory JSON tool bridge received no choices")
    choice = choices[0]
    # Preserve a length finish so MIRIX's existing bounded retry handles it.
    if choice.get("finish_reason") == "length":
        return response_data

    message = choice.get("message") or {}
    raw = message.get("content")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Memory JSON tool bridge received empty content")
    # vLLM/xgrammar has occasionally returned a schema-valid string value with
    # a literal control character after the OpenAI response round trip.  MIRIX
    # already uses ``strict=False`` in its generic JSON helpers for this exact
    # interoperability case.  Accept controls only at JSON decoding here; the
    # bounded schema and the native MIRIX tool validator still validate the
    # decoded object, and arguments are immediately re-serialized canonically.
    payload = json.loads(raw, strict=False)
    if not isinstance(payload, dict):
        raise ValueError("Memory JSON tool bridge payload is not an object")
    name = payload.get("name")
    arguments = payload.get("arguments")
    allowed = set(metadata.get("allowed_names") or [])
    if name not in allowed:
        raise ValueError(f"Memory JSON tool bridge returned disallowed tool: {name}")
    if not isinstance(arguments, dict):
        raise ValueError("Memory JSON tool bridge arguments are not an object")

    canonical_arguments = json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    call_hash = hashlib.sha256(
        f"{response_data.get('id', '')}:{name}:{canonical_arguments}".encode("utf-8")
    ).hexdigest()[:20]
    message["content"] = None
    message["tool_calls"] = [
        {
            "id": f"chatcmpl-tool-{call_hash}",
            "type": "function",
            "function": {"name": name, "arguments": canonical_arguments},
        }
    ]
    choice["message"] = message
    choice["finish_reason"] = "tool_calls"
    return response_data


def convert_core_json_tool_response(
    response_data: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Backward-compatible alias for the generic response converter."""
    return convert_memory_json_tool_response(response_data, metadata)
