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
MEMORY_CHILD_TERMINAL_TOOL_NAMES = frozenset({"finish_memory_update"})


class MemoryJsonToolBridgeError(RuntimeError):
    """A model response could not be converted into one memory tool call."""


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

    # ``finish_memory_update`` is a universal native MIRIX child tool. It must
    # remain available beside the family-specific tools: filtering it out
    # forces a child to invent a write even when the official lifecycle would
    # terminate without a delta.
    functions = [
        deepcopy(tool["function"])
        for tool in tools
        if str(tool.get("function", {}).get("name", ""))
        in family_names | MEMORY_CHILD_TERMINAL_TOOL_NAMES
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
        raise MemoryJsonToolBridgeError(
            "Memory JSON tool bridge received no choices"
        )
    choice = choices[0]
    # Preserve a length finish so MIRIX's existing bounded retry handles it.
    if choice.get("finish_reason") == "length":
        return response_data

    message = choice.get("message") or {}
    payload = _extract_memory_tool_payload(message, metadata)
    name = payload.get("name")
    arguments = payload.get("arguments")
    allowed = set(metadata.get("allowed_names") or [])
    if name not in allowed:
        raise MemoryJsonToolBridgeError(
            f"Memory JSON tool bridge returned disallowed tool: {name}"
        )
    if not isinstance(arguments, dict):
        raise MemoryJsonToolBridgeError(
            "Memory JSON tool bridge arguments are not an object"
        )

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


def _decode_memory_tool_payload(raw: str, source: str) -> dict[str, Any]:
    """Decode one candidate without leaking model text into logs/errors."""
    try:
        # vLLM/xgrammar has occasionally returned a schema-valid string value
        # with a literal control character after the OpenAI response round
        # trip. MIRIX uses ``strict=False`` for the same interoperability case.
        payload = json.loads(raw, strict=False)
    except (TypeError, json.JSONDecodeError) as exc:
        raise MemoryJsonToolBridgeError(
            f"Memory JSON tool bridge received invalid JSON in {source}: "
            f"{type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise MemoryJsonToolBridgeError(
            f"Memory JSON tool bridge payload in {source} is not an object"
        )
    return payload


def _extract_memory_tool_payload(
    message: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Recover a valid payload from common OpenAI-compatible response shapes."""
    errors: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        try:
            return _decode_memory_tool_payload(content, "content")
        except MemoryJsonToolBridgeError as exc:
            errors.append(str(exc))

    # Some local servers still parse a native tool call even when ``tools`` was
    # replaced by ``response_format``. In that case content is legitimately
    # empty; validate and canonicalize the existing call instead of discarding
    # it as an empty answer.
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        try:
            if len(tool_calls) != 1 or not isinstance(tool_calls[0], dict):
                raise MemoryJsonToolBridgeError(
                    "Memory JSON tool bridge requires exactly one native tool call"
                )
            function = tool_calls[0].get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = _decode_memory_tool_payload(
                    arguments, "native tool arguments"
                )
            payload = {"name": function.get("name"), "arguments": arguments}
            allowed = set(metadata.get("allowed_names") or [])
            if payload["name"] not in allowed:
                raise MemoryJsonToolBridgeError(
                    "Memory JSON tool bridge received a disallowed native tool"
                )
            if not isinstance(payload["arguments"], dict):
                raise MemoryJsonToolBridgeError(
                    "Memory JSON tool bridge native arguments are not an object"
                )
            return payload
        except MemoryJsonToolBridgeError as exc:
            errors.append(str(exc))

    # vLLM reasoning parsers may place the entire guided object in an auxiliary
    # field. Only accept it when it is itself a complete JSON object; ordinary
    # chain-of-thought text is ignored and never forwarded as a tool call.
    for key in ("reasoning_content", "reasoning"):
        candidate = message.get(key)
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        try:
            payload = _decode_memory_tool_payload(candidate, key)
            allowed = set(metadata.get("allowed_names") or [])
            if payload.get("name") not in allowed or not isinstance(
                payload.get("arguments"), dict
            ):
                raise MemoryJsonToolBridgeError(
                    f"Memory JSON tool bridge found no valid call in {key}"
                )
            return payload
        except MemoryJsonToolBridgeError as exc:
            errors.append(str(exc))

    if errors:
        raise MemoryJsonToolBridgeError("; ".join(errors))
    raise MemoryJsonToolBridgeError(
        "Memory JSON tool bridge received empty content and no recoverable tool call"
    )


def build_memory_json_retry_request(
    request_data: dict[str, Any],
    metadata: dict[str, Any],
    *,
    attempt: int,
    error: Exception,
) -> dict[str, Any]:
    """Build a fresh corrective request after an unusable guided generation."""
    retry_data = deepcopy(request_data)
    messages = list(retry_data.get("messages") or [])
    allowed_names = ", ".join(metadata.get("allowed_names") or [])
    messages.append(
        {
            "role": "user",
            "content": (
                "The previous generation was unusable "
                f"({type(error).__name__}). Regenerate the complete answer from "
                "scratch; do not continue or repair the previous text. Return "
                "exactly one complete JSON object conforming to response_format, "
                "with no prose or markdown. The tool name must be one of: "
                f"{allowed_names}. Corrective generation attempt: {attempt}."
            ),
        }
    )
    retry_data["messages"] = messages
    return retry_data


def convert_core_json_tool_response(
    response_data: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Backward-compatible alias for the generic response converter."""
    return convert_memory_json_tool_response(response_data, metadata)
