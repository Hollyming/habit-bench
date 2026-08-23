"""Deterministic two-stage JSON bridge for local memory tool calls.

Some OpenAI-compatible servers parse native tool-call text before returning a
response.  A malformed or truncated arguments string therefore becomes HTTP
400 before MIRIX can validate it.  For adapted local memory children we instead
first select one tool with a tiny schema, then request only that tool's exact
argument schema.  The result is converted back to MIRIX's normal OpenAI
tool-call shape.  Tool execution remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from mirix.log import get_logger

logger = get_logger(__name__)


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
    "procedural": frozenset({"procedural_memory_insert", "procedural_memory_update"}),
    "resource": frozenset({"resource_memory_insert", "resource_memory_update"}),
    "knowledge": frozenset({"knowledge_vault_insert", "knowledge_vault_update"}),
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
    names = {str(tool.get("function", {}).get("name", "")) for tool in tools}
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
    """Build the stage-one selector format plus private stage-two metadata."""
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
    # Do not combine heterogeneous tool arguments in one ``anyOf`` grammar.
    # Qwen3/vLLM intermittently emitted malformed content for that complex
    # union.  Stage one is deliberately tiny; stage two (built below) exposes
    # exactly one argument schema, so invalid name/argument cross-pairs are
    # impossible without changing MIRIX's tool lifecycle.
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": names}},
        "required": ["name"],
        "additionalProperties": False,
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": f"mirix_{family}_memory_tool_call",
            "strict": True,
            "schema": schema,
        },
    }
    metadata = {
        "allowed_names": names,
        "argument_schemas": parameters_by_name,
        "family": family,
    }
    return response_format, metadata


def select_memory_json_tool(
    response_data: dict[str, Any], metadata: dict[str, Any]
) -> str:
    """Decode and validate the stage-one tool selector."""
    _, message = _first_choice_and_message(response_data, "selector")
    payload = _extract_json_object(message, "selector")
    allowed = set(metadata.get("allowed_names") or [])
    name = payload.get("name")
    if set(payload) != {"name"} or name not in allowed:
        raise MemoryJsonToolBridgeError(
            f"Memory JSON tool selector returned an invalid tool: {name}"
        )
    return str(name)


def build_memory_json_arguments_request(
    request_data: dict[str, Any],
    metadata: dict[str, Any],
    selected_name: str,
) -> dict[str, Any]:
    """Build stage two with only the selected tool's argument schema."""
    allowed = set(metadata.get("allowed_names") or [])
    schemas = metadata.get("argument_schemas") or {}
    if selected_name not in allowed or selected_name not in schemas:
        raise MemoryJsonToolBridgeError(
            f"No argument schema is available for selected tool: {selected_name}"
        )

    argument_request = deepcopy(request_data)
    messages = list(argument_request.get("messages") or [])
    messages.extend(
        [
            {
                "role": "assistant",
                "content": json.dumps({"name": selected_name}, separators=(",", ":")),
            },
            {
                "role": "user",
                "content": (
                    f"Now provide the arguments for {selected_name}. Return "
                    "exactly one complete JSON object conforming to "
                    "response_format, with no tool wrapper, prose, or markdown."
                ),
            },
        ]
    )
    argument_request["messages"] = messages
    argument_request["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": f"mirix_{metadata.get('family', 'memory')}_{selected_name}_arguments",
            "strict": True,
            "schema": deepcopy(schemas[selected_name]),
        },
    }
    return argument_request


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
    """Convert the legacy wrapped payload to one native tool call.

    New runtime traffic uses :func:`select_memory_json_tool` followed by
    :func:`convert_memory_json_arguments_response`.  This converter remains for
    compatibility with preserved responses and callers of the original bridge.
    """
    choice, message = _first_choice_and_message(response_data, "tool call")
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

    schema = (metadata.get("argument_schemas") or {}).get(name)
    if schema is not None:
        _validate_json_schema(arguments, schema)
    return _inject_native_tool_call(response_data, choice, message, name, arguments)


def convert_memory_json_arguments_response(
    response_data: dict[str, Any],
    metadata: dict[str, Any],
    selected_name: str,
) -> dict[str, Any]:
    """Validate stage-two arguments and restore one native MIRIX tool call."""
    choice, message = _first_choice_and_message(response_data, "arguments")
    schemas = metadata.get("argument_schemas") or {}
    schema = schemas.get(selected_name)
    if schema is None:
        raise MemoryJsonToolBridgeError(
            f"No argument schema is available for selected tool: {selected_name}"
        )
    arguments = _extract_json_object(message, "arguments", repair_schema=schema)
    _validate_json_schema(arguments, schema)
    return _inject_native_tool_call(
        response_data, choice, message, selected_name, arguments
    )


def _inject_native_tool_call(
    response_data: dict[str, Any],
    choice: dict[str, Any],
    message: dict[str, Any],
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
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


def _first_choice_and_message(
    response_data: dict[str, Any], stage: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    choices = response_data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise MemoryJsonToolBridgeError(
            f"Memory JSON {stage} response received no choices"
        )
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise MemoryJsonToolBridgeError(
            f"Memory JSON {stage} response was truncated (finish_reason=length)"
        )
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        raise MemoryJsonToolBridgeError(
            f"Memory JSON {stage} response message is not an object"
        )
    return choice, message


def _decode_memory_tool_payload(
    raw: str,
    source: str,
    *,
    repair_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode one candidate without leaking model text into logs/errors.

    MIRIX itself parses tool arguments with ``json.loads``, ``demjson3`` and
    ``json_repair`` in that order.  Local vLLM can terminate a constrained
    string token with EOS and return ``finish_reason=stop`` even though the
    surrounding JSON object is not closed.  Reuse MIRIX's official tolerant
    parser for that case, then let the bridge's strict schema validator decide
    whether the repaired object is safe to execute.
    """
    standard_error: json.JSONDecodeError | None = None
    try:
        # vLLM/xgrammar has occasionally returned a schema-valid string value
        # with a literal control character after the OpenAI response round
        # trip. MIRIX uses ``strict=False`` for the same interoperability case.
        payload = json.loads(raw, strict=False)
    except json.JSONDecodeError as exc:
        standard_error = exc
        try:
            from mirix.helpers.json_helpers import parse_json

            payload = parse_json(raw)
        except Exception as repair_exc:
            digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[
                :16
            ]
            raise MemoryJsonToolBridgeError(
                f"Memory JSON tool bridge received invalid JSON in {source}: "
                f"JSONDecodeError(line={exc.lineno},column={exc.colno},"
                f"pos={exc.pos},chars={len(raw)},sha256={digest}); "
                f"MIRIXRepairError(type={type(repair_exc).__name__})"
            ) from repair_exc
        digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]
        logger.warning(
            "Recovered malformed memory JSON in %s with MIRIX's tolerant "
            "parser: JSONDecodeError(line=%d,column=%d,pos=%d,chars=%d,"
            "sha256=%s)",
            source,
            exc.lineno,
            exc.colno,
            exc.pos,
            len(raw),
            digest,
        )
    except TypeError as exc:
        raise MemoryJsonToolBridgeError(
            f"Memory JSON tool bridge received non-string JSON in {source}: "
            f"{type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        suffix = " after MIRIX repair" if standard_error is not None else ""
        raise MemoryJsonToolBridgeError(
            f"Memory JSON tool bridge payload in {source}{suffix} is not an object"
        )
    if standard_error is not None and repair_schema is not None:
        payload, removed_fields = _project_repaired_json_to_schema(
            payload, repair_schema
        )
        payload, discarded_items = _discard_repaired_incomplete_array_tail(
            payload, repair_schema
        )
        if removed_fields:
            logger.warning(
                "Removed %d unsupported field(s) while projecting repaired "
                "memory JSON back to its declared schema",
                removed_fields,
            )
        if discarded_items:
            logger.warning(
                "Discarded %d incomplete trailing array item(s) while "
                "preserving the schema-valid prefix of repaired memory JSON",
                discarded_items,
            )
    return payload


def _extract_json_object(
    message: dict[str, Any],
    stage: str,
    *,
    repair_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recover one JSON object without including model text in diagnostics."""
    errors: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        try:
            return _decode_memory_tool_payload(
                content,
                f"{stage} content",
                repair_schema=repair_schema,
            )
        except MemoryJsonToolBridgeError as exc:
            errors.append(str(exc))

    # Retain compatibility with servers that move constrained content into a
    # reasoning field even though the evaluation explicitly disables thinking.
    for key in ("reasoning_content", "reasoning"):
        candidate = message.get(key)
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        try:
            return _decode_memory_tool_payload(
                candidate,
                f"{stage} {key}",
                repair_schema=repair_schema,
            )
        except MemoryJsonToolBridgeError as exc:
            errors.append(str(exc))

    if errors:
        raise MemoryJsonToolBridgeError("; ".join(errors))
    raise MemoryJsonToolBridgeError(
        f"Memory JSON {stage} response contained no JSON object"
    )


def _project_repaired_json_to_schema(
    value: Any, schema: dict[str, Any]
) -> tuple[Any, int]:
    """Remove repair artifacts only from schema-anchored objects.

    MIRIX's tolerant parser can recover a useful object when a local model
    stops before closing its JSON. A truncated property can occasionally be
    recovered as an additional object key (for example ``tree_path`` on a
    semantic item). The official executor consumes only declared fields.

    Projection is deliberately conservative: unknown fields are removed only
    when ``additionalProperties`` is false, the object has at least one
    required field, and every required field is already present. Complete JSON
    is never projected, and repaired objects with missing required fields
    still fail the normal strict validator.
    """
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            projected, removed = _project_repaired_json_to_schema(value, branch)
            try:
                _validate_json_schema(projected, branch)
            except MemoryJsonToolBridgeError:
                continue
            return projected, removed
        return deepcopy(value), 0

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        may_project = (
            schema.get("additionalProperties") is False
            and bool(required)
            and required.issubset(value)
        )
        projected: dict[str, Any] = {}
        removed = 0
        for name, item in value.items():
            child_schema = properties.get(name)
            if child_schema is None:
                if may_project:
                    removed += 1
                    continue
                projected[name] = deepcopy(item)
                continue
            projected[name], child_removed = _project_repaired_json_to_schema(
                item, child_schema
            )
            removed += child_removed
        return projected, removed

    if isinstance(value, list):
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return deepcopy(value), 0
        projected_items = []
        removed = 0
        for item in value:
            projected, child_removed = _project_repaired_json_to_schema(
                item, item_schema
            )
            projected_items.append(projected)
            removed += child_removed
        return projected_items, removed

    return deepcopy(value), 0


def _discard_repaired_incomplete_array_tail(
    value: Any, schema: dict[str, Any]
) -> tuple[Any, int]:
    """Preserve a complete array prefix when JSON repair invents a tail item.

    A response can stop immediately after starting the next object in an array.
    ``json-repair`` then closes that object as an empty or partially populated
    final item.  Executing that fabricated item is unsafe, while rejecting the
    preceding schema-valid items makes deterministic retries reproduce the same
    truncation indefinitely.

    Discard exactly one final item only when the response has already required
    tolerant parsing, at least one earlier item fully satisfies the item schema,
    and the final object is missing required fields. If it contains none of the
    required fields, any parser-invented keys are part of that unusable tail as
    well. Complete JSON never reaches this function. A lone incomplete item,
    an invalid prefix, an otherwise complete object with unexpected fields, or
    a wrong declared-field type remains a strict validation error.
    """
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            normalized, discarded = _discard_repaired_incomplete_array_tail(
                value, branch
            )
            try:
                _validate_json_schema(normalized, branch)
            except MemoryJsonToolBridgeError:
                continue
            return normalized, discarded
        return deepcopy(value), 0

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        normalized: dict[str, Any] = {}
        discarded = 0
        for name, item in value.items():
            child_schema = properties.get(name)
            if not isinstance(child_schema, dict):
                normalized[name] = deepcopy(item)
                continue
            normalized[name], child_discarded = _discard_repaired_incomplete_array_tail(
                item, child_schema
            )
            discarded += child_discarded
        return normalized, discarded

    if isinstance(value, list):
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return deepcopy(value), 0
        normalized_items: list[Any] = []
        discarded = 0
        for item in value:
            normalized, child_discarded = _discard_repaired_incomplete_array_tail(
                item, item_schema
            )
            normalized_items.append(normalized)
            discarded += child_discarded

        if len(normalized_items) < 2 or not _only_missing_required_object_fields(
            normalized_items[-1], item_schema
        ):
            return normalized_items, discarded
        try:
            for prefix_item in normalized_items[:-1]:
                _validate_json_schema(prefix_item, item_schema)
            _validate_json_schema(normalized_items[:-1], schema)
        except MemoryJsonToolBridgeError:
            return normalized_items, discarded
        return normalized_items[:-1], discarded + 1

    return deepcopy(value), 0


def _only_missing_required_object_fields(value: Any, schema: dict[str, Any]) -> bool:
    """Return whether a repaired tail lacks required executable content."""
    if not isinstance(value, dict) or schema.get("type") != "object":
        return False
    required = set(schema.get("required") or [])
    if not required or required.issubset(value):
        return False
    if required.isdisjoint(value):
        return True
    relaxed_schema = deepcopy(schema)
    relaxed_schema["required"] = []
    try:
        _validate_json_schema(value, relaxed_schema)
    except MemoryJsonToolBridgeError:
        return False
    return True


def _validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate the strict JSON-schema subset emitted for MIRIX tools.

    vLLM is responsible for constrained decoding, but accepted benchmark state
    must not depend on the server silently falling back to unconstrained text.
    MIRIX's converted tool schemas use this finite subset, so a small local
    validator avoids adding another runtime dependency.
    """
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                _validate_json_schema(value, branch, path)
                return
            except MemoryJsonToolBridgeError:
                continue
        raise MemoryJsonToolBridgeError(
            f"Memory JSON schema mismatch at {path}: no anyOf branch matched"
        )

    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected]
    if expected is not None and not any(
        _matches_json_type(value, item) for item in expected_types
    ):
        raise MemoryJsonToolBridgeError(
            f"Memory JSON schema mismatch at {path}: expected {expected}"
        )

    if "enum" in schema and value not in schema["enum"]:
        raise MemoryJsonToolBridgeError(
            f"Memory JSON schema mismatch at {path}: value is outside enum"
        )
    if "const" in schema and value != schema["const"]:
        raise MemoryJsonToolBridgeError(
            f"Memory JSON schema mismatch at {path}: value differs from const"
        )

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise MemoryJsonToolBridgeError(
                f"Memory JSON schema mismatch at {path}: missing required keys "
                f"{','.join(sorted(missing))}"
            )
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise MemoryJsonToolBridgeError(
                    f"Memory JSON schema mismatch at {path}: unexpected keys "
                    f"{','.join(sorted(extras))}"
                )
        for name, item in value.items():
            child_schema = properties.get(name)
            if child_schema is not None:
                _validate_json_schema(item, child_schema, f"{path}.{name}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise MemoryJsonToolBridgeError(
                f"Memory JSON schema mismatch at {path}: too few items"
            )
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise MemoryJsonToolBridgeError(
                f"Memory JSON schema mismatch at {path}: too many items"
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_json_schema(item, item_schema, f"{path}[{index}]")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise MemoryJsonToolBridgeError(
                f"Memory JSON schema mismatch at {path}: string is too short"
            )
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise MemoryJsonToolBridgeError(
                f"Memory JSON schema mismatch at {path}: string is too long"
            )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise MemoryJsonToolBridgeError(
                f"Memory JSON schema mismatch at {path}: number is too small"
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise MemoryJsonToolBridgeError(
                f"Memory JSON schema mismatch at {path}: number is too large"
            )


def _matches_json_type(value: Any, expected: str | None) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, False)


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
    stage: str = "selector",
    selected_name: str | None = None,
) -> dict[str, Any]:
    """Build a fresh corrective request after an unusable guided generation."""
    retry_data = deepcopy(request_data)
    messages = list(retry_data.get("messages") or [])
    allowed_names = ", ".join(metadata.get("allowed_names") or [])
    if stage == "arguments":
        constraint = (
            f"Return only the complete argument object for {selected_name}; "
            "do not include a name/arguments wrapper."
        )
    else:
        constraint = (
            "Return only the complete selector object. The tool name must be "
            f"one of: {allowed_names}."
        )
    messages.append(
        {
            "role": "user",
            "content": (
                "The previous generation was unusable "
                f"({type(error).__name__}). Regenerate the complete answer from "
                "scratch; do not continue or repair the previous text. Return "
                "exactly one complete JSON object conforming to response_format, "
                f"with no prose or markdown. {constraint} Corrective generation "
                f"attempt: {attempt}."
            ),
        }
    )
    retry_data["messages"] = messages
    return retry_data


def describe_memory_json_response(response_data: dict[str, Any]) -> str:
    """Return privacy-safe diagnostics without logging generated memory text."""
    choices = response_data.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    content = message.get("content")
    raw = content if isinstance(content, str) else ""
    stripped = raw.strip()
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]
    usage = response_data.get("usage") or {}
    return (
        f"response_id={response_data.get('id', '')} "
        f"finish_reason={choice.get('finish_reason')} "
        f"completion_tokens={usage.get('completion_tokens')} "
        f"content_chars={len(raw)} content_bytes={len(raw.encode('utf-8'))} "
        f"content_sha256={digest} starts_object={stripped.startswith('{')} "
        f"ends_object={stripped.endswith('}')}"
    )


def convert_core_json_tool_response(
    response_data: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Backward-compatible alias for the generic response converter."""
    return convert_memory_json_tool_response(response_data, metadata)
