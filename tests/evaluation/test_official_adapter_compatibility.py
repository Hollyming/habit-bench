from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIRIX_ROOT = PROJECT_ROOT / "third_party/medmemorybench/methods/MIRIX"
if str(MIRIX_ROOT) not in sys.path:
    sys.path.insert(0, str(MIRIX_ROOT))

from mirix.llm_api.local_json_tool_bridge import (  # noqa: E402
    MemoryJsonToolBridgeError,
    build_memory_json_arguments_request,
    build_memory_json_tool_bridge,
    build_memory_json_retry_request,
    convert_memory_json_arguments_response,
    convert_memory_json_tool_response,
    select_memory_json_tool,
)


def _tool(name: str, properties: dict | None = None) -> dict:
    properties = properties or {}
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


class OfficialAdapterCompatibilityTest(unittest.TestCase):
    def test_mirix_bridge_retains_native_finish_tool(self) -> None:
        response_format, metadata = build_memory_json_tool_bridge(
            [
                _tool(
                    "episodic_memory_insert",
                    {"items": {"type": "array", "items": {"type": "string"}}},
                ),
                _tool("finish_memory_update"),
            ]
        )
        self.assertEqual(
            set(metadata["allowed_names"]),
            {"episodic_memory_insert", "finish_memory_update"},
        )
        selector = response_format["json_schema"]["schema"]
        self.assertEqual(
            set(selector["properties"]["name"]["enum"]),
            {"episodic_memory_insert", "finish_memory_update"},
        )
        self.assertEqual(
            metadata["argument_schemas"]["episodic_memory_insert"]
            ["properties"]["items"]["type"],
            "array",
        )

    def test_mirix_bridge_builds_exact_second_stage_and_validates_it(self) -> None:
        selector_request, metadata = build_memory_json_tool_bridge(
            [
                _tool(
                    "episodic_memory_insert",
                    {
                        "items": {
                            "type": "array",
                            "maxItems": 1,
                            "items": {"type": "string", "maxLength": 8},
                        }
                    },
                ),
                _tool("finish_memory_update"),
            ]
        )
        selected = select_memory_json_tool(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"name":"episodic_memory_insert"}'
                        },
                    }
                ]
            },
            metadata,
        )
        request = build_memory_json_arguments_request(
            {
                "model": "Qwen3-4B",
                "messages": [{"role": "user", "content": "remember"}],
                "response_format": selector_request,
            },
            metadata,
            selected,
        )
        self.assertEqual(
            request["response_format"]["json_schema"]["schema"],
            metadata["argument_schemas"][selected],
        )
        converted = convert_memory_json_arguments_response(
            {
                "id": "chatcmpl-args",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"items":["fact"]}'},
                    }
                ],
            },
            metadata,
            selected,
        )
        self.assertEqual(
            converted["choices"][0]["message"]["tool_calls"][0]
            ["function"]["name"],
            selected,
        )
        with self.assertRaisesRegex(MemoryJsonToolBridgeError, "too many items"):
            convert_memory_json_arguments_response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": '{"items":["a","b"]}'},
                        }
                    ]
                },
                metadata,
                selected,
            )

    def test_mirix_bridge_repairs_then_strictly_validates_arguments(self) -> None:
        _, metadata = build_memory_json_tool_bridge(
            [
                _tool(
                    "semantic_memory_insert",
                    {
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "maxLength": 8},
                        }
                    },
                ),
                _tool("finish_memory_update"),
            ]
        )
        converted = convert_memory_json_arguments_response(
            {
                "id": "chatcmpl-repaired",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"items":["fact"]'},
                    }
                ],
            },
            metadata,
            "semantic_memory_insert",
        )
        arguments = json.loads(
            converted["choices"][0]["message"]["tool_calls"][0]
            ["function"]["arguments"]
        )
        self.assertEqual(arguments, {"items": ["fact"]})

        with self.assertRaisesRegex(MemoryJsonToolBridgeError, "too long"):
            convert_memory_json_arguments_response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": '{"items":["far-too-long"]'
                            },
                        }
                    ]
                },
                metadata,
                "semantic_memory_insert",
            )

    def test_mirix_bridge_recovers_native_call_when_content_is_empty(self) -> None:
        response = {
            "id": "chatcmpl-native",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-existing",
                                "type": "function",
                                "function": {
                                    "name": "finish_memory_update",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                }
            ],
        }

        converted = convert_memory_json_tool_response(
            response,
            {
                "allowed_names": ["finish_memory_update"],
                "family": "episodic",
            },
        )

        call = converted["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "finish_memory_update")
        self.assertEqual(json.loads(call["function"]["arguments"]), {})

    def test_mirix_bridge_recovers_guided_json_from_reasoning_field(self) -> None:
        response = {
            "id": "chatcmpl-reasoning",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": (
                            '{"name":"finish_memory_update","arguments":{}}'
                        ),
                    },
                }
            ],
        }

        converted = convert_memory_json_tool_response(
            response,
            {
                "allowed_names": ["finish_memory_update"],
                "family": "core",
            },
        )
        self.assertEqual(
            converted["choices"][0]["message"]["tool_calls"][0]["function"][
                "name"
            ],
            "finish_memory_update",
        )
    def test_mirix_retry_is_a_fresh_corrective_generation(self) -> None:
        original = {
            "model": "Qwen3-4B",
            "messages": [{"role": "user", "content": "remember this"}],
        }
        retry = build_memory_json_retry_request(
            original,
            {"allowed_names": ["finish_memory_update"]},
            attempt=2,
            error=ValueError("empty"),
        )

        self.assertEqual(len(original["messages"]), 1)
        self.assertEqual(len(retry["messages"]), 2)
        correction = retry["messages"][-1]["content"]
        self.assertIn("Regenerate the complete answer from scratch", correction)
        self.assertIn("finish_memory_update", correction)

    def test_mirix_openai_client_retries_empty_guided_response(self) -> None:
        from mirix.llm_api.openai_client import OpenAIClient

        response_payloads = [
            {
                "id": "chatcmpl-empty",
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": ""},
                    }
                ],
            },
            {
                "id": "chatcmpl-selector",
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"name":"finish_memory_update"}',
                        },
                    }
                ],
            },
            {
                "id": "chatcmpl-invalid-arguments",
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"unexpected":1}',
                        },
                    }
                ],
            },
            {
                "id": "chatcmpl-arguments",
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "{}",
                        },
                    }
                ],
            },
        ]
        requests = []

        class _Response:
            object = "chat.completion"

            def __init__(self, payload):
                self.payload = payload

            def model_dump(self):
                return self.payload

        class _Completions:
            async def create(self, **kwargs):
                requests.append(kwargs)
                return _Response(response_payloads.pop(0))

        class _FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = type("_Chat", (), {"completions": _Completions()})()

        client = OpenAIClient.__new__(OpenAIClient)
        client._prepare_client_kwargs = AsyncMock(
            return_value={"api_key": "test", "base_url": "http://test/v1"}
        )
        request = {
            "model": "Qwen3-4B",
            "messages": [{"role": "user", "content": "remember this"}],
            "response_format": {"type": "json_schema"},
            "_mirix_memory_json_tool_bridge": {
                "allowed_names": ["finish_memory_update"],
                "argument_schemas": {
                    "finish_memory_update": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    }
                },
                "family": "core",
            },
        }
        with (
            patch(
                "mirix.llm_api.openai_client.AsyncOpenAI",
                _FakeAsyncOpenAI,
            ),
            patch.dict(
                os.environ,
                {
                    "MIRIX_JSON_TOOL_BRIDGE_ATTEMPTS": "2",
                    "MIRIX_JSON_TOOL_BRIDGE_SEED": "42",
                    "MIRIX_JSON_TOOL_BRIDGE_RETRY_TEMPERATURE": "0.2",
                },
                clear=False,
            ),
        ):
            converted = asyncio.run(client.request(request))

        self.assertEqual(len(requests), 4)
        self.assertEqual(len(requests[0]["messages"]), 1)
        self.assertEqual(len(requests[1]["messages"]), 2)
        self.assertEqual(len(requests[2]["messages"]), 3)
        self.assertEqual(len(requests[3]["messages"]), 4)
        self.assertEqual(requests[0]["seed"], 42)
        self.assertEqual(requests[1]["seed"], 43)
        self.assertEqual(requests[1]["temperature"], 0.2)
        self.assertEqual(requests[2]["seed"], 42)
        self.assertEqual(requests[3]["seed"], 43)
        self.assertEqual(requests[3]["temperature"], 0.2)
        self.assertEqual(
            requests[0]["extra_body"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertEqual(
            requests[1]["extra_body"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertEqual(
            requests[2]["response_format"]["json_schema"]["schema"]
            ["additionalProperties"],
            False,
        )
        self.assertIn(
            "do not include a name/arguments wrapper",
            requests[3]["messages"][-1]["content"],
        )
        self.assertEqual(
            converted["choices"][0]["message"]["tool_calls"][0]["function"][
                "name"
            ],
            "finish_memory_update",
        )
        exhausted = MemoryJsonToolBridgeError("exhausted")
        self.assertIs(client.handle_llm_error(exhausted), exhausted)

    def test_mirix_bridge_serializes_two_stage_calls_per_endpoint(self) -> None:
        from mirix.llm_api.openai_client import OpenAIClient

        active = 0
        maximum_active = 0

        class _Response:
            object = "chat.completion"

            def __init__(self, payload):
                self.payload = payload

            def model_dump(self):
                return self.payload

        class _Completions:
            async def create(self, **kwargs):
                nonlocal active, maximum_active
                active += 1
                maximum_active = max(maximum_active, active)
                await asyncio.sleep(0.01)
                active -= 1
                schema = kwargs["response_format"]["json_schema"]["schema"]
                if "name" in schema.get("properties", {}):
                    content = '{"name":"finish_memory_update"}'
                else:
                    content = "{}"
                return _Response(
                    {
                        "id": "chatcmpl-serialized",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": content},
                            }
                        ],
                    }
                )

        class _FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = type("_Chat", (), {"completions": _Completions()})()

        client = OpenAIClient.__new__(OpenAIClient)
        client._prepare_client_kwargs = AsyncMock(
            return_value={"api_key": "test", "base_url": "http://same/v1"}
        )
        metadata = {
            "allowed_names": ["finish_memory_update"],
            "argument_schemas": {
                "finish_memory_update": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                }
            },
            "family": "core",
        }

        def request():
            return {
                "model": "Qwen3-4B",
                "messages": [{"role": "user", "content": "remember"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "mirix_core_memory_tool_selector",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": ["finish_memory_update"],
                                }
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                },
                "_mirix_memory_json_tool_bridge": metadata,
            }

        async def run_both():
            return await asyncio.gather(
                client.request(request()), client.request(request())
            )

        with (
            patch("mirix.llm_api.openai_client.AsyncOpenAI", _FakeAsyncOpenAI),
            patch.dict(
                os.environ,
                {
                    "MIRIX_JSON_TOOL_BRIDGE_ATTEMPTS": "1",
                    "MIRIX_JSON_TOOL_BRIDGE_SERIALIZE": "1",
                },
                clear=False,
            ),
        ):
            results = asyncio.run(run_both())

        self.assertEqual(maximum_active, 1)
        self.assertEqual(len(results), 2)

    def test_mirix_adapted_profile_uses_five_bounded_bridge_attempts(self) -> None:
        import yaml

        profile = yaml.safe_load(
            (
                PROJECT_ROOT
                / "third_party/medmemorybench/configs/method_config/"
                "mirix_qwen3-8b_adapted.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(profile["agent_params"]["json_tool_bridge_attempts"], 5)

if __name__ == "__main__":
    unittest.main()
