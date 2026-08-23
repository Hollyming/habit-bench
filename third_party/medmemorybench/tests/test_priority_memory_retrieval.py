import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from methods.amem_agent import AMemAgent
from methods.letta_agent import LettaAgent
from methods.lightmem_agent import LightMemAgent
from methods.mem0_agent import Mem0Agent
from methods.memos_agent import MemOSAgent
from methods.memrl_agent import MemRLAgent
from methods.mirix_agent import MIRIXAgent
from methods.base import MemoryRetrievalResult
from src.agent import AgentManager
from src.config import APIConfig, ConfigLoader
from methods.mem0.utils.factory import load_class
from methods.mem0.memory.main import Memory


class _FakeMem0Memory:
    def search(self, *, query, user_id, limit):
        self.last_call = (query, user_id, limit)
        return {
            "results": [
                {"id": "m1", "memory": "prefers oats", "score": 0.9},
                {"id": "m2", "memory": "avoids peanuts", "score": 0.8},
            ]
        }


class _FakeAMemSystem:
    def find_related_memories(self, question, *, k):
        self.last_call = (question, k)
        return "note one\nnote two", [3, 7]


class _FakeSearchSystem:
    def search(self, question, *, top_k):
        self.last_call = (question, top_k)
        return [SimpleNamespace(memory="stored preference", metadata=None)]


class _FakeMemRLService:
    def retrieve_query(self, **kwargs):
        self.last_call = kwargs
        return (
            {
                "selected": [
                    {
                        "memory_id": "r1",
                        "content": "successful routine",
                        "similarity": 0.8,
                        "q_estimate": 0.7,
                        "score": 0.75,
                    }
                ],
                "candidates": ["r1"],
                "simmax": 0.8,
            },
            [0.8],
        )


class _FakeCurrentMemRLService:
    def retrieve_query(self, **kwargs):
        self.last_call = kwargs
        return {
            "selected": [],
            "candidates": [],
            "simmax": 0.0,
        }


class PriorityRetrievalTests(unittest.TestCase):
    def test_adapted_flags_reach_mem0_and_letta_agents(self):
        loader = ConfigLoader()
        manager = AgentManager.__new__(AgentManager)
        manager._api_config = APIConfig()

        manager.method_config = loader.load_method_config("mem0_qwen3-8b_adapted")
        mem0_params = manager._build_agent_params("mem0")
        self.assertTrue(mem0_params["compact_local_prompts"])
        self.assertTrue(mem0_params["strict_json_schema"])
        self.assertEqual(mem0_params["action_generation_attempts"], 5)
        self.assertTrue(mem0_params["normalize_unknown_mutations"])

        manager.method_config = loader.load_method_config("letta_qwen3-8b_adapted")
        letta_params = manager._build_agent_params("letta")
        self.assertFalse(letta_params["use_native_memorize"])
        self.assertFalse(letta_params["use_native_query"])
        self.assertTrue(letta_params["message_buffer_autoclear"])

        manager.method_config = loader.load_method_config("mirix_qwen3-8b_adapted")
        mirix_params = manager._build_agent_params("mirix")
        self.assertTrue(mirix_params["enforce_single_meta_update"])
        self.assertTrue(mirix_params["strict_internal_errors"])
        self.assertTrue(mirix_params["bounded_memory_tool_schema"])
        self.assertTrue(mirix_params["required_tool_choice"])
        self.assertTrue(mirix_params["core_json_tool_bridge"])
        self.assertEqual(mirix_params["json_tool_bridge_attempts"], 5)
        self.assertTrue(mirix_params["normalize_missing_update_ids"])
        self.assertEqual(mirix_params["memory_tool_max_items"], 2)
        self.assertEqual(mirix_params["memory_tool_max_string_chars"], 512)
        self.assertEqual(mirix_params["core_memory_tool_max_string_chars"], 256)
        self.assertEqual(mirix_params["core_memory_max_tokens"], 1024)
        self.assertEqual(mirix_params["memory_agent_max_tokens"], 8192)
        self.assertEqual(mirix_params["core_memory_input_max_chars"], 256)
        self.assertEqual(mirix_params["memory_agent_input_max_chars"], 2048)

    def test_mirix_core_json_tool_bridge_builds_one_bounded_call(self):
        import importlib.util

        bridge_path = (
            Path(__file__).parents[1]
            / "methods"
            / "MIRIX"
            / "mirix"
            / "llm_api"
            / "local_json_tool_bridge.py"
        )
        spec = importlib.util.spec_from_file_location(
            "mirix_local_json_tool_bridge_standalone", bridge_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        parameters = {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["human", "persona"]},
                "content": {"type": "string", "maxLength": 256},
            },
            "required": ["label", "content"],
            "additionalProperties": False,
        }
        tools = [
            {
                "type": "function",
                "function": {"name": name, "parameters": parameters},
            }
            for name in ("core_memory_append", "core_memory_rewrite")
        ]
        # This mirrors the real MIRIX core child: universal helpers are attached
        # in addition to the two core mutation tools. Only the native terminal
        # finish tool remains in the bridge; read/message helpers stay excluded.
        universal_parameters = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }
        tools.extend(
            {
                "type": "function",
                "function": {"name": name, "parameters": universal_parameters},
            }
            for name in (
                "search_in_memory",
                "finish_memory_update",
                "list_memory_within_timerange",
            )
        )
        tools.extend(
            {
                "type": "function",
                "function": {"name": name, "parameters": universal_parameters},
            }
            for name in ("conversation_search", "send_intermediate_message")
        )
        self.assertTrue(module.is_core_memory_tool_request(tools))
        non_core_tools = tools + [
            {
                "type": "function",
                "function": {
                    "name": "episodic_memory_insert",
                    "parameters": universal_parameters,
                },
            }
        ]
        self.assertFalse(module.is_core_memory_tool_request(non_core_tools))
        response_format, metadata = module.build_core_json_tool_bridge(tools)
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(
            metadata["allowed_names"],
            [
                "core_memory_append",
                "core_memory_rewrite",
                "finish_memory_update",
            ],
        )
        self.assertEqual(
            schema["properties"]["name"]["enum"],
            metadata["allowed_names"],
        )
        self.assertEqual(
            metadata["argument_schemas"]["core_memory_append"]["properties"]
            ["content"]["maxLength"],
            256,
        )

        response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"name":"core_memory_append","arguments":{"label":"human","content":"fact"}}',
                    },
                }
            ],
        }
        converted = module.convert_core_json_tool_response(response, metadata)
        choice = converted["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        calls = choice["message"]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "core_memory_append")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"])["content"], "fact"
        )

        bounded_items = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "maxLength": 512}
                        },
                        "required": ["summary"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }
        episodic_tools = [
            {
                "type": "function",
                "function": {
                    "name": "episodic_memory_insert",
                    "parameters": bounded_items,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_episodic_memory",
                    "parameters": universal_parameters,
                },
            },
            tools[-1],
        ]
        self.assertEqual(
            module.identify_memory_tool_family(episodic_tools), "episodic"
        )
        episodic_format, episodic_metadata = module.build_memory_json_tool_bridge(
            episodic_tools
        )
        self.assertEqual(episodic_metadata["family"], "episodic")
        self.assertEqual(
            episodic_format["json_schema"]["schema"]["properties"]["name"]
            ["enum"],
            ["check_episodic_memory", "episodic_memory_insert"],
        )
        self.assertEqual(
            episodic_metadata["argument_schemas"]["episodic_memory_insert"]
            ["properties"]["items"]["maxItems"],
            2,
        )
        self.assertIsNone(
            module.identify_memory_tool_family(episodic_tools + tools[:1])
        )

        procedural_tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": bounded_items,
                },
            }
            for name in ("procedural_memory_insert", "procedural_memory_update")
        ]
        procedural_tools.append(tools[-1])
        procedural_format, procedural_metadata = module.build_memory_json_tool_bridge(
            procedural_tools
        )
        self.assertEqual(procedural_metadata["family"], "procedural")
        self.assertTrue(procedural_format["json_schema"]["strict"])
        self.assertNotIn(
            "send_intermediate_message", procedural_metadata["allowed_names"]
        )

    def test_mirix_missing_update_ids_are_normalized_only_for_adapted_run(self):
        import asyncio
        import importlib.util

        helper_path = (
            Path(__file__).parents[1]
            / "methods"
            / "MIRIX"
            / "mirix"
            / "functions"
            / "update_id_normalization.py"
        )
        spec = importlib.util.spec_from_file_location(
            "mirix_update_id_normalization_standalone", helper_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        kept, missing = module.normalize_existing_update_ids(
            ["stale-name", "sem_real", "stale-name"],
            ["sem_real"],
            allow_missing=True,
        )
        self.assertEqual(kept, ["sem_real"])
        self.assertEqual(missing, ["stale-name", "stale-name"])
        with self.assertRaisesRegex(ValueError, "stale-name"):
            module.normalize_existing_update_ids(
                ["stale-name"], [], allow_missing=False
            )

        class MissingItem(Exception):
            pass

        async def fetch_item(item_id):
            if item_id != "sem_real":
                raise MissingItem(item_id)
            return object()

        kept, missing = asyncio.run(
            module.normalize_existing_update_ids_with_fetch(
                ["stale-name", "sem_real", "stale-name"],
                fetch_item,
                missing_exception=MissingItem,
                allow_missing=True,
            )
        )
        self.assertEqual(kept, ["sem_real"])
        self.assertEqual(missing, ["stale-name"])

    def test_mirix_all_replace_style_update_families_preflight_old_ids(self):
        import ast

        tools_path = (
            Path(__file__).parents[1]
            / "methods"
            / "MIRIX"
            / "mirix"
            / "functions"
            / "function_sets"
            / "memory_tools.py"
        )
        source = tools_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: ast.get_source_segment(source, node) or ""
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef)
        }
        for function_name in (
            "resource_memory_update",
            "procedural_memory_update",
            "semantic_memory_update",
            "knowledge_vault_update",
        ):
            self.assertIn(function_name, functions)
            self.assertIn(
                "normalize_existing_update_ids_with_fetch",
                functions[function_name],
            )

    def test_mirix_adapted_tool_schema_bounds_nested_memory_output(self):
        import importlib.util
        import os
        from typing import List
        from unittest.mock import patch

        from pydantic import BaseModel, Field

        schema_path = (
            Path(__file__).parents[1]
            / "methods"
            / "MIRIX"
            / "mirix"
            / "functions"
            / "schema_generator.py"
        )
        spec = importlib.util.spec_from_file_location(
            "mirix_schema_generator_standalone", schema_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class EpisodicEventForLLM(BaseModel):
            event_type: str = Field(..., description="Event type")
            summary: str = Field(..., description="Short summary")
            details: str = Field(..., description="Event details")
            actor: str = Field(..., description="Event actor")
            occurred_at: str = Field(..., description="Event timestamp")

        def episodic_memory_insert(items: List[EpisodicEventForLLM]):
            """Insert episodic memory items.

            Args:
                items: List of episodic memory items to insert.
            """

        def core_memory_append(label: str, content: str):
            """Append a core-memory delta.

            Args:
                label: Target core-memory label.
                content: Single-line durable delta.
            """

        with patch.dict(
            os.environ,
            {
                "MIRIX_BOUNDED_MEMORY_TOOL_SCHEMA": "1",
                "MIRIX_MEMORY_TOOL_MAX_ITEMS": "4",
                "MIRIX_MEMORY_TOOL_MAX_STRING_CHARS": "1024",
                "MIRIX_CORE_MEMORY_TOOL_MAX_STRING_CHARS": "256",
            },
        ):
            schema = module.generate_schema(episodic_memory_insert)
            core_schema = module.generate_schema(core_memory_append)

        items = schema["parameters"]["properties"]["items"]
        self.assertEqual(items["maxItems"], 4)
        item = items["items"]
        self.assertEqual(item["properties"]["details"]["maxLength"], 1024)
        self.assertEqual(item["properties"]["summary"]["maxLength"], 512)
        core_content = core_schema["parameters"]["properties"]["content"]
        self.assertEqual(core_content["maxLength"], 256)
        self.assertEqual(core_content["pattern"], r"^[^\u0000-\u001F]*$")

    def test_mirix_runtime_bounds_apply_when_vllm_ignores_json_schema(self):
        import importlib.util
        import os
        from unittest.mock import patch

        validator_path = (
            Path(__file__).parents[1]
            / "methods"
            / "MIRIX"
            / "mirix"
            / "agent"
            / "tool_validators.py"
        )
        spec = importlib.util.spec_from_file_location(
            "mirix_tool_validators_standalone", validator_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        payload = {
            "items": [
                {"summary": "s" * 700, "details": "d" * 1400}
                for _ in range(7)
            ]
        }
        with patch.dict(
            os.environ,
            {
                "MIRIX_BOUNDED_MEMORY_TOOL_SCHEMA": "1",
                "MIRIX_MEMORY_TOOL_MAX_ITEMS": "4",
                "MIRIX_MEMORY_TOOL_MAX_STRING_CHARS": "1024",
            },
        ):
            bounded, changes = module.bound_memory_tool_args(
                "episodic_memory_insert", payload
            )
        self.assertEqual(len(bounded["items"]), 4)
        self.assertEqual(len(bounded["items"][0]["summary"]), 512)
        self.assertEqual(len(bounded["items"][0]["details"]), 1024)
        self.assertTrue(changes)

    def test_mirix_runtime_strips_visual_core_line_prefixes(self):
        import importlib.util
        import os
        from unittest.mock import patch

        validator_path = (
            Path(__file__).parents[1]
            / "methods"
            / "MIRIX"
            / "mirix"
            / "agent"
            / "tool_validators.py"
        )
        spec = importlib.util.spec_from_file_location(
            "mirix_tool_validators_core_prefix", validator_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with patch.dict(os.environ, {"MIRIX_BOUNDED_MEMORY_TOOL_SCHEMA": "1"}):
            bounded, changes = module.bound_memory_tool_args(
                "core_memory_append",
                {"label": "human", "content": "Line 6: durable fact\nLine 7: preference"},
            )
        self.assertEqual(bounded["content"], "durable fact preference")
        self.assertIn("content:removed-line-prefix", changes)
        self.assertIn("content:removed-control-whitespace", changes)

    def test_mirix_adapted_input_bounds_apply_to_all_memory_children(self):
        import importlib.util
        import os
        from unittest.mock import patch

        validator_path = (
            Path(__file__).parents[1]
            / "methods"
            / "MIRIX"
            / "mirix"
            / "agent"
            / "tool_validators.py"
        )
        spec = importlib.util.spec_from_file_location(
            "mirix_tool_validators_input_bounds", validator_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        env = {
            "MIRIX_BOUNDED_MEMORY_TOOL_SCHEMA": "1",
            "MIRIX_CORE_INPUT_MAX_CHARS": "512",
            "MIRIX_MEMORY_AGENT_INPUT_MAX_CHARS": "2048",
        }
        with patch.dict(os.environ, env):
            core = module.bound_adapted_memory_input(
                SimpleNamespace(content="c" * 4000), "core"
            )
            semantic = module.bound_adapted_memory_input(
                SimpleNamespace(content="s" * 4000), "semantic"
            )
        self.assertEqual(len(core.content), 512)
        self.assertIn("adapted core middle omitted", core.content)
        self.assertNotIn("\n", core.content)
        self.assertEqual(len(semantic.content), 2048)
        self.assertIn("adapted semantic middle omitted", semantic.content)

    def test_mem0_malformed_json_is_not_silently_treated_as_empty_memory(self):
        with self.assertRaisesRegex(ValueError, "Failed to parse Mem0 JSON"):
            Memory._parse_json_response(None, '{"memory": [{"text": "truncated"}')

    def test_mem0_validates_full_action_batch_before_mutation(self):
        mapping = {"0": "uuid-0", "1": "uuid-1"}
        with self.assertRaisesRegex(ValueError, "unsupported event None"):
            Memory._validate_memory_actions(
                {"memory": [{"id": "new", "text": "fact", "event": None}]},
                mapping,
            )
        delete_then_update = Memory._validate_memory_actions(
            {
                "memory": [
                    {"id": "0", "text": "old", "event": "DELETE"},
                    {"id": "0", "text": "new", "event": "UPDATE"},
                ]
            },
            mapping,
        )
        self.assertEqual(
            delete_then_update,
            [{"id": "0", "text": "new", "event": "UPDATE"}],
        )

        repeated_updates = Memory._validate_memory_actions(
            {
                "memory": [
                    {"id": "0", "text": "intermediate", "event": "UPDATE"},
                    {"id": "0", "text": "final", "event": "UPDATE"},
                ]
            },
            mapping,
        )
        self.assertEqual(
            repeated_updates,
            [{"id": "0", "text": "final", "event": "UPDATE"}],
        )

        actions = Memory._validate_memory_actions(
            {
                "memory": [
                    {"id": "new", "text": "  added fact  ", "event": "ADD"},
                    {"id": "1", "text": "updated fact", "event": "UPDATE"},
                    {"id": "ignored", "text": "unchanged", "event": "NONE"},
                ]
            },
            mapping,
        )
        self.assertEqual([row["event"] for row in actions], ["ADD", "UPDATE"])
        self.assertEqual(actions[0]["text"], "added fact")

        with self.assertRaisesRegex(ValueError, "unknown ID '17'"):
            Memory._validate_memory_actions(
                {"memory": [{"id": "17", "text": "new fact", "event": "UPDATE"}]},
                mapping,
            )
        repaired = Memory._validate_memory_actions(
            {
                "memory": [
                    {"id": "17", "text": "new fact", "event": "UPDATE"},
                    {"id": "18", "text": "stale fact", "event": "DELETE"},
                ]
            },
            mapping,
            normalize_unknown_mutations=True,
        )
        self.assertEqual(repaired, [{"text": "new fact", "event": "ADD"}])

    def test_mem0_strict_schemas_bound_fact_and_action_generation(self):
        memory = Memory.__new__(Memory)
        memory.strict_json_schema = True
        facts = memory._fact_response_format()["json_schema"]["schema"]
        actions = memory._action_response_format()["json_schema"]["schema"]
        self.assertEqual(facts["properties"]["facts"]["maxItems"], 24)
        self.assertEqual(
            facts["properties"]["facts"]["items"]["maxLength"], 512
        )
        action_item = actions["properties"]["memory"]["items"]
        self.assertEqual(action_item["required"], ["id", "text", "event"])
        self.assertEqual(action_item["properties"]["text"]["maxLength"], 512)
        self.assertNotIn(
            None,
            action_item["properties"]["event"]["enum"],
        )

    def test_mem0_factory_never_falls_back_to_external_package(self):
        error = ModuleNotFoundError("vendored dependency missing")
        with patch(
            "methods.mem0.utils.factory.importlib.import_module",
            side_effect=error,
        ) as import_module:
            with self.assertRaisesRegex(
                ModuleNotFoundError,
                "vendored dependency missing",
            ):
                load_class("mem0.embeddings.huggingface.HuggingFaceEmbedding")
        import_module.assert_called_once_with(
            "methods.mem0.embeddings.huggingface"
        )

    def test_mem0_retrieve_does_not_call_answer_model(self):
        agent = Mem0Agent.__new__(Mem0Agent)
        agent._memory = _FakeMem0Memory()
        agent._user_id = "context_4"
        agent._context_id = 4
        agent.retrieve_num = 2
        agent.embedding_model = "fake-embedding"
        agent.embedding_provider = "local"

        result = agent.retrieve("what should I eat?")

        self.assertEqual(result.retrieved_count, 2)
        self.assertEqual(result.memory_context, "- prefers oats\n- avoids peanuts")
        self.assertEqual(result.retrieved_memories[0]["id"], "m1")
        self.assertEqual(
            agent._memory.last_call,
            ("what should I eat?", "context_4", 2),
        )

    def test_amem_retrieve_does_not_call_answer_model(self):
        agent = AMemAgent.__new__(AMemAgent)
        system = _FakeAMemSystem()
        agent._context_id = 5
        agent._amem_systems = {5: system}
        agent.retrieve_num = 2

        result = agent.retrieve("recall my preference")

        self.assertEqual(result.memory_context, "note one\nnote two")
        self.assertEqual(result.retrieved_count, 2)
        self.assertEqual(result.extra["indices"], [3, 7])
        self.assertEqual(system.last_call, ("recall my preference", 2))

    def test_lightmem_retrieve_only(self):
        native = SimpleNamespace(
            retrieve=lambda *, query, limit: "first memory\nsecond memory"
        )
        agent = LightMemAgent.__new__(LightMemAgent)
        agent._context_id = 1
        agent._lightmem_instances = {1: native}
        agent.retrieve_num = 2
        result = agent.retrieve("query")
        self.assertEqual(result.retrieved_count, 2)
        self.assertEqual(result.memory_context, "first memory\nsecond memory")

    def test_memos_retrieve_only(self):
        native = _FakeSearchSystem()
        agent = MemOSAgent.__new__(MemOSAgent)
        agent._context_id = 1
        agent._memory_systems = {1: native}
        agent.retrieve_num = 2
        agent.max_question_tokens = 100
        agent.text_mem_type = "general_text"
        agent._truncate_to_tokens = lambda text, limit: text
        result = agent.retrieve("query")
        self.assertEqual(result.retrieved_count, 1)
        self.assertIn("stored preference", result.memory_context)

    def test_memos_qdrant_state_is_context_isolated(self):
        with TemporaryDirectory() as temp_root:
            agent = MemOSAgent.__new__(MemOSAgent)
            agent.text_mem_type = "general_text"
            agent.memos_backend = "openai"
            agent.memos_model = "fake"
            agent._memos_api_key = "fake"
            agent._memos_api_base = "http://localhost"
            agent.embedding_provider = "local"
            agent.embedding_model_path = "/models/fake"
            agent.embedding_model = "fake"
            agent.embedding_dim = 512
            agent.storage_root = Path(temp_root)
            agent._MemoryConfigFactory = lambda **kwargs: kwargs

            config = agent._build_memory_config(17)

            vector_config = config["config"]["vector_db"]["config"]
            self.assertEqual(
                vector_config["path"],
                str(Path(temp_root) / "context_17" / "qdrant"),
            )

    def test_memrl_retrieve_only(self):
        service = _FakeMemRLService()
        agent = MemRLAgent.__new__(MemRLAgent)
        agent._memory_service = service
        agent.candidate_top_k = 12
        agent.similarity_threshold = 0.2
        agent.retrieve_num = 5
        agent.initial_q = 0.5
        agent.max_question_tokens = 100
        agent.query_memory_item_tokens = 100
        agent.query_memory_context_tokens = 200
        agent._llm_client = SimpleNamespace(count_tokens=len)
        agent._truncate_to_tokens = lambda text, limit: text
        result = agent.retrieve("query")
        self.assertEqual(result.retrieved_count, 1)
        self.assertIn("successful routine", result.memory_context)
        self.assertEqual(service.last_call["task_description"], "query")
        self.assertLessEqual(result.extra["memory_context_tokens"], 200)

    def test_memrl_retrieve_enforces_exact_item_and_context_budget(self):
        service = _FakeMemRLService()
        agent = MemRLAgent.__new__(MemRLAgent)
        agent._memory_service = service
        agent.candidate_top_k = 12
        agent.similarity_threshold = 0.2
        agent.retrieve_num = 5
        agent.initial_q = 0.5
        agent.max_question_tokens = 100
        agent.query_memory_item_tokens = 10
        agent.query_memory_context_tokens = 45
        agent._tokenizer = SimpleNamespace(
            encode=lambda text: list(text), decode=lambda tokens: "".join(tokens)
        )
        agent._llm_client = SimpleNamespace(count_tokens=len)

        result = agent.retrieve("query")

        self.assertEqual(result.retrieved_count, 1)
        self.assertEqual(result.retrieved_memories[0]["memory"], "successful")
        self.assertTrue(result.retrieved_memories[0]["truncated"])
        self.assertLessEqual(len(result.memory_context), 45)
        self.assertEqual(result.extra["memory_context_tokens"], len(result.memory_context))

    def test_memrl_current_dict_return_signature(self):
        service = _FakeCurrentMemRLService()
        agent = MemRLAgent.__new__(MemRLAgent)
        agent._memory_service = service
        agent.candidate_top_k = 12
        agent.similarity_threshold = 0.2
        agent.retrieve_num = 5
        agent.initial_q = 0.5
        agent.max_question_tokens = 100
        agent.query_memory_item_tokens = 100
        agent.query_memory_context_tokens = 200
        agent._llm_client = SimpleNamespace(count_tokens=len)
        agent._truncate_to_tokens = lambda text, limit: text
        result = agent.retrieve("empty query")
        self.assertEqual(result.retrieved_count, 0)
        self.assertEqual(result.memory_context, "")

    def test_mirix_retrieve_only(self):
        agent = MIRIXAgent.__new__(MIRIXAgent)
        agent._retrieve = lambda question: [
            {"memory": "episodic item", "type": "episodic", "score": 0.9}
        ]
        result = agent.retrieve("query")
        self.assertEqual(result.retrieved_count, 1)
        self.assertEqual(result.memory_context, "[episodic] episodic item")

    def test_mirix_meta_agent_seeds_required_core_blocks(self):
        specs = MIRIXAgent._meta_agent_specs()
        core = specs[0]["core_memory_agent"]
        self.assertEqual(
            [block["label"] for block in core["blocks"]],
            ["human", "persona"],
        )
        self.assertIn("episodic_memory_agent", specs)
        self.assertIn("semantic_memory_agent", specs)

    def test_mirix_adapted_meta_flow_updates_once_then_finishes(self):
        rules = MIRIXAgent._meta_agent_tool_flow_spec()
        self.assertEqual(rules[0], {
            "tool_name": "trigger_memory_update",
            "type": "run_first",
        })
        self.assertEqual(rules[1]["children"], ["finish_memory_update"])
        self.assertEqual(rules[2], {
            "tool_name": "finish_memory_update",
            "type": "exit_loop",
        })

    def test_mirix_core_prompt_forbids_transcript_replay(self):
        agent = MIRIXAgent.__new__(MIRIXAgent)
        agent.memory_tool_max_items = 4
        agent.memory_tool_max_string_chars = 1024
        agent.core_memory_tool_max_string_chars = 256
        prompts = agent._bounded_memory_system_prompts()
        core_prompt = prompts["core_memory_agent"]
        self.assertIn("Never quote, summarize, or reproduce", core_prompt)
        self.assertIn("1024 characters", core_prompt)
        self.assertIn("finish_memory_update", core_prompt)

    def test_mirix_memory_children_receive_separate_completion_caps(self):
        agent = MIRIXAgent.__new__(MIRIXAgent)
        agent.core_memory_max_tokens = 1024
        agent.memory_agent_max_tokens = 2048
        self.assertEqual(agent._memory_agent_completion_cap("core_memory_agent"), 1024)
        self.assertEqual(agent._memory_agent_completion_cap("episodic_memory_agent"), 2048)
        self.assertEqual(agent._memory_agent_completion_cap("semantic_memory_agent"), 2048)
        self.assertIsNone(agent._memory_agent_completion_cap("meta_memory_agent"))

    def test_letta_retrieve_only(self):
        passage = SimpleNamespace(
            id="passage-1",
            text="archival item",
            metadata={"source": "test"},
        )
        agent_manager = SimpleNamespace(
            get_agent_by_id=lambda **kwargs: SimpleNamespace(
                embedding_config="embedding-config"
            ),
            list_passages=lambda **kwargs: [passage],
        )
        server = SimpleNamespace(
            user_manager=SimpleNamespace(
                get_user_or_default=lambda **kwargs: "actor"
            ),
            agent_manager=agent_manager,
        )
        agent = LettaAgent.__new__(LettaAgent)
        agent._context_id = 1
        agent.retrieve_num = 5
        agent._ensure_agent = lambda context_id: "agent-1"
        agent._client = SimpleNamespace(server=server, user_id="user-1")
        result = agent.retrieve("query")
        self.assertEqual(result.retrieved_count, 1)
        self.assertIn("archival item", result.memory_context)

    def test_letta_adapted_tool_flow_constrains_each_phase(self):
        captured = []

        class Rule:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        server = SimpleNamespace(
            update_agent=lambda **kwargs: captured.append(kwargs)
        )
        agent = LettaAgent.__new__(LettaAgent)
        agent.enforce_memory_tool_flow = True
        agent._InitToolRule = Rule
        agent._ChildToolRule = Rule
        agent._TerminalToolRule = Rule
        agent._UpdateAgent = Rule
        agent._client = SimpleNamespace(server=server, user="actor")

        agent._configure_agent_tool_flow("agent-1", "memorize")
        memorize_rules = captured[-1]["request"].tool_rules
        self.assertEqual(memorize_rules[0].tool_name, "archival_memory_insert")
        self.assertEqual(memorize_rules[1].children, ["send_message"])
        self.assertEqual(memorize_rules[2].tool_name, "send_message")

        agent._configure_agent_tool_flow("agent-1", "query")
        query_rules = captured[-1]["request"].tool_rules
        self.assertEqual(query_rules[0].tool_name, "archival_memory_search")

    def test_letta_adapted_query_uses_archival_retrieval_then_reader(self):
        agent = LettaAgent.__new__(LettaAgent)
        agent._context_id = 1
        agent.use_native_query = False
        agent.max_question_tokens = 100
        agent._truncate_to_tokens = lambda text, limit: text
        agent._ensure_agent = lambda context_id: "agent-1"
        agent.retrieve = lambda question: MemoryRetrievalResult(
            memory_context="[Memory 1]\nThe user avoids sugar.",
            retrieved_count=1,
            retrieved_memories=[{"memory": "The user avoids sugar."}],
        )
        agent._qa_llm_client = SimpleNamespace(
            chat=lambda messages: SimpleNamespace(
                content="Avoid sugary foods.", input_tokens=12, output_tokens=4
            )
        )

        result = agent.query("What should I avoid?", system_message="Be concise.")
        self.assertEqual(result.output, "Avoid sugary foods.")
        self.assertEqual(result.retrieved_count, 1)
        self.assertEqual(result.extra["method"], "letta_archival_manual_qa")

    def test_letta_adapted_memorize_inserts_vendored_archival_passages(self):
        inserted = []

        def insert_archival_memory(*, agent_id, memory):
            inserted.append((agent_id, memory))
            return [SimpleNamespace(id=f"p-{len(inserted)}", text=memory)]

        agent = LettaAgent.__new__(LettaAgent)
        agent._context_id = 1
        agent.use_native_memorize = False
        agent.memorize_chunk_tokens = 100
        agent.memorize_chunk_overlap_tokens = 0
        agent._ensure_agent = lambda context_id: "agent-1"
        agent._chunk_text_by_tokens = lambda text, size, overlap: [text]
        agent._client = SimpleNamespace(insert_archival_memory=insert_archival_memory)
        agent._memory_chunks = []
        agent._is_initialized = False

        result = agent.memorize("The user avoids sugar.")
        self.assertTrue(result.success)
        self.assertEqual(inserted, [("agent-1", "The user avoids sugar.")])
        self.assertEqual(result.action, "direct_archival_memory_insert")
        self.assertEqual(result.all_passages[0]["id"], "p-1")


if __name__ == "__main__":
    unittest.main()
