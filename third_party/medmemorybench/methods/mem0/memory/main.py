import concurrent
import hashlib
import json
import logging
import os
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict

import pytz
from pydantic import ValidationError

from methods.mem0.configs.base import MemoryConfig, MemoryItem
from methods.mem0.configs.enums import MemoryType
from methods.mem0.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_update_memory_messages,
)
from methods.mem0.memory.base import MemoryBase
from methods.mem0.memory.setup import setup_config
from methods.mem0.memory.storage import SQLiteManager
from methods.mem0.memory.telemetry import capture_event
from methods.mem0.memory.utils import (
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    remove_code_blocks,
)
from methods.mem0.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory

# Setup user config
setup_config()

logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.strict_json_schema = self.config.strict_json_schema
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt

        print(f"[Memory DEBUG] Creating embedding_model...")
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        print(f"[Memory DEBUG] Embedding model created")

        print(f"[Memory DEBUG] Creating vector_store...")
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        print(f"[Memory DEBUG] Vector store created")

        print(f"[Memory DEBUG] Creating llm...")
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        print(f"[Memory DEBUG] LLM created")
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self._max_workers = int(os.environ.get("MEM0_MAX_WORKERS", "2"))
        # Local instruction-following models occasionally return syntactically
        # valid JSON with an invalid action schema (for example, a missing
        # event or two deletes for the same memory).  Validate the complete
        # batch before mutating the vector store and give the model bounded,
        # explicit feedback instead of leaving a partially applied batch.
        self._action_generation_attempts = max(
            1,
            min(
                5,
                int(
                    os.environ.get(
                        "MEM0_ACTION_GENERATION_ATTEMPTS",
                        str(self.config.action_generation_attempts),
                    )
                ),
            ),
        )
        self._normalize_unknown_mutations = bool(
            self.config.normalize_unknown_mutations
        )

        self.enable_graph = False

        if self.config.graph_store.config:
            from methods.mem0.memory.graph_memory import MemoryGraph

            self.graph = MemoryGraph(self.config)
            self.enable_graph = True

        capture_event("mem0.init", self)

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def add(
        self,
        messages,
        user_id=None,
        agent_id=None,
        run_id=None,
        metadata=None,
        filters=None,
        infer=True,
        memory_type=None,
        prompt=None,
    ):
        """
        Create a new memory.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            filters (dict, optional): Filters to apply to the search. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
            result: dict of affected events with each dict has the following key:
              'memories': affected memories
              'graph': affected graph memories

              'memories' and 'graph' is a dict, each with following subkeys:
                'add': added memory
                'update': updated memory
                'delete': deleted memory


        """
        if metadata is None:
            metadata = {}

        filters = filters or {}
        if user_id:
            filters["user_id"] = metadata["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = metadata["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = metadata["run_id"] = run_id

        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=metadata, prompt=prompt)
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        # Note: Rate limiting is now handled by retry mechanisms in LLM and embedding modules

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, metadata, filters, infer)
            future2 = executor.submit(self._add_to_graph, messages, filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        if self.api_version == "v1.0":
            warnings.warn(
                "The current add API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return vector_store_result

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message in messages:
                if message["role"] != "system":
                    message_embeddings = self.embedding_model.embed(message["content"], "add")
                    memory_id = self._create_memory(message["content"], message_embeddings, metadata)
                    returned_memories.append({"id": memory_id, "memory": message["content"], "event": "ADD"})
            return returned_memories

        parsed_messages = parse_messages(messages)

        # DEBUG: 打印原始 messages 结构
        logging.debug(f"[DEBUG] === Raw messages structure ===")
        logging.debug(f"[DEBUG] Number of messages: {len(messages)}")

        if self.custom_fact_extraction_prompt:
            system_prompt = self.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self._fact_response_format(),
        )

        try:
            if isinstance(response, str):
                response = remove_code_blocks(response)
                parsed = self._parse_json_response(response)
                new_retrieved_facts = parsed.get("facts", [])
            else:
                new_retrieved_facts = []
        except Exception as e:
            logging.error(f"Error in new_retrieved_facts: {e}")
            raise RuntimeError("Mem0 fact extraction returned invalid JSON") from e

        if not isinstance(new_retrieved_facts, list):
            logging.warning(f"new_retrieved_facts is not a list: {type(new_retrieved_facts)}")
            new_retrieved_facts = []

        # 过滤非字符串元素，确保 facts 都是有效的字符串
        original_count = len(new_retrieved_facts)
        new_retrieved_facts = [
            fact for fact in new_retrieved_facts
            if isinstance(fact, str) and fact.strip()
        ]
        if len(new_retrieved_facts) < original_count:
            logging.warning(f"Filtered out {original_count - len(new_retrieved_facts)} non-string or empty facts")

        logging.info(f"Extracted {len(new_retrieved_facts)} facts from input")

        if not new_retrieved_facts:
            return []

        # Use batch embedding for efficiency
        retrieved_old_memory = []
        new_message_embeddings = {}

        # Check if batch embedding is available
        if hasattr(self.embedding_model, 'embed_batch') and len(new_retrieved_facts) > 1:
            try:
                all_embeddings = self.embedding_model.embed_batch(new_retrieved_facts, "add")
                for fact, emb in zip(new_retrieved_facts, all_embeddings):
                    new_message_embeddings[fact] = emb
            except Exception as e:
                logging.warning(f"Batch embedding failed, falling back to single: {e}")
                for new_mem in new_retrieved_facts:
                    new_message_embeddings[new_mem] = self.embedding_model.embed(new_mem, "add")
        else:
            for new_mem in new_retrieved_facts:
                new_message_embeddings[new_mem] = self.embedding_model.embed(new_mem, "add")

        # Search for existing memories
        for new_mem in new_retrieved_facts:
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=new_message_embeddings[new_mem],
                limit=5,
                filters=filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload["data"]})

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logging.info(f"Total existing memories: {len(retrieved_old_memory)}")

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        function_calling_prompt = get_update_memory_messages(
            retrieved_old_memory, new_retrieved_facts, self.custom_update_memory_prompt
        )

        validated_actions = None
        last_action_error = None
        for attempt in range(1, self._action_generation_attempts + 1):
            action_prompt = function_calling_prompt
            if last_action_error is not None:
                valid_existing_ids = ", ".join(sorted(temp_uuid_mapping)) or "(none)"
                action_prompt += (
                    "\n\nYour previous action batch was invalid: "
                    f"{last_action_error}. Return a corrected COMPLETE JSON batch. "
                    "Every item must have exactly one supported event, and each existing "
                    "memory ID may be mutated at most once. "
                    f"The only valid IDs for UPDATE or DELETE are: {valid_existing_ids}. "
                    "If no listed ID is the intended target, emit ADD instead."
                )
            try:
                response = self.llm.generate_response(
                    messages=[{"role": "user", "content": action_prompt}],
                    response_format=self._action_response_format(),
                )
                if isinstance(response, str):
                    response = remove_code_blocks(response)
                    response = self._parse_json_response(response)
                validated_actions = self._validate_memory_actions(
                    response,
                    temp_uuid_mapping,
                    normalize_unknown_mutations=(
                        self._normalize_unknown_mutations
                        and attempt == self._action_generation_attempts
                    ),
                )
                if attempt > 1:
                    logging.info(
                        "Mem0 action batch passed validation on attempt %d/%d",
                        attempt,
                        self._action_generation_attempts,
                    )
                break
            except Exception as e:
                last_action_error = str(e)
                if attempt >= self._action_generation_attempts:
                    logging.error(
                        "Mem0 action batch remained invalid after %d attempts: %s",
                        attempt,
                        e,
                    )
                    raise RuntimeError(
                        "Mem0 memory actions failed bounded schema validation"
                    ) from e
                logging.warning(
                    "Mem0 action batch invalid on attempt %d/%d: %s; retrying",
                    attempt,
                    self._action_generation_attempts,
                    e,
                )

        assert validated_actions is not None

        returned_memories = []
        try:
            for resp in validated_actions:
                logging.debug(resp)
                try:
                    if resp.get("event") == "ADD":
                        memory_id = self._create_memory(
                            data=resp.get("text"),
                            existing_embeddings=new_message_embeddings,
                            metadata=metadata,
                        )
                        returned_memories.append(
                            {
                                "id": memory_id,
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                            }
                        )
                    elif resp.get("event") == "UPDATE":
                        resp_id = str(resp.get("id", ""))
                        self._update_memory(
                            memory_id=temp_uuid_mapping[resp_id],
                            data=resp.get("text"),
                            existing_embeddings=new_message_embeddings,
                            metadata=metadata,
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp_id],
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif resp.get("event") == "DELETE":
                        resp_id = str(resp.get("id", ""))
                        self._delete_memory(memory_id=temp_uuid_mapping[resp_id])
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp_id],
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                            }
                        )
                    elif resp.get("event") == "NONE":
                        logging.debug("NOOP for Memory.")
                    else:
                        raise ValueError(f"Unsupported memory event: {resp.get('event')!r}")
                except Exception as e:
                    logging.error(f"Error processing memory action: {e}")
                    raise RuntimeError("Mem0 failed to apply a generated memory action") from e
        except Exception as e:
            logging.error(f"Error in new_memories_with_actions: {e}")
            raise RuntimeError("Mem0 failed while applying generated memory actions") from e

        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": list(filters.keys())},
        )

        return returned_memories

    @staticmethod
    def _json_schema_format(name, schema):
        return {
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        }

    def _fact_response_format(self):
        if not self.strict_json_schema:
            return {"type": "json_object"}
        return self._json_schema_format(
            "mem0_facts",
            {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 512,
                        },
                        "maxItems": 24,
                    }
                },
                "required": ["facts"],
                "additionalProperties": False,
            },
        )

    def _action_response_format(self):
        if not self.strict_json_schema:
            return {"type": "json_object"}
        return self._json_schema_format(
            "mem0_memory_actions",
            {
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 512,
                                },
                                "event": {
                                    "type": "string",
                                    "enum": ["ADD", "UPDATE", "DELETE"],
                                },
                            },
                            "required": ["id", "text", "event"],
                            "additionalProperties": False,
                        },
                        "maxItems": 24,
                    }
                },
                "required": ["memory"],
                "additionalProperties": False,
            },
        )

    @staticmethod
    def _validate_memory_actions(
        payload,
        temp_uuid_mapping,
        *,
        normalize_unknown_mutations=False,
    ):
        """Validate an entire generated action batch before applying mutations."""
        if not isinstance(payload, dict):
            raise TypeError(
                f"Action response must be a JSON object, got {type(payload).__name__}"
            )
        if "memory" not in payload or not isinstance(payload["memory"], list):
            raise TypeError("Action response must contain a list field named 'memory'")

        validated = []
        existing_action_positions = {}
        supported_events = {"ADD", "UPDATE", "DELETE", "NONE"}
        for index, raw_action in enumerate(payload["memory"]):
            if not isinstance(raw_action, dict):
                raise TypeError(
                    f"Memory action {index} must be an object, got "
                    f"{type(raw_action).__name__}"
                )
            event = raw_action.get("event")
            if event not in supported_events:
                raise ValueError(
                    f"Memory action {index} has unsupported event {event!r}"
                )
            if event == "NONE":
                continue

            text = raw_action.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Memory action {index} ({event}) requires non-empty text"
                )

            action = dict(raw_action)
            action["text"] = text.strip()
            if event in {"UPDATE", "DELETE"}:
                action_id = str(raw_action.get("id", ""))
                if action_id not in temp_uuid_mapping:
                    if normalize_unknown_mutations and event == "UPDATE":
                        logging.warning(
                            "Mem0 normalized UPDATE for unknown temporary ID %r "
                            "to ADD on the final bounded retry",
                            action_id,
                        )
                        action["event"] = "ADD"
                        action.pop("id", None)
                        validated.append(action)
                        continue
                    if normalize_unknown_mutations and event == "DELETE":
                        logging.warning(
                            "Mem0 discarded DELETE for unknown temporary ID %r "
                            "on the final bounded retry",
                            action_id,
                        )
                        continue
                    raise ValueError(
                        f"Memory action {index} ({event}) references unknown ID "
                        f"{action_id!r}"
                    )
                action["id"] = action_id
                if action_id in existing_action_positions:
                    previous_index = existing_action_positions[action_id]
                    previous = validated[previous_index]
                    logging.warning(
                        "Mem0 normalized repeated mutation for existing ID %r: "
                        "%s -> %s (last mutation wins)",
                        action_id,
                        previous["event"],
                        event,
                    )
                    validated[previous_index] = action
                    continue
                existing_action_positions[action_id] = len(validated)
            validated.append(action)

        return validated

    def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        filters = {key: memory.payload[key] for key in ["user_id", "agent_id", "run_id"] if memory.payload.get(key)}

        # Prepare base memory item
        memory_item = MemoryItem(
            id=memory.id,
            memory=memory.payload["data"],
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump(exclude={"score"})

        # Add metadata if there are additional keys
        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }
        additional_metadata = {k: v for k, v in memory.payload.items() if k not in excluded_keys}
        if additional_metadata:
            memory_item["metadata"] = additional_metadata

        result = {**memory_item, **filters}

        return result

    def get_all(self, user_id=None, agent_id=None, run_id=None, limit=100):
        """
        List all memories.

        Returns:
            list: List of all memories.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        capture_event("mem0.get_all", self, {"limit": limit, "keys": list(filters.keys())})

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, filters, limit)
            future_graph_entities = executor.submit(self.graph.get_all, filters, limit) if self.enable_graph else None

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories, "relations": graph_entities}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return all_memories
        else:
            return {"results": all_memories}

    def _get_all_from_vector_store(self, filters, limit):
        memories = self.vector_store.list(filters=filters, limit=limit)

        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }
        all_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                ).model_dump(exclude={"score"}),
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories[0]
        ]
        return all_memories

    def search(self, query, user_id=None, agent_id=None, run_id=None, limit=100, filters=None):
        """
        Search for memories.

        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: List of search results.
        """
        filters = filters or {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        capture_event(
            "mem0.search",
            self,
            {"limit": limit, "version": self.api_version, "keys": list(filters.keys())},
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_memories = executor.submit(self._search_vector_store, query, filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.search, query, filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            original_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return original_memories
        else:
            return {"results": original_memories}

    def _search_vector_store(self, query, filters, limit):
        embeddings = self.embedding_model.embed(query, "search")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)

        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }

        original_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                    score=mem.score,
                ).model_dump(),
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories
        ]

        return original_memories

    def update(self, memory_id, data):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (dict): Data to update the memory with.

        Returns:
            dict: Updated memory.
        """
        capture_event("mem0.update", self, {"memory_id": memory_id})

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id})
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        capture_event("mem0.delete_all", self, {"keys": list(filters.keys())})
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id})
        return self.db.get_history(memory_id)

    def _parse_json_response(self, response: str) -> dict:
        """
        解析 LLM 返回的 JSON 响应，支持多种格式容错。

        Args:
            response: LLM 返回的字符串

        Returns:
            dict: 解析后的字典；无法解析时显式失败，避免静默丢失 memory。
        """
        import re

        if not response or not isinstance(response, str):
            return {"memory": []}

        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 对象（从第一个 { 到最后一个 }）
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # 尝试修复常见的 JSON 格式问题
        try:
            # 移除可能的尾部逗号
            fixed = re.sub(r',\s*}', '}', response)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        raise ValueError(
            "Failed to parse Mem0 JSON response; preview=" + repr(response[:200])
        )

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logging.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        self.db.add_history(memory_id, None, data, "ADD", created_at=metadata["created_at"])
        capture_event("mem0._create_memory", self, {"memory_id": memory_id})
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        # Generate embeddings for the summary
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        # Create the memory
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id})

        # Return results in the same format as add()
        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data")

        new_metadata = metadata or {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        if "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")
        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")
        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
        )
        capture_event("mem0._update_memory", self, {"memory_id": memory_id})
        return memory_id

    def _delete_memory(self, memory_id):
        logging.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload["data"]
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(memory_id, prev_value, None, "DELETE", is_deleted=1)
        capture_event("mem0._delete_memory", self, {"memory_id": memory_id})
        return memory_id

    def reset(self):
        """
        Reset the memory store.
        """
        logger.warning("Resetting all memories")
        self.vector_store.delete_col()
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.db.reset()
        capture_event("mem0.reset", self)

    def close(self):
        """Close all resources and release connections."""
        logger.info("Closing Memory instance and releasing resources...")

        # Close LLM client
        if hasattr(self, 'llm') and self.llm:
            if hasattr(self.llm, 'close'):
                try:
                    self.llm.close()
                    logger.debug("Closed LLM client")
                except Exception as e:
                    logger.warning(f"Failed to close LLM client: {e}")
            self.llm = None

        # Close vector store client
        if hasattr(self, 'vector_store') and self.vector_store:
            if hasattr(self.vector_store, 'client') and self.vector_store.client:
                if hasattr(self.vector_store.client, 'close'):
                    try:
                        self.vector_store.client.close()
                        logger.debug("Closed vector store client")
                    except Exception as e:
                        logger.warning(f"Failed to close vector store client: {e}")
            self.vector_store = None

        # Close database connection using SQLiteManager.close()
        if hasattr(self, 'db') and self.db:
            if hasattr(self.db, 'close'):
                try:
                    self.db.close()
                    logger.debug("Closed database connection")
                except Exception as e:
                    logger.warning(f"Failed to close database connection: {e}")
            self.db = None

        # Clear embedding model reference (it holds significant memory)
        if hasattr(self, 'embedding_model'):
            self.embedding_model = None

        logger.info("Memory instance closed")

    def __del__(self):
        """Destructor to ensure resources are released."""
        try:
            self.close()
        except Exception:
            pass

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
