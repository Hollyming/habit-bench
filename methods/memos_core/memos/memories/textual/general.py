import json
import os

from datetime import datetime
from typing import Any

from memos.configs.memory import GeneralTextMemoryConfig
from memos.embedders.factory import ArkEmbedder, EmbedderFactory, OllamaEmbedder
from memos.llms.factory import AzureLLM, LLMFactory, OllamaLLM, OpenAILLM
from memos.log import get_logger
from memos.memories.textual.base import BaseTextMemory
from memos.memories.textual.item import TextualMemoryItem
from memos.templates.mem_reader_prompts import SIMPLE_STRUCT_MEM_READER_PROMPT
from memos.types import MessageList
from memos.vec_dbs.factory import QdrantVecDB, VecDBFactory
from memos.vec_dbs.item import VecDBItem


logger = get_logger(__name__)


class MemoryExtractionError(ValueError):
    """An extraction was returned, but it is not a usable memory batch."""


def _extraction_response_format() -> dict[str, Any]:
    """Constrain format-only retries to the existing official reader contract."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "memory_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "memory list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string", "minLength": 1},
                                "value": {"type": "string", "minLength": 1},
                                "memory_type": {
                                    "type": "string",
                                    "enum": ["LongTermMemory", "UserMemory"],
                                },
                                "tags": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["key", "value", "memory_type", "tags"],
                            "additionalProperties": False,
                        },
                    },
                    "summary": {"type": "string"},
                },
                "required": ["memory list", "summary"],
                "additionalProperties": False,
            },
        },
    }


def _validate_memory_list(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("memory list"), list):
        raise MemoryExtractionError("Expected an object with a 'memory list' array")
    memories = payload["memory list"]
    # Validate the entire batch before constructing items or writing vectors.
    # A malformed row is not an empty extraction and must not be silently lost.
    for index, memory in enumerate(memories):
        if not isinstance(memory, dict):
            raise MemoryExtractionError(f"memory list[{index}] must be an object")
        for field in ("key", "value"):
            if not isinstance(memory.get(field), str) or not memory[field].strip():
                raise MemoryExtractionError(
                    f"memory list[{index}].{field} must be a non-empty string"
                )
    return memories


def _normalize_tags(tags: Any) -> list[str]:
    """Keep only schema-valid tags from an untrusted LLM response."""
    if not isinstance(tags, list):
        return []
    return [tag for tag in tags if isinstance(tag, str)]


class GeneralTextMemory(BaseTextMemory):
    """General textual memory implementation for storing and retrieving memories."""

    def __init__(self, config: GeneralTextMemoryConfig):
        """Initialize memory with the given configuration."""
        # Set mode from class default or override if needed
        self.mode = getattr(self.__class__, "mode", "sync")
        self.config: GeneralTextMemoryConfig = config
        self.extractor_llm: OpenAILLM | OllamaLLM | AzureLLM = LLMFactory.from_config(
            config.extractor_llm
        )
        self.vector_db: QdrantVecDB = VecDBFactory.from_config(config.vector_db)
        self.embedder: OllamaEmbedder | ArkEmbedder = EmbedderFactory.from_config(config.embedder)

    def extract(self, messages: MessageList) -> list[TextualMemoryItem]:
        """Extract memories based on the messages.

        Args:
            messages: List of message dictionaries to extract memories from.

        Returns:
            List of TextualMemoryItem objects representing the extracted memories.
        """

        str_messages = "\n".join(
            [message["role"] + ":" + message["content"] for message in messages]
        )

        prompt = SIMPLE_STRUCT_MEM_READER_PROMPT.replace("${conversation}", str_messages).replace(
            "${custom_tags_prompt}", ""
        )
        failures = []
        for attempt in range(1, 4):
            correction = (
                "\n\nThe previous extraction did not satisfy the output format. "
                "Re-extract from the same conversation and return the complete JSON object. "
                "Every memory list entry requires non-empty string key and value, "
                "memory_type, and a tags array. Include summary. "
                "Use an empty memory list only when the conversation contains no memories."
                if attempt > 1 else ""
            )
            request_kwargs = {}
            if attempt > 1 and isinstance(self.extractor_llm, (OpenAILLM, AzureLLM)):
                request_kwargs["response_format"] = _extraction_response_format()
            try:
                response_text = self.extractor_llm.generate(
                    [{"role": "user", "content": prompt + correction}], **request_kwargs
                )
                memories = _validate_memory_list(self.parse_json_result(response_text))
                break
            except ValueError as exc:
                # Both JSON syntax and schema errors are retryable. Never retry
                # add(): it may already have committed part of a vector batch.
                failures.append(f"{type(exc).__name__}: {exc}")
                logger.warning("Memory extraction format failure, attempt %s/3: %s", attempt, exc)
                if attempt == 3:
                    raise MemoryExtractionError(
                        "Memory extraction failed after 3 attempts: " + " | ".join(failures)
                    ) from exc

        extracted_memories = [
            TextualMemoryItem(
                memory=memory_dict["value"],
                metadata={
                    "key": memory_dict["key"],
                    "source": "conversation",
                    "tags": _normalize_tags(memory_dict.get("tags")),
                    "updated_at": datetime.now().isoformat(),
                },
            )
            for memory_dict in memories
        ]

        return extracted_memories

    def add(self, memories: list[TextualMemoryItem | dict[str, Any]]) -> None:
        """Add memories.

        Args:
            memories: List of TextualMemoryItem objects or dictionaries to add.
        """
        memory_items = [TextualMemoryItem(**m) if isinstance(m, dict) else m for m in memories]

        # Memory encode
        embed_memories = self.embedder.embed([m.memory for m in memory_items])

        # Create vector db items
        vec_db_items = []
        for item, emb in zip(memory_items, embed_memories, strict=True):
            vec_db_items.append(
                VecDBItem(
                    id=item.id,
                    payload=item.model_dump(),
                    vector=emb,
                )
            )

        # Add to vector db
        self.vector_db.add(vec_db_items)

    def update(self, memory_id: str, new_memory: TextualMemoryItem | dict[str, Any]) -> None:
        """Update a memory by memory_id."""
        memory_item = (
            TextualMemoryItem(**new_memory) if isinstance(new_memory, dict) else new_memory
        )
        memory_item.id = memory_id

        vec_db_item = VecDBItem(
            id=memory_item.id,
            payload=memory_item.model_dump(),
            vector=self._embed_one_sentence(memory_item.memory),
        )

        self.vector_db.update(memory_id, vec_db_item)

    def search(self, query: str, top_k: int, info=None, **kwargs) -> list[TextualMemoryItem]:
        """Search for memories based on a query.
        Args:
            query (str): The query to search for.
            top_k (int): The number of top results to return.
        Returns:
            list[TextualMemoryItem]: List of matching memories.
        """
        query_vector = self._embed_one_sentence(query)
        search_results = self.vector_db.search(query_vector, top_k)
        search_results = sorted(  # make higher score first
            search_results, key=lambda x: x.score, reverse=True
        )
        result_memories = [
            TextualMemoryItem(**search_item.payload) for search_item in search_results
        ]
        return result_memories

    def get(self, memory_id: str, user_name: str | None = None) -> TextualMemoryItem:
        """Get a memory by its ID."""
        result = self.vector_db.get_by_id(memory_id)
        if result is None:
            raise ValueError(f"Memory with ID {memory_id} not found")
        return TextualMemoryItem(**result.payload)

    def get_by_ids(self, memory_ids: list[str]) -> list[TextualMemoryItem]:
        """Get memories by their IDs.
        Args:
            memory_ids (list[str]): List of memory IDs to retrieve.
        Returns:
            list[TextualMemoryItem]: List of memories with the specified IDs.
        """
        db_items = self.vector_db.get_by_ids(memory_ids)
        memories = [TextualMemoryItem(**db_item.payload) for db_item in db_items]
        return memories

    def get_all(self) -> list[TextualMemoryItem]:
        """Get all memories.
        Returns:
            list[TextualMemoryItem]: List of all memories.
        """
        all_items = self.vector_db.get_all()
        all_memories = [TextualMemoryItem(**memo.payload) for memo in all_items]
        return all_memories

    def delete(self, memory_ids: list[str]) -> None:
        """Delete a memory."""
        self.vector_db.delete(memory_ids)

    def delete_all(self) -> None:
        """Delete all memories."""
        self.vector_db.delete_collection(self.vector_db.config.collection_name)
        self.vector_db.create_collection()

    def load(self, dir: str) -> None:
        try:
            memory_file = os.path.join(dir, self.config.memory_filename)

            if not os.path.exists(memory_file):
                logger.warning(f"Memory file not found: {memory_file}")
                return

            with open(memory_file, encoding="utf-8") as f:
                memories = json.load(f)

            vec_db_items = [VecDBItem.from_dict(m) for m in memories]
            self.vector_db.add(vec_db_items)
            logger.info(f"Loaded {len(memories)} memories from {memory_file}")

        except FileNotFoundError:
            logger.error(f"Memory file not found in directory: {dir}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from memory file: {e}")
        except Exception as e:
            logger.error(f"An error occurred while loading memories: {e}")

    def dump(self, dir: str) -> None:
        """Dump memories to os.path.join(dir, self.config.memory_filename)"""
        try:
            all_vec_db_items = self.vector_db.get_all()
            json_memories = [memory.to_dict() for memory in all_vec_db_items]

            os.makedirs(dir, exist_ok=True)
            memory_file = os.path.join(dir, self.config.memory_filename)
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(json_memories, f, indent=4, ensure_ascii=False)

            logger.info(f"Dumped {len(all_vec_db_items)} memories to {memory_file}")

        except Exception as e:
            logger.error(f"An error occurred while dumping memories: {e}")
            raise

    def drop(
        self,
    ) -> None:
        pass

    def _embed_one_sentence(self, sentence: str) -> list[float]:
        """Embed a single sentence."""
        return self.embedder.embed([sentence])[0]

    def parse_json_result(self, response_text):
        if not isinstance(response_text, str) or not response_text.strip():
            raise MemoryExtractionError("LLM returned no extraction content")
        json_start = response_text.find("{")
        if json_start == -1:
            raise MemoryExtractionError("LLM response contains no JSON object")
        text = response_text[json_start:].strip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
        # Do not repair a missing closing brace: a truncated generation must
        # be re-extracted, not accepted as a successful partial memory build.
        return json.loads(text)
