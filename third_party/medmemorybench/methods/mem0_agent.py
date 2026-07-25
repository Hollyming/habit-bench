"""Mem0 Agent - automatic memory extraction and retrieval using Mem0 library."""

import os
import time
from typing import Optional, List, Dict, Any

# Disable Mem0 telemetry (PostHog) to avoid SSL errors
os.environ["MEM0_TELEMETRY"] = "false"

from .base import BaseAgent, MemoryBuildResult, AgentResponse, MemoryRetrievalResult
from utils.llm_client import create_llm_client, format_messages, BaseLLMClient, get_usage_tracker


COMPACT_FACT_EXTRACTION_PROMPT = """You extract durable, user-specific memories from a conversation.
Keep diagnoses, symptoms, medications and doses, allergies, test results, treatment decisions,
personal constraints, preferences, plans, and dated changes. Exclude greetings, repetition,
generic medical education, and assistant advice that was not explicitly adopted by the user.
Merge duplicates into short atomic statements in the input language. Return at most 24 facts.
Output JSON only, exactly {"facts":["fact", ...]}; use {"facts":[]} when none."""


COMPACT_UPDATE_MEMORY_PROMPT = """You are a deterministic memory delta planner.
Return only mutations; never echo unchanged memories and never output event NONE.
For each new fact emit at most one action. Use ADD with id "new" only for genuinely new facts.
Use UPDATE or DELETE only with an exact ID present in current memory. If a fact is already
represented, emit no action. Keep text concise, do not invent facts, and never repeat an action.
Output JSON only as {"memory":[...]}; output {"memory":[]} when no mutation is needed."""


class Mem0Agent(BaseAgent):
    """Mem0 Agent for intelligent memory management with automatic extraction and semantic retrieval."""

    METHOD_TYPE = "agentic_memory"
    BIGMODEL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_provider: str = "openai",
        embedding_model_path: Optional[str] = None,
        retrieve_num: int = 5,
        storage_root: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_model_path = embedding_model_path
        self.retrieve_num = retrieve_num
        self.storage_root = storage_root or os.environ.get("MEDMEMORYBENCH_STATE_ROOT")
        self.compact_local_prompts = bool(kwargs.get("compact_local_prompts", False))
        self.strict_json_schema = bool(kwargs.get("strict_json_schema", False))
        self.action_generation_attempts = int(
            kwargs.get("action_generation_attempts", 3)
        )
        self.normalize_unknown_mutations = bool(
            kwargs.get("normalize_unknown_mutations", False)
        )
        # Support both 'chunk_size_tokens' (config file) and 'max_input_tokens' (legacy)
        self.max_input_tokens = int(
            kwargs.get("chunk_size_tokens")
            or kwargs.get("max_input_tokens")
            or 8000
        )
        self.max_context_tokens = int(kwargs.get("max_context_tokens", 120000))
        self.max_question_tokens = int(kwargs.get("max_question_tokens", 4096))

        # API config
        self._api_key = api_key or os.environ.get("BIGMODEL_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = (
            base_url
            or os.environ.get("BIGMODEL_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or self.BIGMODEL_BASE_URL
        )
        self._provider = provider

        # LLM client for Q&A
        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # Mem0 Memory instance
        self._memory = None
        self._user_id = None
        self._agent_start_time = time.time()
        self._qdrant_path = None  # Will be set in _init_mem0

        # Cache for embedding dimensions
        self._embedding_dims_cache: Dict[str, int] = {}

        # Mem0 is initialized lazily after set_context_id().  Initializing here
        # would permanently create the Qdrant collection with no persona ID.

    def _get_embedding_dims(self, model_name_or_path: str) -> int:
        """Get embedding dimensions for a model without loading it repeatedly.

        Uses a cache and known model dimensions to avoid expensive model loading.
        """
        # Check cache first
        if model_name_or_path in self._embedding_dims_cache:
            return self._embedding_dims_cache[model_name_or_path]

        # Known model dimensions (common models)
        known_dims = {
            # BGE models
            "bge-small": 512,
            "bge-base": 768,
            "bge-large": 1024,
            "bge-m3": 1024,
            # Other common models
            "all-MiniLM-L6": 384,
            "all-mpnet-base": 768,
            "paraphrase-multilingual": 768,
            "multi-qa-MiniLM": 384,
        }

        # Try to match known models
        model_lower = model_name_or_path.lower()
        for pattern, dims in known_dims.items():
            if pattern.lower() in model_lower:
                self._embedding_dims_cache[model_name_or_path] = dims
                return dims

        # If not a known model, we need to load it once
        # But this should only happen once per unique model
        try:
            from sentence_transformers import SentenceTransformer
            temp_model = SentenceTransformer(model_name_or_path)
            dims = temp_model.get_sentence_embedding_dimension()
            del temp_model
            self._embedding_dims_cache[model_name_or_path] = dims
            return dims
        except Exception as e:
            print(f"[Mem0] Warning: Failed to get embedding dims for {model_name_or_path}: {e}")
            # Default fallback
            return 512

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    def _split_text_into_chunks(self, text: str, max_tokens: int) -> List[str]:
        if not text.strip():
            return []
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunks.append(self._tokenizer.decode(tokens[start:end]))
            if end >= len(tokens):
                break
            start = end
        return chunks

    def _init_mem0(self) -> None:
        """Initialize Mem0 Memory instance."""
        print(f"[Mem0Agent DEBUG] _init_mem0 starting...")
        from methods.mem0.memory.main import Memory
        from pathlib import Path

        # Create unique Qdrant path for this agent instance to avoid lock conflicts
        # Use context_id if available, otherwise use a unique timestamp
        if self.storage_root:
            context_name = (
                f"persona_{self._context_id}"
                if self._context_id is not None
                else "default"
            )
            qdrant_path = str(Path(self.storage_root).expanduser() / "mem0" / context_name)
            Path(qdrant_path).parent.mkdir(parents=True, exist_ok=True)
        elif self._context_id is not None:
            qdrant_path = f"/tmp/qdrant_mem0_persona_{self._context_id}"
        else:
            import uuid
            qdrant_path = f"/tmp/qdrant_mem0_{uuid.uuid4().hex[:8]}"

        print(f"[Mem0Agent DEBUG] Qdrant path: {qdrant_path}")

        # Build Mem0 config
        mem0_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "api_key": self._api_key,
                    "openai_base_url": self._base_url,
                }
            }
        }
        if self.compact_local_prompts:
            mem0_config["custom_fact_extraction_prompt"] = COMPACT_FACT_EXTRACTION_PROMPT
            mem0_config["custom_update_memory_prompt"] = COMPACT_UPDATE_MEMORY_PROMPT
        mem0_config["strict_json_schema"] = self.strict_json_schema
        mem0_config["action_generation_attempts"] = self.action_generation_attempts
        mem0_config["normalize_unknown_mutations"] = self.normalize_unknown_mutations

        if self.embedding_provider in ("local", "huggingface"):
            model_name_or_path = self.embedding_model_path or self.embedding_model

            embedding_dims = self._get_embedding_dims(model_name_or_path)
            print(f"[Mem0] Using local Embedding model: {model_name_or_path} (dims: {embedding_dims})")

            mem0_config["embedder"] = {
                "provider": "huggingface",
                "config": {
                    "model": model_name_or_path,
                    "embedding_dims": embedding_dims,
                }
            }
            # Configure vector_store dimension with unique path
            collection_name = f"mem0_local_{self._context_id}" if self._context_id is not None else "mem0_local"
            mem0_config["vector_store"] = {
                "provider": "qdrant",
                "config": {
                    "embedding_model_dims": embedding_dims,
                    "collection_name": collection_name,
                    "path": qdrant_path,
                }
            }
        else:
            embedding_dims = 2048 if "embedding-3" in self.embedding_model else 1536
            mem0_config["embedder"] = {
                "provider": "openai",
                "config": {
                    "model": self.embedding_model,
                    "api_key": self._api_key,
                    "openai_base_url": self._base_url,
                }
            }
            # Configure vector_store with unique path
            collection_name = f"mem0_{self.embedding_model.replace('-', '_')}_{self._context_id}" if self._context_id is not None else f"mem0_{self.embedding_model.replace('-', '_')}"
            mem0_config["vector_store"] = {
                "provider": "qdrant",
                "config": {
                    "embedding_model_dims": embedding_dims,
                    "collection_name": collection_name,
                    "path": qdrant_path,
                },
            }

        # Store Qdrant path for cleanup
        self._qdrant_path = qdrant_path

        print(f"[Mem0Agent DEBUG] Creating Memory.from_config...")
        self._memory = Memory.from_config(mem0_config)
        print(f"[Mem0Agent DEBUG] Memory created successfully")

        # Inject usage tracking callback into Mem0's LLM and Embedding
        self._inject_usage_callback()

    def _ensure_memory(self) -> None:
        """Create the vendored Mem0 instance on first memory operation."""
        if self._memory is None:
            self._init_mem0()

    def _inject_usage_callback(self) -> None:
        """Inject usage callback into Mem0's internal LLM and Embedding."""
        # LLM usage callback
        def llm_usage_callback(input_tokens: int, output_tokens: int, latency: float) -> None:
            from utils.llm_client import LLMResponse
            print(f"[Mem0Agent] LLM usage: input={input_tokens}, output={output_tokens}, latency={latency:.2f}s")
            response = LLMResponse(
                content="",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=latency,
                model=self.model,
            )
            get_usage_tracker().record(response)

        # Embedding usage callback (input_tokens only, no output)
        def embedding_usage_callback(input_tokens: int, latency: float) -> None:
            from utils.llm_client import LLMResponse
            print(f"[Mem0Agent] Embedding usage: tokens={input_tokens}, latency={latency:.2f}s")
            response = LLMResponse(
                content="",
                input_tokens=input_tokens,
                output_tokens=0,
                latency=latency,
                model=self.embedding_model,
            )
            get_usage_tracker().record(response)

        # Inject LLM callback
        if self._memory and hasattr(self._memory, "llm") and self._memory.llm:
            if hasattr(self._memory.llm, "set_usage_callback"):
                self._memory.llm.set_usage_callback(llm_usage_callback)
                print(f"[Mem0Agent] LLM usage callback injected: {type(self._memory.llm).__name__}")
            else:
                print(f"[Mem0Agent] Warning: LLM {type(self._memory.llm).__name__} does not support set_usage_callback")
        else:
            print(f"[Mem0Agent] Warning: Unable to inject LLM usage callback - memory.llm not available")

        # Inject Embedding callback (only for OpenAI embeddings, local embeddings don't have token costs)
        if self._memory and hasattr(self._memory, "embedding_model") and self._memory.embedding_model:
            if hasattr(self._memory.embedding_model, "set_usage_callback"):
                # Only inject if using OpenAI embedding (has API token cost)
                if self.embedding_provider not in ("local", "huggingface"):
                    self._memory.embedding_model.set_usage_callback(embedding_usage_callback)
                    print(f"[Mem0Agent] Embedding usage callback injected: {type(self._memory.embedding_model).__name__}")
                else:
                    print(f"[Mem0Agent] Skipping embedding callback for local model (no token cost)")
            else:
                print(f"[Mem0Agent] Warning: Embedding model does not support set_usage_callback")

    def _get_user_id(self) -> str:
        """Get user ID."""
        if self._user_id:
            return self._user_id
        return f"context_{self._context_id}" if self._context_id else "default_user"

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Add text to Mem0 memory."""
        self._ensure_memory()
        user_id = self._get_user_id()

        chunks = self._split_text_into_chunks(text, self.max_input_tokens)
        memory_entries = []

        for chunk in chunks:
            memory_messages = [
                {"role": "user", "content": chunk},
                {"role": "assistant", "content": "I'll make sure to remember this information."}
            ]
            result = self._memory.add(memory_messages, user_id=user_id)
            if result and "results" in result:
                for entry in result.get("results", []):
                    memory_entries.append({
                        "event": entry.get("event", "ADD"),
                        "memory": entry.get("memory", ""),
                        "id": entry.get("id", ""),
                    })

        self._memory_chunks.append(text)
        self._is_initialized = True

        return MemoryBuildResult(
            success=True,
            method="mem0",
            action="add_to_memory",
            input_content=text,
            stored_content=text,
            memory_entries=memory_entries,
            all_passages=list(memory_entries),
            chunk_count=len(self._memory_chunks),
            extra={
                "input_chunks": len(chunks),
                "inserted_count": len(memory_entries),
            },
        )

    def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from Mem0."""
        self._ensure_memory()
        user_id = self._get_user_id()
        results = self._memory.search(query=query, user_id=user_id, limit=self.retrieve_num)

        memories = []
        if results and "results" in results:
            for entry in results["results"]:
                memories.append({
                    "memory": entry.get("memory", ""),
                    "score": entry.get("score", 0.0),
                    "id": entry.get("id", ""),
                })

        return memories

    def retrieve(self, question: str, **kwargs) -> MemoryRetrievalResult:
        """Return Mem0's native retrieval without calling the Q&A model."""
        retrieved_memories = self._retrieve(question)
        formatted_memories = [
            {
                "memory": item["memory"],
                "type": "mem0_retrieval",
                "score": item.get("score", 0.0),
                "id": item.get("id", ""),
            }
            for item in retrieved_memories
        ]
        memory_context = "\n".join(
            f"- {item['memory']}" for item in retrieved_memories if item["memory"]
        )
        return MemoryRetrievalResult(
            memory_context=memory_context,
            retrieved_count=len(formatted_memories),
            retrieved_memories=formatted_memories,
            extra={
                "method": "mem0",
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_provider,
            },
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query the agent."""
        # Retrieve relevant memories
        retrieval = self.retrieve(question)
        retrieved_memories = retrieval.retrieved_memories

        # Bound question tokens first
        full_question = f"{question}\n\nCurrent Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        full_question = self._truncate_to_tokens(full_question, self.max_question_tokens)

        base_system = system_message or ""
        reserved_tokens = self.max_tokens + 400
        available_tokens = max(self.max_context_tokens - reserved_tokens, 0)
        question_tokens = self._llm_client.count_tokens(full_question)
        base_system_tokens = self._llm_client.count_tokens(base_system) if base_system else 0
        memory_budget = max(available_tokens - question_tokens - base_system_tokens, 0)

        # Build memory string
        if retrieved_memories:
            memory_lines: List[str] = []
            used_tokens = 0
            for entry in retrieved_memories:
                line = f"- {entry['memory']}"
                line_tokens = self._llm_client.count_tokens(line)
                if used_tokens + line_tokens <= memory_budget:
                    memory_lines.append(line)
                    used_tokens += line_tokens
                else:
                    break
            memories_str = "\n".join(memory_lines)
            # Build system message with retrieved content
            memory_prompt = f"You are a helpful AI. Answer the question based on the following memories:\n{memories_str}\n"
            if system_message:
                full_system = f"{system_message}\n\n{memory_prompt}"
            else:
                full_system = memory_prompt
        else:
            full_system = system_message

        messages = format_messages(full_question, full_system)

        # Call LLM
        response = self._llm_client.chat(messages)

        # Build retrieved_memories format
        return AgentResponse(
            output=response.content,
            query_time=0.0,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=retrieved_memories,
            extra={
                "method": "mem0",
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_provider,
            }
        )

    def reset(self) -> None:
        """Reset agent and release all resources."""
        import gc
        import shutil
        import logging

        logger = logging.getLogger(__name__)
        super().reset()

        # Close Mem0 Memory using its close() method
        if self._memory:
            try:
                if hasattr(self._memory, 'close'):
                    self._memory.close()
                    logger.info("[Mem0Agent] Closed Memory instance")
                else:
                    # Fallback: manually close resources
                    if hasattr(self._memory, "llm") and self._memory.llm:
                        if hasattr(self._memory.llm, "close"):
                            self._memory.llm.close()
                    if hasattr(self._memory, "vector_store") and self._memory.vector_store:
                        if hasattr(self._memory.vector_store, "client"):
                            self._memory.vector_store.client.close()
            except Exception as e:
                logger.warning(f"[Mem0Agent] Warning: Failed to close Memory: {e}")

            self._memory = None

        # Close the Q&A LLM client
        if self._llm_client:
            try:
                if hasattr(self._llm_client, "client"):
                    client = self._llm_client.client
                    if hasattr(client, "close"):
                        client.close()
                        logger.info("[Mem0Agent] Closed Q&A LLM client")
            except Exception as e:
                logger.warning(f"[Mem0Agent] Warning: Failed to close Q&A LLM client: {e}")
            self._llm_client = None

        # Clean up Qdrant directory to release file locks
        if self._qdrant_path:
            try:
                import os
                if os.path.exists(self._qdrant_path):
                    shutil.rmtree(self._qdrant_path)
                    logger.info(f"[Mem0Agent] Cleaned up Qdrant path: {self._qdrant_path}")
            except Exception as e:
                logger.warning(f"[Mem0Agent] Warning: Failed to clean up Qdrant path: {e}")
            self._qdrant_path = None

        # Force garbage collection to release resources immediately
        gc.collect()

        self._agent_start_time = time.time()

    def set_context_id(self, context_id: int) -> None:
        """Set context ID."""
        super().set_context_id(context_id)
        # Update user_id
        self._user_id = f"context_{context_id}"
        self._agent_start_time = time.time()

    @property
    def memory_count(self) -> int:
        """Get memory count in Mem0."""
        try:
            user_id = self._get_user_id()
            all_memories = self._memory.get_all(user_id=user_id)
            return len(all_memories.get("results", []))
        except Exception:
            return len(self._memory_chunks)
