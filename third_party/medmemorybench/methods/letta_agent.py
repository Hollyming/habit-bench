"""Letta (MemGPT) agent adapter for MedMemoryBench."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import BaseAgent, MemoryBuildResult, AgentResponse, MemoryRetrievalResult
from utils.llm_client import (
    create_llm_client,
    format_messages,
    get_usage_tracker,
    LLMResponse,
)

logger = logging.getLogger(__name__)


class LettaAgent(BaseAgent):
    """Adapter that uses Letta's official Agent API for memory operations.

    This implementation uses Letta's native agent-driven memory system:
    - memorize(): sends message chunks to agent which decides how to store memories
    - query(): sends question to agent which retrieves and responds using memories

    Agent lifecycle:
    - One agent per context_id (persona)
    - Agent is reused within the same persona for memory accumulation
    - Agent is deleted and recreated when switching to a new persona
    """

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
        retrieve_num: int = 5,
        context_window: int = 128000,
        embedding_model: str = "embedding-3",
        embedding_model_path: Optional[str] = None,
        embedding_dim: int = 2048,
        embedding_provider: str = "openai",
        memory_persona: str = "I am an assistant helping with medical QA while preserving long-term memory.",
        memory_human: str = "The user is a patient in a longitudinal medical dialogue setting.",
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.provider = provider
        self.api_key = api_key or os.environ.get("BIGMODEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("BIGMODEL_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or self.BIGMODEL_BASE_URL
        )
        self.retrieve_num = retrieve_num
        self.context_window = context_window
        self.embedding_model = embedding_model
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.embedding_provider = embedding_provider
        self.embedding_chunk_size = int(kwargs.get("embedding_chunk_size", 300))
        self.enforce_memory_tool_flow = bool(kwargs.get("enforce_memory_tool_flow", False))
        self.message_buffer_autoclear = bool(kwargs.get("message_buffer_autoclear", False))
        self.use_native_query = bool(kwargs.get("use_native_query", True))
        self.use_native_memorize = bool(kwargs.get("use_native_memorize", True))

        # Adapted local-model runs can retain Letta's native archival storage and
        # retrieval while using the benchmark's reader for the final answer.  The
        # paper/API configuration keeps Letta's native answer turn by default.
        self._qa_llm_client = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # Token limits for chunking and truncation
        self.max_input_tokens = int(kwargs.get("max_input_tokens", 8000))
        self.max_question_tokens = int(kwargs.get("max_question_tokens", 4096))
        self.max_context_tokens = int(kwargs.get("max_context_tokens", context_window))
        # Chunk size for splitting long memorize inputs
        self.memorize_chunk_tokens = int(kwargs.get("memorize_chunk_tokens", 6000))
        self.memorize_chunk_overlap_tokens = int(kwargs.get("memorize_chunk_overlap_tokens", 200))

        self.memory_persona = memory_persona
        self.memory_human = memory_human
        # Use per-process persistence_root to avoid concurrent SQLite writes
        default_root = f".tmp/letta_runtime_{os.getpid()}"
        self.persistence_root = Path(kwargs.get("persistence_root", default_root)).resolve()

        # Agent management: one agent per context_id
        self._agent_ids: Dict[int, str] = {}
        self._client = None

        self._apply_openai_compatible_env()
        self._load_letta_package()
        self._init_client()

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max_tokens."""
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    def _chunk_text_by_tokens(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into chunks by token count with overlap.

        Args:
            text: Text to split
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks

        Returns:
            List of text chunks
        """
        if not text or chunk_size <= 0:
            return []

        tokens = self._tokenizer.encode(text)
        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start position with overlap
            start = end - overlap if end < len(tokens) else end

        return chunks

    def _apply_openai_compatible_env(self) -> None:
        """Set OpenAI-compatible env vars so vendored Letta can pick up credentials."""
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url
        self.persistence_root.mkdir(parents=True, exist_ok=True)
        os.environ["LETTA_DIR"] = str(self.persistence_root)

    def _load_letta_package(self) -> None:
        """Load vendored Letta package as top-level module name `letta`.

        This method forces the use of the vendored Letta package even if
        a pip-installed version exists, to avoid version conflicts.
        """
        letta_root = Path(__file__).resolve().parent / "letta"
        init_file = letta_root / "__init__.py"
        if not init_file.exists():
            raise ImportError("Letta source not found at methods/letta")

        # Remove any existing letta modules from sys.modules to avoid conflicts
        modules_to_remove = [key for key in sys.modules.keys() if key == "letta" or key.startswith("letta.")]
        for mod_name in modules_to_remove:
            del sys.modules[mod_name]

        # Load vendored letta package
        spec = importlib.util.spec_from_file_location(
            "letta",
            str(init_file),
            submodule_search_locations=[str(letta_root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError("Failed to create import spec for vendored Letta")

        module = importlib.util.module_from_spec(spec)
        sys.modules["letta"] = module
        spec.loader.exec_module(module)

    def _init_client(self) -> None:
        """Initialize Letta client with proper settings."""
        letta_settings = importlib.import_module("letta.settings")
        if self.api_key:
            letta_settings.model_settings.openai_api_key = self.api_key
        if self.base_url:
            letta_settings.model_settings.openai_api_base = self.base_url

        letta_client = importlib.import_module("letta.client.client")
        self._LLMConfig = importlib.import_module("letta.schemas.llm_config").LLMConfig
        self._EmbeddingConfig = importlib.import_module("letta.schemas.embedding_config").EmbeddingConfig
        self._ChatMemory = importlib.import_module("letta.schemas.memory").ChatMemory
        tool_rule_module = importlib.import_module("letta.schemas.tool_rule")
        self._InitToolRule = tool_rule_module.InitToolRule
        self._ChildToolRule = tool_rule_module.ChildToolRule
        self._TerminalToolRule = tool_rule_module.TerminalToolRule
        self._UpdateAgent = importlib.import_module("letta.schemas.agent").UpdateAgent
        self._client = letta_client.create_client(base_url=None, token=None)

    def _configure_agent_tool_flow(self, agent_id: str, phase: str) -> None:
        """Constrain local-model Letta runs to one memory operation and one reply.

        Qwen can otherwise repeatedly call ``conversation_search`` when ingesting a
        medical dialogue.  Besides doing no archival write, that loop eventually
        emits an overlong malformed tool-call list.  The optional adapted-run rule
        keeps all storage and search inside Letta's native tools while making the
        intended phase explicit.  Paper/API configs leave this disabled.
        """
        if not self.enforce_memory_tool_flow:
            return
        phase_tools = {
            "memorize": "archival_memory_insert",
            "query": "archival_memory_search",
        }
        try:
            memory_tool = phase_tools[phase]
        except KeyError as exc:
            raise ValueError(f"Unsupported Letta tool-flow phase: {phase}") from exc

        rules = [
            self._InitToolRule(tool_name=memory_tool),
            self._ChildToolRule(tool_name=memory_tool, children=["send_message"]),
            self._TerminalToolRule(tool_name="send_message"),
        ]
        self._client.server.update_agent(
            agent_id=agent_id,
            request=self._UpdateAgent(tool_rules=rules),
            actor=self._client.user,
        )

    def _get_context_id(self) -> int:
        """Get current context ID, defaulting to 0."""
        return self._context_id if self._context_id is not None else 0

    def _record_usage_to_tracker(self, parsed_response: Dict[str, Any], phase: str, latency: float = 0.0) -> None:
        """Record Letta's token usage to the global LLMUsageTracker.

        Args:
            parsed_response: Parsed response dictionary containing 'usage' field
            phase: Either 'memorize' or 'query'
            latency: Time cost for the API call
        """
        usage = parsed_response.get("usage", {})
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            if prompt_tokens > 0 or completion_tokens > 0:
                llm_response = LLMResponse(
                    content="",
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    latency=latency,
                    model=self.model,
                )
                get_usage_tracker().set_phase(phase)
                get_usage_tracker().record(llm_response)
                logger.debug(f"Recorded Letta usage [{phase}]: prompt={prompt_tokens}, completion={completion_tokens}")

    def _build_llm_config(self):
        """Build LLMConfig for Letta agent creation."""
        endpoint = self.base_url or self.BIGMODEL_BASE_URL
        return self._LLMConfig(
            model=self.model,
            model_endpoint_type=self.provider,
            model_endpoint=endpoint,
            model_wrapper=None,
            context_window=self.context_window,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _build_embedding_config(self):
        """Build EmbeddingConfig for Letta agent creation."""
        if self.embedding_provider == "local" and self.embedding_model_path:
            return self._EmbeddingConfig(
                embedding_endpoint_type="local",
                embedding_endpoint=self.embedding_model_path,
                embedding_model=self.embedding_model,
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=self.embedding_chunk_size,
            )
        else:
            endpoint = self.base_url or self.BIGMODEL_BASE_URL
            return self._EmbeddingConfig(
                embedding_endpoint_type="openai",
                embedding_endpoint=endpoint,
                embedding_model=self.embedding_model,
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=self.embedding_chunk_size,
            )

    def _create_agent(self, context_id: int) -> str:
        """Create a new Letta agent for the given context_id.

        Args:
            context_id: The context/persona ID

        Returns:
            The agent ID string
        """
        name = f"medmemorybench_letta_ctx_{context_id}_{int(time.time())}"
        memory = self._ChatMemory(persona=self.memory_persona, human=self.memory_human)

        agent_metadata = {
            "retrieve_num": self.retrieve_num,
            "embedding_chunk_size": self.embedding_chunk_size,
        }

        state = self._client.create_agent(
            name=name,
            llm_config=self._build_llm_config(),
            embedding_config=self._build_embedding_config(),
            memory=memory,
            metadata=agent_metadata,
            message_buffer_autoclear=self.message_buffer_autoclear,
        )
        agent_id = str(state.id)
        self._agent_ids[context_id] = agent_id
        logger.info(f"Created Letta agent for context {context_id}: {agent_id}")
        return agent_id

    def _delete_agent(self, context_id: int) -> None:
        """Delete the Letta agent for the given context_id.

        Args:
            context_id: The context/persona ID
        """
        agent_id = self._agent_ids.get(context_id)
        if agent_id:
            try:
                self._client.delete_agent(agent_id)
                logger.info(f"Deleted Letta agent for context {context_id}: {agent_id}")
            except Exception as e:
                logger.warning(f"Failed to delete agent {agent_id}: {e}")
            del self._agent_ids[context_id]

    def _ensure_agent(self, context_id: int) -> str:
        """Ensure Letta agent exists for the given context_id.

        Returns existing agent if available, otherwise creates a new one.

        Args:
            context_id: The context/persona ID

        Returns:
            The agent ID string
        """
        existing = self._agent_ids.get(context_id)
        if existing:
            return existing
        return self._create_agent(context_id)

    def _parse_letta_response(self, response) -> Dict[str, Any]:
        """Parse LettaResponse to extract useful information.

        Args:
            response: LettaResponse object from Letta API

        Returns:
            Dictionary with parsed response data including:
            - assistant_message: The agent's response text
            - function_calls: List of function calls made
            - internal_monologue: Agent's internal thoughts
            - usage: Token usage statistics
        """
        import json

        result = {
            "assistant_message": "",
            "function_calls": [],
            "internal_monologue": [],
            "usage": {},
        }

        # Extract usage statistics
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            result["usage"] = {
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

        # Parse messages from response
        if not hasattr(response, "messages") or not response.messages:
            return result

        for msg in response.messages:
            msg_type = getattr(msg, "message_type", None)
            if msg_type is not None:
                msg_type_str = str(msg_type.value) if hasattr(msg_type, "value") else str(msg_type)
            else:
                msg_type_str = type(msg).__name__

            # Handle AssistantMessage
            if msg_type_str == "assistant_message" or "AssistantMessage" in str(type(msg)):
                content = getattr(msg, "content", None)
                if content:
                    if isinstance(content, str):
                        result["assistant_message"] = content
                    elif isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if hasattr(part, "text"):
                                text_parts.append(part.text)
                            elif isinstance(part, str):
                                text_parts.append(part)
                            elif hasattr(part, "content"):
                                text_parts.append(str(part.content))
                        result["assistant_message"] = " ".join(text_parts)
                    elif hasattr(content, "text"):
                        result["assistant_message"] = content.text
                    elif hasattr(content, "message"):
                        result["assistant_message"] = content.message

            # Handle ToolCallMessage / function_call
            elif msg_type_str == "tool_call_message" or "ToolCall" in str(type(msg)):
                tool_call = getattr(msg, "tool_call", None)
                if tool_call:
                    func_name = getattr(tool_call, "name", "")
                    func_args = getattr(tool_call, "arguments", "{}")

                    if isinstance(func_args, str):
                        try:
                            func_args = json.loads(func_args)
                        except json.JSONDecodeError:
                            pass

                    if func_name:
                        result["function_calls"].append({
                            "name": func_name,
                            "arguments": func_args if isinstance(func_args, dict) else str(func_args),
                        })

                        # Letta uses send_message tool to communicate with user
                        if func_name == "send_message" and isinstance(func_args, dict):
                            msg_content = func_args.get("message", "")
                            if msg_content and not result["assistant_message"]:
                                result["assistant_message"] = msg_content
                else:
                    # Fallback for legacy format
                    func_name = getattr(msg, "function_name", None) or getattr(msg, "name", "")
                    func_args = getattr(msg, "function_args", None) or getattr(msg, "arguments", {})

                    if isinstance(func_args, str):
                        try:
                            func_args = json.loads(func_args)
                        except json.JSONDecodeError:
                            pass

                    if func_name:
                        result["function_calls"].append({
                            "name": func_name,
                            "arguments": func_args if isinstance(func_args, dict) else str(func_args),
                        })

                        if func_name == "send_message" and isinstance(func_args, dict):
                            msg_content = func_args.get("message", "")
                            if msg_content and not result["assistant_message"]:
                                result["assistant_message"] = msg_content

            # Handle ReasoningMessage / internal_monologue
            elif msg_type_str == "reasoning_message" or "Reasoning" in str(type(msg)):
                reasoning = getattr(msg, "reasoning", None) or getattr(msg, "content", "")
                if reasoning:
                    result["internal_monologue"].append(reasoning)
            elif msg_type_str == "internal_monologue" or "InternalMonologue" in str(type(msg)):
                monologue = getattr(msg, "internal_monologue", None) or getattr(msg, "content", "")
                if monologue:
                    result["internal_monologue"].append(monologue)

            # Handle ToolReturnMessage / function_return
            elif msg_type_str == "tool_return_message" or "ToolReturn" in str(type(msg)):
                func_return = getattr(msg, "tool_return", None) or getattr(msg, "function_return", None)
                if func_return and result["function_calls"]:
                    result["function_calls"][-1]["return"] = str(func_return)[:1000]
            elif msg_type_str == "function_return" or "FunctionReturn" in str(type(msg)):
                func_return = getattr(msg, "function_return", None)
                if func_return and result["function_calls"]:
                    result["function_calls"][-1]["return"] = str(func_return)[:1000]

        return result

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Store memory using Letta's official agent API.

        Sends message directly to the Letta agent (following official design).
        The agent will internally decide what to store using archival_memory_insert.

        For long inputs, text is split into chunks and sent sequentially.
        """
        context_id = self._get_context_id()
        agent_id = self._ensure_agent(context_id)

        # Split input into chunks if needed
        chunks = self._chunk_text_by_tokens(
            text,
            self.memorize_chunk_tokens,
            self.memorize_chunk_overlap_tokens
        )

        if not chunks:
            chunks = [text] if text else []

        if not self.use_native_memorize:
            start_time = time.time()
            all_passages = []
            all_memory_entries = []
            try:
                for chunk_idx, chunk in enumerate(chunks):
                    passages = self._client.insert_archival_memory(
                        agent_id=agent_id,
                        memory=chunk,
                    )
                    for passage in passages:
                        passage_text = str(getattr(passage, "text", chunk) or chunk)
                        passage_id = str(getattr(passage, "id", "") or "")
                        all_passages.append({
                            "text": passage_text,
                            "id": passage_id,
                            "function": "direct_archival_memory_insert",
                            "chunk_index": chunk_idx,
                        })
                        all_memory_entries.append({
                            "event": "ADD",
                            "function": "direct_archival_memory_insert",
                            "memory": passage_text[:500],
                            "id": passage_id,
                            "chunk_index": chunk_idx,
                        })
            except Exception as e:
                logger.error("Letta direct archival insert failed: %s", e)
                return MemoryBuildResult(
                    success=False,
                    method="letta",
                    action="direct_archival_memory_insert",
                    input_content=text,
                    stored_content="",
                    extraction_result=f"[Error: {e}]",
                    all_passages=all_passages,
                    memory_entries=all_memory_entries,
                    chunk_count=len(all_passages),
                    time_cost=time.time() - start_time,
                    extra={"agent_id": agent_id, "error": str(e)},
                )

            self._memory_chunks.append(text)
            self._is_initialized = True
            return MemoryBuildResult(
                success=True,
                method="letta",
                action="direct_archival_memory_insert",
                input_content=text,
                stored_content=text,
                extraction_result=(
                    f"Inserted {len(all_passages)} passage(s) through Letta's "
                    "vendored archival memory layer."
                ),
                all_passages=all_passages,
                memory_entries=all_memory_entries,
                chunk_count=len(chunks),
                time_cost=time.time() - start_time,
                extra={"agent_id": agent_id, "chunks_processed": len(chunks)},
            )

        self._configure_agent_tool_flow(agent_id, "memorize")
        if self.enforce_memory_tool_flow:
            # Letta continues while step_count > max_chaining_steps, so zero
            # permits the initial archival insert but suppresses heartbeat-driven
            # duplicate tool calls from local Qwen models.
            self._client.server.max_chaining_steps = 0

        total_time_cost = 0.0
        all_memory_entries = []
        all_passages = []
        all_function_calls = []
        all_internal_monologue = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_assistant_message = ""

        for chunk_idx, chunk in enumerate(chunks):
            # Send chunk directly to agent (official Letta design - agent decides what to store)
            start_time = time.time()
            try:
                response = self._client.user_message(agent_id=agent_id, message=chunk)
                chunk_time = time.time() - start_time
                total_time_cost += chunk_time
            except Exception as e:
                logger.error(f"Letta user_message failed for chunk {chunk_idx}: {e}")
                return MemoryBuildResult(
                    success=False,
                    method="letta",
                    action="user_message",
                    input_content=text,
                    stored_content="",
                    extraction_result=f"[Error at chunk {chunk_idx}: {e}]",
                    all_passages=all_passages,
                    memory_entries=all_memory_entries,
                    chunk_count=chunk_idx,
                    time_cost=total_time_cost,
                    extra={"agent_id": agent_id, "error": str(e), "failed_chunk": chunk_idx},
                )

            # Parse response
            parsed = self._parse_letta_response(response)

            # Record token usage
            self._record_usage_to_tracker(parsed, "memorize", chunk_time)

            # Accumulate usage
            chunk_usage = parsed.get("usage", {})
            total_usage["prompt_tokens"] += chunk_usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += chunk_usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += chunk_usage.get("total_tokens", 0)

            # Extract memory operations
            for func_call in parsed["function_calls"]:
                all_function_calls.append(func_call)
                if func_call["name"] in ["archival_memory_insert", "core_memory_append", "core_memory_replace"]:
                    args = func_call.get("arguments", {})
                    if isinstance(args, dict):
                        content = args.get("content", args.get("new_content", ""))
                    else:
                        content = str(args)

                    all_memory_entries.append({
                        "event": "ADD",
                        "function": func_call["name"],
                        "memory": content[:500] if content else "",
                        "chunk_index": chunk_idx,
                    })
                    all_passages.append({
                        "text": content,
                        "function": func_call["name"],
                        "chunk_index": chunk_idx,
                    })

            all_internal_monologue.extend(parsed.get("internal_monologue", []))
            if parsed.get("assistant_message"):
                last_assistant_message = parsed["assistant_message"]

        # Build extraction result summary
        extraction_parts = []
        if all_internal_monologue:
            extraction_parts.append("Agent thinking:\n" + "\n".join(all_internal_monologue[:5]))
        if last_assistant_message:
            extraction_parts.append(f"Agent response: {last_assistant_message[:500]}")
        extraction_result = "\n\n".join(extraction_parts) if extraction_parts else "[No extraction details]"

        self._memory_chunks.append(text)
        self._is_initialized = True

        return MemoryBuildResult(
            success=True,
            method="letta",
            action="user_message",
            input_content=text,
            stored_content=last_assistant_message,
            extraction_result=extraction_result,
            all_passages=all_passages,
            memory_entries=all_memory_entries,
            chunk_count=len(chunks),
            time_cost=total_time_cost,
            extra={
                "agent_id": agent_id,
                "function_calls": all_function_calls,
                "internal_monologue": all_internal_monologue,
                "usage": total_usage,
                "chunks_processed": len(chunks),
            },
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """Query using Letta's official agent API.

        Sends the question to the Letta agent which will use its internal
        archival_memory_search tool to retrieve relevant memories and respond.

        Question is truncated if exceeding max_question_tokens.
        """
        context_id = self._get_context_id()
        agent_id = self._ensure_agent(context_id)

        # Truncate question if needed
        bounded_question = self._truncate_to_tokens(question, self.max_question_tokens)

        if not self.use_native_query:
            start_time = time.time()
            retrieval = self.retrieve(bounded_question)
            retrieval_system = (
                "Use the following relevant archival memories retrieved by Letta "
                "to answer the user's question. If the memories do not contain "
                "the answer, say so.\n\n"
                f"{retrieval.memory_context}"
            )
            if system_message:
                retrieval_system = f"{system_message}\n\n{retrieval_system}"
            get_usage_tracker().set_phase("query")
            response = self._qa_llm_client.chat(
                format_messages(bounded_question, retrieval_system)
            )
            return AgentResponse(
                output=response.content,
                query_time=time.time() - start_time,
                retrieved_count=retrieval.retrieved_count,
                retrieved_memories=retrieval.retrieved_memories,
                extra={
                    "method": "letta_archival_manual_qa",
                    "agent_id": agent_id,
                    "tokens_used": {
                        "input": response.input_tokens,
                        "output": response.output_tokens,
                    },
                },
            )

        # Build query message
        if system_message:
            query_message = f"{system_message}\n\nQuestion: {bounded_question}"
        else:
            query_message = bounded_question

        start_time = time.time()
        try:
            self._configure_agent_tool_flow(agent_id, "query")
            # Switch to restricted context_window for query phase (token budget truncation)
            # Only update once per agent (build phase is always completed before query phase)
            if self.max_context_tokens and self.max_context_tokens < self.context_window:
                if not getattr(self, '_query_context_applied', {}).get(agent_id):
                    query_llm_config = self._build_llm_config().model_copy(
                        update={"context_window": self.max_context_tokens}
                    )
                    self._client.update_agent(agent_id, llm_config=query_llm_config)
                    if not hasattr(self, '_query_context_applied'):
                        self._query_context_applied = {}
                    self._query_context_applied[agent_id] = True

            response = self._client.user_message(agent_id=agent_id, message=query_message)
            query_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Letta user_message failed: {e}")
            raise RuntimeError("Letta query user_message failed") from e

        parsed = self._parse_letta_response(response)
        self._record_usage_to_tracker(parsed, "query", query_time)

        # Extract retrieved memories from search function calls
        retrieved_memories = []
        for func_call in parsed["function_calls"]:
            if func_call["name"] == "archival_memory_search":
                args = func_call.get("arguments", {})
                search_query = args.get("query", "") if isinstance(args, dict) else str(args)
                search_results = func_call.get("return", "")

                # Truncate retrieved memory results
                truncated_results = self._truncate_to_tokens(
                    search_results, self.max_context_tokens // 4
                ) if search_results else ""

                retrieved_memories.append({
                    "memory": truncated_results if truncated_results else f"Search: {search_query}",
                    "type": "archival_memory_search",
                    "query": search_query,
                })

        return AgentResponse(
            output=parsed["assistant_message"],
            query_time=query_time,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=retrieved_memories,
            extra={
                "method": "letta",
                "agent_id": agent_id,
                "function_calls": parsed["function_calls"],
                "internal_monologue": parsed["internal_monologue"],
                "usage": parsed["usage"],
            },
        )

    def retrieve(self, question: str, **kwargs) -> MemoryRetrievalResult:
        """Search Letta archival passages directly, without an agent answer turn."""
        context_id = self._get_context_id()
        agent_id = self._ensure_agent(context_id)
        actor = self._client.server.user_manager.get_user_or_default(
            user_id=self._client.user_id
        )
        agent_state = self._client.server.agent_manager.get_agent_by_id(
            agent_id=agent_id,
            actor=actor,
        )
        passages = self._client.server.agent_manager.list_passages(
            actor=actor,
            agent_id=agent_id,
            query_text=question,
            limit=self.retrieve_num,
            embed_query=True,
            embedding_config=agent_state.embedding_config,
        )
        retrieved_memories = [
            {
                "memory": passage.text,
                "type": "letta_archival_search",
                "id": str(passage.id),
                "metadata": passage.metadata or {},
            }
            for passage in passages
        ]
        return MemoryRetrievalResult(
            memory_context="\n\n".join(
                f"[Memory {index}]\n{item['memory']}"
                for index, item in enumerate(retrieved_memories, start=1)
            ),
            retrieved_count=len(retrieved_memories),
            retrieved_memories=retrieved_memories,
            extra={
                "method": "letta",
                "context_id": context_id,
                "agent_id": agent_id,
            },
        )

    def reset(self) -> None:
        """Reset agent state and delete all created agents."""
        # Delete all agents for this instance
        for context_id in list(self._agent_ids.keys()):
            self._delete_agent(context_id)

        super().reset()
        self._agent_ids = {}

    def set_context_id(self, context_id: int) -> None:
        """Set context ID and handle agent lifecycle.

        When switching to a different context (persona), the old agent is deleted
        to ensure clean separation between personas.
        """
        old_context_id = self._context_id

        # If switching to a different context, delete the old agent
        if old_context_id is not None and old_context_id != context_id:
            if old_context_id in self._agent_ids:
                self._delete_agent(old_context_id)
                logger.info(f"Switched context from {old_context_id} to {context_id}, deleted old agent")

        super().set_context_id(context_id)
