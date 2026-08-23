"""MIRIX Agent - Multi-agent memory system with six-component memory architecture."""

import asyncio
import gc
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Add MIRIX to path
MIRIX_PATH = os.path.join(os.path.dirname(__file__), "MIRIX")
if MIRIX_PATH not in sys.path:
    sys.path.insert(0, MIRIX_PATH)

from .base import BaseAgent, MemoryBuildResult, AgentResponse, MemoryRetrievalResult
from utils.llm_client import (
    create_llm_client,
    format_messages,
    BaseLLMClient,
    get_usage_tracker,
    LLMResponse,
)

logger = logging.getLogger(__name__)


class MIRIXAgent(BaseAgent):
    """MIRIX Agent for intelligent multi-component memory management.

    MIRIX provides a six-component memory system:
    - Core Memory: Essential persona and human blocks
    - Episodic Memory: Event/conversation memories with timestamps
    - Semantic Memory: Concept/fact knowledge
    - Procedural Memory: Step/procedure memories
    - Resource Memory: Resource/file memories
    - Knowledge Vault: Sensitive information storage
    """

    METHOD_TYPE = "agentic_memory"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # Embedding config
        embedding_model: str = "text-embedding-3-small",
        embedding_provider: str = "openai",
        embedding_model_path: Optional[str] = None,
        embedding_dim: int = 1536,
        embedding_device: Optional[str] = None,  # For local embedding: "cuda", "cpu", "mps"
        # Memory retrieval config
        retrieve_num: int = 5,  # Number of memories to retrieve per memory type
        # Chunking config
        memorize_chunk_tokens: int = 2500,
        memorize_chunk_overlap_tokens: int = 200,
        query_memory_item_tokens: int = 300,
        query_memory_context_tokens: int = 1800,
        # Context limits
        max_input_tokens: int = 8000,
        max_question_tokens: int = 4096,
        max_context_tokens: int = 120000,
        # Query mode
        use_native_query: bool = True,  # Use MIRIX native send_message for query
        enforce_single_meta_update: bool = False,
        strict_internal_errors: bool = False,
        bounded_memory_tool_schema: bool = False,
        required_tool_choice: bool = False,
        core_json_tool_bridge: bool = False,
        json_tool_bridge_attempts: int = 3,
        normalize_missing_update_ids: bool = False,
        memory_tool_max_items: int = 4,
        memory_tool_max_string_chars: int = 1024,
        core_memory_tool_max_string_chars: int = 256,
        core_memory_max_tokens: int = 2048,
        memory_agent_max_tokens: int = 10000,
        core_memory_input_max_chars: int = 0,
        memory_agent_input_max_chars: int = 0,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        # Store config
        self._provider = provider
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # Embedding config
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.embedding_device = embedding_device

        # Memory config
        self.retrieve_num = retrieve_num

        # Chunking config
        self.memorize_chunk_tokens = memorize_chunk_tokens
        self.memorize_chunk_overlap_tokens = memorize_chunk_overlap_tokens
        self.query_memory_item_tokens = query_memory_item_tokens
        self.query_memory_context_tokens = query_memory_context_tokens

        # Limits
        self.max_input_tokens = max_input_tokens
        self.max_question_tokens = max_question_tokens
        self.max_context_tokens = max_context_tokens

        # Query mode
        self.use_native_query = use_native_query
        self.enforce_single_meta_update = enforce_single_meta_update
        self.strict_internal_errors = strict_internal_errors
        self.bounded_memory_tool_schema = bounded_memory_tool_schema
        self.required_tool_choice = required_tool_choice
        self.core_json_tool_bridge = core_json_tool_bridge
        self.json_tool_bridge_attempts = max(
            1, min(int(json_tool_bridge_attempts), 5)
        )
        self.normalize_missing_update_ids = normalize_missing_update_ids
        self.memory_tool_max_items = memory_tool_max_items
        self.memory_tool_max_string_chars = memory_tool_max_string_chars
        self.core_memory_tool_max_string_chars = core_memory_tool_max_string_chars
        self.core_memory_max_tokens = core_memory_max_tokens
        self.memory_agent_max_tokens = memory_agent_max_tokens
        self.core_memory_input_max_chars = core_memory_input_max_chars
        self.memory_agent_input_max_chars = memory_agent_input_max_chars

        # These switches are consumed by the vendored MIRIX code while it
        # creates tool schemas and handles an internal LLM failure.  Set them
        # before LocalClient/MetaAgent initialization so a fresh task gets one
        # immutable behavior for its full lifetime.
        if self.strict_internal_errors:
            os.environ["MIRIX_STRICT_INTERNAL_ERRORS"] = "1"
        if self.bounded_memory_tool_schema:
            os.environ["MIRIX_BOUNDED_MEMORY_TOOL_SCHEMA"] = "1"
            os.environ["MIRIX_MEMORY_TOOL_MAX_ITEMS"] = str(memory_tool_max_items)
            os.environ["MIRIX_MEMORY_TOOL_MAX_STRING_CHARS"] = str(
                memory_tool_max_string_chars
            )
            os.environ["MIRIX_CORE_MEMORY_TOOL_MAX_STRING_CHARS"] = str(
                core_memory_tool_max_string_chars
            )
            if core_memory_input_max_chars > 0:
                os.environ["MIRIX_CORE_INPUT_MAX_CHARS"] = str(
                    core_memory_input_max_chars
                )
            if memory_agent_input_max_chars > 0:
                os.environ["MIRIX_MEMORY_AGENT_INPUT_MAX_CHARS"] = str(
                    memory_agent_input_max_chars
                )
        if self.required_tool_choice:
            os.environ["MIRIX_REQUIRED_TOOL_CHOICE"] = "1"
        if self.core_json_tool_bridge:
            os.environ["MIRIX_CORE_JSON_TOOL_BRIDGE"] = "1"
            os.environ["MIRIX_JSON_TOOL_BRIDGE_ATTEMPTS"] = str(
                self.json_tool_bridge_attempts
            )
        if self.normalize_missing_update_ids:
            os.environ["MIRIX_NORMALIZE_MISSING_UPDATE_IDS"] = "1"

        # LLM client for fallback Q&A (uses utils/llm_client for token tracking)
        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # MIRIX components (lazy initialization)
        self._client = None
        self._meta_agent = None
        self._is_client_owner = True  # Track if we own the client for cleanup

        # Token tracking for MIRIX internal calls (memory phase)
        self._mirix_input_tokens = 0
        self._mirix_output_tokens = 0
        self._mirix_step_count = 0

        # Initialize MIRIX
        self._init_mirix()

    def _get_event_loop(self):
        """Get or create event loop for async operations."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        loop = self._get_event_loop()
        return loop.run_until_complete(coro)

    def _init_mirix(self) -> None:
        """Initialize MIRIX LocalClient and MetaAgent."""
        logger.info("[MIRIXAgent] Initializing MIRIX...")

        try:
            # Import MIRIX components
            from mirix.local_client.local_client import LocalClient
            from mirix.schemas.llm_config import LLMConfig
            from mirix.schemas.embedding_config import EmbeddingConfig
            from mirix.schemas.agent import CreateMetaAgent
            from mirix.server.server import ensure_tables_created

            # Ensure database tables are created first
            logger.info("[MIRIXAgent] Ensuring database tables are created...")
            self._run_async(ensure_tables_created())

            # Build LLM config
            llm_config = LLMConfig(
                model=self.model,
                model_endpoint_type=self._get_mirix_endpoint_type(self._provider),
                model_endpoint=self._base_url,
                context_window=self.max_context_tokens,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Build embedding config
            embedding_config = self._build_embedding_config()

            # Create LocalClient
            async def create_client():
                client = await LocalClient.create(
                    debug=False,
                    default_llm_config=llm_config,
                    default_embedding_config=embedding_config,
                )
                return client

            self._client = self._run_async(create_client())
            self._is_client_owner = True
            logger.info("[MIRIXAgent] LocalClient created successfully")

            # Create MetaAgent (always create new one, don't reuse)
            self._create_meta_agent(llm_config, embedding_config)

        except Exception as e:
            logger.error(f"[MIRIXAgent] Failed to initialize MIRIX: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _create_meta_agent(self, llm_config, embedding_config) -> None:
        """Create a new MetaAgent."""
        from mirix.schemas.agent import CreateMetaAgent

        async def setup_meta_agent():
            logger.info("[MIRIXAgent] Creating new MetaAgent...")
            create_request = CreateMetaAgent(
                llm_config=llm_config,
                embedding_config=embedding_config,
                agents=self._meta_agent_specs(),
                system_prompts=(
                    self._bounded_memory_system_prompts()
                    if self.bounded_memory_tool_schema
                    else None
                ),
            )
            meta_agent = await self._client.create_meta_agent(request=create_request)
            if meta_agent and self.bounded_memory_tool_schema:
                # Long timelines can make a small local model replay either the
                # transcript or its complete memory state.  Bound every child
                # that writes memory, with a tighter scalar-core budget.  The
                # schema/prompt/runtime bounds below keep valid calls well under
                # these completion caps; the caps remain a final fail-closed
                # guard against malformed JSON.
                children = await self._client.server.agent_manager.list_agents(
                    parent_id=meta_agent.id,
                    actor=self._client.client,
                )
                for child in children:
                    agent_type = getattr(child.agent_type, "value", child.agent_type)
                    completion_cap = self._memory_agent_completion_cap(agent_type)
                    if completion_cap is None:
                        continue
                    child_llm_config = llm_config.model_copy(
                        update={"max_tokens": completion_cap}
                    )
                    await self._client.server.agent_manager.update_llm_config(
                        agent_id=child.id,
                        llm_config=child_llm_config,
                        actor=self._client.client,
                    )
                    logger.info(
                        "[MIRIXAgent] %s completion cap: %d tokens",
                        agent_type,
                        completion_cap,
                    )
            if meta_agent and self.enforce_single_meta_update:
                # Local Qwen can repeatedly invoke trigger_memory_update against
                # the same stale retrieval snapshot, causing duplicate deletes
                # and inflated writes.  Constrain the adapted flow to one native
                # multi-memory routing call followed by MIRIX's finish tool.
                from mirix.schemas.agent import UpdateAgent

                await self._client.server.agent_manager.update_agent(
                    agent_id=meta_agent.id,
                    agent_update=UpdateAgent(tool_rules=self._meta_agent_tool_flow_spec()),
                    actor=self._client.client,
                )
            return meta_agent

        self._meta_agent = self._run_async(setup_meta_agent())
        if self._meta_agent:
            logger.info(f"[MIRIXAgent] MetaAgent ready: {self._meta_agent.id}")
        else:
            logger.warning("[MIRIXAgent] MetaAgent creation returned None")

    def _memory_agent_completion_cap(self, agent_type: str) -> Optional[int]:
        """Return adapted completion cap for a child that can write memory."""
        if agent_type == "core_memory_agent":
            return self.core_memory_max_tokens
        if agent_type in {
            "resource_memory_agent",
            "semantic_memory_agent",
            "episodic_memory_agent",
            "procedural_memory_agent",
            "knowledge_vault_memory_agent",
        }:
            return self.memory_agent_max_tokens
        return None

    @staticmethod
    def _meta_agent_specs():
        """Return the full MIRIX agent set with required core-memory blocks."""
        return [
            {
                "core_memory_agent": {
                    "blocks": [
                        {"label": "human", "value": ""},
                        {"label": "persona", "value": ""},
                    ]
                }
            },
            "resource_memory_agent",
            "semantic_memory_agent",
            "episodic_memory_agent",
            "procedural_memory_agent",
            "knowledge_vault_memory_agent",
            "meta_memory_agent",
            "reflexion_agent",
            "background_agent",
        ]

    def _bounded_memory_system_prompts(self):
        """Append a local-model delta contract to MIRIX memory-agent prompts."""
        from mirix.prompts import gpt_system

        suffix = f"""

[Adapted local-model output contract]
- The memories in the system prompt are reference state. Never copy or restate
  them as new output.
- Write only the delta supported by the current incoming message.
- A list-valued write argument may contain at most
  {self.memory_tool_max_items} items total. Consolidate overlapping facts.
- Each free-text field must be at most
  {self.memory_tool_max_string_chars} characters; summaries and names must be
  at most 512 characters.
- Make exactly one concise tool call. If there is no new delta, call
  finish_memory_update instead of replaying prior memory.
""".strip()
        names = (
            "core_memory_agent",
            "resource_memory_agent",
            "semantic_memory_agent",
            "episodic_memory_agent",
            "procedural_memory_agent",
            "knowledge_vault_memory_agent",
        )
        prompts = {
            name: f"{gpt_system.get_system_text(f'base/{name}')}\n\n{suffix}"
            for name in names
        }
        prompts["core_memory_agent"] += "\n\n" + f"""

[Core-memory hard limit]
- Never quote, summarize, or reproduce the conversation transcript.
- Ignore the earlier request to extract every detail. Store only stable user
  identity, durable preferences, long-term traits, and enduring care facts.
- `content` is a compact delta, not a rewritten conversation. It must be at
  most {self.core_memory_tool_max_string_chars} characters, contain no newline,
  and be emitted as one plain JSON string with no literal control characters.
- Prefer `finish_memory_update` when the incoming message only repeats a known
  symptom, reassurance, transient event, or clinician advice.
- Do not use `core_memory_rewrite` merely to add a new fact. If a rewrite is
  necessary, keep only durable facts and remain within the same character cap.
""".strip()
        return prompts

    @staticmethod
    def _meta_agent_tool_flow_spec():
        """Bound adapted MetaAgent writes to one routing call per input chunk."""
        return [
            {"tool_name": "trigger_memory_update", "type": "run_first"},
            {
                "tool_name": "trigger_memory_update",
                "type": "constrain_child_tools",
                "children": ["finish_memory_update"],
            },
            {"tool_name": "finish_memory_update", "type": "exit_loop"},
        ]

    def _get_mirix_endpoint_type(self, provider: str) -> str:
        """Convert provider name to MIRIX endpoint type."""
        mapping = {
            "openai": "openai",
            "azure": "azure_openai",
            "anthropic": "anthropic",
            "google": "google_ai",
            "gemini": "google_ai",
        }
        return mapping.get(provider.lower(), "openai")

    def _build_embedding_config(self):
        """Build MIRIX embedding config based on provider."""
        from mirix.schemas.embedding_config import EmbeddingConfig

        provider = self.embedding_provider.lower()

        if provider == "local":
            # Local sentence-transformers model (no API required)
            model_path = self.embedding_model_path or self.embedding_model
            logger.info(f"[MIRIXAgent] Using local embedding: {model_path}")
            return EmbeddingConfig(
                embedding_model=model_path,
                embedding_endpoint_type="local",
                embedding_endpoint=self.embedding_device,  # Device: "cuda", "cpu", "mps", or None for auto
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=512,
            )
        elif provider in ("huggingface", "hugging-face"):
            # HuggingFace TEI server (requires running TEI service)
            model_path = self.embedding_model_path or self.embedding_model
            tei_url = self.embedding_device  # Reuse device field for TEI URL
            if tei_url and tei_url.startswith("http"):
                logger.info(f"[MIRIXAgent] Using HuggingFace TEI: {model_path} at {tei_url}")
                return EmbeddingConfig(
                    embedding_model=model_path,
                    embedding_endpoint_type="hugging-face",
                    embedding_endpoint=tei_url,
                    embedding_dim=self.embedding_dim,
                    embedding_chunk_size=512,
                )
            else:
                # Fall back to local if no TEI URL provided
                logger.info(f"[MIRIXAgent] No TEI URL provided, using local embedding: {model_path}")
                return EmbeddingConfig(
                    embedding_model=model_path,
                    embedding_endpoint_type="local",
                    embedding_endpoint=None,
                    embedding_dim=self.embedding_dim,
                    embedding_chunk_size=512,
                )
        elif provider == "openai":
            # OpenAI embedding API
            base_url = self._base_url or "https://api.openai.com/v1"
            logger.info(f"[MIRIXAgent] Using OpenAI embedding: {self.embedding_model} at {base_url}")
            return EmbeddingConfig(
                embedding_model=self.embedding_model,
                embedding_endpoint_type="openai",
                embedding_endpoint=base_url,
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=8191,
            )
        elif provider == "ollama":
            # Ollama embedding
            ollama_url = self.embedding_model_path or "http://localhost:11434"
            logger.info(f"[MIRIXAgent] Using Ollama embedding: {self.embedding_model} at {ollama_url}")
            return EmbeddingConfig(
                embedding_model=self.embedding_model,
                embedding_endpoint_type="ollama",
                embedding_endpoint=ollama_url,
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=512,
            )
        else:
            # Default to OpenAI-compatible
            base_url = self._base_url or "https://api.openai.com/v1"
            logger.info(f"[MIRIXAgent] Using OpenAI-compatible embedding: {self.embedding_model}")
            return EmbeddingConfig(
                embedding_model=self.embedding_model,
                embedding_endpoint_type="openai",
                embedding_endpoint=base_url,
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=300,
            )

    def _split_text_into_chunks(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
        """Split text into chunks with optional overlap."""
        if not text.strip():
            return []

        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._tokenizer.decode(chunk_tokens))

            if end >= len(tokens):
                break

            # Move start with overlap
            start = end - overlap_tokens if overlap_tokens > 0 else end

        return chunks

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    def _record_mirix_usage(self, usage, latency: float = 0.0) -> None:
        """Record MIRIX usage statistics to global tracker."""
        if usage is None:
            return

        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)
        step_count = getattr(usage, 'step_count', 0)

        self._mirix_input_tokens += input_tokens
        self._mirix_output_tokens += output_tokens
        self._mirix_step_count += step_count

        # Record to global tracker
        if input_tokens > 0 or output_tokens > 0:
            response = LLMResponse(
                content="",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=latency,
                model=self.model,
            )
            get_usage_tracker().record(response)
            logger.debug(f"[MIRIXAgent] Token usage: in={input_tokens}, out={output_tokens}, steps={step_count}")

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Store text into MIRIX memory system.

        This uses MIRIX's native memory extraction pipeline through send_message().
        MIRIX internally routes content to appropriate memory components (Episodic,
        Semantic, Procedural, etc.) based on content analysis.

        Token tracking is done via MirixUsageStatistics from MirixResponse.
        """
        start_time = time.time()

        # Reset MIRIX token counters for this operation
        self._mirix_input_tokens = 0
        self._mirix_output_tokens = 0
        self._mirix_step_count = 0

        # Truncate input if needed
        bounded_text = self._truncate_to_tokens(text, self.max_input_tokens)

        # Split text into chunks
        chunks = self._split_text_into_chunks(
            bounded_text,
            max_tokens=self.memorize_chunk_tokens,
            overlap_tokens=self.memorize_chunk_overlap_tokens
        )

        memory_entries = []
        all_passages = []
        chunk_responses = []
        chunk_errors = []

        async def process_chunks():
            nonlocal memory_entries, all_passages, chunk_responses, chunk_errors

            for i, chunk in enumerate(chunks):
                chunk_start = time.time()
                try:
                    # Send message to MIRIX for memory extraction
                    # MIRIX's MetaAgent will automatically route content to appropriate
                    # memory components (Episodic, Semantic, Procedural, etc.)
                    # NOTE: We don't pass user_id here - MIRIX uses the LocalClient's
                    # default user which was created during initialization.
                    response = await self._client.send_message(
                        agent_id=self._meta_agent.id,
                        role="user",
                        message=chunk,
                    )

                    chunk_latency = time.time() - chunk_start

                    # Extract usage statistics from MirixResponse
                    if response and hasattr(response, 'usage') and response.usage:
                        self._record_mirix_usage(response.usage, chunk_latency)

                    # Record passage info for logging
                    passage_info = {
                        "chunk_index": i,
                        "chunk_tokens": self.count_tokens(chunk),
                        "content_preview": chunk[:500] + "..." if len(chunk) > 500 else chunk,
                        "latency": round(chunk_latency, 3),
                    }

                    # Extract response messages for logging
                    # MIRIX MirixResponse has messages as List[Union[ToolCallMessage, ToolReturnMessage, AssistantMessage, ...]]
                    # Each message has 'message_type' field to identify the type
                    # - ToolCallMessage: has tool_call.name, tool_call.arguments (JSON string)
                    # - ToolReturnMessage: has tool_return (string)
                    # - AssistantMessage: has content (string or list)
                    response_messages = []
                    if response and hasattr(response, 'messages'):
                        logger.debug(f"[MIRIXAgent] Response has {len(response.messages)} messages")
                        for msg in response.messages:
                            msg_type = getattr(msg, 'message_type', None)
                            if hasattr(msg_type, 'value'):
                                msg_type = msg_type.value

                            msg_content = ""
                            func_name = ""
                            func_args = {}

                            if msg_type == 'tool_call_message':
                                # ToolCallMessage has tool_call with name and arguments
                                tool_call = getattr(msg, 'tool_call', None)
                                if tool_call:
                                    func_name = getattr(tool_call, 'name', 'unknown')
                                    args_str = getattr(tool_call, 'arguments', '{}')
                                    try:
                                        if isinstance(args_str, str):
                                            func_args = json.loads(args_str) if args_str else {}
                                        elif isinstance(args_str, dict):
                                            func_args = args_str
                                    except Exception as e:
                                        logger.debug(f"[MIRIXAgent] Failed to parse tool call args: {e}")
                                        func_args = {"raw": str(args_str)[:200]}

                                    args_preview = str(func_args)[:200]
                                    msg_content = f"{func_name}({args_preview}{'...' if len(str(func_args)) > 200 else ''})"

                                    response_messages.append({
                                        "type": "tool_call",
                                        "content": msg_content,
                                        "function_name": func_name,
                                        "arguments": func_args,
                                    })

                                    # Extract memory entries from tool calls
                                    # MIRIX memory-related function names
                                    memory_functions = [
                                        'trigger_memory_update', 'trigger_memory_update_with_instruction',
                                        'episodic_memory_insert', 'episodic_memory_merge', 'episodic_memory_replace',
                                        'semantic_memory_insert', 'semantic_memory_update',
                                        'procedural_memory_insert', 'procedural_memory_update',
                                        'resource_memory_insert', 'knowledge_vault_insert',
                                        'finish_memory_update', 'check_episodic_memory', 'check_semantic_memory',
                                    ]
                                    if func_name in memory_functions or '_memory' in func_name.lower():
                                        memory_entries.append({
                                            "type": "mirix_memory_operation",
                                            "function": func_name,
                                            "arguments": func_args,
                                            "chunk_index": i,
                                        })

                            elif msg_type == 'tool_return_message':
                                # ToolReturnMessage has tool_return
                                tool_return = getattr(msg, 'tool_return', '') or ''
                                msg_content = str(tool_return)[:500]
                                response_messages.append({
                                    "type": "tool_return",
                                    "content": msg_content,
                                })

                            elif msg_type == 'assistant_message':
                                # AssistantMessage has content (string or list)
                                content = getattr(msg, 'content', '')
                                if isinstance(content, str):
                                    msg_content = content[:500]
                                elif isinstance(content, list):
                                    # Extract text from TextContent items
                                    texts = []
                                    for c in content:
                                        if hasattr(c, 'text'):
                                            texts.append(str(c.text))
                                    msg_content = ' '.join(texts)[:500]
                                response_messages.append({
                                    "type": "assistant_message",
                                    "content": msg_content,
                                })

                            elif msg_type == 'internal_monologue':
                                # Internal monologue
                                monologue = getattr(msg, 'internal_monologue', '') or ''
                                msg_content = str(monologue)[:500]
                                response_messages.append({
                                    "type": "internal_monologue",
                                    "content": msg_content,
                                })

                            else:
                                # Other message types
                                response_messages.append({
                                    "type": str(msg_type) if msg_type else "unknown",
                                    "content": str(msg)[:200],
                                })

                    passage_info["response_messages"] = response_messages[:10]  # Limit for logging
                    all_passages.append(passage_info)

                    # Store chunk response summary
                    chunk_responses.append({
                        "chunk_index": i,
                        "message_count": len(response_messages),
                        "usage": {
                            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if response and response.usage else 0,
                            "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if response and response.usage else 0,
                        } if response and response.usage else {},
                    })

                except Exception as e:
                    logger.warning(f"[MIRIXAgent] Failed to process chunk {i}: {e}")
                    chunk_errors.append({"chunk_index": i, "error": str(e)})
                    all_passages.append({
                        "chunk_index": i,
                        "error": str(e),
                        "chunk_tokens": self.count_tokens(chunk),
                    })
                    memory_entries.append({
                        "type": "error",
                        "error": str(e),
                        "chunk_index": i,
                    })

        # Run async memory processing
        self._run_async(process_chunks())

        build_succeeded = not chunk_errors and len(all_passages) == len(chunks)

        # Only mark a complete write as initialized. The evaluator treats any
        # partial MIRIX write as a failed unit and resets the isolated state.
        if build_succeeded:
            self._memory_chunks.append(text)
            self._is_initialized = True

        time_cost = time.time() - start_time

        # Build extraction result summary
        extraction_summary = f"Processed {len(chunks)} chunks through MIRIX MetaAgent\n"
        extraction_summary += f"Total tokens: input={self._mirix_input_tokens}, output={self._mirix_output_tokens}\n"
        extraction_summary += f"Step count: {self._mirix_step_count}\n\n"

        for i, passage in enumerate(all_passages[:5]):
            extraction_summary += f"Chunk {i}: {passage.get('chunk_tokens', 0)} tokens"
            if 'error' in passage:
                extraction_summary += f" [ERROR: {passage['error']}]"
            else:
                extraction_summary += f" [{len(passage.get('response_messages', []))} responses]"
            extraction_summary += "\n"

        return MemoryBuildResult(
            success=build_succeeded,
            method="mirix",
            action="add_to_memory",
            input_content=text,
            stored_content=bounded_text,
            memory_entries=memory_entries,
            chunk_count=len(chunks),
            time_cost=time_cost,
            extraction_result=extraction_summary,
            all_passages=all_passages,
            extra={
                "error": (
                    None
                    if build_succeeded
                    else f"{len(chunk_errors)} of {len(chunks)} chunks failed"
                ),
                "chunk_errors": chunk_errors,
                "mirix_input_tokens": self._mirix_input_tokens,
                "mirix_output_tokens": self._mirix_output_tokens,
                "mirix_step_count": self._mirix_step_count,
                "total_input_tokens": self.count_tokens(text),
                "bounded_input_tokens": self.count_tokens(bounded_text),
                "chunk_responses": chunk_responses,
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_provider,
            }
        )

    def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from MIRIX using all memory types.

        Note: MIRIX's retrieve_memory with memory_type="all" retrieves `limit` items
        from EACH memory type, then merges them. To ensure we return at most
        `retrieve_num` total memories, we truncate the final result.
        """
        async def do_retrieve():
            # Calculate per-type limit to avoid over-fetching
            # With 5 memory types, we want at least 1 from each type
            # But we'll still truncate the final result to retrieve_num
            per_type_limit = max(1, (self.retrieve_num + 4) // 5)  # Ceiling division

            results = await self._client.retrieve_memory(
                agent_id=self._meta_agent.id,
                query=query,
                memory_type="all",  # Search all memory types
                search_method="embedding",  # Use semantic search
                limit=per_type_limit,  # Per-type limit
            )
            return results

        try:
            results = self._run_async(do_retrieve())
            memories = []

            if results and "results" in results:
                for entry in results["results"]:
                    memory_type = entry.get("memory_type", "unknown")

                    # Build memory content based on type
                    content_parts = []

                    # Different memory types have different fields
                    if memory_type == "episodic":
                        if entry.get("summary"):
                            content_parts.append(entry["summary"])
                        if entry.get("details"):
                            content_parts.append(entry["details"])
                    elif memory_type == "semantic":
                        if entry.get("name"):
                            content_parts.append(f"[{entry['name']}]")
                        if entry.get("summary"):
                            content_parts.append(entry["summary"])
                        if entry.get("details"):
                            content_parts.append(entry["details"])
                    elif memory_type == "procedural":
                        if entry.get("summary"):
                            content_parts.append(entry["summary"])
                        if entry.get("steps"):
                            steps = entry["steps"]
                            if isinstance(steps, list):
                                content_parts.append(" -> ".join(steps[:5]))
                            else:
                                content_parts.append(str(steps))
                    elif memory_type == "resource":
                        if entry.get("summary"):
                            content_parts.append(entry["summary"])
                        if entry.get("content"):
                            content_parts.append(entry["content"][:500])
                    elif memory_type == "knowledge_vault":
                        if entry.get("caption"):
                            content_parts.append(entry["caption"])
                    else:
                        # Fallback for unknown types
                        for field in ["summary", "details", "content", "caption", "name"]:
                            if entry.get(field):
                                content_parts.append(str(entry[field]))
                                break

                    content = " | ".join(content_parts) if content_parts else str(entry)

                    # Truncate individual memory items
                    content = self._truncate_to_tokens(content, self.query_memory_item_tokens)

                    memories.append({
                        "memory": content,
                        "type": memory_type,
                        "id": entry.get("id", ""),
                        "raw": entry,
                    })

            # Truncate to retrieve_num to ensure we don't return more than configured
            if len(memories) > self.retrieve_num:
                logger.debug(f"[MIRIXAgent] Truncating memories from {len(memories)} to {self.retrieve_num}")
                memories = memories[:self.retrieve_num]

            return memories

        except Exception as e:
            logger.error(f"[MIRIXAgent] Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("MIRIX native retrieval failed") from e

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query the agent with memory-augmented response.

        If use_native_query is True, uses MIRIX's native send_message() which
        automatically retrieves relevant memories and generates response.
        Otherwise, falls back to manual retrieval + external LLM call.
        """
        start_time = time.time()

        if self.use_native_query:
            return self._query_native(question, system_message, start_time)
        else:
            return self._query_manual(question, system_message, start_time)

    def retrieve(self, question: str, **kwargs) -> MemoryRetrievalResult:
        """Return MIRIX multi-store retrieval without native/external answering."""
        memories = self._retrieve(question)
        formatted = [
            {
                "memory": str(item.get("memory", "")),
                "type": item.get("type", "unknown"),
                **({"score": item["score"]} if "score" in item else {}),
            }
            for item in memories
            if item.get("memory")
        ]
        context = "\n".join(
            f"[{item['type']}] {item['memory']}" for item in formatted
        )
        return MemoryRetrievalResult(
            memory_context=context,
            retrieved_count=len(formatted),
            retrieved_memories=formatted,
            extra={
                "method": "mirix",
                "memory_types_retrieved": sorted({item["type"] for item in formatted}),
            },
        )

    def _query_native(
        self,
        question: str,
        system_message: Optional[str],
        start_time: float
    ) -> AgentResponse:
        """Query using MIRIX native send_message (recommended).

        This leverages MIRIX's built-in memory retrieval and conversation
        capabilities, providing a more integrated experience.
        """
        # Truncate question
        full_question = self._truncate_to_tokens(question, self.max_question_tokens)

        # Add timestamp for context
        query_with_time = f"{full_question}\n\nCurrent Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"

        # If there's a system message, prepend it as context
        if system_message:
            query_with_context = f"[System Context: {system_message}]\n\n{query_with_time}"
        else:
            query_with_context = query_with_time

        response_content = ""
        retrieved_memories = []
        query_usage = {"input": 0, "output": 0}

        async def do_query():
            nonlocal response_content, retrieved_memories, query_usage

            # Use MIRIX native send_message - it will automatically retrieve
            # relevant memories and use them in the response
            # NOTE: We don't pass user_id - MIRIX uses LocalClient's default user
            response = await self._client.send_message(
                agent_id=self._meta_agent.id,
                role="user",
                message=query_with_context,
            )

            # Extract response content and usage
            if response:
                # Record usage to global tracker
                if hasattr(response, 'usage') and response.usage:
                    self._record_mirix_usage(response.usage, time.time() - start_time)
                    query_usage["input"] = getattr(response.usage, 'prompt_tokens', 0)
                    query_usage["output"] = getattr(response.usage, 'completion_tokens', 0)

                # Extract assistant response
                if hasattr(response, 'messages'):
                    for msg in response.messages:
                        msg_type = getattr(msg, 'message_type', '')
                        if msg_type == "assistant_message":
                            response_content = getattr(msg, 'message', '')
                            break

                    # Also extract any function calls that retrieved memories
                    for msg in response.messages:
                        msg_type = getattr(msg, 'message_type', '')
                        if msg_type == "function_return":
                            func_return = getattr(msg, 'function_return', '')
                            # Try to extract memory information from function returns
                            if isinstance(func_return, str) and len(func_return) < 1000:
                                retrieved_memories.append({
                                    "memory": func_return[:300],
                                    "type": "function_return",
                                })

        self._run_async(do_query())

        query_time = time.time() - start_time

        return AgentResponse(
            output=response_content,
            query_time=query_time,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=retrieved_memories[:5],  # Properly set field, limit count for log size
            extra={
                "method": "mirix_native",
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_provider,
                "tokens_used": query_usage,
            }
        )

    def _query_manual(
        self,
        question: str,
        system_message: Optional[str],
        start_time: float
    ) -> AgentResponse:
        """Query using manual retrieval + external LLM (fallback).

        This retrieves memories via MIRIX's retrieve_memory() API,
        then generates response using llm_client (ensuring token tracking).
        """
        # Retrieve relevant memories
        retrieved_memories = self._retrieve(question)

        # Truncate question
        full_question = self._truncate_to_tokens(question, self.max_question_tokens)

        # Add timestamp for context
        full_question = f"{full_question}\n\nCurrent Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"

        # Calculate token budget for memory context
        base_system = system_message or ""
        reserved_tokens = self.max_tokens + 400  # Reserve for output and overhead
        available_tokens = max(self.max_context_tokens - reserved_tokens, 0)
        question_tokens = self._llm_client.count_tokens(full_question)
        base_system_tokens = self._llm_client.count_tokens(base_system) if base_system else 0
        memory_budget = max(available_tokens - question_tokens - base_system_tokens, 0)
        memory_budget = min(memory_budget, self.query_memory_context_tokens)

        # Build memory context with truncation
        memory_lines = []
        used_tokens = 0

        for entry in retrieved_memories:
            memory_text = entry.get("memory", "")
            if not memory_text:
                continue

            line = f"[{entry.get('type', 'memory')}] {memory_text}"
            line_tokens = self._llm_client.count_tokens(line)

            if used_tokens + line_tokens <= memory_budget:
                memory_lines.append(line)
                used_tokens += line_tokens
            else:
                # Try to fit truncated version
                remaining = memory_budget - used_tokens
                if remaining > 50:
                    truncated = self._truncate_to_tokens(line, remaining)
                    memory_lines.append(truncated)
                break

        # Build final prompt with memories
        if memory_lines:
            memories_str = "\n".join(memory_lines)
            memory_prompt = f"Relevant memories from MIRIX:\n{memories_str}\n\nBased on the above memories, please answer the question."

            if system_message:
                full_system = f"{system_message}\n\n{memory_prompt}"
            else:
                full_system = memory_prompt
        else:
            full_system = system_message

        # Call LLM for response (uses llm_client which tracks tokens)
        messages = format_messages(full_question, full_system)
        response = self._llm_client.chat(messages)

        query_time = time.time() - start_time

        # Build retrieved_memories format for evaluation framework
        formatted_memories = [
            {
                "memory": m["memory"][:500],
                "type": m.get("type", "unknown"),
            }
            for m in retrieved_memories
        ]

        return AgentResponse(
            output=response.content,
            query_time=query_time,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=formatted_memories,  # Properly set field
            extra={
                "method": "mirix_manual",
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_provider,
                "memory_types_retrieved": list(set(m.get("type", "unknown") for m in retrieved_memories)),
                "tokens_used": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                },
            }
        )

    def reset(self) -> None:
        """Reset agent state and release MIRIX resources.

        This performs a thorough cleanup for both PostgreSQL and SQLite:
        1. Hard delete all memory data via MIRIX API (works for both database types)
        2. Delete all agents created by the current client
        3. Additionally clean up SQLite files if they exist (for SQLite mode)

        The evaluator discards the manager immediately after reset and constructs
        a new one for the next persona, so recreating a client here only leaked an
        empty MetaAgent and kept the process alive.  Initialization remains in the
        constructor of the next manager.
        """
        logger.info("[MIRIXAgent] Resetting agent state (full database cleanup)...")
        super().reset()

        # Reset token counters
        self._mirix_input_tokens = 0
        self._mirix_output_tokens = 0
        self._mirix_step_count = 0

        # Step 1: Clean up via MIRIX API (works for both PostgreSQL and SQLite)
        if self._client:
            try:
                async def cleanup_all_data():
                    """Clean up all memory data and agents for the current client."""
                    client_id = self._client.client_id

                    # Hard delete ALL memory data using client_manager
                    # This removes: episodic, semantic, procedural, resource, knowledge vault,
                    # messages, and blocks - ensuring complete persona isolation
                    try:
                        logger.info(f"[MIRIXAgent] Deleting all memories for client {client_id}...")
                        await self._client.server.client_manager.delete_memories_by_client_id(client_id)
                        logger.info(f"[MIRIXAgent] Successfully deleted all memories for client {client_id}")
                    except Exception as e:
                        logger.warning(f"[MIRIXAgent] Failed to bulk delete memories: {e}")
                        # Fallback: try to delete individual memory types
                        await fallback_cleanup_memories()

                    # Delete the MetaAgent and all its sub-agents
                    if self._meta_agent:
                        try:
                            await self._client.delete_agent(self._meta_agent.id)
                            logger.info(f"[MIRIXAgent] Deleted MetaAgent: {self._meta_agent.id}")
                        except Exception as e:
                            logger.warning(f"[MIRIXAgent] Failed to delete MetaAgent: {e}")

                    # Clean up any remaining orphaned agents
                    try:
                        agents = await self._client.list_agents()
                        for agent in agents:
                            try:
                                await self._client.delete_agent(agent.id)
                                logger.debug(f"[MIRIXAgent] Cleaned up orphaned agent: {agent.id}")
                            except Exception as e:
                                logger.debug(f"[MIRIXAgent] Failed to cleanup agent {agent.id}: {e}")
                        if agents:
                            logger.info(f"[MIRIXAgent] Cleaned up {len(agents)} orphaned agents")
                    except Exception as e:
                        logger.debug(f"[MIRIXAgent] Failed to list agents for cleanup: {e}")

                async def fallback_cleanup_memories():
                    """Fallback method to clean up memories if bulk delete fails."""
                    try:
                        from mirix.services.episodic_memory_manager import EpisodicMemoryManager
                        from mirix.services.semantic_memory_manager import SemanticMemoryManager
                        from mirix.services.procedural_memory_manager import ProceduralMemoryManager
                        from mirix.services.resource_memory_manager import ResourceMemoryManager
                        from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
                        from mirix.services.message_manager import MessageManager

                        client = self._client.client

                        # Delete each memory type
                        managers = [
                            ("episodic", EpisodicMemoryManager()),
                            ("semantic", SemanticMemoryManager()),
                            ("procedural", ProceduralMemoryManager()),
                            ("resource", ResourceMemoryManager()),
                            ("knowledge_vault", KnowledgeVaultManager()),
                            ("messages", MessageManager()),
                        ]

                        for name, manager in managers:
                            try:
                                count = await manager.delete_by_client_id(actor=client)
                                logger.debug(f"[MIRIXAgent] Fallback deleted {count} {name} records")
                            except Exception as e:
                                logger.warning(f"[MIRIXAgent] Failed to delete {name}: {e}")

                    except Exception as e:
                        logger.error(f"[MIRIXAgent] Fallback cleanup failed: {e}")

                self._run_async(cleanup_all_data())

            except Exception as e:
                logger.warning(f"[MIRIXAgent] API-based data cleanup failed: {e}")
                import traceback
                traceback.print_exc()

        self._meta_agent = None
        self._client = None
        self._is_client_owner = False

        # Force garbage collection to release all references
        gc.collect()

        # Brief pause to allow database operations to settle
        time.sleep(0.3)

        # MIRIX keeps its SQLAlchemy async engine at module scope.  Dropping the
        # LocalClient is therefore not enough to release pooled SQLite handles.
        # Deleting sqlite.db while those handles are alive makes the subsequent
        # reinitialization reuse a connection to an unlinked/read-only file.
        # Dispose the pool first; SQLAlchemy will reconnect on the next use.
        try:
            from mirix.server import server as mirix_server

            engine = getattr(mirix_server, "engine", None)
            if engine is not None:
                self._run_async(engine.dispose())
                logger.info("[MIRIXAgent] Disposed database connection pool")
        except Exception as e:
            logger.warning(f"[MIRIXAgent] Failed to dispose database pool: {e}")

        # Step 2: Clean up SQLite database files if they exist (for SQLite mode)
        # This is a safety measure to ensure complete cleanup regardless of database type
        self._cleanup_sqlite_files()

        # Force garbage collection again after file cleanup
        gc.collect()

        logger.info("[MIRIXAgent] Reset complete - all memory data cleared for persona isolation")

    def _cleanup_sqlite_files(self) -> None:
        """Clean up SQLite database files if they exist.

        This handles the case where MIRIX is configured to use SQLite instead of PostgreSQL.
        It safely removes the SQLite database and all associated journal/WAL files.
        """
        try:
            from pathlib import Path
            mirix_dir = Path(os.environ.get("MIRIX_DIR", str(Path.home() / ".mirix")))
            sqlite_db_path = mirix_dir / "sqlite.db"

            if sqlite_db_path.exists():
                import os
                # Delete SQLite database and all associated files (journal, WAL, SHM)
                for suffix in ['', '-journal', '-wal', '-shm']:
                    db_file = Path(str(sqlite_db_path) + suffix)
                    if db_file.exists():
                        try:
                            os.remove(db_file)
                            logger.info(f"[MIRIXAgent] Deleted SQLite file: {db_file}")
                        except Exception as e:
                            logger.warning(f"[MIRIXAgent] Failed to delete {db_file}: {e}")
            else:
                logger.debug("[MIRIXAgent] No SQLite database found (using PostgreSQL mode)")

        except Exception as e:
            logger.warning(f"[MIRIXAgent] SQLite cleanup error: {e}")

    def cleanup(self) -> None:
        """Full cleanup of MIRIX resources (call when completely done with agent).

        This should be called when the agent is no longer needed, to release
        all database connections and resources.
        """
        logger.info("[MIRIXAgent] Performing full cleanup...")

        # Delete MetaAgent and all agents
        if self._meta_agent and self._client:
            try:
                async def full_cleanup():
                    # Delete all agents
                    agents = await self._client.list_agents()
                    for agent in agents:
                        try:
                            await self._client.delete_agent(agent.id)
                        except Exception:
                            pass

                self._run_async(full_cleanup())
            except Exception as e:
                logger.warning(f"[MIRIXAgent] Cleanup error: {e}")

        self._meta_agent = None
        self._client = None
        self._is_client_owner = False

        # Force garbage collection
        gc.collect()

        logger.info("[MIRIXAgent] Full cleanup complete")

    def set_context_id(self, context_id: int) -> None:
        """Set context ID for distinguishing personas.

        Note: MIRIX uses the LocalClient's default user for all operations,
        so we don't need to pass user_id to MIRIX APIs. The context_id is
        used for logging and internal tracking purposes.
        """
        old_context = self._context_id
        super().set_context_id(context_id)

        # If context changed, log it for debugging
        if old_context != context_id and old_context is not None:
            logger.info(f"[MIRIXAgent] Context changed from {old_context} to {context_id}")

    def get_info(self) -> Dict[str, Any]:
        """Get agent info."""
        info = super().get_info()
        info.update({
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model_path": self.embedding_model_path,
            "embedding_dim": self.embedding_dim,
            "embedding_device": self.embedding_device,
            "retrieve_num": self.retrieve_num,
            "memorize_chunk_tokens": self.memorize_chunk_tokens,
            "memorize_chunk_overlap_tokens": self.memorize_chunk_overlap_tokens,
            "query_memory_item_tokens": self.query_memory_item_tokens,
            "query_memory_context_tokens": self.query_memory_context_tokens,
            "max_input_tokens": self.max_input_tokens,
            "core_memory_max_tokens": self.core_memory_max_tokens,
            "core_json_tool_bridge": self.core_json_tool_bridge,
            "json_tool_bridge_attempts": self.json_tool_bridge_attempts,
            "normalize_missing_update_ids": self.normalize_missing_update_ids,
            "max_question_tokens": self.max_question_tokens,
            "max_context_tokens": self.max_context_tokens,
            "use_native_query": self.use_native_query,
            "meta_agent_id": self._meta_agent.id if self._meta_agent else None,
            "mirix_input_tokens": self._mirix_input_tokens,
            "mirix_output_tokens": self._mirix_output_tokens,
            "mirix_step_count": self._mirix_step_count,
        })
        return info

    @property
    def memory_count(self) -> int:
        """Get memory count."""
        return len(self._memory_chunks)

    def __del__(self):
        """Destructor - attempt cleanup on garbage collection."""
        try:
            if hasattr(self, '_is_client_owner') and self._is_client_owner:
                self.cleanup()
        except Exception:
            pass  # Ignore errors during destruction
