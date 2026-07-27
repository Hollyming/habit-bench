"""MemRL Agent - Value-driven procedural memory with Q-learning based retrieval."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, MemoryBuildResult, AgentResponse, MemoryRetrievalResult
from utils.llm_client import (
    create_llm_client,
    format_messages,
    BaseLLMClient,
    get_usage_tracker,
    LLMResponse,
)

logger = logging.getLogger(__name__)

class TrackedLLMProvider:
    """LLM provider that wraps MedMemoryBench's llm_client for accurate token tracking.

    This class implements the interface expected by MemRL's MemoryService
    (BaseLLM from memrl.providers.base) while using our tracked llm_client
    internally to ensure all API calls are properly recorded.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        model_name: str,
    ):
        """Initialize tracked LLM provider.

        Args:
            llm_client: MedMemoryBench's LLM client with built-in tracking
            model_name: Model name for reference
        """
        self._client = llm_client
        self._model_name = model_name

        # Statistics for reporting
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_latency = 0.0

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Generate response using the tracked LLM client.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        # Extract parameters
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")

        # Call tracked client
        response = self._client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Update local statistics
        self._call_count += 1
        self._total_input_tokens += response.input_tokens
        self._total_output_tokens += response.output_tokens
        self._total_latency += response.latency

        logger.debug(
            f"[TrackedLLM] call={self._call_count}, "
            f"input_tokens={response.input_tokens}, output_tokens={response.output_tokens}, "
            f"latency={response.latency:.2f}s"
        )

        return response.content

    def extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        """Extract keywords from text using LLM.

        Args:
            text: Input text to analyze
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of extracted keywords
        """
        prompt = f"""Extract up to {max_keywords} key concepts or keywords from the following text.
Focus on the most important nouns, actions, and specific entities.
Return only the keywords separated by commas, nothing else.

Text: {text}

Keywords:"""

        messages = [{"role": "user", "content": prompt}]
        response = self.generate(messages, temperature=0, max_tokens=100)

        # Parse keywords from response
        keywords_text = response.strip()
        keywords = []
        for keyword in keywords_text.split(','):
            keyword = keyword.strip().lower()
            # Remove quotes
            keyword = keyword.strip('"\'')
            if keyword and len(keyword) > 1:
                keywords.append(keyword)

        return keywords[:max_keywords]

    def generate_script(self, trajectory: str) -> str:
        """Generate high-level script from trajectory.

        Args:
            trajectory: Detailed task trajectory

        Returns:
            High-level script representation
        """
        prompt = f"""Analyze the following detailed task trajectory and create a concise,
high-level script that captures the essential steps and decision points.

The script should be:
1. Generic enough to apply to similar tasks
2. Specific enough to provide useful guidance
3. 3-5 high-level steps maximum
4. Focus on the strategy and key decisions, not detailed actions

Trajectory:
{trajectory}

High-level script:"""

        messages = [{"role": "user", "content": prompt}]
        return self.generate(messages, temperature=0.7, max_tokens=500)

    def get_stats(self) -> Dict[str, Any]:
        """Get cumulative usage statistics."""
        return {
            "call_count": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_latency": round(self._total_latency, 3),
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_latency = 0.0

@dataclass
class MemoryBuildProgress:
    """Tracks memory building progress and statistics."""
    total_chunks: int = 0
    processed_chunks: int = 0
    successful_builds: int = 0
    failed_builds: int = 0
    total_memories_created: int = 0
    start_time: float = 0.0
    chunk_timings: List[float] = field(default_factory=list)
    build_logs: List[Dict[str, Any]] = field(default_factory=list)

    def start(self, total_chunks: int) -> None:
        """Start progress tracking."""
        self.total_chunks = total_chunks
        self.processed_chunks = 0
        self.successful_builds = 0
        self.failed_builds = 0
        self.total_memories_created = 0
        self.start_time = time.time()
        self.chunk_timings = []
        self.build_logs = []

    def record_chunk(
        self,
        chunk_index: int,
        success: bool,
        memories_created: int,
        chunk_time: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a chunk processing result."""
        self.processed_chunks += 1
        self.chunk_timings.append(chunk_time)

        if success:
            self.successful_builds += 1
            self.total_memories_created += memories_created
        else:
            self.failed_builds += 1

        log_entry = {
            "chunk_index": chunk_index,
            "success": success,
            "memories_created": memories_created,
            "chunk_time": round(chunk_time, 3),
            "timestamp": datetime.now().isoformat(),
        }
        if details:
            log_entry["details"] = details
        self.build_logs.append(log_entry)

        # Log progress
        elapsed = time.time() - self.start_time
        progress_pct = (self.processed_chunks / self.total_chunks) * 100
        avg_time = sum(self.chunk_timings) / len(self.chunk_timings) if self.chunk_timings else 0
        eta = avg_time * (self.total_chunks - self.processed_chunks)

        logger.info(
            f"[MemRL Build] Progress: {self.processed_chunks}/{self.total_chunks} "
            f"({progress_pct:.1f}%) | Memories: {self.total_memories_created} | "
            f"Success: {self.successful_builds} | Failed: {self.failed_builds} | ETA: {eta:.1f}s"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get build progress summary."""
        total_time = time.time() - self.start_time if self.start_time else 0
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "successful_builds": self.successful_builds,
            "failed_builds": self.failed_builds,
            "total_memories_created": self.total_memories_created,
            "total_time": round(total_time, 3),
            "avg_chunk_time": round(sum(self.chunk_timings) / len(self.chunk_timings), 3) if self.chunk_timings else 0,
            "build_logs": self.build_logs,
        }

class MemRLAgent(BaseAgent):
    """MemRL Agent with value-driven procedural memory.

    This agent uses MemRL's official MemoryService for memory management,
    integrating Q-learning based retrieval with ε-greedy exploration.

    Key features:
    - Build: Proceduralization strategy (trajectory + script)
    - Retrieve: Query-based similarity search with Q-value ranking
    - Update: Adjustment strategy with value updates
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
        # MemRL specific parameters
        retrieve_num: int = 5,
        candidate_top_k: int = 12,
        similarity_threshold: float = 0.2,
        utility_mix_lambda: float = 0.6,
        learning_rate: float = 0.2,
        initial_q: float = 0.5,
        max_new_items_per_memorize: int = 8,
        max_experience_tokens: int = 700,
        memorize_chunk_tokens: int = 6000,
        memorize_chunk_overlap_tokens: int = 200,
        query_memory_item_tokens: int = 300,
        query_memory_context_tokens: int = 1800,
        max_concurrent_api_calls: int = 4,
        # Embedding configuration
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "local",
        embedding_model_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        storage_root: Optional[str] = None,
        # Strategy configuration
        build_strategy: str = "proceduralization",
        retrieve_strategy: str = "query",
        update_strategy: str = "adjustment",
        # Q-learning configuration
        epsilon: float = 0.1,
        gamma: float = 0.0,
        q_init_pos: float = 0.5,
        q_init_neg: float = 0.0,
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
        weight_sim: float = 0.5,
        weight_q: float = 0.5,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        # Store configuration
        self.retrieve_num = retrieve_num
        self.candidate_top_k = candidate_top_k
        self.similarity_threshold = similarity_threshold
        self.utility_mix_lambda = utility_mix_lambda
        self.learning_rate = learning_rate
        self.initial_q = initial_q
        self.max_new_items_per_memorize = max_new_items_per_memorize
        self.max_experience_tokens = max_experience_tokens
        self.memorize_chunk_tokens = memorize_chunk_tokens
        self.memorize_chunk_overlap_tokens = memorize_chunk_overlap_tokens
        self.query_memory_item_tokens = query_memory_item_tokens
        self.query_memory_context_tokens = query_memory_context_tokens
        self.max_concurrent_api_calls = max_concurrent_api_calls

        # Token limits
        self.max_input_tokens = int(kwargs.get("max_input_tokens", 8000))
        self.max_context_tokens = int(kwargs.get("max_context_tokens", 120000))
        self.max_question_tokens = int(kwargs.get("max_question_tokens", 4096))

        # Embedding configuration
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = int(embedding_dim) if embedding_dim is not None else None
        self.storage_root = str(Path(storage_root).resolve()) if storage_root else None

        # Strategy configuration
        self.build_strategy = build_strategy
        self.retrieve_strategy = retrieve_strategy
        self.update_strategy = update_strategy

        # Q-learning configuration
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_init_pos = q_init_pos
        self.q_init_neg = q_init_neg
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.weight_sim = weight_sim
        self.weight_q = weight_q

        # API configuration
        self._api_key = (
            api_key
            or os.environ.get("BIGMODEL_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self._base_url = (
            base_url
            or os.environ.get("BIGMODEL_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or self.BIGMODEL_BASE_URL
        )
        self._provider = provider

        # Create LLM client for all API calls (with token tracking)
        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # Progress tracker
        self._build_progress = MemoryBuildProgress()

        # MemRL components (lazy initialized)
        self._memory_service = None
        self._tracked_llm = None
        self._memrl_embedder = None
        self._mos_config_path = None
        self._temp_dir = None

        # Initialize MemRL
        self._init_memrl()

    def _load_memrl_modules(self) -> None:
        """Load vendored MemRL package from methods/MemRL."""
        memrl_root = Path(__file__).resolve().parent / "MemRL"
        if not memrl_root.exists():
            raise ImportError("MemRL source folder not found at methods/MemRL")

        memrl_root_str = str(memrl_root)
        if memrl_root_str not in sys.path:
            sys.path.insert(0, memrl_root_str)

        # Also add MemOS path for compatibility
        memos_src = Path(__file__).resolve().parent / "memOS" / "MemOS" / "src"
        if memos_src.exists():
            memos_src_str = str(memos_src)
            if memos_src_str not in sys.path:
                sys.path.insert(0, memos_src_str)

    def _create_mos_config(self) -> str:
        """Create a temporary MemOS configuration file."""
        self._temp_dir = tempfile.mkdtemp(prefix="memrl_agent_")

        # Embedder config based on provider
        if self.embedding_provider == "local":
            embedder_config = {
                "backend": "sentence_transformer",
                "config": {
                    "model_name_or_path": self.embedding_model_path or self.embedding_model,
                }
            }
        else:
            embedder_config = {
                "backend": "universal_api",
                "config": {
                    "model_name_or_path": self.embedding_model_path or self.embedding_model,
                    "provider": "openai",
                    "api_key": self._api_key,
                    "base_url": self._base_url,
                }
            }

        config = {
            "user_manager": {
                "backend": "sqlite",
                "config": {
                    "db_path": os.path.join(self._temp_dir, "users.db")
                }
            },
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": self.model,
                    "api_key": self._api_key,
                    "api_base": self._base_url,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": self.model,
                            "api_key": self._api_key,
                            "api_base": self._base_url,
                            "temperature": 0.0,
                            "max_tokens": 4096,
                        }
                    },
                    "embedder": embedder_config,
                    "chunker": {
                        "backend": "sentence",
                        "config": {
                            "chunk_size": 512,
                            "chunk_overlap": 128,
                        }
                    }
                }
            }
        }

        config_path = os.path.join(self._temp_dir, "mos_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        return config_path

    def _init_memrl(self) -> None:
        """Initialize MemRL components."""
        logger.info("[MemRLAgent] Initializing MemRL components...")

        # Load MemRL modules
        self._load_memrl_modules()

        # Import MemRL components
        from memrl.providers.embedding import LocalEmbedder, OpenAIEmbedder
        from memrl.service.memory_service import MemoryService
        from memrl.service.strategies import StrategyConfiguration
        from memrl.service.value_driven import RLConfig

        # Create MemOS config
        self._mos_config_path = self._create_mos_config()

        # Create tracked LLM provider using our llm_client
        self._tracked_llm = TrackedLLMProvider(
            llm_client=self._llm_client,
            model_name=self.model,
        )

        # Initialize embedding provider
        if self.embedding_provider == "local":
            self._memrl_embedder = LocalEmbedder(
                model_name=self.embedding_model_path or self.embedding_model,
            )
        else:
            self._memrl_embedder = OpenAIEmbedder(
                api_key=self._api_key,
                base_url=self._base_url,
                model=self.embedding_model,
                token_log_dir=self._temp_dir,
            )

        # Create strategy configuration
        strategy_config = StrategyConfiguration.from_strings(
            build=self.build_strategy,
            retrieve=self.retrieve_strategy,
            update=self.update_strategy,
        )

        # Create RL configuration
        rl_config = RLConfig(
            epsilon=self.epsilon,
            alpha=self.learning_rate,
            gamma=self.gamma,
            q_init_pos=self.q_init_pos,
            q_init_neg=self.q_init_neg,
            success_reward=self.success_reward,
            failure_reward=self.failure_reward,
            sim_threshold=self.similarity_threshold,
            topk=self.candidate_top_k,
            weight_sim=self.weight_sim,
            weight_q=self.weight_q,
        )

        # Create unique user ID for this agent
        user_id = (
            f"memrl_user_{self._context_id}"
            if self._context_id is not None
            else f"memrl_user_{uuid.uuid4().hex[:8]}"
        )

        # Initialize MemoryService with our tracked LLM
        self._memory_service = MemoryService(
            mos_config_path=self._mos_config_path,
            llm_provider=self._tracked_llm,
            embedding_provider=self._memrl_embedder,
            strategy_config=strategy_config,
            user_id=user_id,
            num_workers=self.max_concurrent_api_calls,
            enable_value_driven=True,
            rl_config=rl_config,
            add_similarity_threshold=0.9,
            storage_root=self.storage_root,
            embedding_dim=self.embedding_dim,
        )

        logger.info(f"[MemRLAgent] MemRL initialized successfully")
        logger.info(f"  - User ID: {user_id}")
        logger.info(f"  - Strategy: {strategy_config}")
        logger.info(f"  - Embedding: {self.embedding_provider}/{self.embedding_model}")

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max_tokens."""
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    def _split_text_into_chunks(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 0
    ) -> List[str]:
        """Split text into chunks with optional overlap."""
        if not text.strip():
            return []

        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return [text]

        chunks: List[str] = []
        start = 0
        step = max(1, max_tokens - overlap_tokens)

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._tokenizer.decode(chunk_tokens))

            if end >= len(tokens):
                break
            start += step

        return chunks

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Store text into MemRL memory using official build strategy.

        This method:
        1. Splits input text into manageable chunks
        2. Uses MemRL's build_memory for each chunk (following official implementation)
        3. Tracks progress and provides detailed logging
        4. Records all token usage through the tracked LLM provider
        """
        start_time = time.time()

        # Split text into chunks for processing
        chunks = self._split_text_into_chunks(
            text,
            max_tokens=self.memorize_chunk_tokens,
            overlap_tokens=self.memorize_chunk_overlap_tokens,
        )

        if not chunks:
            return MemoryBuildResult(
                success=False,
                method="memrl",
                action="memorize",
                input_content=text,
                stored_content="",
                memory_entries=[],
                chunk_count=0,
                extra={"error": "No chunks to process"},
            )

        # Start progress tracking
        self._build_progress.start(len(chunks))
        logger.info(f"[MemRL Memory Build] Starting memorization of {len(chunks)} chunks...")

        memory_entries: List[Dict[str, Any]] = []
        all_memory_ids: List[str] = []
        build_details: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            chunk_start = time.time()

            try:
                # Use MemRL's official build_memory method
                # Pass the full chunk as both task_description and trajectory
                # (following official implementation without artificial truncation)
                memory_id = self._memory_service.build_memory(
                    task_description=chunk,  # Full chunk as task description
                    trajectory=chunk,  # Full chunk as trajectory
                    metadata={
                        "source_benchmark": "medmemorybench",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        # Don't assume success - let MemRL handle Q-value initialization
                    }
                )

                chunk_time = time.time() - chunk_start

                # Record entry
                memory_entries.append({
                    "event": "BUILD",
                    "memory_id": memory_id,
                    "chunk_index": i,
                    "content_preview": chunk[:200],
                })
                all_memory_ids.append(memory_id)

                # Record progress
                self._build_progress.record_chunk(
                    chunk_index=i,
                    success=True,
                    memories_created=1,
                    chunk_time=chunk_time,
                    details={"memory_id": memory_id},
                )

                build_details.append({
                    "chunk_index": i,
                    "success": True,
                    "memory_id": memory_id,
                    "chunk_time": round(chunk_time, 3),
                })

            except Exception as e:
                chunk_time = time.time() - chunk_start
                logger.error(f"[MemRL] Failed to build memory for chunk {i}: {e}")

                self._build_progress.record_chunk(
                    chunk_index=i,
                    success=False,
                    memories_created=0,
                    chunk_time=chunk_time,
                    details={"error": str(e)},
                )

                build_details.append({
                    "chunk_index": i,
                    "success": False,
                    "error": str(e),
                    "chunk_time": round(chunk_time, 3),
                })

        total_time = time.time() - start_time
        progress_summary = self._build_progress.get_summary()
        build_succeeded = (
            progress_summary["successful_builds"] == len(chunks)
            and progress_summary["failed_builds"] == 0
        )

        # Update internal state
        if build_succeeded:
            self._memory_chunks.append(text)
            self._is_initialized = True

        # Get LLM usage stats from tracked provider
        llm_stats = self._tracked_llm.get_stats() if self._tracked_llm else {}

        return MemoryBuildResult(
            success=build_succeeded,
            method="memrl",
            action="build_memory",
            input_content=text,
            stored_content=text,
            memory_entries=memory_entries[:10],  # Limit for display
            chunk_count=len(chunks),
            time_cost=total_time,
            extraction_result=f"Built {progress_summary['total_memories_created']} memories from {len(chunks)} chunks",
            all_passages=[{"memory_id": mid} for mid in all_memory_ids],
            extra={
                "error": (
                    None
                    if build_succeeded
                    else f"{progress_summary['failed_builds']} of {len(chunks)} chunks failed"
                ),
                "context_id": self._context_id,
                "total_memories": progress_summary["total_memories_created"],
                "successful_chunks": progress_summary["successful_builds"],
                "failed_chunks": progress_summary["failed_builds"],
                "build_details": build_details,
                "llm_stats": llm_stats,
                "strategy": {
                    "build": self.build_strategy,
                    "retrieve": self.retrieve_strategy,
                    "update": self.update_strategy,
                },
            },
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query using MemRL's value-driven retrieval.

        This method:
        1. Uses MemRL's retrieve_query for value-aware memory retrieval
        2. Applies Q-value based ranking with ε-greedy exploration
        3. Constructs prompt with retrieved memories
        4. Generates response using the tracked LLM client
        """
        start_time = time.time()

        # Truncate question if needed
        bounded_question = self._truncate_to_tokens(question, self.max_question_tokens)

        # Retrieve memories using MemRL's value-driven retrieval
        retrieved_memories: List[Dict[str, Any]] = []
        retrieval_result: Dict[str, Any] = {}

        try:
            # Use MemRL's retrieve_query which combines similarity and Q-value
            result = self._memory_service.retrieve_query(
                task_description=bounded_question,
                k=self.candidate_top_k,
                threshold=self.similarity_threshold,
            )

            # Current vendored MemRL returns the result dict directly.  Older
            # runner code also accepts a legacy ``(dict, diagnostics)`` tuple,
            # so keep the adapter compatible with both source revisions.
            retrieval_result = result[0] if isinstance(result, tuple) else result
            if not isinstance(retrieval_result, dict):
                raise TypeError(
                    "MemRL retrieve_query must return a dict or (dict, diagnostics) tuple"
                )

            # Extract selected memories from result
            selected = retrieval_result.get("selected", [])
            for mem in selected[:self.retrieve_num]:
                # Extract content from memory object
                content = self._extract_memory_content(mem)

                retrieved_memories.append({
                    "memory": content[:self.query_memory_item_tokens * 4],  # Rough char limit
                    "memory_id": mem.get("memory_id"),
                    "similarity": mem.get("similarity", 0.0),
                    "q_value": mem.get("q_estimate", self.initial_q),
                    "score": mem.get("score", 0.0),
                    "type": "memrl_value_driven",
                })

        except Exception as e:
            logger.error(f"[MemRL] Retrieval failed: {e}")
            raise RuntimeError("MemRL value-aware retrieval failed") from e

        retrieval_time = time.time() - start_time

        # Calculate token budget for memory context
        system_tokens = self._llm_client.count_tokens(system_message) if system_message else 0
        reserved_tokens = self.max_tokens + 500
        available_tokens = max(self.max_context_tokens - reserved_tokens - system_tokens, 0)
        question_tokens = self._llm_client.count_tokens(bounded_question)
        memory_budget = min(
            max(available_tokens - question_tokens, 0),
            self.query_memory_context_tokens
        )

        # Build memory context with truncation
        memory_blocks: List[str] = []
        used_tokens = 0

        for idx, mem in enumerate(retrieved_memories):
            mem_text = str(mem.get("memory", "")).strip()
            if not mem_text:
                continue

            mem_tokens = self._llm_client.count_tokens(mem_text)
            if used_tokens + mem_tokens > memory_budget:
                # Truncate this memory to fit
                remaining = memory_budget - used_tokens
                if remaining > 100:
                    truncated = self._truncate_to_tokens(mem_text, remaining)
                    memory_blocks.append(truncated)
                    retrieved_memories[idx]["memory"] = truncated[:2000]
                    retrieved_memories[idx]["truncated"] = True
                break

            memory_blocks.append(mem_text)
            used_tokens += mem_tokens

        # Build final prompt
        full_question = bounded_question
        if memory_blocks:
            memory_context = "\n\n".join([
                f"[Memory {idx + 1} (Q={retrieved_memories[idx].get('q_value', 0):.2f}, "
                f"sim={retrieved_memories[idx].get('similarity', 0):.2f})]\n{block}"
                for idx, block in enumerate(memory_blocks)
            ])
            full_question = f"[Retrieved Memories from MemRL]\n{memory_context}\n\n[Question]\n{bounded_question}"

        # Generate response using tracked LLM client
        messages = format_messages(full_question, system_message)
        response = self._llm_client.chat(messages)

        total_time = time.time() - start_time

        # Get LLM stats from tracked provider
        memrl_llm_stats = self._tracked_llm.get_stats() if self._tracked_llm else {}

        # Format retrieved memories for response
        formatted_memories = [
            {
                "memory": m["memory"][:500] if m.get("memory") else "",
                "memory_id": m.get("memory_id"),
                "similarity": m.get("similarity", 0),
                "q_value": m.get("q_value", 0),
                "score": m.get("score", 0),
                "type": m.get("type"),
            }
            for m in retrieved_memories
        ]

        return AgentResponse(
            output=response.content,
            query_time=total_time,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=formatted_memories,
            extra={
                "method": "memrl",
                "retrieval_time": retrieval_time,
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_provider,
                "retrieval_stats": {
                    "candidates_count": len(retrieval_result.get("candidates", [])),
                    "simmax": retrieval_result.get("simmax", 0),
                },
                "strategy": {
                    "retrieve": self.retrieve_strategy,
                    "epsilon": self.epsilon,
                    "weight_sim": self.weight_sim,
                    "weight_q": self.weight_q,
                },
                "tokens_used": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                },
                "memrl_llm_stats": memrl_llm_stats,
            },
        )

    def retrieve(self, question: str, **kwargs) -> MemoryRetrievalResult:
        """Return MemRL value-aware selections without reader generation."""
        bounded_question = self._truncate_to_tokens(question, self.max_question_tokens)
        raw_result = self._memory_service.retrieve_query(
            task_description=bounded_question,
            k=self.candidate_top_k,
            threshold=self.similarity_threshold,
        )
        result = raw_result[0] if isinstance(raw_result, tuple) else raw_result
        if not isinstance(result, dict):
            raise TypeError(
                "MemRL retrieve_query must return a dict or (dict, diagnostics) tuple"
            )
        retrieved_memories: List[Dict[str, Any]] = []
        blocks: List[str] = []
        for memory in result.get("selected", [])[: self.retrieve_num]:
            raw_content = self._extract_memory_content(memory)
            content = self._truncate_to_tokens(
                raw_content, self.query_memory_item_tokens
            )
            prefix = (
                f"[Memory {len(blocks) + 1} "
                f"(Q={memory.get('q_estimate', self.initial_q):.2f}, "
                f"sim={memory.get('similarity', 0.0):.2f})]\n"
            )
            separator = "\n\n" if blocks else ""
            candidate = "\n\n".join(blocks) + separator + prefix + content
            if self._llm_client.count_tokens(candidate) > self.query_memory_context_tokens:
                prefix_context = "\n\n".join(blocks) + separator + prefix
                remaining = max(
                    self.query_memory_context_tokens
                    - self._llm_client.count_tokens(prefix_context),
                    0,
                )
                content = self._truncate_to_tokens(content, remaining)
                while content and self._llm_client.count_tokens(
                    prefix_context + content
                ) > self.query_memory_context_tokens:
                    remaining -= 1
                    content = self._truncate_to_tokens(content, remaining)
                if not content:
                    break
            entry = {
                "memory": content,
                "memory_id": memory.get("memory_id"),
                "similarity": memory.get("similarity", 0.0),
                "q_value": memory.get("q_estimate", self.initial_q),
                "score": memory.get("score", 0.0),
                "type": "memrl_value_driven",
                "truncated": content != raw_content,
            }
            retrieved_memories.append(entry)
            blocks.append(prefix + content)
        memory_context = "\n\n".join(blocks)
        return MemoryRetrievalResult(
            memory_context=memory_context,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=retrieved_memories,
            extra={
                "method": "memrl",
                "candidates_count": len(result.get("candidates", [])),
                "simmax": result.get("simmax", 0.0),
                "memory_context_tokens": self._llm_client.count_tokens(
                    memory_context
                ),
                "memory_context_token_budget": self.query_memory_context_tokens,
            },
        )

    def _extract_memory_content(self, mem: Dict[str, Any]) -> str:
        """Extract content from a memory object returned by retrieve_query.

        Args:
            mem: Memory dictionary from retrieval result

        Returns:
            Extracted content string
        """
        # Try direct content field first
        content = mem.get("content")
        if content:
            return str(content)

        # Try to get from metadata.full_content
        metadata = mem.get("metadata", {})

        # Handle TextualMemoryMetadata object
        if hasattr(metadata, "model_extra"):
            try:
                content = metadata.model_extra.get("full_content", "")
                if content:
                    return str(content)
            except Exception:
                pass

        # Handle dict-like metadata
        if isinstance(metadata, dict):
            content = metadata.get("full_content", "")
            if content:
                return str(content)

        # Try to get from memory_item
        memory_item = mem.get("memory_item")
        if memory_item:
            # Try memory attribute
            if hasattr(memory_item, "memory"):
                return str(memory_item.memory)
            # Try metadata.full_content
            if hasattr(memory_item, "metadata"):
                item_meta = memory_item.metadata
                if hasattr(item_meta, "model_extra"):
                    try:
                        return str(item_meta.model_extra.get("full_content", ""))
                    except Exception:
                        pass
                if isinstance(item_meta, dict):
                    return str(item_meta.get("full_content", ""))

        return ""

    def reset(self) -> None:
        """Reset agent and release all resources."""
        import gc
        import shutil

        super().reset()

        # Clean up MemRL components
        if self._memory_service:
            try:
                self._memory_service = None
            except Exception as e:
                logger.warning(f"[MemRLAgent] Warning: Failed to cleanup MemoryService: {e}")

        # Reset tracked LLM stats
        if self._tracked_llm:
            self._tracked_llm.reset_stats()
        self._tracked_llm = None
        self._memrl_embedder = None

        # Clean up temp directory
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
                logger.info(f"[MemRLAgent] Cleaned up temp directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"[MemRLAgent] Warning: Failed to clean up temp dir: {e}")
            self._temp_dir = None

        self._mos_config_path = None

        # Reset progress tracker
        self._build_progress = MemoryBuildProgress()

        # Force garbage collection
        gc.collect()

        # Re-initialize if needed for next context
        self._context_id = None

    def set_context_id(self, context_id: int) -> None:
        """Set context ID and reinitialize MemRL for new context."""
        old_context = self._context_id
        super().set_context_id(context_id)

        # If context changed, reinitialize MemRL with new user_id
        if old_context != context_id and self._memory_service is not None:
            logger.info(f"[MemRLAgent] Context changed from {old_context} to {context_id}, reinitializing...")
            self.reset()
            self._context_id = context_id
            self._init_memrl()

    def get_info(self) -> Dict[str, Any]:
        """Get agent info including MemRL specific details."""
        info = super().get_info()

        # Add MemRL specific info
        info.update({
            "memrl_config": {
                "retrieve_num": self.retrieve_num,
                "candidate_top_k": self.candidate_top_k,
                "similarity_threshold": self.similarity_threshold,
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "initial_q": self.initial_q,
                "weight_sim": self.weight_sim,
                "weight_q": self.weight_q,
            },
            "strategy": {
                "build": self.build_strategy,
                "retrieve": self.retrieve_strategy,
                "update": self.update_strategy,
            },
            "embedding": {
                "model": self.embedding_model,
                "provider": self.embedding_provider,
            },
        })

        # Add LLM stats if available
        if self._tracked_llm:
            info["memrl_llm_stats"] = self._tracked_llm.get_stats()

        return info
