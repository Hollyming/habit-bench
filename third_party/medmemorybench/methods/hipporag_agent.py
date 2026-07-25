"""HippoRAG agent adapter for MedMemoryBench."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import AgentResponse, BaseAgent, MemoryBuildResult
from utils.llm_client import (
    BaseLLMClient,
    LLMResponse,
    create_llm_client,
    format_messages,
    get_usage_tracker,
)

logger = logging.getLogger(__name__)


class TrackedLLMWrapper:

    def __init__(
        self,
        llm_client: BaseLLMClient,
        llm_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        seed: int = 0,
        **kwargs,
    ):
        self.llm_client = llm_client
        self.llm_name = llm_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.kwargs = kwargs

        # Emulate LLMConfig structure for HippoRAG internals
        self.llm_config = _LLMConfigProxy(
            llm_name=llm_name,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

    def infer(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Tuple[str, Dict, bool]:
        """Emulate CacheOpenAI.infer() interface, routing calls through llm_client for token tracking.

        Returns:
            Tuple of (response_content, metadata, cache_hit).
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_completion_tokens", kwargs.get("max_tokens", self.max_tokens))

        extra_kwargs = {}
        if "response_format" in kwargs:
            extra_kwargs["response_format"] = kwargs["response_format"]

        try:
            response = self.llm_client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra_kwargs,
            )

            content = response.content

            # Strip markdown fencing from JSON responses
            if kwargs.get("response_format", {}).get("type") == "json_object":
                if content.startswith("```json\n") and content.endswith("```"):
                    content = content[8:-3].strip()

            metadata = {
                "prompt_tokens": response.input_tokens,
                "completion_tokens": response.output_tokens,
                "finish_reason": "stop",
            }

            return content, metadata, False  # cache_hit=False

        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            return "", {"error": str(e), "prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "error"}, False


class _LLMConfigProxy:
    """Proxy that mimics HippoRAG's LLMConfig structure for internal compatibility."""

    def __init__(self, llm_name: str, temperature: float, max_tokens: int, seed: int):
        self.generate_params = {
            "model": llm_name,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "seed": seed,
            "n": 1,
        }


class TrackedEmbeddingWrapper:
    """Wraps embedding calls with the same interface as HippoRAG's BaseEmbeddingModel.

    Supports two modes: local (sentence-transformers) and API (OpenAI-compatible endpoint).
    """

    def __init__(
        self,
        provider: str = "local",
        model: str = "BAAI/bge-small-zh-v1.5",
        model_path: Optional[str] = None,
        dim: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 16,
        max_seq_len: int = 512,
        normalize: bool = True,
        dtype: str = "auto",
        **kwargs,
    ):
        self.provider = provider
        self.model_name = model
        self.embedding_model_name = model  # Attribute name expected by HippoRAG
        self.model_path = model_path or model
        self.dim = dim
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.dtype = dtype
        self.api_key = api_key
        self.base_url = base_url

        # Required by HippoRAG
        self.embedding_size = dim  # Fallback dimension for EmbeddingStore

        self._model = None
        self._openai_client = None
        self._initialized = False

    def _lazy_init(self):
        """Lazily initialize the model on first use."""
        if self._initialized:
            return

        if self.provider == "local":
            self._init_local_model()
        elif self.provider in ["openai", "api"]:
            self._init_api_client()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

        self._initialized = True

    def _init_local_model(self):
        """Initialize a local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local embedding model: {self.model_path}")
            self._model = SentenceTransformer(
                self.model_path,
                trust_remote_code=True,
            )

            if self.dim is None:
                self.dim = self._model.get_sentence_embedding_dimension()
            self.embedding_size = self.dim

            logger.info(f"Embedding model loaded, dim={self.dim}")

        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def _init_api_client(self):
        """Initialize an OpenAI-compatible API client for embeddings."""
        from openai import OpenAI

        self._openai_client = OpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
            base_url=self.base_url or os.environ.get("EMBEDDING_BASE_URL"),
            timeout=60,
        )
        logger.info(f"Embedding API client initialized, model={self.model_name}")

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts into embedding vectors.

        Returns:
            np.ndarray of shape (len(texts), dim).
        """
        self._lazy_init()

        if isinstance(texts, str):
            texts = [texts]

        texts = [t if t and t.strip() else "empty" for t in texts]

        # HippoRAG passes instruction to distinguish query vs document embeddings
        instruction = kwargs.get("instruction", "")
        if instruction:
            texts = [f"{instruction}{t}" for t in texts]

        if self.provider == "local":
            embeddings = self._encode_local(texts)
        else:
            embeddings = self._encode_api(texts)

        if self.normalize and kwargs.get("norm", True):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        return embeddings

    def _encode_local(self, texts: List[str]) -> np.ndarray:
        """Encode using the local sentence-transformers model."""
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,  # Normalization handled in encode()
        )
        return np.array(embeddings)

    def _encode_api(self, texts: List[str]) -> np.ndarray:
        """Encode using the OpenAI-compatible API."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = self._openai_client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Embedding API error: {e}")
                # Fall back to zero vectors
                dim = self.dim or 1536
                all_embeddings.extend([np.zeros(dim) for _ in batch])

        return np.array(all_embeddings)

    def batch_encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Batch encode (alias for encode, provided for interface compatibility)."""
        return self.encode(texts, **kwargs)

class HippoRAGAgent(BaseAgent):
    """HippoRAG adapter for the MedMemoryBench evaluation framework.

    Integrates HippoRAG 2 (OSU NLP Group) while routing all LLM calls
    through llm_client for token usage tracking.
    """

    METHOD_TYPE = "graph_rag"

    def __init__(
        self,
        # Base model config
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # HippoRAG-specific params
        openie_mode: str = "online",  # "online" | "offline"
        # Graph config
        is_directed_graph: bool = False,
        synonymy_edge_sim_threshold: float = 0.8,
        synonymy_edge_topk: int = 2047,
        # Retrieval config
        linking_top_k: int = 5,
        retrieval_top_k: int = 200,
        qa_top_k: int = 5,
        damping: float = 0.5,
        passage_node_weight: float = 0.05,
        # Cache config
        force_index_from_scratch: bool = False,
        force_openie_from_scratch: bool = False,
        save_openie: bool = True,
        # Embedding config
        embedding_provider: str = "local",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        embedding_model_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_batch_size: int = 16,
        embedding_max_seq_len: int = 512,
        # Chunking config
        chunk_size_tokens: int = 8000,
        chunk_overlap_tokens: int = 200,
        # Token limits
        max_input_tokens: int = 8000,
        max_context_tokens: int = 120000,
        working_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.provider = provider
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        self.openie_mode = openie_mode
        self.is_directed_graph = is_directed_graph
        self.synonymy_edge_sim_threshold = synonymy_edge_sim_threshold
        self.synonymy_edge_topk = synonymy_edge_topk
        self.linking_top_k = linking_top_k
        self.retrieval_top_k = retrieval_top_k
        self.qa_top_k = qa_top_k
        self.damping = damping
        self.passage_node_weight = passage_node_weight
        self.force_index_from_scratch = force_index_from_scratch
        self.force_openie_from_scratch = force_openie_from_scratch
        self.save_openie = save_openie

        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        self.embedding_batch_size = embedding_batch_size
        self.embedding_max_seq_len = embedding_max_seq_len

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        self.max_input_tokens = max_input_tokens
        self.max_context_tokens = max_context_tokens

        self.working_dir = working_dir

        self._llm_client = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # Per-context instance pool: {context_id: HippoRAG}
        self._hipporag_instances: Dict[int, Any] = {}
        # Accumulated session texts awaiting graph build: {context_id: [text, ...]}
        self._pending_sessions: Dict[int, List[str]] = {}
        self._session_counts: Dict[int, int] = {}
        # Tracks whether index() has run (used to detect new evaluation units)
        self._indexed_flags: Dict[int, bool] = {}

        self._setup_hipporag_path()
        self._hipporag_modules_loaded = False
        self._shared_embedding_model = None

    def _setup_hipporag_path(self):
        """Add HippoRAG src to sys.path and mock missing optional modules (vllm)."""
        hipporag_src = Path(__file__).resolve().parent / "HippoRAG" / "src"
        if not hipporag_src.exists():
            raise ImportError(f"HippoRAG source folder not found at {hipporag_src}")

        hipporag_src_str = str(hipporag_src)
        if hipporag_src_str not in sys.path:
            sys.path.insert(0, hipporag_src_str)

        # Mock vllm module if not installed (only needed for offline mode)
        if 'vllm' not in sys.modules:
            try:
                import vllm
            except ImportError:
                # Create mock vllm module
                import types
                mock_vllm = types.ModuleType('vllm')
                mock_vllm.SamplingParams = type('SamplingParams', (), {})
                mock_vllm.LLM = type('LLM', (), {})
                sys.modules['vllm'] = mock_vllm

                # Mock vllm.model_executor submodules
                mock_model_executor = types.ModuleType('vllm.model_executor')
                sys.modules['vllm.model_executor'] = mock_model_executor

                mock_guided_decoding = types.ModuleType('vllm.model_executor.guided_decoding')
                sys.modules['vllm.model_executor.guided_decoding'] = mock_guided_decoding

                mock_guided_fields = types.ModuleType('vllm.model_executor.guided_decoding.guided_fields')
                mock_guided_fields.GuidedDecodingRequest = type('GuidedDecodingRequest', (), {})
                sys.modules['vllm.model_executor.guided_decoding.guided_fields'] = mock_guided_fields

                logger.info("vllm module mocked (offline mode not available)")

        logger.info(f"HippoRAG path added: {hipporag_src_str}")

    def _load_hipporag_modules(self):
        """Lazily import HippoRAG core modules."""
        if self._hipporag_modules_loaded:
            return

        from hipporag.HippoRAG import HippoRAG
        from hipporag.utils.config_utils import BaseConfig

        self._HippoRAG = HippoRAG
        self._BaseConfig = BaseConfig
        self._hipporag_modules_loaded = True

        logger.info("HippoRAG modules loaded successfully")

    def _get_shared_embedding_model(self) -> TrackedEmbeddingWrapper:
        """Return the singleton embedding model instance (created on first call)."""
        if self._shared_embedding_model is None:
            self._shared_embedding_model = TrackedEmbeddingWrapper(
                provider=self.embedding_provider,
                model=self.embedding_model,
                model_path=self.embedding_model_path,
                dim=self.embedding_dim,
                api_key=self.embedding_api_key,
                base_url=self.embedding_base_url,
                batch_size=self.embedding_batch_size,
                max_seq_len=self.embedding_max_seq_len,
            )
        return self._shared_embedding_model

    def _build_hipporag_config(self, context_id: int) -> Any:
        """Build a HippoRAG BaseConfig for the given context_id."""
        self._load_hipporag_modules()

        if self.working_dir:
            save_dir = os.path.join(self.working_dir, f"context_{context_id}")
        else:
            save_dir = os.path.join(
                tempfile.gettempdir(),
                "hipporag_medmemorybench",
                f"context_{context_id}",
            )
        os.makedirs(save_dir, exist_ok=True)

        embedding_model_id = self.embedding_model_path if self.embedding_model_path else self.embedding_model
        hipporag_embedding_name = f"Transformers/{embedding_model_id}"

        config = self._BaseConfig(
            dataset=None,
            save_dir=save_dir,
            llm_name=self.model,
            llm_base_url=self._base_url,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            openie_mode=self.openie_mode,
            is_directed_graph=self.is_directed_graph,
            synonymy_edge_sim_threshold=self.synonymy_edge_sim_threshold,
            synonymy_edge_topk=self.synonymy_edge_topk,
            linking_top_k=self.linking_top_k,
            retrieval_top_k=self.retrieval_top_k,
            qa_top_k=self.qa_top_k,
            damping=self.damping,
            passage_node_weight=self.passage_node_weight,
            force_index_from_scratch=self.force_index_from_scratch,
            force_openie_from_scratch=self.force_openie_from_scratch,
            save_openie=self.save_openie,
            embedding_model_name=hipporag_embedding_name,
            embedding_batch_size=self.embedding_batch_size,
            embedding_max_seq_len=self.embedding_max_seq_len,
            embedding_return_as_normalized=True,
        )

        return config

    def _create_tracked_hipporag(self, context_id: int) -> Any:
        """Create a HippoRAG instance with LLM and Embedding components replaced for token tracking."""
        self._load_hipporag_modules()
        config = self._build_hipporag_config(context_id)

        logger.info(f"Creating HippoRAG instance for context_id={context_id}")
        logger.info(f"  openie_mode: {self.openie_mode}")
        logger.info(f"  save_dir: {config.save_dir}")
        logger.info(f"  embedding_model: {self.embedding_model}")

        tracked_llm = TrackedLLMWrapper(
            llm_client=self._llm_client,
            llm_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        tracked_embedding = self._get_shared_embedding_model()

        # Create instance then monkey-patch its internal components
        hipporag = self._HippoRAG(global_config=config)

        # Replace LLM model across all sub-components
        hipporag.llm_model = tracked_llm

        if hasattr(hipporag, 'openie') and hipporag.openie is not None:
            hipporag.openie.llm_model = tracked_llm

        if hasattr(hipporag, 'rerank_filter') and hipporag.rerank_filter is not None:
            hipporag.rerank_filter.llm_infer_fn = tracked_llm.infer

        # Replace embedding model across all embedding stores
        hipporag.embedding_model = tracked_embedding
        if hasattr(hipporag, 'chunk_embedding_store') and hipporag.chunk_embedding_store is not None:
            hipporag.chunk_embedding_store.embedding_model = tracked_embedding
        if hasattr(hipporag, 'entity_embedding_store') and hipporag.entity_embedding_store is not None:
            hipporag.entity_embedding_store.embedding_model = tracked_embedding
        if hasattr(hipporag, 'fact_embedding_store') and hipporag.fact_embedding_store is not None:
            hipporag.fact_embedding_store.embedding_model = tracked_embedding

        logger.info(f"HippoRAG instance created successfully")

        return hipporag

    def _get_context_id(self) -> int:
        return self._context_id if self._context_id is not None else 0

    def _get_hipporag_instance(self, context_id: int) -> Any:
        """Get or create the HippoRAG instance for the given context_id."""
        if context_id not in self._hipporag_instances:
            self._hipporag_instances[context_id] = self._create_tracked_hipporag(context_id)
        return self._hipporag_instances[context_id]

    def _format_input_documents(self, text: str) -> List[str]:
        """Split text into paragraphs for HippoRAG's index() method.

        HippoRAG handles chunking internally via chunk_embedding_store,
        so we only need a simple paragraph split here.
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs
        return [text] if text.strip() else []

    def _load_openie_results(self, hipporag) -> List[Dict]:
        """Load OpenIE extraction results from disk."""
        openie_path = hipporag.openie_results_path
        if os.path.exists(openie_path):
            try:
                with open(openie_path, 'r', encoding='utf-8') as f:
                    openie_data = json.load(f)
                return openie_data.get("docs", [])
            except Exception as e:
                logger.warning(f"Failed to load OpenIE results: {e}")
        return []

    def memorize(self, text: str, is_last_session: bool = False, **kwargs) -> MemoryBuildResult:
        """Memory construction phase.

        Design:
        =======
        The evaluation framework signals via is_last_session whether this is the
        final session in an evaluation unit.
        - Non-last sessions: accumulate only, no graph build.
        - Last session: accumulate then trigger graph construction.

        This ensures noise sessions are accumulated correctly, graph build timing
        is measured in the memorize phase, and each evaluation unit builds independently.

        New evaluation units are detected via _indexed_flags: if the current context
        was already indexed, we clear old state and start fresh.
        """
        context_id = self._get_context_id()
        get_usage_tracker().set_phase("memorize")

        start_time = time.time()

        if context_id not in self._pending_sessions:
            self._pending_sessions[context_id] = []
            self._session_counts[context_id] = 0

        # Detect new evaluation unit: if already indexed, reset state
        if self._indexed_flags.get(context_id, False):
            logger.info(f"[HippoRAG] Context {context_id} was already indexed - new evaluation unit detected")
            logger.info(f"[HippoRAG] Clearing old state for fresh indexing...")

            if context_id in self._pending_sessions:
                old_count = len(self._pending_sessions[context_id])
                self._pending_sessions[context_id] = []
                logger.info(f"[HippoRAG] Cleared {old_count} old pending sessions")

            if context_id in self._hipporag_instances:
                del self._hipporag_instances[context_id]

            self._session_counts[context_id] = 0
            self._indexed_flags[context_id] = False

        self._pending_sessions[context_id].append(text)
        self._session_counts[context_id] += 1
        session_count = self._session_counts[context_id]
        pending_count = len(self._pending_sessions[context_id])

        logger.info(f"[HippoRAG] Session {session_count} accumulated for context_id={context_id} "
                    f"(text length: {len(text)} chars, pending: {pending_count}, is_last: {is_last_session})")

        # Key: trigger graph construction when it is the last session
        if is_last_session:
            logger.info(f"[HippoRAG] Last session received, triggering graph construction for {pending_count} sessions...")

            flush_result = self._flush_pending_sessions(context_id)

            if flush_result and flush_result.get("success"):
                time_cost = time.time() - start_time
                graph_info = flush_result.get("graph_info", {})
                openie_count = flush_result.get("openie_count", 0)
                openie_results = flush_result.get("openie_results", [])

                # Build memory_entries from OpenIE results
                memory_entries = []
                for doc_result in openie_results:
                    entry = {
                        "content": doc_result.get("passage", "")[:500],
                        "entities": doc_result.get("extracted_entities", []),
                        "triples": doc_result.get("extracted_triples", [])[:5],
                    }
                    memory_entries.append(entry)

                return MemoryBuildResult(
                    success=True,
                    method="hipporag",
                    action="index",
                    input_content=text,
                    stored_content=text,
                    memory_entries=memory_entries,
                    all_passages=[{"content": text[:500]}],
                    chunk_count=session_count,
                    time_cost=time_cost,
                    extraction_result=json.dumps({
                        "status": "indexed",
                        "session_number": session_count,
                        "batch_sessions_processed": flush_result.get("session_count", 0),
                        "num_documents": flush_result.get("num_documents", 0),
                        "openie_count": openie_count,
                        "graph_info": graph_info,
                        "sample_entities": memory_entries[0].get("entities", [])[:5] if memory_entries else [],
                        "sample_triples": memory_entries[0].get("triples", [])[:3] if memory_entries else [],
                    }, ensure_ascii=False, default=str)[:5000],
                    extra={
                        "context_id": context_id,
                        "action": "index",
                        "session_number": session_count,
                        "batch_sessions_processed": flush_result.get("session_count", 0),
                        "graph_build_time": flush_result.get("time_cost", 0),
                        "graph_info": graph_info,
                    },
                )
            else:
                # Build failed
                error_msg = flush_result.get("error", "Unknown error") if flush_result else "Flush returned None"
                time_cost = time.time() - start_time
                return MemoryBuildResult(
                    success=False,
                    method="hipporag",
                    action="index_failed",
                    input_content=text,
                    stored_content="",
                    memory_entries=[],
                    all_passages=[],
                    chunk_count=session_count,
                    time_cost=time_cost,
                    extraction_result=f"Error: {error_msg}",
                    extra={
                        "context_id": context_id,
                        "action": "index_failed",
                        "error": error_msg,
                    },
                )
        else:
            # Not the last session, only return accumulation result
            time_cost = time.time() - start_time

            return MemoryBuildResult(
                success=True,
                method="hipporag",
                action="accumulate",
                input_content=text,
                stored_content=text,
                memory_entries=[],
                all_passages=[{"content": text[:500]}],
                chunk_count=session_count,
                time_cost=time_cost,
                extraction_result=json.dumps({
                    "status": "accumulated",
                    "session_number": session_count,
                    "pending_count": pending_count,
                    "message": f"Session {session_count} accumulated. Graph will be built on last session.",
                }, ensure_ascii=False),
                extra={
                    "context_id": context_id,
                    "action": "accumulate",
                    "session_number": session_count,
                    "pending_sessions": pending_count,
                    "total_accumulated_chars": sum(len(s) for s in self._pending_sessions[context_id]),
                },
            )

    def _flush_pending_sessions(self, context_id: int) -> Optional[Dict[str, Any]]:
        """
        Batch-process accumulated sessions and build the knowledge graph.

        This is where HippoRAG index() is actually executed.

        Returns:
            Build result info; returns None if there is no pending content.
        """
        if context_id not in self._pending_sessions or not self._pending_sessions[context_id]:
            logger.info(f"[HippoRAG] No pending sessions for context_id={context_id}")
            return None

        pending_texts = self._pending_sessions[context_id]
        session_count = len(pending_texts)

        logger.info(f"[HippoRAG] Flushing {session_count} accumulated sessions for context_id={context_id}")

        start_time = time.time()

        try:
            hipporag = self._get_hipporag_instance(context_id)
            combined_text = "\n\n".join(pending_texts)
            docs = self._format_input_documents(combined_text)
            logger.info(f"[HippoRAG] Combined {session_count} sessions: {len(combined_text)} chars, "
                        f"split into {len(docs)} documents")

            logger.info(f"[HippoRAG] Calling index() with openie_mode={self.openie_mode}")
            hipporag.index(docs)

            # Reset ready_to_retrieve so prepare_retrieval_objects() re-runs on next query
            # to pick up newly added passages and embeddings
            hipporag.ready_to_retrieve = False
            logger.info(f"[HippoRAG] Reset ready_to_retrieve=False to refresh retrieval cache on next query")

            graph_info = hipporag.get_graph_info()
            openie_results = self._load_openie_results(hipporag)

            time_cost = time.time() - start_time

            logger.info(f"[HippoRAG] Graph construction complete in {time_cost:.2f}s")
            logger.info(f"[HippoRAG] Graph statistics:")
            logger.info(f"  - Phrase nodes: {graph_info.get('num_phrase_nodes', 0)}")
            logger.info(f"  - Passage nodes: {graph_info.get('num_passage_nodes', 0)}")
            logger.info(f"  - Extracted triples: {graph_info.get('num_extracted_triples', 0)}")
            logger.info(f"  - Total nodes: {graph_info.get('num_total_nodes', 0)}")
            logger.info(f"  - Total triples: {graph_info.get('num_total_triples', 0)}")

            if openie_results:
                logger.info(f"[HippoRAG] OpenIE extraction samples ({len(openie_results)} total):")
                for i, doc in enumerate(openie_results[:3]):
                    logger.info(f"  Sample {i+1}:")
                    logger.info(f"    Passage: {doc.get('passage', '')[:100]}...")
                    logger.info(f"    Entities: {doc.get('extracted_entities', [])[:5]}")
                    logger.info(f"    Triples: {doc.get('extracted_triples', [])[:2]}")

            self._pending_sessions[context_id] = []
            self._indexed_flags[context_id] = True
            self._is_initialized = True

            return {
                "success": True,
                "session_count": session_count,
                "num_documents": len(docs),
                "graph_info": graph_info,
                "openie_count": len(openie_results) if openie_results else 0,
                "openie_results": openie_results,  # Include OpenIE results for memorize() usage
                "time_cost": time_cost,
            }

        except Exception as e:
            logger.error(f"[HippoRAG] Graph construction error: {e}")
            import traceback
            traceback.print_exc()

            # Clear accumulation buffer to avoid repeated attempts
            self._pending_sessions[context_id] = []

            return {
                "success": False,
                "session_count": session_count,
                "error": str(e),
                "time_cost": time.time() - start_time,
            }

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Query phase.

        The graph should have been built during memorize() (on the last session).
        This method only handles retrieval and QA.
        """
        context_id = self._get_context_id()
        get_usage_tracker().set_phase("query")

        logger.info(f"[HippoRAG] Starting query for context_id={context_id}")
        logger.info(f"[HippoRAG] Question: {question[:100]}...")

        start_time = time.time()

        try:
            if not self._indexed_flags.get(context_id, False):
                logger.warning("[HippoRAG] No data indexed. Please call memorize() first.")
                return AgentResponse(
                    output="Error: No memory data available. Please memorize content first before querying.",
                    query_time=time.time() - start_time,
                    retrieved_count=0,
                    extra={
                        "method": "hipporag",
                        "error": "no_data_indexed",
                        "message": "No content has been indexed. Call memorize() before query().",
                    },
                )

            hipporag = self._get_hipporag_instance(context_id)

            passage_count = len(hipporag.chunk_embedding_store.get_all_ids())
            if passage_count == 0:
                logger.warning("[HippoRAG] No data indexed. Please call memorize() first.")
                return AgentResponse(
                    output="Error: No memory data available. Please memorize content first before querying.",
                    query_time=time.time() - start_time,
                    retrieved_count=0,
                    extra={
                        "method": "hipporag",
                        "error": "no_data_indexed",
                        "message": "No content has been indexed. Call memorize() before query().",
                    },
                )

            if not hipporag.ready_to_retrieve:
                logger.info("[HippoRAG] Preparing retrieval objects...")
                hipporag.prepare_retrieval_objects()

            # Step 1: Retrieve documents
            query_solutions = hipporag.retrieve(queries=[question])

            if query_solutions and len(query_solutions) > 0:
                solution = query_solutions[0]

                # Step 2: Truncate retrieved docs to fit max_context_tokens
                raw_docs = solution.docs[:self.qa_top_k] if solution.docs else []
                if raw_docs and self.max_context_tokens > 0:
                    question_tokens = self.count_tokens(question)
                    # Reserve tokens for question, QA prompt template overhead, and output
                    available_tokens = max(self.max_context_tokens - question_tokens - 500, 0)

                    truncated_docs = []
                    current_tokens = 0
                    for doc in raw_docs:
                        doc_text = doc if isinstance(doc, str) else str(doc)
                        # Account for the "Wikipedia Title: {doc}\n\n" format used by HippoRAG QA
                        format_overhead = self.count_tokens("Wikipedia Title: \n\n")
                        doc_tokens = self.count_tokens(doc_text)

                        if current_tokens + doc_tokens + format_overhead <= available_tokens:
                            truncated_docs.append(doc_text)
                            current_tokens += doc_tokens + format_overhead
                        else:
                            remaining = available_tokens - current_tokens - format_overhead
                            if remaining > 100:
                                tokens = self._tokenizer.encode(doc_text)
                                truncated_docs.append(self._tokenizer.decode(tokens[:remaining]))
                            break
                    solution.docs = truncated_docs
                else:
                    solution.docs = raw_docs

                # Step 3: QA with truncated docs
                qa_solutions, responses, metadata = hipporag.qa([solution])
                solution = qa_solutions[0] if qa_solutions else solution
                answer = solution.answer if hasattr(solution, 'answer') else str(solution)

                retrieved_docs = []
                if hasattr(solution, 'docs') and solution.docs:
                    for i, doc in enumerate(solution.docs[:self.qa_top_k]):
                        doc_info = {
                            "rank": i + 1,
                            "content": doc[:500] if isinstance(doc, str) else str(doc)[:500],
                        }
                        if hasattr(solution, 'doc_scores') and solution.doc_scores is not None and len(solution.doc_scores) > i:
                            doc_info["score"] = float(solution.doc_scores[i])
                        retrieved_docs.append(doc_info)
            else:
                answer = "No answer generated"
                retrieved_docs = []

            query_time = time.time() - start_time

            logger.info(f"[HippoRAG] Query complete in {query_time:.2f}s")
            logger.info(f"[HippoRAG] Retrieved {len(retrieved_docs)} documents")
            logger.info(f"[HippoRAG] Answer preview: {answer[:100]}...")

            retrieved_memories = [
                {
                    "content": doc.get("content", "")[:500] if isinstance(doc, dict) else str(doc)[:500],
                    "rank": doc.get("rank", i + 1) if isinstance(doc, dict) else i + 1,
                    "score": doc.get("score", 0.0) if isinstance(doc, dict) else 0.0,
                }
                for i, doc in enumerate(retrieved_docs)
            ]

            return AgentResponse(
                output=answer,
                query_time=query_time,
                retrieved_count=len(retrieved_docs),
                retrieved_memories=retrieved_memories,
                extra={
                    "method": "hipporag",
                    "openie_mode": self.openie_mode,
                    "retrieved_docs": retrieved_docs[:5],
                    "retrieval_times": {
                        "ppr_time": hipporag.ppr_time if hasattr(hipporag, 'ppr_time') else 0,
                        "rerank_time": hipporag.rerank_time if hasattr(hipporag, 'rerank_time') else 0,
                    },
                },
            )

        except Exception as e:
            logger.error(f"[HippoRAG] Query error: {e}")
            import traceback
            traceback.print_exc()

            return AgentResponse(
                output=f"Error: {str(e)}",
                query_time=time.time() - start_time,
                retrieved_count=0,
                extra={"error": str(e)},
            )

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self._hipporag_instances = {}
        self._pending_sessions = {}
        self._session_counts = {}
        self._indexed_flags = {}
        logger.info("[HippoRAG] Agent reset, all instances and pending sessions cleared")

    def set_context_id(self, context_id: int) -> None:
        """Set context ID."""
        super().set_context_id(context_id)
        logger.info(f"[HippoRAG] Context ID set to {context_id}")

    def get_info(self) -> Dict[str, Any]:
        """Get agent info."""
        info = super().get_info()
        context_id = self._get_context_id()
        is_indexed = self._indexed_flags.get(context_id, False)

        info.update({
            "openie_mode": self.openie_mode,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "linking_top_k": self.linking_top_k,
            "retrieval_top_k": self.retrieval_top_k,
            "qa_top_k": self.qa_top_k,
            "damping": self.damping,
            "num_instances": len(self._hipporag_instances),
            "pending_sessions": {k: len(v) for k, v in self._pending_sessions.items()},
            "is_indexed": is_indexed,
        })
        return info
