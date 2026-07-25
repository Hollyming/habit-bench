"""BM25 RAG Agent - sparse retrieval with BM25 algorithm."""

import logging
from typing import Optional, List

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import create_llm_client, format_messages, BaseLLMClient

logger = logging.getLogger(__name__)


class BM25RAGAgent(BaseAgent):
    """BM25 sparse retrieval based RAG agent. Supports Chinese and English tokenization."""

    METHOD_TYPE = "rag"
    DEFAULT_MAX_CONTEXT_TOKENS = 120000

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
        language: str = "auto",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_context_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.language = language
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_context_tokens = max_context_tokens or self.DEFAULT_MAX_CONTEXT_TOKENS

        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        self._bm25 = None
        self._tokenized_corpus: List[List[str]] = []
        self._chunks: List[str] = []

    def _detect_language(self, text: str) -> str:
        """Detect text language based on Chinese character ratio."""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return "zh" if chinese_chars / max(len(text), 1) > 0.1 else "en"

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        lang = self.language if self.language != "auto" else self._detect_language(text)

        if lang == "zh":
            try:
                import jieba
                return list(jieba.cut(text))
            except ImportError:
                return list(text)
        else:
            try:
                from nltk.tokenize import word_tokenize
                return word_tokenize(text.lower())
            except (ImportError, LookupError):
                return text.lower().split()

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into token-based chunks."""
        if not text.strip():
            return []

        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(self._tokenizer.decode(tokens[start:end]))
            start = end - self.chunk_overlap
            if end == len(tokens):
                break

        return chunks

    def _build_index(self) -> None:
        """Build BM25 index from chunks."""
        if not self._chunks:
            return

        from rank_bm25 import BM25Okapi
        self._tokenized_corpus = [self._tokenize(doc) for doc in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Add text to memory and build BM25 index."""
        self._memory_chunks.append(text)
        self._chunks.extend(self._split_text_into_chunks(text))
        self._is_initialized = True
        self._build_index()

        return MemoryBuildResult(
            success=True,
            method="bm25_rag",
            action="build_bm25_index",
            input_content=text,
            stored_content=text,
            chunk_count=len(self._chunks),
            extra={"original_doc_count": len(self._memory_chunks)},
        )

    def _retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents using BM25."""
        if self._bm25 is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]

        return [self._chunks[i] for i in top_indices if scores[i] > 0]

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query the agent with BM25 retrieval."""
        retrieved_docs = self._retrieve(question)
        truncated_docs = self.truncate_docs_to_context(
            docs=retrieved_docs,
            question=question,
            system_message=system_message,
            max_context_tokens=self.max_context_tokens,
        )

        if truncated_docs:
            context = "\n\n".join([
                f"[Memory {i+1}]\n{doc}"
                for i, doc in enumerate(truncated_docs)
            ])
            full_message = f"{context}\n\n{question}"
        else:
            full_message = question

        messages = format_messages(full_message, system_message)
        response = self._llm_client.chat(messages)

        formatted_memories = [
            {"memory": doc[:500] + "..." if len(doc) > 500 else doc, "type": "bm25_retrieval"}
            for doc in truncated_docs
        ]

        return AgentResponse(
            output=response.content,
            query_time=0.0,
            retrieved_count=len(truncated_docs),
            retrieved_memories=formatted_memories,
            extra={
                "method": "bm25_rag",
                "k1": self.k1,
                "b": self.b,
                "language": self.language,
                "original_retrieved_count": len(retrieved_docs),
                "truncated_count": len(truncated_docs),
            }
        )

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self._bm25 = None
        self._tokenized_corpus = []
        self._chunks = []

    @property
    def has_index(self) -> bool:
        """Check if BM25 index is built."""
        return self._bm25 is not None

    @property
    def chunk_count(self) -> int:
        """Get current chunk count."""
        return len(self._chunks)
