import os
import time
import logging
import warnings
from typing import Literal, Optional, List

from openai import OpenAI

from methods.mem0.configs.embeddings.base import BaseEmbedderConfig
from methods.mem0.embeddings.base import EmbeddingBase

logger = logging.getLogger(__name__)


class OpenAIEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        self.config.embedding_dims = self.config.embedding_dims or 1536

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if os.environ.get("OPENAI_API_BASE"):
            warnings.warn(
                "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                "Please use 'OPENAI_BASE_URL' instead.",
                DeprecationWarning,
            )

        # Configure httpx client with aggressive timeout and minimal connection reuse
        # to handle proxy (ClashX) connection issues that cause CLOSED state connections
        import httpx
        import socket

        # Shorter timeout to fail fast when proxy connections die
        timeout = httpx.Timeout(timeout=30.0, connect=15.0, read=30.0, write=30.0)

        # Minimal connection pooling - proxy connections are unreliable for keepalive
        limits = httpx.Limits(
            max_keepalive_connections=1,  # Minimal keepalive
            max_connections=5,
            keepalive_expiry=10.0,  # Very short expiry - proxy connections die fast
        )

        # Create transport with socket-level options for better dead connection detection
        transport = httpx.HTTPTransport(
            retries=0,  # We handle retries ourselves
            limits=limits,
            socket_options=[
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),  # Enable TCP keepalive
                (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),  # Keepalive interval: 10s
                (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),  # Max keepalive probes
            ]
        )

        http_client = httpx.Client(timeout=timeout, transport=transport)

        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")

        # Retry logic for rate limiting and transient errors
        max_retries = 5
        last_exception = None

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.embeddings.create(
                    input=[text],
                    model=self.config.model,
                    dimensions=self.config.embedding_dims
                )
                latency = time.time() - start_time

                # Track token usage if callback is set
                if self._usage_callback and response.usage:
                    self._usage_callback(response.usage.total_tokens, latency)

                return response.data[0].embedding
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check if it's a retryable error (rate limit, timeout, server error, connection error)
                is_retryable = any(keyword in error_msg for keyword in [
                    "rate limit", "429", "timeout", "502", "503", "504",
                    "connection", "overloaded", "capacity", "ssl", "eof",
                    "reset", "refused", "closed", "broken", "pipe", "aborted"
                ])

                if is_retryable and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5s, 10s, 15s, 20s
                    logger.warning(
                        f"[Mem0 Embedding] Error: {e}, waiting {wait_time}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    print(f"⚠️  [Mem0 Embedding] Error: {e}")
                    print(f"   Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")

                    # On connection errors, recreate the HTTP client to get fresh connections
                    if attempt >= 2:
                        try:
                            logger.warning("[Mem0 Embedding] Recreating HTTP client after repeated errors...")
                            print("   Recreating HTTP client...")
                            self._recreate_client()
                        except Exception as recreate_error:
                            logger.warning(f"[Mem0 Embedding] Failed to recreate client: {recreate_error}")

                    time.sleep(wait_time)
                else:
                    if attempt >= max_retries - 1:
                        logger.error(f"[Mem0 Embedding] Failed after {max_retries} retries: {e}")
                    raise

        raise last_exception

    def _recreate_client(self):
        """Recreate HTTP client with fresh connections."""
        import httpx
        import socket

        # Close the old client if possible
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
            except Exception:
                pass

        # Create a new HTTP client with aggressive settings for proxy reliability
        timeout = httpx.Timeout(timeout=30.0, connect=15.0, read=30.0, write=30.0)
        limits = httpx.Limits(
            max_keepalive_connections=1,
            max_connections=5,
            keepalive_expiry=10.0,
        )
        transport = httpx.HTTPTransport(
            retries=0,
            limits=limits,
            socket_options=[
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
                (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
            ]
        )
        http_client = httpx.Client(timeout=timeout, transport=transport)

        # Recreate the OpenAI client
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    def embed_batch(self, texts: List[str], memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[List[float]]:
        """
        Get embeddings for multiple texts in a single API call.

        Args:
            texts (List[str]): List of texts to embed.
            memory_action (optional): The type of embedding to use.
        Returns:
            List[List[float]]: List of embedding vectors.
        """
        if not texts:
            return []

        # Clean texts
        cleaned_texts = [text.replace("\n", " ") for text in texts]

        # Retry logic for rate limiting and transient errors
        max_retries = 5
        last_exception = None

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.embeddings.create(
                    input=cleaned_texts,
                    model=self.config.model,
                    dimensions=self.config.embedding_dims
                )
                latency = time.time() - start_time

                # Track token usage if callback is set
                if self._usage_callback and response.usage:
                    self._usage_callback(response.usage.total_tokens, latency)

                # Sort by index to ensure correct order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                is_retryable = any(keyword in error_msg for keyword in [
                    "rate limit", "429", "timeout", "502", "503", "504",
                    "connection", "overloaded", "capacity", "ssl", "eof",
                    "reset", "refused", "closed", "broken", "pipe", "aborted"
                ])

                if is_retryable and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    logger.warning(
                        f"[Mem0 Embedding Batch] Error: {e}, waiting {wait_time}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    print(f"⚠️  [Mem0 Embedding Batch] Error: {e}")
                    print(f"   Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")

                    # On connection errors, recreate the HTTP client to get fresh connections
                    if attempt >= 2:
                        try:
                            logger.warning("[Mem0 Embedding Batch] Recreating HTTP client after repeated errors...")
                            print("   Recreating HTTP client...")
                            self._recreate_client()
                        except Exception as recreate_error:
                            logger.warning(f"[Mem0 Embedding Batch] Failed to recreate client: {recreate_error}")

                    time.sleep(wait_time)
                else:
                    if attempt >= max_retries - 1:
                        logger.error(f"[Mem0 Embedding Batch] Failed after {max_retries} retries: {e}")
                    raise

        raise last_exception
