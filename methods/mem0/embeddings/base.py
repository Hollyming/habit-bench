from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional

from methods.mem0.configs.embeddings.base import BaseEmbedderConfig


class EmbeddingBase(ABC):
    """Initialized a base embedding class

    :param config: Embedding configuration option class, defaults to None
    :type config: Optional[BaseEmbedderConfig], optional
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        if config is None:
            self.config = BaseEmbedderConfig()
        else:
            self.config = config
        # Usage callback: (input_tokens: int, latency: float) -> None
        self._usage_callback: Optional[Callable[[int, float], None]] = None

    def set_usage_callback(self, callback: Callable[[int, float], None]) -> None:
        """Set callback for tracking embedding token usage."""
        self._usage_callback = callback

    @abstractmethod
    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]]):
        """
        Get the embedding for the given text.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        pass
