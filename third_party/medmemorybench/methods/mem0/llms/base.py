from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable

from methods.mem0.configs.llms.base import BaseLlmConfig


class LLMBase(ABC):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        if config is None:
            self.config = BaseLlmConfig()
        else:
            self.config = config
        self._usage_callback: Optional[Callable[[int, int, float], None]] = None

    def set_usage_callback(self, callback: Callable[[int, int, float], None]) -> None:
        self._usage_callback = callback

    @abstractmethod
    def generate_response(self, messages, tools: Optional[List[Dict]] = None, tool_choice: str = "auto"):
        pass
