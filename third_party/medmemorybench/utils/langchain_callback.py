"""LangChain callback handler for token usage tracking."""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback handler to track LLM token usage in LangChain."""

    def __init__(self, model_name: str = "langchain"):
        super().__init__()
        self._model_name = model_name
        self._start_time: Optional[float] = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self._start_time = time.time()

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List], **kwargs: Any) -> None:
        self._start_time = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        latency = time.time() - self._start_time if self._start_time else 0.0
        self._start_time = None

        input_tokens = 0
        output_tokens = 0

        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)

            if input_tokens == 0 and output_tokens == 0:
                usage = response.llm_output.get("usage", {})
                if usage:
                    input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

        if input_tokens == 0 and output_tokens == 0 and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "generation_info") and gen.generation_info:
                        usage = gen.generation_info.get("usage", {})
                        if usage:
                            input_tokens += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                            output_tokens += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

                    if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                        metadata = gen.message.usage_metadata
                        if metadata:
                            input_tokens += getattr(metadata, "input_tokens", 0)
                            output_tokens += getattr(metadata, "output_tokens", 0)

        if input_tokens == 0 and output_tokens == 0:
            logger.warning(
                "[TokenUsageCallback] No token usage data found in LLM response "
                f"(model={self._model_name}). This LLM call will NOT be counted "
                "in token statistics. Check if the API provider returns usage data."
            )
            return

        from utils.llm_client import get_usage_tracker, LLMResponse

        llm_response = LLMResponse(
            content="",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=latency,
            model=self._model_name,
        )
        get_usage_tracker().record(llm_response)
