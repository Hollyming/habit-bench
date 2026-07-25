import json
import time

from collections.abc import Generator

import openai

from openai._types import NOT_GIVEN
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from memos.configs.llm import AzureLLMConfig, OpenAILLMConfig
from memos.llms.base import BaseLLM
from memos.llms.utils import remove_thinking_tags
from memos.log import get_logger
from memos.types import MessageList
from memos.utils import timed_with_status


logger = get_logger(__name__)


# ============================================================================
# Token Usage Tracking Integration
# ============================================================================
# This integration allows MemOS LLM calls to be tracked by the global usage tracker
# from utils.llm_client, enabling comprehensive token statistics across all phases.

def _try_record_usage(response, latency: float, model: str) -> None:
    """Try to record token usage to the global tracker if available.

    This function attempts to import and use the global LLM usage tracker.
    If the tracker is not available (e.g., when MemOS is used standalone),
    it silently skips recording without affecting normal operation.
    """
    try:
        from utils.llm_client import get_usage_tracker, LLMResponse

        if response and hasattr(response, 'usage') and response.usage:
            llm_response = LLMResponse(
                content="",  # Content not needed for tracking
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                latency=latency,
                model=model,
                raw_response=None,
            )
            get_usage_tracker().record(llm_response)
    except ImportError:
        # utils.llm_client not available, skip tracking
        pass
    except Exception as e:
        # Log but don't fail on tracking errors
        logger.debug(f"Token tracking skipped: {e}")


def _use_max_completion_tokens(model: str) -> bool:
    """Check if model requires max_completion_tokens instead of max_tokens.

    New OpenAI models (gpt-4o, gpt-5, o1, o3, etc.) require max_completion_tokens.
    """
    new_patterns = ["gpt-5", "gpt-4o", "o1-", "o3-"]
    model_lower = model.lower()
    return any(p in model_lower for p in new_patterns)


class OpenAILLM(BaseLLM):
    """OpenAI LLM class via openai.chat.completions.create."""

    def __init__(self, config: OpenAILLMConfig):
        self.config = config
        self.client = openai.Client(
            api_key=config.api_key, base_url=config.api_base, default_headers=config.default_headers
        )
        self.use_backup_client = config.backup_client
        if self.use_backup_client:
            self.backup_client = openai.Client(
                api_key=config.backup_api_key,
                base_url=config.backup_api_base,
                default_headers=config.backup_headers,
            )
            logger.info(
                f"OpenAI LLM instance initialized with backup "
                f"(model={config.backup_model_name_or_path})"
            )
        else:
            self.backup_client = None
            logger.info("OpenAI LLM instance initialized")

    def _parse_response(self, response) -> str:
        """Extract text content from a chat completion response."""
        if not response.choices:
            logger.warning("OpenAI response has no choices")
            return ""

        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            return self.tool_call_parser(tool_calls)
        response_content = response.choices[0].message.content
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content:
            reasoning_content = f"<think>{reasoning_content}</think>"
        if self.config.remove_think_prefix:
            return remove_thinking_tags(response_content or "")
        if reasoning_content:
            return reasoning_content + (response_content or "")
        return response_content or ""

    @timed_with_status(
        log_prefix="OpenAI LLM",
        log_extra_args=lambda self, messages, **kwargs: {
            "model_name_or_path": kwargs.get("model_name_or_path", self.config.model_name_or_path),
            "messages": messages,
        },
    )
    def generate(self, messages: MessageList, **kwargs) -> str:
        """Generate a response from OpenAI LLM, optionally overriding generation params."""
        model = kwargs.get("model_name_or_path", self.config.model_name_or_path)
        max_tokens_value = kwargs.get("max_tokens", self.config.max_tokens)

        request_body = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "extra_body": kwargs.get("extra_body", self.config.extra_body),
            "tools": kwargs.get("tools", NOT_GIVEN),
        }

        # Use max_completion_tokens for new models, max_tokens for older ones
        if _use_max_completion_tokens(model):
            request_body["max_completion_tokens"] = max_tokens_value
        else:
            request_body["max_tokens"] = max_tokens_value

        start_time = time.perf_counter()
        logger.info(f"OpenAI LLM Request body: {request_body}")

        try:
            response = self.client.chat.completions.create(**request_body)
            cost_time = time.perf_counter() - start_time
            logger.info(
                f"Request body: {request_body}, Response from OpenAI: "
                f"{response.model_dump_json()}, Cost time: {cost_time}"
            )
            # Record token usage to global tracker
            _try_record_usage(response, cost_time, request_body["model"])
            return self._parse_response(response)
        except Exception as e:
            if not self.use_backup_client:
                raise
            logger.warning(
                f"Primary LLM request failed with {type(e).__name__}: {e}, "
                f"falling back to backup client"
            )
            backup_body = {
                **request_body,
                "model": self.config.backup_model_name_or_path or request_body["model"],
            }
            backup_response = self.backup_client.chat.completions.create(**backup_body)
            cost_time = time.perf_counter() - start_time
            logger.info(
                f"Backup LLM request succeeded, Response: "
                f"{backup_response.model_dump_json()}, Cost time: {cost_time}"
            )
            # Record token usage for backup response
            _try_record_usage(backup_response, cost_time, backup_body["model"])
            return self._parse_response(backup_response)

    @timed_with_status(
        log_prefix="OpenAI LLM Stream",
        log_extra_args=lambda self, messages, **kwargs: {
            "model_name_or_path": self.config.model_name_or_path
        },
    )
    def generate_stream(self, messages: MessageList, **kwargs) -> Generator[str, None, None]:
        """Stream response from OpenAI LLM with optional reasoning support."""
        if kwargs.get("tools"):
            logger.info("stream api not support tools")
            return

        model = self.config.model_name_or_path
        max_tokens_value = kwargs.get("max_tokens", self.config.max_tokens)

        request_body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "extra_body": kwargs.get("extra_body", self.config.extra_body),
            "tools": kwargs.get("tools", NOT_GIVEN),
        }

        # Use max_completion_tokens for new models, max_tokens for older ones
        if _use_max_completion_tokens(model):
            request_body["max_completion_tokens"] = max_tokens_value
        else:
            request_body["max_tokens"] = max_tokens_value

        logger.info(f"OpenAI LLM Stream Request body: {request_body}")
        response = self.client.chat.completions.create(**request_body)

        reasoning_started = False

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Support for custom 'reasoning_content' (if present in OpenAI-compatible models like Qwen, DeepSeek)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if not reasoning_started and not self.config.remove_think_prefix:
                    yield "<think>"
                    reasoning_started = True
                yield delta.reasoning_content
            elif hasattr(delta, "content") and delta.content:
                if reasoning_started and not self.config.remove_think_prefix:
                    yield "</think>"
                    reasoning_started = False
                yield delta.content

        # Ensure we close the <think> block if not already done
        if reasoning_started and not self.config.remove_think_prefix:
            yield "</think>"

    def tool_call_parser(self, tool_calls: list[ChatCompletionMessageToolCall]) -> list[dict]:
        """Parse tool calls from OpenAI response."""
        return [
            {
                "tool_call_id": tool_call.id,
                "function_name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            }
            for tool_call in tool_calls
        ]


class AzureLLM(BaseLLM):
    """Azure OpenAI LLM class with singleton pattern."""

    def __init__(self, config: AzureLLMConfig):
        self.config = config
        self.client = openai.AzureOpenAI(
            azure_endpoint=config.base_url,
            api_version=config.api_version,
            api_key=config.api_key,
        )
        logger.info("Azure LLM instance initialized")

    def generate(self, messages: MessageList, **kwargs) -> str:
        """Generate a response from Azure OpenAI LLM."""
        model = self.config.model_name_or_path
        max_tokens_value = kwargs.get("max_tokens", self.config.max_tokens)

        request_params = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "tools": kwargs.get("tools", NOT_GIVEN),
            "extra_body": kwargs.get("extra_body", self.config.extra_body),
        }

        # Use max_completion_tokens for new models, max_tokens for older ones
        if _use_max_completion_tokens(model):
            request_params["max_completion_tokens"] = max_tokens_value
        else:
            request_params["max_tokens"] = max_tokens_value

        start_time = time.perf_counter()
        response = self.client.chat.completions.create(**request_params)
        cost_time = time.perf_counter() - start_time
        logger.info(f"Response from Azure OpenAI: {response.model_dump_json()}")
        # Record token usage to global tracker
        _try_record_usage(response, cost_time, model)
        if not response.choices:
            logger.warning("Azure OpenAI response has no choices")
            return ""

        if response.choices[0].message.tool_calls:
            return self.tool_call_parser(response.choices[0].message.tool_calls)
        response_content = response.choices[0].message.content
        if self.config.remove_think_prefix:
            return remove_thinking_tags(response_content or "")
        else:
            return response_content or ""

    def generate_stream(self, messages: MessageList, **kwargs) -> Generator[str, None, None]:
        """Stream response from Azure OpenAI LLM with optional reasoning support."""
        if kwargs.get("tools"):
            logger.info("stream api not support tools")
            return

        model = self.config.model_name_or_path
        max_tokens_value = kwargs.get("max_tokens", self.config.max_tokens)

        request_params = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "extra_body": kwargs.get("extra_body", self.config.extra_body),
        }

        # Use max_completion_tokens for new models, max_tokens for older ones
        if _use_max_completion_tokens(model):
            request_params["max_completion_tokens"] = max_tokens_value
        else:
            request_params["max_tokens"] = max_tokens_value

        response = self.client.chat.completions.create(**request_params)

        reasoning_started = False

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Support for custom 'reasoning_content' (if present in OpenAI-compatible models like Qwen, DeepSeek)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if not reasoning_started and not self.config.remove_think_prefix:
                    yield "<think>"
                    reasoning_started = True
                yield delta.reasoning_content
            elif hasattr(delta, "content") and delta.content:
                if reasoning_started and not self.config.remove_think_prefix:
                    yield "</think>"
                    reasoning_started = False
                yield delta.content

        # Ensure we close the <think> block if not already done
        if reasoning_started and not self.config.remove_think_prefix:
            yield "</think>"

    def tool_call_parser(self, tool_calls: list[ChatCompletionMessageToolCall]) -> list[dict]:
        """Parse tool calls from OpenAI response."""
        return [
            {
                "tool_call_id": tool_call.id,
                "function_name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            }
            for tool_call in tool_calls
        ]
