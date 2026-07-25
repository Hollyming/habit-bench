"""Long Context Agent - concatenates all memory into context for LLM."""

from typing import Optional, List

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import create_llm_client, format_messages


class LongContextAgent(BaseAgent):
    """Baseline method using LLM's long context capability."""

    METHOD_TYPE = "baseline"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_context_tokens: int = 100000,
        truncation_strategy: str = "oldest_first",
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.max_context_tokens = max_context_tokens
        self.truncation_strategy = truncation_strategy

        self._llm_client = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        self._context = ""

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Add text to memory context."""
        prev_context_len = len(self._context)

        self._memory_chunks.append(text)

        if self._context:
            self._context += "\n\n" + text
        else:
            self._context = text

        self._truncate_if_needed()
        self._is_initialized = True

        return MemoryBuildResult(
            success=True,
            method="long_context",
            action="append_context",
            input_content=text,
            stored_content=text,
            memory_entries=[],
            chunk_count=len(self._memory_chunks),
            extra={
                "context_length_before": prev_context_len,
                "context_length_after": len(self._context),
            }
        )

    def _truncate_if_needed(self) -> None:
        """Truncate context if exceeds limit."""
        current_tokens = self.count_tokens(self._context)

        if current_tokens <= self.max_context_tokens:
            return

        if self.truncation_strategy == "oldest_first":
            while self._memory_chunks and self.count_tokens(self._context) > self.max_context_tokens:
                self._memory_chunks.pop(0)
                self._context = "\n\n".join(self._memory_chunks)
        else:
            encoded = self._tokenizer.encode(self._context)
            self._context = self._tokenizer.decode(encoded[-self.max_context_tokens:])

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query the agent."""
        # Concatenate context and question; truncation already done in memorize phase
        if self._context:
            full_message = f"{self._context}\n\n{question}"
        else:
            full_message = question

        messages = format_messages(full_message, system_message)
        response = self._llm_client.chat(messages)

        return AgentResponse(
            output=response.content,
            query_time=0.0,
            retrieved_count=0,
            extra={
                "context_tokens": self.count_tokens(self._context) if self._context else 0,
                "method": "long_context",
            }
        )

    def reset(self) -> None:
        """Reset agent."""
        super().reset()
        self._context = ""

    @property
    def context(self) -> str:
        """Get current context."""
        return self._context

    @property
    def context_tokens(self) -> int:
        """Get current context token count."""
        return self.count_tokens(self._context) if self._context else 0
