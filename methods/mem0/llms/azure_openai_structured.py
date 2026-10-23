import os
import time
from typing import Dict, List, Optional

from openai import AzureOpenAI

from methods.mem0.configs.llms.base import BaseLlmConfig
from methods.mem0.llms.base import LLMBase


class AzureOpenAIStructuredLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-4o-2024-08-06"

        api_key = os.getenv("LLM_AZURE_OPENAI_API_KEY") or self.config.azure_kwargs.api_key
        azure_deployment = os.getenv("LLM_AZURE_DEPLOYMENT") or self.config.azure_kwargs.azure_deployment
        azure_endpoint = os.getenv("LLM_AZURE_ENDPOINT") or self.config.azure_kwargs.azure_endpoint
        api_version = os.getenv("LLM_AZURE_API_VERSION") or self.config.azure_kwargs.api_version
        default_headers = self.config.azure_kwargs.default_headers

        self.client = AzureOpenAI(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
            http_client=self.config.http_client,
            default_headers=default_headers,
        )

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> str:
        """
        Generate a response based on the given messages using Azure OpenAI.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries, each containing a 'role' and 'content' key.
            response_format (Optional[str]): The desired format of the response. Defaults to None.

        Returns:
            str: The generated response.
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        start_time = time.time()
        response = self.client.chat.completions.create(**params)
        latency = time.time() - start_time

        if self._usage_callback and response.usage:
            self._usage_callback(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                latency,
            )

        return self._parse_response(response, tools)
