from __future__ import annotations

from typing import Any
from urllib.parse import quote

from core.settings import LLMSettings
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import LLMTransport, OpenAILLM


class AzureOpenAILLM(OpenAILLM):
    def __init__(self, settings: LLMSettings, transport: LLMTransport | None = None, timeout: float = 30.0) -> None:
        super().__init__(settings, transport=transport, timeout=timeout)

    def _chat_url(self) -> str:
        endpoint = self.settings.azure_endpoint.rstrip("/")
        deployment = quote(self.settings.deployment_name or self.settings.model, safe="")
        api_version = self.settings.api_version or "2024-02-15-preview"
        return f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["api-key"] = self.settings.api_key
        return headers

    def chat(self, messages: Any) -> str:
        if not self.settings.azure_endpoint:
            raise ValueError("azure validation error: azure_endpoint is required")
        return super().chat(messages)


LLMFactory.register_provider("azure", AzureOpenAILLM)
