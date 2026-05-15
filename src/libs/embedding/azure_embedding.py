from __future__ import annotations

from typing import Any
from urllib.parse import quote

from core.settings import EmbeddingSettings
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import EmbeddingTransport, OpenAIEmbedding


class AzureOpenAIEmbedding(OpenAIEmbedding):
    def __init__(self, settings: EmbeddingSettings, transport: EmbeddingTransport | None = None, timeout: float = 30.0) -> None:
        super().__init__(settings, transport=transport, timeout=timeout)

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        if not self.settings.azure_endpoint:
            raise ValueError("azure validation error: azure_endpoint is required")
        return super().embed(texts, trace=trace)

    def _embeddings_url(self) -> str:
        endpoint = self.settings.azure_endpoint.rstrip("/")
        deployment = quote(self.settings.deployment_name or self.settings.model, safe="")
        api_version = self.settings.api_version or "2024-02-15-preview"
        return f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["api-key"] = self.settings.api_key
        return headers

    def _payload(self, texts: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {"input": texts}
        if self.settings.dimensions is not None:
            payload["dimensions"] = self.settings.dimensions
        return payload


EmbeddingFactory.register_provider("azure", AzureOpenAIEmbedding)
