from typing import Any

import pytest

from core.settings import EmbeddingSettings
from libs.embedding.azure_embedding import AzureOpenAIEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAIEmbedding, OpenAIEmbeddingError


class FakeTransport:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {"data": [{"embedding": [1, 2.5]}, {"embedding": [3, 4]}]}
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((url, headers, payload, timeout))
        return self.response


def test_factory_routes_openai_and_azure_embedding() -> None:
    assert isinstance(EmbeddingFactory.create(EmbeddingSettings(provider="openai", model="text-embedding-3-small")), OpenAIEmbedding)
    assert isinstance(
        EmbeddingFactory.create(
            EmbeddingSettings(
                provider="azure",
                model="text-embedding-ada-002",
                azure_endpoint="https://example.openai.azure.com",
                deployment_name="embeddings",
            )
        ),
        AzureOpenAIEmbedding,
    )


def test_openai_embed_uses_openai_embeddings_payload() -> None:
    transport = FakeTransport()
    embedding = OpenAIEmbedding(
        EmbeddingSettings(provider="openai", model="text-embedding-3-small", api_key="secret", dimensions=2),
        transport=transport,
    )

    result = embedding.embed(["hello", "world"])

    url, headers, payload, timeout = transport.calls[0]
    assert result == [[1.0, 2.5], [3.0, 4.0]]
    assert url == "https://api.openai.com/v1/embeddings"
    assert headers["Authorization"] == "Bearer secret"
    assert payload == {"model": "text-embedding-3-small", "input": ["hello", "world"], "dimensions": 2}
    assert timeout == 30.0


def test_openai_embed_uses_custom_base_url() -> None:
    transport = FakeTransport({"data": [{"embedding": [0.1]}]})
    embedding = OpenAIEmbedding(
        EmbeddingSettings(provider="openai", model="text-embedding-3-small", base_url="https://gateway.example/v1/"),
        transport=transport,
    )

    result = embedding.embed(["hello"])

    assert result == [[0.1]]
    assert transport.calls[0][0] == "https://gateway.example/v1/embeddings"


def test_azure_embed_uses_deployment_endpoint_and_api_key_header() -> None:
    transport = FakeTransport({"data": [{"embedding": [1]}, {"embedding": [2]}]})
    embedding = AzureOpenAIEmbedding(
        EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            api_key="secret",
            azure_endpoint="https://example.openai.azure.com/",
            api_version="2024-05-01-preview",
            deployment_name="embedding deployment",
        ),
        transport=transport,
    )

    result = embedding.embed(["hello", "world"])

    url, headers, payload, _ = transport.calls[0]
    assert result == [[1.0], [2.0]]
    assert url == "https://example.openai.azure.com/openai/deployments/embedding%20deployment/embeddings?api-version=2024-05-01-preview"
    assert headers["api-key"] == "secret"
    assert "Authorization" not in headers
    assert payload == {"input": ["hello", "world"]}


def test_embed_validation_error_mentions_provider_and_error_type() -> None:
    embedding = OpenAIEmbedding(EmbeddingSettings(provider="openai", model="text-embedding-3-small"), transport=FakeTransport())

    with pytest.raises(OpenAIEmbeddingError, match="openai validation error"):
        embedding.embed([])


def test_embed_response_error_mentions_provider_and_error_type() -> None:
    embedding = OpenAIEmbedding(
        EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        transport=FakeTransport({"data": [{"embedding": [1.0]}]}),
    )

    with pytest.raises(OpenAIEmbeddingError, match="openai response error"):
        embedding.embed(["hello", "world"])


def test_embed_transport_error_hides_api_key() -> None:
    def failing_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise OSError("secret")

    embedding = OpenAIEmbedding(
        EmbeddingSettings(provider="openai", model="text-embedding-3-small", api_key="secret"),
        transport=failing_transport,
    )

    with pytest.raises(OpenAIEmbeddingError) as error:
        embedding.embed(["hello"])

    assert "secret" not in str(error.value)
