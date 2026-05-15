from typing import Any
from urllib.error import URLError

import pytest

from core.settings import EmbeddingSettings
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.ollama_embedding import OllamaEmbedding, OllamaEmbeddingError


class FakeTransport:
    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.response = response or {"embeddings": [[1, 2.5], [3, 4]]}
        self.error = error
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((url, headers, payload, timeout))
        if self.error:
            raise self.error
        return self.response


def test_factory_routes_ollama_embedding_provider() -> None:
    embedding = EmbeddingFactory.create(EmbeddingSettings(provider="ollama", model="nomic-embed-text"))

    assert isinstance(embedding, OllamaEmbedding)


def test_ollama_embed_uses_default_endpoint_and_batch_payload() -> None:
    transport = FakeTransport()
    embedding = OllamaEmbedding(EmbeddingSettings(provider="ollama", model="nomic-embed-text"), transport=transport)

    result = embedding.embed(["hello", "world"])

    url, headers, payload, timeout = transport.calls[0]
    assert result == [[1.0, 2.5], [3.0, 4.0]]
    assert url == "http://localhost:11434/api/embed"
    assert headers == {"Content-Type": "application/json"}
    assert payload == {"model": "nomic-embed-text", "input": ["hello", "world"]}
    assert timeout == 30.0


def test_ollama_embed_uses_configured_base_url() -> None:
    transport = FakeTransport({"embeddings": [[0.1]]})
    embedding = OllamaEmbedding(
        EmbeddingSettings(provider="ollama", model="mxbai-embed-large", base_url="http://127.0.0.1:11435/"),
        transport=transport,
    )

    result = embedding.embed(["hello"])

    assert result == [[0.1]]
    assert transport.calls[0][0] == "http://127.0.0.1:11435/api/embed"


def test_ollama_embed_accepts_single_embedding_response() -> None:
    transport = FakeTransport({"embedding": [1, 2, 3]})
    embedding = OllamaEmbedding(EmbeddingSettings(provider="ollama", model="nomic-embed-text"), transport=transport)

    result = embedding.embed(["hello"])

    assert result == [[1.0, 2.0, 3.0]]


def test_ollama_connection_error_is_readable_and_does_not_leak_config() -> None:
    transport = FakeTransport(error=URLError("connection refused"))
    embedding = OllamaEmbedding(
        EmbeddingSettings(provider="ollama", model="secret-model", api_key="secret-key", base_url="http://private-host:11434"),
        transport=transport,
    )

    with pytest.raises(OllamaEmbeddingError) as error:
        embedding.embed(["hello"])

    message = str(error.value)
    assert "ollama connection error" in message
    assert "secret-key" not in message
    assert "private-host" not in message


def test_ollama_validation_error_is_readable() -> None:
    embedding = OllamaEmbedding(EmbeddingSettings(provider="ollama", model="nomic-embed-text"), transport=FakeTransport())

    with pytest.raises(OllamaEmbeddingError, match="ollama validation error"):
        embedding.embed([])


def test_ollama_response_error_is_readable() -> None:
    embedding = OllamaEmbedding(
        EmbeddingSettings(provider="ollama", model="nomic-embed-text"),
        transport=FakeTransport({"embeddings": [[1.0]]}),
    )

    with pytest.raises(OllamaEmbeddingError, match="ollama response error"):
        embedding.embed(["hello", "world"])
