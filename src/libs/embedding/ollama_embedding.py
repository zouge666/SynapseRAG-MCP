from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


OllamaEmbeddingTransport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class OllamaEmbeddingError(RuntimeError):
    pass


class OllamaEmbedding(BaseEmbedding):
    default_base_url = "http://localhost:11434"

    def __init__(self, settings: EmbeddingSettings, transport: OllamaEmbeddingTransport | None = None, timeout: float = 30.0) -> None:
        super().__init__(settings)
        self.transport = transport or self._default_transport
        self.timeout = timeout

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        self._validate_texts(texts)
        response = self._send(self._embed_url(), {"Content-Type": "application/json"}, {"model": self.settings.model, "input": texts})
        return self._extract_embeddings(response, len(texts))

    def _embed_url(self) -> str:
        return f"{self._base_url()}/api/embed"

    def _base_url(self) -> str:
        return (self.settings.base_url or self.default_base_url).rstrip("/")

    def _send(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.transport(url, headers, payload, self.timeout)
        except OllamaEmbeddingError:
            raise
        except HTTPError as error:
            raise OllamaEmbeddingError(f"ollama http error: {error.code}") from error
        except URLError as error:
            raise OllamaEmbeddingError(f"ollama connection error: {error.reason}") from error
        except OSError as error:
            raise OllamaEmbeddingError(f"ollama transport error: {type(error).__name__}") from error

    def _default_transport(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            data = response.read().decode("utf-8")
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise OllamaEmbeddingError("ollama response error: expected object")
        return parsed

    def _extract_embeddings(self, response: dict[str, Any], expected_count: int) -> list[list[float]]:
        embeddings = response.get("embeddings")
        if embeddings is None and expected_count == 1:
            embeddings = [response.get("embedding")]
        if not isinstance(embeddings, list):
            raise OllamaEmbeddingError("ollama response error: embeddings must be list")
        if len(embeddings) != expected_count:
            raise OllamaEmbeddingError("ollama response error: embedding count mismatch")
        result: list[list[float]] = []
        for index, embedding in enumerate(embeddings):
            if not isinstance(embedding, list) or not all(isinstance(value, int | float) for value in embedding):
                raise OllamaEmbeddingError(f"ollama response error: embeddings[{index}] must be numeric list")
            result.append([float(value) for value in embedding])
        return result

    def _validate_texts(self, texts: list[str]) -> None:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, list) or not texts:
            raise OllamaEmbeddingError("ollama validation error: texts must be a non-empty list")
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise OllamaEmbeddingError(f"ollama validation error: texts[{index}] must be string")


EmbeddingFactory.register_provider("ollama", OllamaEmbedding)
