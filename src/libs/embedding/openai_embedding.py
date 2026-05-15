from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


EmbeddingTransport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class OpenAIEmbeddingError(RuntimeError):
    pass


class OpenAIEmbedding(BaseEmbedding):
    default_base_url = "https://api.openai.com/v1"

    def __init__(self, settings: EmbeddingSettings, transport: EmbeddingTransport | None = None, timeout: float = 30.0) -> None:
        super().__init__(settings)
        self.transport = transport or self._default_transport
        self.timeout = timeout

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        self._validate_texts(texts)
        response = self._send(self._embeddings_url(), self._headers(), self._payload(texts))
        return self._extract_embeddings(response, len(texts))

    def _embeddings_url(self) -> str:
        return f"{self._base_url()}/embeddings"

    def _base_url(self) -> str:
        return (self.settings.base_url or self.default_base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _payload(self, texts: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.settings.model, "input": texts}
        if self.settings.dimensions is not None:
            payload["dimensions"] = self.settings.dimensions
        return payload

    def _send(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.transport(url, headers, payload, self.timeout)
        except OpenAIEmbeddingError:
            raise
        except HTTPError as error:
            raise OpenAIEmbeddingError(f"{self.settings.provider} http error: {error.code}") from error
        except URLError as error:
            raise OpenAIEmbeddingError(f"{self.settings.provider} connection error: {error.reason}") from error
        except OSError as error:
            raise OpenAIEmbeddingError(f"{self.settings.provider} transport error: {type(error).__name__}") from error

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
            raise OpenAIEmbeddingError(f"{self.settings.provider} response error: expected object")
        return parsed

    def _extract_embeddings(self, response: dict[str, Any], expected_count: int) -> list[list[float]]:
        data = response.get("data")
        if not isinstance(data, list):
            raise OpenAIEmbeddingError(f"{self.settings.provider} response error: data must be list")
        if len(data) != expected_count:
            raise OpenAIEmbeddingError(f"{self.settings.provider} response error: embedding count mismatch")
        embeddings: list[list[float]] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise OpenAIEmbeddingError(f"{self.settings.provider} response error: data[{index}] must be object")
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not all(isinstance(value, int | float) for value in embedding):
                raise OpenAIEmbeddingError(f"{self.settings.provider} response error: data[{index}].embedding must be numeric list")
            embeddings.append([float(value) for value in embedding])
        return embeddings

    def _validate_texts(self, texts: list[str]) -> None:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, list) or not texts:
            raise OpenAIEmbeddingError(f"{self.settings.provider} validation error: texts must be a non-empty list")
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise OpenAIEmbeddingError(f"{self.settings.provider} validation error: texts[{index}] must be string")


EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
