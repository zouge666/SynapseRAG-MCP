from __future__ import annotations

from collections.abc import Callable

from core.settings import EmbeddingSettings, Settings
from libs.embedding.base_embedding import BaseEmbedding


EmbeddingBuilder = Callable[[EmbeddingSettings], BaseEmbedding]


class EmbeddingFactory:
    _providers: dict[str, EmbeddingBuilder] = {}

    @classmethod
    def register_provider(cls, provider: str, builder: EmbeddingBuilder) -> None:
        key = cls._normalize_provider(provider)
        cls._providers[key] = builder

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        key = cls._normalize_provider(provider)
        cls._providers.pop(key, None)

    @classmethod
    def create(cls, settings: Settings | EmbeddingSettings) -> BaseEmbedding:
        embedding_settings = settings.embedding if isinstance(settings, Settings) else settings
        key = cls._normalize_provider(embedding_settings.provider)
        builder = cls._providers.get(key)
        if builder is None:
            raise ValueError(f"unsupported embedding provider: {embedding_settings.provider}")
        return builder(embedding_settings)

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        key = provider.strip().lower()
        if not key:
            raise ValueError("embedding.provider is required")
        return key
