from __future__ import annotations

from collections.abc import Callable

from core.settings import RerankSettings, Settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker


RerankerBuilder = Callable[[RerankSettings], BaseReranker]


class RerankerFactory:
    _providers: dict[str, RerankerBuilder] = {"none": NoneReranker}

    @classmethod
    def register_provider(cls, provider: str, builder: RerankerBuilder) -> None:
        key = cls._normalize_provider(provider)
        cls._providers[key] = builder

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        key = cls._normalize_provider(provider)
        if key != "none":
            cls._providers.pop(key, None)

    @classmethod
    def create(cls, settings: Settings | RerankSettings) -> BaseReranker:
        rerank_settings = settings.rerank if isinstance(settings, Settings) else settings
        key = cls._normalize_provider(rerank_settings.backend)
        builder = cls._providers.get(key)
        if builder is None:
            raise ValueError(f"unsupported reranker backend: {rerank_settings.backend}")
        return builder(rerank_settings)

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        key = provider.strip().lower()
        if not key:
            raise ValueError("rerank.backend is required")
        return key
