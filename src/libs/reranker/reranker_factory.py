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
        if key == "llm" and isinstance(settings, Settings):
            return cls._create_llm_reranker(settings, rerank_settings)
        builder = cls._providers.get(key)
        if builder is None:
            cls._load_builtin_provider(key)
            builder = cls._providers.get(key)
        if builder is None:
            raise ValueError(f"unsupported reranker backend: {rerank_settings.backend}")
        return builder(rerank_settings)

    @staticmethod
    def _create_llm_reranker(settings: Settings, rerank_settings: RerankSettings) -> BaseReranker:
        from dataclasses import replace

        from libs.llm.llm_factory import LLMFactory
        from libs.reranker.llm_reranker import LLMReranker

        llm_settings = settings.llm
        if not llm_settings.api_key:
            raise ValueError("rerank llm backend requires an LLM API key; configure it in the LLM section of Settings")
        if rerank_settings.model.strip():
            llm_settings = replace(llm_settings, model=rerank_settings.model.strip())
        return LLMReranker(rerank_settings, llm=LLMFactory.create(llm_settings))

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        key = provider.strip().lower()
        if not key:
            raise ValueError("rerank.backend is required")
        return key

    @staticmethod
    def _load_builtin_provider(provider: str) -> None:
        if provider == "llm":
            from libs.reranker.llm_reranker import LLMReranker

            RerankerFactory.register_provider("llm", LLMReranker)
        elif provider == "cross_encoder":
            from libs.reranker.cross_encoder_reranker import CrossEncoderReranker

            RerankerFactory.register_provider("cross_encoder", CrossEncoderReranker)
