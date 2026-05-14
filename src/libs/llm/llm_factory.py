from __future__ import annotations

from collections.abc import Callable

from core.settings import LLMSettings, Settings
from libs.llm.base_llm import BaseLLM


LLMBuilder = Callable[[LLMSettings], BaseLLM]


class LLMFactory:
    _providers: dict[str, LLMBuilder] = {}

    @classmethod
    def register_provider(cls, provider: str, builder: LLMBuilder) -> None:
        key = cls._normalize_provider(provider)
        cls._providers[key] = builder

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        key = cls._normalize_provider(provider)
        cls._providers.pop(key, None)

    @classmethod
    def create(cls, settings: Settings | LLMSettings) -> BaseLLM:
        llm_settings = settings.llm if isinstance(settings, Settings) else settings
        key = cls._normalize_provider(llm_settings.provider)
        builder = cls._providers.get(key)
        if builder is None:
            raise ValueError(f"unsupported LLM provider: {llm_settings.provider}")
        return builder(llm_settings)

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        key = provider.strip().lower()
        if not key:
            raise ValueError("llm.provider is required")
        return key
