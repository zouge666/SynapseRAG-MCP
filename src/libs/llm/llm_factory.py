from __future__ import annotations

from collections.abc import Callable

from core.settings import LLMSettings, Settings
from libs.llm.base_llm import BaseLLM
from libs.llm.base_vision_llm import BaseVisionLLM


LLMBuilder = Callable[[LLMSettings], BaseLLM]
VisionLLMBuilder = Callable[[LLMSettings], BaseVisionLLM]


class LLMFactory:
    _providers: dict[str, LLMBuilder] = {}
    _vision_providers: dict[str, VisionLLMBuilder] = {}

    @classmethod
    def register_provider(cls, provider: str, builder: LLMBuilder) -> None:
        key = cls._normalize_provider(provider)
        cls._providers[key] = builder

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        key = cls._normalize_provider(provider)
        cls._providers.pop(key, None)

    @classmethod
    def register_vision_provider(cls, provider: str, builder: VisionLLMBuilder) -> None:
        key = cls._normalize_provider(provider)
        cls._vision_providers[key] = builder

    @classmethod
    def unregister_vision_provider(cls, provider: str) -> None:
        key = cls._normalize_provider(provider)
        cls._vision_providers.pop(key, None)

    @classmethod
    def create(cls, settings: Settings | LLMSettings) -> BaseLLM:
        llm_settings = settings.llm if isinstance(settings, Settings) else settings
        key = cls._normalize_provider(llm_settings.provider)
        builder = cls._providers.get(key)
        if builder is None:
            cls._load_builtin_provider(key)
            builder = cls._providers.get(key)
        if builder is None:
            raise ValueError(f"unsupported LLM provider: {llm_settings.provider}")
        return builder(llm_settings)

    @classmethod
    def create_vision_llm(cls, settings: Settings | LLMSettings) -> BaseVisionLLM:
        llm_settings = settings.vision_llm or settings.llm if isinstance(settings, Settings) else settings
        key = cls._normalize_provider(llm_settings.provider)
        builder = cls._vision_providers.get(key)
        if builder is None:
            cls._load_builtin_vision_provider(key)
            builder = cls._vision_providers.get(key)
        if builder is None:
            raise ValueError(f"unsupported vision LLM provider: {llm_settings.provider}")
        return builder(llm_settings)

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        key = provider.strip().lower()
        if not key:
            raise ValueError("llm.provider is required")
        return key

    @staticmethod
    def _load_builtin_provider(provider: str) -> None:
        if provider == "openai":
            from libs.llm.openai_llm import OpenAILLM

            LLMFactory.register_provider("openai", OpenAILLM)
        elif provider == "azure":
            from libs.llm.azure_llm import AzureOpenAILLM

            LLMFactory.register_provider("azure", AzureOpenAILLM)
        elif provider == "deepseek":
            from libs.llm.deepseek_llm import DeepSeekLLM

            LLMFactory.register_provider("deepseek", DeepSeekLLM)
        elif provider == "ollama":
            from libs.llm.ollama_llm import OllamaLLM

            LLMFactory.register_provider("ollama", OllamaLLM)

    @staticmethod
    def _load_builtin_vision_provider(provider: str) -> None:
        if provider == "azure":
            from libs.llm.azure_vision_llm import AzureVisionLLM

            LLMFactory.register_vision_provider("azure", AzureVisionLLM)
