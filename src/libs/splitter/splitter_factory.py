from __future__ import annotations

from collections.abc import Callable, Mapping

from libs.splitter.base_splitter import BaseSplitter


SplitterBuilder = Callable[[object], BaseSplitter]


class SplitterFactory:
    _providers: dict[str, SplitterBuilder] = {}

    @classmethod
    def register_provider(cls, provider: str, builder: SplitterBuilder) -> None:
        key = cls._normalize_provider(provider)
        cls._providers[key] = builder

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        key = cls._normalize_provider(provider)
        cls._providers.pop(key, None)

    @classmethod
    def create(cls, settings: object) -> BaseSplitter:
        splitter_settings = cls._splitter_settings(settings)
        provider = cls._provider(splitter_settings)
        key = cls._normalize_provider(provider)
        builder = cls._providers.get(key)
        if builder is None:
            raise ValueError(f"unsupported splitter provider: {provider}")
        return builder(splitter_settings)

    @staticmethod
    def _splitter_settings(settings: object) -> object:
        if isinstance(settings, Mapping):
            return settings.get("splitter", settings)
        return getattr(settings, "splitter", settings)

    @staticmethod
    def _provider(settings: object) -> str:
        if isinstance(settings, Mapping):
            value = settings.get("provider", settings.get("backend"))
        else:
            value = getattr(settings, "provider", getattr(settings, "backend", None))
        if not isinstance(value, str):
            raise ValueError("splitter.provider is required")
        return value

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        key = provider.strip().lower()
        if not key:
            raise ValueError("splitter.provider is required")
        return key
