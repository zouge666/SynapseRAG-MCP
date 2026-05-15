from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings, VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore


VectorStoreBuilder = Callable[[VectorStoreSettings], BaseVectorStore]


class VectorStoreFactory:
    _providers: dict[str, VectorStoreBuilder] = {}

    @classmethod
    def register_provider(cls, provider: str, builder: VectorStoreBuilder) -> None:
        key = cls._normalize_provider(provider)
        cls._providers[key] = builder

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        key = cls._normalize_provider(provider)
        cls._providers.pop(key, None)

    @classmethod
    def create(cls, settings: Settings | VectorStoreSettings) -> BaseVectorStore:
        vector_store_settings = settings.vector_store if isinstance(settings, Settings) else settings
        key = cls._normalize_provider(vector_store_settings.backend)
        builder = cls._providers.get(key)
        if builder is None:
            cls._load_builtin_provider(key)
            builder = cls._providers.get(key)
        if builder is None:
            raise ValueError(f"unsupported vector store backend: {vector_store_settings.backend}")
        return builder(vector_store_settings)

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        key = provider.strip().lower()
        if not key:
            raise ValueError("vector_store.backend is required")
        return key

    @staticmethod
    def _load_builtin_provider(provider: str) -> None:
        if provider == "chroma":
            from libs.vector_store.chroma_store import ChromaStore

            VectorStoreFactory.register_provider("chroma", ChromaStore)
