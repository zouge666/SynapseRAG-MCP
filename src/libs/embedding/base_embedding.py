from __future__ import annotations

from abc import ABC, abstractmethod

from core.settings import EmbeddingSettings


class BaseEmbedding(ABC):
    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    @abstractmethod
    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        raise NotImplementedError
