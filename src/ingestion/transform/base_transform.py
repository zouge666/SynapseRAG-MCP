from __future__ import annotations

from abc import ABC, abstractmethod

from core import Chunk


class BaseTransform(ABC):
    @abstractmethod
    def transform(self, chunks: list[Chunk], trace: object | None = None) -> list[Chunk]:
        raise NotImplementedError
