from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.settings import VectorStoreSettings


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    def __init__(self, settings: VectorStoreSettings) -> None:
        self.settings = settings

    @abstractmethod
    def upsert(self, records: list[VectorRecord], trace: object | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: object | None = None,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError
