from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.settings import RerankSettings


@dataclass(frozen=True)
class RerankCandidate:
    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    def __init__(self, settings: RerankSettings) -> None:
        self.settings = settings

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        raise NotImplementedError


class NoneReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        return list(candidates)
