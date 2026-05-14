from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    retrieved_ids: list[str]
    golden_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    def __init__(self, settings: object | None = None) -> None:
        self.settings = settings

    @abstractmethod
    def evaluate(self, case: EvaluationCase, trace: object | None = None) -> dict[str, float]:
        raise NotImplementedError
