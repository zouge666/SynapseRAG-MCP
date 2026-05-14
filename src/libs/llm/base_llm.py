from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from core.settings import LLMSettings


class BaseLLM(ABC):
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    @abstractmethod
    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        raise NotImplementedError
