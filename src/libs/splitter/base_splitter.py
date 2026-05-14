from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSplitter(ABC):
    def __init__(self, settings: object) -> None:
        self.settings = settings

    @abstractmethod
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        raise NotImplementedError
