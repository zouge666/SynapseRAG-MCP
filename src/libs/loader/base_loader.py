from __future__ import annotations

from abc import ABC, abstractmethod

from core import Document


class LoaderError(RuntimeError):
    pass


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Document:
        raise NotImplementedError
