from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.settings import LLMSettings


@dataclass(frozen=True)
class VisionLLMResponse:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVisionLLM(ABC):
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: object | None = None,
    ) -> VisionLLMResponse:
        raise NotImplementedError
