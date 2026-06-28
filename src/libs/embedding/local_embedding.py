from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class LocalEmbeddingError(ValueError):
    pass


class LocalEmbedding(BaseEmbedding):
    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        self._validate_texts(texts)
        dimensions = self._dimensions()
        vectors = [self._embed_text(text, dimensions) for text in texts]
        self._record(trace, "local_embedding", {"count": len(vectors), "dimension": dimensions})
        return vectors

    def _embed_text(self, text: str, dimensions: int) -> list[float]:
        values = [0.0] * dimensions
        for token, count in Counter(self._tokens(text)).items():
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % dimensions
            values[index] += float(count)
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [round(value / norm, 8) for value in values]

    def _tokens(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]", text.lower())
        return tokens or [text.strip().lower()]

    def _dimensions(self) -> int:
        value = self.settings.dimensions
        if not isinstance(value, int) or value <= 0:
            raise LocalEmbeddingError("local embedding dimensions must be a positive integer")
        return value

    def _validate_texts(self, texts: list[str]) -> None:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, list) or not texts:
            raise LocalEmbeddingError("local embedding texts must be a non-empty list")
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise LocalEmbeddingError(f"local embedding texts[{index}] must be string")

    def _record(self, trace: object | None, name: str, details: dict[str, int]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)


EmbeddingFactory.register_provider("local", LocalEmbedding)
