from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core import RetrievalResult


class CitationGeneratorError(ValueError):
    pass


@dataclass(frozen=True)
class Citation:
    id: int
    source: str
    page: int | str | None
    chunk_id: str
    score: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
        }


class CitationGenerator:
    def generate(self, retrieval_results: list[RetrievalResult]) -> list[Citation]:
        if not isinstance(retrieval_results, list) or not all(isinstance(result, RetrievalResult) for result in retrieval_results):
            raise CitationGeneratorError("retrieval_results must be a list of RetrievalResult")
        return [self._citation(index, result) for index, result in enumerate(retrieval_results, start=1)]

    def _citation(self, index: int, result: RetrievalResult) -> Citation:
        metadata = result.metadata
        source = metadata.get("source_path") or metadata.get("source") or "unknown"
        page = metadata.get("page", metadata.get("page_number"))
        return Citation(
            id=index,
            source=str(source),
            page=page if isinstance(page, int | str) else None,
            chunk_id=result.chunk_id,
            score=float(result.score),
            text=result.text,
        )
