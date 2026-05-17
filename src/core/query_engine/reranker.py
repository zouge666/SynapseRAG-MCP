from __future__ import annotations

from dataclasses import replace
from typing import Any

from core import RetrievalResult
from libs.reranker.base_reranker import BaseReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory


class RerankerError(ValueError):
    pass


class Reranker:
    def __init__(self, settings: object, backend: BaseReranker | None = None) -> None:
        self.settings = settings
        self.backend = backend

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        trace: object | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise RerankerError("query must be a non-empty string")
        self._validate_results(candidates)
        if not candidates:
            self._record(trace, {"count": 0, "fallback": False, "enabled": self._enabled(), "backend": self._backend_name()})
            return []
        if not self._enabled():
            results = list(candidates)
            self._record(trace, {"count": len(results), "fallback": False, "enabled": False, "backend": self._backend_name()})
            return results
        try:
            ranked = self._backend().rerank(query, self._to_candidates(candidates), trace=trace)
            results = self._to_results(ranked, candidates)
            self._record(trace, {"count": len(results), "fallback": False, "enabled": True, "backend": self._backend_name()})
            return results
        except Exception as error:
            results = list(candidates)
            self._record(
                trace,
                {
                    "count": len(results),
                    "fallback": True,
                    "enabled": True,
                    "backend": self._backend_name(),
                    "error": str(error),
                },
            )
            return results

    def _backend(self) -> BaseReranker:
        if self.backend is None:
            self.backend = RerankerFactory.create(self.settings)
        return self.backend

    def _to_candidates(self, results: list[RetrievalResult]) -> list[RerankCandidate]:
        return [
            RerankCandidate(id=result.chunk_id, text=result.text, score=float(result.score), metadata=dict(result.metadata))
            for result in results
        ]

    def _to_results(self, ranked: list[RerankCandidate], original: list[RetrievalResult]) -> list[RetrievalResult]:
        if not isinstance(ranked, list) or not all(isinstance(candidate, RerankCandidate) for candidate in ranked):
            raise RerankerError("reranker backend must return a list of RerankCandidate")
        original_by_id = {result.chunk_id: result for result in original}
        seen: set[str] = set()
        results = []
        for candidate in ranked:
            if candidate.id not in original_by_id:
                raise RerankerError(f"reranker backend returned unknown candidate: {candidate.id}")
            if candidate.id in seen:
                raise RerankerError(f"reranker backend returned duplicate candidate: {candidate.id}")
            seen.add(candidate.id)
            source = original_by_id[candidate.id]
            results.append(replace(source, score=float(candidate.score), metadata=dict(source.metadata)))
        for result in original:
            if result.chunk_id not in seen:
                results.append(result)
        return results

    def _validate_results(self, candidates: list[RetrievalResult]) -> None:
        if not isinstance(candidates, list) or not all(isinstance(candidate, RetrievalResult) for candidate in candidates):
            raise RerankerError("candidates must be a list of RetrievalResult")

    def _enabled(self) -> bool:
        rerank = self._rerank_settings()
        if isinstance(rerank, dict):
            return bool(rerank.get("enabled", True))
        return bool(getattr(rerank, "enabled", True))

    def _backend_name(self) -> str:
        rerank = self._rerank_settings()
        if isinstance(rerank, dict):
            value = rerank.get("backend", "none")
        else:
            value = getattr(rerank, "backend", "none")
        return value if isinstance(value, str) and value else "none"

    def _rerank_settings(self) -> Any:
        if isinstance(self.settings, dict):
            return self.settings.get("rerank", self.settings)
        return getattr(self.settings, "rerank", self.settings)

    def _record(self, trace: object | None, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage("reranker", details)
