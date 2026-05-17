from __future__ import annotations

from dataclasses import replace
from typing import Any

from core import RetrievalResult


class RRFusionError(ValueError):
    pass


class RRFusion:
    def __init__(self, k: int = 60) -> None:
        if not isinstance(k, int) or k <= 0:
            raise RRFusionError("k must be a positive integer")
        self.k = k

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
        trace: object | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(top_k, int) or top_k <= 0:
            return []
        self._validate_results(dense_results, "dense_results")
        self._validate_results(sparse_results, "sparse_results")
        scores: dict[str, float] = {}
        records: dict[str, RetrievalResult] = {}
        ranks: dict[str, dict[str, int]] = {}
        self._accumulate("dense", dense_results, scores, records, ranks)
        self._accumulate("sparse", sparse_results, scores, records, ranks)
        fused = [
            self._fused_result(records[chunk_id], scores[chunk_id], ranks[chunk_id])
            for chunk_id in sorted(scores, key=lambda item: (-scores[item], item))
        ]
        result = fused[:top_k]
        self._record(trace, "fusion.rrf", {"count": len(result), "dense_count": len(dense_results), "sparse_count": len(sparse_results), "k": self.k})
        return result

    def _accumulate(
        self,
        source: str,
        results: list[RetrievalResult],
        scores: dict[str, float],
        records: dict[str, RetrievalResult],
        ranks: dict[str, dict[str, int]],
    ) -> None:
        for index, result in enumerate(results, start=1):
            chunk_id = result.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.k + index)
            ranks.setdefault(chunk_id, {})[source] = index
            current = records.get(chunk_id)
            if current is None or result.score > current.score:
                records[chunk_id] = result

    def _fused_result(self, result: RetrievalResult, score: float, ranks: dict[str, int]) -> RetrievalResult:
        metadata = dict(result.metadata)
        metadata["rrf_ranks"] = dict(ranks)
        return replace(result, score=score, metadata=metadata)

    def _validate_results(self, results: list[RetrievalResult], name: str) -> None:
        if not isinstance(results, list) or not all(isinstance(result, RetrievalResult) for result in results):
            raise RRFusionError(f"{name} must be a list of RetrievalResult")

    def _record(self, trace: object | None, name: str, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)
