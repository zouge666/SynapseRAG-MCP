from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter
from typing import Any

from core import RetrievalResult
from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFusion
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever


class HybridSearchError(ValueError):
    pass


class HybridSearch:
    def __init__(
        self,
        settings: object,
        query_processor: QueryProcessor | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        fusion: RRFusion | None = None,
    ) -> None:
        self.settings = settings
        self.query_processor = query_processor or QueryProcessor()
        self.dense_retriever = dense_retriever or DenseRetriever(settings)
        self.sparse_retriever = sparse_retriever or SparseRetriever(settings)
        self.fusion = fusion or RRFusion()

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: object | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise HybridSearchError("query must be a non-empty string")
        if not isinstance(top_k, int) or top_k <= 0:
            return []
        query_start = perf_counter()
        processed = self.query_processor.process(query, filters=filters, trace=trace)
        self._record(
            trace,
            "query_processing",
            {
                "method": type(self.query_processor).__name__,
                "keyword_count": len(processed.keywords),
                "filters": dict(processed.filters),
            },
            self._elapsed_ms(query_start),
        )
        dense_results, dense_error, sparse_results, sparse_error = self._retrieve(processed, top_k, trace)
        if dense_error is not None and sparse_error is not None:
            raise HybridSearchError(f"dense and sparse retrieval failed: dense={dense_error}; sparse={sparse_error}")
        fused_top_k = max(top_k, len(dense_results) + len(sparse_results))
        fusion_start = perf_counter()
        fused = self.fusion.fuse(dense_results, sparse_results, top_k=fused_top_k, trace=trace)
        self._record(
            trace,
            "fusion",
            {
                "method": type(self.fusion).__name__,
                "count": len(fused),
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "top_k": fused_top_k,
            },
            self._elapsed_ms(fusion_start),
        )
        filtered = self._apply_metadata_filters(fused, processed.filters)
        results = filtered[:top_k]
        details: dict[str, Any] = {
            "count": len(results),
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "filtered_count": len(filtered),
            "top_k": top_k,
            "filters": dict(processed.filters),
        }
        if dense_error is not None:
            details["dense_error"] = str(dense_error)
        if sparse_error is not None:
            details["sparse_error"] = str(sparse_error)
        self._record(trace, "hybrid_search", details)
        return results

    def _apply_metadata_filters(
        self,
        candidates: list[RetrievalResult],
        filters: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        if not isinstance(candidates, list) or not all(isinstance(candidate, RetrievalResult) for candidate in candidates):
            raise HybridSearchError("candidates must be a list of RetrievalResult")
        if not filters:
            return list(candidates)
        if not isinstance(filters, dict):
            raise HybridSearchError("filters must be a mapping")
        return [candidate for candidate in candidates if self._matches_filters(candidate.metadata, filters)]

    def _retrieve(
        self,
        processed: object,
        top_k: int,
        trace: object | None,
    ) -> tuple[list[RetrievalResult], Exception | None, list[RetrievalResult], Exception | None]:
        dense_top_k = self._route_top_k("dense", top_k)
        sparse_top_k = self._route_top_k("sparse", top_k)
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_start = perf_counter()
            sparse_start = perf_counter()
            dense_future = executor.submit(self.dense_retriever.retrieve, processed.query, dense_top_k, processed.filters, trace)
            sparse_future = executor.submit(self.sparse_retriever.retrieve, processed.keywords, sparse_top_k, trace)
            dense_results, dense_error = self._future_results(dense_future, "dense")
            self._record_route(trace, "dense_retrieval", type(self.dense_retriever).__name__, dense_results, dense_error, dense_top_k, dense_start)
            sparse_results, sparse_error = self._future_results(sparse_future, "sparse")
            self._record_route(trace, "sparse_retrieval", type(self.sparse_retriever).__name__, sparse_results, sparse_error, sparse_top_k, sparse_start)
        return dense_results, dense_error, sparse_results, sparse_error

    def _future_results(self, future: Future[list[RetrievalResult]], route: str) -> tuple[list[RetrievalResult], Exception | None]:
        try:
            results = future.result()
            self._validate_results(results, route)
            return results, None
        except Exception as error:
            return [], error

    def _validate_results(self, results: list[RetrievalResult], route: str) -> None:
        if not isinstance(results, list) or not all(isinstance(result, RetrievalResult) for result in results):
            raise HybridSearchError(f"{route} results must be a list of RetrievalResult")

    def _route_top_k(self, route: str, requested: int) -> int:
        retrieval = self._settings_value("retrieval", {})
        key = f"top_k_{route}"
        value = None
        if isinstance(retrieval, dict):
            value = retrieval.get(key)
        else:
            value = getattr(retrieval, key, None)
        return max(requested, value) if isinstance(value, int) and value > 0 else requested

    def _settings_value(self, key: str, default: Any) -> Any:
        if isinstance(self.settings, dict):
            return self.settings.get(key, default)
        return getattr(self.settings, key, default)

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, expected in filters.items():
            if expected is None or key not in metadata:
                continue
            if not self._matches_value(metadata[key], expected):
                return False
        return True

    def _matches_value(self, actual: Any, expected: Any) -> bool:
        if isinstance(expected, list | tuple | set):
            return any(self._matches_value(actual, item) for item in expected)
        if isinstance(actual, list | tuple | set):
            return any(self._matches_value(item, expected) for item in actual)
        return actual == expected

    def _record_route(
        self,
        trace: object | None,
        name: str,
        method: str,
        results: list[RetrievalResult],
        error: Exception | None,
        top_k: int,
        start: float,
    ) -> None:
        details: dict[str, Any] = {"method": method, "count": len(results), "top_k": top_k}
        if error is not None:
            details["error"] = str(error)
        self._record(trace, name, details, self._elapsed_ms(start))

    def _elapsed_ms(self, start: float) -> float:
        return round((perf_counter() - start) * 1000, 3)

    def _record(self, trace: object | None, name: str, details: dict[str, Any], duration_ms: float | None = None) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details, duration_ms=duration_ms)
