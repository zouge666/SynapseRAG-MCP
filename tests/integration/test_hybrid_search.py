import pytest

from core import RetrievalResult
from core.query_engine import HybridSearch, HybridSearchError, RRFusion
from core.query_engine.query_processor import ProcessedQuery
from core.trace import TraceContext


class FakeQueryProcessor:
    def __init__(self, keywords: list[str] | None = None) -> None:
        self.keywords = keywords or ["rag", "graph"]
        self.calls: list[dict[str, object]] = []

    def process(self, query: str, filters: dict[str, object] | None = None, trace: object | None = None) -> ProcessedQuery:
        self.calls.append({"query": query, "filters": filters or {}, "trace": trace})
        return ProcessedQuery(query=query, normalized_query=query.lower(), keywords=list(self.keywords), filters=dict(filters or {}))


class FakeDenseRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters or {}, "trace": trace})
        if self.error is not None:
            raise self.error
        return list(self.results[:top_k])


class FakeSparseRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[dict[str, object]] = []

    def retrieve(self, keywords: list[str], top_k: int, trace: object | None = None) -> list[RetrievalResult]:
        self.calls.append({"keywords": keywords, "top_k": top_k, "trace": trace})
        if self.error is not None:
            raise self.error
        return list(self.results[:top_k])


def result(
    chunk_id: str,
    score: float,
    collection: str = "docs",
    doc_type: str = "pdf",
    text: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"text {chunk_id}",
        metadata={"source_path": f"docs/{chunk_id}.pdf", "collection": collection, "doc_type": doc_type},
    )


def test_hybrid_search_processes_routes_fuses_and_returns_text_metadata() -> None:
    query_processor = FakeQueryProcessor(keywords=["hybrid", "rag"])
    dense_retriever = FakeDenseRetriever([result("a", 0.9), result("b", 0.8)])
    sparse_retriever = FakeSparseRetriever([result("b", 9.0), result("c", 8.0)])
    search = HybridSearch({}, query_processor, dense_retriever, sparse_retriever, RRFusion(k=60))

    results = search.search("Hybrid RAG", top_k=3, filters={"collection": "docs"})

    assert [item.chunk_id for item in results] == ["b", "a", "c"]
    assert results[0].text == "text b"
    assert results[0].metadata["source_path"] == "docs/b.pdf"
    assert query_processor.calls[0]["filters"] == {"collection": "docs"}
    assert dense_retriever.calls[0]["query"] == "Hybrid RAG"
    assert dense_retriever.calls[0]["filters"] == {"collection": "docs"}
    assert sparse_retriever.calls[0]["keywords"] == ["hybrid", "rag"]


def test_hybrid_search_applies_metadata_filters_after_fusion() -> None:
    search = HybridSearch(
        {},
        FakeQueryProcessor(),
        FakeDenseRetriever([result("a", 0.9, collection="docs", doc_type="pdf"), result("b", 0.8, collection="notes", doc_type="pdf")]),
        FakeSparseRetriever([result("c", 6.0, collection="docs", doc_type="markdown")]),
        RRFusion(k=60),
    )

    results = search.search("metadata filters", top_k=5, filters={"collection": "docs", "doc_type": ["pdf", "markdown"]})

    assert [item.chunk_id for item in results] == ["a", "c"]


def test_hybrid_search_degrades_when_dense_route_fails() -> None:
    trace = TraceContext(trace_type="query")
    search = HybridSearch(
        {},
        FakeQueryProcessor(),
        FakeDenseRetriever(error=RuntimeError("dense down")),
        FakeSparseRetriever([result("s1", 5.0), result("s2", 4.0)]),
        RRFusion(k=60),
    )

    results = search.search("fallback", top_k=2, trace=trace)

    assert [item.chunk_id for item in results] == ["s1", "s2"]
    assert trace.stages[-1]["name"] == "hybrid_search"
    assert "dense down" in trace.stages[-1]["details"]["dense_error"]


def test_hybrid_search_degrades_when_sparse_route_fails() -> None:
    trace = TraceContext(trace_type="query")
    search = HybridSearch(
        {},
        FakeQueryProcessor(),
        FakeDenseRetriever([result("d1", 0.9), result("d2", 0.8)]),
        FakeSparseRetriever(error=RuntimeError("sparse down")),
        RRFusion(k=60),
    )

    results = search.search("fallback", top_k=2, trace=trace)

    assert [item.chunk_id for item in results] == ["d1", "d2"]
    assert "sparse down" in trace.stages[-1]["details"]["sparse_error"]


def test_hybrid_search_raises_when_both_routes_fail() -> None:
    search = HybridSearch(
        {},
        FakeQueryProcessor(),
        FakeDenseRetriever(error=RuntimeError("dense down")),
        FakeSparseRetriever(error=RuntimeError("sparse down")),
        RRFusion(k=60),
    )

    with pytest.raises(HybridSearchError, match="dense and sparse retrieval failed"):
        search.search("no route", top_k=2)


def test_hybrid_search_top_k_zero_returns_empty_without_work() -> None:
    query_processor = FakeQueryProcessor()
    dense_retriever = FakeDenseRetriever([result("a", 0.9)])
    sparse_retriever = FakeSparseRetriever([result("b", 3.0)])
    search = HybridSearch({}, query_processor, dense_retriever, sparse_retriever, RRFusion())

    assert search.search("skip", top_k=0) == []
    assert query_processor.calls == []
    assert dense_retriever.calls == []
    assert sparse_retriever.calls == []


def test_hybrid_search_can_be_imported_from_package() -> None:
    from core.query_engine import HybridSearch as ExportedHybridSearch

    assert ExportedHybridSearch is HybridSearch
