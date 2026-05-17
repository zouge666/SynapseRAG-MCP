import pytest

from core import RetrievalResult
from core.query_engine import Reranker, RerankerError
from core.settings import RerankSettings
from core.trace import TraceContext
from libs.reranker.base_reranker import BaseReranker, RerankCandidate


class ScoreDescendingBackend(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        return sorted(
            [
                RerankCandidate(id=candidate.id, text=candidate.text, score=candidate.score + 10, metadata=dict(candidate.metadata))
                for candidate in candidates
            ],
            key=lambda candidate: (-candidate.score, candidate.id),
        )


class FailingBackend(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        raise RuntimeError("rerank unavailable")


class UnknownBackend(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        return [RerankCandidate(id="missing", text="missing", score=99)]


def result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text {chunk_id}",
        metadata={"source_path": f"docs/{chunk_id}.pdf"},
    )


def settings(enabled: bool = True, backend: str = "fake", top_m: int = 30) -> RerankSettings:
    return RerankSettings(enabled=enabled, backend=backend, top_m=top_m)


def test_core_reranker_orders_candidates_and_preserves_result_shape() -> None:
    trace = TraceContext(trace_type="query")
    candidates = [result("a", 0.2), result("b", 0.8), result("c", 0.5)]
    reranker = Reranker(settings(), backend=ScoreDescendingBackend(settings()))

    ranked = reranker.rerank("find best", candidates, trace=trace)

    assert [item.chunk_id for item in ranked] == ["b", "c", "a"]
    assert ranked[0].score == pytest.approx(10.8)
    assert ranked[0].text == "text b"
    assert ranked[0].metadata == {"source_path": "docs/b.pdf"}
    assert trace.stages[-1]["name"] == "reranker"
    assert trace.stages[-1]["details"] == {"count": 3, "fallback": False, "enabled": True, "backend": "fake"}


def test_core_reranker_disabled_keeps_original_order_without_backend_call() -> None:
    trace = TraceContext(trace_type="query")
    candidates = [result("a", 0.2), result("b", 0.8)]
    reranker = Reranker(settings(enabled=False), backend=FailingBackend(settings()))

    ranked = reranker.rerank("find best", candidates, trace=trace)

    assert ranked == candidates
    assert ranked is not candidates
    assert trace.stages[-1]["details"] == {"count": 2, "fallback": False, "enabled": False, "backend": "fake"}


def test_core_reranker_fallback_keeps_candidates_when_backend_fails() -> None:
    trace = TraceContext(trace_type="query")
    candidates = [result("a", 0.2), result("b", 0.8)]
    reranker = Reranker(settings(backend="cross_encoder"), backend=FailingBackend(settings(backend="cross_encoder")))

    ranked = reranker.rerank("find best", candidates, trace=trace)

    assert ranked == candidates
    assert ranked is not candidates
    assert trace.stages[-1]["details"]["fallback"] is True
    assert trace.stages[-1]["details"]["backend"] == "cross_encoder"
    assert trace.stages[-1]["details"]["error"] == "rerank unavailable"


def test_core_reranker_fallback_handles_invalid_backend_output() -> None:
    trace = TraceContext(trace_type="query")
    candidates = [result("a", 0.2)]
    reranker = Reranker(settings(), backend=UnknownBackend(settings()))

    ranked = reranker.rerank("find best", candidates, trace=trace)

    assert ranked == candidates
    assert trace.stages[-1]["details"]["fallback"] is True
    assert "unknown candidate" in trace.stages[-1]["details"]["error"]


def test_core_reranker_validates_inputs() -> None:
    reranker = Reranker(settings(), backend=ScoreDescendingBackend(settings()))

    with pytest.raises(RerankerError, match="query"):
        reranker.rerank("", [result("a", 0.2)])
    with pytest.raises(RerankerError, match="candidates"):
        reranker.rerank("query", ["bad"])


def test_core_reranker_empty_candidates_records_trace() -> None:
    trace = TraceContext(trace_type="query")
    reranker = Reranker(settings(), backend=ScoreDescendingBackend(settings()))

    assert reranker.rerank("query", [], trace=trace) == []
    assert trace.stages[-1]["details"] == {"count": 0, "fallback": False, "enabled": True, "backend": "fake"}


def test_core_reranker_can_be_imported_from_package() -> None:
    from core.query_engine import Reranker as ExportedReranker

    assert ExportedReranker is Reranker
