from types import SimpleNamespace

from core import RetrievalResult
from observability.dashboard.pages.query_console import run_dashboard_query


class FakeSearch:
    def search(self, query, top_k, filters=None, trace=None):
        trace.record_stage("hybrid_search", {"count": 1, "filters": filters or {}})
        return [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.75,
                text="海盐桂花拿铁售价36元。",
                metadata={"source_path": "pdfs/cafe.pdf", "chunk_index": 0},
            )
        ]


class FakeReranker:
    def rerank(self, query, candidates, trace=None):
        trace.record_stage("rerank", {"count": len(candidates), "enabled": False})
        return list(candidates)


def test_dashboard_query_returns_results_and_persists_trace() -> None:
    written = []
    settings = SimpleNamespace(
        rerank=SimpleNamespace(enabled=False),
        observability=SimpleNamespace(trace_path="logs/test-traces.jsonl"),
    )

    result = run_dashboard_query(
        settings,
        "海盐桂花拿铁售价多少？",
        "default",
        5,
        search=FakeSearch(),
        reranker=FakeReranker(),
        trace_writer=lambda trace, path: written.append((trace, path)) or trace,
    )

    assert result.results[0].text == "海盐桂花拿铁售价36元。"
    assert result.rerank_applied is False
    assert written[0][0]["trace_type"] == "query"
    assert written[0][0]["status"] == "success"
    assert written[0][1] == "logs/test-traces.jsonl"
