from types import SimpleNamespace

from core import RetrievalResult
from core.settings import LLMSettings
from observability.dashboard.pages.query_console import _dimension_warning, run_dashboard_answer, run_dashboard_query


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


class EmptySearch:
    def search(self, query, top_k, filters=None, trace=None):
        return []


class FakeReranker:
    def rerank(self, query, candidates, trace=None):
        trace.record_stage("rerank", {"count": len(candidates), "enabled": False})
        return list(candidates)


class FallbackReranker:
    def rerank(self, query, candidates, trace=None):
        trace.record_stage("rerank", {"count": len(candidates), "enabled": True, "backend": "llm", "fallback": True, "error": "llm reranker fallback: boom"})
        return list(candidates)


class FakeLLM:
    def __init__(self, reply="拿铁36元 [1]。", error=None):
        self.settings = LLMSettings(provider="fake", model="fake-1", api_key="x")
        self.reply = reply
        self.error = error

    def chat(self, messages):
        if self.error is not None:
            raise self.error
        return self.reply


def make_settings():
    return SimpleNamespace(
        rerank=SimpleNamespace(enabled=False),
        observability=SimpleNamespace(trace_path="logs/test-traces.jsonl"),
        llm=LLMSettings(provider="fake", model="fake-1", api_key="x"),
    )


def collect(written):
    return lambda trace, path: written.append((trace, path)) or trace


def test_dashboard_query_returns_results_and_persists_trace() -> None:
    written = []

    result = run_dashboard_query(
        make_settings(),
        "海盐桂花拿铁售价多少？",
        "default",
        5,
        search=FakeSearch(),
        reranker=FakeReranker(),
        trace_writer=collect(written),
    )

    assert result.results[0].text == "海盐桂花拿铁售价36元。"
    assert result.rerank_applied is False
    assert result.answer is None
    assert written[0][0]["trace_type"] == "query"
    assert written[0][0]["status"] == "success"
    assert written[0][1] == "logs/test-traces.jsonl"


def test_dashboard_answer_returns_llm_answer_with_generation_stage() -> None:
    written = []

    result = run_dashboard_answer(
        make_settings(),
        "海盐桂花拿铁售价多少？",
        "default",
        5,
        llm=FakeLLM(),
        search=FakeSearch(),
        reranker=FakeReranker(),
        trace_writer=collect(written),
    )

    assert result.answer is not None
    assert result.answer.answer == "拿铁36元 [1]。"
    assert result.answer_error is None
    trace = written[0][0]
    assert trace["metadata"]["generate_answer"] is True
    assert trace["status"] == "success"
    stage_names = [stage["name"] for stage in trace["stages"]]
    assert stage_names == ["hybrid_search", "rerank", "generation"]


def test_dashboard_answer_llm_failure_keeps_retrieval_results() -> None:
    written = []

    result = run_dashboard_answer(
        make_settings(),
        "海盐桂花拿铁售价多少？",
        "default",
        5,
        llm=FakeLLM(error=RuntimeError("http 500")),
        search=FakeSearch(),
        reranker=FakeReranker(),
        trace_writer=collect(written),
    )

    assert result.answer is None
    assert "generation failed" in result.answer_error
    assert len(result.results) == 1
    trace = written[0][0]
    assert trace["status"] == "success"
    assert "generation.error" in [stage["name"] for stage in trace["stages"]]


def test_dashboard_answer_without_results_skips_generation() -> None:
    written = []

    result = run_dashboard_answer(
        make_settings(),
        "nothing matches",
        "default",
        5,
        llm=FakeLLM(),
        search=EmptySearch(),
        reranker=FakeReranker(),
        trace_writer=collect(written),
    )

    assert result.results == []
    assert result.answer is None
    assert result.answer_error is None
    assert "generation" not in [stage["name"] for stage in written[0][0]["stages"]]


def test_dashboard_query_surfaces_rerank_fallback_error() -> None:
    written = []

    result = run_dashboard_query(
        make_settings(),
        "海盐桂花拿铁售价多少？",
        "default",
        5,
        search=FakeSearch(),
        reranker=FallbackReranker(),
        trace_writer=collect(written),
    )

    assert result.rerank_error == "llm reranker fallback: boom"
    assert len(result.results) == 1


def test_dashboard_query_without_fallback_has_no_rerank_error() -> None:
    result = run_dashboard_query(
        make_settings(),
        "海盐桂花拿铁售价多少？",
        "default",
        5,
        search=FakeSearch(),
        reranker=FakeReranker(),
        trace_writer=collect([]),
    )

    assert result.rerank_error is None


def make_embedding_settings(dimensions, provider="ollama"):
    return SimpleNamespace(embedding=SimpleNamespace(provider=provider, dimensions=dimensions))


def test_dimension_warning_flags_mismatch() -> None:
    warning = _dimension_warning(make_embedding_settings(768), 128)

    assert warning is not None
    assert "128" in warning and "768" in warning and "ollama" in warning


def test_dimension_warning_is_silent_when_matching_or_unknown() -> None:
    assert _dimension_warning(make_embedding_settings(768), 768) is None
    assert _dimension_warning(make_embedding_settings(768), None) is None
    assert _dimension_warning(make_embedding_settings(None), 128) is None
