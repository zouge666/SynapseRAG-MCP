import importlib.util
import sys
from pathlib import Path

from core import RetrievalResult


QUERY_PATH = Path(__file__).resolve().parents[2] / "scripts" / "query.py"
SPEC = importlib.util.spec_from_file_location("query_script", QUERY_PATH)
query_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = query_script
SPEC.loader.exec_module(query_script)


class FakeSearch:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters or {}, "trace": trace})
        if trace is not None:
            trace.record_stage("hybrid_search", {"count": len(self.results)})
        return list(self.results[:top_k])


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        trace: object | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "candidates": candidates, "trace": trace})
        if trace is not None:
            trace.record_stage("reranker", {"count": len(candidates), "fallback": False})
        return sorted(candidates, key=lambda item: (-item.score, item.chunk_id))


def result(chunk_id: str, score: float, page: int = 1) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text for {chunk_id}",
        metadata={"source_path": f"docs/{chunk_id}.pdf", "page": page},
    )


def test_query_script_runs_search_and_reranker(capsys) -> None:
    search = FakeSearch([result("a", 0.2, 1), result("b", 0.9, 2)])
    reranker = FakeReranker()

    code = query_script.main(
        ["--query", "find beta", "--top-k", "2", "--collection", "docs"],
        search_factory=lambda settings: search,
        reranker_factory=lambda settings: reranker,
    )

    output = capsys.readouterr()
    assert code == 0
    assert "1. score=0.9000 source=docs/b.pdf page=2" in output.out
    assert "2. score=0.2000 source=docs/a.pdf page=1" in output.out
    assert search.calls[0]["query"] == "find beta"
    assert search.calls[0]["filters"] == {"collection": "docs"}
    assert reranker.calls[0]["query"] == "find beta"


def test_query_script_no_rerank_keeps_candidate_order() -> None:
    search = FakeSearch([result("a", 0.2), result("b", 0.9)])

    script_result = query_script.run_query(
        query="find beta",
        top_k=2,
        collection=None,
        no_rerank=True,
        settings_path="config/settings.yaml",
        search_factory=lambda settings: search,
        reranker_factory=lambda settings: (_ for _ in ()).throw(AssertionError("reranker should not run")),
    )

    assert [item.chunk_id for item in script_result.results] == ["a", "b"]
    assert script_result.rerank_applied is False
    assert script_result.trace.stages[-1]["name"] == "reranker"
    assert script_result.trace.stages[-1]["details"] == {"skipped": True, "count": 2}


def test_query_script_verbose_prints_candidates_final_results_and_trace(capsys) -> None:
    search = FakeSearch([result("a", 0.2), result("b", 0.9)])
    reranker = FakeReranker()

    code = query_script.main(
        ["--query", "find beta", "--top-k", "2", "--verbose"],
        search_factory=lambda settings: search,
        reranker_factory=lambda settings: reranker,
    )

    output = capsys.readouterr()
    assert code == 0
    assert "Candidates before rerank:" in output.out
    assert "Final results:" in output.out
    assert "Trace:" in output.out
    assert "- hybrid_search:" in output.out
    assert "- reranker:" in output.out


def test_query_script_reports_no_data_before_search(monkeypatch, capsys) -> None:
    monkeypatch.setattr(query_script, "has_ingested_data", lambda settings, collection: False)

    code = query_script.main(["--query", "anything"])

    output = capsys.readouterr()
    assert code == 0
    assert "未找到相关文档，请先运行 ingest.py 摄取数据" in output.out


def test_query_script_rejects_invalid_top_k(capsys) -> None:
    code = query_script.main(["--query", "anything", "--top-k", "0"], search_factory=lambda settings: FakeSearch([]))

    output = capsys.readouterr()
    assert code == 1
    assert "top_k must be a positive integer" in output.err
