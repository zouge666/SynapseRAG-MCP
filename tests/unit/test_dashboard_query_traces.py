import json
from pathlib import Path

from observability.dashboard.pages.query_traces import _stage_rows, _trace_label, _trace_rows
from observability.dashboard.services.trace_service import TraceService


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def query_trace(trace_id: str, query: str, started_at: str) -> dict:
    return {
        "trace_id": trace_id,
        "trace_type": "query",
        "metadata": {"query": query},
        "started_at": started_at,
        "finished_at": started_at,
        "status": "success",
        "total_elapsed_ms": 12.0,
        "stages": [
            {"name": "query_processing", "elapsed_ms": 1.0, "details": {"method": "query_processor", "query": query, "keyword_count": 2}},
            {"name": "dense_retrieval", "elapsed_ms": 2.0, "details": {"method": "dense", "count": 3, "top_k": 5}},
            {"name": "sparse_retrieval", "elapsed_ms": 3.0, "details": {"method": "sparse", "count": 2, "top_k": 5}},
            {"name": "fusion", "elapsed_ms": 4.0, "details": {"method": "rrf", "count": 4}},
            {"name": "rerank", "elapsed_ms": 5.0, "details": {"method": "reranker", "enabled": True, "count": 4}},
        ],
    }


def test_trace_service_searches_query_traces_by_query_text(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    write_jsonl(
        path,
        [
            query_trace("trace-1", "hybrid rag", "2026-06-01T10:00:00+00:00"),
            query_trace("trace-2", "metadata filters", "2026-06-02T10:00:00+00:00"),
            {**query_trace("trace-3", "ingest docs", "2026-06-03T10:00:00+00:00"), "trace_type": "ingestion"},
        ],
    )
    service = TraceService(path)

    traces = service.search_query_traces("metadata")

    assert [trace["trace_id"] for trace in traces] == ["trace-2"]
    assert [trace["trace_id"] for trace in service.search_query_traces()] == ["trace-2", "trace-1"]
    assert service.summary_for_trace(traces[0]).metadata["query"] == "metadata filters"


def test_trace_service_builds_query_waterfall_comparison_and_rerank_rows(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    trace = query_trace("trace-1", "hybrid rag", "2026-06-01T10:00:00+00:00")
    write_jsonl(path, [trace])
    service = TraceService(path)

    waterfall = service.query_waterfall_rows(trace)
    comparison = service.retrieval_comparison_rows(trace)
    rerank = service.rerank_rows(trace)

    assert [row["stage"] for row in waterfall] == ["query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank"]
    assert comparison == [
        {"route": "dense", "count": 3, "top_k": 5, "elapsed_ms": 2.0, "method": "dense", "error": ""},
        {"route": "sparse", "count": 2, "top_k": 5, "elapsed_ms": 3.0, "method": "sparse", "error": ""},
    ]
    assert rerank[0]["details"]["enabled"] is True
    assert _stage_rows(waterfall)[0] == {"stage": "query_processing", "elapsed_ms": 1.0, "method": "query_processor"}


def test_query_trace_page_helpers_format_rows_and_labels() -> None:
    trace = query_trace("trace-1", "hybrid rag", "2026-06-01T10:00:00+00:00")

    rows = _trace_rows([trace])

    assert rows[0]["query"] == "hybrid rag"
    assert rows[0]["elapsed_ms"] == 12.0
    assert _trace_label(trace) == "hybrid rag success"
