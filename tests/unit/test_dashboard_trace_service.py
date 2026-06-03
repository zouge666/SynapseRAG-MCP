import json
from pathlib import Path

from observability.dashboard.pages.ingestion_traces import _stage_rows, _trace_label, _trace_rows
from observability.dashboard.services.trace_service import TraceService, TraceSummary


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\nnot-json\n", encoding="utf-8")


def ingestion_trace(trace_id: str, source_path: str, started_at: str, elapsed: float) -> dict:
    return {
        "trace_id": trace_id,
        "trace_type": "ingestion",
        "metadata": {"source_path": source_path, "collection": "docs"},
        "started_at": started_at,
        "finished_at": started_at,
        "status": "success",
        "total_elapsed_ms": elapsed,
        "stages": [
            {"name": "load", "elapsed_ms": 1.0, "details": {"method": "pdf_loader", "count": 1}},
            {"name": "split", "elapsed_ms": 2.0, "details": {"method": "recursive", "count": 2}},
            {"name": "transform", "elapsed_ms": 3.0, "details": {"method": "metadata_enricher"}},
            {"name": "embed", "elapsed_ms": 4.0, "details": {"method": "batch_processor"}},
            {"name": "upsert", "elapsed_ms": 5.0, "details": {"method": "chroma"}},
            {"name": "pipeline", "elapsed_ms": 6.0, "details": {"status": "success"}},
        ],
    }


def test_trace_service_reads_filters_and_sorts_ingestion_traces(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    older = ingestion_trace("trace-1", "docs/a.pdf", "2026-06-01T10:00:00+00:00", 10.0)
    newer = ingestion_trace("trace-2", "docs/b.pdf", "2026-06-02T10:00:00+00:00", 20.0)
    query = {**ingestion_trace("trace-q", "docs/q.pdf", "2026-06-03T10:00:00+00:00", 30.0), "trace_type": "query"}
    write_jsonl(path, [older, query, newer])
    service = TraceService(path)

    traces = service.ingestion_traces()
    summaries = service.summaries("ingestion")

    assert [trace["trace_id"] for trace in traces] == ["trace-2", "trace-1"]
    assert [summary.trace_id for summary in summaries] == ["trace-2", "trace-1"]
    assert isinstance(summaries[0], TraceSummary)
    assert service.get_trace("trace-1")["metadata"]["source_path"] == "docs/a.pdf"
    assert service.query_traces()[0]["trace_id"] == "trace-q"


def test_trace_service_builds_stage_and_waterfall_rows(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    trace = ingestion_trace("trace-1", "docs/a.pdf", "2026-06-01T10:00:00+00:00", 10.0)
    write_jsonl(path, [trace])
    service = TraceService(path)

    rows = service.stage_rows(trace)
    waterfall = service.ingestion_waterfall_rows(trace)

    assert rows[0] == {"stage": "load", "elapsed_ms": 1.0, "method": "pdf_loader", "details": {"method": "pdf_loader", "count": 1}}
    assert [row["stage"] for row in waterfall] == ["load", "split", "transform", "embed", "upsert"]
    assert _stage_rows(waterfall)[0] == {"stage": "load", "elapsed_ms": 1.0, "method": "pdf_loader"}


def test_ingestion_trace_page_helpers_format_rows_and_labels() -> None:
    trace = ingestion_trace("trace-1", "docs/a.pdf", "2026-06-01T10:00:00+00:00", 10.0)

    rows = _trace_rows([trace])

    assert rows[0]["source_path"] == "docs/a.pdf"
    assert rows[0]["collection"] == "docs"
    assert rows[0]["elapsed_ms"] == 10.0
    assert _trace_label(trace) == "docs/a.pdf success"
