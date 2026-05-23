import json
from time import sleep

import pytest

from core.trace import TraceCollector, TraceCollectorError, TraceContext, TraceContextError


def test_trace_context_defaults_to_query_trace() -> None:
    trace = TraceContext()

    assert trace.trace_type == "query"
    assert trace.status == "running"
    assert trace.finished_at is None


def test_trace_context_records_stage_and_serializes_finished_state() -> None:
    trace = TraceContext(trace_type="ingestion", metadata={"source_path": "docs/sample.pdf"})
    trace.record_stage("load", {"count": 1}, duration_ms=12.3456)

    data = trace.finish("success")

    assert data["trace_id"] == trace.trace_id
    assert data["trace_type"] == "ingestion"
    assert data["metadata"] == {"source_path": "docs/sample.pdf"}
    assert data["status"] == "success"
    assert data["started_at"]
    assert data["finished_at"]
    assert data["total_elapsed_ms"] >= 0
    assert data["duration_ms"] == data["total_elapsed_ms"]
    assert data["stages"][0]["name"] == "load"
    assert data["stages"][0]["details"] == {"count": 1}
    assert data["stages"][0]["duration_ms"] == 12.3456
    assert data["stages"][0]["elapsed_ms"] == 12.346
    json.dumps(data)


def test_trace_context_elapsed_ms_returns_total_and_stage_values() -> None:
    trace = TraceContext(trace_type="query")
    sleep(0.001)
    trace.record_stage("dense_retrieval", {"count": 2}, duration_ms=5.0)

    assert trace.elapsed_ms() >= 0
    assert trace.elapsed_ms("dense_retrieval") == 5.0


def test_trace_context_finish_is_idempotent() -> None:
    trace = TraceContext(trace_type="query")

    first = trace.finish("success")
    second = trace.finish("success")

    assert second["finished_at"] == first["finished_at"]
    assert second["total_elapsed_ms"] == first["total_elapsed_ms"]


def test_trace_context_validates_inputs() -> None:
    with pytest.raises(TraceContextError, match="trace_type"):
        TraceContext(trace_type="other")
    with pytest.raises(TraceContextError, match="stage name"):
        TraceContext().record_stage("", {})
    with pytest.raises(TraceContextError, match="duration_ms"):
        TraceContext().record_stage("bad", {}, duration_ms=-1)
    with pytest.raises(TraceContextError, match="stage not found"):
        TraceContext().elapsed_ms("missing")


def test_trace_collector_collects_and_writes_trace_dict() -> None:
    written = []
    trace = TraceContext(trace_type="query", metadata={"query": "alpha"})
    collector = TraceCollector(writer=written.append)

    data = collector.collect(trace)

    assert collector.traces == [data]
    assert written == [data]
    assert data["trace_type"] == "query"


def test_trace_collector_rejects_invalid_trace() -> None:
    with pytest.raises(TraceCollectorError, match="TraceContext"):
        TraceCollector().collect({"trace_id": "bad"})
