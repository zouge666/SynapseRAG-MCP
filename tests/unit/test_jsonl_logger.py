import json
import logging
from pathlib import Path

import pytest

from observability.logger import JSONFormatter, LoggerError, get_trace_logger, write_trace


def trace_dict() -> dict[str, object]:
    return {
        "trace_id": "trace-1",
        "trace_type": "query",
        "started_at": "2026-05-24T00:00:00+00:00",
        "finished_at": "2026-05-24T00:00:01+00:00",
        "total_elapsed_ms": 1.0,
        "stages": [{"name": "query_processing", "elapsed_ms": 1.0}],
    }


def test_json_formatter_merges_trace_dict_into_json_record() -> None:
    record = logging.LogRecord(
        name="synapserag.trace",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="trace",
        args=(),
        exc_info=None,
    )
    record.trace = trace_dict()

    data = json.loads(JSONFormatter().format(record))

    assert data["message"] == "trace"
    assert data["level"] == "INFO"
    assert data["trace_id"] == "trace-1"
    assert data["trace_type"] == "query"
    assert data["stages"][0]["name"] == "query_processing"


def test_write_trace_appends_json_line(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "traces.jsonl"

    write_trace(trace_dict(), path=path, logger=get_trace_logger(path, name="test.trace.write"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["trace_id"] == "trace-1"
    assert data["trace_type"] == "query"
    assert data["total_elapsed_ms"] == 1.0


def test_write_trace_appends_multiple_lines(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "traces.jsonl"
    logger = get_trace_logger(path, name="test.trace.append")

    write_trace(trace_dict(), path=path, logger=logger)
    write_trace({**trace_dict(), "trace_id": "trace-2", "trace_type": "ingestion"}, path=path, logger=logger)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["trace_id"] for record in records] == ["trace-1", "trace-2"]
    assert records[1]["trace_type"] == "ingestion"


def test_get_trace_logger_reuses_handler_for_same_path(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"

    first = get_trace_logger(path, name="test.trace.reuse")
    second = get_trace_logger(path, name="test.trace.reuse")

    assert first is second
    assert len(second.handlers) == 1


def test_write_trace_rejects_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(LoggerError, match="trace_dict"):
        write_trace(["bad"], path=tmp_path / "traces.jsonl", logger=get_trace_logger(tmp_path / "traces.jsonl", name="test.trace.bad"))
