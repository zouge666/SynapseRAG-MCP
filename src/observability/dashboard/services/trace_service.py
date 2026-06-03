from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceSummary:
    trace_id: str
    trace_type: str
    status: str
    started_at: str
    finished_at: str | None
    total_elapsed_ms: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.total_elapsed_ms,
            "metadata": dict(self.metadata),
        }


class TraceService:
    def __init__(self, trace_path: str | Path = "logs/traces.jsonl") -> None:
        self.trace_path = Path(trace_path)

    def list_traces(self, trace_type: str | None = None) -> list[dict[str, Any]]:
        traces = [trace for trace in self._read_jsonl() if self._matches_trace_type(trace, trace_type)]
        return sorted(traces, key=self._sort_key, reverse=True)

    def ingestion_traces(self) -> list[dict[str, Any]]:
        return self.list_traces("ingestion")

    def query_traces(self) -> list[dict[str, Any]]:
        return self.list_traces("query")

    def search_query_traces(self, keyword: str = "") -> list[dict[str, Any]]:
        value = keyword.strip().lower()
        traces = self.query_traces()
        if not value:
            return traces
        return [trace for trace in traces if value in self._query_text(trace).lower()]

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        for trace in self.list_traces():
            if trace.get("trace_id") == trace_id:
                return trace
        return None

    def summaries(self, trace_type: str | None = None) -> list[TraceSummary]:
        return [self._summary(trace) for trace in self.list_traces(trace_type)]

    def summary_for_trace(self, trace: dict[str, Any]) -> TraceSummary:
        return self._summary(trace)

    def stage_rows(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for stage in trace.get("stages", []):
            if not isinstance(stage, dict):
                continue
            details = stage.get("details", {})
            if not isinstance(details, dict):
                details = {}
            rows.append(
                {
                    "stage": stage.get("name", ""),
                    "elapsed_ms": float(stage.get("elapsed_ms") or stage.get("duration_ms") or 0.0),
                    "method": details.get("method", ""),
                    "details": details,
                }
            )
        return rows

    def ingestion_waterfall_rows(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        wanted = {"load", "split", "transform", "embed", "upsert"}
        return [row for row in self.stage_rows(trace) if row["stage"] in wanted]

    def query_waterfall_rows(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        wanted = {"query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank"}
        return [row for row in self.stage_rows(trace) if row["stage"] in wanted]

    def retrieval_comparison_rows(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for row in self.stage_rows(trace):
            if row["stage"] not in {"dense_retrieval", "sparse_retrieval"}:
                continue
            details = row["details"]
            rows.append(
                {
                    "route": row["stage"].replace("_retrieval", ""),
                    "count": int(details.get("count") or 0),
                    "top_k": int(details.get("top_k") or 0),
                    "elapsed_ms": row["elapsed_ms"],
                    "method": row["method"],
                    "error": details.get("error", ""),
                }
            )
        return rows

    def rerank_rows(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        return [row for row in self.stage_rows(trace) if row["stage"] == "rerank"]

    def _query_text(self, trace: dict[str, Any]) -> str:
        metadata = trace.get("metadata", {})
        if isinstance(metadata, dict):
            value = metadata.get("query") or metadata.get("question") or metadata.get("text")
            if isinstance(value, str):
                return value
        for row in self.stage_rows(trace):
            details = row["details"]
            value = details.get("query") or details.get("text")
            if isinstance(value, str):
                return value
        return str(trace.get("trace_id", ""))

    def _read_jsonl(self) -> list[dict[str, Any]]:
        if not self.trace_path.exists():
            return []
        traces = []
        with self.trace_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    traces.append(item)
        return traces

    def _matches_trace_type(self, trace: dict[str, Any], trace_type: str | None) -> bool:
        return trace_type is None or trace.get("trace_type") == trace_type

    def _sort_key(self, trace: dict[str, Any]) -> str:
        return str(trace.get("finished_at") or trace.get("started_at") or trace.get("timestamp") or "")

    def _summary(self, trace: dict[str, Any]) -> TraceSummary:
        return TraceSummary(
            trace_id=str(trace.get("trace_id", "")),
            trace_type=str(trace.get("trace_type", "")),
            status=str(trace.get("status", "")),
            started_at=str(trace.get("started_at", "")),
            finished_at=trace.get("finished_at") if isinstance(trace.get("finished_at"), str) else None,
            total_elapsed_ms=float(trace.get("total_elapsed_ms") or trace.get("duration_ms") or 0.0),
            metadata=dict(trace.get("metadata", {}) if isinstance(trace.get("metadata"), dict) else {}),
        )
