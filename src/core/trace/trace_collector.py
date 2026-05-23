from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.trace.trace_context import TraceContext


class TraceCollectorError(ValueError):
    pass


class TraceCollector:
    def __init__(self, writer: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.writer = writer
        self.traces: list[dict[str, Any]] = []

    def collect(self, trace: TraceContext) -> dict[str, Any]:
        if not isinstance(trace, TraceContext):
            raise TraceCollectorError("trace must be a TraceContext")
        data = trace.to_dict()
        self.traces.append(data)
        if self.writer is not None:
            self.writer(data)
        return data
