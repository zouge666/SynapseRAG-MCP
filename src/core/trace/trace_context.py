from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4


class TraceContextError(ValueError):
    pass


@dataclass
class TraceContext:
    trace_type: str = "query"
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    stages: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    status: str = "running"
    total_elapsed_ms: float | None = None
    _started_time: float = field(default_factory=perf_counter, repr=False)
    _finished_time: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.trace_type not in {"query", "ingestion"}:
            raise TraceContextError("trace_type must be query or ingestion")
        if not isinstance(self.metadata, dict):
            raise TraceContextError("metadata must be a mapping")
        if not isinstance(self.stages, list):
            raise TraceContextError("stages must be a list")

    def record_stage(self, name: str, details: dict[str, Any] | None = None, duration_ms: float | None = None) -> None:
        if not isinstance(name, str) or not name:
            raise TraceContextError("stage name must be a non-empty string")
        if details is not None and not isinstance(details, dict):
            raise TraceContextError("stage details must be a mapping")
        elapsed = self._elapsed(duration_ms)
        self.stages.append(
            {
                "name": name,
                "details": details or {},
                "duration_ms": duration_ms,
                "elapsed_ms": elapsed,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def finish(self, status: str = "success") -> dict[str, Any]:
        if not isinstance(status, str) or not status:
            raise TraceContextError("status must be a non-empty string")
        self.status = status
        if self.finished_at is None:
            self.finished_at = datetime.now(timezone.utc).isoformat()
            self._finished_time = perf_counter()
            self.total_elapsed_ms = self.elapsed_ms()
        elif self.total_elapsed_ms is None:
            self.total_elapsed_ms = self.elapsed_ms()
        return self.to_dict()

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        if stage_name is not None:
            for stage in reversed(self.stages):
                if stage.get("name") == stage_name:
                    value = stage.get("elapsed_ms", stage.get("duration_ms"))
                    return float(value or 0.0)
            raise TraceContextError(f"stage not found: {stage_name}")
        end = self._finished_time if self._finished_time is not None else perf_counter()
        return round((end - self._started_time) * 1000, 3)

    def to_dict(self) -> dict[str, Any]:
        total_elapsed = self.total_elapsed_ms if self.total_elapsed_ms is not None else self.elapsed_ms()
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "metadata": self.metadata,
            "stages": self.stages,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "total_elapsed_ms": total_elapsed,
            "duration_ms": total_elapsed,
        }

    def _elapsed(self, duration_ms: float | None) -> float:
        if duration_ms is None:
            return self.elapsed_ms()
        if not isinstance(duration_ms, int | float) or duration_ms < 0:
            raise TraceContextError("duration_ms must be a non-negative number")
        return round(float(duration_ms), 3)
