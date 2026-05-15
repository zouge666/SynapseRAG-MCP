from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4


@dataclass
class TraceContext:
    trace_type: str = "ingestion"
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    stages: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    status: str = "running"
    _started_time: float = field(default_factory=perf_counter, repr=False)

    def record_stage(self, name: str, details: dict[str, Any] | None = None, duration_ms: float | None = None) -> None:
        self.stages.append(
            {
                "name": name,
                "details": details or {},
                "duration_ms": duration_ms,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def finish(self, status: str = "success") -> dict[str, Any]:
        self.status = status
        self.finished_at = datetime.now(timezone.utc).isoformat()
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "metadata": self.metadata,
            "stages": self.stages,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "duration_ms": round((perf_counter() - self._started_time) * 1000, 3),
        }
