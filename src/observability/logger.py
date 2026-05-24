from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LoggerError(ValueError):
    pass


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace = getattr(record, "trace", None)
        if isinstance(trace, dict):
            payload.update(trace)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_trace_logger(path: str | Path = "logs/traces.jsonl", name: str = "synapserag.trace") -> logging.Logger:
    log_path = Path(path)
    logger = logging.getLogger(name)
    active_path = getattr(logger, "_synapserag_trace_path", None)
    if logger.handlers and active_path == str(log_path):
        return logger
    for handler in list(logger.handlers):
        if getattr(handler, "_synapserag_trace_handler", False):
            logger.removeHandler(handler)
            handler.close()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(JSONFormatter())
    handler._synapserag_trace_handler = True
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger._synapserag_trace_path = str(log_path)
    return logger


def write_trace(trace_dict: dict[str, Any], path: str | Path = "logs/traces.jsonl", logger: logging.Logger | None = None) -> dict[str, Any]:
    if not isinstance(trace_dict, dict):
        raise LoggerError("trace_dict must be a mapping")
    active_logger = logger or get_trace_logger(path)
    active_logger.info("trace", extra={"trace": dict(trace_dict)})
    for handler in active_logger.handlers:
        handler.flush()
    return trace_dict
