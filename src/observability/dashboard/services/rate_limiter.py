from __future__ import annotations

import threading
import time
from typing import Callable


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._windows: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        now = self.clock()
        with self._lock:
            window_start, count = self._windows.get(key, (now, 0))
            if now - window_start >= window_seconds:
                window_start, count = now, 0
            if count >= limit:
                self._windows[key] = (window_start, count)
                return False
            self._windows[key] = (window_start, count + 1)
            return True

    def remaining(self, key: str, limit: int, window_seconds: int) -> int:
        now = self.clock()
        with self._lock:
            window_start, count = self._windows.get(key, (now, 0))
            if now - window_start >= window_seconds:
                return limit
            return max(0, limit - count)


UPLOAD_LIMIT_PER_SESSION = 3
OWNER_LLM_HOURLY_LIMIT = 30
OWNER_LLM_DAILY_LIMIT = 200


def upload_allowed(limiter: RateLimiter, session_id: str) -> bool:
    return limiter.allow(f"upload:{session_id}", UPLOAD_LIMIT_PER_SESSION, 3600)


def owner_llm_allowed(limiter: RateLimiter) -> bool:
    if not limiter.allow("owner-llm:hour", OWNER_LLM_HOURLY_LIMIT, 3600):
        return False
    if not limiter.allow("owner-llm:day", OWNER_LLM_DAILY_LIMIT, 86400):
        return False
    return True
