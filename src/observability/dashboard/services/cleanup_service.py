from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from observability.dashboard.services.session_context import SESSIONS_ROOT, read_marker


GRACE_PERIOD_SECONDS = 3600


class CleanupService:
    def __init__(self, root: Path = SESSIONS_ROOT, clock: Callable[[], float] = time.time) -> None:
        self.root = root
        self.clock = clock
        self._thread_started = False
        self._lock = threading.Lock()

    def cleanup_expired(self) -> int:
        if not self.root.is_dir():
            return 0
        removed = 0
        now = self.clock()
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            marker = read_marker(entry)
            if marker is not None:
                expired = now >= float(marker["expires_at"])
            else:
                try:
                    expired = now >= entry.stat().st_mtime + GRACE_PERIOD_SECONDS
                except OSError:
                    continue
            if expired and self._safe_remove(entry):
                removed += 1
        return removed

    def ensure_cleanup_thread(self, interval_seconds: int = 60) -> None:
        with self._lock:
            if self._thread_started:
                return
            self._thread_started = True
        thread = threading.Thread(target=self._loop, args=(interval_seconds,), name="synapserag-cleanup", daemon=True)
        thread.start()

    def _loop(self, interval_seconds: int) -> None:
        while True:
            time.sleep(interval_seconds)
            try:
                self.cleanup_expired()
            except Exception:
                continue

    def _safe_remove(self, target: Path) -> bool:
        try:
            resolved_root = self.root.resolve()
            resolved = target.resolve()
            if resolved == resolved_root or resolved_root not in resolved.parents:
                return False
            shutil.rmtree(target, ignore_errors=True)
            return True
        except OSError:
            return False
