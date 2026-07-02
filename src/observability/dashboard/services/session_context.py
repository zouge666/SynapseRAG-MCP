from __future__ import annotations

import ipaddress
import json
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from core.settings import EmbeddingSettings, LLMSettings, Settings


SESSIONS_ROOT = Path(tempfile.gettempdir()) / "synapserag_sessions"
SESSION_TTL_SECONDS = 3600
SESSION_MARKER = "session.json"
SESSION_COLLECTION = "session"

Clock = Callable[[], float]

LOCAL_EMBEDDING = EmbeddingSettings(provider="local", model="local-hash", dimensions=128)


def is_safe_remote_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname
    if host == "localhost" or host.endswith((".local", ".internal")):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return True


class SessionError(RuntimeError):
    pass


class SessionExpiredError(SessionError):
    pass


@dataclass(frozen=True)
class SessionPaths:
    root: Path
    uploads: Path
    chroma: Path
    bm25: Path
    images: Path
    image_db: Path
    integrity_db: Path
    logs: Path
    trace: Path


@dataclass
class SessionContext:
    session_id: str
    mode: str
    paths: SessionPaths
    settings: Settings
    created_at: float
    expires_at: float
    display_name: str = ""
    clock: Clock = time.time
    sessions_root: Path | None = None

    @classmethod
    def create(
        cls,
        base_settings: Settings,
        mode: str,
        llm: LLMSettings | None = None,
        embedding: EmbeddingSettings | None = None,
        session_id: str | None = None,
        root: Path | None = None,
        clock: Clock = time.time,
    ) -> "SessionContext":
        if mode not in {"guest", "admin"}:
            raise SessionError(f"unsupported session mode: {mode}")
        active_id = session_id or secrets.token_urlsafe(12)
        if not active_id or Path(active_id).name != active_id:
            raise SessionError("invalid session id")
        paths = _session_paths(root or SESSIONS_ROOT, active_id)
        paths.uploads.mkdir(parents=True, exist_ok=True)
        paths.logs.mkdir(parents=True, exist_ok=True)
        now = clock()
        sessions_root = root or SESSIONS_ROOT
        context = cls(
            session_id=active_id,
            mode=mode,
            paths=paths,
            settings=build_session_settings(base_settings, paths, llm, embedding),
            created_at=now,
            expires_at=now + SESSION_TTL_SECONDS,
            clock=clock,
            sessions_root=sessions_root,
        )
        context._write_marker()
        return context

    def is_expired(self) -> bool:
        return self.clock() >= self.expires_at

    def remaining_seconds(self) -> int:
        return max(0, int(self.expires_at - self.clock()))

    def require_active(self) -> None:
        if self.is_expired():
            self.destroy()
            raise SessionExpiredError("This session has expired. Start a new session to continue.")

    def reset_workspace(self) -> None:
        for name in ("uploads", "chroma", "bm25", "images"):
            target = self.paths.root / name
            if target.exists():
                shutil.rmtree(self._guarded(target), ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)
        for db_file in (self.paths.image_db, self.paths.integrity_db, self.paths.trace, self.paths.logs / "app.log"):
            if db_file.exists():
                db_file.unlink()
        self.paths.logs.mkdir(parents=True, exist_ok=True)

    def destroy(self) -> None:
        if self.paths.root.exists():
            shutil.rmtree(self._guarded(self.paths.root), ignore_errors=True)

    def _guarded(self, target: Path) -> Path:
        sessions_root = (self.sessions_root or self.paths.root.parent).resolve()
        resolved = target.resolve()
        if resolved == sessions_root or sessions_root not in resolved.parents:
            raise SessionError(f"refusing to touch path outside the session root: {target.name}")
        return target

    def _write_marker(self) -> None:
        marker = {"session_id": self.session_id, "mode": self.mode, "created_at": self.created_at, "expires_at": self.expires_at}
        (self.paths.root / SESSION_MARKER).write_text(json.dumps(marker), encoding="utf-8")


def build_session_settings(base: Settings, paths: SessionPaths, llm: LLMSettings | None, embedding: EmbeddingSettings | None = None) -> Settings:
    active_llm = llm or LLMSettings(provider="none", model="none")
    session_ingestion = replace(
        base.ingestion,
        bm25_path=str(paths.bm25),
        image_root=str(paths.images),
        image_db_path=str(paths.image_db),
        integrity_db_path=str(paths.integrity_db),
        chunk_refiner=replace(base.ingestion.chunk_refiner, use_llm=False),
        metadata_enricher=replace(base.ingestion.metadata_enricher, use_llm=False),
        image_captioner=replace(base.ingestion.image_captioner, enabled=False),
    )
    return replace(
        base,
        llm=active_llm,
        embedding=embedding or LOCAL_EMBEDDING,
        vector_store=replace(base.vector_store, persist_path=str(paths.chroma), collection=SESSION_COLLECTION),
        ingestion=session_ingestion,
        rerank=replace(base.rerank, enabled=False, backend="none"),
        evaluation=replace(base.evaluation, enabled=False),
        observability=replace(base.observability, log_path=str(paths.logs / "app.log"), trace_path=str(paths.trace)),
    )


def read_marker(session_dir: Path) -> dict[str, Any] | None:
    marker_path = session_dir / SESSION_MARKER
    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("expires_at"), (int, float)):
        return None
    return data


def _session_paths(root: Path, session_id: str) -> SessionPaths:
    session_root = root / session_id
    return SessionPaths(
        root=session_root,
        uploads=session_root / "uploads",
        chroma=session_root / "chroma",
        bm25=session_root / "bm25",
        images=session_root / "images",
        image_db=session_root / "image_index.db",
        integrity_db=session_root / "ingestion_history.db",
        logs=session_root / "logs",
        trace=session_root / "logs" / "traces.jsonl",
    )


def sanitize_error(message: str, secrets_to_hide: list[str]) -> str:
    sanitized = str(message)
    for secret in secrets_to_hide:
        if secret and secret in sanitized:
            sanitized = sanitized.replace(secret, "***")
    return sanitized
