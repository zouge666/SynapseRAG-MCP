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

from core.settings import EmbeddingSettings, LLMSettings, Settings, SettingsError, load_settings, validate_settings


SESSIONS_ROOT = Path(tempfile.gettempdir()) / "synapserag_sessions"
SESSION_TTL_SECONDS = 3600
SESSION_MARKER = "session.json"
SESSION_COLLECTION = "session"

Clock = Callable[[], float]

LOCAL_EMBEDDING = EmbeddingSettings(provider="local", model="local-hash", dimensions=128)

_PUBLIC_PLACEHOLDERS = {
    ("llm", "provider"): "none",
    ("llm", "model"): "none",
    ("embedding", "provider"): "local",
    ("embedding", "model"): "local-hash",
    ("rerank", "backend"): "none",
}


def load_public_base_settings(path: str = "config/settings.yaml") -> Settings:
    try:
        return load_settings(path)
    except SettingsError:
        pass
    import yaml

    from core import settings as core_settings

    config_path = Path(path)
    if not config_path.exists():
        raise SettingsError(f"settings file not found: {path}")
    core_settings._load_dotenv()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SettingsError("settings root must be a mapping")
    raw = core_settings._expand_env_vars(raw)
    for (section, key), placeholder in _PUBLIC_PLACEHOLDERS.items():
        node = raw.get(section)
        if isinstance(node, dict) and not node.get(key):
            node[key] = placeholder
    settings = core_settings._parse_settings(raw)
    validate_settings(settings)
    return settings


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


GUEST_EMBEDDING_LOCAL_LABEL = "local (free hash)"
GUEST_EMBEDDING_OPENAI_LABEL = "openai-compatible"


def apply_guest_embedding(session: "SessionContext", provider_label: str, model: str, base_url: str, api_key: str, dimensions: int) -> str | None:
    if provider_label == GUEST_EMBEDDING_LOCAL_LABEL:
        new_embedding = LOCAL_EMBEDDING
    else:
        if not model.strip():
            return "Enter the embedding model name."
        if not is_safe_remote_url(base_url):
            return "Embedding base URL must be a public https address."
        if not api_key.strip():
            return "Enter your embedding API key, or switch back to local (free hash)."
        if dimensions <= 0:
            return "Dimensions must be a positive integer."
        new_embedding = EmbeddingSettings(provider="openai", model=model.strip(), base_url=base_url.strip(), api_key=api_key.strip(), dimensions=dimensions)
    current = session.settings.embedding
    changed = (new_embedding.provider, new_embedding.model, new_embedding.dimensions) != (current.provider, current.model, current.dimensions)
    session.settings = replace(session.settings, embedding=new_embedding)
    if changed:
        session.reset_workspace()
    return None


GUEST_RERANK_BACKENDS = ["none", "llm"]


def apply_guest_rerank(session: "SessionContext", backend: str, top_m: int, top_k_final: int) -> str | None:
    if backend not in GUEST_RERANK_BACKENDS:
        return f"Unsupported rerank backend: {backend}"
    if backend == "llm" and not session.settings.llm.api_key:
        return "The llm rerank backend uses your session LLM key. Configure it in the Session LLM section first."
    if top_k_final <= 0 or top_m <= 0:
        return "Top K and Top M must be positive integers."
    session.settings = replace(
        session.settings,
        rerank=replace(session.settings.rerank, enabled=backend != "none", backend=backend, top_m=int(top_m)),
        retrieval=replace(session.settings.retrieval, top_k_final=int(top_k_final)),
    )
    return None


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
            settings=build_session_settings(base_settings, paths, llm, embedding, admin=mode == "admin"),
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


def build_session_settings(base: Settings, paths: SessionPaths, llm: LLMSettings | None, embedding: EmbeddingSettings | None = None, admin: bool = False) -> Settings:
    active_llm = llm or LLMSettings(provider="none", model="none")
    if admin:
        session_ingestion = replace(
            base.ingestion,
            bm25_path=str(paths.bm25),
            image_root=str(paths.images),
            image_db_path=str(paths.image_db),
            integrity_db_path=str(paths.integrity_db),
        )
        rerank = base.rerank
        evaluation = base.evaluation
    else:
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
        rerank = replace(base.rerank, enabled=False, backend="none")
        evaluation = replace(base.evaluation, enabled=False)
    return replace(
        base,
        llm=active_llm,
        embedding=embedding or LOCAL_EMBEDDING,
        vector_store=replace(base.vector_store, persist_path=str(paths.chroma), collection=SESSION_COLLECTION),
        ingestion=session_ingestion,
        rerank=rerank,
        evaluation=evaluation,
        observability=replace(base.observability, log_path=str(paths.logs / "app.log"), trace_path=str(paths.trace)),
    )


def settings_to_raw(settings: Settings) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "app": {"name": settings.app.name, "environment": settings.app.environment},
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "api_key": settings.llm.api_key,
            "base_url": settings.llm.base_url,
            "max_image_size": settings.llm.max_image_size,
        },
        "embedding": {
            "provider": settings.embedding.provider,
            "model": settings.embedding.model,
            "dimensions": settings.embedding.dimensions,
            "api_key": settings.embedding.api_key,
            "base_url": settings.embedding.base_url,
        },
        "vector_store": {
            "backend": settings.vector_store.backend,
            "persist_path": settings.vector_store.persist_path,
            "collection": settings.vector_store.collection,
        },
        "splitter": {
            "provider": settings.splitter.provider,
            "chunk_size": settings.splitter.chunk_size,
            "chunk_overlap": settings.splitter.chunk_overlap,
        },
        "ingestion": {
            "chunk_refiner": {"use_llm": settings.ingestion.chunk_refiner.use_llm, "prompt_path": settings.ingestion.chunk_refiner.prompt_path},
            "metadata_enricher": {"use_llm": settings.ingestion.metadata_enricher.use_llm},
            "image_captioner": {"enabled": settings.ingestion.image_captioner.enabled, "prompt_path": settings.ingestion.image_captioner.prompt_path},
            "bm25_path": settings.ingestion.bm25_path,
            "image_root": settings.ingestion.image_root,
            "image_db_path": settings.ingestion.image_db_path,
            "integrity_db_path": settings.ingestion.integrity_db_path,
        },
        "retrieval": {
            "sparse_backend": settings.retrieval.sparse_backend,
            "fusion_algorithm": settings.retrieval.fusion_algorithm,
            "top_k_dense": settings.retrieval.top_k_dense,
            "top_k_sparse": settings.retrieval.top_k_sparse,
            "top_k_final": settings.retrieval.top_k_final,
        },
        "rerank": {
            "enabled": settings.rerank.enabled,
            "backend": settings.rerank.backend,
            "model": settings.rerank.model,
            "top_m": settings.rerank.top_m,
        },
        "evaluation": {"enabled": settings.evaluation.enabled, "backends": list(settings.evaluation.backends)},
        "observability": {"log_path": settings.observability.log_path, "trace_path": settings.observability.trace_path},
    }
    if settings.vision_llm is not None:
        raw["vision_llm"] = {
            "provider": settings.vision_llm.provider,
            "model": settings.vision_llm.model,
            "api_key": settings.vision_llm.api_key,
            "base_url": settings.vision_llm.base_url,
            "max_image_size": settings.vision_llm.max_image_size,
        }
    return raw


def apply_admin_session_settings(session: "SessionContext", raw: dict[str, Any]) -> str | None:
    from core import settings as core_settings

    try:
        parsed = core_settings._parse_settings(raw)
        validate_settings(parsed)
    except SettingsError as error:
        return str(error)
    embedding = session.settings.embedding
    embedding_changed = (parsed.embedding.provider, parsed.embedding.model, parsed.embedding.dimensions) != (embedding.provider, embedding.model, embedding.dimensions)
    session.settings = _isolate_session_paths(parsed, session.paths)
    if embedding_changed:
        session.reset_workspace()
    return None


def _isolate_session_paths(settings: Settings, paths: SessionPaths) -> Settings:
    return replace(
        settings,
        vector_store=replace(settings.vector_store, persist_path=str(paths.chroma), collection=SESSION_COLLECTION),
        ingestion=replace(
            settings.ingestion,
            bm25_path=str(paths.bm25),
            image_root=str(paths.images),
            image_db_path=str(paths.image_db),
            integrity_db_path=str(paths.integrity_db),
        ),
        observability=replace(settings.observability, log_path=str(paths.logs / "app.log"), trace_path=str(paths.trace)),
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
