from pathlib import Path

import pytest

from core.settings import load_settings
from observability.dashboard.services.session_context import (
    LOCAL_EMBEDDING,
    SESSION_COLLECTION,
    SessionContext,
    SessionError,
    build_session_settings,
    sanitize_error,
)


class FakeClock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def base_settings():
    return load_settings("config/settings.yaml")


def make_session(base_settings, tmp_path: Path, clock: FakeClock, mode: str = "guest", session_id: str | None = None) -> SessionContext:
    return SessionContext.create(base_settings, mode, session_id=session_id, root=tmp_path / "sessions", clock=clock)


def test_two_sessions_use_completely_different_paths(base_settings, tmp_path) -> None:
    clock = FakeClock()
    first = make_session(base_settings, tmp_path, clock)
    second = make_session(base_settings, tmp_path, clock)

    assert first.session_id != second.session_id
    first_paths = {str(first.paths.chroma), str(first.paths.bm25), str(first.paths.images), str(first.paths.integrity_db), str(first.paths.trace)}
    second_paths = {str(second.paths.chroma), str(second.paths.bm25), str(second.paths.images), str(second.paths.integrity_db), str(second.paths.trace)}
    assert first_paths.isdisjoint(second_paths)


def test_session_settings_point_into_session_root_and_base_is_untouched(base_settings, tmp_path) -> None:
    session = make_session(base_settings, tmp_path, FakeClock())

    assert str(session.paths.root) in session.settings.vector_store.persist_path
    assert str(session.paths.root) in session.settings.ingestion.bm25_path
    assert str(session.paths.root) in session.settings.ingestion.integrity_db_path
    assert str(session.paths.root) in session.settings.observability.trace_path
    assert session.settings.vector_store.collection == SESSION_COLLECTION
    assert session.settings.embedding == LOCAL_EMBEDDING
    assert session.settings.rerank.enabled is False
    assert session.settings.ingestion.chunk_refiner.use_llm is False
    assert session.settings.ingestion.metadata_enricher.use_llm is False
    assert session.settings.ingestion.image_captioner.enabled is False

    assert base_settings.vector_store.persist_path == "data/db/chroma"
    assert base_settings.embedding.provider == "ollama"
    assert base_settings.rerank.enabled is True


def test_ttl_absolute_3599_alive_3601_expired_and_cleaned(base_settings, tmp_path) -> None:
    clock = FakeClock()
    session = make_session(base_settings, tmp_path, clock)
    assert session.paths.root.exists()

    clock.now += 3599
    session.require_active()
    assert session.paths.root.exists()

    clock.now += 2
    with pytest.raises(Exception, match="expired"):
        session.require_active()
    assert not session.paths.root.exists()


def test_reset_workspace_clears_derived_data_but_keeps_session(base_settings, tmp_path) -> None:
    clock = FakeClock()
    session = make_session(base_settings, tmp_path, clock)
    (session.paths.chroma).mkdir(parents=True, exist_ok=True)
    (session.paths.chroma / "session.json").write_text("{}", encoding="utf-8")
    (session.paths.integrity_db).write_text("db", encoding="utf-8")

    session.reset_workspace()

    assert session.paths.root.exists()
    assert not (session.paths.chroma / "session.json").exists()
    assert not session.paths.integrity_db.exists()
    assert not session.is_expired()


def test_destroy_removes_entire_workspace(base_settings, tmp_path) -> None:
    session = make_session(base_settings, tmp_path, FakeClock())
    (session.paths.images).mkdir(parents=True, exist_ok=True)
    (session.paths.images / "x.png").write_bytes(b"x")

    session.destroy()

    assert not session.paths.root.exists()


def test_rejects_path_traversal_session_id(base_settings, tmp_path) -> None:
    with pytest.raises(SessionError, match="invalid session id"):
        make_session(base_settings, tmp_path, FakeClock(), session_id="../escape")


def test_guarded_delete_refuses_escape(base_settings, tmp_path) -> None:
    session = make_session(base_settings, tmp_path, FakeClock())
    session.paths = type(session.paths)(
        root=tmp_path, uploads=tmp_path, chroma=tmp_path, bm25=tmp_path, images=tmp_path,
        image_db=tmp_path / "a.db", integrity_db=tmp_path / "b.db", logs=tmp_path, trace=tmp_path / "t.jsonl",
    )
    with pytest.raises(SessionError, match="outside the session root"):
        session.destroy()


def test_sanitize_error_hides_secrets() -> None:
    assert sanitize_error("request failed with key sk-abc123", ["sk-abc123"]) == "request failed with key ***"
    assert sanitize_error("plain error", ["sk-abc123"]) == "plain error"


def test_build_session_settings_without_llm_disables_answer_generation(base_settings, tmp_path) -> None:
    from observability.dashboard.services.session_context import _session_paths

    settings = build_session_settings(base_settings, _session_paths(tmp_path, "s1"), None)

    assert settings.llm.api_key == ""


def test_is_safe_remote_url_accepts_public_https() -> None:
    from observability.dashboard.services.session_context import is_safe_remote_url

    assert is_safe_remote_url("https://api.deepseek.com/v1") is True
    assert is_safe_remote_url("https://api.openai.com/v1") is True
    assert is_safe_remote_url("https://8.8.8.8/v1") is True


def test_is_safe_remote_url_rejects_non_https_and_local() -> None:
    from observability.dashboard.services.session_context import is_safe_remote_url

    assert is_safe_remote_url("http://api.deepseek.com/v1") is False
    assert is_safe_remote_url("https://localhost:11434") is False
    assert is_safe_remote_url("https://127.0.0.1:8080") is False
    assert is_safe_remote_url("https://192.168.1.10") is False
    assert is_safe_remote_url("https://10.0.0.5") is False
    assert is_safe_remote_url("https://169.254.169.254/latest/meta-data") is False
    assert is_safe_remote_url("https://printer.local") is False
    assert is_safe_remote_url("not-a-url") is False


def test_session_embedding_override(base_settings, tmp_path) -> None:
    from core.settings import EmbeddingSettings

    custom = EmbeddingSettings(provider="openai", model="BAAI/bge-m3", dimensions=1024, api_key="sk-x")
    session = SessionContext.create(base_settings, "admin", embedding=custom, root=tmp_path)
    assert session.settings.embedding.model == "BAAI/bge-m3"
    assert session.settings.embedding.dimensions == 1024


def test_session_embedding_defaults_to_local_hash(base_settings, tmp_path) -> None:
    session = SessionContext.create(base_settings, "guest", root=tmp_path)
    assert session.settings.embedding.provider == "local"
