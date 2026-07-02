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


def test_load_public_base_settings_tolerates_missing_llm_env(monkeypatch) -> None:
    from core import settings as core_settings
    from observability.dashboard.services.session_context import load_public_base_settings

    for var in ("LLM_MODEL", "LLM_BASE_URL", "API_KEY", "EMBEDDING_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(core_settings, "_load_dotenv", lambda *args, **kwargs: None)

    settings = load_public_base_settings()

    assert settings.app.name == "synapserag-mcp"
    assert settings.llm.model == "none"
    assert settings.retrieval.top_k_final > 0


def test_load_public_base_settings_prefers_real_values_when_present(monkeypatch) -> None:
    from core import settings as core_settings
    from observability.dashboard.services.session_context import load_public_base_settings

    monkeypatch.setattr(core_settings, "_load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("API_KEY", "sk-test")

    settings = load_public_base_settings()

    assert settings.llm.model == "deepseek-chat"


def test_apply_guest_embedding_openai_replaces_and_resets_workspace(base_settings, tmp_path) -> None:
    from observability.dashboard.services.session_context import apply_guest_embedding

    session = SessionContext.create(base_settings, "guest", root=tmp_path)
    session.paths.chroma.mkdir(parents=True, exist_ok=True)
    marker = session.paths.chroma / "old_index.bin"
    marker.write_text("stale")

    error = apply_guest_embedding(session, "openai-compatible", "BAAI/bge-m3", "https://api.siliconflow.cn/v1", "sk-guest-emb", 1024)

    assert error is None
    assert session.settings.embedding.provider == "openai"
    assert session.settings.embedding.model == "BAAI/bge-m3"
    assert session.settings.embedding.dimensions == 1024
    assert not marker.exists()
    assert session.paths.uploads.exists()


def test_apply_guest_embedding_rejects_unsafe_url_and_missing_key(base_settings, tmp_path) -> None:
    from observability.dashboard.services.session_context import apply_guest_embedding

    session = SessionContext.create(base_settings, "guest", root=tmp_path)
    assert apply_guest_embedding(session, "openai-compatible", "m", "http://192.168.0.1", "sk-x", 1024) is not None
    assert apply_guest_embedding(session, "openai-compatible", "m", "https://api.siliconflow.cn/v1", "", 1024) is not None
    assert apply_guest_embedding(session, "openai-compatible", "", "https://api.siliconflow.cn/v1", "sk-x", 1024) is not None
    assert session.settings.embedding.provider == "local"


def test_apply_guest_embedding_local_keeps_workspace_when_unchanged(base_settings, tmp_path) -> None:
    from observability.dashboard.services.session_context import apply_guest_embedding

    session = SessionContext.create(base_settings, "guest", root=tmp_path)
    session.paths.chroma.mkdir(parents=True, exist_ok=True)
    marker = session.paths.chroma / "keep.bin"
    marker.write_text("data")

    assert apply_guest_embedding(session, "local (free hash)", "", "", "", 128) is None
    assert marker.exists()
    assert session.settings.embedding.provider == "local"


def test_apply_guest_rerank_requires_llm_key(base_settings, tmp_path) -> None:
    from observability.dashboard.services.session_context import apply_guest_rerank

    session = SessionContext.create(base_settings, "guest", root=tmp_path)
    assert apply_guest_rerank(session, "llm", 30, 5) is not None
    assert session.settings.rerank.enabled is False


def test_apply_guest_rerank_llm_with_key(base_settings, tmp_path) -> None:
    from core.settings import LLMSettings
    from observability.dashboard.services.session_context import apply_guest_rerank

    session = SessionContext.create(base_settings, "guest", llm=LLMSettings(provider="openai", model="m", api_key="sk-x"), root=tmp_path)
    assert apply_guest_rerank(session, "llm", 25, 8) is None
    assert session.settings.rerank.enabled is True
    assert session.settings.rerank.backend == "llm"
    assert session.settings.rerank.top_m == 25
    assert session.settings.retrieval.top_k_final == 8


def test_apply_guest_rerank_none_disables(base_settings, tmp_path) -> None:
    from observability.dashboard.services.session_context import apply_guest_rerank

    session = SessionContext.create(base_settings, "guest", root=tmp_path)
    assert apply_guest_rerank(session, "none", 30, 3) is None
    assert session.settings.rerank.enabled is False
    assert session.settings.retrieval.top_k_final == 3


def test_admin_session_keeps_base_rerank_and_evaluation(base_settings, tmp_path) -> None:
    admin = SessionContext.create(base_settings, "admin", root=tmp_path)
    guest = SessionContext.create(base_settings, "guest", root=tmp_path)

    assert base_settings.rerank.enabled is True
    assert admin.settings.rerank.enabled is True
    assert admin.settings.rerank.backend == base_settings.rerank.backend
    assert admin.settings.evaluation.enabled == base_settings.evaluation.enabled
    assert guest.settings.rerank.enabled is False
    assert guest.settings.evaluation.enabled is False


def test_settings_to_raw_roundtrips_through_parser(base_settings, tmp_path) -> None:
    from core import settings as core_settings
    from observability.dashboard.services.session_context import settings_to_raw

    session = SessionContext.create(base_settings, "admin", root=tmp_path)
    parsed = core_settings._parse_settings(settings_to_raw(session.settings))
    assert parsed == session.settings


def test_apply_admin_session_settings_updates_rerank_and_keeps_session_paths(base_settings, tmp_path) -> None:
    from observability.dashboard.services.session_context import apply_admin_session_settings, settings_to_raw

    session = SessionContext.create(base_settings, "admin", root=tmp_path)
    raw = settings_to_raw(session.settings)
    raw["rerank"]["enabled"] = False
    raw["rerank"]["backend"] = "none"
    raw["retrieval"]["top_k_final"] = 7
    raw["vector_store"]["persist_path"] = "data/db/chroma"

    assert apply_admin_session_settings(session, raw) is None
    assert session.settings.rerank.enabled is False
    assert session.settings.retrieval.top_k_final == 7
    assert session.settings.vector_store.persist_path == str(session.paths.chroma)
    assert session.settings.vector_store.collection == SESSION_COLLECTION
    assert str(session.paths.root) in session.settings.observability.trace_path


def test_apply_admin_session_settings_rejects_invalid_and_keeps_old(base_settings, tmp_path) -> None:
    from observability.dashboard.services.session_context import apply_admin_session_settings, settings_to_raw

    session = SessionContext.create(base_settings, "admin", root=tmp_path)
    before = session.settings
    raw = settings_to_raw(session.settings)
    raw["llm"]["model"] = ""

    assert apply_admin_session_settings(session, raw) is not None
    assert session.settings is before


def test_apply_admin_session_settings_resets_workspace_on_embedding_change(base_settings, tmp_path) -> None:
    from observability.dashboard.services.session_context import apply_admin_session_settings, settings_to_raw

    session = SessionContext.create(base_settings, "admin", root=tmp_path)
    session.paths.chroma.mkdir(parents=True, exist_ok=True)
    marker = session.paths.chroma / "old_index.bin"
    marker.write_text("stale")
    raw = settings_to_raw(session.settings)
    raw["embedding"]["model"] = "BAAI/bge-m3"
    raw["embedding"]["dimensions"] = 1024

    assert apply_admin_session_settings(session, raw) is None
    assert session.settings.embedding.model == "BAAI/bge-m3"
    assert not marker.exists()
