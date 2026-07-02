from types import SimpleNamespace

import pytest

from core.settings import load_settings
from observability.dashboard import runtime
from observability.dashboard.services.rate_limiter import RateLimiter
from observability.dashboard.services.session_context import SessionContext


class FakeSt:
    def __init__(self) -> None:
        self.session_state = {}
        self.warnings = []

    def warning(self, message) -> None:
        self.warnings.append(str(message))


@pytest.fixture
def base_settings():
    return load_settings("config/settings.yaml")


def test_mask_is_noop_in_local_mode(monkeypatch) -> None:
    monkeypatch.delenv(runtime.PUBLIC_ENV_VAR, raising=False)
    text = "failed at /Users/someone/private/doc.pdf with key sk-abc123secret"
    assert runtime.mask(text) == text


def test_mask_hides_absolute_paths_in_public_mode(monkeypatch) -> None:
    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")
    assert runtime.mask("error at /Users/someone/private/doc.pdf") == "error at •••/doc.pdf"
    assert runtime.mask("/tmp/synapserag_sessions/abc123/uploads/9f0e.pdf deleted") == "•••/9f0e.pdf deleted"


def test_mask_hides_secret_tokens_in_public_mode(monkeypatch) -> None:
    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")
    assert runtime.mask("key sk-fake-owner-key-000000 leaked") == "key ••• leaked"
    assert runtime.mask("token abcdefghijklmnopqrstuvwx0123456789 inside") == "token ••• inside"


def test_mask_obj_recurses_in_public_mode(monkeypatch) -> None:
    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")
    payload = {
        "source_path": "/Users/someone/docs/menu.pdf",
        "count": 3,
        "chunks": [{"path": "/tmp/synapserag_sessions/x/y.pdf"}, "plain text"],
    }
    masked = runtime.mask_obj(payload)
    assert masked["source_path"] == "•••/menu.pdf"
    assert masked["count"] == 3
    assert masked["chunks"][0]["path"] == "•••/y.pdf"
    assert masked["chunks"][1] == "plain text"


def test_mask_obj_is_noop_in_local_mode(monkeypatch) -> None:
    monkeypatch.delenv(runtime.PUBLIC_ENV_VAR, raising=False)
    payload = {"source_path": "/Users/someone/docs/menu.pdf"}
    assert runtime.mask_obj(payload) == payload


def test_require_settings_local_mode_ignores_session(monkeypatch) -> None:
    monkeypatch.delenv(runtime.PUBLIC_ENV_VAR, raising=False)
    settings = runtime.require_settings(FakeSt())
    assert settings is not None


def test_require_settings_public_without_session_warns(monkeypatch) -> None:
    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")
    st = FakeSt()
    assert runtime.require_settings(st) is None
    assert any("Start a guest session" in warning for warning in st.warnings)


def test_require_settings_public_returns_active_session_settings(monkeypatch, base_settings, tmp_path) -> None:
    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")
    session = SessionContext.create(base_settings, "guest", root=tmp_path)
    st = FakeSt()
    st.session_state["session_context"] = session
    assert runtime.require_settings(st) is session.settings
    assert st.warnings == []


def test_require_settings_public_expires_session(monkeypatch, base_settings, tmp_path) -> None:
    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")
    now = [1_000_000.0]
    session = SessionContext.create(base_settings, "guest", root=tmp_path, clock=lambda: now[0])
    st = FakeSt()
    st.session_state["session_context"] = session

    now[0] += 3700

    assert runtime.require_settings(st) is None
    assert "session_context" not in st.session_state
    assert not session.paths.root.exists()
    assert any("expired" in warning for warning in st.warnings)


def test_admin_session_disables_masking(monkeypatch, base_settings, tmp_path) -> None:
    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")
    session = SessionContext.create(base_settings, "admin", root=tmp_path)
    st = FakeSt()
    st.session_state["session_context"] = session

    assert runtime.require_settings(st) is session.settings
    assert runtime.admin_view() is True
    text = "error at /tmp/synapserag_sessions/abc/uploads/9f0e.pdf with sk-fake-owner-key-000000"
    assert runtime.mask(text) == text
    assert runtime.mask_obj({"source_path": "/tmp/x/y.pdf"})["source_path"] == "/tmp/x/y.pdf"


def test_guest_session_keeps_masking(monkeypatch, base_settings, tmp_path) -> None:
    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")
    session = SessionContext.create(base_settings, "guest", root=tmp_path)
    st = FakeSt()
    st.session_state["session_context"] = session

    runtime.require_settings(st)
    assert runtime.admin_view() is False
    assert runtime.mask("/tmp/synapserag_sessions/abc/y.pdf") == "•••/y.pdf"


def test_admin_view_resets_when_session_lost(monkeypatch, base_settings, tmp_path) -> None:
    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")
    session = SessionContext.create(base_settings, "admin", root=tmp_path)
    st = FakeSt()
    st.session_state["session_context"] = session
    runtime.require_settings(st)
    assert runtime.admin_view() is True

    st.session_state.pop("session_context")
    assert runtime.require_settings(st) is None
    assert runtime.admin_view() is False
    assert runtime.mask("/tmp/synapserag_sessions/abc/y.pdf") == "•••/y.pdf"


def test_owner_llm_quota_ok_without_admin_session(monkeypatch) -> None:
    st = FakeSt()
    assert runtime.owner_llm_quota_ok(st) is True


def test_owner_llm_quota_ok_guest_not_throttled(monkeypatch, base_settings, tmp_path) -> None:
    session = SessionContext.create(base_settings, "guest", root=tmp_path)
    st = FakeSt()
    st.session_state["session_context"] = session
    assert runtime.owner_llm_quota_ok(st) is True


def test_owner_llm_quota_blocks_admin_when_hourly_exhausted(monkeypatch, base_settings, tmp_path) -> None:
    limiter = RateLimiter()
    monkeypatch.setattr(runtime, "owner_llm_limiter", limiter)
    session = SessionContext.create(base_settings, "admin", root=tmp_path)
    st = FakeSt()
    st.session_state["session_context"] = session

    for _ in range(30):
        assert runtime.owner_llm_quota_ok(st) is True
    assert runtime.owner_llm_quota_ok(st) is False


def test_llm_missing_notice_only_when_public_without_key(monkeypatch, base_settings, tmp_path) -> None:
    from types import SimpleNamespace

    st = FakeSt()
    st.info = lambda message: st.warnings.append(str(message))
    monkeypatch.delenv(runtime.PUBLIC_ENV_VAR, raising=False)
    assert runtime.llm_missing_notice(st, SimpleNamespace(llm=SimpleNamespace(api_key=""))) is False

    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")
    assert runtime.llm_missing_notice(st, SimpleNamespace(llm=SimpleNamespace(api_key="sk-x"))) is False
    assert st.warnings == []
    assert runtime.llm_missing_notice(st, SimpleNamespace(llm=SimpleNamespace(api_key=""))) is True
    assert any("Settings page" in warning for warning in st.warnings)
