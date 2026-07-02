import contextlib
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from core.settings import LLMSettings, load_settings
from ingestion import IngestionPipeline
from observability.dashboard.pages.query_console import run_dashboard_query
from observability.dashboard import runtime
from observability.dashboard.public_pages import upload_page
from observability.dashboard.services.data_service import DataService
from observability.dashboard.services.rate_limiter import RateLimiter
from observability.dashboard.services.session_context import SESSION_COLLECTION, SessionContext


GUEST_KEY = "sk-guest-test-secret-12345"


class FakeClock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


@pytest.fixture
def base_settings():
    return load_settings("config/settings.yaml")


def make_session(base_settings, tmp_path: Path, name: str, llm: LLMSettings | None = None) -> SessionContext:
    return SessionContext.create(base_settings, "guest", llm=llm, session_id=name, root=tmp_path / "sessions", clock=FakeClock())


def ingest(session: SessionContext, pdf: Path) -> None:
    result = IngestionPipeline(session.settings).run(str(pdf), collection=SESSION_COLLECTION)
    assert result.status in {"success", "skipped"}


def test_sessions_cannot_see_each_others_collections_or_documents(base_settings, tmp_path) -> None:
    alpha = make_session(base_settings, tmp_path, "alpha")
    beta = make_session(base_settings, tmp_path, "beta")
    alpha_pdf = tmp_path / "alpha.pdf"
    beta_pdf = tmp_path / "beta.pdf"
    make_pdf(alpha_pdf, "The alpha-only menu features a yuzu tonic priced at 9 USD.")
    make_pdf(beta_pdf, "The beta-only handbook describes a midnight maintenance window.")

    ingest(alpha, alpha_pdf)
    ingest(beta, beta_pdf)

    alpha_results = run_dashboard_query(alpha.settings, "yuzu tonic price", SESSION_COLLECTION, 5)
    beta_results = run_dashboard_query(beta.settings, "yuzu tonic price", SESSION_COLLECTION, 5)

    assert alpha_results.results and "yuzu" in alpha_results.results[0].text
    assert all("yuzu" not in item.text for item in beta_results.results)

    alpha_docs = DataService(alpha.settings).list_documents()
    beta_docs = DataService(beta.settings).list_documents()
    assert len(alpha_docs) == 1 and len(beta_docs) == 1
    assert alpha_docs[0]["source_path"] != beta_docs[0]["source_path"]


def test_guest_key_never_written_to_session_files_or_global_config(base_settings, tmp_path) -> None:
    config_before = hashlib.sha256(Path("config/settings.yaml").read_bytes()).hexdigest()
    session = make_session(base_settings, tmp_path, "guest-key", llm=LLMSettings(provider="openai", model="m", api_key=GUEST_KEY))
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, "A tiny document about tea.")

    ingest(session, pdf)
    run_dashboard_query(session.settings, "tea", SESSION_COLLECTION, 5)

    leaked = []
    for path in session.paths.root.rglob("*"):
        if path.is_file():
            try:
                if GUEST_KEY in path.read_text(encoding="utf-8", errors="ignore"):
                    leaked.append(str(path))
            except OSError:
                pass
    assert leaked == []
    assert hashlib.sha256(Path("config/settings.yaml").read_bytes()).hexdigest() == config_before


class FakeUploaded:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getbuffer(self) -> bytes:
        return self._data


class FakeSt:
    def __init__(self) -> None:
        self.session_state = {}
        self.errors = []
        self.successes = []

    def error(self, message) -> None:
        self.errors.append(str(message))

    def success(self, message) -> None:
        self.successes.append(str(message))

    def spinner(self, *args, **kwargs):
        return contextlib.nullcontext()


def make_ingest_context(base_settings, tmp_path: Path, monkeypatch) -> tuple[SessionContext, FakeSt]:
    session = make_session(base_settings, tmp_path, "ingest-test")
    fake_st = FakeSt()
    monkeypatch.setattr(runtime, "upload_limiter", RateLimiter(clock=FakeClock()))
    return session, fake_st


def test_original_pdf_deleted_after_successful_processing(base_settings, tmp_path, monkeypatch) -> None:
    session, fake_st = make_ingest_context(base_settings, tmp_path, monkeypatch)
    pdf = tmp_path / "real.pdf"
    make_pdf(pdf, "Fresh coffee notes.")
    uploaded = FakeUploaded("../../evil name.pdf", pdf.read_bytes())

    upload_page.process_upload(fake_st, session, uploaded)

    assert fake_st.errors == []
    assert fake_st.successes
    assert list(session.paths.uploads.iterdir()) == []
    assert DataService(session.settings).list_documents()


def test_original_pdf_deleted_after_failed_processing(base_settings, tmp_path, monkeypatch) -> None:
    session, fake_st = make_ingest_context(base_settings, tmp_path, monkeypatch)

    class ExplodingPipeline:
        def __init__(self, settings) -> None:
            pass

        def run(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(upload_page, "IngestionPipeline", ExplodingPipeline)
    pdf = tmp_path / "real.pdf"
    make_pdf(pdf, "Failure path content.")
    uploaded = FakeUploaded("doc.pdf", pdf.read_bytes())

    upload_page.process_upload(fake_st, session, uploaded)

    assert fake_st.errors and "boom" in fake_st.errors[0]
    assert list(session.paths.uploads.iterdir()) == []


def test_new_upload_replaces_old_workspace(base_settings, tmp_path, monkeypatch) -> None:
    session, fake_st = make_ingest_context(base_settings, tmp_path, monkeypatch)
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    make_pdf(first, "First document about alpha beans.")
    make_pdf(second, "Second document about beta roast.")

    upload_page.process_upload(fake_st, session, FakeUploaded("first.pdf", first.read_bytes()))
    upload_page.process_upload(fake_st, session, FakeUploaded("second.pdf", second.read_bytes()))

    results = run_dashboard_query(session.settings, "alpha beans", SESSION_COLLECTION, 5)
    assert all("alpha" not in item.text for item in results.results)
    results = run_dashboard_query(session.settings, "beta roast", SESSION_COLLECTION, 5)
    assert results.results and "beta" in results.results[0].text


def test_upload_rate_limit_blocks_fourth_upload(base_settings, tmp_path, monkeypatch) -> None:
    session, fake_st = make_ingest_context(base_settings, tmp_path, monkeypatch)
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, "rate limit content")

    for _ in range(3):
        upload_page.process_upload(fake_st, session, FakeUploaded("doc.pdf", pdf.read_bytes()))
    assert fake_st.errors == []

    upload_page.process_upload(fake_st, session, FakeUploaded("doc.pdf", pdf.read_bytes()))

    assert fake_st.errors and "Upload limit" in fake_st.errors[-1]


def test_upload_limit_tracks_client_ip_across_sessions(base_settings, tmp_path, monkeypatch) -> None:
    from observability.dashboard.public_pages.upload_page import _client_upload_key

    monkeypatch.setattr(runtime, "upload_limiter", RateLimiter(clock=FakeClock()))
    headers = {"X-Forwarded-For": "203.0.113.7, 10.1.2.3"}
    fake_st = FakeSt()
    fake_st.context = SimpleNamespace(headers=headers)
    first = make_session(base_settings, tmp_path, "ip-test-a")
    second = make_session(base_settings, tmp_path, "ip-test-b")
    assert _client_upload_key(fake_st, first) == "ip:203.0.113.7"
    assert _client_upload_key(fake_st, second) == "ip:203.0.113.7"
    plain_st = FakeSt()
    assert _client_upload_key(plain_st, first) == "session:ip-test-a"
