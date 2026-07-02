import sys
from types import ModuleType, SimpleNamespace

from observability.dashboard import app, runtime
from observability.dashboard.pages import data_browser, evaluation_panel, ingestion_manager, ingestion_traces, llm_chat, overview, query_console, query_traces, settings_page
from observability.dashboard.public_pages import session_page, start_page


class FakeContext:
    def __init__(self, status_calls=None):
        self.status_calls = status_calls if status_calls is not None else []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, name):
        def call(*args, **kwargs):
            if name == "button":
                return False
            if name == "checkbox":
                return kwargs.get("value", False)
            if name == "number_input":
                return kwargs.get("value", 1)
            if name == "multiselect":
                return kwargs.get("default", [])
            if name == "selectbox":
                options = args[1] if len(args) > 1 else kwargs.get("options", [])
                return options[0] if options else None
            if name == "text_input":
                return kwargs.get("value", "")
            if name == "tabs":
                labels = args[0] if args else kwargs.get("tabs", [])
                return [FakeContext(self.status_calls) for _ in labels]
            if name == "file_uploader":
                return None
            if name == "columns":
                spec = args[0] if args else 1
                count = len(spec) if isinstance(spec, list) else int(spec)
                return [FakeContext(self.status_calls) for _ in range(count)]
            if name in {"expander", "container", "empty", "status", "spinner", "progress"}:
                return FakeContext(self.status_calls)
            return None

        return call


class FakeStreamlit(ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.status_calls = []
        self.warnings = []
        self.session_state = {}
        self.pages = []
        self.secrets = {}

    def Page(self, render, title: str, icon: str, url_path: str, default: bool = False, visibility: str = "visible"):
        page = SimpleNamespace(render=render, title=title, icon=icon, url_path=url_path, default=default, visibility=visibility)
        self.pages.append(page)
        return page

    def navigation(self, pages):
        self.pages = list(pages)
        return SimpleNamespace(run=lambda: None)

    def warning(self, message) -> None:
        self.warnings.append(str(message))

    @property
    def sidebar(self):
        return FakeContext(self.status_calls)

    def __getattr__(self, name):
        return getattr(FakeContext(self.status_calls), name)


def install_fake_streamlit(monkeypatch) -> FakeStreamlit:
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    return fake


def stub_housekeeping(monkeypatch) -> None:
    stub = SimpleNamespace(ensure_cleanup_thread=lambda **kwargs: None, cleanup_expired=lambda: 0)
    monkeypatch.setattr(runtime, "_cleanup_service", stub)


def test_public_app_registers_full_dashboard_nav(monkeypatch) -> None:
    fake = install_fake_streamlit(monkeypatch)
    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")
    stub_housekeeping(monkeypatch)

    app.main()

    assert [page.title for page in fake.pages] == [
        "Start",
        "System Overview",
        "Data Browser",
        "Ingestion Manager",
        "Query",
        "LLM",
        "Ingestion Traces",
        "Query Traces",
        "Evaluation",
        "Session",
        "Settings",
    ]
    assert [page.url_path for page in fake.pages] == [
        "",
        "overview",
        "data-browser",
        "ingestion-manager",
        "query",
        "llm",
        "ingestion-traces",
        "query-traces",
        "evaluation",
        "session",
        "settings",
    ]
    assert [page.title for page in fake.pages if page.default] == ["Start"]
    assert all(page.visibility == "visible" for page in fake.pages)


def test_local_app_nav_unchanged(monkeypatch) -> None:
    fake = install_fake_streamlit(monkeypatch)
    monkeypatch.delenv(runtime.PUBLIC_ENV_VAR, raising=False)

    app.main()

    visible = [page for page in fake.pages if page.visibility == "visible"]
    assert [page.title for page in visible] == [
        "System Overview",
        "Data Browser",
        "Ingestion Manager",
        "Query",
        "LLM",
        "Ingestion Traces",
        "Query Traces",
        "Evaluation",
        "Settings",
    ]
    assert [page.url_path for page in visible if page.default] == [""]
    assert [page.url_path for page in fake.pages if page.visibility == "hidden"] == ["overview"]


def test_public_pages_gate_without_session(monkeypatch) -> None:
    fake = install_fake_streamlit(monkeypatch)
    monkeypatch.setenv(runtime.PUBLIC_ENV_VAR, "1")

    pages = [start_page, overview, data_browser, ingestion_manager, query_console, llm_chat, ingestion_traces, query_traces, evaluation_panel, session_page, settings_page]
    for page in pages:
        page.render()

    assert any("Start a guest session" in warning for warning in fake.warnings)
