import sys
from types import ModuleType, SimpleNamespace

from observability.dashboard import app
from observability.dashboard.pages import data_browser, evaluation_panel, ingestion_manager, ingestion_traces, overview, query_traces


class FakeContext:
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
            if name in {"text_input"}:
                return args[1] if len(args) > 1 else kwargs.get("value", "")
            if name == "file_uploader":
                return None
            if name == "columns":
                spec = args[0] if args else 1
                count = len(spec) if isinstance(spec, list) else int(spec)
                return [FakeContext() for _ in range(count)]
            if name in {"expander", "status"}:
                return FakeContext()
            if name == "empty":
                return FakeContext()
            if name == "progress":
                return FakeContext()
            return None

        return call


class FakeStreamlit(ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.sidebar = FakeContext()
        self.pages = []

    def Page(self, render, title: str, icon: str):
        page = SimpleNamespace(render=render, title=title, icon=icon)
        self.pages.append(page)
        return page

    def navigation(self, pages):
        self.pages = list(pages)
        return SimpleNamespace(run=lambda: [page.render() for page in pages])

    def __getattr__(self, name):
        return getattr(FakeContext(), name)


def install_fake_streamlit(monkeypatch) -> FakeStreamlit:
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    return fake


def test_dashboard_app_registers_and_runs_all_pages(monkeypatch) -> None:
    fake = install_fake_streamlit(monkeypatch)

    app.main()

    assert [page.title for page in fake.pages] == [
        "System Overview",
        "Data Browser",
        "Ingestion Manager",
        "Ingestion Traces",
        "Query Traces",
        "Evaluation",
    ]


def test_dashboard_pages_render_without_python_exceptions(monkeypatch) -> None:
    install_fake_streamlit(monkeypatch)

    for page in [overview, data_browser, ingestion_manager, ingestion_traces, query_traces, evaluation_panel]:
        page.render()
