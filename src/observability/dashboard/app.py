from __future__ import annotations

import functools

from observability.dashboard.pages import data_browser, evaluation_panel, ingestion_manager, ingestion_traces, llm_chat, overview, query_console, query_traces, settings_page
from observability.dashboard.public_pages import session_page, start_page


def _apply_app_styles(st) -> None:
    """Keep the dashboard compact and make the native sidebar easier to scan."""
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] .main .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width: 1600px;
            padding-top: 4.25rem;
            padding-bottom: 2.5rem;
            padding-left: clamp(1.5rem, 5vw, 7rem);
            padding-right: clamp(1.5rem, 5vw, 7rem);
        }

        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: fit-content !important;
            max-width: fit-content !important;
        }

        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: fit-content !important;
            min-width: 12.5rem !important;
            max-width: 17rem !important;
        }

        [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarNav"] a {
            white-space: nowrap;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.8rem;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
            height: auto;
            min-height: 2rem;
            margin-bottom: 0.35rem;
            justify-content: flex-end;
        }

        [data-testid="stSidebar"] [data-testid="stLogoSpacer"] {
            display: none;
        }

        [data-testid="stSidebarNav"] {
            padding-top: 0.35rem;
        }

        [data-testid="stSidebarNav"] a {
            min-height: 2.85rem;
            padding: 0.55rem 0.75rem;
            font-size: 1.05rem;
        }

        [data-testid="stSidebarNav"] a span {
            font-size: 1.05rem;
        }

        [data-testid="stSidebarNav"] [data-testid="stIconMaterial"] {
            font-size: 1.42rem;
        }

        /* Keep an icon rail visible when the native sidebar is collapsed. */
        [data-testid="stSidebar"][aria-expanded="false"] {
            min-width: 4.5rem !important;
            max-width: 4.5rem !important;
            height: 100vh !important;
            transform: translateX(0) !important;
            overflow: visible;
        }

        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 4.5rem !important;
            overflow: hidden;
        }

        [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarHeader"] {
            display: none;
        }

        [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarUserContent"],
        [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavSeparator"] {
            display: none;
        }

        [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"] a {
            justify-content: center;
            padding-left: 0;
            padding-right: 0;
        }

        [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"] a > span:not(:has([data-testid="stIconMaterial"])) {
            display: none;
        }

        [data-testid="stButtonGroup"] button[data-selected] {
            background-color: rgb(38, 132, 78) !important;
            border-color: rgb(38, 132, 78) !important;
            color: rgb(255, 255, 255) !important;
        }

        .synapserag-component-card {
            min-height: 7.1rem;
            margin-bottom: 0.65rem;
            padding: 0.85rem 0.95rem;
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 0.65rem;
            background: rgba(250, 250, 252, 0.55);
        }

        .synapserag-component-card__name {
            margin-bottom: 0.25rem;
            font-weight: 700;
            font-size: 1.05rem;
        }

        .synapserag-component-card__provider {
            color: rgb(128, 132, 149);
            font-size: 0.88rem;
        }

        .synapserag-component-card__detail {
            margin: 0.25rem 0 0.45rem;
            font-size: 1rem;
        }

        .synapserag-component-card__status {
            font-size: 0.88rem;
            color: rgb(38, 132, 78);
        }

        .synapserag-component-card__status--disabled {
            color: rgb(128, 132, 149);
        }

        .sr-section-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 0.75rem;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid rgba(49, 51, 63, 0.15);
            font-size: 1.12rem;
            font-weight: 700;
        }

        .sr-section-title::before {
            content: "";
            width: 0.5rem;
            height: 1.1rem;
            border-radius: 0.25rem;
            background: rgb(38, 132, 78);
        }

        [data-testid="stLayoutWrapper"]:has(.sr-section-title--pipeline) > [data-testid="stVerticalBlock"] {
            display: flex;
            flex-direction: row !important;
            flex-wrap: wrap;
            column-gap: 1.25rem;
            row-gap: 0.9rem;
            align-items: flex-end;
        }

        [data-testid="stLayoutWrapper"]:has(.sr-section-title--pipeline) > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {
            flex: 0 0 auto;
            width: auto !important;
            min-width: 0;
        }

        [data-testid="stLayoutWrapper"]:has(.sr-section-title--pipeline) > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has([data-testid="stButtonGroup"]) {
            padding: 0.55rem 0.8rem 0.7rem;
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 0.6rem;
            background: rgba(250, 250, 252, 0.65);
        }

        [data-testid="stLayoutWrapper"]:has(.sr-section-title--pipeline) > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.sr-section-title),
        [data-testid="stLayoutWrapper"]:has(.sr-section-title--pipeline) > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has([data-testid="stCaptionContainer"]),
        [data-testid="stLayoutWrapper"]:has(.sr-section-title--pipeline) > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has([data-testid="stAlert"]),
        [data-testid="stLayoutWrapper"]:has(.sr-section-title--pipeline) > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
            flex: 1 1 100%;
        }

        [data-testid="stLayoutWrapper"]:has(.sr-section-title) [data-testid="stButton"] button {
            width: auto;
            min-width: 7rem;
            padding-left: 1.4rem;
            padding-right: 1.4rem;
        }

        [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]),
        [data-testid="stHorizontalBlock"]:has(.synapserag-component-card) {
            flex-wrap: wrap;
            column-gap: clamp(0.75rem, 1.6vw, 2.25rem);
            row-gap: clamp(0.75rem, 1.6vw, 2.25rem);
        }

        [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
            flex: 1 1 9.5rem !important;
            min-width: 9.5rem !important;
            width: auto !important;
        }

        [data-testid="stHorizontalBlock"]:has(.synapserag-component-card) > [data-testid="stColumn"] {
            flex: 1 1 15rem !important;
            min-width: 15rem !important;
            width: auto !important;
        }

        @media (max-width: 768px) {
            [data-testid="stSidebar"][aria-expanded="false"] {
                min-width: 0 !important;
                max-width: 0 !important;
                height: 0 !important;
                transform: translateX(-100%) !important;
            }

            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 0 !important;
            }

            [data-testid="stAppViewContainer"] .main .block-container,
            [data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _auto_expand_sidebar() -> None:
    try:
        from streamlit.components.v1 import html as components_html
    except Exception:
        return
    components_html(
        """
        <script>
        (function expandSidebar(attemptsLeft) {
            const doc = window.parent.document;
            if (!doc) return;
            if (window.parent.innerWidth <= 768) return;
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar || sidebar.getAttribute('aria-expanded') !== 'false') return;
            const button = doc.querySelector('button[data-testid="stExpandSidebarButton"]');
            if (button) {
                button.click();
            } else if (attemptsLeft > 0) {
                window.setTimeout(() => expandSidebar(attemptsLeft - 1), 100);
            }
        })(10);
        </script>
        """,
        height=0,
    )


def _with_sidebar_auto_expand(render_fn):
    @functools.wraps(render_fn)
    def _wrapped() -> None:
        _auto_expand_sidebar()
        render_fn()

    return _wrapped


def main() -> None:
    import streamlit as st

    from observability.dashboard import runtime

    st.set_page_config(page_title="SynapseRAG MCP", page_icon="SR", layout="wide")
    _apply_app_styles(st)
    if runtime.is_public():
        runtime.start_public_housekeeping()
    pages = _local_pages(st) if not runtime.is_public() else _public_pages(st)
    navigation = st.navigation(pages)
    navigation.run()


def _local_pages(st) -> list:
    return [
        st.Page(overview.render, title="System Overview", icon=":material/dashboard:", url_path="", default=True),
        st.Page(overview.render, title="System Overview", icon=":material/dashboard:", url_path="overview", visibility="hidden"),
        st.Page(
            _with_sidebar_auto_expand(data_browser.render),
            title="Data Browser",
            icon=":material/folder_open:",
            url_path="data-browser",
        ),
        st.Page(
            ingestion_manager.render,
            title="Ingestion Manager",
            icon=":material/upload_file:",
            url_path="ingestion-manager",
        ),
        st.Page(query_console.render, title="Query", icon=":material/question_answer:", url_path="query"),
        st.Page(llm_chat.render, title="LLM", icon=":material/chat:", url_path="llm"),
        st.Page(
            _with_sidebar_auto_expand(ingestion_traces.render),
            title="Ingestion Traces",
            icon=":material/timeline:",
            url_path="ingestion-traces",
        ),
        st.Page(
            _with_sidebar_auto_expand(query_traces.render),
            title="Query Traces",
            icon=":material/search:",
            url_path="query-traces",
        ),
        st.Page(
            _with_sidebar_auto_expand(evaluation_panel.render),
            title="Evaluation",
            icon=":material/analytics:",
            url_path="evaluation",
        ),
        st.Page(settings_page.render, title="Settings", icon=":material/settings:", url_path="settings"),
    ]


def _public_pages(st) -> list:
    return [
        st.Page(start_page.render, title="Start", icon=":material/login:", url_path="", default=True),
        st.Page(overview.render, title="System Overview", icon=":material/dashboard:", url_path="overview"),
        st.Page(
            _with_sidebar_auto_expand(data_browser.render),
            title="Data Browser",
            icon=":material/folder_open:",
            url_path="data-browser",
        ),
        st.Page(
            ingestion_manager.render,
            title="Ingestion Manager",
            icon=":material/upload_file:",
            url_path="ingestion-manager",
        ),
        st.Page(query_console.render, title="Query", icon=":material/question_answer:", url_path="query"),
        st.Page(llm_chat.render, title="LLM", icon=":material/chat:", url_path="llm"),
        st.Page(
            _with_sidebar_auto_expand(ingestion_traces.render),
            title="Ingestion Traces",
            icon=":material/timeline:",
            url_path="ingestion-traces",
        ),
        st.Page(
            _with_sidebar_auto_expand(query_traces.render),
            title="Query Traces",
            icon=":material/search:",
            url_path="query-traces",
        ),
        st.Page(
            _with_sidebar_auto_expand(evaluation_panel.render),
            title="Evaluation",
            icon=":material/analytics:",
            url_path="evaluation",
        ),
        st.Page(session_page.render, title="Session", icon=":material/hourglass_empty:", url_path="session"),
        st.Page(settings_page.render, title="Settings", icon=":material/settings:", url_path="settings"),
    ]


if __name__ == "__main__":
    main()
