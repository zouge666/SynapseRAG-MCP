from __future__ import annotations

from observability.dashboard.pages import data_browser, evaluation_panel, ingestion_manager, ingestion_traces, overview, query_console, query_traces


def _apply_app_styles(st) -> None:
    """Keep the dashboard compact and make the native sidebar easier to scan."""
    st.markdown(
        """
        <style>
        /* The Streamlit defaults leave a desktop dashboard with unnecessary whitespace. */
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1600px;
            padding-top: 2.1rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.8rem;
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
            transform: translateX(0) !important;
            overflow: visible;
        }

        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 4.5rem !important;
            overflow: hidden;
        }

        [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"] a {
            justify-content: center;
            padding-left: 0;
            padding-right: 0;
        }

        [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"] a span:not([data-testid="stIconMaterial"]) {
            display: none;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="SynapseRAG MCP", page_icon="SR", layout="wide")
    _apply_app_styles(st)
    pages = [
        st.Page(overview.render, title="System Overview", icon=":material/dashboard:", url_path="overview", default=True),
        st.Page(data_browser.render, title="Data Browser", icon=":material/folder_open:", url_path="data-browser"),
        st.Page(
            ingestion_manager.render,
            title="Ingestion Manager",
            icon=":material/upload_file:",
            url_path="ingestion-manager",
        ),
        st.Page(query_console.render, title="Query", icon=":material/question_answer:", url_path="query"),
        st.Page(
            ingestion_traces.render,
            title="Ingestion Traces",
            icon=":material/timeline:",
            url_path="ingestion-traces",
        ),
        st.Page(query_traces.render, title="Query Traces", icon=":material/search:", url_path="query-traces"),
        st.Page(evaluation_panel.render, title="Evaluation", icon=":material/analytics:", url_path="evaluation"),
    ]
    navigation = st.navigation(pages)
    navigation.run()


if __name__ == "__main__":
    main()
