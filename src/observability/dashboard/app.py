from __future__ import annotations

from observability.dashboard.pages import data_browser, evaluation_panel, ingestion_manager, ingestion_traces, overview, query_traces


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="SynapseRAG MCP", page_icon="SR", layout="wide")
    pages = [
        st.Page(overview.render, title="System Overview", icon=":material/dashboard:"),
        st.Page(data_browser.render, title="Data Browser", icon=":material/folder_open:"),
        st.Page(ingestion_manager.render, title="Ingestion Manager", icon=":material/upload_file:"),
        st.Page(ingestion_traces.render, title="Ingestion Traces", icon=":material/timeline:"),
        st.Page(query_traces.render, title="Query Traces", icon=":material/search:"),
        st.Page(evaluation_panel.render, title="Evaluation", icon=":material/analytics:"),
    ]
    navigation = st.navigation(pages)
    navigation.run()


if __name__ == "__main__":
    main()
