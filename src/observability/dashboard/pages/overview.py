from __future__ import annotations

from typing import Any

from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.config_service import ConfigService


def render() -> None:
    import streamlit as st

    service = ConfigService()
    try:
        settings = service.load()
    except Exception as error:
        st.error(f"Failed to load settings: {error}")
        return

    st.title("SynapseRAG MCP")
    summary = service.app_summary(settings)
    left, middle, right = st.columns(3)
    left.metric("Environment", summary["environment"])
    middle.metric("Vector Backend", settings.vector_store.backend)
    right.metric("Collection", settings.vector_store.collection)

    st.subheader("Components")
    components = service.component_dicts(settings)
    columns = st.columns(4)
    for index, component in enumerate(components):
        with columns[index % len(columns)]:
            st.markdown(f"**{component['name']}**")
            st.caption(component["provider"])
            st.write(component["detail"])
            st.status(component["status"], expanded=False)
            metadata = component.get("metadata") or {}
            if metadata:
                st.json(metadata, expanded=False)

    st.subheader("Data Assets")
    stats = _collection_stats(settings)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Records", stats.get("record_count", 0))
    metric_columns[1].metric("Documents", stats.get("document_count", 0))
    metric_columns[2].metric("Sources", stats.get("source_count", 0))
    metric_columns[3].metric("Persisted", "yes" if stats.get("persisted") else "no")
    st.json(stats, expanded=False)


def _collection_stats(settings: Any) -> dict[str, Any]:
    try:
        return ChromaStore(settings.vector_store).get_collection_stats()
    except Exception as error:
        return {
            "collection": settings.vector_store.collection,
            "record_count": 0,
            "document_count": 0,
            "source_count": 0,
            "persisted": False,
            "error": str(error),
        }
