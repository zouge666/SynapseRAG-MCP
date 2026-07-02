from __future__ import annotations

from html import escape
from typing import Any

from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.config_service import ConfigService


def render() -> None:
    import streamlit as st

    from observability.dashboard import runtime

    service = ConfigService()
    try:
        settings = runtime.require_settings(st)
    except Exception as error:
        st.error(f"Failed to load settings: {error}")
        return
    if settings is None:
        return

    st.title("SynapseRAG MCP")
    summary = service.app_summary(settings)
    left, middle, right = st.columns(3)
    left.metric("Environment", summary["environment"])
    middle.metric("Vector Backend", settings.vector_store.backend)
    right.metric("Collection", settings.vector_store.collection)

    st.subheader("Components")
    components = [runtime.mask_obj(component) for component in service.component_dicts(settings)]
    for start in range(0, len(components), 4):
        columns = st.columns(4)
        for column, component in zip(columns, components[start : start + 4]):
            with column:
                st.markdown(_component_card(component), unsafe_allow_html=True)

    st.subheader("Data Assets")
    stats = _collection_stats(settings)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Records", stats.get("record_count", 0))
    metric_columns[1].metric("Documents", stats.get("document_count", 0))
    metric_columns[2].metric("Sources", stats.get("source_count", 0))
    metric_columns[3].metric("Persisted", "yes" if stats.get("persisted") else "no")
    st.json(runtime.mask_obj(stats), expanded=False)


def _component_card(component: dict[str, Any]) -> str:
    """Render a fixed-height summary so component rows stay visually aligned."""
    is_disabled = component["status"] == "disabled"
    status_class = " synapserag-component-card__status--disabled" if is_disabled else ""
    status_text = "○ Disabled" if is_disabled else "✓ Configured"
    metadata = component.get("metadata") or {}
    metadata_text = _metadata_summary(metadata)
    metadata_html = (
        f'<div class="synapserag-component-card__provider">{escape(metadata_text)}</div>' if metadata_text else ""
    )
    return (
        '<div class="synapserag-component-card">'
        f'<div class="synapserag-component-card__name">{escape(str(component["name"]))}</div>'
        f'<div class="synapserag-component-card__provider">{escape(str(component["provider"]))}</div>'
        f'<div class="synapserag-component-card__detail">{escape(str(component["detail"]))}</div>'
        f'<div class="synapserag-component-card__status{status_class}">{status_text}</div>'
        f"{metadata_html}"
        "</div>"
    )


def _metadata_summary(metadata: dict[str, Any]) -> str:
    return " · ".join(f"{key}: {value}" for key, value in metadata.items())


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
