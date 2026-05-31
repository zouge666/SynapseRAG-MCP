from core.settings import VectorStoreSettings
from libs.vector_store.base_vector_store import VectorRecord
from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.config_service import ConfigService


def test_config_service_formats_dashboard_components() -> None:
    service = ConfigService("config/settings.yaml")
    settings = service.load()

    components = service.component_dicts(settings)
    names = {component["name"] for component in components}

    assert service.app_summary(settings)["name"] == "synapserag-mcp"
    assert "LLM" in names
    assert "Embedding" in names
    assert "Vector Store" in names
    assert "Observability" in names
    assert next(component for component in components if component["name"] == "Vector Store")["metadata"]["persist_path"] == "data/db/chroma"


def test_chroma_store_collection_stats_counts_records_documents_and_sources(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path), collection="docs"))
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0], text="alpha", metadata={"source_path": "docs/a.pdf", "document_id": "doc-a"}),
            VectorRecord(id="b", vector=[0.5], text="beta", metadata={"source_path": "docs/a.pdf", "document_id": "doc-a"}),
            VectorRecord(id="c", vector=[0.0], text="gamma", metadata={"source_path": "docs/c.pdf", "document_id": "doc-c"}),
        ]
    )

    stats = store.get_collection_stats()

    assert stats["collection"] == "docs"
    assert stats["record_count"] == 3
    assert stats["document_count"] == 2
    assert stats["source_count"] == 2
    assert stats["persisted"] is True
