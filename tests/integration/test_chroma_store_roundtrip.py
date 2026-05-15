import pytest

from core.settings import VectorStoreSettings
from libs.vector_store.base_vector_store import VectorRecord
from libs.vector_store.chroma_store import ChromaStore, ChromaStoreError
from libs.vector_store.vector_store_factory import VectorStoreFactory


def test_factory_routes_chroma_store(tmp_path) -> None:
    settings = VectorStoreSettings(backend="chroma", persist_path=str(tmp_path), collection="docs")

    store = VectorStoreFactory.create(settings)

    assert isinstance(store, ChromaStore)


def test_chroma_store_upsert_query_roundtrip_with_top_k_and_filters(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path), collection="docs"))
    records = [
        VectorRecord(id="alpha", vector=[1.0, 0.0], text="alpha text", metadata={"collection": "docs", "kind": "guide"}),
        VectorRecord(id="beta", vector=[0.8, 0.2], text="beta text", metadata={"collection": "docs", "kind": "guide"}),
        VectorRecord(id="gamma", vector=[0.0, 1.0], text="gamma text", metadata={"collection": "docs", "kind": "api"}),
    ]

    count = store.upsert(records)
    results = store.query([1.0, 0.0], top_k=2, filters={"kind": "guide"})

    assert count == 3
    assert [result.id for result in results] == ["alpha", "beta"]
    assert results[0].text == "alpha text"
    assert results[0].metadata == {"collection": "docs", "kind": "guide"}
    assert results[0].score > results[1].score


def test_chroma_store_persists_records_between_instances(tmp_path) -> None:
    settings = VectorStoreSettings(backend="chroma", persist_path=str(tmp_path), collection="docs")
    first = ChromaStore(settings)
    first.upsert([VectorRecord(id="alpha", vector=[1.0, 0.0], text="alpha text", metadata={"kind": "guide"})])

    second = ChromaStore(settings)
    results = second.query([1.0, 0.0], top_k=1)

    assert [result.id for result in results] == ["alpha"]
    assert results[0].text == "alpha text"


def test_chroma_store_upsert_replaces_existing_record(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path), collection="docs"))
    store.upsert([VectorRecord(id="alpha", vector=[1.0, 0.0], text="old", metadata={"version": 1})])

    store.upsert([VectorRecord(id="alpha", vector=[0.0, 1.0], text="new", metadata={"version": 2})])
    results = store.query([0.0, 1.0], top_k=1)

    assert len(results) == 1
    assert results[0].id == "alpha"
    assert results[0].text == "new"
    assert results[0].metadata == {"version": 2}


def test_chroma_store_rejects_invalid_query(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path), collection="docs"))
    store.upsert([VectorRecord(id="alpha", vector=[1.0, 0.0], text="alpha", metadata={})])

    with pytest.raises(ChromaStoreError, match="top_k"):
        store.query([1.0, 0.0], top_k=0)

    with pytest.raises(ChromaStoreError, match="dimensions"):
        store.query([1.0], top_k=1)
