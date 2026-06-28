import pytest

from core.settings import VectorStoreSettings, load_settings
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord, VectorSearchResult
from libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    def __init__(self, settings: VectorStoreSettings) -> None:
        super().__init__(settings)
        self.records: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord], trace: object | None = None) -> int:
        for record in records:
            self.records[record.id] = record
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[VectorSearchResult]:
        filters = filters or {}
        results = []
        for record in self.records.values():
            if any(record.metadata.get(key) != value for key, value in filters.items()):
                continue
            score = sum(left * right for left, right in zip(vector, record.vector))
            results.append(VectorSearchResult(id=record.id, score=score, text=record.text, metadata=record.metadata))
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def get_by_ids(self, ids: list[str], trace: object | None = None) -> list[VectorRecord]:
        return [self.records[record_id] for record_id in ids if record_id in self.records]

    def delete_by_metadata(self, filters: dict[str, object]) -> int:
        if not isinstance(filters, dict) or not filters:
            raise ValueError("filters must be a non-empty object")
        removed = [record_id for record_id, record in self.records.items() if all(record.metadata.get(key) == value for key, value in filters.items())]
        for record_id in removed:
            self.records.pop(record_id, None)
        return len(removed)


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    VectorStoreFactory.unregister_provider("fake")
    VectorStoreFactory.unregister_provider("chroma")
    yield
    VectorStoreFactory.unregister_provider("fake")
    VectorStoreFactory.unregister_provider("chroma")


def test_vector_store_contract_upsert_and_query_shape() -> None:
    settings = VectorStoreSettings(backend="fake", persist_path="memory")
    store = FakeVectorStore(settings)
    records = [
        VectorRecord(id="a", vector=[1.0, 0.0], text="alpha", metadata={"collection": "docs"}),
        VectorRecord(id="b", vector=[0.5, 0.5], text="beta", metadata={"collection": "docs"}),
        VectorRecord(id="c", vector=[0.0, 1.0], text="gamma", metadata={"collection": "other"}),
    ]

    count = store.upsert(records)
    results = store.query([1.0, 0.0], top_k=2, filters={"collection": "docs"})

    assert count == 3
    assert all(isinstance(result, VectorSearchResult) for result in results)
    assert [result.id for result in results] == ["a", "b"]
    assert results[0].text == "alpha"
    assert results[0].metadata == {"collection": "docs"}
    assert [record.id for record in store.get_by_ids(["c", "missing", "a"])] == ["c", "a"]


def test_factory_creates_registered_backend_from_vector_store_settings() -> None:
    VectorStoreFactory.register_provider("fake", FakeVectorStore)
    settings = VectorStoreSettings(backend="fake", persist_path="memory")

    store = VectorStoreFactory.create(settings)

    assert isinstance(store, FakeVectorStore)
    assert store.settings.persist_path == "memory"


def test_factory_creates_registered_backend_from_project_settings() -> None:
    VectorStoreFactory.register_provider("chroma", FakeVectorStore)
    settings = load_settings("config/settings.yaml")

    store = VectorStoreFactory.create(settings)

    assert isinstance(store, FakeVectorStore)
    assert store.settings.backend == "chroma"


def test_factory_rejects_unknown_backend() -> None:
    settings = VectorStoreSettings(backend="missing", persist_path="memory")

    with pytest.raises(ValueError, match="unsupported vector store backend: missing"):
        VectorStoreFactory.create(settings)


def test_vector_store_contract_delete_by_metadata_removes_only_matching_records() -> None:
    settings = VectorStoreSettings(backend="fake", persist_path="memory")
    store = FakeVectorStore(settings)
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0], text="alpha", metadata={"source_path": "docs/a.pdf", "collection": "docs"}),
            VectorRecord(id="b", vector=[1.0], text="beta", metadata={"source_path": "docs/a.pdf", "collection": "docs"}),
            VectorRecord(id="c", vector=[1.0], text="gamma", metadata={"source_path": "docs/c.pdf", "collection": "docs"}),
        ]
    )

    deleted = store.delete_by_metadata({"source_path": "docs/a.pdf"})

    assert deleted == 2
    assert [record.id for record in store.get_by_ids(["a", "b", "c"])] == ["c"]
    assert [result.id for result in store.query([1.0], top_k=5, filters={"collection": "docs"})] == ["c"]


def test_vector_store_contract_delete_by_metadata_returns_zero_for_no_match() -> None:
    settings = VectorStoreSettings(backend="fake", persist_path="memory")
    store = FakeVectorStore(settings)
    store.upsert([VectorRecord(id="a", vector=[1.0], text="alpha", metadata={"source_path": "docs/a.pdf"})])

    deleted = store.delete_by_metadata({"source_path": "docs/missing.pdf"})

    assert deleted == 0
    assert [record.id for record in store.get_by_ids(["a"])] == ["a"]


@pytest.mark.parametrize("filters", [{}, []])
def test_vector_store_contract_delete_by_metadata_rejects_invalid_filters(filters: object) -> None:
    settings = VectorStoreSettings(backend="fake", persist_path="memory")
    store = FakeVectorStore(settings)

    with pytest.raises(ValueError, match="filters must be a non-empty object"):
        store.delete_by_metadata(filters)
