from types import SimpleNamespace

import pytest

from core import ChunkRecord
from core.trace import TraceContext
from ingestion.storage import VectorUpserter, VectorUpserterError
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord, VectorSearchResult


class FakeVectorStore(BaseVectorStore):
    def __init__(self, count_offset: int = 0) -> None:
        super().__init__(SimpleNamespace(backend="fake", persist_path="memory"))
        self.records: dict[str, VectorRecord] = {}
        self.calls = []
        self.count_offset = count_offset

    def upsert(self, records: list[VectorRecord], trace: object | None = None) -> int:
        self.calls.append({"records": records, "trace": trace})
        for record in records:
            self.records[record.id] = record
        return len(records) + self.count_offset

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[VectorSearchResult]:
        return []


def record(text: str = "Alpha text", chunk_id: str = "chunk-1", chunk_index: int = 0, source_path: str = "docs/sample.pdf") -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        text=text,
        metadata={"source_path": source_path, "chunk_index": chunk_index},
        dense_vector=[1.0, 2.0],
    )


def test_same_chunk_twice_produces_same_vector_id() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(SimpleNamespace(), vector_store=store)
    first = upserter.upsert([record()])
    second = upserter.upsert([record()])

    assert first == second
    assert len(store.records) == 1


def test_content_change_changes_vector_id() -> None:
    upserter = VectorUpserter(SimpleNamespace(), vector_store=FakeVectorStore())

    first = upserter.upsert([record("Alpha text")])[0]
    second = upserter.upsert([record("Changed text")])[0]

    assert first != second


def test_chunk_index_change_changes_vector_id() -> None:
    upserter = VectorUpserter(SimpleNamespace(), vector_store=FakeVectorStore())

    first = upserter.upsert([record(chunk_index=0)])[0]
    second = upserter.upsert([record(chunk_index=1)])[0]

    assert first != second


def test_source_path_change_changes_vector_id() -> None:
    upserter = VectorUpserter(SimpleNamespace(), vector_store=FakeVectorStore())

    first = upserter.upsert([record(source_path="docs/a.pdf")])[0]
    second = upserter.upsert([record(source_path="docs/b.pdf")])[0]

    assert first != second


def test_batch_upsert_preserves_order() -> None:
    records = [record("Alpha", "chunk-1", 0), record("Beta", "chunk-2", 1), record("Gamma", "chunk-3", 2)]

    ids = VectorUpserter(SimpleNamespace(), vector_store=FakeVectorStore()).upsert(records)

    assert ids == [VectorUpserter(SimpleNamespace(), vector_store=FakeVectorStore())._stable_id(item) for item in records]


def test_vector_record_shape_preserves_text_vector_and_metadata() -> None:
    store = FakeVectorStore()
    original = record("Alpha text", "chunk-original", 7)

    vector_id = VectorUpserter(SimpleNamespace(), vector_store=store).upsert([original])[0]
    vector_record = store.records[vector_id]

    assert vector_record.text == "Alpha text"
    assert vector_record.vector == [1.0, 2.0]
    assert vector_record.metadata["source_path"] == "docs/sample.pdf"
    assert vector_record.metadata["chunk_index"] == 7
    assert vector_record.metadata["chunk_id"] == "chunk-original"


def test_empty_upsert_returns_empty_list_and_does_not_call_store() -> None:
    store = FakeVectorStore()

    ids = VectorUpserter(SimpleNamespace(), vector_store=store).upsert([])

    assert ids == []
    assert store.calls == []


def test_missing_dense_vector_raises() -> None:
    bad_record = ChunkRecord(id="chunk-1", text="Alpha", metadata={"source_path": "docs/sample.pdf"}, dense_vector=None)

    with pytest.raises(VectorUpserterError, match="dense_vector"):
        VectorUpserter(SimpleNamespace(), vector_store=FakeVectorStore()).upsert([bad_record])


def test_empty_dense_vector_raises() -> None:
    bad_record = ChunkRecord(id="chunk-1", text="Alpha", metadata={"source_path": "docs/sample.pdf"}, dense_vector=[])

    with pytest.raises(VectorUpserterError, match="non-empty"):
        VectorUpserter(SimpleNamespace(), vector_store=FakeVectorStore()).upsert([bad_record])


def test_upsert_count_mismatch_raises() -> None:
    with pytest.raises(VectorUpserterError, match="count"):
        VectorUpserter(SimpleNamespace(), vector_store=FakeVectorStore(count_offset=-1)).upsert([record()])


def test_upsert_passes_trace_to_vector_store_and_records_stage() -> None:
    trace = TraceContext()
    store = FakeVectorStore()

    VectorUpserter(SimpleNamespace(), vector_store=store).upsert([record()], trace=trace)

    assert store.calls[0]["trace"] is trace
    assert trace.stages[0]["name"] == "vector_upserter"
    assert trace.stages[0]["details"] == {"count": 1}


def test_vector_upserter_can_be_imported_from_package() -> None:
    from ingestion.storage import VectorUpserter as ExportedVectorUpserter

    assert ExportedVectorUpserter is VectorUpserter
