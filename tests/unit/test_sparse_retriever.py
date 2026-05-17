from types import SimpleNamespace

import pytest

from core import ChunkRecord, RetrievalResult
from core.query_engine import SparseRetriever, SparseRetrieverError
from core.trace import TraceContext
from ingestion.storage import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord, VectorSearchResult
from libs.vector_store.chroma_store import ChromaStore, ChromaStoreError
from core.settings import VectorStoreSettings


class FakeVectorStore(BaseVectorStore):
    def __init__(self, records: list[VectorRecord]) -> None:
        super().__init__(SimpleNamespace(backend="fake", persist_path="memory", collection="default"))
        self.records = {record.id: record for record in records}
        self.calls = []

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
        return []

    def get_by_ids(self, ids: list[str], trace: object | None = None) -> list[VectorRecord]:
        self.calls.append({"ids": ids, "trace": trace})
        return [self.records[record_id] for record_id in ids if record_id in self.records]


def record(chunk_id: str, sparse_vector: dict[str, float], text: str, source_path: str = "docs/sample.pdf") -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        text=text,
        metadata={"source_path": source_path, "sparse_token_count": int(sum(sparse_vector.values()))},
        sparse_vector=sparse_vector,
    )


def vector_record(chunk_id: str, text: str, source_path: str = "docs/sample.pdf") -> VectorRecord:
    return VectorRecord(id=chunk_id, vector=[1.0, 0.0], text=text, metadata={"source_path": source_path, "chunk_id": chunk_id})


def indexer() -> BM25Indexer:
    return BM25Indexer().build(
        [
            record("chunk-1", {"alpha": 2.0, "beta": 1.0}, "alpha alpha beta", "docs/a.pdf"),
            record("chunk-2", {"alpha": 1.0, "gamma": 3.0}, "alpha gamma gamma gamma", "docs/a.pdf"),
            record("chunk-3", {"delta": 1.0}, "delta", "docs/b.pdf"),
        ]
    )


def test_sparse_retriever_queries_bm25_and_hydrates_vector_records() -> None:
    store = FakeVectorStore(
        [
            vector_record("chunk-1", "alpha alpha beta", "docs/a.pdf"),
            vector_record("chunk-2", "alpha gamma gamma gamma", "docs/a.pdf"),
            vector_record("chunk-3", "delta", "docs/b.pdf"),
        ]
    )
    trace = TraceContext(trace_type="query")

    results = SparseRetriever(SimpleNamespace(), bm25_indexer=indexer(), vector_store=store).retrieve(["delta", "alpha"], top_k=3, trace=trace)

    assert [result.chunk_id for result in results] == ["chunk-3", "chunk-2", "chunk-1"]
    assert results[0].chunk_id == "chunk-3"
    assert results[0].score == pytest.approx(indexer().query(["delta"], top_k=1)[0]["score"])
    assert results[0].text == "delta"
    assert results[0].metadata == {"source_path": "docs/b.pdf", "chunk_id": "chunk-3"}
    assert store.calls == [{"ids": ["chunk-3", "chunk-2", "chunk-1"], "trace": trace}]
    assert trace.stages[-1]["name"] == "sparse_retriever"
    assert trace.stages[-1]["details"] == {"count": 3, "top_k": 3, "keyword_count": 2}


def test_sparse_retriever_skips_missing_vector_records() -> None:
    store = FakeVectorStore([vector_record("chunk-3", "delta", "docs/b.pdf")])

    results = SparseRetriever(SimpleNamespace(), bm25_indexer=indexer(), vector_store=store).retrieve(["delta", "alpha"], top_k=3)

    assert [result.chunk_id for result in results] == ["chunk-3"]


def test_sparse_retriever_returns_empty_for_no_keywords_or_top_k() -> None:
    retriever = SparseRetriever(SimpleNamespace(), bm25_indexer=indexer(), vector_store=FakeVectorStore([]))

    assert retriever.retrieve([], top_k=3) == []
    assert retriever.retrieve(["alpha"], top_k=0) == []


def test_sparse_retriever_rejects_invalid_keywords() -> None:
    with pytest.raises(SparseRetrieverError, match="keywords"):
        SparseRetriever(SimpleNamespace(), bm25_indexer=indexer(), vector_store=FakeVectorStore([])).retrieve("alpha", top_k=3)


def test_sparse_retriever_loads_index_from_settings_path(tmp_path) -> None:
    bm25_path = tmp_path / "bm25"
    saved = BM25Indexer(bm25_path).build(
        [
            record("chunk-3", {"delta": 1.0}, "delta", "docs/b.pdf"),
        ]
    )
    saved.save()
    settings = {"ingestion": {"bm25_path": str(bm25_path)}}
    store = FakeVectorStore([vector_record("chunk-3", "delta", "docs/b.pdf")])

    results = SparseRetriever(settings, vector_store=store).retrieve(["delta"], top_k=1)

    assert [result.chunk_id for result in results] == ["chunk-3"]


def test_chroma_store_get_by_ids_preserves_input_order_and_records(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path), collection="docs"))
    store.upsert(
        [
            VectorRecord(id="chunk-1", vector=[1.0, 0.0], text="alpha", metadata={"source_path": "docs/a.pdf"}),
            VectorRecord(id="chunk-2", vector=[0.0, 1.0], text="beta", metadata={"source_path": "docs/b.pdf"}),
        ]
    )

    records = store.get_by_ids(["chunk-2", "missing", "chunk-1"])

    assert [(record.id, record.text) for record in records] == [("chunk-2", "beta"), ("chunk-1", "alpha")]


def test_chroma_store_get_by_ids_rejects_invalid_ids(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path), collection="docs"))

    with pytest.raises(ChromaStoreError, match="ids"):
        store.get_by_ids(["valid", ""])


def test_sparse_retriever_can_be_imported_from_package() -> None:
    from core.query_engine import SparseRetriever as ExportedSparseRetriever

    assert ExportedSparseRetriever is SparseRetriever
