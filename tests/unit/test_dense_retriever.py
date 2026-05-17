from types import SimpleNamespace

import pytest

from core import RetrievalResult
from core.query_engine import DenseRetriever, DenseRetrieverError
from core.settings import VectorStoreSettings
from core.trace import TraceContext
from libs.embedding.base_embedding import BaseEmbedding
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord, VectorSearchResult
from libs.vector_store.chroma_store import ChromaStore


class FakeEmbedding(BaseEmbedding):
    def __init__(self, vectors: list[list[float]] | object) -> None:
        self.vectors = vectors
        self.calls = []

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        self.calls.append({"texts": texts, "trace": trace})
        return self.vectors


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        super().__init__(SimpleNamespace(backend="fake", persist_path="memory", collection="default"))
        self.calls = []
        self.results = [
            VectorSearchResult(id="vec-1", score=0.91, text="Alpha text", metadata={"chunk_id": "chunk-1", "source_path": "docs/a.pdf"}),
            VectorSearchResult(id="vec-2", score=0.72, text="Beta text", metadata={"source_path": "docs/b.pdf"}),
        ]

    def upsert(self, records: list[VectorRecord], trace: object | None = None) -> int:
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[VectorSearchResult]:
        self.calls.append({"vector": vector, "top_k": top_k, "filters": filters, "trace": trace})
        return self.results[:top_k]

    def get_by_ids(self, ids: list[str], trace: object | None = None) -> list[VectorRecord]:
        return []


def test_retrieve_embeds_query_and_calls_vector_store() -> None:
    embedding = FakeEmbedding([[0.1, 0.2]])
    store = FakeVectorStore()
    trace = TraceContext(trace_type="query")

    results = DenseRetriever(SimpleNamespace(), embedding_client=embedding, vector_store=store).retrieve(
        "alpha query",
        top_k=2,
        filters={"collection": "docs"},
        trace=trace,
    )

    assert embedding.calls == [{"texts": ["alpha query"], "trace": trace}]
    assert store.calls == [{"vector": [0.1, 0.2], "top_k": 2, "filters": {"collection": "docs"}, "trace": trace}]
    assert results == [
        RetrievalResult(chunk_id="chunk-1", score=0.91, text="Alpha text", metadata={"chunk_id": "chunk-1", "source_path": "docs/a.pdf"}),
        RetrievalResult(chunk_id="vec-2", score=0.72, text="Beta text", metadata={"source_path": "docs/b.pdf"}),
    ]
    assert trace.stages[-1]["name"] == "dense_retriever"
    assert trace.stages[-1]["details"] == {"count": 2, "top_k": 2, "filters": {"collection": "docs"}}


def test_retrieve_top_k_zero_returns_empty_without_calls() -> None:
    embedding = FakeEmbedding([[0.1]])
    store = FakeVectorStore()

    results = DenseRetriever(SimpleNamespace(), embedding_client=embedding, vector_store=store).retrieve("alpha", top_k=0)

    assert results == []
    assert embedding.calls == []
    assert store.calls == []


def test_invalid_query_raises() -> None:
    with pytest.raises(DenseRetrieverError, match="query"):
        DenseRetriever(SimpleNamespace(), embedding_client=FakeEmbedding([[0.1]]), vector_store=FakeVectorStore()).retrieve("", top_k=1)


def test_embedding_count_mismatch_raises() -> None:
    with pytest.raises(DenseRetrieverError, match="one query vector"):
        DenseRetriever(SimpleNamespace(), embedding_client=FakeEmbedding([]), vector_store=FakeVectorStore()).retrieve("alpha", top_k=1)


def test_empty_embedding_vector_raises() -> None:
    with pytest.raises(DenseRetrieverError, match="non-empty numeric"):
        DenseRetriever(SimpleNamespace(), embedding_client=FakeEmbedding([[]]), vector_store=FakeVectorStore()).retrieve("alpha", top_k=1)


def test_invalid_vector_store_result_raises() -> None:
    store = FakeVectorStore()
    store.results = [{"id": "bad"}]

    with pytest.raises(DenseRetrieverError, match="VectorSearchResult"):
        DenseRetriever(SimpleNamespace(), embedding_client=FakeEmbedding([[0.1]]), vector_store=store).retrieve("alpha", top_k=1)


def test_retrieval_result_serializes_to_dict_and_json() -> None:
    result = RetrievalResult(chunk_id="chunk-1", score=0.5, text="Alpha", metadata={"source_path": "docs/a.pdf"})

    assert result.to_dict() == {"chunk_id": "chunk-1", "score": 0.5, "text": "Alpha", "metadata": {"source_path": "docs/a.pdf"}}
    assert RetrievalResult.from_json(result.to_json()) == result


def test_chroma_query_returns_text_field(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path), collection="docs"))
    store.upsert([VectorRecord(id="vec-1", vector=[1.0, 0.0], text="Stored chunk text", metadata={"source_path": "docs/a.pdf"})])

    result = store.query([1.0, 0.0], top_k=1)[0]

    assert result.text == "Stored chunk text"


def test_dense_retriever_can_be_imported_from_package() -> None:
    from core.query_engine import DenseRetriever as ExportedDenseRetriever

    assert ExportedDenseRetriever is DenseRetriever
