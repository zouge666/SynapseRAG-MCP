from types import SimpleNamespace

import pytest

from core import Chunk, ChunkRecord
from core.trace import TraceContext
from ingestion.embedding import DenseEncoder, DenseEncoderError
from libs.embedding.base_embedding import BaseEmbedding


class FakeEmbedding(BaseEmbedding):
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        super().__init__(SimpleNamespace(provider="fake", model="fake-embedding"))
        self.vectors = vectors
        self.calls = []

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        self.calls.append({"texts": texts, "trace": trace})
        if self.vectors is not None:
            return self.vectors
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


def chunk(text: str = "Alpha text", chunk_id: str = "chunk-1") -> Chunk:
    return Chunk(id=chunk_id, text=text, metadata={"source_path": "docs/sample.pdf", "chunk_index": 0}, start_offset=0, end_offset=len(text), source_ref="doc-1")


def test_dense_encoder_encodes_chunk_texts_to_chunk_records() -> None:
    embedding = FakeEmbedding()
    chunks = [chunk("Alpha", "chunk-1"), chunk("Beta text", "chunk-2")]

    records = DenseEncoder(SimpleNamespace(), embedding=embedding).encode(chunks)

    assert all(isinstance(record, ChunkRecord) for record in records)
    assert [record.id for record in records] == ["chunk-1", "chunk-2"]
    assert [record.text for record in records] == ["Alpha", "Beta text"]
    assert [record.dense_vector for record in records] == [[5.0, 0.0], [9.0, 1.0]]
    assert embedding.calls[0]["texts"] == ["Alpha", "Beta text"]


def test_dense_encoder_preserves_metadata_copy() -> None:
    original = chunk("Alpha", "chunk-1")

    record = DenseEncoder(SimpleNamespace(), embedding=FakeEmbedding()).encode([original])[0]
    record.metadata["chunk_index"] = 99

    assert original.metadata["chunk_index"] == 0


def test_dense_encoder_returns_empty_records_for_empty_chunks() -> None:
    embedding = FakeEmbedding()

    records = DenseEncoder(SimpleNamespace(), embedding=embedding).encode([])

    assert records == []
    assert embedding.calls == []


def test_dense_encoder_records_trace_stage() -> None:
    trace = TraceContext()

    DenseEncoder(SimpleNamespace(), embedding=FakeEmbedding()).encode([chunk("Alpha")], trace=trace)

    assert trace.stages[0]["name"] == "dense_encoder"
    assert trace.stages[0]["details"] == {"count": 1, "dimension": 2}


def test_dense_encoder_passes_trace_to_embedding() -> None:
    trace = TraceContext()
    embedding = FakeEmbedding()

    DenseEncoder(SimpleNamespace(), embedding=embedding).encode([chunk("Alpha")], trace=trace)

    assert embedding.calls[0]["trace"] is trace


def test_dense_encoder_rejects_vector_count_mismatch() -> None:
    encoder = DenseEncoder(SimpleNamespace(), embedding=FakeEmbedding([[1.0, 2.0]]))

    with pytest.raises(DenseEncoderError, match="count"):
        encoder.encode([chunk("Alpha", "chunk-1"), chunk("Beta", "chunk-2")])


def test_dense_encoder_rejects_dimension_mismatch() -> None:
    encoder = DenseEncoder(SimpleNamespace(), embedding=FakeEmbedding([[1.0, 2.0], [3.0]]))

    with pytest.raises(DenseEncoderError, match="same dimension"):
        encoder.encode([chunk("Alpha", "chunk-1"), chunk("Beta", "chunk-2")])


def test_dense_encoder_rejects_empty_vectors() -> None:
    encoder = DenseEncoder(SimpleNamespace(), embedding=FakeEmbedding([[]]))

    with pytest.raises(DenseEncoderError, match="must not be empty"):
        encoder.encode([chunk("Alpha")])


def test_dense_encoder_rejects_non_numeric_values() -> None:
    encoder = DenseEncoder(SimpleNamespace(), embedding=FakeEmbedding([[1.0, "bad"]]))

    with pytest.raises(DenseEncoderError, match="only numbers"):
        encoder.encode([chunk("Alpha")])


def test_dense_encoder_can_be_imported_from_package() -> None:
    from ingestion.embedding import DenseEncoder as ExportedDenseEncoder

    assert ExportedDenseEncoder is DenseEncoder
