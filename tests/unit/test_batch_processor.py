from types import SimpleNamespace

import pytest

from core import Chunk, ChunkRecord
from core.trace import TraceContext
from ingestion.embedding import BatchProcessor, BatchProcessorError


class FakeDenseEncoder:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, chunks: list[Chunk], trace: object | None = None) -> list[ChunkRecord]:
        self.calls.append([chunk.id for chunk in chunks])
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                dense_vector=[float(index), float(len(chunk.text))],
            )
            for index, chunk in enumerate(chunks)
        ]


class FakeSparseEncoder:
    def __init__(self, id_suffix: str = "") -> None:
        self.calls = []
        self.id_suffix = id_suffix

    def encode(self, chunks: list[Chunk], trace: object | None = None) -> list[ChunkRecord]:
        self.calls.append([chunk.id for chunk in chunks])
        return [
            ChunkRecord(
                id=f"{chunk.id}{self.id_suffix}",
                text=chunk.text,
                metadata={**chunk.metadata, "sparse_token_count": len(chunk.text.split()), "sparse_unique_terms": len(set(chunk.text.split()))},
                sparse_vector={token: 1.0 for token in chunk.text.lower().split()},
            )
            for chunk in chunks
        ]


class ShortDenseEncoder:
    def encode(self, chunks: list[Chunk], trace: object | None = None) -> list[ChunkRecord]:
        return []


def chunk(index: int) -> Chunk:
    text = f"alpha {index}"
    return Chunk(
        id=f"chunk-{index}",
        text=text,
        metadata={"source_path": "docs/sample.pdf", "chunk_index": index},
        start_offset=index * 10,
        end_offset=index * 10 + len(text),
        source_ref="doc-1",
    )


def chunks(count: int) -> list[Chunk]:
    return [chunk(index) for index in range(count)]


def test_iter_batches_splits_five_chunks_into_three_batches() -> None:
    processor = BatchProcessor(SimpleNamespace(), dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder(), batch_size=2)

    batches = processor.iter_batches(chunks(5))

    assert [[chunk.id for chunk in batch] for batch in batches] == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]


def test_process_drives_dense_and_sparse_encoders_per_batch() -> None:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()

    BatchProcessor(SimpleNamespace(), dense_encoder=dense, sparse_encoder=sparse, batch_size=2).process(chunks(5))

    assert dense.calls == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]
    assert sparse.calls == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]


def test_process_preserves_order_and_merges_vectors() -> None:
    records = BatchProcessor(SimpleNamespace(), dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder(), batch_size=2).process(chunks(3))

    assert [record.id for record in records] == ["chunk-0", "chunk-1", "chunk-2"]
    assert records[0].dense_vector == [0.0, 7.0]
    assert records[0].sparse_vector == {"alpha": 1.0, "0": 1.0}
    assert records[0].metadata["sparse_token_count"] == 2


def test_process_returns_empty_list_for_empty_chunks() -> None:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()

    records = BatchProcessor(SimpleNamespace(), dense_encoder=dense, sparse_encoder=sparse, batch_size=2).process([])

    assert records == []
    assert dense.calls == []
    assert sparse.calls == []


def test_batch_size_can_come_from_settings() -> None:
    settings = SimpleNamespace(ingestion=SimpleNamespace(batch_size=2))
    processor = BatchProcessor(settings, dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder())

    assert len(processor.iter_batches(chunks(5))) == 3


def test_dict_batch_size_can_come_from_settings() -> None:
    processor = BatchProcessor({"ingestion": {"batch_size": 2}}, dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder())

    assert processor.batch_size == 2


def test_invalid_batch_size_raises() -> None:
    with pytest.raises(BatchProcessorError, match="positive integer"):
        BatchProcessor(SimpleNamespace(), dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder(), batch_size=0)


def test_encoder_count_mismatch_raises() -> None:
    processor = BatchProcessor(SimpleNamespace(), dense_encoder=ShortDenseEncoder(), sparse_encoder=FakeSparseEncoder(), batch_size=2)

    with pytest.raises(BatchProcessorError, match="counts"):
        processor.process(chunks(2))


def test_encoder_id_mismatch_raises() -> None:
    processor = BatchProcessor(SimpleNamespace(), dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder("-bad"), batch_size=2)

    with pytest.raises(BatchProcessorError, match="ids"):
        processor.process(chunks(2))


def test_process_records_trace_stages() -> None:
    trace = TraceContext()

    BatchProcessor(SimpleNamespace(), dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder(), batch_size=2).process(chunks(3), trace=trace)

    assert [stage["name"] for stage in trace.stages] == ["batch_processor.batch", "batch_processor.batch", "batch_processor"]
    assert trace.stages[-1]["details"]["count"] == 3
    assert trace.stages[-1]["details"]["batch_size"] == 2
    assert trace.stages[-1]["details"]["batch_count"] == 2


def test_batch_processor_can_be_imported_from_package() -> None:
    from ingestion.embedding import BatchProcessor as ExportedBatchProcessor

    assert ExportedBatchProcessor is BatchProcessor
