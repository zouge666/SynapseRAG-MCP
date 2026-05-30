from pathlib import Path
from types import SimpleNamespace

from core import Chunk, ChunkRecord, Document
from ingestion import IngestionPipeline


class FakeIntegrityChecker:
    def __init__(self) -> None:
        self.success = None

    def compute_sha256(self, path: str) -> str:
        return "hash-1"

    def should_skip(self, file_hash: str) -> bool:
        return False

    def mark_success(self, file_hash: str, file_path: str, file_size: int | None = None, chunk_count: int | None = None) -> None:
        self.success = {"file_hash": file_hash, "file_path": file_path, "file_size": file_size, "chunk_count": chunk_count}

    def mark_failed(self, file_hash: str, error_msg: str, file_path: str = "", file_size: int | None = None) -> None:
        raise AssertionError(error_msg)


class FakeLoader:
    def load(self, path: str) -> Document:
        return Document(id="doc-1", text="Alpha\n\nBeta", metadata={"source_path": path})


class FakeChunker:
    def split_document(self, document: Document) -> list[Chunk]:
        return [
            Chunk(id="chunk-1", text="Alpha", metadata={"source_path": document.metadata["source_path"]}, start_offset=0, end_offset=5, source_ref=document.id),
            Chunk(id="chunk-2", text="Beta", metadata={"source_path": document.metadata["source_path"]}, start_offset=7, end_offset=11, source_ref=document.id),
        ]


class FakeTransform:
    def transform(self, chunks: list[Chunk], trace: object | None = None) -> list[Chunk]:
        return chunks


class FakeBatchProcessor:
    def process(self, chunks: list[Chunk], trace: object | None = None) -> list[ChunkRecord]:
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                dense_vector=[1.0, 0.0],
                sparse_vector={chunk.text.lower(): 1.0},
            )
            for chunk in chunks
        ]


class FakeVectorUpserter:
    def upsert(self, records: list[ChunkRecord], trace: object | None = None) -> list[str]:
        return [record.id for record in records]


class FakeBM25Indexer:
    def __init__(self) -> None:
        self.records = []
        self.saved = False

    def upsert(self, records: list[ChunkRecord]) -> None:
        self.records.extend(records)

    def save(self) -> None:
        self.saved = True


class FakeImageStorage:
    pass


def make_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        SimpleNamespace(),
        loader=FakeLoader(),
        integrity_checker=FakeIntegrityChecker(),
        chunker=FakeChunker(),
        transforms=[FakeTransform()],
        batch_processor=FakeBatchProcessor(),
        vector_upserter=FakeVectorUpserter(),
        bm25_indexer=FakeBM25Indexer(),
        image_storage=FakeImageStorage(),
    )


def test_ingestion_pipeline_calls_progress_callback_for_each_stage(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"pdf")
    progress = []
    pipeline = make_pipeline()

    result = pipeline.run(source_path, collection="docs", on_progress=lambda stage, current, total: progress.append((stage, current, total)))

    assert result.status == "success"
    assert progress == [
        ("integrity", 1, 7),
        ("load", 2, 7),
        ("image_store", 3, 7),
        ("split", 4, 7),
        ("transform", 5, 7),
        ("encode", 6, 7),
        ("store", 7, 7),
    ]


def test_ingestion_pipeline_runs_without_progress_callback(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"pdf")
    pipeline = make_pipeline()

    result = pipeline.run(source_path, collection="docs", on_progress=None)

    assert result.status == "success"
    assert result.chunk_count == 2
    assert result.vector_ids == ["chunk-1", "chunk-2"]
