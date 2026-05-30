from pathlib import Path
from types import SimpleNamespace

import pytest

from core import Chunk, ChunkRecord, Document
from ingestion import IngestionPipeline, IngestionPipelineError, IngestionResult
from ingestion.storage import BM25Indexer, ImageStorage


class FakeIntegrityChecker:
    def __init__(self, file_hash: str = "hash-1", skip: bool = False) -> None:
        self.file_hash = file_hash
        self.skip = skip
        self.success = None
        self.failed = None

    def compute_sha256(self, path: str) -> str:
        return self.file_hash

    def should_skip(self, file_hash: str) -> bool:
        return self.skip

    def mark_success(self, file_hash: str, file_path: str, file_size: int | None = None, chunk_count: int | None = None) -> None:
        self.success = {"file_hash": file_hash, "file_path": file_path, "file_size": file_size, "chunk_count": chunk_count}

    def mark_failed(self, file_hash: str, error_msg: str, file_path: str = "", file_size: int | None = None) -> None:
        self.failed = {"file_hash": file_hash, "error_msg": error_msg, "file_path": file_path, "file_size": file_size}


class FakeLoader:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
        self.calls = []

    def load(self, path: str) -> Document:
        self.calls.append(path)
        return Document(
            id="hash-1",
            text="Alpha section\n[IMAGE: img-1]\n\nBeta section",
            metadata={
                "source_path": path,
                "doc_type": "pdf",
                "title": "Sample",
                "images": [
                    {
                        "id": "img-1",
                        "path": str(self.image_path),
                        "page": 2,
                        "text_offset": 14,
                        "text_length": 14,
                        "position": {},
                    }
                ],
            },
        )


class FakeChunker:
    def __init__(self) -> None:
        self.documents = []

    def split_document(self, document: Document) -> list[Chunk]:
        self.documents.append(document)
        return [
            Chunk(
                id="chunk-1",
                text="Alpha section [IMAGE: img-1]",
                metadata={**document.metadata, "chunk_index": 0},
                start_offset=0,
                end_offset=27,
                source_ref=document.id,
            ),
            Chunk(
                id="chunk-2",
                text="Beta section",
                metadata={"source_path": document.metadata["source_path"], "chunk_index": 1},
                start_offset=29,
                end_offset=41,
                source_ref=document.id,
            ),
        ]


class FakeTransform:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    def transform(self, chunks: list[Chunk], trace: object | None = None) -> list[Chunk]:
        self.calls.append([chunk.id for chunk in chunks])
        if self.fail:
            raise RuntimeError("boom")
        return [
            Chunk(
                id=chunk.id,
                text=chunk.text,
                metadata={**chunk.metadata, "transformed": True},
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                source_ref=chunk.source_ref,
            )
            for chunk in chunks
        ]


class FakeBatchProcessor:
    def __init__(self) -> None:
        self.calls = []

    def process(self, chunks: list[Chunk], trace: object | None = None) -> list[ChunkRecord]:
        self.calls.append([chunk.id for chunk in chunks])
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata={**chunk.metadata, "sparse_token_count": len(chunk.text.split())},
                dense_vector=[float(index + 1), 0.5],
                sparse_vector={token.lower(): 1.0 for token in chunk.text.split()},
            )
            for index, chunk in enumerate(chunks)
        ]


class FakeVectorUpserter:
    def __init__(self) -> None:
        self.records = []

    def upsert(self, records: list[ChunkRecord], trace: object | None = None) -> list[str]:
        self.records.extend(records)
        return [f"vec-{record.id}" for record in records]


def pipeline_parts(tmp_path: Path, skip: bool = False, fail_transform: bool = False):
    source_path = tmp_path / "simple.pdf"
    image_path = tmp_path / "image.png"
    source_path.write_bytes(b"%PDF fake")
    image_path.write_bytes(b"image-bytes")
    integrity = FakeIntegrityChecker(skip=skip)
    loader = FakeLoader(image_path)
    chunker = FakeChunker()
    transform = FakeTransform(fail=fail_transform)
    batch_processor = FakeBatchProcessor()
    vector_upserter = FakeVectorUpserter()
    bm25_indexer = BM25Indexer(tmp_path / "db" / "bm25")
    image_storage = ImageStorage(image_root=tmp_path / "images", db_path=tmp_path / "db" / "image_index.db")
    pipeline = IngestionPipeline(
        SimpleNamespace(),
        loader=loader,
        integrity_checker=integrity,
        chunker=chunker,
        transforms=[transform],
        batch_processor=batch_processor,
        vector_upserter=vector_upserter,
        bm25_indexer=bm25_indexer,
        image_storage=image_storage,
    )
    return SimpleNamespace(
        source_path=source_path,
        integrity=integrity,
        loader=loader,
        chunker=chunker,
        transform=transform,
        batch_processor=batch_processor,
        vector_upserter=vector_upserter,
        bm25_indexer=bm25_indexer,
        image_storage=image_storage,
        pipeline=pipeline,
    )


def test_pipeline_runs_full_ingestion_flow(tmp_path: Path) -> None:
    parts = pipeline_parts(tmp_path)
    progress = []

    result = parts.pipeline.run(parts.source_path, collection="docs", on_progress=lambda stage, current, total: progress.append((stage, current, total)))

    assert isinstance(result, IngestionResult)
    assert result.status == "success"
    assert result.skipped is False
    assert result.chunk_count == 2
    assert result.vector_ids == ["vec-chunk-1", "vec-chunk-2"]
    assert result.image_count == 1
    assert result.trace["status"] == "success"
    assert result.trace["trace_type"] == "ingestion"
    stages = {stage["name"]: stage for stage in result.trace["stages"]}
    for name in ("load", "split", "transform", "embed", "upsert"):
        assert name in stages
        assert stages[name]["elapsed_ms"] >= 0
        assert stages[name]["details"]["method"]
    assert stages["load"]["details"]["document_id"] == "hash-1"
    assert stages["split"]["details"]["count"] == 2
    assert stages["transform"]["details"]["count"] == 2
    assert stages["embed"]["details"]["count"] == 2
    assert stages["upsert"]["details"]["count"] == 2
    assert progress == [
        ("integrity", 1, 7),
        ("load", 2, 7),
        ("image_store", 3, 7),
        ("split", 4, 7),
        ("transform", 5, 7),
        ("encode", 6, 7),
        ("store", 7, 7),
    ]
    assert parts.loader.calls == [str(parts.source_path)]
    assert parts.transform.calls == [["chunk-1", "chunk-2"]]
    assert parts.batch_processor.calls == [["chunk-1", "chunk-2"]]
    assert [record.id for record in parts.vector_upserter.records] == ["chunk-1", "chunk-2"]
    assert parts.integrity.success["chunk_count"] == 2
    assert parts.bm25_indexer.index_path.exists()
    assert parts.bm25_indexer.query("alpha", top_k=1)[0]["chunk_id"] == "chunk-1"
    image_path = Path(parts.image_storage.get_path("img-1"))
    assert image_path.read_bytes() == b"image-bytes"
    assert image_path.parent == tmp_path / "images" / "docs"
    assert parts.chunker.documents[0].metadata["images"][0]["path"] == str(image_path)


def test_pipeline_skips_successful_hash_without_force(tmp_path: Path) -> None:
    parts = pipeline_parts(tmp_path, skip=True)

    result = parts.pipeline.run(parts.source_path, collection="docs")

    assert result.status == "skipped"
    assert result.skipped is True
    assert result.chunk_count == 0
    assert parts.loader.calls == []
    assert parts.integrity.success is None


def test_pipeline_force_reprocesses_successful_hash(tmp_path: Path) -> None:
    parts = pipeline_parts(tmp_path, skip=True)

    result = parts.pipeline.run(parts.source_path, collection="docs", force=True)

    assert result.status == "success"
    assert parts.loader.calls == [str(parts.source_path)]
    assert parts.integrity.success["file_hash"] == "hash-1"


def test_pipeline_wraps_failed_stage_and_marks_failed(tmp_path: Path) -> None:
    parts = pipeline_parts(tmp_path, fail_transform=True)

    with pytest.raises(IngestionPipelineError, match="transform failed: boom"):
        parts.pipeline.run(parts.source_path, collection="docs")

    assert parts.integrity.failed["file_hash"] == "hash-1"
    assert "transform failed: boom" in parts.integrity.failed["error_msg"]


def test_pipeline_can_be_imported_from_package() -> None:
    from ingestion import IngestionPipeline as ExportedIngestionPipeline

    assert ExportedIngestionPipeline is IngestionPipeline
