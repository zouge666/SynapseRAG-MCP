from pathlib import Path

from core import ChunkRecord
from core.settings import VectorStoreSettings
from ingestion import CollectionStats, DeleteResult, DocumentDetail, DocumentInfo, DocumentManager
from ingestion.storage import BM25Indexer, ImageStorage
from libs.loader import SQLiteIntegrityChecker
from libs.vector_store.base_vector_store import VectorRecord
from libs.vector_store.chroma_store import ChromaStore


def make_manager(tmp_path: Path) -> tuple[DocumentManager, ChromaStore, BM25Indexer, ImageStorage, SQLiteIntegrityChecker, str, str]:
    collection = "docs"
    source_path = str(tmp_path / "sample.pdf")
    file_hash = "hash-a"
    chroma_store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma"), collection=collection))
    bm25_indexer = BM25Indexer(tmp_path / "bm25")
    image_storage = ImageStorage(image_root=tmp_path / "images", db_path=tmp_path / "image_index.db")
    file_integrity = SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db"))
    vector_records = [
        VectorRecord(id="vec-1", vector=[1.0, 0.0], text="Alpha", metadata={"source_path": source_path, "file_hash": file_hash, "chunk_id": "chunk-1", "chunk_index": 0}),
        VectorRecord(id="vec-2", vector=[0.0, 1.0], text="Beta", metadata={"source_path": source_path, "file_hash": file_hash, "chunk_id": "chunk-2", "chunk_index": 1}),
    ]
    chunk_records = [
        ChunkRecord(id="chunk-1", text="Alpha", metadata={"source_path": source_path, "file_hash": file_hash, "sparse_token_count": 1}, sparse_vector={"alpha": 1.0}),
        ChunkRecord(id="chunk-2", text="Beta", metadata={"source_path": source_path, "file_hash": file_hash, "sparse_token_count": 1}, sparse_vector={"beta": 1.0}),
    ]
    chroma_store.upsert(vector_records)
    bm25_indexer.upsert(chunk_records)
    bm25_indexer.save()
    image_storage.save_image("img-1", b"image", collection=collection, doc_hash=file_hash)
    file_integrity.mark_success(file_hash, source_path, file_size=10, chunk_count=2)
    return DocumentManager(chroma_store, bm25_indexer, image_storage, file_integrity), chroma_store, bm25_indexer, image_storage, file_integrity, source_path, file_hash


def test_document_manager_lists_documents_with_counts(tmp_path: Path) -> None:
    manager, _, _, _, _, source_path, file_hash = make_manager(tmp_path)

    documents = manager.list_documents("docs")

    assert len(documents) == 1
    assert isinstance(documents[0], DocumentInfo)
    assert documents[0].source_path == source_path
    assert documents[0].file_hash == file_hash
    assert documents[0].chunk_count == 2
    assert documents[0].image_count == 1
    assert documents[0].collection == "docs"
    assert documents[0].processed_at is not None


def test_document_manager_returns_document_detail(tmp_path: Path) -> None:
    manager, _, _, _, _, source_path, file_hash = make_manager(tmp_path)

    detail = manager.get_document_detail(file_hash)

    assert isinstance(detail, DocumentDetail)
    assert detail.document.source_path == source_path
    assert [chunk["id"] for chunk in detail.chunks] == ["chunk-1", "chunk-2"]
    assert [image["image_id"] for image in detail.images] == ["img-1"]


def test_document_manager_collection_stats(tmp_path: Path) -> None:
    manager, _, _, _, _, _, _ = make_manager(tmp_path)

    stats = manager.get_collection_stats("docs")

    assert isinstance(stats, CollectionStats)
    assert stats.to_dict() == {"collection": "docs", "document_count": 1, "chunk_count": 2, "image_count": 1}


def test_document_manager_deletes_document_from_all_stores(tmp_path: Path) -> None:
    manager, chroma_store, bm25_indexer, image_storage, file_integrity, source_path, file_hash = make_manager(tmp_path)

    result = manager.delete_document(source_path, "docs")

    assert isinstance(result, DeleteResult)
    assert result.deleted is True
    assert result.vector_count == 2
    assert result.bm25_count == 2
    assert result.image_count == 1
    assert result.integrity_count == 1
    assert manager.list_documents("docs") == []
    assert chroma_store.get_by_metadata({}) == []
    assert bm25_indexer.query("alpha beta") == []
    assert image_storage.list_images(collection="docs", doc_hash=file_hash) == []
    assert file_integrity.list_processed(status="success") == []


def test_document_manager_delete_missing_document_returns_empty_result(tmp_path: Path) -> None:
    manager, _, _, _, _, _, _ = make_manager(tmp_path)

    result = manager.delete_document("docs/missing.pdf", "docs")

    assert result.deleted is False
    assert result.to_dict()["vector_count"] == 0
