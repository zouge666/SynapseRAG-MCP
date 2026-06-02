from pathlib import Path

from core import ChunkRecord
from core.settings import VectorStoreSettings, load_settings
from ingestion import DocumentManager
from ingestion.storage import BM25Indexer, ImageStorage
from libs.loader import SQLiteIntegrityChecker
from libs.vector_store.base_vector_store import VectorRecord
from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.data_service import DataService, image_exists


def make_service(tmp_path: Path) -> tuple[DataService, str, str]:
    collection = "docs"
    source_path = str(tmp_path / "sample.pdf")
    file_hash = "hash-a"
    image_path = tmp_path / "image.png"
    chroma_store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma"), collection=collection))
    bm25_indexer = BM25Indexer(tmp_path / "bm25")
    image_storage = ImageStorage(image_root=tmp_path / "images", db_path=tmp_path / "image_index.db")
    file_integrity = SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db"))
    chroma_store.upsert(
        [
            VectorRecord(
                id="vec-1",
                vector=[1.0, 0.0],
                text="Alpha",
                metadata={"source_path": source_path, "file_hash": file_hash, "chunk_id": "chunk-1", "chunk_index": 0, "image_refs": ["img-1"]},
            ),
            VectorRecord(
                id="vec-2",
                vector=[0.0, 1.0],
                text="Beta",
                metadata={"source_path": source_path, "file_hash": file_hash, "chunk_id": "chunk-2", "chunk_index": 1},
            ),
        ]
    )
    bm25_indexer.upsert(
        [
            ChunkRecord(id="chunk-1", text="Alpha", metadata={"source_path": source_path, "file_hash": file_hash, "sparse_token_count": 1}, sparse_vector={"alpha": 1.0}),
            ChunkRecord(id="chunk-2", text="Beta", metadata={"source_path": source_path, "file_hash": file_hash, "sparse_token_count": 1}, sparse_vector={"beta": 1.0}),
        ]
    )
    image_path.write_bytes(b"image")
    image_storage.save_image("img-1", image_path, collection=collection, doc_hash=file_hash)
    file_integrity.mark_success(file_hash, source_path, file_size=10, chunk_count=2)
    manager = DocumentManager(chroma_store, bm25_indexer, image_storage, file_integrity)
    return DataService(settings=load_settings("config/settings.yaml"), document_manager=manager), source_path, file_hash


def test_data_service_lists_documents_as_dashboard_rows(tmp_path: Path) -> None:
    service, source_path, file_hash = make_service(tmp_path)

    documents = service.list_documents("docs")

    assert documents == [
        {
            "doc_id": file_hash,
            "source_path": source_path,
            "collection": "docs",
            "file_hash": file_hash,
            "chunk_count": 2,
            "image_count": 1,
            "processed_at": documents[0]["processed_at"],
            "metadata": documents[0]["metadata"],
        }
    ]
    assert documents[0]["processed_at"] is not None


def test_data_service_returns_document_detail_chunks_and_images(tmp_path: Path) -> None:
    service, _, file_hash = make_service(tmp_path)

    detail = service.get_document_detail(file_hash)
    chunks = service.get_chunks(file_hash)
    images = service.list_images("docs", file_hash)

    assert detail["document"]["file_hash"] == file_hash
    assert [chunk["id"] for chunk in chunks] == ["chunk-1", "chunk-2"]
    assert [image["image_id"] for image in images] == ["img-1"]
    assert image_exists(images[0]) is True
    assert image_exists({"file_path": "missing.png"}) is False
