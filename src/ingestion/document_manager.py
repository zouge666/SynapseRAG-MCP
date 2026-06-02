from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    source_path: str
    collection: str
    file_hash: str = ""
    chunk_count: int = 0
    image_count: int = 0
    processed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "collection": self.collection,
            "file_hash": self.file_hash,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "processed_at": self.processed_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DocumentDetail:
    document: DocumentInfo
    chunks: list[dict[str, Any]]
    images: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "chunks": [dict(chunk) for chunk in self.chunks],
            "images": [dict(image) for image in self.images],
        }


@dataclass(frozen=True)
class DeleteResult:
    source_path: str
    collection: str
    vector_count: int = 0
    bm25_count: int = 0
    image_count: int = 0
    integrity_count: int = 0

    @property
    def deleted(self) -> bool:
        return self.vector_count + self.bm25_count + self.image_count + self.integrity_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "collection": self.collection,
            "vector_count": self.vector_count,
            "bm25_count": self.bm25_count,
            "image_count": self.image_count,
            "integrity_count": self.integrity_count,
            "deleted": self.deleted,
        }


@dataclass(frozen=True)
class CollectionStats:
    collection: str
    document_count: int = 0
    chunk_count: int = 0
    image_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
        }


class DocumentManager:
    def __init__(self, chroma_store: Any, bm25_indexer: Any, image_storage: Any, file_integrity: Any) -> None:
        self.chroma_store = chroma_store
        self.bm25_indexer = bm25_indexer
        self.image_storage = image_storage
        self.file_integrity = file_integrity

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        active_collection = self._collection(collection)
        records = self._vector_records(active_collection)
        integrity_by_source = self._integrity_by_source()
        grouped: dict[str, dict[str, Any]] = {}
        for record in records:
            metadata = dict(getattr(record, "metadata", {}) or {})
            source_path = self._source_path(metadata)
            if not source_path:
                continue
            item = grouped.setdefault(
                source_path,
                {
                    "doc_id": self._document_id(metadata, source_path),
                    "source_path": source_path,
                    "collection": active_collection,
                    "file_hash": self._file_hash(metadata),
                    "chunk_count": 0,
                    "metadata": {},
                },
            )
            item["chunk_count"] += 1
            item["metadata"].update(metadata)
            if not item["file_hash"]:
                item["file_hash"] = self._file_hash(metadata)
            if item["doc_id"] == source_path:
                item["doc_id"] = self._document_id(metadata, source_path)
        for source_path, record in integrity_by_source.items():
            if source_path not in grouped:
                grouped[source_path] = {
                    "doc_id": record.get("file_hash") or source_path,
                    "source_path": source_path,
                    "collection": active_collection,
                    "file_hash": record.get("file_hash", ""),
                    "chunk_count": int(record.get("chunk_count") or 0),
                    "metadata": {},
                }
            if not grouped[source_path]["file_hash"]:
                grouped[source_path]["file_hash"] = record.get("file_hash", "")
        documents = []
        for source_path, item in grouped.items():
            integrity = integrity_by_source.get(source_path, {})
            file_hash = str(item.get("file_hash") or integrity.get("file_hash") or "")
            documents.append(
                DocumentInfo(
                    doc_id=str(item["doc_id"]),
                    source_path=source_path,
                    collection=active_collection,
                    file_hash=file_hash,
                    chunk_count=int(item["chunk_count"]),
                    image_count=self._image_count(active_collection, file_hash),
                    processed_at=integrity.get("processed_at"),
                    metadata=dict(item["metadata"]),
                )
            )
        return sorted(documents, key=lambda document: document.source_path)

    def get_document_detail(self, doc_id: str) -> DocumentDetail:
        documents = self.list_documents()
        for document in documents:
            if doc_id in {document.doc_id, document.source_path, document.file_hash}:
                chunks = [self._chunk_dict(record) for record in self._records_for_source(document.source_path, document.collection)]
                images = self._images(document.collection, document.file_hash)
                return DocumentDetail(document=document, chunks=chunks, images=images)
        raise ValueError(f"document not found: {doc_id}")

    def delete_document(self, source_path: str, collection: str) -> DeleteResult:
        documents = [document for document in self.list_documents(collection) if document.source_path == source_path]
        file_hashes = {document.file_hash for document in documents if document.file_hash}
        file_hashes.update(record.get("file_hash", "") for record in self._integrity_by_source().values() if record.get("file_path") == source_path)
        file_hashes = {file_hash for file_hash in file_hashes if file_hash}
        vector_count = self._delete_vectors(source_path)
        bm25_count = int(self.bm25_indexer.remove_document(source_path)) if hasattr(self.bm25_indexer, "remove_document") else 0
        if bm25_count and hasattr(self.bm25_indexer, "save"):
            self.bm25_indexer.save()
        image_count = sum(int(self.image_storage.delete_images(collection=collection, doc_hash=file_hash)) for file_hash in file_hashes)
        integrity_count = sum(1 for file_hash in file_hashes if self.file_integrity.remove_record(file_hash))
        return DeleteResult(
            source_path=source_path,
            collection=collection,
            vector_count=vector_count,
            bm25_count=bm25_count,
            image_count=image_count,
            integrity_count=integrity_count,
        )

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        active_collection = self._collection(collection)
        documents = self.list_documents(active_collection)
        return CollectionStats(
            collection=active_collection,
            document_count=len(documents),
            chunk_count=sum(document.chunk_count for document in documents),
            image_count=sum(document.image_count for document in documents),
        )

    def _collection(self, collection: str | None = None) -> str:
        if collection:
            return collection
        return str(getattr(self.chroma_store, "collection", "default") or "default")

    def _vector_records(self, collection: str) -> list[Any]:
        store_collection = str(getattr(self.chroma_store, "collection", collection) or collection)
        if store_collection != collection:
            return []
        if hasattr(self.chroma_store, "get_by_metadata"):
            return list(self.chroma_store.get_by_metadata({}))
        records = getattr(self.chroma_store, "records", {})
        if isinstance(records, dict):
            return list(records.values())
        return list(records or [])

    def _records_for_source(self, source_path: str, collection: str) -> list[Any]:
        return [record for record in self._vector_records(collection) if self._source_path(getattr(record, "metadata", {}) or {}) == source_path]

    def _delete_vectors(self, source_path: str) -> int:
        if hasattr(self.chroma_store, "delete_by_metadata"):
            removed = int(self.chroma_store.delete_by_metadata({"source_path": source_path}))
            if removed:
                return removed
            return int(self.chroma_store.delete_by_metadata({"source": source_path}))
        return 0

    def _integrity_by_source(self) -> dict[str, dict[str, Any]]:
        if not hasattr(self.file_integrity, "list_processed"):
            return {}
        records = self.file_integrity.list_processed(status="success")
        by_source = {}
        for record in records:
            source_path = record.get("file_path")
            if isinstance(source_path, str) and source_path:
                by_source[source_path] = dict(record)
        return by_source

    def _chunk_dict(self, record: Any) -> dict[str, Any]:
        metadata = dict(getattr(record, "metadata", {}) or {})
        return {
            "id": metadata.get("chunk_id") or getattr(record, "id", ""),
            "vector_id": getattr(record, "id", ""),
            "text": getattr(record, "text", ""),
            "metadata": metadata,
        }

    def _images(self, collection: str, file_hash: str) -> list[dict[str, Any]]:
        if not file_hash:
            return []
        return [dict(image) for image in self.image_storage.list_images(collection=collection, doc_hash=file_hash)]

    def _image_count(self, collection: str, file_hash: str) -> int:
        return len(self._images(collection, file_hash))

    def _source_path(self, metadata: dict[str, Any]) -> str:
        source_path = metadata.get("source_path") or metadata.get("source")
        return source_path if isinstance(source_path, str) else ""

    def _file_hash(self, metadata: dict[str, Any]) -> str:
        for key in ("file_hash", "doc_hash", "document_id", "doc_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    def _document_id(self, metadata: dict[str, Any], source_path: str) -> str:
        return self._file_hash(metadata) or source_path
