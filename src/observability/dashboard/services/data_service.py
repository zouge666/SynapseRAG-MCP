from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.settings import Settings
from ingestion import DocumentManager
from ingestion.storage import BM25Indexer, ImageStorage
from libs.loader import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.config_service import ConfigService


class DataService:
    def __init__(self, settings: Settings | None = None, document_manager: DocumentManager | None = None) -> None:
        self.settings = settings or ConfigService().load()
        self.document_manager = document_manager or self._document_manager(self.settings)

    def list_collections(self) -> list[str]:
        store = getattr(self.document_manager, "chroma_store", None)
        bound = getattr(store, "collection", None)
        default = bound if isinstance(bound, str) and bound else (self.settings.vector_store.collection or "default")
        names = {default}
        get_records = getattr(store, "get_by_metadata", None)
        if callable(get_records):
            try:
                records = get_records({})
            except Exception:
                records = []
            for record in records:
                metadata = getattr(record, "metadata", None)
                if isinstance(metadata, dict):
                    name = metadata.get("collection")
                    if isinstance(name, str) and name.strip():
                        names.add(name.strip())
        return sorted(names)

    def collection_dimensions(self, collection: str) -> int | None:
        persist_path = getattr(self.settings.vector_store, "persist_path", "")
        if not isinstance(persist_path, str) or not persist_path:
            return None
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", collection).strip("._") or "default"
        store_path = Path(persist_path) / f"{safe_name}.json"
        if not store_path.is_file():
            return None
        try:
            with store_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None
        records = data.get("records") if isinstance(data, dict) else None
        if not isinstance(records, list):
            return None
        for record in records:
            vector = record.get("vector") if isinstance(record, dict) else None
            if isinstance(vector, list) and vector:
                return len(vector)
        return None

    def list_documents(self, collection: str | None = None) -> list[dict[str, Any]]:
        return [document.to_dict() for document in self.document_manager.list_documents(collection or self._default_collection())]

    def get_document_detail(self, doc_id: str) -> dict[str, Any]:
        return self.document_manager.get_document_detail(doc_id).to_dict()

    def get_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        return self.get_document_detail(doc_id)["chunks"]

    def list_images(self, collection: str, doc_hash: str) -> list[dict[str, Any]]:
        return [dict(image) for image in self.document_manager.image_storage.list_images(collection=collection, doc_hash=doc_hash)]

    def _document_manager(self, settings: Settings) -> DocumentManager:
        return DocumentManager(
            ChromaStore(settings.vector_store),
            BM25Indexer(self._setting(settings, "bm25_path", "data/db/bm25")),
            ImageStorage(
                image_root=self._setting(settings, "image_root", "data/images"),
                db_path=self._setting(settings, "image_db_path", "data/db/image_index.db"),
            ),
            SQLiteIntegrityChecker(self._setting(settings, "integrity_db_path", "data/db/ingestion_history.db")),
        )

    def _default_collection(self) -> str:
        return self.settings.vector_store.collection or "default"

    def _setting(self, settings: Settings, name: str, default: str) -> str:
        ingestion = getattr(settings, "ingestion", None)
        value = getattr(ingestion, name, None) if ingestion is not None else None
        return value if isinstance(value, str) and value else default


def image_exists(image: dict[str, Any]) -> bool:
    path = image.get("file_path")
    return isinstance(path, str) and Path(path).is_file()
