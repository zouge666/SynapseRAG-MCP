from __future__ import annotations

import hashlib
from typing import Any

from core import ChunkRecord
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserterError(ValueError):
    pass


class VectorUpserter:
    def __init__(self, settings: object, vector_store: BaseVectorStore | None = None) -> None:
        self.settings = settings
        self.vector_store = vector_store

    def upsert(self, records: list[ChunkRecord], trace: object | None = None) -> list[str]:
        vector_records = [self._vector_record(record) for record in records]
        if vector_records:
            count = self._store().upsert(vector_records, trace=trace)
            if count != len(vector_records):
                raise VectorUpserterError("vector store upsert count must match records count")
        self._record(trace, "vector_upserter", {"count": len(vector_records)})
        return [record.id for record in vector_records]

    def _vector_record(self, record: ChunkRecord) -> VectorRecord:
        if record.dense_vector is None:
            raise VectorUpserterError("dense_vector is required")
        if not record.dense_vector or not all(isinstance(value, int | float) for value in record.dense_vector):
            raise VectorUpserterError("dense_vector must be a non-empty numeric list")
        metadata = dict(record.metadata)
        metadata["chunk_id"] = record.id
        vector_id = self._stable_id(record)
        return VectorRecord(
            id=vector_id,
            vector=[float(value) for value in record.dense_vector],
            text=record.text,
            metadata=metadata,
        )

    def _stable_id(self, record: ChunkRecord) -> str:
        source_path = str(record.metadata.get("source_path", record.metadata.get("source", "")))
        chunk_index = str(record.metadata.get("chunk_index", ""))
        content_hash = hashlib.sha256(record.text.encode("utf-8")).hexdigest()[:8]
        digest = hashlib.sha256(f"{source_path}:{chunk_index}:{content_hash}".encode("utf-8")).hexdigest()[:16]
        return f"vec_{digest}"

    def _store(self) -> BaseVectorStore:
        if self.vector_store is None:
            self.vector_store = VectorStoreFactory.create(self.settings)
        return self.vector_store

    def _record(self, trace: object | None, name: str, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)
