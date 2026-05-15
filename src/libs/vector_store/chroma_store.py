from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from core.settings import VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord, VectorSearchResult
from libs.vector_store.vector_store_factory import VectorStoreFactory


class ChromaStoreError(RuntimeError):
    pass


class ChromaStore(BaseVectorStore):
    def __init__(self, settings: VectorStoreSettings) -> None:
        super().__init__(settings)
        self.persist_path = Path(settings.persist_path)
        self.collection = settings.collection or "default"
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.store_path = self.persist_path / f"{self._safe_collection_name(self.collection)}.json"
        self.records = self._load()

    def upsert(self, records: list[VectorRecord], trace: object | None = None) -> int:
        if not isinstance(records, list):
            raise ChromaStoreError("chroma validation error: records must be a list")
        for record in records:
            self._validate_record(record)
            self.records[record.id] = record
        self._save()
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: object | None = None,
    ) -> list[VectorSearchResult]:
        self._validate_vector(vector, "query vector")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ChromaStoreError("chroma validation error: top_k must be a positive integer")
        active_filters = filters or {}
        results: list[VectorSearchResult] = []
        for record in self.records.values():
            if not self._matches_filters(record, active_filters):
                continue
            score = self._cosine_similarity(vector, record.vector)
            results.append(VectorSearchResult(id=record.id, score=score, text=record.text, metadata=dict(record.metadata)))
        return sorted(results, key=lambda result: (-result.score, result.id))[:top_k]

    def _load(self) -> dict[str, VectorRecord]:
        if not self.store_path.exists():
            return {}
        with self.store_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict):
            raise ChromaStoreError("chroma storage error: root must be object")
        records = raw.get("records", [])
        if not isinstance(records, list):
            raise ChromaStoreError("chroma storage error: records must be list")
        loaded: dict[str, VectorRecord] = {}
        for item in records:
            if not isinstance(item, dict):
                raise ChromaStoreError("chroma storage error: record must be object")
            record = VectorRecord(
                id=self._text(item, "id"),
                vector=self._vector(item, "vector"),
                text=self._text(item, "text"),
                metadata=self._metadata(item.get("metadata", {})),
            )
            loaded[record.id] = record
        return loaded

    def _save(self) -> None:
        data = {
            "collection": self.collection,
            "records": [
                {"id": record.id, "vector": record.vector, "text": record.text, "metadata": record.metadata}
                for record in sorted(self.records.values(), key=lambda item: item.id)
            ],
        }
        with self.store_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, sort_keys=True)

    def _validate_record(self, record: VectorRecord) -> None:
        if not isinstance(record, VectorRecord):
            raise ChromaStoreError("chroma validation error: record must be VectorRecord")
        if not record.id:
            raise ChromaStoreError("chroma validation error: record.id is required")
        if not isinstance(record.text, str):
            raise ChromaStoreError("chroma validation error: record.text must be string")
        self._validate_vector(record.vector, "record.vector")
        if not isinstance(record.metadata, dict):
            raise ChromaStoreError("chroma validation error: record.metadata must be object")

    def _validate_vector(self, vector: list[float], label: str) -> None:
        if not isinstance(vector, list) or not vector:
            raise ChromaStoreError(f"chroma validation error: {label} must be a non-empty list")
        if not all(isinstance(value, int | float) for value in vector):
            raise ChromaStoreError(f"chroma validation error: {label} must contain only numbers")

    def _matches_filters(self, record: VectorRecord, filters: dict[str, Any]) -> bool:
        if not isinstance(filters, dict):
            raise ChromaStoreError("chroma validation error: filters must be object")
        return all(record.metadata.get(key) == value for key, value in filters.items())

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ChromaStoreError("chroma query error: vector dimensions must match")
        numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _safe_collection_name(self, collection: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", collection).strip("._")
        return value or "default"

    def _text(self, item: dict[str, Any], key: str) -> str:
        value = item.get(key)
        if not isinstance(value, str):
            raise ChromaStoreError(f"chroma storage error: {key} must be string")
        return value

    def _vector(self, item: dict[str, Any], key: str) -> list[float]:
        value = item.get(key)
        if not isinstance(value, list) or not all(isinstance(part, int | float) for part in value):
            raise ChromaStoreError(f"chroma storage error: {key} must be numeric list")
        return [float(part) for part in value]

    def _metadata(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ChromaStoreError("chroma storage error: metadata must be object")
        return dict(value)


VectorStoreFactory.register_provider("chroma", ChromaStore)
