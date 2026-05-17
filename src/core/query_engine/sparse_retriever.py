from __future__ import annotations

from typing import Any

from core import RetrievalResult
from ingestion.storage import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from libs.vector_store.vector_store_factory import VectorStoreFactory


class SparseRetrieverError(ValueError):
    pass


class SparseRetriever:
    def __init__(
        self,
        settings: object,
        bm25_indexer: BM25Indexer | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.bm25_indexer = bm25_indexer
        self.vector_store = vector_store

    def retrieve(self, keywords: list[str], top_k: int, trace: object | None = None) -> list[RetrievalResult]:
        if not isinstance(keywords, list) or not all(isinstance(keyword, str) for keyword in keywords):
            raise SparseRetrieverError("keywords must be a list of strings")
        active_keywords = [keyword for keyword in keywords if keyword]
        if not isinstance(top_k, int) or top_k <= 0 or not active_keywords:
            return []
        scored = self._indexer().query(active_keywords, top_k=top_k)
        ids = [str(item["chunk_id"]) for item in scored]
        records = self._store().get_by_ids(ids, trace=trace)
        records_by_id = {record.id: record for record in records}
        results = []
        for item in scored:
            chunk_id = str(item["chunk_id"])
            record = records_by_id.get(chunk_id)
            if record is None:
                continue
            results.append(self._result(record, float(item["score"])))
        self._record(trace, "sparse_retriever", {"count": len(results), "top_k": top_k, "keyword_count": len(active_keywords)})
        return results

    def _indexer(self) -> BM25Indexer:
        if self.bm25_indexer is None:
            self.bm25_indexer = BM25Indexer.from_persist_path(self._bm25_path())
        return self.bm25_indexer

    def _store(self) -> BaseVectorStore:
        if self.vector_store is None:
            self.vector_store = VectorStoreFactory.create(self.settings)
        return self.vector_store

    def _bm25_path(self) -> str:
        ingestion = getattr(self.settings, "ingestion", None)
        value = getattr(ingestion, "bm25_path", None) if ingestion is not None else None
        if value is None and isinstance(self.settings, dict):
            value = self.settings.get("ingestion", {}).get("bm25_path", self.settings.get("bm25_path"))
        return value if isinstance(value, str) and value else "data/db/bm25"

    def _result(self, record: VectorRecord, score: float) -> RetrievalResult:
        if not isinstance(record, VectorRecord):
            raise SparseRetrieverError("vector store record must be VectorRecord")
        metadata = dict(record.metadata)
        chunk_id = metadata.get("chunk_id", record.id)
        if not isinstance(chunk_id, str) or not chunk_id:
            raise SparseRetrieverError("retrieval result chunk_id must be a non-empty string")
        return RetrievalResult(chunk_id=chunk_id, score=score, text=record.text, metadata=metadata)

    def _record(self, trace: object | None, name: str, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)
