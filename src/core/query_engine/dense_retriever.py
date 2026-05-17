from __future__ import annotations

from typing import Any

from core import RetrievalResult
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.base_vector_store import BaseVectorStore, VectorSearchResult
from libs.vector_store.vector_store_factory import VectorStoreFactory


class DenseRetrieverError(ValueError):
    pass


class DenseRetriever:
    def __init__(
        self,
        settings: object,
        embedding_client: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: object | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise DenseRetrieverError("query must be a non-empty string")
        if not isinstance(top_k, int) or top_k <= 0:
            return []
        query_vector = self._query_vector(query, trace)
        results = self._store().query(query_vector, top_k=top_k, filters=filters, trace=trace)
        retrieval_results = [self._result(result) for result in results]
        self._record(trace, "dense_retriever", {"count": len(retrieval_results), "top_k": top_k, "filters": filters or {}})
        return retrieval_results

    def _query_vector(self, query: str, trace: object | None) -> list[float]:
        vectors = self._embedding().embed([query], trace=trace)
        if not isinstance(vectors, list) or len(vectors) != 1:
            raise DenseRetrieverError("embedding result must contain one query vector")
        vector = vectors[0]
        if not isinstance(vector, list) or not vector or not all(isinstance(value, int | float) for value in vector):
            raise DenseRetrieverError("query vector must be a non-empty numeric list")
        return [float(value) for value in vector]

    def _embedding(self) -> BaseEmbedding:
        if self.embedding_client is None:
            self.embedding_client = EmbeddingFactory.create(self.settings)
        return self.embedding_client

    def _store(self) -> BaseVectorStore:
        if self.vector_store is None:
            self.vector_store = VectorStoreFactory.create(self.settings)
        return self.vector_store

    def _result(self, result: VectorSearchResult) -> RetrievalResult:
        if not isinstance(result, VectorSearchResult):
            raise DenseRetrieverError("vector store query result must be VectorSearchResult")
        metadata = dict(result.metadata)
        chunk_id = metadata.get("chunk_id", result.id)
        if not isinstance(chunk_id, str) or not chunk_id:
            raise DenseRetrieverError("retrieval result chunk_id must be a non-empty string")
        return RetrievalResult(
            chunk_id=chunk_id,
            score=float(result.score),
            text=result.text,
            metadata=metadata,
        )

    def _record(self, trace: object | None, name: str, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)
