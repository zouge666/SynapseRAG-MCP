from __future__ import annotations

from typing import Any

from core import Chunk, ChunkRecord
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class DenseEncoderError(ValueError):
    pass


class DenseEncoder:
    def __init__(self, settings: object, embedding: BaseEmbedding | None = None) -> None:
        self.settings = settings
        self.embedding = embedding

    def encode(self, chunks: list[Chunk], trace: object | None = None) -> list[ChunkRecord]:
        if not chunks:
            self._record(trace, "dense_encoder", {"count": 0, "dimension": 0})
            return []
        texts = [chunk.text for chunk in chunks]
        vectors = self._embedding().embed(texts, trace=trace)
        self._validate_vectors(vectors, len(chunks))
        records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                dense_vector=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._record(trace, "dense_encoder", {"count": len(records), "dimension": len(vectors[0])})
        return records

    def _embedding(self) -> BaseEmbedding:
        if self.embedding is None:
            self.embedding = EmbeddingFactory.create(self.settings)
        return self.embedding

    def _validate_vectors(self, vectors: list[list[float]], expected_count: int) -> None:
        if not isinstance(vectors, list):
            raise DenseEncoderError("embedding result must be a list")
        if len(vectors) != expected_count:
            raise DenseEncoderError("embedding vector count must match chunks count")
        if not vectors:
            raise DenseEncoderError("embedding result must not be empty")
        dimension = len(vectors[0])
        if dimension == 0:
            raise DenseEncoderError("embedding vectors must not be empty")
        for index, vector in enumerate(vectors):
            if not isinstance(vector, list):
                raise DenseEncoderError(f"embedding vector {index} must be a list")
            if len(vector) != dimension:
                raise DenseEncoderError("embedding vectors must have the same dimension")
            if not all(isinstance(value, int | float) for value in vector):
                raise DenseEncoderError(f"embedding vector {index} must contain only numbers")

    def _record(self, trace: object | None, name: str, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)
