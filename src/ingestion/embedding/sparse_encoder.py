from __future__ import annotations

import re
from collections import Counter
from typing import Any

from core import Chunk, ChunkRecord


class SparseEncoder:
    default_stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }

    token_pattern = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]")

    def __init__(self, stop_words: set[str] | None = None) -> None:
        self.stop_words = {word.lower() for word in (stop_words if stop_words is not None else self.default_stop_words)}

    def encode(self, chunks: list[Chunk], trace: object | None = None) -> list[ChunkRecord]:
        records = [self._encode_chunk(chunk) for chunk in chunks]
        vocabulary = set()
        total_terms = 0
        for record in records:
            sparse_vector = record.sparse_vector or {}
            vocabulary.update(sparse_vector)
            total_terms += int(record.metadata.get("sparse_token_count", 0))
        self._record(trace, "sparse_encoder", {"count": len(records), "total_terms": total_terms, "vocabulary_size": len(vocabulary)})
        return records

    def _encode_chunk(self, chunk: Chunk) -> ChunkRecord:
        tokens = self._tokens(chunk.text)
        counts = Counter(tokens)
        sparse_vector = {term: float(counts[term]) for term in sorted(counts)}
        metadata = dict(chunk.metadata)
        metadata["sparse_token_count"] = len(tokens)
        metadata["sparse_unique_terms"] = len(sparse_vector)
        return ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            sparse_vector=sparse_vector,
        )

    def _tokens(self, text: str) -> list[str]:
        normalized = text.lower()
        tokens = self.token_pattern.findall(normalized)
        return [token for token in tokens if token not in self.stop_words]

    def _record(self, trace: object | None, name: str, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)
