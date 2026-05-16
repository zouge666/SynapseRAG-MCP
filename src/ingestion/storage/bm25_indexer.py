from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any

from core import ChunkRecord


class BM25IndexerError(ValueError):
    pass


class BM25Indexer:
    token_pattern = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]")

    def __init__(self, persist_path: str | Path = "data/db/bm25", k1: float = 1.5, b: float = 0.75) -> None:
        self.persist_path = Path(persist_path)
        self.index_path = self.persist_path / "index.pkl"
        self.k1 = k1
        self.b = b
        self.records: dict[str, dict[str, Any]] = {}
        self.inverted_index: dict[str, dict[str, Any]] = {}
        self.document_count = 0
        self.average_doc_length = 0.0

    def build(self, records: list[ChunkRecord]) -> BM25Indexer:
        self.records = {}
        self.upsert(records)
        return self

    def upsert(self, records: list[ChunkRecord]) -> None:
        for record in records:
            self.records[record.id] = self._record_state(record)
        self._rebuild_index()

    def query(self, query: str | list[str], top_k: int = 10) -> list[dict[str, float | str]]:
        if top_k <= 0:
            return []
        terms = self._query_terms(query)
        scores: dict[str, float] = {}
        for term in terms:
            entry = self.inverted_index.get(term)
            if entry is None:
                continue
            idf = float(entry["idf"])
            for posting in entry["postings"]:
                chunk_id = posting["chunk_id"]
                tf = float(posting["tf"])
                doc_length = float(posting["doc_length"])
                scores[chunk_id] = scores.get(chunk_id, 0.0) + self._score(tf, doc_length, idf)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [{"chunk_id": chunk_id, "score": score} for chunk_id, score in ranked[:top_k]]

    def save(self) -> None:
        self.persist_path.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as file:
            pickle.dump(self.to_dict(), file)

    def load(self) -> BM25Indexer:
        with self.index_path.open("rb") as file:
            data = pickle.load(file)
        self._load_dict(data)
        return self

    @classmethod
    def from_persist_path(cls, persist_path: str | Path) -> BM25Indexer:
        return cls(persist_path).load()

    def remove_document(self, source: str) -> int:
        removed = [
            chunk_id
            for chunk_id, record in self.records.items()
            if record.get("metadata", {}).get("source_path") == source or record.get("metadata", {}).get("source") == source
        ]
        for chunk_id in removed:
            self.records.pop(chunk_id, None)
        if removed:
            self._rebuild_index()
        return len(removed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "records": self.records,
            "inverted_index": self.inverted_index,
            "document_count": self.document_count,
            "average_doc_length": self.average_doc_length,
        }

    def _record_state(self, record: ChunkRecord) -> dict[str, Any]:
        sparse_vector = record.sparse_vector or {}
        if not all(isinstance(term, str) and isinstance(weight, int | float) for term, weight in sparse_vector.items()):
            raise BM25IndexerError("sparse_vector must map terms to numeric weights")
        return {
            "id": record.id,
            "text": record.text,
            "metadata": dict(record.metadata),
            "sparse_vector": {term: float(weight) for term, weight in sparse_vector.items() if float(weight) > 0},
            "doc_length": self._doc_length(record),
        }

    def _doc_length(self, record: ChunkRecord) -> float:
        value = record.metadata.get("sparse_token_count")
        if isinstance(value, int | float) and value >= 0:
            return float(value)
        return float(sum((record.sparse_vector or {}).values()))

    def _rebuild_index(self) -> None:
        self.document_count = len(self.records)
        if self.document_count == 0:
            self.average_doc_length = 0.0
            self.inverted_index = {}
            return
        self.average_doc_length = sum(float(record["doc_length"]) for record in self.records.values()) / self.document_count
        postings_by_term: dict[str, list[dict[str, float | str]]] = {}
        for chunk_id, record in self.records.items():
            doc_length = float(record["doc_length"])
            for term, tf in record["sparse_vector"].items():
                postings_by_term.setdefault(term, []).append({"chunk_id": chunk_id, "tf": float(tf), "doc_length": doc_length})
        self.inverted_index = {
            term: {
                "idf": self._idf(len(postings)),
                "postings": sorted(postings, key=lambda posting: str(posting["chunk_id"])),
            }
            for term, postings in sorted(postings_by_term.items())
        }

    def _idf(self, document_frequency: int) -> float:
        if self.document_count == 0 or document_frequency <= 0:
            return 0.0
        return math.log((self.document_count - document_frequency + 0.5) / (document_frequency + 0.5))

    def _score(self, tf: float, doc_length: float, idf: float) -> float:
        if tf <= 0:
            return 0.0
        average = self.average_doc_length or 1.0
        denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / average)
        return idf * (tf * (self.k1 + 1)) / denominator

    def _query_terms(self, query: str | list[str]) -> list[str]:
        if isinstance(query, str):
            raw_terms = self.token_pattern.findall(query.lower())
        elif isinstance(query, list):
            raw_terms = []
            for item in query:
                if isinstance(item, str):
                    raw_terms.extend(self.token_pattern.findall(item.lower()))
        else:
            raise BM25IndexerError("query must be a string or list of strings")
        terms = []
        for term in raw_terms:
            if term and term not in terms:
                terms.append(term)
        return terms

    def _load_dict(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise BM25IndexerError("persisted index must be a mapping")
        self.k1 = float(data.get("k1", self.k1))
        self.b = float(data.get("b", self.b))
        self.records = data.get("records", {})
        self.inverted_index = data.get("inverted_index", {})
        self.document_count = int(data.get("document_count", len(self.records)))
        self.average_doc_length = float(data.get("average_doc_length", 0.0))
