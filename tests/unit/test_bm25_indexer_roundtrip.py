import math

import pytest

from core import ChunkRecord
from ingestion.storage import BM25Indexer, BM25IndexerError


def record(chunk_id: str, sparse_vector: dict[str, float], source_path: str = "docs/sample.pdf", text: str | None = None) -> ChunkRecord:
    content = text or " ".join(sparse_vector)
    return ChunkRecord(
        id=chunk_id,
        text=content,
        metadata={"source_path": source_path, "sparse_token_count": int(sum(sparse_vector.values()))},
        sparse_vector=sparse_vector,
    )


def corpus() -> list[ChunkRecord]:
    return [
        record("chunk-1", {"alpha": 2.0, "beta": 1.0}, "docs/a.pdf", "alpha alpha beta"),
        record("chunk-2", {"alpha": 1.0, "gamma": 3.0}, "docs/a.pdf", "alpha gamma gamma gamma"),
        record("chunk-3", {"delta": 1.0}, "docs/b.pdf", "delta"),
    ]


def test_build_creates_inverted_index_with_postings() -> None:
    indexer = BM25Indexer().build(corpus())

    assert indexer.document_count == 3
    assert indexer.inverted_index["alpha"]["postings"] == [
        {"chunk_id": "chunk-1", "tf": 2.0, "doc_length": 3.0},
        {"chunk_id": "chunk-2", "tf": 1.0, "doc_length": 4.0},
    ]
    assert indexer.inverted_index["delta"]["postings"] == [{"chunk_id": "chunk-3", "tf": 1.0, "doc_length": 1.0}]


def test_idf_uses_spec_formula() -> None:
    indexer = BM25Indexer().build(corpus())

    assert indexer.inverted_index["delta"]["idf"] == pytest.approx(math.log((3 - 1 + 0.5) / (1 + 0.5)))
    assert indexer.inverted_index["alpha"]["idf"] == pytest.approx(math.log((3 - 2 + 0.5) / (2 + 0.5)))


def test_query_returns_stable_top_ids() -> None:
    indexer = BM25Indexer().build(corpus())

    results = indexer.query("delta alpha", top_k=3)

    assert [result["chunk_id"] for result in results] == ["chunk-3", "chunk-2", "chunk-1"]


def test_query_accepts_keyword_list() -> None:
    indexer = BM25Indexer().build(corpus())

    results = indexer.query(["delta", "missing"], top_k=5)

    assert results == [{"chunk_id": "chunk-3", "score": pytest.approx(indexer.query("delta")[0]["score"])}]


def test_save_and_load_roundtrip(tmp_path) -> None:
    BM25Indexer(tmp_path).build(corpus()).save()

    loaded = BM25Indexer(tmp_path).load()

    assert loaded.document_count == 3
    assert loaded.inverted_index == BM25Indexer(tmp_path).build(corpus()).inverted_index
    assert loaded.query("delta", top_k=1)[0]["chunk_id"] == "chunk-3"


def test_from_persist_path_loads_index(tmp_path) -> None:
    BM25Indexer(tmp_path).build(corpus()).save()

    loaded = BM25Indexer.from_persist_path(tmp_path)

    assert loaded.query("delta", top_k=1)[0]["chunk_id"] == "chunk-3"


def test_upsert_replaces_existing_record_and_rebuilds_index() -> None:
    indexer = BM25Indexer().build([record("chunk-1", {"alpha": 1.0})])

    indexer.upsert([record("chunk-1", {"beta": 2.0}), record("chunk-2", {"beta": 1.0})])

    assert "alpha" not in indexer.inverted_index
    assert [posting["chunk_id"] for posting in indexer.inverted_index["beta"]["postings"]] == ["chunk-1", "chunk-2"]


def test_build_replaces_previous_index() -> None:
    indexer = BM25Indexer().build([record("chunk-1", {"alpha": 1.0})])

    indexer.build([record("chunk-2", {"beta": 1.0})])

    assert "alpha" not in indexer.inverted_index
    assert list(indexer.records) == ["chunk-2"]


def test_remove_document_removes_matching_source_path() -> None:
    indexer = BM25Indexer().build(corpus())

    removed = indexer.remove_document("docs/a.pdf")

    assert removed == 2
    assert list(indexer.records) == ["chunk-3"]
    assert "alpha" not in indexer.inverted_index
    assert indexer.query("delta")[0]["chunk_id"] == "chunk-3"


def test_empty_index_query_returns_empty_list() -> None:
    assert BM25Indexer().query("alpha") == []


def test_top_k_zero_returns_empty_list() -> None:
    assert BM25Indexer().build(corpus()).query("alpha", top_k=0) == []


def test_invalid_query_type_raises() -> None:
    with pytest.raises(BM25IndexerError, match="query"):
        BM25Indexer().query({"alpha": 1})


def test_indexer_can_be_imported_from_package() -> None:
    from ingestion.storage import BM25Indexer as ExportedBM25Indexer

    assert ExportedBM25Indexer is BM25Indexer
